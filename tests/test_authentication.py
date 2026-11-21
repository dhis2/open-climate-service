"""Tests for named API tokens, tiers and the route table (CLIM-1292), and web sign-in (CLIM-1293)."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from http.cookiejar import DefaultCookiePolicy
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message

from open_geo_service import config as api_config
from open_geo_service.auth import middleware, sessions
from open_geo_service.auth.route_table import ROUTE_TABLE, route_tier
from open_geo_service.auth.tokens import Tier, configured_tokens
from open_geo_service.main import app
from open_geo_service.openeo.jobs import get_openeo_job_service
from open_geo_service.system.templates import actions_closed

from .conftest import MountedClientFactory

OPERATOR_SECRET = "o" * 32
CLIENT_SECRET = "c" * 40
OTHER_SECRET = "d" * 40
DECLARED = {route: tier for tier, routes in ROUTE_TABLE.items() for route in routes}


def _live_routes() -> tuple[set[str], set[str]]:
    """Every method and path the application serves, including routes left out of OpenAPI, and
    the paths of entries that serve any method below them, such as a `Mount`."""
    live: set[str] = set()
    mounts: set[str] = set()
    # FastAPI 0.162 keeps an included router as one entry in `app.routes`; its routes, with the
    # prefix applied, come from `effective_route_contexts()`.
    for entry in app.routes:
        included = getattr(entry, "effective_route_contexts", None)
        for route in included() if included else [entry]:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", None)
            if not methods:
                mounts.add(path)
            for method in methods or set():
                live.add(f"{'GET' if method == 'HEAD' else method} {path}")
    return live, mounts


def _under(route: str, mount: str) -> bool:
    path = route.split(" ", 1)[1]
    return path == mount or path.startswith(f"{mount.rstrip('/')}/")


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "abc", path)


def _configure(monkeypatch: pytest.MonkeyPatch, tokens: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(api_config, "_cache", {**api_config.get_config(), "authentication": {"tokens": tokens}})


@pytest.fixture
def auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _turn_on(monkeypatch)


def _turn_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEOSERVICE_TEST_OPERATOR_TOKEN", OPERATOR_SECRET)
    monkeypatch.setenv("GEOSERVICE_TEST_CLIENT_TOKEN", CLIENT_SECRET)
    monkeypatch.setenv("GEOSERVICE_TEST_OTHER_TOKEN", OTHER_SECRET)
    _configure(
        monkeypatch,
        [
            {"id": "caps", "token_env": "GEOSERVICE_TEST_OPERATOR_TOKEN", "tier": "operator"},
            {"id": "notebook", "token_env": "GEOSERVICE_TEST_CLIENT_TOKEN", "tier": "authenticated"},
            {"id": "other", "token_env": "GEOSERVICE_TEST_OTHER_TOKEN", "tier": "authenticated"},
        ],
    )


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _finished(client: TestClient, job_id: str, headers: dict[str, str]) -> str:
    """Start a batch job and return the status it ends in."""
    client.post(f"/jobs/{job_id}/results", headers=headers)
    deadline = time.monotonic() + 30
    while (status := client.get(f"/jobs/{job_id}", headers=headers).json()["status"]) not in ("error", "finished"):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    return str(status)


def _basic(token_id: str, secret: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{token_id}:{secret}".encode()).decode()}


# --- the route table ------------------------------------------------------------------


def test_every_served_route_is_in_the_route_table() -> None:
    """A new route fails here until it is given a tier; until then it needs the operator tier."""
    live, mounts = _live_routes()
    assert sorted(live - DECLARED.keys()) == [], "add each route to ROUTE_TABLE in auth/route_table.py"
    # A mount (/mcp, say) is listed by the methods and paths it serves, e.g. "POST /mcp/{path:path}".
    assert sorted(m for m in mounts if not any(_under(r, m) for r in DECLARED)) == [], "give each mount a tier"
    mounted = {route for route in DECLARED if any(_under(route, m) for m in mounts)}
    assert sorted(DECLARED.keys() - live - mounted) == [], "routes listed in ROUTE_TABLE that the app no longer serves"
    decided = {route: route_tier(route.split(" ")[0], _concrete(route.split(" ")[1])) for route in DECLARED}
    assert {route: tier for route, tier in decided.items() if tier != DECLARED[route]} == {}
    assert route_tier("GET", "/not-a-route") == Tier.OPERATOR


@pytest.mark.parametrize("route", sorted(route for route, tier in DECLARED.items() if tier > Tier.PUBLIC))
def test_gated_routes_need_a_token(auth_on: None, client: TestClient, route: str) -> None:
    method, path = route.split(" ", 1)
    response = client.request(method, _concrete(path), json={})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["code"] == "AuthenticationRequired"


# --- the middleware -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/", "/.well-known/openeo", "/openapi.json", "/credentials/oidc", "/stac/catalog.json"]
)
def test_reads_and_discovery_stay_public(auth_on: None, client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_refuses_before_reading_the_body(auth_on: None) -> None:
    sent: list[Message] = []

    async def receive() -> Message:
        raise AssertionError("the body was read")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/jobs",
        "raw_path": b"/jobs",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", b"1000000000")],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "scheme": "http",
    }
    asyncio.run(app(scope, receive, send))

    assert sent[0]["status"] == 401


def test_a_token_is_granted_its_tier(auth_on: None, client: TestClient) -> None:
    assert client.get("/me", headers=_bearer(CLIENT_SECRET)).json()["user_id"] == "token:notebook"
    assert client.get("/jobs", headers=_bearer(CLIENT_SECRET)).status_code == 200
    # openEO clients prefix the secret with the method they authenticated by.
    assert client.get("/jobs", headers=_bearer(f"basic//{CLIENT_SECRET}")).status_code == 200

    refused = client.post("/manage/sync", headers=_bearer(CLIENT_SECRET))
    assert refused.status_code == 403
    assert refused.json()["code"] == "PermissionsInsufficient"
    assert client.post("/schedules/sync/abc/pause", headers=_bearer(OPERATOR_SECRET)).status_code not in (401, 403)


def test_an_invalid_credential_is_refused_even_on_a_public_route(auth_on: None, client: TestClient) -> None:
    for headers in (_bearer("x" * 32), {"Authorization": "Token abc"}, {"Authorization": "Basic !!"}):
        response = client.get("/datasets", headers=headers)
        assert response.status_code == 401
        assert response.json()["code"] == "TokenInvalid"


def test_a_path_no_route_serves_is_not_found(auth_on: None, client: TestClient) -> None:
    # The openEO client asks for job logs, which Open Geo Service does not serve.
    assert client.get("/jobs/abc/logs").status_code == 404
    assert client.post("/datasets").status_code == 401


def test_without_tokens_nothing_is_checked(client: TestClient) -> None:
    assert client.get("/jobs", headers=_bearer("anything")).status_code == 200
    assert client.get("/me").json()["user_id"] == "anonymous"


@pytest.mark.parametrize("root_path", ["/ocs", "/ocs/"])
def test_a_mount_prefix_does_not_hide_a_gated_route(
    auth_on: None, mounted_client_factory: MountedClientFactory, root_path: str
) -> None:
    mounted = mounted_client_factory(root_path)

    assert mounted.get("/jobs").status_code == 401
    assert mounted.get("/stac/catalog.json").status_code == 200


def test_read_only_applies_before_the_tier(auth_on: None, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setitem(api_config._cache or {}, "read_only", True)

    assert client.post("/manage/sync", headers=_bearer(OPERATOR_SECRET)).json()["id"] == "read-only"
    # Read-only lets POST /result through, but never lowers its tier.
    assert client.post("/result", json={}).status_code == 401
    assert client.post("/result", json={}, headers=_bearer(CLIENT_SECRET)).status_code != 401


# --- ownership --------------------------------------------------------------------------


def test_a_job_is_reached_only_by_its_owner_and_operators(auth_on: None, client: TestClient) -> None:
    graph = {"add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}
    created = client.post("/jobs", json={"process": {"process_graph": graph}}, headers=_bearer(CLIENT_SECRET))
    job_id = created.json()["id"]

    def listed(secret: str) -> bool:
        return job_id in {job["id"] for job in client.get("/jobs", headers=_bearer(secret)).json()["jobs"]}

    assert listed(CLIENT_SECRET) and listed(OPERATOR_SECRET) and not listed(OTHER_SECRET)
    other = _bearer(OTHER_SECRET)
    for method, path in [
        ("GET", f"/jobs/{job_id}"),
        ("PATCH", f"/jobs/{job_id}"),
        ("POST", f"/jobs/{job_id}/results"),
        ("GET", f"/jobs/{job_id}/results/result.geojson"),
        ("DELETE", f"/jobs/{job_id}"),
    ]:
        assert client.request(method, path, json={}, headers=other).status_code == 404, f"{method} {path}"
    assert client.get(f"/jobs/{job_id}", headers=_bearer(OPERATOR_SECRET)).status_code == 200
    assert client.delete(f"/jobs/{job_id}", headers=_bearer(CLIENT_SECRET)).status_code == 204

    # The route table cannot see a `dataset_id` inside the job body: publishing is refused at the write.
    publishes = {
        "sum": {"process_id": "add", "arguments": {"x": 1, "y": 2}},
        "save": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "sum"}, "format": "ZARR", "options": {"dataset_id": "mine"}},
            "result": True,
        },
    }
    mine = _bearer(CLIENT_SECRET)
    job_id = client.post("/jobs", json={"process": {"process_graph": publishes}}, headers=mine).json()["id"]
    assert _finished(client, job_id, mine) == "error"
    assert "needs an operator token" in (get_openeo_job_service().get_job_or_404(job_id).error_message or "")


def test_a_stored_workflow_is_reached_only_by_its_owner_and_operators(auth_on: None, client: TestClient) -> None:
    graph = {"process_graph": {"add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}}
    assert client.put("/process_graphs/mine", json=graph, headers=_bearer(CLIENT_SECRET)).status_code == 200

    def listed(headers: dict[str, str]) -> bool:
        return "mine" in {item["id"] for item in client.get("/process_graphs", headers=headers).json()["processes"]}

    assert listed(_bearer(CLIENT_SECRET)) and listed(_bearer(OPERATOR_SECRET))
    assert not listed(_bearer(OTHER_SECRET)) and not listed({})
    assert "aggregate_by_features" in {item["id"] for item in client.get("/process_graphs").json()["processes"]}
    other = _bearer(OTHER_SECRET)
    assert client.get("/process_graphs/mine", headers=other).status_code == 404
    assert client.get("/workflows/mine").status_code == 404
    assert client.put("/process_graphs/mine", json=graph, headers=other).status_code == 403
    assert client.delete("/process_graphs/mine", headers=other).status_code == 404
    # Nor can it be run by id, synchronously or in a batch job.
    calls_mine = {"process_graph": {"m": {"process_id": "mine", "arguments": {}, "result": True}}}
    assert client.post("/result", json={"process": calls_mine}, headers=_bearer(CLIENT_SECRET)).json() == 3
    assert client.post("/result", json={"process": calls_mine}, headers=other).status_code == 400
    job_id = client.post("/jobs", json={"process": calls_mine}, headers=other).json()["id"]
    assert _finished(client, job_id, other) == "error"
    assert "mine not found" in (get_openeo_job_service().get_job_or_404(job_id).error_message or "")
    # Taking a built-in id would replace it for every caller and pipeline.
    built_in = "/process_graphs/aggregate_by_features"
    assert client.put(built_in, json=graph, headers=_bearer(CLIENT_SECRET)).status_code == 403
    assert client.delete("/process_graphs/mine", headers=_bearer(OPERATOR_SECRET)).status_code == 204


def test_records_stored_with_authentication_off_are_operator_only(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    # Authentication off stores no owner: its principal shares the anonymous caller's id.
    graph = {"process_graph": {"add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}}
    assert client.put("/process_graphs/before", json=graph).status_code == 200
    job_id = client.post("/jobs", json={"process": graph}).json()["id"]
    _turn_on(monkeypatch)

    def listed(headers: dict[str, str]) -> bool:
        return "before" in {item["id"] for item in client.get("/process_graphs", headers=headers).json()["processes"]}

    assert not listed({}) and not listed(_bearer(CLIENT_SECRET)) and listed(_bearer(OPERATOR_SECRET))
    assert client.get("/process_graphs/before").status_code == 404
    assert client.get(f"/jobs/{job_id}", headers=_bearer(CLIENT_SECRET)).status_code == 404
    assert client.get(f"/jobs/{job_id}", headers=_bearer(OPERATOR_SECRET)).status_code == 200
    assert client.delete("/process_graphs/before", headers=_bearer(OPERATOR_SECRET)).status_code == 204


# --- openEO -----------------------------------------------------------------------------


def test_openeo_basic_exchange_returns_the_secret(auth_on: None, client: TestClient) -> None:
    assert client.get("/credentials/basic", headers=_basic("notebook", CLIENT_SECRET)).json() == {
        "access_token": CLIENT_SECRET
    }
    assert client.get("/credentials/basic", headers=_basic("caps", CLIENT_SECRET)).status_code == 401
    for headers in ({}, _bearer(CLIENT_SECRET)):
        challenge = client.get("/credentials/basic", headers=headers)
        assert challenge.status_code == 401
        assert challenge.headers["WWW-Authenticate"].startswith("Basic")

    endpoints = {endpoint["path"] for endpoint in client.get("/").json()["endpoints"]}
    assert "/credentials/basic" in endpoints


def test_capabilities_omit_basic_auth_when_authentication_is_off(client: TestClient) -> None:
    endpoints = {endpoint["path"] for endpoint in client.get("/").json()["endpoints"]}
    assert "/credentials/basic" not in endpoints


# --- web sign-in --------------------------------------------------------------------------

SAME_ORIGIN = {"Origin": "https://testserver"}
HTML = {"Accept": "text/html"}


@pytest.fixture
def browser(client: TestClient) -> TestClient:
    """The client over HTTPS, which sign-in needs. It keeps no cookies: each test sends the
    session cookie by hand, so which requests carry it is visible in the test."""
    client.base_url = httpx.URL("https://testserver")
    client.cookies.jar.set_policy(DefaultCookiePolicy(blocked_domains=["testserver.local"]))
    return client


def _session(client: TestClient, secret: str) -> dict[str, str]:
    """Sign in through the form and return the cookie header the browser would send."""
    signed_in = client.post(
        "/sign-in", data={"token": secret, "next": "/datasets"}, headers=SAME_ORIGIN, follow_redirects=False
    )
    assert signed_in.status_code == 303 and signed_in.headers["location"] == "/datasets"
    set_cookie = signed_in.headers["set-cookie"]
    assert {"HttpOnly", "Secure", "SameSite=Lax", "Path=/"} <= {part.strip() for part in set_cookie.split(";")}
    assert secret not in set_cookie
    return {"Cookie": set_cookie.split(";")[0]}


def test_web_sign_in_opens_the_forms_its_tier_allows(
    auth_on: None, browser: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = browser
    pause = "/schedules/sync/nothing/pause"
    # A page rendered outside a request shows no operator forms.
    assert actions_closed() == "sign-in"
    signed_out = client.get("/data-sources", headers=HTML).text
    assert 'href="/sign-in">Sign in</a>' in signed_out and "Sign out" not in signed_out
    assert "Sign in</a> with an operator token to" in client.get("/", headers=HTML).text
    assert client.post(pause, headers=SAME_ORIGIN).status_code == 401

    came_from = client.get("/sign-in", headers={"Referer": "https://testserver/data-sources?q=1"})
    assert 'name="next" value="/data-sources?q=1"' in came_from.text
    monkeypatch.setattr(middleware, "_last_failure_log", float("-inf"))
    wrong = client.post("/sign-in", data={"token": "x" * 40}, headers=SAME_ORIGIN)
    assert wrong.status_code == 401 and "not valid" in wrong.text
    assert "invalid credential" in caplog.text
    other_site = {"Origin": "https://elsewhere.example"}
    refused = client.post("/sign-in", data={"token": OPERATOR_SECRET}, headers=other_site)
    assert refused.status_code == 403 and "GEOSERVICE_BASE_URL" in refused.json()["message"]
    # Only a path on this instance is returned to.
    off_site = client.post(
        "/sign-in", data={"token": CLIENT_SECRET, "next": "//elsewhere.example/"}, headers=SAME_ORIGIN
    )
    assert off_site.url.path == "/"

    # An authenticated token signs in, but the pages' forms need an operator.
    client_cookie = _session(client, CLIENT_SECRET)
    page = client.get("/", headers={**client_cookie, **HTML}).text
    assert "Sign out" in page and "You are signed in as notebook, which is not an operator token" in page
    assert client.post(pause, headers={**client_cookie, **SAME_ORIGIN}).status_code == 403

    cookie = _session(client, OPERATOR_SECRET)
    assert "with an operator token to" not in client.get("/", headers={**cookie, **HTML}).text
    assert client.post(pause, headers={**cookie, **SAME_ORIGIN}).status_code not in (401, 403)
    assert client.post(pause, headers={**cookie, "Sec-Fetch-Site": "same-origin"}).status_code not in (401, 403)
    # A write from another site, the same host on another scheme, or with no sign of where it
    # came from, is refused.
    assert client.post(pause, headers={**cookie, **other_site}).status_code == 403
    assert client.post(pause, headers={**cookie, "Origin": "http://testserver"}).status_code == 403
    assert client.post(pause, headers=cookie).status_code == 403
    # The page never gets the token back.
    assert client.get("/credentials/basic", headers=cookie).status_code == 401

    signed_out_now = client.post("/sign-out", headers={**cookie, **SAME_ORIGIN}, follow_redirects=False)
    assert signed_out_now.status_code == 303 and "Max-Age=0" in signed_out_now.headers["set-cookie"]
    # The ended session is no credential: pages stay readable and the stale cookie is cleared.
    stale = client.get("/datasets", headers={**cookie, **HTML})
    assert stale.status_code == 200 and "Max-Age=0" in stale.headers["set-cookie"]
    assert client.post(pause, headers={**cookie, **SAME_ORIGIN}).status_code == 401
    # Signing in again with the stale cookie sets only the new one.
    again = client.post(
        "/sign-in", data={"token": OPERATOR_SECRET}, headers={**cookie, **SAME_ORIGIN}, follow_redirects=False
    )
    set_cookies = again.headers.get_list("set-cookie")
    assert len(set_cookies) == 1 and "Max-Age=28800" in set_cookies[0]


def test_a_session_ends_when_it_expires_or_its_token_changes(
    auth_on: None, monkeypatch: pytest.MonkeyPatch, browser: TestClient
) -> None:
    client = browser
    cookie = _session(client, OPERATOR_SECRET)
    assert "caps" in client.get("/", headers={**cookie, **HTML}).text
    monkeypatch.setenv("GEOSERVICE_TEST_OPERATOR_TOKEN", "n" * 32)
    assert "caps" not in client.get("/", headers={**cookie, **HTML}).text

    cookie = _session(client, "n" * 32)
    monkeypatch.setattr(sessions, "LIFETIME_SECONDS", -1)
    expiring = _session(client, "n" * 32)
    assert "caps" in client.get("/", headers={**cookie, **HTML}).text
    assert "caps" not in client.get("/", headers={**expiring, **HTML}).text


def test_sign_in_is_offered_only_over_https_or_on_this_machine(auth_on: None, client: TestClient) -> None:
    over_http = client.get("/sign-in")
    assert "needs HTTPS" in over_http.text and 'name="token"' not in over_http.text
    refused = client.post("/sign-in", data={"token": OPERATOR_SECRET}, headers={"Origin": "http://testserver"})
    assert refused.status_code == 403 and "set-cookie" not in refused.headers
    client.base_url = httpx.URL("http://localhost:8000")
    # The server offers the form; the browser shows it only in a secure context, which a proxy
    # rewriting Host to a loopback address cannot fake.
    local = client.get("/sign-in").text
    assert 'name="token"' in local and 'id="sign-in" hidden' in local and "window.isSecureContext" in local


def test_without_tokens_there_is_no_sign_in(client: TestClient) -> None:
    assert client.get("/sign-in").status_code == 404
    page = client.get("/data-sources", headers=HTML).text
    assert "Sign in" not in page and "Sign out" not in page


# --- configuration -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "error"),
    [
        ([{"id": "a", "token_env": "GEOSERVICE_TEST_UNSET", "tier": "operator"}], "GEOSERVICE_TEST_UNSET is not set"),
        ([{"id": "a", "token_env": "GEOSERVICE_TEST_SHORT", "tier": "operator"}], "shorter than 32"),
        ([{"id": "a", "token_env": "GEOSERVICE_TEST_OPERATOR_TOKEN", "tier": "admin"}], "invalid"),
        (
            [
                {"id": "a", "token_env": "GEOSERVICE_TEST_OPERATOR_TOKEN", "tier": "operator"},
                {"id": "a", "token_env": "GEOSERVICE_TEST_CLIENT_TOKEN", "tier": "operator"},
            ],
            "id is used by an earlier token",
        ),
        (
            [
                {"id": "a", "token_env": "GEOSERVICE_TEST_OPERATOR_TOKEN", "tier": "operator"},
                {"id": "b", "token_env": "GEOSERVICE_TEST_SAME", "tier": "operator"},
            ],
            "secret in GEOSERVICE_TEST_SAME is used by an earlier token",
        ),
    ],
)
def test_an_unusable_token_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tokens: list[dict[str, Any]], error: str
) -> None:
    monkeypatch.setenv("GEOSERVICE_TEST_OPERATOR_TOKEN", OPERATOR_SECRET)
    monkeypatch.setenv("GEOSERVICE_TEST_CLIENT_TOKEN", CLIENT_SECRET)
    monkeypatch.setenv("GEOSERVICE_TEST_SHORT", "short")
    monkeypatch.setenv("GEOSERVICE_TEST_SAME", OPERATOR_SECRET)
    monkeypatch.delenv("GEOSERVICE_TEST_UNSET", raising=False)
    _configure(monkeypatch, tokens)

    with pytest.raises(ValueError, match=error) as raised:
        configured_tokens()
    assert OPERATOR_SECRET not in str(raised.value)
    with pytest.raises(ValueError, match=error), TestClient(app):
        pass
