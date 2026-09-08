"""Tests for read-only mode (CLIM-861)."""

from __future__ import annotations

import re
from collections.abc import Generator, Iterator

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.main import app
from open_climate_service.read_only import is_blocked

# The full mutating surface, written with the app's own path templates so
# test_every_mutating_route_is_covered_by_the_policy can compare exactly rather than
# approximately. Anything mutating that is intended to stay open goes in _OPEN_WRITES.
_MUTATING_ROUTES = [
    ("POST", "/ingestions"),
    ("DELETE", "/ingestions/jobs/{job_id}"),
    ("POST", "/exports/{export_id}"),
    ("POST", "/jobs"),
    ("PATCH", "/jobs/{job_id}"),
    ("DELETE", "/jobs/{job_id}"),
    ("POST", "/jobs/{job_id}/results"),
    ("DELETE", "/jobs/{job_id}/results"),
    ("POST", "/manage/ingest"),
    ("POST", "/manage/sync"),
    ("PUT", "/process_graphs/{process_graph_id}"),
    ("DELETE", "/process_graphs/{process_graph_id}"),
    ("POST", "/sync/{dataset_id}"),
]

# Mutating routes deliberately left open by read-only mode.
_OPEN_WRITES = {("POST", "/result")}


def _concrete(path: str) -> str:
    """Substitute path templates so a declared route can be requested over HTTP."""
    return re.sub(r"\{[^}]+\}", "abc", path)


# Reads a casual visitor must keep. Status is not asserted — only that the request is not
# refused by read-only mode, since several depend on ingested data this test suite lacks.
_PUBLIC_READS = [
    "/",
    "/health",
    "/info",
    "/collections",
    "/processes",
    "/process_graphs",
    "/file_formats",
    "/extent",
    "/datasets",
    "/stac",
    "/map",
    "/openeo",
    "/.well-known/openeo",
]


@pytest.fixture
def read_only() -> Iterator[None]:
    """Force the instance read-only for the duration of a test."""
    original = api_config.get_config
    api_config.get_config = lambda: {**original(), "read_only": True}  # type: ignore[assignment]
    try:
        yield
    finally:
        api_config.get_config = original  # type: ignore[assignment]


@pytest.fixture
def ro_client(read_only: None) -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


# --- the policy as a pure function ---------------------------------------------------


@pytest.mark.parametrize(("method", "path"), _MUTATING_ROUTES)
def test_policy_blocks_every_mutating_route(method: str, path: str) -> None:
    assert is_blocked(method, _concrete(path))


def test_policy_allows_synchronous_result_execution() -> None:
    """POST /result is a POST only because the graph travels in the body."""
    assert not is_blocked("POST", "/result")
    assert not is_blocked("POST", "/result/")  # trailing slash is the same rule


def test_policy_closes_the_admin_console_including_get() -> None:
    """/manage is an admin UI behind a GET, so method filtering would miss it."""
    assert is_blocked("GET", "/manage")
    assert is_blocked("GET", "/manage/ingest")


def test_policy_closes_batch_jobs_for_reads_too() -> None:
    """There is no request identity, so the job namespace is shared between visitors."""
    assert is_blocked("GET", "/jobs")
    assert is_blocked("GET", "/jobs/abc")
    assert is_blocked("GET", "/jobs/abc/results")
    assert is_blocked("GET", "/jobs/abc/results/result.zarr/zarr.json")


def test_policy_closes_export_delivery_for_reads_too() -> None:
    """Delivery exposes server credentials, so status and reports are closed too."""
    assert is_blocked("GET", "/exports")
    assert is_blocked("GET", "/exports/rainfall-monthly/jobs/abc")


@pytest.mark.parametrize("path", _PUBLIC_READS)
def test_policy_allows_public_reads(path: str) -> None:
    assert not is_blocked("GET", path)


def test_policy_allows_data_endpoints() -> None:
    assert not is_blocked("GET", "/zarr/some_dataset/zarr.json")
    assert not is_blocked("GET", "/icechunk/some_dataset/refs/branch.main/ref.json")
    assert not is_blocked("GET", "/stac/collections/some_dataset")


def test_policy_never_blocks_cors_preflight() -> None:
    """A rejected preflight surfaces as an opaque CORS error instead of a readable 403."""
    assert not is_blocked("OPTIONS", "/ingestions")
    assert not is_blocked("OPTIONS", "/manage")
    assert not is_blocked("OPTIONS", "/jobs")


def _live_mutating_routes() -> set[tuple[str, str]]:
    write_methods = {"POST", "PUT", "PATCH", "DELETE"}
    live: set[tuple[str, str]] = set()
    for route in app.routes:
        methods: set[str] = getattr(route, "methods", None) or set()
        for method in methods & write_methods:
            live.add((method, getattr(route, "path", "")))
    return live


def test_every_mutating_route_is_covered_by_the_policy() -> None:
    """Guard against a new mutating route slipping in undecided.

    Enumerates the live app and compares **exactly** against the declared lists, so adding
    a write endpoint forces a deliberate choice: block it (add to _MUTATING_ROUTES, which
    also gives it an HTTP-level test) or leave it open (add to _OPEN_WRITES).
    """
    live = _live_mutating_routes()
    declared = set(_MUTATING_ROUTES) | _OPEN_WRITES

    undeclared = sorted(f"{m} {p}" for m, p in live - declared)
    assert not undeclared, (
        f"mutating routes not accounted for by the read-only policy: {undeclared}. "
        "Add each to _MUTATING_ROUTES (blocked) or _OPEN_WRITES (deliberately open)."
    )

    stale = sorted(f"{m} {p}" for m, p in declared - live)
    assert not stale, f"declared routes that no longer exist in the app: {stale}"


def test_declared_expectations_match_the_policy() -> None:
    """Every route declared blocked is blocked, and every route declared open is open."""
    assert [(m, p) for m, p in _MUTATING_ROUTES if not is_blocked(m, p)] == []
    assert [(m, p) for m, p in _OPEN_WRITES if is_blocked(m, p)] == []


# --- enforcement over HTTP -----------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), _MUTATING_ROUTES)
def test_mutating_routes_are_refused_with_403(ro_client: TestClient, method: str, path: str) -> None:
    response = ro_client.request(method, _concrete(path), json={})
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["code"] == "PermissionsInsufficient"
    assert "read-only" in body["message"]
    # The message should name what the caller can do instead.
    assert "/result" in body["message"]


def test_admin_console_is_refused(ro_client: TestClient) -> None:
    assert ro_client.get("/manage").status_code == 403


@pytest.mark.parametrize("path", ["/health", "/info", "/collections", "/processes"])
def test_public_reads_still_work(ro_client: TestClient, path: str) -> None:
    assert ro_client.get(path).status_code == 200


def test_refusal_carries_cors_headers(ro_client: TestClient) -> None:
    """CORS must wrap the refusal, or a browser sees a network error rather than the 403."""
    response = ro_client.post("/ingestions", json={}, headers={"Origin": "https://example.org"})
    assert response.status_code == 403
    assert response.headers.get("access-control-allow-origin") == "*"


def test_preflight_for_a_blocked_route_still_succeeds(ro_client: TestClient) -> None:
    response = ro_client.options(
        "/ingestions",
        headers={
            "Origin": "https://example.org",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_info_advertises_read_only(ro_client: TestClient) -> None:
    assert ro_client.get("/info").json()["read_only"] is True


def test_landing_page_hides_the_manage_link(ro_client: TestClient) -> None:
    assert "/manage" not in ro_client.get("/").text


def _capabilities(client: TestClient) -> dict[str, list[str]]:
    """Return the openEO capabilities endpoint map.

    The capabilities document is served by ``GET /`` under content negotiation — ``?f=json``
    selects it over the HTML landing page. Note ``/openeo`` is *not* it: that route redirects
    to the hosted openEO web editor.
    """
    response = client.get("/", params={"f": "json"})
    assert response.status_code == 200
    return {e["path"]: e["methods"] for e in response.json()["endpoints"]}


def test_capabilities_omit_the_write_endpoints(ro_client: TestClient) -> None:
    endpoints = _capabilities(ro_client)
    assert "/jobs" not in endpoints
    assert "/jobs/{job_id}" not in endpoints
    assert "/jobs/{job_id}/results" not in endpoints
    assert endpoints["/process_graphs/{process_graph_id}"] == ["GET"]
    assert endpoints["/result"] == ["POST"]
    assert endpoints["/collections"] == ["GET"]


# --- default (writable) behaviour is unchanged ---------------------------------------


def test_disabled_by_default() -> None:
    assert api_config.is_read_only() is False


def test_writable_instance_does_not_refuse(client: TestClient) -> None:
    """Without the flag, a mutating route reaches its handler (and fails on its own terms)."""
    response = client.post("/ingestions", json={})
    assert response.status_code != 403


def test_writable_instance_serves_the_admin_console(client: TestClient) -> None:
    assert client.get("/manage").status_code == 200


def test_writable_capabilities_keep_the_write_endpoints(client: TestClient) -> None:
    endpoints = _capabilities(client)
    assert endpoints["/jobs"] == ["GET", "POST"]
    assert endpoints["/jobs/{job_id}"] == ["GET", "PATCH", "DELETE"]
    assert endpoints["/process_graphs/{process_graph_id}"] == ["GET", "PUT", "DELETE"]


def test_info_reports_writable_by_default(client: TestClient) -> None:
    assert client.get("/info").json()["read_only"] is False


# --- config parsing -------------------------------------------------------------------


def test_rejects_a_non_boolean_value() -> None:
    original = api_config.get_config
    api_config.get_config = lambda: {**original(), "read_only": "true"}  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="must be true or false"):
            api_config.is_read_only()
    finally:
        api_config.get_config = original  # type: ignore[assignment]


def test_accepts_an_explicit_false() -> None:
    original = api_config.get_config
    api_config.get_config = lambda: {**original(), "read_only": False}  # type: ignore[assignment]
    try:
        assert api_config.is_read_only() is False
    finally:
        api_config.get_config = original  # type: ignore[assignment]


def test_policy_is_inert_when_the_flag_is_off(client: TestClient) -> None:
    """is_blocked() describes the policy; the middleware only applies it when configured."""
    assert is_blocked("POST", "/ingestions")  # policy says yes...
    assert client.post("/ingestions", json={}).status_code != 403  # ...but it is not applied
