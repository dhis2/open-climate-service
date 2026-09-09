"""Absolute URLs must carry the configured public scheme (CLIM-974).

Behind a TLS-terminating proxy the request arrives over plain HTTP, so anything built from
`request.base_url` or `request.url` emits `http://` on an HTTPS deployment. In a browser that
is active mixed content: the fetches are blocked and the map viewer renders an empty
catalogue, while curl against every endpoint still returns 200. HSTS hides it entirely, so
the deployment where it was found looked fine.
"""

import urllib.parse

import pytest
from fastapi.testclient import TestClient

from open_climate_service.shared.urls import BASE_URL_ENV

from .conftest import MountedClientFactory

_CONFIGURED = "https://ocs-demo-nepal.dhis2.org"


@pytest.fixture
def https_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose requests arrive over HTTP while the public origin is HTTPS.

    This is exactly the proxied deployment: TestClient's own base URL is `http://testserver`,
    standing in for what uvicorn sees behind the proxy.
    """
    from open_climate_service.main import app

    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    return TestClient(app)


# -- the helper -----------------------------------------------------------------------------


def test_the_configured_origin_wins_over_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    request = _fake_request("http://internal:9000/", "/map")
    assert absolute_base(request) == _CONFIGURED


def test_the_request_is_the_fallback_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local development, served directly with no proxy and no configuration."""
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    request = _fake_request("http://localhost:9000/", "/map")
    assert absolute_base(request) == "http://localhost:9000"


@pytest.mark.parametrize("configured", [f"{_CONFIGURED}/", f"  {_CONFIGURED}  ", f"{_CONFIGURED}///"])
def test_a_sloppy_base_url_is_normalised(monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
    """A trailing slash or stray whitespace in the deployment config must not double up in
    every link on the page."""
    from open_climate_service.shared.urls import absolute_base, absolute_url

    monkeypatch.setenv(BASE_URL_ENV, configured)
    request = _fake_request("http://internal:9000/", "/map")
    assert absolute_base(request) == _CONFIGURED
    assert absolute_url(request, "/collections") == f"{_CONFIGURED}/collections"


def test_a_blank_base_url_is_not_a_configured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty environment variable is how a compose file spells "unset"; treating it as an
    origin would emit links with no host at all."""
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.setenv(BASE_URL_ENV, "   ")
    request = _fake_request("http://localhost:9000/", "/map")
    assert absolute_base(request) == "http://localhost:9000"


def test_the_self_url_keeps_the_requested_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/stac` and `/stac/catalog.json` are one document, and each must name itself."""
    from open_climate_service.shared.urls import self_url

    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    assert self_url(_fake_request("http://internal:9000/", "/stac")) == f"{_CONFIGURED}/stac"
    assert self_url(_fake_request("http://internal:9000/", "/stac/catalog.json")) == f"{_CONFIGURED}/stac/catalog.json"


def test_the_self_url_drops_the_query_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OCS document varies by query string, so reflecting client input into a served
    document buys nothing."""
    from open_climate_service.shared.urls import self_url

    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    request = _fake_request("http://internal:9000/", "/stac", query="a=1&b=2")
    assert self_url(request) == f"{_CONFIGURED}/stac"


def _fake_request(base: str, path: str, query: str = "", root_path: str = "", app_root_path: str | None = None):
    """A Request with just enough scope for the URL helpers.

    `root_path` is what uvicorn or an embedding `Mount` sets; `app_root_path` is what Starlette
    records for the outermost app, which under a `Mount` differs from `root_path`.
    """
    from fastapi import Request

    split = urllib.parse.urlsplit(base)
    scope = {
        "type": "http",
        "scheme": split.scheme,
        "server": (split.hostname, split.port or 80),
        "path": path,
        "query_string": query.encode(),
        "headers": [(b"host", split.netloc.encode())],
        "root_path": root_path,
    }
    if app_root_path is not None:
        scope["app_root_path"] = app_root_path
    return Request(scope)


# -- the two reported endpoints -------------------------------------------------------------


def test_the_map_viewer_html_carries_no_origin(https_client: TestClient) -> None:
    """The reported defect is `http://` fetch targets on an `https://` page (CLIM-974).

    Mount-relative targets fix it without naming an origin, which matters: with the configured
    origin baked in, opening the viewer through a port-forward fetches the *public* instance's
    catalogue — permitted by the wildcard CORS — and the operator validates an ingest against
    another instance's data.

    Scoped to the HTML the server renders, which is all this greps. Once a collection is
    selected the page loads raster data from the `zarr.href` inside the collection JSON, which
    `build_collection` builds with `absolute_url` and so names the configured origin. On a
    port-forward the dropdown is local while the chunks come from the configured instance —
    arguably correct STAC behaviour, and not covered here; see the PR description.
    """
    body = https_client.get("/map").text

    assert "http://testserver" not in body
    assert _CONFIGURED not in body, "the viewer should not name any origin"
    for path in ("/extent", "/collections"):
        assert f'fetch("{path}")' in body


def test_the_manage_console_posts_to_the_origin_it_was_reached_on(https_client: TestClient) -> None:
    """Form actions and redirects stay relative for the same reason: an operator on a
    port-forward pressing Ingest must not POST to the configured public instance."""
    body = https_client.get("/manage").text

    assert _CONFIGURED not in body
    # Strict: the bare substring also appears in the `action="http://testserver/..."` form this
    # test exists to reject, so matching on it alone would pass against the very bug.
    assert 'action="/manage/ingest"' in body


def test_the_landing_page_links_within_the_instance_it_is_served_from(https_client: TestClient) -> None:
    """HTML explicitly: `/` content-negotiates, and the openEO capabilities on the JSON side
    *must* carry the configured origin because other processes consume those links."""
    body = https_client.get("/", headers={"Accept": "text/html"}).text

    assert _CONFIGURED not in body
    assert 'href="/map"' in body


def test_the_capabilities_json_still_carries_the_configured_origin(https_client: TestClient) -> None:
    """The other half of the split. Links in a document read by another process have to be
    absolute and have to name the public origin."""
    links = https_client.get("/", headers={"Accept": "application/json"}).json()["links"]

    assert any(link["href"].startswith(_CONFIGURED) for link in links)
    assert not any("testserver" in link["href"] for link in links)


def test_the_stac_self_link_uses_the_configured_scheme(https_client: TestClient) -> None:
    """`self` was built from the raw request URL while its sibling links already used the
    configured base — so one link in the document disagreed with the rest."""
    links = {link["rel"]: link["href"] for link in https_client.get("/stac").json()["links"]}

    assert links["self"] == f"{_CONFIGURED}/stac"
    assert links["root"] == f"{_CONFIGURED}/stac/catalog.json"


def test_the_catalog_json_self_link_names_its_own_path(https_client: TestClient) -> None:
    links = {link["rel"]: link["href"] for link in https_client.get("/stac/catalog.json").json()["links"]}

    assert links["self"] == f"{_CONFIGURED}/stac/catalog.json"


def test_no_served_document_leaks_the_internal_origin(https_client: TestClient) -> None:
    """A sweep rather than a list, so a new absolute URL built from the request is caught here
    instead of on a deployment. Non-browser STAC clients do not implement HSTS, so an `http`
    link is followed as written."""
    for path in ("/", "/map", "/stac", "/stac/catalog.json", "/collections", "/processes"):
        response = https_client.get(path, headers={"Accept": "application/json"} if path == "/" else {})
        assert response.status_code == 200, path
        assert "http://testserver" not in response.text, path


# -- degenerate configuration ----------------------------------------------------------------


@pytest.mark.parametrize("configured", ["/", "///", "  /  "])
def test_a_base_url_that_is_only_slashes_falls_back_to_the_request(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    """A value that is nothing but slashes has no origin in it to use.

    Judged before the slashes come off, `"/"` passes as configured and then strips to the empty
    string: every href in every served document loses its origin and `/openeo` redirects to
    `editor.openeo.org/?server=`.
    """
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.setenv(BASE_URL_ENV, configured)
    assert absolute_base(_fake_request("http://localhost:9000/", "/stac")) == "http://localhost:9000"


def test_the_self_link_does_not_double_a_mount_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary mounted deployment, not an exotic one.

    Uvicorn sets `scope["path"] = root_path + path`, so with `--root-path /ocs` behind a
    stripping proxy a request for `/stac` arrives with `request.url.path == "/ocs/stac"` while
    the fallback origin `request.base_url` already ends in `/ocs/`. Appending that unstripped
    gives `/ocs/ocs/stac` for `self` while every other link stays correct, and a STAC client
    following `self` 404s.

    The scope is built by hand rather than through `mounted_client` because
    `TestClient(root_path=...)` does not prepend the prefix to `path` the way uvicorn does, and
    because the configured origin must be unset for the fallback to be reached at all.
    """
    from open_climate_service.shared.urls import self_url

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    request = _fake_request("http://host/", "/ocs/stac", root_path="/ocs")
    assert self_url(request) == "http://host/ocs/stac"


def test_a_configured_origin_is_unaffected_by_a_mount_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix belongs to the deployment, so a configured public origin that already
    includes it must not have it stripped or doubled."""
    from open_climate_service.shared.urls import self_url

    monkeypatch.setenv(BASE_URL_ENV, "https://example.org/ocs")
    request = _fake_request("http://host/", "/ocs/stac", root_path="/ocs")
    assert self_url(request) == "https://example.org/ocs/stac"


# -- mount prefix ----------------------------------------------------------------------------
#
# Three possible positions on the same links. Two are wrong in opposite directions, so the
# tests below pin all three rather than only the right one:
#
#   absolute, configured origin -> submits to the *public* instance from a port-forward
#   bare leading slash          -> drops a deployment prefix, proxy 404
#   mount-relative path         -> inherits the page's origin, resolves under the prefix


@pytest.fixture
def mounted_client(monkeypatch: pytest.MonkeyPatch, mounted_client_factory: MountedClientFactory) -> TestClient:
    """Served under `/ocs` the way uvicorn does it, with a configured public origin that is
    *not* the request's and that carries no path of its own."""
    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    return mounted_client_factory("/ocs")


def test_an_origin_only_base_url_composes_with_the_asgi_prefix(mounted_client: TestClient) -> None:
    """`ROOT_PATH=/ocs` with `CLIMATE_SERVICE_BASE_URL=https://host` is the documented pairing.

    Returning the configured value verbatim put the HTML links under `/ocs` while every STAC
    and openEO href named the origin root: one document, two service roots, and the absolute
    half 404s at the proxy.
    """
    payload = mounted_client.get("/stac").json()
    for link in payload["links"]:
        assert link["href"].startswith(f"{_CONFIGURED}/ocs/"), link

    well_known = mounted_client.get("/.well-known/openeo").json()
    assert all(v["url"].startswith(f"{_CONFIGURED}/ocs") for v in well_known["versions"]), well_known


def test_the_manage_console_posts_under_the_mount_prefix(mounted_client: TestClient) -> None:
    """A page served at `/ocs/manage` that posts to `/manage/ingest` reaches the proxy, not the
    app, and gets a 404."""
    body = mounted_client.get("/manage").text

    assert 'action="/ocs/manage/ingest"' in body
    assert 'action="/manage/ingest"' not in body
    # Still no origin: the prefix is a path, so the page's own scheme and host are inherited.
    assert _CONFIGURED not in body


def test_the_landing_page_links_under_the_mount_prefix(mounted_client: TestClient) -> None:
    body = mounted_client.get("/", headers={"Accept": "text/html"}).text

    assert 'href="/ocs/map"' in body
    assert 'href="/map"' not in body
    assert _CONFIGURED not in body


def test_the_viewer_fetches_under_the_mount_prefix(mounted_client: TestClient) -> None:
    """Without the prefix the viewer fetched `/extent` from the origin root and rendered
    empty, which looks like missing data rather than a broken URL."""
    body = mounted_client.get("/map").text

    assert 'fetch("/ocs/extent")' in body
    assert 'fetch("/ocs/collections")' in body
    # The per-collection href too, not just the two literal fetch targets: a viewer that lists
    # datasets under the mount and then resolves each one outside it shows a catalogue where every
    # entry 404s on selection.
    assert "`/ocs/collections/${col.id}`" in body
    assert _CONFIGURED not in body


def test_an_unmounted_instance_gains_no_prefix(https_client: TestClient) -> None:
    """The prefix is empty at the root, so the same expression serves both deployments."""
    body = https_client.get("/manage").text

    assert 'action="/manage/ingest"' in body
    assert "//manage" not in body, "empty prefix must not leave a doubled slash"


@pytest.mark.parametrize(("reported", "expected"), [("/ocs/", "/ocs"), ("/ocs", "/ocs"), ("/", ""), ("", "")])
def test_mount_prefix_strips_a_trailing_slash(reported: str, expected: str) -> None:
    """`f"{mount_prefix(request)}/manage"` has to be right for every form the server reports."""
    from open_climate_service.shared.urls import mount_prefix

    assert mount_prefix(_fake_request("http://host/", "/manage", root_path=reported)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("/ocs", "/ocs"), ("/ocs/", "/ocs"), ("ocs", "/ocs"), (" /ocs/ ", "/ocs"), ("/", ""), ("", ""), ("  ", "")],
)
def test_the_root_path_setting_is_normalised(monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
    """Uvicorn joins `root_path + path` verbatim, so `ROOT_PATH=/ocs/` would put `/ocs//manage`
    on the scope, and `ocs` would render relative links. Both are the operator's most likely
    spellings after the canonical one."""
    from open_climate_service.shared.urls import configured_root_path

    monkeypatch.setenv("ROOT_PATH", raw)
    assert configured_root_path() == expected


def test_the_app_reads_the_root_path_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ROOT_PATH` set in `.env` must work under `make run` and bare uvicorn, which never go
    through `cli.py`. The app owns the setting, so every launcher gets it."""
    from open_climate_service.main import create_app

    monkeypatch.setenv("ROOT_PATH", "/ocs/")
    client = TestClient(create_app())

    assert 'action="/ocs/manage/ingest"' in client.get("/manage").text
    self_href = next(link["href"] for link in client.get("/stac").json()["links"] if link["rel"] == "self")
    assert self_href == "http://testserver/ocs/stac"


@pytest.mark.parametrize("root_path", ["/ocs", "/ocs/"])
def test_the_route_path_strips_the_prefix_the_way_starlette_does(root_path: str) -> None:
    """A trailing slash in `root_path` doubles the slash on the joined path. Starlette routes
    `/ocs//manage` under `/ocs/` to `/manage`; the read-only policy must see the same path."""
    from open_climate_service.shared.urls import route_path, self_url

    request = _fake_request("http://host/", root_path + "/manage", root_path=root_path)
    assert route_path(request) == "/manage"
    assert self_url(request) == "http://host/ocs/manage"


def test_the_mount_prefix_falls_back_to_the_base_url_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployment that declares its prefix only in `CLIMATE_SERVICE_BASE_URL`.

    Without the fallback, in-page links come out unprefixed: an instance behind
    `https://host/ocs/` renders `/map`, which 404s at the proxy. `ROOT_PATH` states this more
    directly and `root_path` wins whenever set, but the fallback has to keep working for
    deployments that cannot set it.
    """
    from open_climate_service.shared.urls import mount_prefix

    monkeypatch.setenv(BASE_URL_ENV, "https://host/ocs")
    assert mount_prefix(_fake_request("https://host/", "/")) == "/ocs"

    monkeypatch.setenv(BASE_URL_ENV, "https://host")
    assert mount_prefix(_fake_request("https://host/", "/")) == ""


def test_the_self_link_keeps_the_prefix_of_an_embedding_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under an embedding mount the fallback origin needs the prefix added to it.

    Starlette builds `base_url` from `app_root_path`, which under `Mount("/ocs", app)` is the
    outer root with no prefix at all, while `request.url.path` carries `/ocs`. Stripping the
    prefix off the path without adding it to the origin drops it altogether and `self` comes
    back as `/stac` — a 404.
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from open_climate_service.main import app as inner

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    outer = Starlette(routes=[Mount("/ocs", app=inner)])
    payload = TestClient(outer).get("/ocs/stac").json()
    self_href = next(link["href"] for link in payload["links"] if link["rel"] == "self")
    assert self_href == "http://testserver/ocs/stac"


def test_stripping_a_prefix_respects_segment_boundaries() -> None:
    """`startswith` alone turned `/stac` under a `/st` prefix into `/ac`."""
    from open_climate_service.shared.urls import strip_mount

    assert strip_mount("/stac", "/st") == "/stac"
    assert strip_mount("/ocs/stac", "/ocs") == "/stac"
    assert strip_mount("/ocs", "/ocs") == "/"
    assert strip_mount("/stac", "") == "/stac"
    assert strip_mount("/ocs//stac", "/ocs/") == "/stac", "a trailing-slash prefix consumes the doubled slash"


def test_the_self_link_does_not_double_a_configured_path_under_an_embedding_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured origin already is the service root, prefix included, so nothing adds it
    twice.

    Varying how much of the prefix comes off the path by which origin it is appended to leaves
    the prefix on the path under `Mount("/ocs", app)` while the configured path supplies it a
    second time: `https://public.example/ocs/ocs/stac`.
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from open_climate_service.main import app as inner

    monkeypatch.setenv(BASE_URL_ENV, "https://public.example/ocs")
    outer = Starlette(routes=[Mount("/ocs", app=inner)])
    payload = TestClient(outer).get("/ocs/stac").json()
    self_href = next(link["href"] for link in payload["links"] if link["rel"] == "self")
    assert self_href == "https://public.example/ocs/stac"


def test_every_link_of_one_document_agrees_under_an_embedding_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`self` is not the only link in the document, and the others are built differently.

    `root` and the child links come from `absolute_base` rather than the request path, so an
    origin without the prefix puts both forms in one STAC document — `self` with `/ocs`, `root`
    without — and a client following `root` gets a 404.
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from open_climate_service.main import app as inner

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    outer = Starlette(routes=[Mount("/ocs", app=inner)])
    payload = TestClient(outer).get("/ocs/stac").json()

    for link in payload["links"]:
        assert link["href"].startswith("http://testserver/ocs/"), link


def test_a_host_named_like_the_prefix_does_not_lose_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deciding whether `base_url` already carries the prefix by `endswith` compared a URL
    against a path: `http://ocs` ends with `/ocs`, so a compose service named after its mount
    served `self` as `http://ocs/stac`."""
    from open_climate_service.shared.urls import self_url

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    request = _fake_request("http://ocs/", "/ocs/stac", root_path="/ocs", app_root_path="")
    assert self_url(request) == "http://ocs/ocs/stac"


def test_a_mount_inside_a_root_path_carries_both_prefixes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--root-path /api` with `Mount("/ocs", app)` inside it: Starlette composes the child
    `root_path` as `/api/ocs` while `base_url` stops at the outer `/api`. Appending the whole
    child prefix to that gave `/api/api/ocs`."""
    from open_climate_service.shared.urls import self_url

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    request = _fake_request("http://host/", "/api/ocs/stac", root_path="/api/ocs", app_root_path="/api")
    assert self_url(request) == "http://host/api/ocs/stac"


# -- unusable configuration ------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured",
    ["ocs-demo-nepal.dhis2.org", "ocs-demo-nepal.dhis2.org/ocs", "https://", "//host/ocs"],
)
def test_a_base_url_without_a_scheme_and_host_falls_back_to_the_request(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    """Trimming instead of parsing accepts anything non-empty as an origin.

    A host with no scheme gives `ocs-demo-nepal.dhis2.org/stac`, which a browser resolves
    against the current page as a *relative* path; `https://` gives `https:/stac`. Both leave
    the API returning 200 and every served document carrying broken links. Falling back to the
    request origin is wrong in the way this variable exists to fix, but it is well-formed and it
    logs.
    """
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.setenv(BASE_URL_ENV, configured)
    assert absolute_base(_fake_request("http://localhost:9000/", "/stac")) == "http://localhost:9000"


@pytest.mark.parametrize("configured", ["https://host/ocs?x=1", "https://host/ocs#frag", "https://host/ocs/#"])
def test_a_base_url_is_parsed_rather_than_trimmed(monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
    """A query string or fragment otherwise lands in the middle of every absolute URL.

    `https://host/ocs?x=1` gives `https://host/ocs?x=1/stac` while in-page links stay correct,
    because two readings of one variable disagree. Both go through the same parse.
    """
    from open_climate_service.shared.urls import absolute_url, mount_prefix

    monkeypatch.setenv(BASE_URL_ENV, configured)
    request = _fake_request("http://internal:9000/", "/stac")
    assert absolute_url(request, "/stac") == "https://host/ocs/stac"
    assert mount_prefix(request) == "/ocs"


def test_an_unusable_base_url_is_logged_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence is how this survives: the links are broken but every response is a 200.

    The emissions are counted on the emitting logger with a handler of this test's own, rather
    than through `caplog`. `caplog` captures at the root, which this package's logger does not
    propagate to, so asserting through it means monkeypatching `propagate` and then trusting
    that nothing else in the process has put a second capture in that chain. Counting at the
    source is the same assertion without that dependency.
    """
    import logging

    from open_climate_service.shared.urls import _split_configured_base, absolute_base

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    emitter = logging.getLogger("open_climate_service.shared.urls")
    handler = _Collect(level=logging.WARNING)
    emitter.addHandler(handler)
    monkeypatch.setenv(BASE_URL_ENV, "ocs-demo-nepal.dhis2.org")
    try:
        for _ in range(3):
            absolute_base(_fake_request("http://localhost:9000/", "/stac"))

        assert len(records) == 1, "warned once per distinct value"
        assert BASE_URL_ENV in records[0].getMessage()

        # The parse cache is memoisation, not the deduplicator. Losing it must not repeat an
        # operator-facing warning, which is what made this assertion depend on execution order.
        _split_configured_base.cache_clear()
        absolute_base(_fake_request("http://localhost:9000/", "/stac"))

        assert len(records) == 1, "still once, even with the parse cache cleared"
    finally:
        emitter.removeHandler(handler)


def test_the_asgi_prefix_ignores_the_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """`asgi_prefix` describes the incoming path; `mount_prefix` describes outgoing links.

    Swapping them is what let a base URL of `https://host/jobs` strip `/jobs` off a real route.
    """
    from open_climate_service.shared.urls import asgi_prefix, mount_prefix

    monkeypatch.setenv(BASE_URL_ENV, "https://host/jobs")
    request = _fake_request("https://host/", "/jobs")

    assert asgi_prefix(request) == ""
    assert mount_prefix(request) == "/jobs"
