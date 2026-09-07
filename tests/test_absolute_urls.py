"""Absolute URLs must carry the configured public scheme (CLIM-974).

Behind a TLS-terminating proxy the request arrives over plain HTTP, so anything built from
`request.base_url` or `request.url` emits `http://` on an HTTPS deployment. In a browser that
is active mixed content: the fetches are blocked and the map viewer renders an empty
catalogue, while curl against every endpoint still returns 200. HSTS hides it entirely, so
the deployment where it was found looked fine.
"""

import pytest
from fastapi.testclient import TestClient

from open_climate_service.shared.urls import BASE_URL_ENV

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


def _fake_request(base: str, path: str, query: str = ""):
    """A Request with just enough scope for the URL helpers."""
    from fastapi import Request

    host = base.split("://", 1)[1].rstrip("/")
    hostname, _, port = host.partition(":")
    return Request(
        {
            "type": "http",
            "scheme": base.split("://", 1)[0],
            "server": (hostname, int(port) if port else 80),
            "path": path,
            "query_string": query.encode(),
            "headers": [(b"host", host.encode())],
            "root_path": "",
        }
    )


# -- the two reported endpoints -------------------------------------------------------------


def test_the_map_viewer_carries_no_origin_at_all(https_client: TestClient) -> None:
    """The reported defect was `http://` fetch targets on an `https://` page. Root-relative
    targets fix it without naming an origin, which also removes a hazard the first fix
    introduced: with the configured origin baked in, opening the viewer through a port-forward
    fetched the *public* instance's catalogue — permitted by the wildcard CORS — and the
    operator would validate an ingest against another instance's data (CLIM-974 review).
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
    assert 'action="/manage/ingest"' in body or "/manage/ingest" in body


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
    """Truthiness was tested before the trailing slashes were stripped, so `"/"` survived the
    check, stripped to the empty string, and was returned — every href in every served
    document lost its origin, and `/openeo` redirected to `editor.openeo.org/?server=`
    (CLIM-974 review).
    """
    from open_climate_service.shared.urls import absolute_base

    monkeypatch.setenv(BASE_URL_ENV, configured)
    assert absolute_base(_fake_request("http://localhost:9000/", "/stac")) == "http://localhost:9000"


def test_the_self_link_does_not_double_a_mount_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behind a non-stripping proxy in front of `--root-path /ocs`, the fallback origin ends
    with the prefix and `request.url.path` carries it again, so `self` became
    `/ocs/ocs/stac` while every other link stayed correct — a STAC client following `self`
    would 404 (CLIM-974 review).
    """
    from fastapi import Request

    from open_climate_service.shared.urls import self_url

    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("host", 80),
            "path": "/ocs/stac",
            "root_path": "/ocs",
            "query_string": b"",
            "headers": [(b"host", b"host")],
        }
    )
    assert self_url(request) == "http://host/ocs/stac"


def test_a_configured_origin_is_unaffected_by_a_mount_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix belongs to the deployment, so a configured public origin that already
    includes it must not have it stripped or doubled."""
    from fastapi import Request

    from open_climate_service.shared.urls import self_url

    monkeypatch.setenv(BASE_URL_ENV, "https://example.org/ocs")
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("host", 80),
            "path": "/ocs/stac",
            "root_path": "/ocs",
            "query_string": b"",
            "headers": [(b"host", b"host")],
        }
    )
    assert self_url(request) == "https://example.org/ocs/stac"


# -- mount prefix ----------------------------------------------------------------------------
#
# Three positions on the same links, all of which have been in this file's history. Two are
# wrong in opposite directions, so the tests below pin all three rather than the survivor:
#
#   absolute, configured origin -> submits to the *public* instance from a port-forward
#   bare leading slash          -> drops a deployment prefix, proxy 404
#   mount-relative path         -> inherits the page's origin, resolves under the prefix


@pytest.fixture
def mounted_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Served under `/ocs`, with a configured public origin that is *not* the request's."""
    from open_climate_service.main import app

    monkeypatch.setenv(BASE_URL_ENV, _CONFIGURED)
    return TestClient(app, root_path="/ocs")


def test_the_manage_console_posts_under_the_mount_prefix(mounted_client: TestClient) -> None:
    """A page served at `/ocs/manage` posting to `/manage/ingest` reached the proxy, not the
    app, and returned 404."""
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


def test_mount_prefix_strips_a_trailing_slash() -> None:
    """`f"{mount_prefix(request)}/manage"` has to be right for every form the server reports."""
    from open_climate_service.shared.urls import mount_prefix

    for reported, expected in (("/ocs/", "/ocs"), ("/ocs", "/ocs"), ("/", ""), ("", "")):
        request = _fake_request("http://host/", "/manage")
        request.scope["root_path"] = reported
        assert mount_prefix(request) == expected, reported
