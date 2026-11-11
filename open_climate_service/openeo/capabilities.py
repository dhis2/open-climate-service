"""openEO capabilities document builder."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from open_climate_service import config as api_config
from open_climate_service.openeo.schemas import OpenEOCapabilities, OpenEOEndpoint

API_VERSION = "1.2.0"
STAC_VERSION = "1.1.0"


def _readable_endpoints(endpoints: list[OpenEOEndpoint]) -> list[OpenEOEndpoint]:
    """Drop methods and paths that read-only mode refuses, keeping the rest intact.

    Derived from the read-only policy itself rather than a second hand-maintained list, so
    the advertised surface cannot drift from the enforced one.
    """
    from open_climate_service.read_only import is_blocked

    filtered: list[OpenEOEndpoint] = []
    for endpoint in endpoints:
        allowed = [method for method in endpoint.methods if not is_blocked(method, endpoint.path)]
        if allowed:
            filtered.append(OpenEOEndpoint(path=endpoint.path, methods=allowed))
    return filtered


def build_capabilities(base_url: str) -> OpenEOCapabilities:
    """Return the openEO capabilities document."""
    backend_version = _pkg_version("open-climate-service")
    name = api_config.get_name()

    endpoints = [
        OpenEOEndpoint(path="/", methods=["GET"]),
        OpenEOEndpoint(path="/credentials/oidc", methods=["GET"]),
        OpenEOEndpoint(path="/file_formats", methods=["GET"]),
        OpenEOEndpoint(path="/service_types", methods=["GET"]),
        OpenEOEndpoint(path="/me", methods=["GET"]),
        OpenEOEndpoint(path="/collections", methods=["GET"]),
        OpenEOEndpoint(path="/collections/{collection_id}", methods=["GET"]),
        OpenEOEndpoint(path="/processes", methods=["GET"]),
        OpenEOEndpoint(path="/process_graphs", methods=["GET"]),
        OpenEOEndpoint(path="/process_graphs/{process_graph_id}", methods=["GET", "PUT", "DELETE"]),
        OpenEOEndpoint(path="/jobs", methods=["GET", "POST"]),
        OpenEOEndpoint(path="/jobs/{job_id}", methods=["GET", "PATCH", "DELETE"]),
        OpenEOEndpoint(path="/jobs/{job_id}/results", methods=["GET", "POST", "DELETE"]),
        OpenEOEndpoint(path="/result", methods=["POST"]),
        OpenEOEndpoint(path="/health", methods=["GET"]),
    ]

    # A read-only instance must not advertise what it will refuse: openEO clients build
    # their capability set from this list, so leaving the write endpoints in would have
    # them offer batch jobs and UDP storage that then 403. The spec permits a backend to
    # declare only the endpoints it supports.
    if api_config.is_read_only():
        endpoints = _readable_endpoints(endpoints)

    links = [
        {"rel": "self", "href": base_url, "type": "application/json"},
        {"rel": "version-history", "href": "https://openeo.org/documentation/1.0/", "type": "text/html"},
        {
            "rel": "data",
            "href": f"{base_url}/collections",
            "type": "application/json",
            "title": "Published datasets",
        },
        {
            "rel": "related",
            "href": f"{base_url}/stac/catalog.json",
            "type": "application/json",
            "title": "STAC catalog",
        },
    ]

    return OpenEOCapabilities(
        api_version=API_VERSION,
        backend_version=backend_version,
        stac_version=STAC_VERSION,
        id=api_config.get_id(),
        title=name,
        description=(
            "openEO-compatible backend for the Open Climate Service. "
            "Access and process ERA5-Land climate datasets for health and climate analysis."
        ),
        production=False,
        endpoints=endpoints,
        links=links,
    )
