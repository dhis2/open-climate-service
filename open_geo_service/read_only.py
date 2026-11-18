"""Read-only mode: refuse every route above the public tier when ``read_only: true`` is configured.

Serves a public instance that can be browsed but not modified. The decision is read from the
route table (`auth/route_table.py`), so a route is never public to one check and gated to the
other, and the access middleware applies it before any credential is looked at: no token
lifts read-only mode.

``POST /result`` is the one exception. It is a POST only because the process graph travels in
the request body, and it cannot publish a dataset: the synchronous handler rejects ZARR output
(``openeo/routes.py``), so the only route that writes to the managed store is the batch-job path,
which read-only closes. Read-only never changes a route's tier, so with authentication on,
``POST /result`` still needs a token.

Read-only governs HTTP only. Ingestion on a read-only instance is an operator task run on the
host. There is deliberately **no** loopback or trusted-CIDR escape hatch: behind a reverse proxy
on the same host, every public request arrives from ``127.0.0.1``, so such an exemption would
silently open the instance to everyone while appearing to work in testing.
"""

from __future__ import annotations

from open_geo_service.auth.route_table import route_tier
from open_geo_service.auth.tokens import Tier

MESSAGE = (
    "This instance is read-only: {method} {path} is not available. "
    "Data can be browsed via /collections, /stac/catalog.json, /zarr and /datasets, and process graphs "
    "can be executed synchronously via POST /result."
)


def is_blocked(method: str, path: str) -> bool:
    """Return True when read-only mode should refuse this request.

    A pure function of the request line, so the capabilities document and the API page can
    leave out what the instance will refuse.
    """
    method = method.upper()
    # A refused preflight surfaces in the browser as an opaque CORS failure instead of the 403.
    if method == "OPTIONS":
        return False
    if method == "POST" and (path.rstrip("/") or "/") == "/result":
        return False
    return route_tier(method, path) > Tier.PUBLIC
