"""The public origin for absolute URLs OCS puts into documents it serves.

Behind a TLS-terminating reverse proxy, the request uvicorn sees is plain HTTP: the proxy
speaks HTTPS to the client and forwards over HTTP. So `request.base_url` reports the `http`
scheme even though every client reached the service over `https`, and any absolute URL built
from the request carries a scheme that is wrong for the outside world (CLIM-974).

`CLIMATE_SERVICE_BASE_URL` is the operator's statement of the public origin, so it wins.
The request is the fallback for a direct deployment where nothing is configured, which is
the local-development case.

The symptom is easy to misread. An HTTPS page fetching `http://` URLs is active mixed
content, so the browser blocks the requests and the map viewer renders an empty catalogue,
while curl against every endpoint returns 200 — the endpoints are not the problem. HSTS masks
it entirely for clients that have already seen the header, so the deployment can look correct
while non-browser STAC clients, which do not implement HSTS, still fail.

Use these helpers rather than `request.base_url` or `request.url` for anything that leaves the
process, so the rule lives in one place instead of being repeated at each call site.
"""

import functools
import logging
import os
import urllib.parse

from fastapi import Request

logger = logging.getLogger(__name__)

BASE_URL_ENV = "CLIMATE_SERVICE_BASE_URL"


@functools.lru_cache(maxsize=8)
def _parse_configured_base(raw: str) -> str:
    """Normalise `CLIMATE_SERVICE_BASE_URL`, or return `""` when it is unusable.

    Parsed, not string-trimmed, because a path is appended to whatever comes back and every
    consumer must read the value the same way. A surviving query string turns
    `https://host/ocs?x=1` into `https://host/ocs?x=1/stac`.

    A value with no scheme or no host is refused — `host.example` yields a schemeless
    `host.example/stac`, which a browser resolves as a relative path, and `"/"` and `https://`
    yield nothing to build on. Refusing falls back to the request origin: wrong in the way this
    variable exists to fix, but well-formed and logged.

    Cached on the raw string, so the warning appears once per distinct value rather than once
    per request.
    """
    value = raw.strip()
    if not value:
        return ""
    split = urllib.parse.urlsplit(value)
    if not split.scheme or not split.netloc:
        logger.warning(
            "%s=%r is not a usable absolute URL (needs a scheme and a host); "
            "falling back to the request origin, so absolute URLs will name the internal address",
            BASE_URL_ENV,
            value,
        )
        return ""
    if split.query or split.fragment:
        logger.warning(
            "%s=%r carries a query string or fragment; ignoring them, since a path is appended to this value",
            BASE_URL_ENV,
            value,
        )
    return urllib.parse.urlunsplit((split.scheme, split.netloc, split.path.rstrip("/"), "", ""))


def configured_base() -> str:
    """The normalised configured public origin, or `""` when unset or unusable."""
    return _parse_configured_base(os.getenv(BASE_URL_ENV, ""))


def absolute_base(request: Request) -> str:
    """The public service root — origin and mount prefix — without a trailing slash.

    The root of this service *as the outside world addresses it*, so appending a route path
    always yields a working URL. One rule for every caller, so no document can mix a prefixed
    link with an unprefixed one.

    `request.base_url` is the fallback when nothing is configured, and it needs the prefix added:
    Starlette builds it from `app_root_path`, which under an embedding `Mount("/ocs", app)` is
    the outer root with no prefix at all.
    """
    configured = configured_base()
    if configured:
        return configured
    base = str(request.base_url).rstrip("/")
    prefix = asgi_prefix(request)
    if prefix and not base.endswith(prefix):
        base += prefix
    return base


def absolute_url(request: Request, path: str) -> str:
    """`path` resolved against the public origin."""
    return f"{absolute_base(request)}/{path.lstrip('/')}"


def mount_prefix(request: Request) -> str:
    """The path prefix this instance is served under, or `""` when it is at the root.

    For links and form actions *within* a served page: a path carries no scheme and no host, so
    it inherits both from the page — which is what fixes the mixed-content defect — while still
    resolving under a deployment prefix. An origin would not, since an operator on a port-forward
    would then submit forms to the configured public instance.

    Two places can carry the prefix, and this reconciles them so callers have one answer. ASGI
    `root_path` wins, which is what `ROOT_PATH` sets through `cli.py`. The fallback is the path
    of `CLIMATE_SERVICE_BASE_URL`, for deployments that declare the prefix only there: without
    it an instance behind `https://host/ocs/` renders `/map`, which 404s at the proxy. That
    fallback cannot distinguish a proxied request from a direct one, so it prefixes links on a
    port-forward too — a reason to prefer `ROOT_PATH`, not to drop the fallback.

    Returned without a trailing slash, so `f"{mount_prefix(request)}/manage"` is right whether
    or not there is a prefix.
    """
    root_path = str(request.scope.get("root_path", "")).rstrip("/")
    if root_path:
        return root_path
    return urllib.parse.urlsplit(configured_base()).path.rstrip("/")


def asgi_prefix(request: Request) -> str:
    """The prefix the routing layer has put on `request.url.path`, or `""`.

    Distinct from `mount_prefix`, and the two must not be swapped: this one describes the
    *incoming* path, so it is what to remove before matching a path against a route. It never
    consults `CLIMATE_SERVICE_BASE_URL`, because that variable states a public origin and its
    path is not necessarily an ASGI mount — a base URL of `https://host/jobs` would otherwise
    strip `/jobs` off a real route and turn read-only mode's `GET /jobs` into a 200.
    """
    return str(request.scope.get("root_path", "")).rstrip("/")


def strip_mount(path: str, prefix: str) -> str:
    """`path` with `prefix` removed, only when it ends on a path-segment boundary.

    `startswith` alone would turn `/stac` under a `/st` prefix into `/ac`. Unreachable through
    uvicorn, but the guard costs one comparison and mirrors Starlette's own `get_route_path`.
    """
    if not prefix or not path.startswith(prefix):
        return path
    remainder = path[len(prefix) :]
    if remainder and not remainder.startswith("/"):
        return path
    return remainder or "/"


def self_url(request: Request) -> str:
    """The public URL of the document being requested.

    The path comes from the request — `/stac` and `/stac/catalog.json` are the same document
    and each must name itself — while the origin comes from the configuration. The query
    string is deliberately dropped: no OCS document varies by it, and reflecting arbitrary
    client input back into a served document is not worth the trouble it invites.

    The ASGI prefix comes off the path because `absolute_base` supplies it — one rule, whichever
    origin is in play. It is **required in the ordinary deployment**, not defensive: uvicorn sets
    `scope["path"] = root_path + path` (`uvicorn/protocols/http/h11_impl.py`), so behind a
    stripping proxy with `--root-path /ocs` a request for `/stac` arrives with
    `request.url.path == "/ocs/stac"`. Appending that unstripped gives `/ocs/ocs/stac` for `self`
    while every other link stays correct, and a STAC client following `self` 404s.

    `TestClient(root_path=...)` does not prepend the prefix to `path` the way uvicorn does, so a
    test covering this has to build the scope by hand, or it passes either way.
    """
    return absolute_url(request, strip_mount(request.url.path, asgi_prefix(request)))
