"""The public origin for absolute URLs OCS puts into documents it serves.

Behind a TLS-terminating reverse proxy, the request uvicorn sees is plain HTTP: the proxy
speaks HTTPS to the client and forwards over HTTP. So `request.base_url` reports the `http`
scheme even though every client reached the service over `https`, and any absolute URL built
from the request carries a scheme that is wrong for the outside world (CLIM-974).

`CLIMATE_SERVICE_BASE_URL` is the operator's statement of the public origin, so it wins.
The request is the fallback for a direct deployment where nothing is configured, which is
the local-development case.

The failure this prevents is easy to misread. An HTTPS page fetching `http://` URLs is
active mixed content, so the browser blocks the requests and the map viewer renders an empty
catalogue — while curl against every endpoint still returns 200, because the endpoints were
never the problem. HSTS masks it completely: the browser rewrites the scheme before the
request leaves, so a deployment with HSTS at the edge looks correct and only clients that
have not yet seen the HSTS header (and non-browser STAC clients, which do not implement it
at all) see the defect.

Use these helpers rather than `request.base_url` or `request.url` for anything that leaves
the process. Reading the environment variable at each call site is the same one-line
expression repeated, and the sites that forgot it are exactly the bug.
"""

import os
import urllib.parse

from fastapi import Request

BASE_URL_ENV = "CLIMATE_SERVICE_BASE_URL"


def absolute_base(request: Request) -> str:
    """The configured public origin, or the request's own, without a trailing slash.

    Trailing slashes are stripped *before* the value is judged usable. Testing truthiness
    first accepted `CLIMATE_SERVICE_BASE_URL="/"`, stripped it to the empty string and
    returned that, so every href in every served document lost its origin.
    """
    configured = os.getenv(BASE_URL_ENV, "").strip().rstrip("/")
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


def absolute_url(request: Request, path: str) -> str:
    """`path` resolved against the public origin."""
    return f"{absolute_base(request)}/{path.lstrip('/')}"


def mount_prefix(request: Request) -> str:
    """The path prefix this instance is served under, or `""` when it is at the root.

    For links and form actions *within* a served page: a path carries no scheme and no host, so
    it inherits both from the page — which is what fixes the mixed-content defect — while still
    resolving under a deployment prefix. An origin would not, since an operator on a port-forward
    would then submit forms to the configured public instance.

    Two places can carry the prefix and this reconciles them, so callers have one answer. ASGI
    `root_path` wins when set. Nothing in the shipped entry points sets it — `cli.py` passes only
    host and port — so the usual case is a proxy prefix declared solely in the path of
    `CLIMATE_SERVICE_BASE_URL`, and that is the fallback. Without it, an instance behind
    `https://host/ocs/` renders `/map`, which 404s at the proxy.

    Returned without a trailing slash, so `f"{mount_prefix(request)}/manage"` is right whether
    or not there is a prefix.
    """
    root_path = str(request.scope.get("root_path", "")).rstrip("/")
    if root_path:
        return root_path
    configured = os.getenv(BASE_URL_ENV, "").strip().rstrip("/")
    if not configured:
        return ""
    return urllib.parse.urlsplit(configured).path.rstrip("/")


def strip_mount(path: str, prefix: str) -> str:
    """`path` with `prefix` removed, only when it ends on a path-segment boundary.

    `startswith` alone would turn `/stac` under a `/st` prefix into `/ac`. Unreachable through
    uvicorn, but the guard costs one comparison and mirrors what Starlette's own
    `get_route_path` does.
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

    `root_path` is removed before the path is appended, and this is **required in the ordinary
    deployment** rather than a defensive measure. Uvicorn sets `scope["path"] = root_path + path`
    (`uvicorn/protocols/http/h11_impl.py`), so behind a normal stripping proxy with
    `--root-path /ocs` the request arrives as `/stac`, `request.url.path` becomes `/ocs/stac`,
    and the fallback origin `request.base_url` already ends in `/ocs/`. Appending the path
    unstripped would give `/ocs/ocs/stac` for `self` while every other link stayed correct, and
    a STAC client following `self` would 404.

    `TestClient(root_path=...)` does not prepend the prefix to `path` the way uvicorn does, so
    a test that wants this behaviour has to build the scope by hand and unset the configured
    origin, or it will pass whether or not the strip is here.
    """
    # `app_root_path` rather than `root_path`: `request.base_url` is built from the former, so
    # the strip has to match it or the two disagree. Under `outer.mount("/ocs", create_app())`
    # Starlette sets `root_path` on the inner app while `base_url` keeps the outer prefix, and
    # stripping `root_path` there removed a prefix `base_url` had never added.
    scope = request.scope
    prefix = str(scope.get("app_root_path", scope.get("root_path", "")))
    return absolute_url(request, strip_mount(request.url.path, prefix))
