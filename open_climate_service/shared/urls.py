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

    For links and form actions *within* a served page, which need to be same-origin but
    mount-correct. A path carries no scheme and no host, so it inherits both from the page —
    which is what fixes the mixed-content defect — while still resolving under a deployment
    prefix.

    The two wrong answers this sits between, both of which shipped in this file's history:

    * `absolute_base()` names an origin, so an operator reaching the console through a
      port-forward would submit forms to the configured *public* instance. Silently, since CORS
      is `allow_origins=["*"]`.
    * A bare leading slash resolves from the origin root, so under `--root-path /ocs` a page at
      `/ocs/manage` posted to `/manage` and the proxy returned 404.

    Returned without a trailing slash, so `f"{mount_prefix(request)}/manage"` is right whether
    or not there is a prefix.
    """
    return str(request.scope.get("root_path", "")).rstrip("/")


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
    a STAC client following `self` would 404. Verified against uvicorn booted with
    `root_path="/ocs"`: with the strip `self` is `.../ocs/stac`, without it `.../ocs/ocs/stac`.

    `TestClient(root_path=...)` does not prepend the prefix to `path` the way uvicorn does, so
    the `mounted_client` tests cannot reproduce the doubling. The coverage lives in
    `test_the_self_link_does_not_double_a_mount_prefix`, which builds the scope by hand and
    unsets the configured origin so the fallback is exercised; removing the strip fails it.
    """
    path = request.url.path
    root_path = request.scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        path = path[len(root_path) :] or "/"
    return absolute_url(request, path)
