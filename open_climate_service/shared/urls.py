"""The public origin and mount prefix for the URLs OCS puts into documents it serves.

Behind a TLS-terminating reverse proxy the request uvicorn sees is plain HTTP, so any absolute
URL built from `request.base_url` carries the wrong scheme for the outside world (CLIM-974).
An HTTPS page fetching those URLs is active mixed content: the browser blocks the requests and
the map viewer renders an empty catalogue, while curl against every endpoint returns 200.

Two rules, one per kind of URL:

- A URL that leaves the process (STAC links, openEO capabilities, the `/openeo` redirect) is
  absolute and names the public service root: `absolute_base`, `absolute_url`, `self_url`.
- A URL used inside a page the browser is already on (form actions, redirects, in-page fetches)
  is a path with the mount prefix and no origin: `mount_prefix`. It inherits scheme and host
  from the page, so an operator on a port-forward never submits a form to the public instance.

Two settings feed them. `ROOT_PATH` is the deployment prefix and sets ASGI `root_path`.
`CLIMATE_SERVICE_BASE_URL` is the public origin, needed when the proxy's forwarded headers are
not trusted; a path in it is the service root as the outside world addresses it, and when it
has none the ASGI prefix is appended, so the two settings compose.

Use these helpers rather than `request.base_url` or `request.url` for anything that leaves the
process, and `route_path` rather than `request.url.path` for anything that matches a path
against a route.
"""

import functools
import logging
import os
import urllib.parse
from typing import NamedTuple

from fastapi import Request

logger = logging.getLogger(__name__)

BASE_URL_ENV = "CLIMATE_SERVICE_BASE_URL"
ROOT_PATH_ENV = "ROOT_PATH"


class ConfiguredBase(NamedTuple):
    """`CLIMATE_SERVICE_BASE_URL` split into its origin and its path, each `""` when absent."""

    origin: str
    path: str


_UNUSABLE = (
    "%s=%r is not a usable absolute URL (needs a scheme and a host); "
    "falling back to the request origin, so absolute URLs will name the internal address"
)
_HAS_QUERY = "%s=%r carries a query string or fragment; ignoring them, since a path is appended to this value"

_warned_base_urls: set[str] = set()


def _forget_base_url_warnings() -> None:
    """Forget which values have been warned about, so a test starts from silence."""
    _warned_base_urls.clear()


@functools.lru_cache(maxsize=8)
def _split_configured_base(raw: str) -> tuple[ConfiguredBase, str]:
    """Split the raw value, and name the problem with it rather than reporting it.

    Pure and cached, so it is only ever memoisation. Reporting lives in
    `_parse_configured_base`, because a warning emitted from inside a cache appears or not
    depending on whether the entry survived, which is not something a caller can reason about.
    """
    value = raw.strip()
    if not value:
        return ConfiguredBase("", ""), ""
    split = urllib.parse.urlsplit(value)
    if not split.scheme or not split.netloc:
        return ConfiguredBase("", ""), _UNUSABLE
    base = ConfiguredBase(f"{split.scheme}://{split.netloc}", split.path.rstrip("/"))
    return base, (_HAS_QUERY if split.query or split.fragment else "")


def _parse_configured_base(raw: str) -> ConfiguredBase:
    """Parse `CLIMATE_SERVICE_BASE_URL`, refusing a value with no scheme or no host.

    Parsed rather than trimmed so that every consumer reads the value the same way and a query
    string or fragment cannot end up in the middle of a link. A schemeless value would give
    links a browser resolves as relative paths, so it is refused and the request origin is
    used instead, with a warning, once per distinct value.
    """
    base, problem = _split_configured_base(raw)
    if problem and raw not in _warned_base_urls:
        _warned_base_urls.add(raw)
        logger.warning(problem, BASE_URL_ENV, raw.strip())
    return base


def configured_base() -> str:
    """The normalised configured public root, or `""` when unset or unusable."""
    origin, path = _parse_configured_base(os.getenv(BASE_URL_ENV, ""))
    return origin + path


def configured_root_path() -> str:
    """`ROOT_PATH` normalised to a leading slash and no trailing one, or `""` when unset.

    Read by `create_app`, so it applies under every launcher rather than only the
    `climate-service` entry point. Normalised because uvicorn joins `root_path + path`
    verbatim: a trailing slash would put `/ocs//manage` on the scope, which routes but no
    longer matches a closed prefix in read-only mode.
    """
    value = os.getenv(ROOT_PATH_ENV, "").strip().strip("/")
    return f"/{value}" if value else ""


def asgi_prefix(request: Request) -> str:
    """The ASGI `root_path` without a trailing slash, or `""` when served at the root.

    Never consults `CLIMATE_SERVICE_BASE_URL`: that variable states a public origin, and its
    path is not necessarily an ASGI mount.
    """
    return str(request.scope.get("root_path", "")).rstrip("/")


def mount_prefix(request: Request) -> str:
    """The prefix for links and form actions within a served page, or `""` at the root.

    ASGI `root_path` wins. The fallback is the path of `CLIMATE_SERVICE_BASE_URL`, for
    deployments that declare the prefix only there; it cannot tell a proxied request from a
    direct one, so it prefixes links on a port-forward too, which is why `ROOT_PATH` is the
    preferred setting.
    """
    return asgi_prefix(request) or _parse_configured_base(os.getenv(BASE_URL_ENV, "")).path


def absolute_base(request: Request) -> str:
    """The public service root, origin and mount prefix, without a trailing slash.

    The configured origin wins over the request's. The prefix is the configured path when
    there is one, else the ASGI prefix, so `ROOT_PATH` with an origin-only base URL still
    yields links under the prefix. Built from the scope's scheme and host plus `root_path`
    rather than from `request.base_url`, which Starlette derives from the outer app's root
    under an embedding `Mount` and so may or may not already carry the prefix.
    """
    origin, path = _parse_configured_base(os.getenv(BASE_URL_ENV, ""))
    prefix = path or asgi_prefix(request)
    if origin:
        return origin + prefix
    return f"{request.url.scheme}://{request.url.netloc}{prefix}"


def absolute_url(request: Request, path: str) -> str:
    """`path` resolved against the public service root."""
    return f"{absolute_base(request)}/{path.lstrip('/')}"


def strip_mount(path: str, prefix: str) -> str:
    """`path` with `prefix` removed, only when it ends on a path-segment boundary.

    The same rule Starlette routes by: `/stac` under a `/st` prefix stays `/stac`. Given the
    raw `root_path`, a trailing slash in it consumes the slash that uvicorn's `root_path + path`
    join doubles, so `/ocs//manage` under `/ocs/` is `/manage`. Returns `/` rather than `""`
    for an exact match.
    """
    if not prefix or not path.startswith(prefix):
        return path
    remainder = path[len(prefix) :]
    if remainder and not remainder.startswith("/"):
        return path
    return remainder or "/"


def route_path(request: Request) -> str:
    """The request path as the app's routes see it: `request.url.path` less the ASGI prefix.

    Uvicorn sets `scope["path"] = root_path + path`, so under `--root-path /ocs` a request for
    `/manage` arrives as `/ocs/manage`. Anything that matches the incoming path against a
    declared route must use this rather than `request.url.path`, or it fails to match under a
    prefix. The raw `root_path` is used, not `asgi_prefix`, so that a trailing slash in it
    strips the same way Starlette strips it.
    """
    return strip_mount(request.url.path, str(request.scope.get("root_path", "")))


def self_url(request: Request) -> str:
    """The public URL of the document being requested, without its query string.

    The path comes from the request, so `/stac` and `/stac/catalog.json` each name themselves,
    and the origin and prefix come from `absolute_base`. No OCS document varies by query
    string, so it is dropped rather than reflected into a served document.
    """
    return absolute_url(request, route_path(request))
