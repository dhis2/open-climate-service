"""Read-only mode: refuse state-changing requests when ``read_only: true`` is configured.

Serves a public instance that can be browsed but not modified. The policy is deliberately
expressed as **paths**, not HTTP methods, because method is the wrong axis here:

- ``POST /result`` is a POST only because the process graph travels in the request body.
  It is the instance's compute path and stays open. It cannot publish a dataset — the
  synchronous handler rejects ZARR output outright (``openeo/routes.py``), so the only
  route that writes to the managed store is the batch-job path, which read-only closes.
- ``GET /manage`` is an admin console behind a GET. Filtering by method would leave it
  fully browsable with buttons that fail, which reads as broken rather than locked.

Batch jobs are closed entirely rather than merely made unwritable. There is no request
identity — ``/me`` reports ``anonymous`` for every caller — so the job namespace is shared:
``GET /jobs`` would list every visitor's jobs and ``DELETE /jobs/{id}`` would let anyone
remove anyone else's. Partial protection would leave that intact.

Read-only governs HTTP only. Ingestion on a read-only instance is an operator task run on
the host, which is what lets this be absolute — there is no exemption, token or trusted
header to misconfigure into a bypass. In particular there is deliberately **no** loopback
or trusted-CIDR escape hatch: behind a reverse proxy on the same host, every public request
arrives from ``127.0.0.1``, so such an exemption would silently open the instance to
everyone while appearing to work in testing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.responses import JSONResponse, Response

from open_climate_service import config as api_config
from open_climate_service.shared.urls import asgi_prefix, strip_mount

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Mutating paths that remain open. Matched exactly (after stripping a trailing slash).
_ALLOWED_WRITE_PATHS = frozenset({"/result"})

# Path trees closed to every method, including GET: the admin console and batch jobs.
_CLOSED_PREFIXES = ("/manage", "/jobs")

_MESSAGE = (
    "This instance is read-only: {method} {path} is not available. "
    "Data can be browsed via /collections, /stac, /zarr and /datasets, and process graphs "
    "can be executed synchronously via POST /result."
)


def _normalise(path: str) -> str:
    """Strip a trailing slash so ``/result/`` and ``/result`` match the same rule."""
    return path.rstrip("/") or "/"


def is_blocked(method: str, path: str) -> bool:
    """Return True when read-only mode should refuse this request.

    Pure function of the request line so the policy can be tested without a client, and
    so the route-coverage test can assert every mutating route in the app is decided.
    """
    method = method.upper()
    path = _normalise(path)

    # Never block CORS preflight: the browser must be able to learn that the real request
    # is disallowed, and a rejected preflight surfaces as an opaque CORS failure instead
    # of the 403 below. The actual request is still refused.
    if method == "OPTIONS":
        return False

    if any(path == prefix or path.startswith(f"{prefix}/") for prefix in _CLOSED_PREFIXES):
        return True

    return method in _WRITE_METHODS and path not in _ALLOWED_WRITE_PATHS


async def read_only_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Refuse state-changing and admin requests when the instance is configured read-only.

    The ASGI prefix is removed before the policy sees the path, because `is_blocked` matches
    route paths as the app declares them. Left on, `/ocs/manage` under `--root-path /ocs`
    matches no closed prefix and read-only mode **fails open**, serving the admin console and
    the job listing while `POST /ocs/result` is still refused by the method check — a partial
    failure that looks like a working configuration.

    `asgi_prefix`, never `mount_prefix`: the latter falls back to the path of
    `CLIMATE_SERVICE_BASE_URL`, a statement about the public origin rather than the incoming
    path. A base URL of `https://host/jobs` would strip `/jobs` off the real route and serve the
    job listing with a 200.
    """
    path = strip_mount(request.url.path, asgi_prefix(request))
    if api_config.is_read_only() and is_blocked(request.method, path):
        message = _MESSAGE.format(method=request.method, path=path)
        return JSONResponse(
            status_code=403,
            # `detail` matches FastAPI's HTTPException shape for ordinary HTTP clients;
            # `code`/`message` are the openEO error fields, so openEO clients surface
            # something meaningful rather than a bare status.
            content={"id": "read-only", "code": "PermissionsInsufficient", "message": message, "detail": message},
        )
    return await call_next(request)
