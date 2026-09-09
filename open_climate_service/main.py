"""Open Climate Service -- Climate and earth observation data API for DHIS2."""

import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import open_climate_service.startup  # noqa: F401  # pyright: ignore[reportUnusedImport]
from open_climate_service.automation.service import get_workflow_automation_service
from open_climate_service.data_registry import routes as dataset_template_routes
from open_climate_service.extents import routes as extent_routes
from open_climate_service.ingestions import routes as ingestion_routes
from open_climate_service.jobs.service import get_job_service
from open_climate_service.openeo import routes as openeo_routes
from open_climate_service.openeo.jobs import get_openeo_job_service
from open_climate_service.read_only import read_only_middleware
from open_climate_service.scheduler import routes as scheduler_routes
from open_climate_service.scheduler.service import get_scheduler_service
from open_climate_service.stac import routes as stac_routes
from open_climate_service.system import routes as system_routes

logger = logging.getLogger(__name__)

# Hosted browser tools that read a Zarr store directly over HTTP. They are the reason a local
# instance needs any cross-origin story at all: each is a public page fetching a store that may
# be on localhost, which browsers gate behind the private/local network rules handled below.
DEFAULT_ZARR_BROWSER_ORIGINS = (
    "https://inspect.geozarr.org",
    # source-cooperative/zarr-viewer, the reference viewer for GeoZarr stores (CLIM-852).
    # Omitting it is a dead end an operator cannot diagnose: the viewer reports "your connection
    # looks slow or unstable" for what is actually a blocked cross-origin request.
    "https://source-cooperative.github.io",
)


def _zarr_browser_access_origins() -> set[str]:
    """Return allowlisted remote origins permitted to inspect local Zarr endpoints."""
    raw = os.getenv("CLIMATE_SERVICE_ZARR_BROWSER_ORIGINS", ",".join(DEFAULT_ZARR_BROWSER_ORIGINS))
    return {origin.strip() for origin in raw.split(",") if origin.strip()}


def _pna_trusted_origins() -> set[str]:
    """Return origins allowed to make Private Network Access requests.

    Defaults to the openEO editor and the Zarr inspector.  Override with the
    CLIMATE_SERVICE_PNA_ORIGINS environment variable (comma-separated).
    """
    default = "https://editor.openeo.org," + os.getenv(
        "CLIMATE_SERVICE_ZARR_BROWSER_ORIGINS", ",".join(DEFAULT_ZARR_BROWSER_ORIGINS)
    )
    raw = os.getenv("CLIMATE_SERVICE_PNA_ORIGINS", default)
    return {o.strip() for o in raw.split(",") if o.strip()}


def _append_vary_value(response: Response, value: str) -> None:
    """Append one token to the Vary header without clobbering existing values."""
    existing = response.headers.get("Vary")
    if existing is None:
        response.headers["Vary"] = value
        return

    values = [item.strip() for item in existing.split(",") if item.strip()]
    if value not in values:
        response.headers["Vary"] = ", ".join([*values, value])


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Run lightweight startup recovery hooks for the application lifecycle."""
    from open_climate_service.plugins_diagnostics import log_plugin_loading

    log_plugin_loading()
    job_service = get_job_service()
    job_service.recover_pending_jobs()
    openeo_service = get_openeo_job_service()
    openeo_service.recover_pending_jobs()
    automation_service = get_workflow_automation_service()
    automation_service.start()
    job_service.set_event_consumer(automation_service.consume)
    try:
        automation_service.replay()
    except Exception:
        # Replay must not take down startup: a partial replay is idempotent and the consumer
        # above still handles every newly persisted event.
        logger.exception("Workflow automation replay failed; continuing startup")
    scheduler_service = get_scheduler_service()
    scheduler_service.start()
    try:
        yield
    finally:
        job_service.set_event_consumer(None)
        scheduler_service.shutdown()
        job_service.shutdown()
        openeo_service.shutdown()


def create_app() -> FastAPI:
    """Create and configure the Open Climate Service FastAPI application.

    This is the public entry point for embedding the API in a larger application:

        from open_climate_service.main import create_app
        app = create_app()
    """
    _app = FastAPI(lifespan=_lifespan)

    # Registered *before* CORS so it ends up innermost: Starlette applies the most recently
    # added middleware outermost, so CORS wraps this and a 403 still carries the headers a
    # browser needs in order to read it, rather than surfacing as an opaque network error.
    # The config is read per request, so the flag can be flipped without rebuilding the app.
    _app.middleware("http")(read_only_middleware)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @_app.middleware("http")
    async def add_browser_access_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Handle private/local network preflights and Zarr browser CORS headers.

        Browsers gate a public-origin page (editor.openeo.org, a hosted Zarr viewer) fetching a
        localhost resource, and the server has to opt in. Chrome has shipped two spellings of
        that opt-in as the spec evolved — the older Private Network Access
        (``Access-Control-Request/Allow-Private-Network``) and the newer Local Network Access
        (``...-Local-Network-Access``) — so both are answered here rather than betting on one.

        A server header is necessary but, on current Chrome, no longer sufficient: reaching the
        loopback address space also needs a user permission grant, refused by default. When that
        is what blocks a request the console says "Permission was denied for this request to
        access the `loopback` address space" — distinct from a missing-header CORS failure, and
        not something the server can fix. See docs/zarr_and_geozarr.md.
        """
        origin = request.headers.get("origin")
        is_pna_preflight = (
            request.method == "OPTIONS"
            and (
                request.headers.get("access-control-request-private-network") == "true"
                or request.headers.get("access-control-request-local-network-access") == "true"
            )
            and origin is not None
        )

        # Short-circuit PNA preflight for trusted origins (openEO editor, Zarr inspectors).
        if is_pna_preflight and origin in _pna_trusted_origins():
            response = Response(status_code=200)
            response.headers["Access-Control-Allow-Origin"] = str(origin)
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Access-Control-Allow-Local-Network-Access"] = "true"
            response.headers["Access-Control-Allow-Methods"] = request.headers.get(
                "access-control-request-method", "GET, POST, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = request.headers.get(
                "access-control-request-headers", "*"
            )
            response.headers["Vary"] = "Origin"
            return response

        response = await call_next(request)

        # Extra CORS + PNA headers for Zarr inspector origins on /zarr paths.
        allowed_zarr_origin = origin if origin in _zarr_browser_access_origins() else None
        if allowed_zarr_origin and (request.url.path == "/zarr" or request.url.path.startswith("/zarr/")):
            response.headers["Access-Control-Allow-Origin"] = allowed_zarr_origin
            _append_vary_value(response, "Origin")
            response.headers.setdefault("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            response.headers.setdefault(
                "Access-Control-Allow-Headers",
                request.headers.get("access-control-request-headers", "*"),
            )
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Access-Control-Allow-Local-Network-Access"] = "true"

        return response

    _app.include_router(system_routes.router, tags=["System"])
    _app.include_router(openeo_routes.capabilities_router, tags=["openEO"])
    _app.include_router(stac_routes.router, prefix="/stac", tags=["STAC"])
    _app.include_router(openeo_routes.collections_router, prefix="/collections", tags=["openEO"])
    _app.include_router(openeo_routes.jobs_router, prefix="/jobs", tags=["openEO"])
    _app.include_router(openeo_routes.udp_router, prefix="/process_graphs", tags=["openEO"])
    _app.include_router(openeo_routes.result_router, prefix="/result", tags=["openEO"])
    _app.include_router(extent_routes.router, prefix="/extent", tags=["Extent"])
    _app.include_router(dataset_template_routes.router, prefix="/dataset-templates", tags=["Dataset templates"])
    _app.include_router(ingestion_routes.datasets_router, prefix="/datasets", tags=["Datasets"])
    _app.include_router(ingestion_routes.ingestions_router, prefix="/ingestions", tags=["Ingestions"])
    _app.include_router(ingestion_routes.zarr_router, prefix="/zarr", tags=["Zarr"])
    _app.include_router(ingestion_routes.icechunk_router, prefix="/icechunk", tags=["Icechunk"])
    _app.include_router(ingestion_routes.sync_router, prefix="/sync", tags=["Sync"])
    _app.include_router(scheduler_routes.router, prefix="/schedules", tags=["Schedules"])
    _app.include_router(openeo_routes.processes_router, prefix="/processes", tags=["openEO"])

    return _app


app = create_app()
