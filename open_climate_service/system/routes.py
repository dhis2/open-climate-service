"""Root API endpoints."""

import asyncio
import json
import sys
import urllib.parse
from collections.abc import AsyncIterator
from importlib.metadata import version as _pkg_version
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.responses import RedirectResponse, StreamingResponse

from open_climate_service import config as api_config
from open_climate_service.shared.urls import absolute_base, mount_prefix

from .schemas import AppInfo, HealthStatus, Status
from .templates import (
    ROOT_RESPONSES,
    app_version,
    render_landing,
    render_manage,
    render_maps,
    wants_json,
)

router = APIRouter()


async def _sse_events(queue: asyncio.Queue[dict[str, Any] | None]) -> AsyncIterator[str]:
    while True:
        item = await queue.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"


@router.get("/", response_class=Response, responses=ROOT_RESPONSES)
def read_index(request: Request) -> Response:
    """Return openEO capabilities (JSON) or the landing page (HTML)."""
    if wants_json(request):
        from open_climate_service.openeo.capabilities import build_capabilities

        # openEO capabilities are consumed by other processes, so their links must be absolute
        # and must name the public origin. The HTML page is navigated in a browser that is
        # already on the right origin, so it uses relative paths instead.
        caps = build_capabilities(absolute_base(request))
        return JSONResponse(caps.model_dump())
    return HTMLResponse(render_landing(app_version, mount_prefix(request)))


@router.get("/map", response_class=HTMLResponse, include_in_schema=False)
def maps(request: Request) -> HTMLResponse:
    """Return the interactive map viewer."""
    return HTMLResponse(render_maps(mount_prefix(request)))


@router.get("/openeo", response_class=HTMLResponse, include_in_schema=False)
def openeo_editor(request: Request) -> RedirectResponse:
    """Redirect to the openEO Web Editor pre-connected to this backend."""
    base = absolute_base(request)
    params = urllib.parse.urlencode({"server": base, "server-title": api_config.get_name()})
    return RedirectResponse(f"https://editor.openeo.org/?{params}", status_code=302)


@router.get("/manage", response_class=HTMLResponse, include_in_schema=False)
def manage(
    request: Request,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Return the management interface for ingestion and sync operations."""
    return HTMLResponse(render_manage(app_version, mount_prefix(request), message=message, error=error))


def _refusal(status_code: int, message: str) -> JSONResponse:
    """The answer to an ingest or sync request that cannot start.

    JSON rather than a stream, so the page that posted can show the message in place: the
    streams below only begin once the request is known to be runnable.
    """
    return JSONResponse(status_code=status_code, content={"error": message})


def _job_stream(work: Any, finished_message: str) -> StreamingResponse:
    """Run *work* in a thread and stream its progress as server-sent events.

    Events are progress updates (`done`, `total`, `message`), then exactly one of `finished`
    (with a message) or `error`. The stream then ends.
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(done: int | None, total: int | None, message: str | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"done": done, "total": total, "message": message})

    async def run() -> None:
        from fastapi import HTTPException

        try:
            await asyncio.to_thread(lambda: work(on_progress))
            loop.call_soon_threadsafe(queue.put_nowait, {"finished": True, "message": finished_message})
        except HTTPException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc.detail)})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(run())
    return StreamingResponse(_sse_events(queue), media_type="text/event-stream")


@router.post("/manage/ingest", include_in_schema=False)
async def manage_ingest(request: Request) -> Response:
    """Ingest a dataset template from its page's form, streaming progress via SSE."""
    from fastapi import HTTPException

    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.data_registry.services.datasets import get_dataset
    from open_climate_service.extents.services import get_extent_or_404
    from open_climate_service.ingestions.services import create_artifact, ensure_ingestable

    try:
        form = await request.form()
        dataset_id = str(form.get("dataset_id", "")).strip()
        # A blank field submits "" rather than being absent, so normalise to None.
        start = str(form.get("start", "")).strip() or None
        end = str(form.get("end", "")).strip() or None
        publish = "publish" in form
        overwrite = "overwrite" in form

        template = get_dataset(dataset_id)
        if template is None:
            return _refusal(404, f"Dataset template '{dataset_id}' not found")

        # Both checks belong here rather than inside create_artifact: the work below runs in an
        # event stream, where a refusal arrives as an event after a 200 instead of as the
        # refusal it is. A workflow output has nothing to fetch from...
        ensure_ingestable(template)
        # ...and only a forecast may leave the start blank.
        if start is None and not registry_datasets.is_future_facing(template):
            return _refusal(400, f"Start period is required for '{dataset_id}': its periods are not in the future")

        extent = get_extent_or_404()
        resolved_bbox = list(extent["bbox"])
        country_code = extent.get("country_code")
    except HTTPException as exc:
        return _refusal(exc.status_code, str(exc.detail))
    except Exception as exc:
        return _refusal(400, str(exc))

    return _job_stream(
        lambda on_progress: create_artifact(
            dataset=template,
            start=start,
            end=end,
            bbox=resolved_bbox,
            country_code=country_code,
            overwrite=overwrite,
            publish=publish,
            on_progress=on_progress,
        ),
        f"Ingested {template.get('name', dataset_id)}",
    )


@router.post("/manage/sync", include_in_schema=False)
async def manage_sync(request: Request) -> Response:
    """Sync a dataset from its page's form, streaming progress via SSE."""
    from fastapi import HTTPException

    from open_climate_service.ingestions.services import get_latest_artifact_for_dataset_or_404, sync_dataset

    try:
        form = await request.form()
        dataset_id = str(form.get("dataset_id", "")).strip()
        end = str(form.get("end", "")).strip() or None
        publish = "publish" in form
        if not dataset_id:
            return _refusal(400, "Dataset ID is required")
        # Resolve the dataset before the stream opens, for the same reason as ingest: an id
        # that names nothing is a client mistake, and inside the stream it would reach the page
        # as an error event on a 200.
        get_latest_artifact_for_dataset_or_404(dataset_id)
    except HTTPException as exc:
        return _refusal(exc.status_code, str(exc.detail))
    except Exception as exc:
        return _refusal(400, str(exc))

    return _job_stream(
        lambda on_progress: sync_dataset(dataset_id=dataset_id, end=end, publish=publish, on_progress=on_progress),
        "Sync completed",
    )


@router.get("/health")
def health() -> HealthStatus:
    """Return health status for container health checks."""
    return HealthStatus(status=Status.HEALTHY)


@router.get("/info")
def info() -> AppInfo:
    """Return application version and environment info."""
    return AppInfo(
        app_version=_pkg_version("open-climate-service"),
        python_version=sys.version,
        uvicorn_version=_pkg_version("uvicorn"),
        read_only=api_config.is_read_only(),
    )
