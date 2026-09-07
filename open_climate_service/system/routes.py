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


def _manage_url(mount: str, *, error: str | None = None, message: str | None = None) -> str:
    """A `/manage` URL carrying one banner, with the text percent-encoded.

    One place rather than at each redirect: a prefix added to most of them and missed on one is
    a bug that only shows on the path nobody re-tests, and the miss landed on the *success*
    redirect while every error path around it stayed correct.
    """
    banner = "error" if error is not None else "message"
    text = error if error is not None else message
    return f"{mount}/manage?{banner}={urllib.parse.quote(str(text))}"


@router.post("/manage/ingest", include_in_schema=False)
async def manage_ingest(request: Request) -> Response:
    """Handle ingest form submission and stream progress via SSE."""
    from fastapi import HTTPException

    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.data_registry.services.datasets import get_dataset
    from open_climate_service.extents.services import get_extent_or_404
    from open_climate_service.ingestions.services import create_artifact

    mount = mount_prefix(request)
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
            return RedirectResponse(
                _manage_url(mount, error=f"Dataset template '{dataset_id}' not found"), status_code=303
            )

        # Validate the blank start here rather than leaving it to create_artifact. The work
        # below runs inside an SSE stream, and a response that has already begun cannot
        # redirect — the operator would get a progress bar that fails mid-flight instead of
        # the error banner. Only a forecast may omit it (see temporal_direction).
        if start is None and not registry_datasets.is_future_facing(template):
            return RedirectResponse(
                _manage_url(
                    mount,
                    error=f"Start period is required for '{dataset_id}': its periods are not in the future",
                ),
                status_code=303,
            )

        extent = get_extent_or_404()
        resolved_bbox = list(extent["bbox"])
        country_code = extent.get("country_code")
    except HTTPException as exc:
        return RedirectResponse(_manage_url(mount, error=str(exc.detail)), status_code=303)
    except Exception as exc:
        return RedirectResponse(_manage_url(mount, error=str(exc)), status_code=303)

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(done: int | None, total: int | None, message: str | None) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"done": done, "total": total, "message": message},
        )

    async def run() -> None:
        try:
            await asyncio.to_thread(
                lambda: create_artifact(
                    dataset=template,
                    start=start,
                    end=end,
                    bbox=resolved_bbox,
                    country_code=country_code,
                    overwrite=overwrite,
                    publish=publish,
                    on_progress=on_progress,
                )
            )
            name = str(template.get("name", dataset_id))
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"redirect": _manage_url(mount, message=f"Ingested {name}")},
            )
        except HTTPException as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"error": str(exc.detail), "redirect": _manage_url(mount, error=str(exc.detail))},
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"error": str(exc), "redirect": _manage_url(mount, error=str(exc))},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(run())
    return StreamingResponse(_sse_events(queue), media_type="text/event-stream")


@router.post("/manage/sync", include_in_schema=False)
async def manage_sync(request: Request) -> Response:
    """Handle sync form submission and stream progress via SSE."""
    from fastapi import HTTPException

    from open_climate_service.ingestions.services import sync_dataset

    mount = mount_prefix(request)
    try:
        form = await request.form()
        dataset_id = str(form.get("dataset_id", "")).strip()
        end = str(form.get("end", "")).strip() or None
        publish = "publish" in form

        if not dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID is required")
    except HTTPException as exc:
        return RedirectResponse(_manage_url(mount, error=str(exc.detail)), status_code=303)
    except Exception as exc:
        return RedirectResponse(_manage_url(mount, error=str(exc)), status_code=303)

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(done: int | None, total: int | None, message: str | None) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"done": done, "total": total, "message": message},
        )

    async def run() -> None:
        try:
            await asyncio.to_thread(
                lambda: sync_dataset(dataset_id=dataset_id, end=end, publish=publish, on_progress=on_progress)
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"redirect": _manage_url(mount, message="Sync completed")},
            )
        except HTTPException as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"error": str(exc.detail), "redirect": _manage_url(mount, error=str(exc.detail))},
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"error": str(exc), "redirect": _manage_url(mount, error=str(exc))},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(run())
    return StreamingResponse(_sse_events(queue), media_type="text/event-stream")


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
