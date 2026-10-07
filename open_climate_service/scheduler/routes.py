"""Operator routes for dataset sync schedules: status, and the stored ones' lifecycle (CLIM-1242).

File-configured schedules are listed but cannot be changed here; a stored one for the same
dataset would be shadowed, so the API refuses to create it. Every change to the store is
followed by a scheduler reload, so a saved schedule takes effect without a restart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler import store
from open_climate_service.scheduler.schemas import ScheduleListResponse, ScheduleStatus
from open_climate_service.scheduler.service import (
    get_scheduler_service,
    resolve_schedule_template,
    validate_schedule_target,
)
from open_climate_service.scheduler.store import StoredSchedule
from open_climate_service.shared.urls import mount_prefix
from open_climate_service.system.templates import wants_json

router = APIRouter()

_TRUE = {"on", "true", "1", "yes"}


def _require_writable() -> None:
    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; schedules cannot be changed")


def _json_request(request: Request) -> bool:
    return "application/json" in request.headers.get("content-type", "")


def _answer_json(request: Request) -> bool:
    """JSON for a JSON body or a bodiless API call; a form post goes back to the page."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return True
    if "form" in content_type:
        return False
    return wants_json(request)


def _status_or_404(dataset_id: str) -> ScheduleStatus:
    status = get_scheduler_service().schedule_for(dataset_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No schedule for dataset '{dataset_id}'")
    return status


def _stored_or_409(dataset_id: str) -> StoredSchedule:
    """The stored schedule, refusing a file-configured dataset, which the page cannot change."""
    stored = store.get_schedule(dataset_id)
    if stored is None:
        if get_scheduler_service().schedule_for(dataset_id) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"The schedule for '{dataset_id}' is configured in climate-service.yaml and is read-only here",
            )
        raise HTTPException(status_code=404, detail=f"No stored schedule for dataset '{dataset_id}'")
    return stored


def _stored_status(dataset_id: str) -> ScheduleStatus | None:
    """The stored entry's own status row, which is the shadowed one when the file also has it."""
    return next(
        (
            item
            for item in get_scheduler_service().status().schedules
            if item.dataset_id == dataset_id and item.source == "store"
        ),
        None,
    )


async def _body(request: Request) -> dict[str, Any]:
    """A JSON body as is, or a form turned into the schedule shape."""
    if _json_request(request):
        payload: Any = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Schedule body must be an object")
        return {str(key): value for key, value in payload.items()}
    form = await request.form()
    field = {key: str(value).strip() for key, value in form.items() if isinstance(value, str)}
    body: dict[str, Any] = {
        "dataset_id": field.get("dataset_id", ""),
        "cron": field.get("cron", ""),
        "publish": field.get("publish", "").lower() in _TRUE,
        "enabled": field.get("enabled", "").lower() in _TRUE,
    }
    if field.get("max_attempts"):
        body["max_attempts"] = field["max_attempts"]
    return body


def _render(
    request: Request,
    tab: str,
    *,
    error: str | None = None,
    draft: dict[str, Any] | None = None,
    editing: str | None = None,
) -> HTMLResponse:
    from open_climate_service.system.templates import render_schedules_page

    return HTMLResponse(
        render_schedules_page(
            get_scheduler_service().status(),
            schedule_choices() if tab == "create" else [],
            mount_prefix(request),
            tab=tab,
            error=error,
            draft=draft,
            editing=editing,
        ),
        status_code=400 if error else 200,
    )


def schedule_choices() -> list[dict[str, Any]]:
    """Datasets a stored schedule can target: ingested, syncable, not configured in the file."""
    from open_climate_service.ingestions.services import list_datasets

    file_datasets = {item.dataset_id for item in get_scheduler_service().status().schedules if item.source == "file"}
    choices: list[dict[str, Any]] = []
    for item in list_datasets().items:
        if item.item_type != "coverage":
            continue
        template = resolve_schedule_template(item.dataset_id)
        try:
            validate_schedule_target(template, item.dataset_id)
        except ValueError:
            continue
        choices.append(
            {
                "id": item.dataset_id,
                "name": item.dataset_name,
                "cadence": item.period_type,
                "coverage": item.extent.temporal,
                "configured": item.dataset_id in file_datasets,
            }
        )
    return choices


def _parse(body: dict[str, Any], request: Request, *, editing: str | None) -> StoredSchedule | HTMLResponse:
    try:
        return StoredSchedule.model_validate(body)
    except ValidationError as exc:
        message = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        if _json_request(request):
            raise HTTPException(status_code=422, detail=message) from exc
        return _render(request, "create", error=message, draft=body, editing=editing)


def _check_target(schedule: StoredSchedule, request: Request, *, editing: str | None) -> HTMLResponse | None:
    """Refuse a dataset the clock could not sync and, on create, one the file already schedules.

    An existing stored schedule stays editable while shadowed: it is kept for the day the file
    entry goes, and its settings must be changeable before then.
    """
    dataset_id = schedule.dataset_id
    try:
        validate_schedule_target(resolve_schedule_template(dataset_id), dataset_id)
    except ValueError as exc:
        if _json_request(request):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _render(request, "create", error=str(exc), draft=schedule.model_dump(mode="json"), editing=editing)
    if editing is not None:
        return None
    status = get_scheduler_service().schedule_for(dataset_id)
    if status is not None and status.source == "file":
        detail = f"'{dataset_id}' is scheduled in climate-service.yaml; a stored schedule for it would be shadowed"
        if _json_request(request):
            raise HTTPException(status_code=409, detail=detail)
        return _render(request, "create", error=detail, draft=schedule.model_dump(mode="json"), editing=editing)
    return None


def _after_change(request: Request, dataset_id: str, *, status_code: int = 200) -> Response:
    """Reload the clock, then answer with the schedule's new status or go back to the page."""
    service = get_scheduler_service()
    service.reload()
    if _answer_json(request):
        status = _stored_status(dataset_id) or service.schedule_for(dataset_id)
        payload = status.model_dump(mode="json") if status is not None else {"dataset_id": dataset_id}
        reload_error = service.status().reload_error
        if reload_error:
            payload["reload_error"] = reload_error
        return JSONResponse(status_code=status_code, content=payload)
    return RedirectResponse(f"{mount_prefix(request)}/schedules", status_code=303)


# --- routes ------------------------------------------------------------------------------------


@router.get("", response_model=ScheduleListResponse, response_class=Response)
def list_schedules(request: Request) -> Response:
    """The merged schedules with their runtime status, as JSON, or the Sync schedules page."""
    status = get_scheduler_service().status()
    if wants_json(request):
        return JSONResponse(status.model_dump(mode="json"))
    return _render(request, "list")


@router.get("/new", response_class=HTMLResponse, include_in_schema=False)
def new_schedule(request: Request) -> HTMLResponse:
    """The schedule form, optionally prefilled with a dataset."""
    dataset_id = request.query_params.get("dataset", "")
    return _render(request, "create", draft={"dataset_id": dataset_id} if dataset_id else None)


@router.post("", response_class=Response)
async def create_schedule(request: Request) -> Response:
    """Save a new schedule and start running it."""
    _require_writable()
    body = await _body(request)
    parsed = _parse(body, request, editing=None)
    if isinstance(parsed, HTMLResponse):
        return parsed
    refused = _check_target(parsed, request, editing=None)
    if refused is not None:
        return refused
    try:
        store.save_schedule(parsed, create=True)
    except ValueError as exc:
        if _json_request(request):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _render(request, "create", error=str(exc), draft=body)
    return _after_change(request, parsed.dataset_id, status_code=201)


@router.get("/{dataset_id}", response_model=ScheduleStatus)
def get_schedule(dataset_id: str) -> ScheduleStatus:
    """One dataset's effective schedule and runtime status."""
    return _status_or_404(dataset_id)


@router.get("/{dataset_id}/edit", response_class=HTMLResponse, include_in_schema=False)
def edit_schedule(request: Request, dataset_id: str) -> HTMLResponse:
    """The schedule form, prefilled with a stored schedule."""
    stored = _stored_or_409(dataset_id)
    return _render(request, "create", draft=stored.model_dump(mode="json"), editing=dataset_id)


async def _update(request: Request, dataset_id: str) -> Response:
    _require_writable()
    existing = _stored_or_409(dataset_id)
    body = await _body(request)
    body["dataset_id"] = dataset_id
    body.setdefault("enabled", existing.enabled)
    parsed = _parse(body, request, editing=dataset_id)
    if isinstance(parsed, HTMLResponse):
        return parsed
    refused = _check_target(parsed, request, editing=dataset_id)
    if refused is not None:
        return refused
    store.save_schedule(parsed, create=False)
    return _after_change(request, dataset_id)


@router.put("/{dataset_id}", response_class=Response)
async def update_schedule(request: Request, dataset_id: str) -> Response:
    """Replace a stored schedule's cron, publish flag, attempts or pause switch."""
    return await _update(request, dataset_id)


@router.post("/{dataset_id}", response_class=Response, include_in_schema=False)
async def save_schedule_form(request: Request, dataset_id: str) -> Response:
    """The page's edit form; the same as PUT."""
    return await _update(request, dataset_id)


@router.post("/{dataset_id}/pause", response_class=Response)
def pause_schedule(request: Request, dataset_id: str) -> Response:
    """Stop a stored schedule from firing without removing it."""
    _require_writable()
    _stored_or_409(dataset_id)
    store.set_enabled(dataset_id, False)
    return _after_change(request, dataset_id)


@router.post("/{dataset_id}/resume", response_class=Response)
def resume_schedule(request: Request, dataset_id: str) -> Response:
    """Let a paused stored schedule fire again."""
    _require_writable()
    _stored_or_409(dataset_id)
    store.set_enabled(dataset_id, True)
    return _after_change(request, dataset_id)


@router.delete("/{dataset_id}", status_code=204)
def delete_schedule(dataset_id: str) -> Response:
    """Remove a stored schedule; a file-configured one is refused."""
    _require_writable()
    _stored_or_409(dataset_id)
    store.delete_schedule(dataset_id)
    get_scheduler_service().reload()
    return Response(status_code=204)


@router.post("/{dataset_id}/delete", response_class=Response, include_in_schema=False)
async def delete_schedule_form(request: Request, dataset_id: str) -> Response:
    """The page's delete action; needs the confirmation box ticked."""
    _require_writable()
    _stored_or_409(dataset_id)
    form = await request.form()
    if str(form.get("confirm", "")).strip().lower() not in _TRUE:
        raise HTTPException(status_code=400, detail="Tick the confirmation to delete the schedule")
    store.delete_schedule(dataset_id)
    get_scheduler_service().reload()
    return RedirectResponse(f"{mount_prefix(request)}/schedules", status_code=303)
