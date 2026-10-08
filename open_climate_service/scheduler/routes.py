"""Operator routes for schedules (CLIM-1242).

``GET /schedules`` lists everything on the instance's clock. Sync schedules, the only kind so far,
live under ``/schedules/sync/{dataset_id}``, keyed by the dataset they sync because there is one
per dataset. They are set up and edited from the dataset's page, and paused, resumed or deleted
from either page.

Every schedule is stored in one shared JSON file. A change is followed by a scheduler
reload, so a saved schedule takes effect without a restart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler import store
from open_climate_service.scheduler.presets import cron_from_form
from open_climate_service.scheduler.schemas import ScheduleListResponse, ScheduleStatus
from open_climate_service.scheduler.service import (
    get_scheduler_service,
    resolve_schedule_template,
    validate_schedule_target,
)
from open_climate_service.scheduler.store import ScheduleStoreUnreadable, StoredSchedule
from open_climate_service.shared.urls import mount_prefix
from open_climate_service.system.templates import wants_json

router = APIRouter()

_TRUE = {"on", "true", "1", "yes"}
_EDITABLE = {"cron", "publish", "max_attempts", "enabled"}
T = TypeVar("T")


def _require_writable() -> None:
    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; schedules cannot be changed")


def _json_request(request: Request) -> bool:
    return "application/json" in request.headers.get("content-type", "")


def _form_request(request: Request) -> bool:
    return "form" in request.headers.get("content-type", "")


def _answer_json(request: Request) -> bool:
    """JSON for a JSON body or a bodiless API call; a form post goes back to a page."""
    if _json_request(request):
        return True
    if _form_request(request):
        return False
    return wants_json(request)


def _reading(action: Callable[[], T]) -> T:
    """Run a store read or write; an unreadable store is a 503 with its reason, not a 500."""
    try:
        return action()
    except ScheduleStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=f"Stored schedules cannot be read: {exc}") from exc


def _status_or_404(dataset_id: str) -> ScheduleStatus:
    status = get_scheduler_service().schedule_for(dataset_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No sync schedule for dataset '{dataset_id}'")
    return status


def _stored_or_404(dataset_id: str) -> StoredSchedule:
    """Return the dataset's stored schedule, or report that it does not exist."""
    stored = _reading(lambda: store.get_schedule(dataset_id))
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No stored sync schedule for dataset '{dataset_id}'")
    return stored


async def _body(request: Request) -> tuple[dict[str, Any], str | None]:
    """A JSON body as is, or a form turned into the schedule shape, plus where a form returns to.

    An unticked `enabled` checkbox is absent from a form post and means False. Publication
    is not an editable field in the form: on edit an omitted value keeps the saved setting.
    """
    if _json_request(request):
        payload: Any = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Schedule body must be an object")
        return {str(key): value for key, value in payload.items()}, None
    form = await request.form()
    field = {key: str(value).strip() for key, value in form.items() if isinstance(value, str)}
    body: dict[str, Any] = {
        "dataset_id": field.get("dataset_id", ""),
        "cron": field.get("cron", ""),
        "enabled": field.get("enabled", "").lower() in _TRUE,
    }
    if "frequency" in field:
        for key in ("frequency", "check_time", "weekday", "month_day", "cron"):
            body[f"_ui_{key}"] = field.get(key, "")
        try:
            body["cron"] = cron_from_form(field)
        except ValueError as exc:
            body["_ui_error"] = str(exc)
    if "publish" in field:
        body["publish"] = field["publish"].lower() in _TRUE
    if field.get("max_attempts"):
        body["max_attempts"] = field["max_attempts"]
    return body, field.get("return_to") or None


def _back(request: Request, dataset_id: str, return_to: str | None) -> RedirectResponse:
    """Back to the page the form came from: the dataset page, or the Schedules list."""
    mount = mount_prefix(request)
    if return_to == "dataset":
        return RedirectResponse(f"{mount}/datasets/{dataset_id}#schedule", status_code=303)
    return RedirectResponse(f"{mount}/schedules", status_code=303)


def _refuse(request: Request, dataset_id: str, status_code: int, message: str, draft: dict[str, Any]) -> Response:
    """Refuse a save: the reason as JSON, or the dataset page with the reason and the draft kept."""
    if _answer_json(request):
        raise HTTPException(status_code=status_code, detail=message)
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.system.templates import render_dataset_page, render_schedules_page

    try:
        record = ingestion_services.get_dataset_or_404(dataset_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        # A malformed or stale form has no dataset page to return to. Keep the
        # validation status and reason rather than replacing them with a 404.
        status = get_scheduler_service().status()
        return HTMLResponse(
            render_schedules_page(status, mount_prefix(request), change_warning=message), status_code=status_code
        )
    page = render_dataset_page(record, mount_prefix(request), schedule_error=message, schedule_draft=draft)
    return HTMLResponse(page, status_code=status_code)


def _parse(request: Request, dataset_id: str, body: dict[str, Any]) -> StoredSchedule | Response:
    if "_ui_error" in body:
        return _refuse(request, dataset_id, 422, str(body["_ui_error"]), body)
    try:
        values = body
        if _form_request(request):
            values = {key: value for key, value in body.items() if not key.startswith("_ui_")}
        return StoredSchedule.model_validate(values)
    except ValidationError as exc:
        message = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        return _refuse(request, dataset_id, 422, message, body)


def _check_target(request: Request, schedule: StoredSchedule) -> Response | None:
    """Refuse a dataset the clock could not sync."""
    dataset_id = schedule.dataset_id
    draft = schedule.model_dump(mode="json")
    try:
        validate_schedule_target(resolve_schedule_template(dataset_id), dataset_id)
    except ValueError as exc:
        return _refuse(request, dataset_id, 422, str(exc), draft)
    return None


def _after_change(request: Request, dataset_id: str, return_to: str | None, *, status_code: int = 200) -> Response:
    """Report a persisted change separately from whether the clock accepted it."""
    service = get_scheduler_service()
    service.reload()
    scheduler_status = service.status()
    if scheduler_status.reload_error:
        message = (
            "The schedule was saved, but the scheduler could not apply it; the previous schedule remains in force. "
            f"Reason: {scheduler_status.reload_error}"
        )
        if _answer_json(request):
            return JSONResponse(
                status_code=409,
                content={"detail": message, "stored": True, "applied": False, "dataset_id": dataset_id},
            )
        if return_to == "dataset":
            saved = _reading(lambda: store.get_schedule(dataset_id))
            return _refuse(request, dataset_id, 409, message, saved.model_dump(mode="json") if saved else {})
        from open_climate_service.system.templates import render_schedules_page

        return HTMLResponse(
            render_schedules_page(scheduler_status, mount_prefix(request), change_warning=message), status_code=409
        )
    if _answer_json(request):
        status = service.schedule_for(dataset_id)
        payload = status.model_dump(mode="json") if status is not None else {"dataset_id": dataset_id}
        return JSONResponse(status_code=status_code, content=payload)
    return _back(request, dataset_id, return_to)


async def _form_return_to(request: Request) -> str | None:
    """Where a bodiless form action (pause, resume, delete) returns to."""
    if not _form_request(request):
        return None
    form = await request.form()
    value = form.get("return_to")
    return value if isinstance(value, str) and value else None


# --- routes ------------------------------------------------------------------------------------


@router.get("", response_model=ScheduleListResponse, response_class=Response)
def list_schedules(request: Request) -> Response:
    """Everything on the clock with its runtime status, as JSON, or the Schedules page."""
    status = get_scheduler_service().status()
    if wants_json(request):
        return JSONResponse(status.model_dump(mode="json"))
    from open_climate_service.system.templates import render_schedules_page

    return HTMLResponse(render_schedules_page(status, mount_prefix(request)))


@router.post("/sync", response_class=Response)
async def create_sync_schedule(request: Request) -> Response:
    """Save a new sync schedule for a dataset and start running it."""
    _require_writable()
    body, return_to = await _body(request)
    if _form_request(request):
        body.setdefault("publish", False)
    dataset_id = str(body.get("dataset_id", ""))
    parsed = _parse(request, dataset_id, body)
    if not isinstance(parsed, StoredSchedule):
        return parsed
    refused = _check_target(request, parsed)
    if refused is not None:
        return refused
    try:
        _reading(lambda: store.save_schedule(parsed, create=True))
    except ValueError as exc:
        return _refuse(request, dataset_id, 409, str(exc), body)
    return _after_change(request, parsed.dataset_id, return_to, status_code=201)


@router.get("/sync/{dataset_id}", response_model=ScheduleStatus)
def get_sync_schedule(dataset_id: str) -> ScheduleStatus:
    """One dataset's effective sync schedule and runtime status."""
    return _status_or_404(dataset_id)


async def _update(request: Request, dataset_id: str) -> Response:
    _require_writable()
    existing = _stored_or_404(dataset_id)
    body, return_to = await _body(request)
    # JSON and form updates both preserve settings that were not offered by the caller.
    body = {**existing.model_dump(include=_EDITABLE), **body}
    body["dataset_id"] = dataset_id
    parsed = _parse(request, dataset_id, body)
    if not isinstance(parsed, StoredSchedule):
        return parsed
    refused = _check_target(request, parsed)
    if refused is not None:
        return refused
    try:
        _reading(lambda: store.save_schedule(parsed, create=False))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _after_change(request, dataset_id, return_to)


@router.put("/sync/{dataset_id}", response_class=Response)
async def update_sync_schedule(request: Request, dataset_id: str) -> Response:
    """Change a stored sync schedule; settings left out of the body are kept."""
    return await _update(request, dataset_id)


@router.post("/sync/{dataset_id}", response_class=Response, include_in_schema=False)
async def save_sync_schedule_form(request: Request, dataset_id: str) -> Response:
    """The dataset page's edit form; the same as PUT."""
    return await _update(request, dataset_id)


@router.post("/sync/{dataset_id}/pause", response_class=Response)
async def pause_sync_schedule(request: Request, dataset_id: str) -> Response:
    """Stop a stored sync schedule from firing without removing it."""
    _require_writable()
    try:
        _reading(lambda: store.set_enabled(dataset_id, False))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _after_change(request, dataset_id, await _form_return_to(request))


@router.post("/sync/{dataset_id}/resume", response_class=Response)
async def resume_sync_schedule(request: Request, dataset_id: str) -> Response:
    """Let a paused stored sync schedule fire again."""
    _require_writable()
    try:
        _reading(lambda: store.set_enabled(dataset_id, True))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _after_change(request, dataset_id, await _form_return_to(request))


@router.delete("/sync/{dataset_id}", status_code=204)
def delete_sync_schedule(dataset_id: str) -> Response:
    """Remove a stored sync schedule."""
    _require_writable()
    if not _reading(lambda: store.delete_schedule(dataset_id)):
        raise HTTPException(status_code=404, detail=f"No stored sync schedule for dataset '{dataset_id}'")
    service = get_scheduler_service()
    service.reload()
    if reason := service.status().reload_error:
        message = "The schedule was deleted from storage, but the scheduler could not apply the change: " + reason
        return JSONResponse(
            status_code=409,
            content={
                "detail": message,
                "stored": False,
                "applied": False,
                "dataset_id": dataset_id,
            },
        )
    return Response(status_code=204)


@router.get("/sync/{dataset_id}/delete", response_class=HTMLResponse, include_in_schema=False)
def confirm_sync_schedule_delete(request: Request, dataset_id: str) -> HTMLResponse:
    """Confirmation page for browsers without JavaScript, including orphaned schedules."""
    _require_writable()
    _stored_or_404(dataset_id)
    from open_climate_service.system.templates import render_schedule_delete_page

    return HTMLResponse(render_schedule_delete_page(dataset_id, mount_prefix(request)))


@router.post("/sync/{dataset_id}/delete", response_class=Response, include_in_schema=False)
async def delete_sync_schedule_form(request: Request, dataset_id: str) -> Response:
    """The pages' delete action; needs the confirmation the dialog or the panel sends."""
    _require_writable()
    form = await request.form()
    if str(form.get("confirm", "")).strip().lower() not in _TRUE:
        raise HTTPException(status_code=400, detail="Confirm the deletion to delete the schedule")
    if not _reading(lambda: store.delete_schedule(dataset_id)):
        raise HTTPException(status_code=404, detail=f"No stored sync schedule for dataset '{dataset_id}'")
    service = get_scheduler_service()
    service.reload()
    status = service.status()
    if status.reload_error:
        from open_climate_service.system.templates import render_schedules_page

        message = (
            "The schedule was deleted from storage, but the scheduler could not apply the change; "
            f"the previous schedule may still run. Reason: {status.reload_error}"
        )
        return HTMLResponse(
            render_schedules_page(status, mount_prefix(request), change_warning=message), status_code=409
        )
    return_to = form.get("return_to")
    return _back(request, dataset_id, return_to if isinstance(return_to, str) else None)
