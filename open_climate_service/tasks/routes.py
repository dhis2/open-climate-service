"""The tasks API: list, create, change, pause, run and remove every kind of task (CLIM-1378).

One resource for what used to be three: sync schedules, workflow triggers in
``climate-service.yaml`` and their delivery blocks. A write is validated against the whole set
of tasks under the store lock, then applied at once: the clock reloads, and automation picks up
the change through the clock's reload listener. No restart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from open_climate_service import config as api_config
from open_climate_service.runs.service import RunView
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult
from open_climate_service.tasks import store
from open_climate_service.tasks.models import Task
from open_climate_service.tasks.store import TaskStoreUnreadable

router = APIRouter()


class TaskStatus(Task):
    """A task with how it starts, in words, when it runs next, and how its latest runs went.

    The last run comes from the stored run records, so it survives a restart; ``next_run`` is
    known only in the process that runs the clock.
    """

    starts: str
    next_run: datetime | None = None
    last_run: RunView | None = None
    consecutive_failures: int = 0


class TaskList(BaseModel):
    """Every task, by id."""

    tasks: list[TaskStatus]


def _require_writable() -> None:
    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; tasks cannot be changed")


def _validate_structure(tasks: list[Task]) -> None:
    """Rules across tasks: one sync schedule per dataset, a deliver task after an existing workflow task."""
    from open_climate_service.automation.config import compile_tasks
    from open_climate_service.scheduler.store import _one_per_dataset

    _one_per_dataset(tasks)
    compile_tasks(tasks)


def check_target(task: Task) -> None:
    """Refuse a sync or refresh task whose dataset or collection cannot be run. Raises ValueError."""
    from open_climate_service.scheduler.service import resolve_schedule_template, validate_schedule_target

    if task.kind == "sync":
        validate_schedule_target(resolve_schedule_template(task.target), task.target)
    elif task.kind == "refresh":
        from open_climate_service.features.services import refreshable_feature_template_or_error

        try:
            refreshable_feature_template_or_error(task.target)
        except HTTPException as exc:
            raise ValueError(f"Refresh task {task.id!r}: {exc.detail}") from None


def _validator(written: Task) -> Any:
    """Check a write: the structure of all tasks, and everything the written task depends on.

    Only the written task's own target is resolved, so a task whose dataset has since gone does
    not block editing an unrelated one. Raises ValueError, naming the task.
    """

    def check(tasks: list[Task]) -> None:
        from open_climate_service.automation.config import AutomationConfig, compile_tasks
        from open_climate_service.automation.service import validate_automation

        _validate_structure(tasks)
        check_target(written)
        involved = {written.id}
        if written.kind == "deliver" and written.after is not None and written.after.task is not None:
            involved.add(written.after.task)
        triggers = [trigger for trigger in compile_tasks(tasks).workflow_triggers if trigger.id in involved]
        validate_automation(AutomationConfig(workflow_triggers=triggers))

    return check


def _apply() -> None:
    """Make the clock and automation run what the store now holds."""
    from open_climate_service.scheduler.service import get_scheduler_service

    get_scheduler_service().reload()


def _status(task: Task) -> TaskStatus:
    from open_climate_service.runs import service as runs
    from open_climate_service.scheduler.service import get_scheduler_service

    scheduler = get_scheduler_service()
    next_run: Any = None
    if task.kind == "sync":
        sync = scheduler.schedule_for(task.target)
        next_run = sync.next_check if sync is not None else None
    else:
        next_run, _ = scheduler.task_status(task.id)
    latest = runs.list_runs(task_id=task.id, limit=1)
    return TaskStatus(
        **task.model_dump(),
        starts=task.when,
        next_run=next_run,
        last_run=runs.view(latest[0]) if latest else None,
        consecutive_failures=runs.consecutive_failures(task.id),
    )


def _get_or_404(task_id: str) -> Task:
    try:
        task = store.get_task(task_id)
    except TaskStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=404, detail=f"No task with id '{task_id}'")
    return task


def _write(action: Any) -> Any:
    try:
        return action()
    except TaskStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _listing() -> TaskList:
    try:
        tasks = store.list_tasks()
    except TaskStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TaskList(tasks=[_status(task) for task in tasks])


def _page(request: Request, *, error: str | None = None, draft: dict[str, Any] | None = None) -> HTMLResponse:
    from open_climate_service.scheduler.service import get_scheduler_service
    from open_climate_service.shared.urls import mount_prefix
    from open_climate_service.system.templates import render_tasks_page

    html = render_tasks_page(
        _listing().tasks, get_scheduler_service().status(), mount_prefix(request), error=error, draft=draft
    )
    return HTMLResponse(html, status_code=400 if error else 200)


def _back(request: Request) -> RedirectResponse:
    from open_climate_service.shared.urls import mount_prefix

    return RedirectResponse(f"{mount_prefix(request)}/tasks", status_code=303)


@router.get("", response_model=TaskList)
def list_tasks(request: Request) -> Any:
    """Every task, with how it starts and its latest run, as JSON, or the Automation page."""
    from open_climate_service.system.templates import wants_json

    if not wants_json(request):
        return _page(request)
    return _listing()


@router.post("/form", include_in_schema=False)
async def create_task_form(request: Request) -> Response:
    """The Automation page's Add a task form."""
    import json

    _require_writable()
    form = {key: str(value).strip() for key, value in (await request.form()).items() if isinstance(value, str)}
    body: dict[str, Any] = {"id": form.get("id", ""), "kind": form.get("kind", ""), "target": form.get("target", "")}
    if form.get("cron"):
        body["cron"] = form["cron"]
    if form.get("after_type") and form.get("after_id"):
        body["after"] = {form["after_type"]: form["after_id"]}
    if form.get("arguments"):
        try:
            body["arguments"] = json.loads(form["arguments"])
        except json.JSONDecodeError as exc:
            return _page(request, error=f"Workflow arguments are not valid JSON: {exc}", draft=form)
    if body["kind"] == "deliver":
        body["dry_run"] = form.get("dry_run") == "true"
    try:
        task = Task.model_validate(body)
        store.save_task(task, create=True, check=_validator(task))
    except ValidationError as exc:
        return _page(request, error="; ".join(str(error["msg"]) for error in exc.errors()), draft=form)
    except ValueError as exc:
        return _page(request, error=str(exc), draft=form)
    _apply()
    return _back(request)


@router.post("/{task_id}/delete", include_in_schema=False)
def delete_task_form(request: Request, task_id: str) -> Response:
    """The Automation page's Delete button."""
    _require_writable()
    _get_or_404(task_id)
    try:
        store.delete_task(task_id, check=_validate_structure)
    except ValueError as exc:
        return _page(request, error=str(exc))
    _apply()
    return _back(request)


@router.post("", response_model=TaskStatus, status_code=201)
def create_task(body: dict[str, Any]) -> TaskStatus:
    """Save a new task and start running it."""
    _require_writable()
    try:
        task = Task.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    saved = _write(lambda: store.save_task(task, create=True, check=_validator(task)))
    _apply()
    return _status(saved)


@router.get("/{task_id}", response_model=TaskStatus)
def get_task(task_id: str) -> TaskStatus:
    """One task, with how it starts and its latest run."""
    return _status(_get_or_404(task_id))


@router.put("/{task_id}", response_model=TaskStatus)
def update_task(task_id: str, body: dict[str, Any]) -> TaskStatus:
    """Replace a task. Fields left out keep their stored value; the id cannot change."""
    _require_writable()
    existing = _get_or_404(task_id)
    merged = {**existing.model_dump(), **body, "id": task_id}
    try:
        task = Task.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    saved = _write(lambda: store.save_task(task, create=False, check=_validator(task)))
    _apply()
    return _status(saved)


@router.post("/{task_id}/pause", response_model=TaskStatus)
def pause_task(task_id: str, request: Request) -> Any:
    """Stop a task from running without removing it."""
    _require_writable()
    _get_or_404(task_id)
    saved = _write(lambda: store.set_enabled(task_id, False))
    _apply()
    return _back(request) if request.query_params.get("next") else _status(saved)


@router.post("/{task_id}/resume", response_model=TaskStatus)
def resume_task(task_id: str, request: Request) -> Any:
    """Let a paused task run again."""
    _require_writable()
    _get_or_404(task_id)
    saved = _write(lambda: store.set_enabled(task_id, True))
    _apply()
    return _back(request) if request.query_params.get("next") else _status(saved)


@router.post("/{task_id}/run", response_model=CheckResult, status_code=202)
def run_task(task_id: str, request: Request) -> Any:
    """Run a task now, by hand. A deliver task runs after its workflow task and cannot be run alone."""
    _require_writable()
    from open_climate_service.scheduler.service import get_scheduler_service

    task = _get_or_404(task_id)
    if task.kind == "deliver":
        raise HTTPException(status_code=409, detail="A deliver task runs after its workflow task; run that instead")
    if not task.enabled:
        raise HTTPException(status_code=409, detail=f"Task '{task_id}' is paused")
    result = get_scheduler_service().run_task_now(task, cause=f"manual:{uuid4()}")
    if result.outcome == CheckOutcome.ERROR:
        if request.query_params.get("next"):
            return _page(request, error=result.message)
        raise HTTPException(status_code=409, detail=result.message)
    return _back(request) if request.query_params.get("next") else result


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str) -> Response:
    """Remove a task. A workflow task that a deliver task follows cannot be removed first."""
    _require_writable()
    _get_or_404(task_id)
    _write(lambda: store.delete_task(task_id, check=_validate_structure))
    _apply()
    return Response(status_code=204)
