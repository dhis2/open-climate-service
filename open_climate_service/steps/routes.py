"""The steps API: list, create, change, pause, run and remove every kind of step (CLIM-1378).

One resource for what used to be three: sync schedules, workflow triggers in
``climate-service.yaml`` and their delivery blocks. A write is validated against the whole set
of steps under the store lock, then applied at once: the clock reloads, and automation picks up
the change through the clock's reload listener. No restart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult
from open_climate_service.steps import store
from open_climate_service.steps.models import Step
from open_climate_service.steps.store import StepStoreUnreadable

router = APIRouter()


class StepStatus(Step):
    """A step with how it starts, in words, and its latest run on this process's clock."""

    starts: str
    next_run: datetime | None = None
    last_run: datetime | None = None
    last_outcome: CheckOutcome | None = None
    last_message: str | None = None
    last_job_id: str | None = None


class StepList(BaseModel):
    """Every step, by id."""

    steps: list[StepStatus]


def _require_writable() -> None:
    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; steps cannot be changed")


def _validate_structure(steps: list[Step]) -> None:
    """Rules across steps: one sync schedule per dataset, a deliver step after an existing workflow step."""
    from open_climate_service.automation.config import compile_steps
    from open_climate_service.scheduler.store import _one_per_dataset

    _one_per_dataset(steps)
    compile_steps(steps)


def _validator(written: Step) -> Any:
    """Check a write: the structure of all steps, and everything the written step depends on.

    Only the written step's own target is resolved, so a step whose dataset has since gone does
    not block editing an unrelated one. Raises ValueError, naming the step.
    """

    def check(steps: list[Step]) -> None:
        from open_climate_service.automation.config import AutomationConfig, compile_steps
        from open_climate_service.automation.service import validate_automation
        from open_climate_service.scheduler.service import resolve_schedule_template, validate_schedule_target

        _validate_structure(steps)
        if written.kind == "sync":
            validate_schedule_target(resolve_schedule_template(written.target), written.target)
        elif written.kind == "refresh":
            from open_climate_service.features.services import refreshable_feature_template_or_error

            try:
                refreshable_feature_template_or_error(written.target)
            except HTTPException as exc:
                raise ValueError(f"Refresh step {written.id!r}: {exc.detail}") from None
        involved = {written.id}
        if written.kind == "deliver" and written.after is not None and written.after.step is not None:
            involved.add(written.after.step)
        triggers = [trigger for trigger in compile_steps(steps).workflow_triggers if trigger.id in involved]
        validate_automation(AutomationConfig(workflow_triggers=triggers))

    return check


def _apply() -> None:
    """Make the clock and automation run what the store now holds."""
    from open_climate_service.scheduler.service import get_scheduler_service

    get_scheduler_service().reload()


def _status(step: Step) -> StepStatus:
    from open_climate_service.scheduler.service import get_scheduler_service

    scheduler = get_scheduler_service()
    next_run: Any = None
    result: CheckResult | None = None
    if step.kind == "sync":
        sync = scheduler.schedule_for(step.target)
        if sync is not None:
            next_run, last = sync.next_check, sync
            return StepStatus(
                **step.model_dump(),
                starts=step.when,
                next_run=next_run,
                last_run=last.last_check,
                last_outcome=last.last_outcome,
                last_message=last.last_message,
                last_job_id=last.last_job_id,
            )
    else:
        next_run, result = scheduler.step_status(step.id)
    return StepStatus(
        **step.model_dump(),
        starts=step.when,
        next_run=next_run,
        last_run=result.checked_at if result else None,
        last_outcome=result.outcome if result else None,
        last_message=result.message if result else None,
        last_job_id=result.job_id if result else None,
    )


def _get_or_404(step_id: str) -> Step:
    try:
        step = store.get_step(step_id)
    except StepStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if step is None:
        raise HTTPException(status_code=404, detail=f"No step with id '{step_id}'")
    return step


def _write(action: Any) -> Any:
    try:
        return action()
    except StepStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=StepList)
def list_steps() -> StepList:
    """Every step, with how it starts and its latest run."""
    try:
        steps = store.list_steps()
    except StepStoreUnreadable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StepList(steps=[_status(step) for step in steps])


@router.post("", response_model=StepStatus, status_code=201)
def create_step(body: dict[str, Any]) -> StepStatus:
    """Save a new step and start running it."""
    _require_writable()
    try:
        step = Step.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    saved = _write(lambda: store.save_step(step, create=True, check=_validator(step)))
    _apply()
    return _status(saved)


@router.get("/{step_id}", response_model=StepStatus)
def get_step(step_id: str) -> StepStatus:
    """One step, with how it starts and its latest run."""
    return _status(_get_or_404(step_id))


@router.put("/{step_id}", response_model=StepStatus)
def update_step(step_id: str, body: dict[str, Any]) -> StepStatus:
    """Replace a step. Fields left out keep their stored value; the id cannot change."""
    _require_writable()
    existing = _get_or_404(step_id)
    merged = {**existing.model_dump(), **body, "id": step_id}
    try:
        step = Step.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    saved = _write(lambda: store.save_step(step, create=False, check=_validator(step)))
    _apply()
    return _status(saved)


@router.post("/{step_id}/pause", response_model=StepStatus)
def pause_step(step_id: str) -> StepStatus:
    """Stop a step from running without removing it."""
    _require_writable()
    _get_or_404(step_id)
    saved = _write(lambda: store.set_enabled(step_id, False))
    _apply()
    return _status(saved)


@router.post("/{step_id}/resume", response_model=StepStatus)
def resume_step(step_id: str) -> StepStatus:
    """Let a paused step run again."""
    _require_writable()
    _get_or_404(step_id)
    saved = _write(lambda: store.set_enabled(step_id, True))
    _apply()
    return _status(saved)


@router.post("/{step_id}/run", response_model=CheckResult, status_code=202)
def run_step(step_id: str) -> CheckResult:
    """Run a step now, by hand. A deliver step runs after its workflow step and cannot be run alone."""
    _require_writable()
    from open_climate_service.scheduler.service import get_scheduler_service

    step = _get_or_404(step_id)
    if step.kind == "deliver":
        raise HTTPException(status_code=409, detail="A deliver step runs after its workflow step; run that instead")
    if not step.enabled:
        raise HTTPException(status_code=409, detail=f"Step '{step_id}' is paused")
    result = get_scheduler_service().run_step_now(step, cause=f"manual:{uuid4()}")
    if result.outcome == CheckOutcome.ERROR:
        raise HTTPException(status_code=409, detail=result.message)
    return result


@router.delete("/{step_id}", status_code=204)
def delete_step(step_id: str) -> Response:
    """Remove a step. A workflow step that a deliver step follows cannot be removed first."""
    _require_writable()
    _get_or_404(step_id)
    _write(lambda: store.delete_step(step_id, check=_validate_structure))
    _apply()
    return Response(status_code=204)
