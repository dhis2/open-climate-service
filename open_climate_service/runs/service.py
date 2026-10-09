"""Record each run of a task, and answer what a run did, from the job it started (CLIM-1378).

A run is written when a task is started (by its cron, by an event, or by hand) and names the job
it submitted. Its status is read from that job when asked, so a run never disagrees with its job
and nothing has to be updated when the job finishes. The run that caused it (a sync whose update
started a workflow, a workflow whose job a delivery sends) is linked as its parent, so one sync
can be followed through every run it set off.

Runs live in the operational database, so the last outcome of each task survives a restart
(CLIM-919), and repeated failures are a count rather than something to notice.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from open_climate_service.shared.time import utc_now
from open_climate_service.state import db

logger = logging.getLogger(__name__)

RunCause = Literal["cron", "event", "manual"]
JobKind = Literal["native", "openeo"]
RunStatus = Literal["queued", "running", "retrying", "succeeded", "failed", "cancelled", "skipped", "refused"]

_FAILED = {"failed", "refused"}


class Run(BaseModel):
    """One run of a task, as recorded when it started."""

    id: str
    task_id: str
    kind: str
    cause: RunCause
    cause_ref: str | None = Field(default=None, description="The event or cron fire that started it.")
    parent_run_id: str | None = Field(default=None, description="The run whose result started this one.")
    job_kind: JobKind | None = None
    job_id: str | None = None
    outcome: str = Field(description="What starting it did: submitted, skipped (and why), or error.")
    message: str = ""
    started_at: datetime


class RunView(Run):
    """A run with its live status, read from its job."""

    status: RunStatus
    job_href: str | None = None
    detail: str | None = Field(default=None, description="The job's latest message or error.")


def record_run(
    *,
    task_id: str,
    kind: str,
    cause: RunCause,
    outcome: str,
    message: str = "",
    job_kind: JobKind | None = None,
    job_id: str | None = None,
    cause_ref: str | None = None,
    parent_run_id: str | None = None,
) -> Run | None:
    """Write one run. Never raises: losing a record must not stop the work it describes."""
    run = Run(
        id=str(uuid4()),
        task_id=task_id,
        kind=kind,
        cause=cause,
        cause_ref=cause_ref,
        parent_run_id=parent_run_id,
        job_kind=job_kind,
        job_id=job_id,
        outcome=str(outcome),
        message=message,
        started_at=utc_now(),
    )
    try:
        with db.write() as connection:
            connection.execute(
                "INSERT INTO runs (id, task_id, kind, cause, cause_ref, parent_run_id, job_kind, job_id, outcome, "
                "message, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.task_id,
                    run.kind,
                    run.cause,
                    run.cause_ref,
                    run.parent_run_id,
                    run.job_kind,
                    run.job_id,
                    run.outcome,
                    run.message,
                    run.started_at.isoformat(),
                ),
            )
    except Exception:
        logger.exception("Could not record a run of task %s", task_id)
        return None
    return run


def _row(row: object) -> Run:
    return Run.model_validate(dict(row))  # type: ignore[call-overload]


def list_runs(*, task_id: str | None = None, parent_run_id: str | None = None, limit: int = 50) -> list[Run]:
    """Newest first."""
    clauses, params = [], []
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(task_id)
    if parent_run_id is not None:
        clauses.append("parent_run_id = ?")
        params.append(parent_run_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.read() as connection:
        rows = connection.execute(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ?",  # noqa: S608  # fixed clauses
            (*params, limit),
        ).fetchall()
    return [_row(row) for row in rows]


def get_run(run_id: str) -> Run | None:
    """One run, or None."""
    with db.read() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row(row) if row is not None else None


def run_for_job(job_id: str | None) -> Run | None:
    """The run that started ``job_id``, to link the runs its result causes."""
    if not job_id:
        return None
    try:
        with db.read() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE job_id = ? ORDER BY started_at DESC LIMIT 1", (job_id,)
            ).fetchone()
    except Exception:
        logger.exception("Could not look up the run of job %s", job_id)
        return None
    return _row(row) if row is not None else None


def _native_status(job_id: str) -> tuple[RunStatus, str | None]:
    from open_climate_service.jobs import store as job_store

    record = job_store.get_job_record(job_id)
    if record is None:
        return "failed", "The job record is gone"
    status = {
        "accepted": "queued",
        "running": "running",
        "retrying": "retrying",
        "successful": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }[str(record.status)]
    detail = record.error.message if record.error is not None else record.progress.message
    return status, detail  # type: ignore[return-value]


def _openeo_status(job_id: str) -> tuple[RunStatus, str | None]:
    from open_climate_service.openeo.jobs import store_get_job

    record = store_get_job(job_id)
    if record is None:
        return "failed", "The job record is gone"
    status = {
        "created": "queued",
        "queued": "queued",
        "running": "running",
        "finished": "succeeded",
        "error": "failed",
        "canceled": "cancelled",
    }[str(record.status.value if hasattr(record.status, "value") else record.status)]
    return status, record.error_message  # type: ignore[return-value]


def view(run: Run) -> RunView:
    """The run with its job's current status."""
    if run.job_id is None or run.job_kind is None:
        status: RunStatus = "refused" if run.outcome == "error" else "skipped"
        return RunView(**run.model_dump(), status=status, detail=run.message or None)
    try:
        if run.job_kind == "openeo":
            status, detail = _openeo_status(run.job_id)
            href = f"/jobs/{run.job_id}"
        else:
            status, detail = _native_status(run.job_id)
            href = f"/ingestions/jobs/{run.job_id}"
    except Exception as exc:
        return RunView(**run.model_dump(), status="failed", detail=f"Could not read the job: {exc}")
    return RunView(**run.model_dump(), status=status, job_href=href, detail=detail)


def consecutive_failures(task_id: str, limit: int = 20) -> int:
    """How many of the task's latest runs failed in a row: the signal worth telling someone about."""
    count = 0
    for run in list_runs(task_id=task_id, limit=limit):
        if view(run).status not in _FAILED:
            break
        count += 1
    return count
