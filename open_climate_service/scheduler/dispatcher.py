"""Plan and submit scheduled dataset syncs without owning timing logic."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from fastapi import HTTPException
from pydantic import BaseModel, Field

from open_climate_service import config as api_config
from open_climate_service.ingestions.job_submission import submit_sync_job
from open_climate_service.jobs.models import JobStatus
from open_climate_service.jobs.service import get_job_service
from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.shared.time import utc_now

_ACTIVE_JOB_STATUSES = {JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.RETRYING}
# All native jobs that may write the same managed dataset store. Enqueuing a
# scheduled sync while one is active only burns retry attempts on the store lock.
_ACTIVE_WRITER_PROCESS_IDS = {"ingestion", "sync", "scheduled-sync"}


class CheckOutcome(StrEnum):
    """Outcome of one scheduled dataset check."""

    SUBMITTED = "submitted"
    NOT_MATERIALIZED = "not_materialized"
    ALREADY_RUNNING = "already_running"
    READ_ONLY = "read_only"
    ERROR = "error"


class CheckResult(BaseModel):
    """Operational result of one check, shared through the data directory (``scheduler.results``)."""

    schedule_id: str
    dataset_id: str
    outcome: CheckOutcome
    message: str
    job_id: str | None = None
    checked_at: datetime = Field(default_factory=utc_now)


def _has_active_writer_job(dataset_id: str) -> bool:
    return any(
        record.process_id in _ACTIVE_WRITER_PROCESS_IDS
        and record.status in _ACTIVE_JOB_STATUSES
        and record.request.get("dataset_id") == dataset_id
        for record in get_job_service().list_jobs().jobs
    )


def _dataset_is_materialized(dataset_id: str) -> bool:
    """Return True when the managed dataset has at least one local artifact.

    Uses the same local lookup the sync job performs before planning, so the
    guard can never disagree with the job's 404. Reads the local artifacts index
    only (no network), so this is safe in the clock callback and deliberately
    distinct from planning, which still runs inside the submitted native job.
    """
    from open_climate_service.ingestions.services import get_latest_artifact_for_dataset_or_404

    try:
        get_latest_artifact_for_dataset_or_404(dataset_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return False
        raise
    return True


def enqueue_sync(schedule: DatasetSyncSchedule) -> CheckResult:
    """Enqueue one due sync without doing upstream work in the clock callback."""
    if api_config.is_read_only():
        return CheckResult(
            schedule_id=schedule.schedule_id,
            dataset_id=schedule.dataset_id,
            outcome=CheckOutcome.READ_ONLY,
            message="Scheduler is disabled for a read-only instance",
        )

    if _has_active_writer_job(schedule.dataset_id):
        return CheckResult(
            schedule_id=schedule.schedule_id,
            dataset_id=schedule.dataset_id,
            outcome=CheckOutcome.ALREADY_RUNNING,
            message="An active sync job already exists for this dataset",
        )

    if not _dataset_is_materialized(schedule.dataset_id):
        return CheckResult(
            schedule_id=schedule.schedule_id,
            dataset_id=schedule.dataset_id,
            outcome=CheckOutcome.NOT_MATERIALIZED,
            message="Dataset has not been ingested yet; skipping scheduled sync",
        )

    job = submit_sync_job(
        dataset_id=schedule.dataset_id,
        end=None,
        publish=schedule.publish,
        label="scheduled-sync",
        max_attempts=schedule.max_attempts,
    )
    return CheckResult(
        schedule_id=schedule.schedule_id,
        dataset_id=schedule.dataset_id,
        outcome=CheckOutcome.SUBMITTED,
        message="Scheduled sync submitted",
        job_id=job.job_id,
    )
