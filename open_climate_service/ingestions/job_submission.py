"""Shared submission helpers for native ingestion jobs."""

from open_climate_service.ingestions.processes import execute_sync
from open_climate_service.jobs.models import JobRecord
from open_climate_service.jobs.service import get_job_service

INGESTION_JOB_HREF_BASE = "/ingestions/jobs"


def submit_sync_job(
    *,
    dataset_id: str,
    end: str | None,
    publish: bool,
    label: str = "sync",
    max_attempts: int = 1,
    outcome_source: str | None = None,
) -> JobRecord:
    """Submit a sync through the native job service used by HTTP and schedules."""
    return get_job_service().submit_callable_job(
        func=execute_sync,
        label=label,
        request={"dataset_id": dataset_id, "end": end, "publish": publish},
        max_attempts=max_attempts,
        job_href_base=INGESTION_JOB_HREF_BASE,
        outcome_source=outcome_source,
    )
