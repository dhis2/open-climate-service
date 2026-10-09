"""Run one step now, whatever its kind: on its cron, or by hand (CLIM-1378)."""

from __future__ import annotations

from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult, enqueue_sync
from open_climate_service.steps.models import Step


def run_step(step: Step, cause: str) -> CheckResult:
    """Start one run of ``step`` and say what was submitted.

    ``cause`` identifies the run: the cron fire time, or a unique id for a run by hand. A
    workflow step uses it for its deterministic job id. Sync and refresh submit a native job,
    a workflow step an openEO job through the automation service, so its delivery step and
    retries apply exactly as for a run started by an event.
    """
    if step.kind == "sync":
        result = enqueue_sync(
            DatasetSyncSchedule(
                dataset_id=step.target,
                cron=step.cron or "0 0 * * *",
                publish=step.publish,
                max_attempts=step.max_attempts,
            )
        )
        return result.model_copy(update={"schedule_id": step.id})
    if step.kind == "refresh":
        from open_climate_service.features.services import execute_feature_refresh
        from open_climate_service.ingestions.job_submission import INGESTION_JOB_HREF_BASE
        from open_climate_service.jobs.service import get_job_service

        job = get_job_service().submit_callable_job(
            func=execute_feature_refresh,
            label="feature refresh",
            request={"collection_id": step.target, "publish": step.publish},
            max_attempts=step.max_attempts,
            job_href_base=INGESTION_JOB_HREF_BASE,
        )
        return CheckResult(
            schedule_id=step.id,
            dataset_id=step.target,
            outcome=CheckOutcome.SUBMITTED,
            message="Feature collection refresh submitted",
            job_id=job.job_id,
        )
    if step.kind == "workflow":
        from open_climate_service.automation.service import get_workflow_automation_service

        job_id = get_workflow_automation_service().run_now(step.id, cause)
        return CheckResult(
            schedule_id=step.id,
            dataset_id=step.target,
            outcome=CheckOutcome.SUBMITTED,
            message=f"Workflow {step.target} submitted",
            job_id=job_id,
        )
    raise ValueError(f"A {step.kind} step runs after the step it follows, not on its own")
