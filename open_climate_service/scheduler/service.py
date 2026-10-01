"""APScheduler adapter for in-process dataset synchronization schedules."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.scheduler.config import DatasetSyncSchedule, SchedulerConfig, get_scheduler_config
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult, enqueue_sync
from open_climate_service.scheduler.schemas import ScheduleListResponse, ScheduleStatus

logger = logging.getLogger(__name__)


class SchedulerService:
    """Own the process-local clock while delegating sync decisions to OCS."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], SchedulerConfig] = get_scheduler_config,
        dispatcher: Callable[[DatasetSyncSchedule], CheckResult] = enqueue_sync,
        template_loader: Callable[[str], dict[str, Any] | None] = registry_datasets.get_dataset,
    ) -> None:
        self._config_loader = config_loader
        self._dispatcher = dispatcher
        self._template_loader = template_loader
        self._scheduler: AsyncIOScheduler | None = None
        self._config: SchedulerConfig | None = None
        self._last_results: dict[str, CheckResult] = {}

    def start(self) -> None:
        """Validate configuration and start callbacks when this process is enabled."""
        config = self._config_loader()
        self._config = config
        if not config.enabled:
            logger.info("Dataset scheduler is disabled")
            return
        if api_config.is_read_only():
            logger.info("Dataset scheduler will not start on a read-only instance")
            return

        scheduler = AsyncIOScheduler(timezone=config.timezone_info)
        registered = 0
        for schedule in config.dataset_sync:
            try:
                self._validate_target(schedule)
            except ValueError as exc:
                self._last_results[schedule.schedule_id] = CheckResult(
                    schedule_id=schedule.schedule_id,
                    dataset_id=schedule.dataset_id,
                    outcome=CheckOutcome.ERROR,
                    message=str(exc),
                )
                logger.error("Scheduled dataset %s was not registered: %s", schedule.dataset_id, exc)
                continue
            trigger = CronTrigger.from_crontab(schedule.cron, timezone=config.timezone_info)
            scheduler.add_job(
                self.check_now,
                trigger=trigger,
                args=[schedule],
                id=schedule.schedule_id,
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )
            registered += 1
        scheduler.start()
        self._scheduler = scheduler
        logger.warning(
            "Dataset scheduler enabled in process %d with %d schedule(s); "
            "exactly one OCS process may enable scheduling",
            os.getpid(),
            registered,
        )

    def _validate_target(self, schedule: DatasetSyncSchedule) -> None:
        """Reject unsupported targets without taking down unrelated API routes."""
        template = self._template_loader(schedule.dataset_id)
        if template is None:
            raise ValueError(f"Scheduled dataset {schedule.dataset_id!r} has no registered data source")
        if registry_datasets.is_future_facing(template):
            raise ValueError(
                f"Scheduled dataset {schedule.dataset_id!r} is future-facing; forecast refresh requires "
                "overlapping-window rematerialization and is not supported yet"
            )
        sync = template.get("sync")
        if not isinstance(sync, dict) or sync.get("kind") == "static":
            raise ValueError(f"Scheduled dataset {schedule.dataset_id!r} is not syncable")

    def shutdown(self) -> None:
        """Stop future callbacks without waiting for submitted native jobs."""
        if self._scheduler is None:
            return
        self._scheduler.shutdown(wait=False)
        self._scheduler = None
        logger.info("Dataset scheduler stopped")

    def check_now(self, schedule: DatasetSyncSchedule) -> CheckResult:
        """Run one isolated check and retain an operator-visible result."""
        try:
            result = self._dispatcher(schedule)
        except Exception as exc:
            result = CheckResult(
                schedule_id=schedule.schedule_id,
                dataset_id=schedule.dataset_id,
                outcome=CheckOutcome.ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("Scheduled sync check failed for %s", schedule.dataset_id)
        self._last_results[schedule.schedule_id] = result
        log = logger.warning if result.outcome == CheckOutcome.ERROR else logger.info
        log(
            "Scheduled sync check for %s: %s (%s)",
            schedule.dataset_id,
            result.outcome,
            result.message,
        )
        return result

    def status(self) -> ScheduleListResponse:
        """Return configuration plus volatile next/last-check state."""
        config = self._config or self._config_loader()
        apscheduler_jobs = {}
        if self._scheduler is not None:
            apscheduler_jobs = {job.id: job for job in self._scheduler.get_jobs()}

        schedules: list[ScheduleStatus] = []
        for schedule in config.dataset_sync:
            result = self._last_results.get(schedule.schedule_id)
            job = apscheduler_jobs.get(schedule.schedule_id)
            schedules.append(
                ScheduleStatus(
                    schedule_id=schedule.schedule_id,
                    dataset_id=schedule.dataset_id,
                    cron=schedule.cron,
                    timezone=config.timezone,
                    publish=schedule.publish,
                    max_attempts=schedule.max_attempts,
                    next_check=getattr(job, "next_run_time", None),
                    last_check=result.checked_at if result else None,
                    last_outcome=result.outcome if result else None,
                    last_message=result.message if result else None,
                    last_job_id=result.job_id if result else None,
                )
            )
        return ScheduleListResponse(
            enabled=config.enabled,
            running=self._scheduler is not None,
            timezone=config.timezone,
            schedules=schedules,
        )


_service: SchedulerService | None = None


def get_scheduler_service() -> SchedulerService:
    """Return the process-local scheduler singleton."""
    global _service
    if _service is None:
        _service = SchedulerService()
    return _service
