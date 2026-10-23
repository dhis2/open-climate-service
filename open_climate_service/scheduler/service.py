"""APScheduler adapter for in-process dataset synchronization schedules."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import HTTPException

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.scheduler.config import (
    DatasetSyncSchedule,
    EffectiveSchedule,
    SchedulerConfig,
    effective_schedules,
    get_scheduler_config,
)
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult, enqueue_sync
from open_climate_service.scheduler.results import SharedResults, SharedResultsStore
from open_climate_service.scheduler.schemas import ClockStatus, ScheduleListResponse, ScheduleStatus
from open_climate_service.scheduler.store import StoredSchedule, list_schedules, store_stamp
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

_WATCH_JOB_ID = "scheduler:store-watch"
_BEAT_JOB_ID = "scheduler:heartbeat"
WATCH_SECONDS = 30
"""How often the clock owner looks for a store change made by another process, and reports that it runs."""
CLOCK_LIVE = timedelta(seconds=3 * WATCH_SECONDS)
"""How old the clock's last heartbeat may be before other processes stop trusting that it runs."""


def validate_schedule_target(template: dict[str, Any] | None, dataset_id: str) -> None:
    """Reject a dataset the clock cannot sync, naming why.

    Shared by the clock, which skips such an entry without taking down unrelated routes, and
    by the schedule API, which refuses to save one.
    """
    if template is None:
        raise ValueError(f"Scheduled dataset {dataset_id!r} has no registered data source")
    if registry_datasets.is_future_facing(template):
        raise ValueError(
            f"Scheduled dataset {dataset_id!r} is future-facing; forecast refresh requires "
            "overlapping-window rematerialization and is not supported yet"
        )
    sync = template.get("sync")
    if not isinstance(sync, dict) or sync.get("kind") == "static":
        raise ValueError(f"Scheduled dataset {dataset_id!r} is not syncable")


def resolve_schedule_template(
    dataset_id: str, template_loader: Callable[[str], dict[str, Any] | None] | None = None
) -> dict[str, Any] | None:
    """Resolve a managed dataset to its source template, with a direct-template fallback."""
    from open_climate_service.ingestions.services import get_latest_artifact_for_dataset_or_404

    load_template = template_loader or registry_datasets.get_dataset
    try:
        latest = get_latest_artifact_for_dataset_or_404(dataset_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return load_template(dataset_id)
    return load_template(latest.source_dataset_id or latest.dataset_id)


@dataclass(frozen=True)
class _Plan:
    """What a load resolved to, before anything is touched.

    ``runnable`` holds every effective entry whose target resolved and whose trigger built,
    with that trigger; ``refused`` holds the effective entries the clock will not run, with
    the reason recorded for the status.
    """

    config: SchedulerConfig
    effective: list[EffectiveSchedule]
    runnable: list[tuple[EffectiveSchedule, CronTrigger]]
    refused: dict[str, CheckResult]
    stamp: str | None = None


class SchedulerService:
    """Own the process-local clock while delegating sync decisions to OCS.

    The effective schedule list comes only from the shared store (CLIM-1242).
    ``start`` registers it once; ``reload`` re-reads the store, resolves
    and validates the whole before touching anything, then reconciles the running jobs
    by id. A load that cannot be validated leaves the previous list in force.
    """

    def __init__(
        self,
        *,
        config_loader: Callable[[], SchedulerConfig] = get_scheduler_config,
        store_loader: Callable[[], list[StoredSchedule]] = list_schedules,
        dispatcher: Callable[[DatasetSyncSchedule], CheckResult] = enqueue_sync,
        template_loader: Callable[[str], dict[str, Any] | None] | None = None,
        stamp_loader: Callable[[], str | None] = store_stamp,
        results: SharedResultsStore | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._store_loader = store_loader
        self._stamp_loader = stamp_loader
        self._dispatcher = dispatcher
        # Resolved at call time, so the registry the rest of the process sees is the one used.
        self._template_loader = template_loader
        self._scheduler: AsyncIOScheduler | None = None
        self._config: SchedulerConfig | None = None
        self._effective: list[EffectiveSchedule] = []
        self._runnable: list[tuple[EffectiveSchedule, CronTrigger]] = []
        self._stamp: str | None = None
        # This process's own refusals and checks. Check results are also recorded in
        # ``self._results``, shared with every process on the data directory; status reports
        # whichever is newer.
        self._last_results: dict[str, CheckResult] = {}
        self._results = results if results is not None else SharedResultsStore()
        # Schedules the last load would not run; the clock never fires them, so no next check.
        self._refused: set[str] = set()
        self._reload_error: str | None = None
        self._lock = threading.Lock()

    # --- loading -------------------------------------------------------------------------------

    def _plan(self, config: SchedulerConfig, effective: list[EffectiveSchedule]) -> _Plan:
        """Resolve every effective entry's target and trigger without touching the clock."""
        runnable: list[tuple[EffectiveSchedule, CronTrigger]] = []
        refused: dict[str, CheckResult] = {}
        for entry in effective:
            if not entry.effective:
                continue
            schedule = entry.schedule
            try:
                validate_schedule_target(
                    resolve_schedule_template(schedule.dataset_id, self._template_loader), schedule.dataset_id
                )
                trigger = CronTrigger.from_crontab(schedule.cron, timezone=config.timezone_info)
            except ValueError as exc:
                refused[entry.schedule_id] = CheckResult(
                    schedule_id=entry.schedule_id,
                    dataset_id=schedule.dataset_id,
                    outcome=CheckOutcome.ERROR,
                    message=str(exc),
                )
                logger.error("Scheduled dataset %s was not registered: %s", schedule.dataset_id, exc)
                continue
            runnable.append((entry, trigger))
        return _Plan(config=config, effective=effective, runnable=runnable, refused=refused)

    def _load(self, config: SchedulerConfig | None = None) -> _Plan:
        """Read the clock config and store, then resolve the schedules.

        The stamp is taken before the store is read, so a write that lands between the two
        moves the stamp past what was loaded and the watch reloads once more.
        """
        stamp = self._stamp_loader()
        config = config if config is not None else self._config_loader()
        stored = self._store_loader()
        plan = self._plan(config, effective_schedules(stored))
        return _Plan(plan.config, plan.effective, plan.runnable, plan.refused, stamp)

    def _apply(
        self, scheduler: AsyncIOScheduler, runnable: list[tuple[EffectiveSchedule, CronTrigger]], known: set[str]
    ) -> None:
        """Make the clock run exactly ``runnable`` among the ``known`` schedule ids."""
        for entry, trigger in runnable:
            self._add(scheduler, entry, trigger)
        wanted = {entry.schedule_id for entry, _ in runnable}
        for schedule_id in sorted(known - wanted):
            self._remove(scheduler, schedule_id)

    def _add(self, scheduler: AsyncIOScheduler, entry: EffectiveSchedule, trigger: CronTrigger) -> None:
        scheduler.add_job(
            self.check_now,
            trigger=trigger,
            args=[entry.schedule],
            id=entry.schedule_id,
            coalesce=True,
            max_instances=1,
            replace_existing=True,
        )

    @staticmethod
    def _remove(scheduler: AsyncIOScheduler, schedule_id: str) -> None:
        if scheduler.get_job(schedule_id) is not None:
            scheduler.remove_job(schedule_id)

    def start(self) -> None:
        """Validate configuration and start callbacks when this process is enabled."""
        with self._lock:
            config = self._config_loader()
            try:
                plan = self._load(config)
            except Exception as exc:
                # Keep the API available for repair, but never run schedules from another
                # source or pretend a broken store is an empty, healthy one.
                logger.exception("Stored schedules could not be read; the clock has no schedules")
                plan = self._plan(config, [])
                self._reload_error = f"{type(exc).__name__}: {exc}"
            self._config = plan.config
            self._effective = plan.effective
            self._runnable = plan.runnable
            self._stamp = plan.stamp
            self._last_results.update(plan.refused)
            self._refused = set(plan.refused)
            if not plan.config.enabled:
                logger.info("Dataset scheduler is disabled")
                return
            if api_config.is_read_only():
                logger.info("Dataset scheduler will not start on a read-only instance")
                return

            scheduler = AsyncIOScheduler(timezone=plan.config.timezone_info)
            for entry, trigger in plan.runnable:
                self._add(scheduler, entry, trigger)
            # Writes from another process on the shared data directory reach this clock through
            # the store file, not through this process's routes; watch it.
            scheduler.add_job(
                self.reload_if_changed,
                trigger=IntervalTrigger(seconds=WATCH_SECONDS),
                id=_WATCH_JOB_ID,
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )
            # Tells every process on the data directory that a clock runs, and where.
            scheduler.add_job(
                self.beat,
                trigger=IntervalTrigger(seconds=WATCH_SECONDS),
                id=_BEAT_JOB_ID,
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )
            scheduler.start()
            self._scheduler = scheduler
            self.beat()
            logger.warning(
                "Dataset scheduler enabled in process %d with %d schedule(s); "
                "exactly one OCS process may enable scheduling",
                os.getpid(),
                len(plan.runnable),
            )

    def reload(self) -> None:
        """Re-read the store, resolve the whole list, then reconcile the running jobs.

        Nothing is touched until every entry has resolved: the store has been parsed, each
        effective entry's target checked and its trigger built. When the load fails, the
        previous list stays in force and ``status`` reports why. An entry whose target no
        longer resolves is taken off the clock and reported, so a job never keeps firing with
        a configuration the status no longer describes. When the clock is not running
        (disabled, or a read-only instance) only the list used by ``status`` changes.
        """
        with self._lock:
            try:
                plan = self._load()
            except Exception as exc:
                self._reload_error = f"{type(exc).__name__}: {exc}; the previous schedules stay in force"
                logger.error("Schedule reload refused; keeping the previous schedules: %s", exc)
                return
            previous = {entry.schedule_id for entry in self._effective}
            runnable_ids = {entry.schedule_id for entry, _ in plan.runnable}
            scheduler = self._scheduler
            if scheduler is not None:
                known = previous | {entry.schedule_id for entry in plan.effective}
                try:
                    self._apply(scheduler, plan.runnable, known)
                except Exception as exc:
                    # The plan was sound, so this is the clock itself refusing. Put the
                    # previous jobs back so the status keeps describing what runs.
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.exception("The clock refused the reloaded schedules; restoring the previous ones")
                    try:
                        self._apply(scheduler, self._runnable, known)
                    except Exception as restore_exc:
                        logger.exception("The previous schedules could not be restored either")
                        self._reload_error = (
                            "the clock refused the change and the previous schedules could not be restored; "
                            f"the clock may run settings this list does not show ({reason}; then "
                            f"{type(restore_exc).__name__}: {restore_exc})"
                        )
                        self._stamp = plan.stamp
                        return
                    self._reload_error = (
                        f"the clock refused the change and the previous schedules were restored ({reason})"
                    )
                    self._stamp = plan.stamp
                    return
            self._reload_error = None
            self._config = plan.config
            self._effective = plan.effective
            self._runnable = plan.runnable
            self._stamp = plan.stamp
            self._last_results = {key: value for key, value in self._last_results.items() if key in runnable_ids}
            self._last_results.update(plan.refused)
            self._refused = set(plan.refused)
            if scheduler is not None:
                # Only the clock's process writes the shared results; a read-only one never does.
                try:
                    self._results.keep_only({entry.schedule_id for entry in plan.effective})
                except Exception:
                    logger.exception("Results of removed schedules could not be pruned")
            logger.info("Schedules reloaded: %d runnable", len(plan.runnable))

    def reload_if_changed(self) -> bool:
        """Reload when another process changed the store since this one last loaded it.

        Runs on the clock every ``WATCH_SECONDS`` in the process that owns it, so a pause or
        an edit handled by an API-only replica reaches the clock without that replica knowing
        where the clock is. Returns whether a reload ran.
        """
        current_stamp = self._stamp_loader()
        with self._lock:
            loaded_stamp = self._stamp
        if current_stamp == loaded_stamp:
            return False
        logger.info("Stored schedules changed outside this process; reloading")
        self.reload()
        return True

    def shutdown(self) -> None:
        """Stop future callbacks without waiting for submitted native jobs."""
        if self._scheduler is None:
            return
        self._scheduler.shutdown(wait=False)
        self._scheduler = None
        try:
            self._results.stop_beating()
        except Exception:
            logger.exception("The clock's heartbeat could not be cleared")
        logger.info("Dataset scheduler stopped")

    def beat(self) -> None:
        """Report through the shared data directory that the clock runs in this process."""
        try:
            self._results.beat(self._config.timezone if self._config is not None else "UTC")
        except Exception:
            logger.exception("The clock's heartbeat could not be recorded")

    # --- running -------------------------------------------------------------------------------

    def check_now(self, schedule: DatasetSyncSchedule) -> CheckResult:
        """Run one isolated check and record its result for every process to report."""
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
        # Kept here too, so this process still reports its own last check if the file cannot be written.
        self._last_results[schedule.schedule_id] = result
        if not api_config.is_read_only():
            try:
                self._results.record(result)
            except Exception:
                logger.exception("The result of the check for %s could not be recorded", schedule.dataset_id)
        log = logger.warning if result.outcome == CheckOutcome.ERROR else logger.info
        log(
            "Scheduled sync check for %s: %s (%s)",
            schedule.dataset_id,
            result.outcome,
            result.message,
        )
        return result

    # --- reading -------------------------------------------------------------------------------

    def effective(self) -> list[EffectiveSchedule]:
        """The stored list the clock runs, or would run if enabled.

        Looks for a store change first, so a process that does not own the clock, and so has
        no watch job, still lists what another process saved. The check is one digest of a
        small file.
        """
        self.reload_if_changed()
        with self._lock:
            if self._config is None:
                plan = self._load()
                self._config, self._effective = plan.config, plan.effective
                self._stamp = plan.stamp
                self._last_results.update(plan.refused)
                self._refused = set(plan.refused)
            return list(self._effective)

    def schedule_for(self, dataset_id: str) -> ScheduleStatus | None:
        """The schedule of one dataset, with its runtime state, or None."""
        return next((item for item in self.status().schedules if item.dataset_id == dataset_id), None)

    def _clock_status(self, shared: SharedResults) -> ClockStatus | None:
        """Where the clock runs: here, or the process whose heartbeat is recent; None when neither."""
        if self._scheduler is not None:
            beat = shared.clock if shared.clock is not None and shared.clock.is_this_process() else None
            return ClockStatus(here=True, beat_at=beat.beat_at if beat else utc_now())
        beat = shared.clock
        if beat is None or not beat.is_live(CLOCK_LIVE):
            return None
        return ClockStatus(here=False, beat_at=beat.beat_at)

    @staticmethod
    def _clock_timezone(shared: SharedResults, clock: ClockStatus | None, config: SchedulerConfig) -> str:
        """The zone the clock's crons fire in: its own when it runs elsewhere and said so, else ours."""
        beat = shared.clock
        if clock is None or clock.here or beat is None or not beat.timezone:
            return config.timezone
        try:
            ZoneInfo(beat.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            return config.timezone
        return beat.timezone

    @staticmethod
    def _next_from_cron(cron: str, timezone: str) -> datetime | None:
        """When the clock, running in another process, next fires this cron in its ``timezone``."""
        zone = ZoneInfo(timezone)
        try:
            trigger = CronTrigger.from_crontab(cron, timezone=zone)
        except ValueError:
            return None
        fire: datetime | None = trigger.get_next_fire_time(None, utc_now().astimezone(zone))
        return fire

    def status(self) -> ScheduleListResponse:
        """Return configuration plus the next and last check of each schedule.

        Last checks come from the shared results file, so every process on the data directory
        reports what the clock's process ran, also after that process restarts. The next check
        is the clock job's when the clock runs here; elsewhere it is computed from the cron, and
        only while a clock reports that it runs, since otherwise nothing would fire. A paused
        entry has no clock job, so its runtime fields are empty.
        """
        effective = self.effective()
        config = self._config
        assert config is not None
        apscheduler_jobs = {}
        if self._scheduler is not None:
            apscheduler_jobs = {job.id: job for job in self._scheduler.get_jobs()}
        shared = self._results.read()
        clock = self._clock_status(shared)
        timezone = self._clock_timezone(shared, clock, config)

        schedules: list[ScheduleStatus] = []
        for entry in effective:
            schedule = entry.schedule
            result: CheckResult | None = None
            next_check: datetime | None = None
            job = None
            if entry.effective:
                found = [
                    r for r in (self._last_results.get(entry.schedule_id), shared.results.get(entry.schedule_id)) if r
                ]
                result = max(found, key=lambda r: r.checked_at) if found else None
                job = apscheduler_jobs.get(entry.schedule_id)
                if job is not None:
                    next_check = getattr(job, "next_run_time", None)
                elif clock is not None and not clock.here and entry.schedule_id not in self._refused:
                    next_check = self._next_from_cron(schedule.cron, timezone)
            schedules.append(
                ScheduleStatus(
                    schedule_id=entry.schedule_id,
                    dataset_id=schedule.dataset_id,
                    cron=schedule.cron,
                    timezone=timezone,
                    publish=schedule.publish,
                    max_attempts=schedule.max_attempts,
                    enabled=entry.enabled,
                    effective=entry.effective,
                    registered=job is not None,
                    next_check=next_check,
                    last_check=result.checked_at if result else None,
                    last_outcome=result.outcome if result else None,
                    last_message=result.message if result else None,
                    last_job_id=result.job_id if result else None,
                )
            )
        return ScheduleListResponse(
            enabled=config.enabled,
            running=self._scheduler is not None,
            clock=clock,
            timezone=timezone,
            reload_error=self._reload_error,
            schedules=schedules,
        )


_service: SchedulerService | None = None


def get_scheduler_service() -> SchedulerService:
    """Return the process-local scheduler singleton."""
    global _service
    if _service is None:
        _service = SchedulerService()
    return _service
