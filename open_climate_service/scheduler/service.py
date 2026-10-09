"""APScheduler adapter for in-process dataset synchronization schedules."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
from open_climate_service.scheduler.schemas import ScheduleListResponse, ScheduleStatus
from open_climate_service.scheduler.store import StoredSchedule, list_schedules, store_stamp
from open_climate_service.shared.time import utc_now
from open_climate_service.tasks.models import Task

logger = logging.getLogger(__name__)

_WATCH_JOB_ID = "scheduler:store-watch"
_TASK_JOB_PREFIX = "task:"

CLOCK_LEASE = "scheduler"
LEASE_TTL_SECONDS = 3 * 30
"""How long the clock's owner keeps it without renewing. Renewed every watch, so a crashed
owner is replaced within this time by a process on standby (CLIM-997)."""


def _db_lease(holder: str, ttl: float) -> bool:
    from open_climate_service.state import db

    return db.acquire_lease(CLOCK_LEASE, holder, ttl)


def _db_release(holder: str) -> None:
    from open_climate_service.state import db

    db.release_lease(CLOCK_LEASE, holder)


def _cron_tasks() -> list[Task]:
    """Enabled refresh and workflow tasks on a cron: the tasks the clock runs besides syncs."""
    from open_climate_service.tasks.store import list_tasks

    return [
        task for task in list_tasks() if task.kind in ("refresh", "workflow") and task.cron is not None and task.enabled
    ]


def _run_task(task: Task, cause: str) -> CheckResult:
    from open_climate_service.tasks.dispatch import run_task

    return run_task(task, cause)


WATCH_SECONDS = 30
"""How often the clock owner looks for a store change made by another process."""


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
    # Refresh and workflow tasks on a cron (CLIM-1378), each with its trigger.
    tasks: tuple[tuple[Task, CronTrigger], ...] = ()


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
        tasks_loader: Callable[[], list[Task]] = _cron_tasks,
        task_runner: Callable[[Task, str], CheckResult] = _run_task,
        lease: Callable[[str, float], bool] = _db_lease,
        release: Callable[[str], None] = _db_release,
    ) -> None:
        # Every process with the scheduler enabled starts a clock, but only the lease holder runs
        # jobs on it; the others stand by and take over when the holder stops renewing. So
        # several replicas, or several workers on one machine, fire each schedule once.
        import socket
        from uuid import uuid4

        self._holder = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._lease = lease
        self._release = release
        self._leader = False
        self._tasks_loader = tasks_loader
        self._task_runner = task_runner
        self._tasks: tuple[tuple[Task, CronTrigger], ...] = ()
        # Called after every reload, so automation sees a task change made by another process.
        self._reload_listeners: list[Callable[[], None]] = []
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
        self._last_results: dict[str, CheckResult] = {}
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
        tasks = tuple(
            (task, CronTrigger.from_crontab(task.cron, timezone=config.timezone_info))
            for task in self._tasks_loader()
            if task.cron is not None
        )
        return _Plan(plan.config, plan.effective, plan.runnable, plan.refused, stamp, tasks)

    def add_reload_listener(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` after each reload of the store, in this process or after another's write."""
        self._reload_listeners.append(listener)

    def _apply_tasks(self, scheduler: AsyncIOScheduler, tasks: tuple[tuple[Task, CronTrigger], ...]) -> None:
        """Make the clock run exactly ``tasks`` among the task jobs."""
        wanted = {_TASK_JOB_PREFIX + task.id for task, _ in tasks}
        for job in list(scheduler.get_jobs()):
            if job.id.startswith(_TASK_JOB_PREFIX) and job.id not in wanted:
                scheduler.remove_job(job.id)
        for task, trigger in tasks:
            scheduler.add_job(
                self.run_task_now,
                trigger=trigger,
                args=[task],
                id=_TASK_JOB_PREFIX + task.id,
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )

    def run_task_now(self, task: Task, cause: str | None = None) -> CheckResult:
        """Run one cron task and keep its result for the status, as for a sync."""
        if cause is None:
            cause = f"cron:{utc_now().replace(second=0, microsecond=0).isoformat()}"
        try:
            result = self._task_runner(task, cause)
        except Exception as exc:
            result = CheckResult(
                schedule_id=task.id,
                dataset_id=task.target,
                outcome=CheckOutcome.ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("Task %s failed to start", task.id)
        self._last_results[_TASK_JOB_PREFIX + task.id] = result
        return result

    def task_status(self, task_id: str) -> tuple[object | None, CheckResult | None]:
        """The next fire time of a cron task on this clock, and its last result."""
        job = self._scheduler.get_job(_TASK_JOB_PREFIX + task_id) if self._scheduler is not None else None
        return getattr(job, "next_run_time", None), self._last_results.get(_TASK_JOB_PREFIX + task_id)

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
            if not plan.config.enabled:
                logger.info("Dataset scheduler is disabled")
                return
            if api_config.is_read_only():
                logger.info("Dataset scheduler will not start on a read-only instance")
                return

            self._tasks = plan.tasks
            scheduler = AsyncIOScheduler(timezone=plan.config.timezone_info)
            self._leader = self._try_lease()
            if self._leader:
                self._activate(scheduler)
            # Every watch renews or seeks the lease, and picks up writes another process made.
            scheduler.add_job(
                self.watch,
                trigger=IntervalTrigger(seconds=WATCH_SECONDS),
                id=_WATCH_JOB_ID,
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )
            scheduler.start()
            self._scheduler = scheduler
            if self._leader:
                logger.warning(
                    "Scheduler clock owned by %s with %d schedule(s) and %d other cron task(s)",
                    self._holder,
                    len(plan.runnable),
                    len(plan.tasks),
                )
            else:
                logger.warning("Scheduler on standby in %s; another process owns the clock", self._holder)

    def _try_lease(self) -> bool:
        try:
            return self._lease(self._holder, LEASE_TTL_SECONDS)
        except Exception:
            logger.exception("Could not reach the clock lease; this process runs no schedules meanwhile")
            return False

    def _activate(self, scheduler: AsyncIOScheduler) -> None:
        for entry, trigger in self._runnable:
            self._add(scheduler, entry, trigger)
        self._apply_tasks(scheduler, self._tasks)

    def _deactivate(self, scheduler: AsyncIOScheduler) -> None:
        for job in list(scheduler.get_jobs()):
            if job.id != _WATCH_JOB_ID:
                scheduler.remove_job(job.id)

    def watch(self) -> None:
        """Renew or take the clock lease, then reload if another process changed the store.

        A process that wins the lease starts running every schedule; one that loses it (it was
        paused too long, or the database moved on without it) stops, so two clocks never fire
        the same schedule.
        """
        held = self._try_lease()
        with self._lock:
            scheduler = self._scheduler
            if scheduler is not None and held and not self._leader:
                self._activate(scheduler)
                logger.warning("Scheduler clock taken over by %s", self._holder)
            elif scheduler is not None and not held and self._leader:
                self._deactivate(scheduler)
                logger.warning("Scheduler clock lost by %s; standing by", self._holder)
            self._leader = held
        self.reload_if_changed()

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
            scheduler = self._scheduler if self._leader else None
            if scheduler is not None:
                known = previous | {entry.schedule_id for entry in plan.effective}
                try:
                    self._apply(scheduler, plan.runnable, known)
                    self._apply_tasks(scheduler, plan.tasks)
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
            self._tasks = plan.tasks
            logger.info("Schedules reloaded: %d runnable, %d other cron tasks", len(plan.runnable), len(plan.tasks))
        for listener in self._reload_listeners:
            try:
                listener()
            except Exception:
                logger.exception("A reload listener failed")

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
        if self._leader:
            self._leader = False
            try:
                self._release(self._holder)
            except Exception:
                logger.exception("Could not give up the clock lease; it expires on its own")
        logger.info("Dataset scheduler stopped")

    # --- running -------------------------------------------------------------------------------

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
            return list(self._effective)

    def schedule_for(self, dataset_id: str) -> ScheduleStatus | None:
        """The schedule of one dataset, with its runtime state, or None."""
        return next((item for item in self.status().schedules if item.dataset_id == dataset_id), None)

    def status(self) -> ScheduleListResponse:
        """Return configuration plus volatile next/last-check state.

        A paused entry has no clock job, so its runtime fields are empty.
        """
        effective = self.effective()
        config = self._config
        assert config is not None
        apscheduler_jobs = {}
        if self._scheduler is not None:
            apscheduler_jobs = {job.id: job for job in self._scheduler.get_jobs()}

        schedules: list[ScheduleStatus] = []
        for entry in effective:
            schedule = entry.schedule
            result = self._last_results.get(entry.schedule_id) if entry.effective else None
            job = apscheduler_jobs.get(entry.schedule_id) if entry.effective else None
            schedules.append(
                ScheduleStatus(
                    schedule_id=entry.schedule_id,
                    dataset_id=schedule.dataset_id,
                    cron=schedule.cron,
                    timezone=config.timezone,
                    publish=schedule.publish,
                    max_attempts=schedule.max_attempts,
                    enabled=entry.enabled,
                    effective=entry.effective,
                    registered=job is not None,
                    next_check=getattr(job, "next_run_time", None),
                    last_check=result.checked_at if result else None,
                    last_outcome=result.outcome if result else None,
                    last_message=result.message if result else None,
                    last_job_id=result.job_id if result else None,
                )
            )
        return ScheduleListResponse(
            enabled=config.enabled,
            running=self._scheduler is not None and self._leader,
            clock_holder=self._holder if self._leader else None,
            timezone=config.timezone,
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
