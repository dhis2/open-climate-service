"""Sync schedules, as the ``sync`` tasks of the one tasks store (CLIM-1242, CLIM-1378).

The schedule API and pages keep their dataset-keyed shape: one sync schedule per dataset, at
``/schedules/sync/{dataset_id}``. Underneath, each is a task with ``kind: sync`` and id
``sync-<dataset_id>`` in ``<data_dir>/tasks.json``, next to the workflow and delivery steps,
so there is one store and one reload for everything the clock and automation run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.shared.time import utc_now
from open_climate_service.tasks import store as task_store
from open_climate_service.tasks.models import Task
from open_climate_service.tasks.store import TaskStoreUnreadable

ScheduleStoreUnreadable = TaskStoreUnreadable
"""Kept as the schedule API's name for an unreadable store."""

store_stamp = task_store.store_stamp


class StoredSchedule(DatasetSyncSchedule):
    """A sync schedule an operator saved, with its pause switch and its history."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


def sync_task_id(dataset_id: str) -> str:
    """The task id of a dataset's sync schedule."""
    return f"sync-{dataset_id}"


def schedules_path() -> Path:
    """The file holding sync schedules: the tasks store."""
    return task_store.tasks_path()


def _to_schedule(task: Task) -> StoredSchedule:
    assert task.cron is not None
    return StoredSchedule(
        dataset_id=task.target,
        cron=task.cron,
        publish=task.publish,
        max_attempts=task.max_attempts,
        enabled=task.enabled,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _to_task(schedule: StoredSchedule) -> Task:
    return Task(
        id=sync_task_id(schedule.dataset_id),
        kind="sync",
        target=schedule.dataset_id,
        cron=schedule.cron,
        publish=schedule.publish,
        max_attempts=schedule.max_attempts,
        enabled=schedule.enabled,
        created_at=schedule.created_at,
        updated_at=schedule.updated_at,
    )


def _scheduled(tasks: list[Task]) -> list[Task]:
    """Sync tasks on a cron. A by-hand sync task has no schedule to list."""
    return [task for task in tasks if task.kind == "sync" and task.cron is not None]


def _one_per_dataset(tasks: list[Any]) -> None:
    datasets = [task.target for task in tasks if task.kind == "sync" and task.cron is not None]
    duplicates = sorted({dataset for dataset in datasets if datasets.count(dataset) > 1})
    if duplicates:
        raise ValueError(f"A dataset has one sync schedule; more than one for {', '.join(duplicates)}")


def list_schedules() -> list[StoredSchedule]:
    """Every sync schedule, by dataset id. Raises when the store is unreadable."""
    schedules = [_to_schedule(task) for task in _scheduled(task_store.list_tasks("sync"))]
    return sorted(schedules, key=lambda schedule: schedule.dataset_id)


def get_schedule(dataset_id: str) -> StoredSchedule | None:
    """One dataset's sync schedule, or None."""
    return next((schedule for schedule in list_schedules() if schedule.dataset_id == dataset_id), None)


def save_schedule(schedule: StoredSchedule, *, create: bool) -> StoredSchedule:
    """Create or replace one dataset's sync schedule.

    ``create`` refuses an existing schedule for the dataset; an update refuses a missing one.
    """
    if create and get_schedule(schedule.dataset_id) is not None:
        raise ValueError(f"A schedule for dataset '{schedule.dataset_id}' already exists")
    if not create and get_schedule(schedule.dataset_id) is None:
        raise ValueError(f"No stored schedule for dataset '{schedule.dataset_id}'")
    saved = task_store.save_task(_to_task(schedule), create=create, check=_one_per_dataset)
    return _to_schedule(saved)


def set_enabled(dataset_id: str, enabled: bool) -> StoredSchedule:
    """Pause or resume one dataset's sync schedule."""
    if get_schedule(dataset_id) is None:
        raise ValueError(f"No stored schedule for dataset '{dataset_id}'")
    return _to_schedule(task_store.set_enabled(sync_task_id(dataset_id), enabled))


def delete_schedule(dataset_id: str) -> bool:
    """Remove one dataset's sync schedule; True when something was removed."""
    if get_schedule(dataset_id) is None:
        return False
    return task_store.delete_task(sync_task_id(dataset_id))
