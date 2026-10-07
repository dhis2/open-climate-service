"""Operator-managed sync schedules, stored beside the instance configuration file.

The file keeps the schedules an operator wrote by hand in ``climate-service.yaml``; this store
keeps the ones created from the web interface or the API (CLIM-1242). The two are merged by
``scheduler.config.merge_schedules``, file first, so a stored entry for a dataset the file also
configures is shadowed rather than a second clock for the same dataset.

One JSON document holds every stored schedule, keyed by dataset id, because the one-per-dataset
rule makes the dataset id the natural key and the set is small. Writes take the cross-process
lock the other indexes use and replace the file atomically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.shared.persistence import atomic_json, index_lock
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)


class StoredSchedule(DatasetSyncSchedule):
    """A sync schedule an operator saved, with its pause switch and its history."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ScheduleStoreUnreadable(Exception):
    """The schedules file exists but cannot be parsed; nothing stored is trusted until fixed."""


def schedules_path() -> Path:
    """Where stored schedules live: ``<data_dir>/schedules.json``."""
    return api_config.get_data_root() / "schedules.json"


def store_stamp() -> tuple[int, int] | None:
    """What the clock owner watches: the file's modification time and size, or None when absent.

    Every write replaces the file atomically, so a change made by any process on the shared
    data directory moves the stamp; the process that owns the clock reloads when it does.
    """
    try:
        stat = schedules_path().stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _read_raw(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScheduleStoreUnreadable(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScheduleStoreUnreadable(f"{path} must hold a mapping of dataset id to schedule")
    return raw


def _parse(raw: dict[str, Any], path: Path) -> dict[str, StoredSchedule]:
    schedules: dict[str, StoredSchedule] = {}
    for dataset_id, value in raw.items():
        try:
            schedule = StoredSchedule.model_validate(value)
        except ValidationError as exc:
            raise ScheduleStoreUnreadable(f"{path}: schedule for {dataset_id!r} is invalid: {exc}") from exc
        if schedule.dataset_id != dataset_id:
            raise ScheduleStoreUnreadable(
                f"{path}: schedule stored under {dataset_id!r} names dataset {schedule.dataset_id!r}"
            )
        schedules[dataset_id] = schedule
    return schedules


def list_schedules() -> list[StoredSchedule]:
    """Every stored schedule, by dataset id. Raises when the file is unreadable."""
    path = schedules_path()
    schedules = _parse(_read_raw(path), path)
    return [schedules[key] for key in sorted(schedules)]


def get_schedule(dataset_id: str) -> StoredSchedule | None:
    """One stored schedule, or None."""
    path = schedules_path()
    return _parse(_read_raw(path), path).get(dataset_id)


def _write(schedules: dict[str, StoredSchedule], path: Path) -> None:
    atomic_json(path, {key: schedules[key].model_dump(mode="json") for key in sorted(schedules)})


def save_schedule(schedule: StoredSchedule, *, create: bool) -> StoredSchedule:
    """Create or replace one stored schedule under the store lock.

    ``create`` refuses an existing dataset id, so two operators adding the same dataset at
    once cannot both believe they created it; an update refuses a missing one.
    """
    path = schedules_path()
    with index_lock(path):
        schedules = _parse(_read_raw(path), path)
        existing = schedules.get(schedule.dataset_id)
        if create and existing is not None:
            raise ValueError(f"A schedule for dataset '{schedule.dataset_id}' already exists")
        if not create and existing is None:
            raise ValueError(f"No stored schedule for dataset '{schedule.dataset_id}'")
        stamped = schedule.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else schedule.created_at,
                "updated_at": utc_now(),
            }
        )
        schedules[schedule.dataset_id] = stamped
        _write(schedules, path)
    return stamped


def set_enabled(dataset_id: str, enabled: bool) -> StoredSchedule:
    """Pause or resume one stored schedule."""
    path = schedules_path()
    with index_lock(path):
        schedules = _parse(_read_raw(path), path)
        existing = schedules.get(dataset_id)
        if existing is None:
            raise ValueError(f"No stored schedule for dataset '{dataset_id}'")
        if existing.enabled == enabled:
            return existing
        updated = existing.model_copy(update={"enabled": enabled, "updated_at": utc_now()})
        schedules[dataset_id] = updated
        _write(schedules, path)
    return updated


def delete_schedule(dataset_id: str) -> bool:
    """Remove one stored schedule; True when something was removed."""
    path = schedules_path()
    with index_lock(path):
        schedules = _parse(_read_raw(path), path)
        if dataset_id not in schedules:
            return False
        del schedules[dataset_id]
        _write(schedules, path)
    return True
