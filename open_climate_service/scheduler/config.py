"""Validated instance configuration for scheduled dataset synchronization.

Two sources feed the scheduler: the ``scheduler`` block of ``climate-service.yaml``, which an
operator edits by hand, and the schedules saved from the web interface or the API
(``scheduler.store``). ``merge_schedules`` combines them into the one effective list the clock
runs, file first.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_climate_service import config as api_config

if TYPE_CHECKING:
    from open_climate_service.scheduler.store import StoredSchedule

ScheduleSource = Literal["file", "store"]


class DatasetSyncSchedule(BaseModel):
    """One cron-driven check of an existing managed dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    cron: str = Field(min_length=1)
    publish: bool = True
    max_attempts: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_cron(self) -> "DatasetSyncSchedule":
        try:
            CronTrigger.from_crontab(self.cron)
        except ValueError as exc:
            raise ValueError(f"invalid five-field cron expression {self.cron!r}: {exc}") from exc
        return self

    @property
    def schedule_id(self) -> str:
        """Return the stable identifier derived from the one-schedule-per-dataset rule."""
        return f"dataset-sync:{self.dataset_id}"


class SchedulerConfig(BaseModel):
    """Scheduler configuration loaded from ``climate-service.yaml``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timezone: str = "UTC"
    dataset_sync: list[DatasetSyncSchedule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_configuration(self) -> "SchedulerConfig":
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown scheduler timezone {self.timezone!r}") from exc
        dataset_ids = [schedule.dataset_id for schedule in self.dataset_sync]
        duplicates = sorted({dataset_id for dataset_id in dataset_ids if dataset_ids.count(dataset_id) > 1})
        if duplicates:
            raise ValueError(f"only one scheduler entry is allowed per dataset: {duplicates}")
        return self

    @property
    def timezone_info(self) -> ZoneInfo:
        """Return the validated IANA timezone."""
        return ZoneInfo(self.timezone)


class EffectiveSchedule(BaseModel):
    """One entry of the merged schedule list, with where it came from and whether it runs.

    A file entry always runs while the scheduler is enabled; the file has no pause switch
    short of removing the entry. A stored entry runs when it is enabled and not shadowed by a
    file entry for the same dataset. A shadowed entry stays listed, marked ineffective, so an
    operator can see why their saved schedule does nothing and which entry wins.
    """

    model_config = ConfigDict(frozen=True)

    schedule: DatasetSyncSchedule
    source: ScheduleSource
    enabled: bool = True
    shadowed: bool = False

    @property
    def dataset_id(self) -> str:
        return self.schedule.dataset_id

    @property
    def schedule_id(self) -> str:
        return self.schedule.schedule_id

    @property
    def effective(self) -> bool:
        """Whether this is the entry the clock would run: enabled and not shadowed.

        Whether it actually runs also depends on the scheduler being enabled in this process
        and on the dataset resolving; ``ScheduleStatus.registered`` reports that.
        """
        return self.enabled and not self.shadowed


def merge_schedules(config: SchedulerConfig, stored: Iterable[StoredSchedule]) -> list[EffectiveSchedule]:
    """File entries first, then stored ones; a stored entry for a file-configured dataset is shadowed."""
    effective: list[EffectiveSchedule] = [
        EffectiveSchedule(schedule=schedule, source="file") for schedule in config.dataset_sync
    ]
    file_datasets = {schedule.dataset_id for schedule in config.dataset_sync}
    for item in sorted(stored, key=lambda entry: entry.dataset_id):
        effective.append(
            EffectiveSchedule(
                schedule=DatasetSyncSchedule(
                    dataset_id=item.dataset_id, cron=item.cron, publish=item.publish, max_attempts=item.max_attempts
                ),
                source="store",
                enabled=item.enabled,
                shadowed=item.dataset_id in file_datasets,
            )
        )
    return effective


def get_scheduler_config() -> SchedulerConfig:
    """Load scheduler configuration from the instance configuration."""
    raw = api_config.get_config().get("scheduler", {})
    if not isinstance(raw, dict):
        raise ValueError("scheduler in CLIMATE_SERVICE_CONFIG must be a mapping")
    return SchedulerConfig.model_validate(raw)
