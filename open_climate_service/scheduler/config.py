"""Validated clock settings and schedules loaded from the shared schedules store."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_climate_service import config as api_config

if TYPE_CHECKING:
    from open_climate_service.scheduler.store import StoredSchedule


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

    @model_validator(mode="after")
    def validate_configuration(self) -> "SchedulerConfig":
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown scheduler timezone {self.timezone!r}") from exc
        return self

    @property
    def timezone_info(self) -> ZoneInfo:
        """Return the validated IANA timezone."""
        return ZoneInfo(self.timezone)


class EffectiveSchedule(BaseModel):
    """One stored schedule and whether its clock job is enabled."""

    model_config = ConfigDict(frozen=True)

    schedule: DatasetSyncSchedule
    enabled: bool = True

    @property
    def dataset_id(self) -> str:
        return self.schedule.dataset_id

    @property
    def schedule_id(self) -> str:
        return self.schedule.schedule_id

    @property
    def effective(self) -> bool:
        """Whether this is the entry the clock would run: enabled.

        Whether it actually runs also depends on the scheduler being enabled in this process
        and on the dataset resolving; ``ScheduleStatus.registered`` reports that.
        """
        return self.enabled


def effective_schedules(stored: Iterable[StoredSchedule]) -> list[EffectiveSchedule]:
    """Return the single store's schedules in dataset-id order."""
    effective: list[EffectiveSchedule] = []
    for item in sorted(stored, key=lambda entry: entry.dataset_id):
        effective.append(
            EffectiveSchedule(
                schedule=DatasetSyncSchedule(
                    dataset_id=item.dataset_id, cron=item.cron, publish=item.publish, max_attempts=item.max_attempts
                ),
                enabled=item.enabled,
            )
        )
    return effective


def get_scheduler_config() -> SchedulerConfig:
    """Load scheduler configuration from the instance configuration."""
    raw = api_config.get_config().get("scheduler", {})
    if not isinstance(raw, dict):
        raise ValueError("scheduler in CLIMATE_SERVICE_CONFIG must be a mapping")
    if "dataset_sync" in raw:
        raise ValueError(
            "scheduler.dataset_sync is no longer supported; run 'climate-service migrate-schedules' "
            "to copy entries to <data_dir>/schedules.json, then remove dataset_sync from climate-service.yaml"
        )
    return SchedulerConfig.model_validate(raw)
