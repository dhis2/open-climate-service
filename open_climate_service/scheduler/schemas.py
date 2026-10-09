"""Public status schemas for scheduled dataset synchronization."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from open_climate_service.scheduler.dispatcher import CheckOutcome


class ScheduleStatus(BaseModel):
    """Configuration and volatile runtime status for one dataset schedule."""

    schedule_id: str
    kind: Literal["sync"] = Field(default="sync", description="What the schedule runs; only dataset sync so far.")
    dataset_id: str
    cron: str
    timezone: str
    publish: bool
    max_attempts: int
    source: Literal["store"] = "store"
    enabled: bool = True
    effective: bool = Field(default=True, description="Enabled: the entry the clock would run.")
    registered: bool = Field(default=False, description="Whether a clock job exists for it in this process now.")
    next_check: datetime | None = None
    last_check: datetime | None = None
    last_outcome: CheckOutcome | None = None
    last_message: str | None = None
    last_job_id: str | None = None


class ScheduleListResponse(BaseModel):
    """Status of the process-local scheduler and configured schedules."""

    enabled: bool
    running: bool
    timezone: str
    clock_holder: str | None = Field(
        default=None,
        description="The process running the clock, when it is this one. Others stand by and take over if it stops.",
    )
    reload_error: str | None = Field(
        default=None,
        description="Why the last reload was refused; the previous working schedules stay in force until it is fixed.",
    )
    schedules: list[ScheduleStatus] = Field(default_factory=list)
