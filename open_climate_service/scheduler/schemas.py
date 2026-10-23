"""Public status schemas for scheduled dataset synchronization."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from open_climate_service.scheduler.dispatcher import CheckOutcome


class ScheduleStatus(BaseModel):
    """Configuration, next check and last check for one dataset schedule."""

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


class ClockStatus(BaseModel):
    """A clock that runs on this data directory, as it last reported.

    Which process and host it is stays in the shared file: OCS needs it to recognise its own
    heartbeat, a client does not, and a public instance should not publish it.
    """

    here: bool = Field(description="Whether the clock runs in the process answering.")
    beat_at: datetime = Field(description="When it last reported that it runs.")


class ScheduleListResponse(BaseModel):
    """Status of the scheduler and configured schedules.

    ``enabled`` and ``running`` describe the answering process only. ``clock`` says whether a
    clock runs anywhere on the shared data directory, so a process that does not run it can
    tell "scheduling runs elsewhere" from "scheduling is off".
    """

    enabled: bool = Field(description="Whether this process's configuration turns the clock on.")
    running: bool = Field(description="Whether the clock runs in this process.")
    clock: ClockStatus | None = Field(
        default=None,
        description="The process running the clock; None when no process has reported one recently.",
    )
    timezone: str
    reload_error: str | None = Field(
        default=None,
        description="Why the last reload was refused; the previous working schedules stay in force until it is fixed.",
    )
    schedules: list[ScheduleStatus] = Field(default_factory=list)
