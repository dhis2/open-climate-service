"""Public status schemas for scheduled dataset synchronization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from open_climate_service.scheduler.dispatcher import CheckOutcome


class ScheduleCreate(BaseModel):
    """A new sync schedule for a managed dataset; one per dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, description="The managed dataset to keep current.")
    cron: str = Field(min_length=1, description="Five-field cron expression, in the instance's timezone.")
    publish: bool = Field(default=True, description="Publish each new version the sync makes.")
    max_attempts: int = Field(default=3, ge=1, description="Attempts per scheduled check, the first included.")
    enabled: bool = Field(default=True, description="False saves the schedule paused.")


def _without_null(schema: dict[str, Any]) -> None:
    """Advertise each property as its own type: a client leaves a setting out, it never sends null."""
    for prop in schema.get("properties", {}).values():
        options = prop.pop("anyOf", None)
        if options:
            prop.update(next(option for option in options if option.get("type") != "null"))
        if "default" in prop and prop["default"] is None:
            del prop["default"]


class ScheduleChange(BaseModel):
    """A change to a stored sync schedule; settings left out are kept, and none may be null.

    Only the fields a client sets are read, so the ``None`` defaults stand for "left out".
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_without_null)

    cron: str | None = Field(default=None, min_length=1)
    publish: bool | None = None
    max_attempts: int | None = Field(default=None, ge=1)
    enabled: bool | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _not_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("leave the setting out to keep it; it cannot be null")
        return value


class ScheduleLinks(BaseModel):
    """Where to follow a schedule: itself, its dataset and the sync it last submitted."""

    self: str
    dataset: str
    last_job: str | None = Field(default=None, description="The last submitted sync job; poll it for its status.")


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


class ScheduleView(ScheduleStatus):
    """A schedule as clients see it: its status with links into this deployment."""

    links: ScheduleLinks


class ClockStatus(BaseModel):
    """A clock that runs on this data directory, as it last reported.

    Which process and host it is stays in the shared file: OCS needs it to recognise its own
    heartbeat, a client does not, and a public instance should not publish it.
    """

    here: bool = Field(description="Whether the clock runs in the process answering.")
    beat_at: datetime = Field(description="When it last reported that it runs.")


class SchedulerState(BaseModel):
    """Status of the scheduler.

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


class ScheduleListResponse(SchedulerState):
    """The scheduler's status and its schedules, as the scheduler reports them."""

    schedules: list[ScheduleStatus] = Field(default_factory=list)


class ScheduleListView(SchedulerState):
    """The scheduler's status and its schedules, each with its links."""

    schedules: list[ScheduleView] = Field(default_factory=list)
