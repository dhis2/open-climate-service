"""Pydantic schemas and execution primitives for native jobs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DATASET_UPDATED_EVENT_TYPE = "dataset.updated"
"""Event type persisted when a sync actually appends or rematerializes a dataset."""


class JobCancelledError(Exception):
    """Raised by a job implementation when cooperative cancellation is honored."""


class JobStatus(StrEnum):
    """Persisted lifecycle states for native jobs."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    """Native job categories."""

    PROCESS = "process"


class JobLink(BaseModel):
    """Hypermedia link for a job resource."""

    href: str
    rel: str
    title: str | None = None


class JobProgress(BaseModel):
    """Progress details reported by a running job."""

    done: int | None = None
    total: int | None = None
    percent: float | None = None
    message: str | None = None


class JobError(BaseModel):
    """Serialized error details for a failed job."""

    type: str
    message: str


class JobEventDraft(BaseModel):
    """Domain event returned by a job before completion metadata is assigned."""

    type: str
    source: str
    data: dict[str, Any] = Field(default_factory=dict)


class JobExecutionResult(BaseModel):
    """A callable result accompanied by domain events created on success."""

    result: Any = None
    events: list[JobEventDraft] = Field(default_factory=list)


class JobEvent(JobEventDraft):
    """Domain event persisted atomically with a successful native job."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(validation_alias="eventID", serialization_alias="eventID")
    time: datetime


class JobRecord(BaseModel):
    """Persisted native job record."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(validation_alias="jobID", serialization_alias="jobID")
    process_id: str = Field(validation_alias="processID", serialization_alias="processID")
    type: JobType = JobType.PROCESS
    status: JobStatus
    created_at: datetime = Field(validation_alias="createdAt", serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, validation_alias="startedAt", serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, validation_alias="finishedAt", serialization_alias="finishedAt")
    attempt: int = 0
    max_attempts: int = Field(default=1, validation_alias="maxAttempts", serialization_alias="maxAttempts")
    executor_kind: str = Field(default="thread", validation_alias="executorKind", serialization_alias="executorKind")
    request: dict[str, Any] = Field(default_factory=dict)
    progress: JobProgress = Field(default_factory=JobProgress)
    result: Any | None = None
    events: list[JobEvent] = Field(default_factory=list)
    error: JobError | None = None
    cancel_requested: bool = Field(
        default=False,
        validation_alias="cancelRequested",
        serialization_alias="cancelRequested",
    )
    retry_after: int | None = Field(
        default=None,
        validation_alias="retryAfter",
        serialization_alias="retryAfter",
    )
    cursor: dict[str, Any] | None = None
    links: list[JobLink] = Field(default_factory=list)


class JobListResponse(BaseModel):
    """Envelope response for native jobs."""

    jobs: list[JobRecord] = Field(default_factory=list)
    links: list[JobLink] = Field(default_factory=list)
