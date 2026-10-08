"""Pydantic models for the openEO API surface."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# openEO capabilities
# ---------------------------------------------------------------------------


class OpenEOEndpoint(BaseModel):
    """One declared endpoint in the openEO capabilities document."""

    path: str
    methods: list[str]


class OpenEOCapabilities(BaseModel):
    """openEO capabilities document returned by GET /."""

    api_version: str
    backend_version: str
    stac_version: str
    id: str
    title: str
    description: str
    production: bool
    endpoints: list[OpenEOEndpoint]
    links: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# openEO jobs
# ---------------------------------------------------------------------------


class OpenEOJobStatus(StrEnum):
    """openEO-compliant job lifecycle states."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    CANCELED = "canceled"


class OpenEOJobRecord(BaseModel):
    """Persisted openEO job record."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str | None = None
    description: str | None = None
    process: dict[str, Any] = Field(default_factory=dict)
    status: OpenEOJobStatus
    created: datetime
    updated: datetime | None = None
    plan: str | None = None
    budget: float | None = None
    usage: dict[str, Any] | None = None
    logs: str | None = None
    links: list[dict[str, Any]] = Field(default_factory=list)

    # Internal fields not exposed in openEO responses
    _error_message: str | None = None
    _cancel_requested: bool = False

    # Persisted internal state (underscore prefix excluded from openEO output)
    error_message: str | None = Field(default=None, exclude=True)
    cancel_requested: bool = Field(default=False, exclude=True)
    # Set on jobs created by a workflow trigger; automated delivery keys off them.
    trigger_id: str | None = Field(default=None, exclude=True)
    source_event_id: str | None = Field(default=None, exclude=True)
    # When the job last reached FINISHED. `updated` moves on every later mutation, so it
    # cannot tell whether a job finished before a delivery activation boundary.
    finished_at: datetime | None = Field(default=None, exclude=True)
    # The delivery this job owed when it finished ({"export", "mode"}), written with FINISHED.
    delivery_due: dict[str, str] | None = Field(default=None, exclude=True)
    # Execution attempts under this one job ID. Triggered jobs retry up to `max_attempts`;
    # `retry_at` is when a job waiting out a backoff becomes due again.
    attempt: int = Field(default=0, exclude=True)
    max_attempts: int = Field(default=1, exclude=True)
    retry_at: datetime | None = Field(default=None, exclude=True)
    # Set once the running attempt has passed its point of no return (a managed store commit):
    # from then on the job finishes rather than being cancelled half-published.
    publishing: bool = Field(default=False, exclude=True)


class OpenEOJobCreate(BaseModel):
    """Request body for POST /jobs."""

    process: dict[str, Any]
    title: str | None = None
    description: str | None = None
    plan: str | None = None
    budget: float | None = None


class OpenEOJobUpdate(BaseModel):
    """Request body for PATCH /jobs/{job_id}."""

    title: str | None = None
    description: str | None = None
    process: dict[str, Any] | None = None
    plan: str | None = None
    budget: float | None = None


class OpenEOJobListResponse(BaseModel):
    """Envelope for GET /jobs."""

    jobs: list[OpenEOJobRecord]
    links: list[dict[str, Any]] = Field(default_factory=list)


class OpenEOJobResultLink(BaseModel):
    """One asset link in GET /jobs/{id}/results."""

    href: str
    type: str | None = None
    title: str | None = None
    roles: list[str] = Field(default_factory=list)


class OpenEOJobResults(BaseModel):
    """Response for GET /jobs/{id}/results."""

    type: str = "Feature"
    stac_version: str
    stac_extensions: list[str] = Field(default_factory=list)
    id: str
    geometry: dict[str, Any] | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, Any] = Field(default_factory=dict)
    links: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# openEO workflows (user-defined processes)
# ---------------------------------------------------------------------------


class WorkflowRecord(BaseModel):
    """Persisted workflow (user-defined process)."""

    id: str
    summary: str | None = None
    description: str | None = None
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    returns: dict[str, Any] | None = None
    process_graph: dict[str, Any] = Field(default_factory=dict)
    links: list[dict[str, Any]] = Field(default_factory=list)


class WorkflowListResponse(BaseModel):
    """Envelope for GET /process_graphs."""

    processes: list[WorkflowRecord]
    links: list[dict[str, Any]] = Field(default_factory=list)
