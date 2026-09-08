"""Structured outcome of one export delivery attempt."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExportOutcome(StrEnum):
    """Distinct terminal states a delivery attempt can reach."""

    SUCCESS = "success"
    """The target accepted the payload and reported no conflicts or errors."""

    PARTIAL = "partial"
    """The target reported a mixed import summary with conflicts or failures."""

    REJECTED = "rejected"
    """The target rejected the payload outright (validation or import errors)."""

    DRY_RUN = "dry_run"
    """A dry-run validation completed; nothing was written."""

    CANCELLED = "cancelled"
    """Delivery was cancelled before or during submission."""

    UNKNOWN = "unknown"
    """The remote outcome could not be reconciled (e.g. timeout after POST)."""


class ExportReport(BaseModel):
    """Authoritative summary of a delivery attempt, persisted on the delivery job."""

    model_config = ConfigDict(extra="forbid")

    plugin_id: str
    connection_id: str | None = None
    dry_run: bool = False
    outcome: ExportOutcome
    message: str | None = None
    payload_sha256: str
    attempts: int = 1
    chunks: int = 1
    submitted: int = 0
    imported: int = 0
    updated: int = 0
    ignored: int = 0
    deleted: int = 0
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    remote_task_ids: list[str] = Field(default_factory=list)
    created_at: str
    finished_at: str
