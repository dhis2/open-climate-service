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


# Ordered by severity; a merged report takes the first matching outcome.
_OUTCOME_PRECEDENCE: tuple[ExportOutcome, ...] = (
    ExportOutcome.REJECTED,
    ExportOutcome.UNKNOWN,
    ExportOutcome.CANCELLED,
    ExportOutcome.PARTIAL,
)


def merge_chunk_reports(
    reports: list[ExportReport],
    *,
    plugin_id: str,
    connection_id: str | None,
    dry_run: bool,
    payload_sha256: str,
    created_at: str,
    finished_at: str,
    submitted: int | None = None,
    cancelled_early: bool = False,
    message: str | None = None,
) -> ExportReport:
    """Combine per-chunk reports into one authoritative delivery report.

    Counts, conflicts, remote task IDs, and attempts are summed. The aggregate
    outcome is the most severe chunk outcome; a dry run that fully validated
    reports ``dry_run``, otherwise a clean run reports ``success``. An early
    cancellation is recorded as ``cancelled`` unless a chunk was rejected or its
    outcome is unknown.
    """
    outcomes: set[ExportOutcome]
    if reports:
        attempts = sum(report.attempts for report in reports)
        imported = sum(report.imported for report in reports)
        updated = sum(report.updated for report in reports)
        ignored = sum(report.ignored for report in reports)
        deleted = sum(report.deleted for report in reports)
        conflicts = [conflict for report in reports for conflict in report.conflicts]
        remote_task_ids = [task_id for report in reports for task_id in report.remote_task_ids]
        chunks = len(reports)
        total_submitted = submitted if submitted is not None else sum(report.submitted for report in reports)
        outcomes = {report.outcome for report in reports}
    else:
        attempts = 0
        imported = updated = ignored = deleted = 0
        conflicts = []
        remote_task_ids = []
        chunks = 0
        total_submitted = submitted if submitted is not None else 0
        outcomes = set()

    outcome = ExportOutcome.SUCCESS
    for candidate in _OUTCOME_PRECEDENCE:
        if candidate in outcomes or (candidate == ExportOutcome.CANCELLED and cancelled_early):
            outcome = candidate
            break
    if outcome == ExportOutcome.SUCCESS:
        outcome = ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS

    return ExportReport(
        plugin_id=plugin_id,
        connection_id=connection_id,
        dry_run=dry_run,
        outcome=outcome,
        message=message,
        payload_sha256=payload_sha256,
        attempts=attempts,
        chunks=chunks,
        submitted=total_submitted,
        imported=imported,
        updated=updated,
        ignored=ignored,
        deleted=deleted,
        conflicts=conflicts,
        remote_task_ids=remote_task_ids,
        created_at=created_at,
        finished_at=finished_at,
    )
