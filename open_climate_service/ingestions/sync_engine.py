"""Sync engine for managed datasets.

This module owns the two sync responsibilities:

- `plan_sync(...)` decides what EO API should do for one managed dataset
- `run_sync(...)` applies that plan through the existing artifact creation flow

Keeping both operations here gives the route/service layer a stable entry point
and keeps future scheduler-driven sync jobs on the same code path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from open_climate_service import config as api_config
from open_climate_service.ingestions.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    SyncAction,
    SyncDetail,
    SyncKind,
    SyncResponse,
)
from open_climate_service.publications.services import managed_dataset_id_for
from open_climate_service.shared.time import (
    datetime_to_period_string,
    dekad_bounds,
    dekad_start,
    normalize_period_string,
    parse_hourly_period_string,
    parse_period_string_to_datetime,
    utc_now,
    utc_today,
)
from open_climate_service.streaming.store import read_committed_period_ids

logger = logging.getLogger(__name__)


class SyncConfigurationError(RuntimeError):
    """Raised when server-side sync configuration is invalid."""


def plan_sync(
    *,
    source_dataset: dict[str, Any],
    latest_artifact: ArtifactRecord,
    requested_end: str | None,
) -> SyncDetail:
    """Return the sync decision for one managed dataset without changing local state.

    The planner reads the latest materialized coverage plus the dataset template's
    declared `sync_kind` and produces the action EO API should take next.

    Current first-pass behavior:

    - temporal datasets compare the latest local period against the requested end
    - release datasets compare the current materialized release against the requested end
    - static datasets are marked as not syncable

    This planner deliberately does not download data or persist artifacts.
    """
    sync_kind_value = source_dataset.get("sync", {}).get("kind")
    if not isinstance(sync_kind_value, str) or not sync_kind_value:
        raise ValueError("source_dataset must define sync.kind for sync planning")
    sync_kind = SyncKind(sync_kind_value)
    current_start = latest_artifact.request_scope.start

    if sync_kind == SyncKind.STATIC:
        # Static datasets are never synced. Report NOT_SYNCABLE directly, reading the
        # coverage end from the artifact (None for a non-temporal dataset such as a
        # climatology) — don't call _sync_current_end, which requires a temporal end.
        static_end = latest_artifact.coverage.temporal.end
        return SyncDetail(
            source_dataset_id=latest_artifact.dataset_id,
            sync_kind=sync_kind,
            action=SyncAction.NOT_SYNCABLE,
            reason="static_dataset",
            message="This dataset is static and is not syncable.",
            current_start=current_start,
            current_end=static_end,
            target_end=static_end,
            target_end_source="current_coverage",
        )

    current_end = _sync_current_end(source_dataset=source_dataset, latest_artifact=latest_artifact)
    period_type = str(source_dataset["period_type"])
    normalized_requested_end = requested_end.strip() if isinstance(requested_end, str) else None
    normalized_requested_end = normalized_requested_end or None
    target_end_source = "request" if normalized_requested_end is not None else "default_today"
    if normalized_requested_end is not None:
        target_end = normalize_period_string(normalized_requested_end, period_type)
    else:
        target_end = _default_target_end(period_type=period_type)

    if sync_kind == SyncKind.TEMPORAL:
        next_period_start = _next_period_start(current_end, period_type=period_type)
        if next_period_start > target_end:
            return SyncDetail(
                source_dataset_id=latest_artifact.dataset_id,
                sync_kind=sync_kind,
                action=SyncAction.NO_OP,
                reason="no_new_period",
                message=(
                    f"Data already exists through {current_end}; target {target_end} does not require a new download."
                ),
                current_start=current_start,
                current_end=current_end,
                target_end=target_end,
                target_end_source=target_end_source,
            )
        # Enumerate the plugin's actually-available periods before committing to a sync.
        # The full list is stored on the SyncDetail so the orchestrator can reuse it
        # without re-querying at execution time.
        available_periods = _query_available_periods(source_dataset, next_period_start, target_end)
        # available_periods is None  → no plugin, use target_end as-is
        # available_periods is []    → plugin exists but nothing available → NO_OP
        # available_periods is [...]  → clamp target_end to the last available period
        if available_periods == []:
            return SyncDetail(
                source_dataset_id=latest_artifact.dataset_id,
                sync_kind=sync_kind,
                action=SyncAction.NO_OP,
                reason="no_new_period",
                message=f"Data already exists through {current_end}; no new periods available from the source.",
                current_start=current_start,
                current_end=current_end,
                target_end=target_end,
                target_end_source=target_end_source,
            )
        if available_periods is not None and available_periods[-1] < target_end:
            target_end = available_periods[-1]
            target_end_source = "plugin_availability"
        action = SyncAction.APPEND if _supports_append(source_dataset, latest_artifact) else SyncAction.REMATERIALIZE
        reason = "new_periods_available_for_append" if action == SyncAction.APPEND else "new_periods_available"
        # Only pass periods for APPEND — it covers [next_period_start, target_end].
        # For REMATERIALIZE the orchestrator must enumerate the full [current_start, target_end]
        # range so we pass None and let it call plugin.periods() with the correct bounds.
        periods_for_orchestrator = available_periods if action == SyncAction.APPEND else None
        return SyncDetail(
            source_dataset_id=latest_artifact.dataset_id,
            sync_kind=sync_kind,
            action=action,
            reason=reason,
            message=_sync_plan_message(
                action=action,
                current_end=current_end,
                target_end=target_end,
                delta_start=next_period_start,
                delta_end=target_end,
            ),
            current_start=current_start,
            current_end=current_end,
            target_end=target_end,
            target_end_source=target_end_source,
            delta_start=next_period_start,
            delta_end=target_end,
            periods=periods_for_orchestrator,
        )

    # Clamp target_end to what the plugin reports as actually available so we
    # never plan a rematerialization into future/unpublished releases.
    available_periods = _query_available_periods(source_dataset, current_end, target_end)
    if available_periods is not None:
        if not available_periods:
            return SyncDetail(
                source_dataset_id=latest_artifact.dataset_id,
                sync_kind=sync_kind,
                action=SyncAction.NO_OP,
                reason="no_new_release",
                message=f"No new release available beyond {current_end}.",
                current_start=current_start,
                current_end=current_end,
                target_end=target_end,
                target_end_source=target_end_source,
            )
        target_end = available_periods[-1]
        target_end_source = "plugin_availability"

    if current_end >= target_end:
        return SyncDetail(
            source_dataset_id=latest_artifact.dataset_id,
            sync_kind=sync_kind,
            action=SyncAction.NO_OP,
            reason="no_new_release",
            message=(
                f"Release {current_end} is already available locally; target {target_end} "
                "does not require a new download."
            ),
            current_start=current_start,
            current_end=current_end,
            target_end=target_end,
            target_end_source=target_end_source,
        )

    return SyncDetail(
        source_dataset_id=latest_artifact.dataset_id,
        sync_kind=sync_kind,
        action=SyncAction.REMATERIALIZE,
        reason="new_release_available",
        message=f"A newer release is available: {target_end}. Sync will rematerialize the dataset.",
        current_start=current_start,
        current_end=current_end,
        target_end=target_end,
        target_end_source=target_end_source,
    )


def run_sync(
    *,
    latest_artifact: ArtifactRecord,
    source_dataset: dict[str, Any],
    requested_end: str | None,
    country_code: str | None,
    publish: bool,
    create_artifact_fn: Callable[..., ArtifactRecord],
    get_dataset_fn: Callable[[str], Any],
    on_progress: Callable[..., Any] | None = None,
) -> SyncResponse:
    """Plan and execute one sync operation for a managed dataset.

    `run_sync(...)` is the engine entry point that non-HTTP callers can reuse
    later, for example scheduled jobs. It keeps all sync outcomes on the same
    response model:

    - `up_to_date` when no new upstream state is planned
    - `not_syncable` for templates that should not be synced
    - `completed` when EO API rematerializes a fresh backing artifact
    """
    sync_detail = plan_sync(
        source_dataset=source_dataset,
        latest_artifact=latest_artifact,
        requested_end=requested_end,
    )
    dataset_id = managed_dataset_id_for(latest_artifact)
    logger.info(
        "Sync plan for dataset '%s': action=%s reason=%s current=%s..%s target=%s delta=%s..%s",
        dataset_id,
        sync_detail.action,
        sync_detail.reason,
        sync_detail.current_start,
        sync_detail.current_end,
        sync_detail.target_end,
        sync_detail.delta_start,
        sync_detail.delta_end,
    )

    if sync_detail.action == SyncAction.NO_OP:
        logger.info("Sync skipped for dataset '%s': already current", dataset_id)
        return SyncResponse(
            sync_id=None,
            status="up_to_date",
            message="Managed dataset is already current with upstream source state.",
            dataset=get_dataset_fn(dataset_id),
            sync_detail=sync_detail,
        )

    if sync_detail.action == SyncAction.NOT_SYNCABLE:
        logger.info("Sync skipped for dataset '%s': dataset is not syncable", dataset_id)
        return SyncResponse(
            sync_id=None,
            status="not_syncable",
            message="Managed dataset is not syncable under its configured sync policy.",
            dataset=get_dataset_fn(dataset_id),
            sync_detail=sync_detail,
        )

    # Execution always goes through the ingestion materialization path so sync
    # does not grow a second write/storage implementation. Temporal APPEND
    # extends the committed store; release-style REMATERIALIZE rewrites the
    # managed artifact against the latest planned upstream state.
    if sync_detail.current_start is None:
        raise ValueError("Sync execution requires current_start for rematerialize or append actions")
    if sync_detail.target_end is None:
        raise ValueError("Sync execution requires target_end for rematerialize or append actions")
    download_start = sync_detail.delta_start if sync_detail.action == SyncAction.APPEND else None
    logger.info(
        "Sync executing for dataset '%s': action=%s artifact_scope=%s..%s download_scope=%s..%s publish=%s",
        dataset_id,
        sync_detail.action,
        sync_detail.current_start,
        sync_detail.target_end,
        download_start or sync_detail.current_start,
        sync_detail.delta_end if download_start is not None else sync_detail.target_end,
        publish,
    )
    artifact = create_artifact_fn(
        dataset=source_dataset,
        start=sync_detail.current_start,
        end=sync_detail.target_end,
        download_start=download_start,
        download_end=sync_detail.delta_end if download_start is not None else None,
        bbox=list(latest_artifact.request_scope.bbox) if latest_artifact.request_scope.bbox is not None else None,
        country_code=country_code,
        overwrite=False,
        publish=publish,
        on_progress=on_progress,
        periods=sync_detail.periods,
    )
    logger.info(
        "Sync completed for dataset '%s': artifact_id=%s action=%s",
        dataset_id,
        artifact.artifact_id,
        sync_detail.action,
    )
    return SyncResponse(
        sync_id=artifact.artifact_id,
        status="completed",
        message=_sync_completed_message(sync_detail.action),
        dataset=get_dataset_fn(managed_dataset_id_for(artifact)),
        sync_detail=sync_detail,
    )


def _sync_completed_message(action: SyncAction) -> str:
    """Return a user-facing completion message for the executed sync action."""
    if action == SyncAction.APPEND:
        return "Managed dataset was synced by appending missing periods to the committed store."
    return "Managed dataset was rematerialized against the latest planned upstream state."


def _sync_plan_message(
    *,
    action: SyncAction,
    current_end: str,
    target_end: str,
    delta_start: str,
    delta_end: str,
) -> str:
    """Return a human-readable sync plan summary."""
    if action == SyncAction.APPEND:
        return (
            f"Data exists through {current_end}. Sync will append missing periods "
            f"{delta_start} through {delta_end} and extend coverage through {target_end}."
        )
    return f"Data exists through {current_end}. Sync will rematerialize the dataset through {target_end}."


def _query_available_periods(
    source_dataset: dict[str, Any],
    start: str,
    target_end: str,
) -> list[str] | None:
    """Ask the plugin which periods are actually available.

    Returns the full period list (possibly empty) so the result can be reused
    at execution time, avoiding a second query in the orchestrator.

    Returns None only when the dataset has no plugin (non-streaming) — the
    caller falls back to the requested target_end without clamping.

    Returns [] when the plugin exists but has no periods for the requested
    range — the caller should treat this as NO_OP.
    """
    from open_climate_service.shared.plugin_loader import instantiate_plugin

    plugin_path = source_dataset.get("ingestion", {}).get("plugin")
    if not plugin_path:
        return None
    params: dict[str, Any] = source_dataset.get("ingestion", {}).get("params") or {}
    plugin = instantiate_plugin(plugin_path, params)
    return asyncio.run(plugin.periods(start, target_end))


def _next_period_start(latest_period_end: str, *, period_type: str) -> str:
    """Return the next dataset-native period after a covered temporal end value.

    This helper is part of sync planning because temporal datasets need to know
    whether another period could exist beyond the current materialized coverage.
    """
    if period_type == "hourly":
        timestamp = parse_hourly_period_string(latest_period_end)
        return datetime_to_period_string(timestamp + timedelta(hours=1), period_type)
    if period_type == "daily":
        current = date.fromisoformat(latest_period_end)
        return (current + timedelta(days=1)).isoformat()
    if period_type == "dekadal":
        # Step by the covered dekad's own length rather than a fixed 10 days, since the
        # third dekad of a month runs 8-11 days.
        _, dekad_end = dekad_bounds(latest_period_end)
        return (dekad_end + timedelta(days=1)).isoformat()
    if period_type == "weekly":
        current = parse_period_string_to_datetime(latest_period_end).date()
        next_week = datetime.combine(current + timedelta(days=7), time(0))
        return datetime_to_period_string(next_week, period_type)
    if period_type == "monthly":
        current = date.fromisoformat(f"{latest_period_end}-01")
        month = current.month + 1
        year = current.year + (1 if month == 13 else 0)
        month = 1 if month == 13 else month
        return f"{year:04d}-{month:02d}"
    if period_type == "yearly":
        return str(int(latest_period_end) + 1)
    raise ValueError(f"Unsupported period_type '{period_type}' for sync")


def _default_target_end(*, period_type: str) -> str:
    """Return the default sync target in the dataset-native period format."""
    today = utc_today()
    if period_type == "hourly":
        return datetime_to_period_string(utc_now(), period_type)
    if period_type == "daily":
        return today.isoformat()
    if period_type == "dekadal":
        return dekad_start(today).isoformat()
    if period_type == "weekly":
        return datetime_to_period_string(utc_now(), period_type)
    if period_type == "monthly":
        return f"{today.year:04d}-{today.month:02d}"
    if period_type == "yearly":
        return str(today.year)
    raise ValueError(f"Unsupported period_type '{period_type}' for sync")


def _icechunk_path_for(artifact: ArtifactRecord) -> str | None:
    """Return the Icechunk store path for a plugin-backed artifact."""
    if artifact.format != ArtifactFormat.ICECHUNK:
        return None
    return artifact.path or (artifact.asset_paths[0] if artifact.asset_paths else None)


def _supports_append(source_dataset: dict[str, Any], latest_artifact: ArtifactRecord) -> bool:
    """Return whether this template opts into store-based append sync execution."""
    if source_dataset.get("sync", {}).get("execution") != SyncAction.APPEND.value:
        return False
    if latest_artifact.format == ArtifactFormat.ICECHUNK:
        return True
    logger.info(
        "Sync append execution for dataset '%s' requires an existing Icechunk artifact; falling back to rematerialize",
        source_dataset.get("id", "<unknown>"),
    )
    return False


def _is_plugin_backed(source_dataset: dict[str, Any]) -> bool:
    """Return whether the source dataset uses the streaming plugin contract."""
    ingestion = source_dataset.get("ingestion")
    if not isinstance(ingestion, dict):
        return False
    plugin = ingestion.get("plugin")
    return isinstance(plugin, str) and bool(plugin)


def _artifact_storage_roots() -> tuple[Path, ...]:
    """Return the trusted local roots that may contain managed artifact stores."""
    return (api_config.get_download_root().resolve(),)


def _resolve_local_artifact_path(raw_path: str | None) -> tuple[Path | None, str | None]:
    """Return a trusted local artifact path plus a fallback reason when unavailable."""
    if raw_path is None:
        return None, None
    # urlparse misreads Windows drive letters as URI schemes — "C:/path" produces
    # scheme="c". Detect drive-letter paths explicitly so they skip URI parsing
    # and reach the shared absolute-path and trusted-root checks below.
    is_windows_path = len(raw_path) >= 3 and raw_path[0].isalpha() and raw_path[1] == ":" and raw_path[2] in ("\\", "/")
    if is_windows_path:
        candidate = Path(raw_path)
    else:
        parsed = urlparse(raw_path)
        if parsed.scheme and parsed.scheme != "file":
            return None, "non-local URI"
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                return None, "non-local file URI"
            candidate = Path(unquote(parsed.path))
        else:
            candidate = Path(raw_path)
    if not candidate.is_absolute():
        return None, "relative path"
    resolved = candidate.resolve(strict=False)
    for root in _artifact_storage_roots():
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return resolved, None
    return None, "untrusted local path"


def _sync_current_end(*, source_dataset: dict[str, Any], latest_artifact: ArtifactRecord) -> str:
    """Return the current sync end using the same source of truth as execution.

    For plugin-backed Icechunk datasets, sync execution consults committed store
    periods directly. Planning should do the same so it does not depend on
    potentially stale artifact metadata after prior interrupted or external
    updates.

    This adds a small read cost to plan-only requests for plugin-backed
    datasets, but keeps planning aligned with execution on the committed store
    state rather than a cached metadata snapshot.
    """
    end = latest_artifact.coverage.temporal.end
    if end is None:
        # Sync planning only applies to temporal datasets; a non-temporal (ordinal)
        # dataset such as a day-of-year climatology has no temporal end to advance from.
        raise ValueError(
            f"Dataset '{source_dataset.get('id', '<unknown>')}' has no temporal coverage and cannot be synced"
        )
    if not _is_plugin_backed(source_dataset):
        return end
    raw_artifact_path = _icechunk_path_for(latest_artifact)
    if raw_artifact_path is None:
        return end
    local_artifact_path, path_reason = _resolve_local_artifact_path(raw_artifact_path)
    if local_artifact_path is None:
        logger.warning(
            "Sync planning skipped committed-store inspection for dataset '%s' because artifact path is %s: %s",
            source_dataset.get("id", "<unknown>"),
            path_reason or "unsupported",
            raw_artifact_path,
        )
        return end
    try:
        committed = read_committed_period_ids(
            local_artifact_path,
            str(source_dataset["period_type"]),
        )
    except Exception as exc:
        logger.warning(
            "Sync planning failed to read committed periods for dataset '%s' from '%s'; "
            "falling back to artifact metadata: %s",
            source_dataset.get("id", "<unknown>"),
            str(local_artifact_path),
            exc,
        )
        return end
    if not committed:
        return end
    try:
        return normalize_period_string(
            max(committed, key=parse_period_string_to_datetime),
            str(source_dataset["period_type"]),
        )
    except Exception as exc:
        logger.warning(
            "Sync planning found malformed committed periods for dataset '%s' in '%s'; "
            "falling back to artifact metadata: %s",
            source_dataset.get("id", "<unknown>"),
            str(local_artifact_path),
            exc,
        )
        return end
