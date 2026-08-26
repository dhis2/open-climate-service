"""Services for ingestion, dataset persistence, and publication metadata."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import portalocker
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response
from zarr.core.buffer import default_buffer_prototype

from open_climate_service import config as api_config
from open_climate_service.data_accessor.services.accessor import get_data_coverage_for_paths
from open_climate_service.data_manager.services import downloader
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents.services import get_extent
from open_climate_service.ingestions.artifact_paths import decode_record_paths, encode_record_paths
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactListResponse,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    DatasetAccessLink,
    DatasetDetailRecord,
    DatasetListResponse,
    DatasetPublication,
    DatasetRecord,
    DatasetVersionRecord,
    IngestionListResponse,
    IngestionResponse,
    PublicationStatus,
    SyncAction,
    SyncDetail,
    SyncResponse,
)
from open_climate_service.ingestions.sync_engine import SyncConfigurationError, plan_sync, run_sync
from open_climate_service.publications.services import managed_dataset_id_for, publish_artifact
from open_climate_service.shared.time import (
    datetime_to_period_string,
    dekad_start,
    next_period_string,
    normalize_period_string,
    utc_now,
    utc_today,
)
from open_climate_service.streaming.orchestrator import run_streaming_ingest_sync
from open_climate_service.streaming.protocol import IngestionPlugin
from open_climate_service.streaming.store import (
    open_or_create_repo,
    read_committed_period_ids_ordered,
)

logger = logging.getLogger(__name__)

# Per-store threading locks prevent two concurrent ingest/sync runs from writing
# to the same Icechunk store simultaneously (which causes MVCC commit conflicts).
_store_locks: dict[str, threading.Lock] = {}
_store_locks_mutex = threading.Lock()

# Consolidated zarr metadata cache: (store_path, snapshot_id) → metadata dict.
# Building consolidated metadata requires reading every zarr.json from the store;
# the result is stable for the lifetime of a snapshot, so we cache it keyed to
# the Icechunk branch tip.  Capped at 512 entries; the oldest half is evicted when
# the limit is hit (one entry per ingest per dataset, so 512 entries ≈ 85 ingests
# across 6 datasets before any eviction occurs).
_consolidated_metadata_cache: dict[str, dict[str, object]] = {}
_MAX_CONSOLIDATED_CACHE_ENTRIES = 512

# Icechunk artifact path cache: dataset_id → (records_mtime, artifact).
# Avoids reading and deserializing the full artifact index on every chunk request
# from the /icechunk/ endpoint.  Invalidated when records.json changes on disk.
_icechunk_artifact_cache: dict[str, tuple[float, "ArtifactRecord"]] = {}


@dataclass(frozen=True)
class _StreamingMaterializationPlan:
    """Write shape selected before a streaming ingest mutates its store."""

    action: SyncAction
    start: str
    end: str
    periods: list[str] | None
    has_committed_periods: bool


@dataclass(frozen=True)
class _StoreNormalizationResult:
    """Outcome of normalizing a store, including whether its directory was swapped."""

    completed: bool
    swapped: bool = False


def _acquire_store_lock(store_path: Path) -> threading.Lock:
    """Return the exclusive lock for store_path, creating it if needed."""
    key = str(store_path.resolve())
    with _store_locks_mutex:
        if key not in _store_locks:
            _store_locks[key] = threading.Lock()
        return _store_locks[key]


class _IcechunkReadableStore(Protocol):
    def list_dir(self, prefix: str) -> AsyncIterator[str]: ...
    def exists(self, key: str) -> Awaitable[bool]: ...
    def get(self, key: str, prototype: Any) -> Awaitable[Any]: ...


class _IcechunkSession:
    """Thin holder for a readonly Icechunk session and its branch-tip snapshot id."""

    __slots__ = ("store", "snapshot_id", "store_path")

    def __init__(self, store: _IcechunkReadableStore, snapshot_id: str, store_path: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id
        self.store_path = store_path

    @property
    def cache_key(self) -> str:
        return f"{self.store_path}:{self.snapshot_id}"


def _resolve_artifacts_dir() -> Path:
    return api_config.get_data_root() / "artifacts"


ARTIFACTS_DIR = _resolve_artifacts_dir()
ARTIFACTS_INDEX_PATH = ARTIFACTS_DIR / "records.json"


def ensure_store() -> None:
    """Create the artifact metadata store if it does not exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ARTIFACTS_INDEX_PATH.exists():
        ARTIFACTS_INDEX_PATH.write_text("[]\n", encoding="utf-8")


def list_artifacts() -> ArtifactListResponse:
    """Return all stored artifacts."""
    return ArtifactListResponse(items=_materialized_records(_load_records()))


def group_datasets() -> dict[str, list[ArtifactRecord]]:
    """Return artifact records grouped by stable managed dataset id."""
    grouped: dict[str, list[ArtifactRecord]] = {}
    for record in list_artifacts().items:
        grouped.setdefault(managed_dataset_id_for(record), []).append(record)
    return grouped


def latest_published_zarr_artifacts_by_dataset() -> dict[str, ArtifactRecord]:
    """Return the latest published Zarr/Icechunk artifact for each dataset id."""
    result: dict[str, ArtifactRecord] = {}
    for dataset_id, artifacts in group_datasets().items():
        latest = max(artifacts, key=lambda artifact: artifact.created_at)
        if latest.publication.status != PublicationStatus.PUBLISHED:
            continue
        if latest.format != ArtifactFormat.ICECHUNK:
            continue
        result[dataset_id] = latest
    return dict(sorted(result.items()))


def list_ingestions() -> IngestionListResponse:
    """Return ingestion run records for operational/admin use."""
    records = sorted(_materialized_records(_load_records()), key=lambda record: record.created_at, reverse=True)
    items = [_build_ingestion_response(record) for record in records]
    return IngestionListResponse(items=items)


def get_artifact_or_404(artifact_id: str) -> ArtifactRecord:
    """Return a single artifact or raise 404."""
    for record in _materialized_records(_load_records()):
        if record.artifact_id == artifact_id:
            return record
    raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")


def get_dataset_for_artifact_or_404(artifact_id: str) -> DatasetDetailRecord:
    """Return the managed dataset view corresponding to an internal artifact id."""
    artifact = get_artifact_or_404(artifact_id)
    return get_dataset_or_404(managed_dataset_id_for(artifact))


def get_dataset_summary_for_artifact_or_404(artifact_id: str) -> DatasetRecord:
    """Return the managed dataset summary corresponding to an internal artifact id."""
    artifact = get_artifact_or_404(artifact_id)
    dataset_id = managed_dataset_id_for(artifact)
    artifacts = group_datasets().get(dataset_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return _build_dataset_record(dataset_id, artifacts)


def get_ingestion_or_404(artifact_id: str) -> IngestionResponse:
    """Return an ingestion run record resolved to the managed dataset summary."""
    return _build_ingestion_response(get_artifact_or_404(artifact_id))


def list_datasets() -> DatasetListResponse:
    """Return managed datasets grouped by stable dataset id."""
    grouped = group_datasets()
    items = [_build_dataset_record(dataset_id, artifacts) for dataset_id, artifacts in sorted(grouped.items())]
    return DatasetListResponse(items=items)


def get_dataset_or_404(dataset_id: str) -> DatasetDetailRecord:
    """Return one managed dataset or raise 404."""
    grouped = group_datasets()
    artifacts = grouped.get(dataset_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return _build_dataset_detail_record(dataset_id, artifacts)


def get_latest_artifact_for_dataset_or_404(dataset_id: str) -> ArtifactRecord:
    """Return the latest artifact backing a managed dataset."""
    grouped = group_datasets()
    artifacts = grouped.get(dataset_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return max(artifacts, key=lambda artifact: artifact.created_at)


def create_artifact(
    *,
    dataset: dict[str, object],
    start: str | None,
    end: str | None,
    bbox: list[float] | None,
    country_code: str | None,
    overwrite: bool,
    publish: bool,
    download_start: str | None = None,
    download_end: str | None = None,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, object]], None] | None = None,
    periods: list[str] | None = None,
) -> ArtifactRecord:
    """Materialize one managed dataset artifact and persist its metadata.

    Source dataset materialization is plugin-backed and always writes an
    Icechunk store. Sync requests may still pass `download_start` and
    `download_end`, but those only describe the requested append window. The
    streaming engine remains store-authoritative and appends only periods that
    are actually missing from the committed store.
    """
    period_type = str(dataset["period_type"])
    start = _resolve_request_start(start, dataset=dataset, period_type=period_type)
    end = _normalize_optional_request_period(end, period_type=period_type, field_name="end")
    download_start = _normalize_optional_request_period(
        download_start, period_type=period_type, field_name="download_start"
    )
    download_end = _normalize_optional_request_period(download_end, period_type=period_type, field_name="download_end")
    _validate_download_scope(
        start=start,
        end=end,
        download_start=download_start,
        download_end=download_end,
    )
    resolved_download_end = download_end if download_end is not None else end
    if resolved_download_end is None:
        if registry_datasets.is_future_facing(dataset):
            # "Now" is this dataset's *start*, so using it as the end would collapse a
            # seven-day forecast to a single day. Offer a generous horizon instead and let
            # the plugin clip to whatever it actually publishes. `request_scope.end` stays
            # None, so the coverage check does not hold the plugin to this wider bound.
            resolved_download_end = _forecast_horizon(dataset, period_type)
        else:
            resolved_download_end = _current_request_period(period_type)
    request_scope = ArtifactRequestScope(
        start=start,
        end=end,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]) if bbox is not None else None,
    )
    ingestion = dataset.get("ingestion")
    plugin_path = ingestion.get("plugin") if isinstance(ingestion, dict) else None
    if isinstance(plugin_path, str) and plugin_path:
        return _create_streaming_artifact(
            dataset=dataset,
            plugin_path=plugin_path,
            start=start,
            end=resolved_download_end,
            bbox=bbox,
            country_code=country_code,
            overwrite=overwrite,
            publish=publish,
            request_scope=request_scope,
            on_progress=on_progress,
            is_cancel_requested=is_cancel_requested,
            save_cursor=save_cursor,
            periods=periods,
        )
    raise HTTPException(status_code=500, detail=f"Dataset '{dataset['id']}' does not define ingestion.plugin")


def _period_order_key(period: str, period_type: str) -> str:
    """Return a lexically sortable key for one normalized period id."""
    normalized = normalize_period_string(period, period_type)
    if period_type == "climatology":
        try:
            return f"{int(normalized):03d}"
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Source returned invalid climatology period '{period}'",
            ) from exc
    return normalized


def _normalize_ordered_periods(
    periods: list[str], *, period_type: str, source: str, require_ordered: bool = True
) -> list[str]:
    """Normalize periods and optionally require a unique, ascending sequence."""
    try:
        normalized = [normalize_period_string(period, period_type) for period in periods]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"{source} returned an invalid period sequence: {exc}") from exc
    keys = [_period_order_key(period, period_type) for period in normalized]
    if require_ordered and (len(set(normalized)) != len(normalized) or keys != sorted(keys)):
        raise HTTPException(
            status_code=409,
            detail=f"{source} periods must be unique and in ascending order; refusing to mutate the store",
        )
    return normalized


def _periods_are_contiguous(periods: list[str], period_type: str) -> bool:
    """Return whether each period immediately follows the previous one."""
    return all(next_period_string(previous, period_type) == current for previous, current in zip(periods, periods[1:]))


def _validate_source_periods(
    periods: list[str],
    *,
    start: str,
    end: str,
    period_type: str,
    scope: str,
) -> list[str]:
    """Normalize and validate one source-provided materialization sequence."""
    available = _normalize_ordered_periods(periods, period_type=period_type, source="Ingestion plugin")
    if not available:
        raise HTTPException(status_code=409, detail=f"Source has no data for the requested {scope}")
    if available[0] != start:
        raise HTTPException(
            status_code=409,
            detail=f"Source cannot materialize the requested {scope} from {start}; first available is {available[0]}",
        )
    if _period_order_key(available[-1], period_type) > _period_order_key(end, period_type):
        raise HTTPException(
            status_code=409,
            detail=f"Source returned period {available[-1]} beyond the requested {scope} ending {end}",
        )
    if not _periods_are_contiguous(available, period_type):
        raise HTTPException(
            status_code=409,
            detail=f"Source returned a non-contiguous period sequence for the requested {scope}",
        )
    return available


def _plan_streaming_materialization(
    *,
    plugin: IngestionPlugin,
    store_path: Path,
    start: str,
    end: str,
    period_type: str,
    overwrite: bool,
    periods: list[str] | None,
) -> _StreamingMaterializationPlan:
    """Plan a contiguous union before allowing streaming ingest to write.

    A forward request can append only when the committed coordinate is an exact
    prefix of every source-valid period in the union. Earlier requests, gaps, or
    non-monotonic committed coordinates require a sibling-store rematerialization.
    """
    committed = _normalize_ordered_periods(
        read_committed_period_ids_ordered(
            store_path,
            period_type,
            time_dim=str(getattr(plugin, "time_dim", "t")),
        ),
        period_type=period_type,
        source="Committed store",
        require_ordered=False,
    )
    if overwrite:
        available = _validate_source_periods(
            periods if periods is not None else asyncio.run(plugin.periods(start, end)),
            start=start,
            end=end,
            period_type=period_type,
            scope="temporal scope",
        )
        return _StreamingMaterializationPlan(
            action=SyncAction.REMATERIALIZE,
            start=start,
            end=available[-1],
            periods=available,
            has_committed_periods=bool(committed),
        )

    if not committed:
        available = _validate_source_periods(
            periods if periods is not None else asyncio.run(plugin.periods(start, end)),
            start=start,
            end=end,
            period_type=period_type,
            scope="temporal scope",
        )
        return _StreamingMaterializationPlan(
            action=SyncAction.APPEND,
            start=start,
            end=available[-1],
            periods=available,
            has_committed_periods=False,
        )

    current_start = min(committed, key=lambda value: _period_order_key(value, period_type))
    current_end = max(committed, key=lambda value: _period_order_key(value, period_type))
    materialization_start = min((current_start, start), key=lambda value: _period_order_key(value, period_type))
    materialization_end = max((current_end, end), key=lambda value: _period_order_key(value, period_type))
    committed_is_contiguous = _periods_are_contiguous(committed, period_type)

    # A request wholly contained by an already contiguous store cannot add data.
    # Reuse the current artifact without asking a rolling source to enumerate
    # historical periods that it may no longer expose.
    if materialization_start == current_start and materialization_end == current_end and committed_is_contiguous:
        return _StreamingMaterializationPlan(
            action=SyncAction.NO_OP,
            start=current_start,
            end=current_end,
            periods=committed,
            has_committed_periods=True,
        )

    # A contiguous store that only needs a forward extension requires the source
    # to enumerate the missing delta, not reproduce committed history. Reuse a
    # sync planner's prefetched delta when present; direct ingestion queries it here.
    if materialization_start == current_start and committed_is_contiguous:
        expected_start = next_period_string(current_end, period_type)
        source_periods = _normalize_ordered_periods(
            periods if periods is not None else asyncio.run(plugin.periods(expected_start, materialization_end)),
            period_type=period_type,
            source="Ingestion plugin",
        )
        # Plugins should honor the requested bounds, but accepting an already
        # committed leading prefix keeps broader source enumerators compatible.
        delta = [
            period
            for period in source_periods
            if _period_order_key(period, period_type) >= _period_order_key(expected_start, period_type)
        ]
        if not delta:
            return _StreamingMaterializationPlan(
                action=SyncAction.NO_OP,
                start=current_start,
                end=current_end,
                periods=committed,
                has_committed_periods=True,
            )
        if delta[0] != expected_start or not _periods_are_contiguous(delta, period_type):
            raise HTTPException(
                status_code=409,
                detail=(f"Source append periods must form a contiguous sequence beginning at {expected_start}"),
            )
        if _period_order_key(delta[-1], period_type) > _period_order_key(materialization_end, period_type):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Source returned period {delta[-1]} beyond the requested temporal union ending "
                    f"{materialization_end}"
                ),
            )
        return _StreamingMaterializationPlan(
            action=SyncAction.APPEND,
            start=current_start,
            end=delta[-1],
            periods=[*committed, *delta],
            has_committed_periods=True,
        )

    available = _validate_source_periods(
        asyncio.run(plugin.periods(materialization_start, materialization_end)),
        start=materialization_start,
        end=materialization_end,
        period_type=period_type,
        scope="temporal union",
    )
    available_set = set(available)
    missing_committed = [period for period in committed if period not in available_set]
    if missing_committed:
        raise HTTPException(
            status_code=409,
            detail=(
                "Source can no longer reproduce committed period(s) required for the temporal union: "
                + ", ".join(missing_committed[:5])
            ),
        )

    committed_is_prefix = available[: len(committed)] == committed
    action = (
        SyncAction.APPEND
        if committed_is_prefix and materialization_start == current_start
        else SyncAction.REMATERIALIZE
    )
    return _StreamingMaterializationPlan(
        action=action,
        start=materialization_start,
        end=available[-1],
        periods=available,
        has_committed_periods=True,
    )


def _create_streaming_artifact(
    *,
    dataset: dict[str, object],
    plugin_path: str,
    start: str,
    end: str,
    bbox: list[float] | None,
    country_code: str | None,
    overwrite: bool,
    publish: bool,
    request_scope: ArtifactRequestScope,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, object]], None] | None = None,
    periods: list[str] | None = None,
) -> ArtifactRecord:
    """Create or update one plugin-backed Icechunk artifact.

    The same helper is used for both initial ingest and store-based sync. The
    streaming orchestrator enumerates the plugin's periods for the full requested
    range, then appends only periods that are not already committed in the target
    Icechunk-backed store.
    """
    if bbox is None:
        raise HTTPException(status_code=400, detail="Streaming ingest requires a bounding box")

    ingestion = dataset.get("ingestion")
    raw_params = ingestion.get("params") if isinstance(ingestion, dict) else None
    if raw_params is None:
        params: dict[str, object] = {}
    elif isinstance(raw_params, dict):
        params = dict(raw_params)
    else:
        raise HTTPException(status_code=500, detail="ingestion.params must be an object")
    if country_code is not None:
        params["country_code"] = country_code

    plugin = _load_streaming_plugin(plugin_path, params=params)
    store_path = downloader.get_icechunk_path(dataset)

    lock = _acquire_store_lock(store_path)
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail=f"An ingest or sync is already running for dataset '{dataset['id']}'. Wait for it to finish.",
        )
    replacement_path: Path | None = None
    rollback_repo: Any | None = None
    rollback_branch: str | None = None
    rollback_snapshot: str | None = None
    published_swap_pending = False
    store_committed = False
    plugin_handed_to_orchestrator = False
    try:
        # First thing under the lock, before anything looks at the store. A swap killed between
        # its two renames leaves the published path missing and the data at `.retired`; ingest
        # would read that as a brand-new store and write only the requested delta into a fresh
        # one, after which recovery finds the target present and strands the whole history in
        # `.retired`. Healing before any inspection makes the next sync an ordinary append.
        recover_interrupted_swap(store_path)

        plan = _plan_streaming_materialization(
            plugin=plugin,
            store_path=store_path,
            start=start,
            end=end,
            period_type=str(dataset["period_type"]),
            overwrite=overwrite,
            periods=periods,
        )
        if plan.action == SyncAction.NO_OP:
            existing = get_latest_artifact_for_dataset_or_404(str(dataset["id"]))
            if publish and existing.publication.status != PublicationStatus.PUBLISHED:
                return publish_artifact_record(existing.artifact_id)
            return existing
        materialization_scope = request_scope.model_copy(update={"start": plan.start, "end": plan.end})

        ingest_path = store_path
        if plan.action == SyncAction.REMATERIALIZE and store_path.exists():
            # Build a prepend/overwrite beside the published store. Fetching is the least
            # reliable part of ingestion, so the current data remains readable until the
            # contiguous replacement has been fetched, validated, and normalized.
            replacement_path = store_path.with_name(f"{store_path.name}.replacement")
            _remove_store_path(replacement_path)
            ingest_path = replacement_path
        elif plan.has_committed_periods:
            # Icechunk commits each fetched period independently. Keep the pre-ingest
            # snapshot reachable so an exception after any commit can restore the public
            # branch instead of leaving a partial append or a stale pyramid behind.
            repo = open_or_create_repo(store_path)
            snapshot = repo.lookup_branch("main")
            branch = f"ocs-ingest-rollback-{uuid4().hex}"
            repo.create_branch(branch, snapshot)
            rollback_repo = repo
            rollback_snapshot = snapshot
            rollback_branch = branch

        plugin_handed_to_orchestrator = True
        result = run_streaming_ingest_sync(
            plugin=plugin,
            params=params,
            dataset=dataset,
            bbox=bbox,
            start=plan.start,
            end=plan.end,
            store_path=ingest_path,
            period_type=str(dataset["period_type"]),
            on_progress=on_progress,
            is_cancel_requested=is_cancel_requested,
            save_cursor=save_cursor,
            periods=plan.periods,
        )
        if result.periods_written == 0 and not ingest_path.exists():
            raise HTTPException(status_code=409, detail="Source has no data for the requested temporal scope")

        coverage_data = get_data_coverage_for_paths(dataset, icechunk_path=str(ingest_path.resolve()))
        if not coverage_data.get("has_data", True):
            raise HTTPException(
                status_code=409,
                detail="Materialized artifact contains no data for the requested scope",
            )

        _spatial_wgs84_data = coverage_data["coverage"].get("spatial_wgs84")
        coverage = ArtifactCoverage(
            temporal=CoverageTemporal(**coverage_data["coverage"]["temporal"]),
            spatial=CoverageSpatial(**coverage_data["coverage"]["spatial"]),
            spatial_wgs84=CoverageSpatial(**_spatial_wgs84_data) if _spatial_wgs84_data else None,
        )
        # Temporal datasets validate against the cumulative materialization scope, not
        # the raw user request. The latter remains on the record as operation provenance.
        # A brand-new store may legitimately clamp its end to source availability; an
        # update was already planned from the source's exact available period sequence.
        if coverage.temporal.start is not None:
            coverage_matches_plan = (
                _temporal_coverage_matches_request_scope(coverage.temporal, materialization_scope)
                if plan.has_committed_periods
                else _temporal_coverage_matches_streaming_request_scope(coverage.temporal, materialization_scope)
            )
            if not coverage_matches_plan:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Materialized artifact coverage does not match the planned contiguous scope: "
                        f"coverage={coverage.temporal.start}..{coverage.temporal.end}, "
                        f"plan={materialization_scope.start}..{materialization_scope.end}"
                    ),
                )

        normalization = _maybe_build_pyramid(
            ingest_path,
            dataset,
            retain_previous=plan.has_committed_periods and replacement_path is None,
        )
        if plan.has_committed_periods and not normalization.completed:
            raise RuntimeError(f"Could not normalize '{dataset['id']}'; the dataset update was rolled back")
        if normalization.swapped and ingest_path == store_path:
            # Pyramid normalization replaced the published repository but retained
            # the previous one until its matching artifact record is durable. From
            # here, rollback first restores that repository and then resets its main
            # branch to the pre-ingest snapshot retained above.
            published_swap_pending = True
        if replacement_path is not None:
            _swap_store(replacement_path, store_path, retain_previous=True)
            published_swap_pending = True

        record = ArtifactRecord(
            artifact_id=str(uuid4()),
            dataset_id=str(dataset["id"]),
            dataset_name=str(dataset["name"]),
            variable=str(dataset["variable"]),
            period_type=str(dataset.get("period_type")) if dataset.get("period_type") is not None else None,
            format=ArtifactFormat.ICECHUNK,
            path=str(store_path.resolve()),
            asset_paths=[str(store_path.resolve())],
            variables=[str(dataset["variable"])],
            request_scope=request_scope,
            coverage=coverage,
            created_at=datetime.now(UTC),
            publication=ArtifactPublication(),
        )
        stored_record = _upsert_artifact_record(
            record,
            publish=publish,
            overwrite=overwrite,
        )
        store_committed = True
        if published_swap_pending:
            try:
                _finalize_store_swap(store_path)
                published_swap_pending = False
            except Exception:
                # The record and replacement are already durable. Retaining the old
                # directory is only a cleanup leak; restoring it would make the record
                # point at stale data. A later run may retry the cleanup.
                logger.warning("Could not remove retired store '%s' after commit", store_path, exc_info=True)
            # A swapped-out repository either disappeared with successful cleanup or
            # remains only as a retired fallback. Never try to operate on its temporary
            # branch through the newly published repository path.
            rollback_repo = None
            rollback_branch = None
            rollback_snapshot = None
        if publish and stored_record.publication.status != PublicationStatus.PUBLISHED:
            return publish_artifact_record(stored_record.artifact_id)
        return stored_record
    finally:
        try:
            if published_swap_pending and not store_committed:
                try:
                    _rollback_store_swap(store_path)
                    published_swap_pending = False
                except Exception:
                    logger.error(
                        "Could not restore the previous store for '%s' after artifact registration failed",
                        store_path,
                        exc_info=True,
                    )
            elif published_swap_pending:
                try:
                    _finalize_store_swap(store_path)
                    published_swap_pending = False
                except Exception:
                    logger.warning("Could not clean up retired store '%s'", store_path, exc_info=True)
            if not plugin_handed_to_orchestrator:
                close_plugin = getattr(plugin, "close", None)
                if callable(close_plugin):
                    try:
                        close_plugin()
                    except Exception:
                        logger.warning("Could not close ingestion plugin after planning failure", exc_info=True)
            if rollback_repo is not None and rollback_branch is not None and rollback_snapshot is not None:
                try:
                    if not store_committed:
                        rollback_repo.reset_branch("main", rollback_snapshot)
                    rollback_repo.delete_branch(rollback_branch)
                except Exception:
                    logger.error(
                        "Could not %s append transaction for '%s'",
                        "roll back" if not store_committed else "clean up",
                        store_path,
                        exc_info=True,
                    )
            if replacement_path is not None:
                # Failed fetches and validations leave only a disposable partial replacement. A
                # successful swap has already moved this path away, making cleanup a no-op.
                try:
                    _remove_store_path(replacement_path)
                except Exception:
                    # Cleanup is best-effort: the published store is still intact and the next
                    # overwrite removes this path before reuse. Do not mask the ingest failure.
                    logger.warning("Could not remove replacement store '%s'", replacement_path, exc_info=True)
        finally:
            # Releasing the in-process writer lock must not depend on filesystem cleanup.
            lock.release()


def _retired_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.retired")


def _remove_store_path(path: Path) -> None:
    """Remove a disposable store path whether it is a directory, file, or symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def recover_interrupted_swap(target: Path) -> bool:
    """Restore a store left behind by a swap that was killed between its two renames.

    ``_swap_store`` moves the published store aside before moving the rebuilt one in. An
    exception between the two is handled there, but a SIGKILL or a host restart is not: the
    published path then does not exist and the data sits at ``<name>.retired``, which no
    reader looks for. Called before the store is opened, so the next sync heals it rather
    than reporting an unreadable dataset.

    Returns True when a recovery was performed.
    """
    retired = _retired_path(target)
    if target.exists() or not retired.is_dir():
        return False
    retired.rename(target)
    logger.warning(
        "Recovered '%s' from '%s': a previous store swap was interrupted between its two "
        "renames, leaving the published path missing.",
        target.name,
        retired.name,
    )
    return True


def _swap_store(staging: Path, target: Path, *, retain_previous: bool = False) -> None:
    """Move ``staging`` into ``target``'s place, keeping the original until the swap lands.

    Two directory renames on one filesystem rather than a copy, so the cost is metadata
    rather than bytes. Two caveats, both real:

    * **The target does not exist between the renames.** A reader that opens the store in that
      window fails. The window is two metadata operations wide, but it is not zero.
    * **Only an exception is rolled back here.** A process kill or host restart between the
      renames leaves the published path missing, which :func:`recover_interrupted_swap`
      repairs on the next sync.

    Neither is fixable by reordering: POSIX has no atomic directory exchange that Python
    exposes portably. The durable answer is to publish through a pointer that can be switched
    atomically, which is CLIM-880 — this keeps the window small and recoverable meanwhile.
    """
    retired = _retired_path(target)
    shutil.rmtree(retired, ignore_errors=True)
    target.rename(retired)
    try:
        staging.rename(target)
    except Exception:
        retired.rename(target)  # leave the store exactly as it was
        raise
    if not retain_previous:
        shutil.rmtree(retired, ignore_errors=True)


def _finalize_store_swap(target: Path) -> None:
    """Discard the previous store after its replacement record is durable."""
    _remove_store_path(_retired_path(target))


def _rollback_store_swap(target: Path) -> None:
    """Restore the retained store when replacement metadata cannot be persisted."""
    retired = _retired_path(target)
    if not retired.exists():
        return
    failed = target.with_name(f"{target.name}.failed")
    _remove_store_path(failed)
    target.rename(failed)
    try:
        retired.rename(target)
    except Exception:
        failed.rename(target)
        raise
    _remove_store_path(failed)


def _maybe_build_pyramid(
    store_path: Path,
    dataset: dict[str, object],
    *,
    retain_previous: bool = False,
) -> _StoreNormalizationResult:
    """Apply GeoZarr conventions and a multiscale pyramid to the committed Icechunk store.

    Streaming ingest writes a flat store with root GeoZarr attrs but no
    ``spatial_ref`` CRS coordinate or pyramid levels. This rewrites the store via
    ``write_to_icechunk_store`` to add them, so every Icechunk store follows the
    same GeoZarr layout regardless of size.

    ``write_to_icechunk_store`` would overwrite the very store it is reading from, so the
    rewrite goes to a sibling store which is then swapped in. That keeps the source readable
    while topozarr streams the pyramid out of it, which is what lets the build stay lazy
    instead of materialising the whole store in RAM.

    A *temporal append* to an already-normalised flat store produces nothing new, though:
    ``spatial_ref`` is a scalar that survives the append and streaming refreshes the root
    attrs on every commit. We detect that case and skip the read-rewrite entirely, avoiding
    the write amplification of re-emitting the whole store on every sync.

    Returns whether normalization completed and whether it swapped the store.
    Errors remain logged and swallowed so a brand-new plain flat artifact can
    still be registered; callers updating an existing store use the result to
    roll back instead.
    """
    from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

    # Belt and braces: `_create_streaming_artifact` already heals this under the lock before
    # ingest, which is the call that matters. Kept because this function is also entered
    # directly, and because it is idempotent — a present target makes it a no-op.
    recover_interrupted_swap(store_path)

    try:
        ds = open_icechunk_dataset(store_path)
    except Exception:
        logger.warning("Could not open Icechunk store for GeoZarr write; skipping", exc_info=True)
        return _StoreNormalizationResult(completed=False)

    try:
        already_normalized = "spatial_ref" in ds.coords or "spatial_ref" in ds.variables
        if already_normalized and "x" in ds.sizes and "y" in ds.sizes and not downloader.needs_pyramid(ds):
            # A flat, already-normalized store needs no read-rewrite — but streaming ingests
            # append one period at a time, leaving the time coordinate chunked at size 1.
            # That makes a map client fetch one tiny chunk per timestep to read the axis
            # (thousands of requests for a multi-year daily store), so re-chunk just the
            # coordinate here (cheap; data variables untouched).
            t_dim = next((d for d in ("t", "time", "valid_time") if d in ds.sizes), None)
            if t_dim is not None:
                try:
                    downloader.ensure_time_coordinate_chunking(store_path, t_dim)
                except Exception:
                    logger.warning("Time-coordinate re-chunk failed for '%s'", store_path.name, exc_info=True)
            logger.info(
                "Store '%s' is already GeoZarr-normalized and flat; skipping read-rewrite",
                store_path.name,
            )
            return _StoreNormalizationResult(completed=True)
        # `ds` reads lazily from `store_path`, and the rewrite overwrites that same store —
        # the aliasing is why this used to materialise everything first. Build into a sibling
        # store and swap it in, so the source stays readable while topozarr streams the
        # pyramid out of it. The cost moves from RAM (the whole store: 3.47 GB for Uganda's
        # GPP) to transient disk (one extra copy, reclaimed on success).
        staging = store_path.with_name(f"{store_path.name}.rebuild")
        shutil.rmtree(staging, ignore_errors=True)
        try:
            downloader.write_to_icechunk_store(
                ds,
                staging,
                pyramid_method=downloader.resampling_method_from_template(dataset),
                commit_message="Applied GeoZarr conventions",
            )
            ds.close()  # drop the read session before the directory moves under it
            _swap_store(staging, store_path, retain_previous=retain_previous)
        finally:
            # A failed build leaves a partial copy of the whole store behind. That matters most
            # when the failure *was* disk exhaustion: the flat-store fallback below would
            # otherwise run with the transient copy still occupying the space. A successful
            # swap has already moved `staging` away, so this is a no-op then.
            shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        logger.error(
            "GeoZarr/pyramid write failed for '%s'; flat Icechunk artifact will be used as-is",
            store_path.name,
            exc_info=True,
        )
        return _StoreNormalizationResult(completed=False)
    finally:
        ds.close()
    return _StoreNormalizationResult(completed=True, swapped=True)


def _load_streaming_plugin(plugin_path: str, *, params: dict[str, object]) -> IngestionPlugin:
    """Load and instantiate one streaming plugin class from a dotted import path.

    Template-defined `ingestion.params` are treated as plugin
    configuration and passed to the constructor here. The same params are also
    forwarded later to `fetch_period(...)` so plugins may keep configuration in
    constructor state, per-call kwargs, or both.
    """
    from open_climate_service.shared.plugin_loader import instantiate_plugin

    module_path, _, attr_name = plugin_path.rpartition(".")
    if not module_path or not attr_name:
        raise HTTPException(status_code=500, detail=f"Invalid ingestion.plugin path '{plugin_path}'")
    try:
        plugin = instantiate_plugin(plugin_path, dict(params))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load ingestion.plugin '%s'", plugin_path, exc_info=exc)
        raise HTTPException(status_code=500, detail=f"Failed to load ingestion.plugin '{plugin_path}'") from exc
    return plugin


def register_artifact_record(record: ArtifactRecord, *, publish: bool) -> ArtifactRecord:
    """Store a pre-built artifact record and optionally publish it to STAC.

    Uses overwrite semantics: re-publishing a managed dataset (e.g. re-running an
    openEO ``save_result`` for the same ``dataset_id``) replaces the existing
    record's metadata — name, coverage, paths — rather than silently keeping the
    stale record, while preserving its artifact id and publication state.
    """
    stored = _upsert_artifact_record(record, publish=publish, overwrite=True)
    if publish and stored.publication.status != PublicationStatus.PUBLISHED:
        return publish_artifact_record(stored.artifact_id)
    return stored


def publish_artifact_record(artifact_id: str) -> ArtifactRecord:
    """Publish an artifact and persist publication metadata."""
    published = publish_artifact(get_artifact_or_404(artifact_id))

    def mutate(records: list[ArtifactRecord]) -> ArtifactRecord:
        for index, record in enumerate(records):
            if record.artifact_id != artifact_id:
                continue
            records[index] = published
            return published
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")

    return _mutate_records(mutate)


def sync_dataset(
    *,
    dataset_id: str,
    end: str | None,
    publish: bool,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
) -> SyncResponse:
    """Resolve sync inputs and delegate managed-dataset sync to the sync engine.

    The service layer stays thin on purpose: it validates that the requested
    public dataset id resolves to a managed dataset plus a source template, then
    hands execution to `sync_engine.run_sync(...)`.
    """
    latest_artifact = get_latest_artifact_for_dataset_or_404(dataset_id)
    source_dataset = registry_datasets.get_dataset(latest_artifact.dataset_id)
    if source_dataset is None:
        raise HTTPException(status_code=404, detail=f"Source dataset '{latest_artifact.dataset_id}' not found")
    extent = get_extent()
    resolved_country_code = extent.get("country_code") if extent else None
    try:
        return run_sync(
            latest_artifact=latest_artifact,
            source_dataset=source_dataset,
            requested_end=end,
            country_code=resolved_country_code,
            publish=publish,
            create_artifact_fn=create_artifact,
            get_dataset_fn=get_dataset_or_404,
            on_progress=on_progress,
        )
    except SyncConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def plan_sync_dataset(
    *,
    dataset_id: str,
    end: str | None,
) -> SyncDetail:
    """Return the sync plan for a managed dataset without downloading or writing artifacts."""
    latest_artifact = get_latest_artifact_for_dataset_or_404(dataset_id)
    source_dataset = registry_datasets.get_dataset(latest_artifact.dataset_id)
    if source_dataset is None:
        raise HTTPException(status_code=404, detail=f"Source dataset '{latest_artifact.dataset_id}' not found")
    try:
        return plan_sync(
            latest_artifact=latest_artifact,
            source_dataset=source_dataset,
            requested_end=end,
        )
    except SyncConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_dataset_zarr_store_file_or_404(
    dataset_id: str, relative_path: str, range_header: str | None = None
) -> Response | dict[str, object]:
    """Serve a file, metadata document, or directory listing within a dataset Zarr store."""
    artifact = get_latest_artifact_for_dataset_or_404(dataset_id)
    session = _open_icechunk_store_or_404(artifact)
    return _get_icechunk_store_path_or_404(
        dataset_id=dataset_id, session=session, relative_path=relative_path, range_header=range_header
    )


def serve_icechunk_file(dataset_id: str, file_path: str) -> FileResponse:
    """Serve a raw Icechunk store file for native SDK access via icechunk.http_storage()."""
    artifact = _get_latest_published_icechunk_artifact_cached(dataset_id)
    store_root = Path(artifact.path or artifact.asset_paths[0]).resolve()

    full_path = (store_root / file_path).resolve()
    if not full_path.is_relative_to(store_root):
        raise HTTPException(status_code=404, detail="Not found")
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"Icechunk store file not found: {file_path}")

    return FileResponse(full_path, media_type="application/octet-stream")


def _get_latest_published_icechunk_artifact_cached(dataset_id: str) -> ArtifactRecord:
    """Return the latest published Icechunk artifact, cached by records.json mtime."""
    try:
        mtime = ARTIFACTS_INDEX_PATH.stat().st_mtime if ARTIFACTS_INDEX_PATH.exists() else 0.0
    except OSError:
        mtime = 0.0
    cached = _icechunk_artifact_cache.get(dataset_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    artifact = _get_latest_published_icechunk_artifact(dataset_id)
    _icechunk_artifact_cache[dataset_id] = (mtime, artifact)
    return artifact


def _get_latest_published_icechunk_artifact(dataset_id: str) -> ArtifactRecord:
    """Return the latest published Icechunk artifact for dataset_id, or raise 404/409."""
    artifacts = latest_published_zarr_artifacts_by_dataset()
    artifact = artifacts.get(dataset_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    if artifact.format != ArtifactFormat.ICECHUNK:
        raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' is not Icechunk-backed")
    return artifact


def _open_icechunk_store_or_404(artifact: ArtifactRecord) -> _IcechunkSession:
    """Open the readonly Icechunk session store for a published artifact."""
    if artifact.format != ArtifactFormat.ICECHUNK:
        raise HTTPException(status_code=409, detail="Artifact is not an Icechunk-backed Zarr store")
    store_path = artifact.path or (artifact.asset_paths[0] if artifact.asset_paths else None)
    if store_path is None:
        raise HTTPException(status_code=409, detail="Artifact has no resolvable store path")
    path = Path(store_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Icechunk store path does not exist on disk")

    import icechunk

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open(storage)
    snapshot_id = repo.lookup_branch("main")
    session_store = cast(_IcechunkReadableStore, repo.readonly_session("main").store)
    return _IcechunkSession(store=session_store, snapshot_id=snapshot_id, store_path=str(path))


def _run_async(awaitable: Any) -> Any:
    """Run an async Icechunk store operation from sync route/service code.

    Most calls arrive from sync FastAPI handlers with no running event loop.
    If a loop is already running in this thread, run the awaitable in a short-
    lived worker thread instead of nesting event loops in-process.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(awaitable)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            error["value"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    if "value" not in result:
        raise RuntimeError("async runner completed without a result or error")
    return result["value"]


def _icechunk_list_dir(store: _IcechunkReadableStore, prefix: str) -> list[str]:
    async def collect() -> list[str]:
        items: list[str] = []
        async for item in store.list_dir(prefix):
            items.append(item)
        return items

    return cast(list[str], _run_async(collect()))


def _icechunk_exists(store: _IcechunkReadableStore, key: str) -> bool:
    """Whether *key* exists in the store; False for anything Icechunk cannot address.

    A browser probing an unknown store asks for keys that are not Zarr keys at all, and Icechunk
    rejects those two different ways: ``KeyError`` for Zarr v2 spellings (``.zgroup``,
    ``.zarray``), and ``IcechunkError: invalid zarr key format`` for paths outside the Zarr
    namespace entirely (``repo``, ``refs/branch.main/ref.json`` — the layout probes zarr-viewer
    uses to decide whether a URL is an Icechunk repository).

    Both mean "not here", so both must answer False. Letting the second escape turned a probe
    into a 500, and because an unhandled 500 is raised above the CORS middleware it reached the
    browser with no ``Access-Control-Allow-Origin`` — reported as a CORS failure, which is not
    where anyone would look for a missing-key bug.
    """
    from icechunk import IcechunkError

    try:
        return cast(bool, _run_async(store.exists(key)))
    except (KeyError, IcechunkError):
        return False


def _icechunk_get(store: _IcechunkReadableStore, key: str) -> bytes | None:
    buffer: Any = _run_async(store.get(key, prototype=default_buffer_prototype()))
    if buffer is None:
        return None
    if hasattr(buffer, "to_bytes"):
        return cast(bytes, buffer.to_bytes())
    return cast(bytes, buffer)


def _build_icechunk_consolidated_metadata(session: _IcechunkSession) -> dict[str, object]:
    """Recursively build zarr v3 consolidated metadata for all groups and arrays in the store.

    Icechunk explicitly blocks ``zarr.consolidate_metadata()`` because its transactional
    snapshot system makes a static .zmetadata incompatible with multi-writer safety.
    We generate equivalent consolidated metadata dynamically at serve time so that HTTP
    clients can open the store with a single request (``consolidated=True``) without
    needing server-side directory enumeration.

    Results are cached per (store_path, snapshot_id) so repeated requests within the
    lifetime of a snapshot — which is the common case between ingests — read once and
    serve many times.
    """
    cached = _consolidated_metadata_cache.get(session.cache_key)
    if cached is not None:
        return cached

    if len(_consolidated_metadata_cache) >= _MAX_CONSOLIDATED_CACHE_ENTRIES:
        evict_count = _MAX_CONSOLIDATED_CACHE_ENTRIES // 2
        for key in list(_consolidated_metadata_cache)[:evict_count]:
            # pop(..., None) rather than del: this can run in a worker thread, so a
            # concurrent request may have already evicted the same key.
            _consolidated_metadata_cache.pop(key, None)

    node_metadata: dict[str, object] = {}

    def traverse(prefix: str) -> None:
        try:
            children = _icechunk_list_dir(session.store, prefix)
        except Exception:
            logger.warning("Failed to list Icechunk store prefix '%s' for consolidated metadata", prefix, exc_info=True)
            return
        for child in sorted(children):
            path = f"{prefix}/{child}" if prefix else child
            meta_bytes = _icechunk_get(session.store, f"{path}/zarr.json")
            if meta_bytes is not None:
                meta: dict[str, object] = json.loads(meta_bytes.decode("utf-8"))
                node_metadata[path] = meta
                if meta.get("node_type") == "group":
                    traverse(path)

    traverse("")
    result: dict[str, object] = {"kind": "inline", "must_understand": False, "metadata": node_metadata}
    _consolidated_metadata_cache[session.cache_key] = result
    return result


def _normalize_icechunk_relative_path(relative_path: str) -> str:
    """Normalize a requested Icechunk key path and reject unsafe segments."""
    if "\\" in relative_path:
        raise HTTPException(status_code=400, detail="Zarr path must use '/' separators")
    target = relative_path.strip("/")
    if target == "":
        return ""
    segments = target.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise HTTPException(status_code=400, detail="Zarr path contains invalid segments")
    return target


def _serve_bytes_ranged(content: bytes, media_type: str, range_header: str | None) -> Response:
    """Serve bytes with HTTP range request support (RFC 7233).

    Zarrita's sharding_indexed codec requires range requests to read the shard index
    suffix and individual inner chunks without downloading the full shard file.
    """
    total = len(content)
    base_headers: dict[str, str] = {"Accept-Ranges": "bytes", "Content-Length": str(total)}
    if not range_header or not range_header.startswith("bytes="):
        return Response(content=content, media_type=media_type, headers=base_headers)
    try:
        spec = range_header[len("bytes=") :].strip()
        if spec.startswith("-"):
            suffix_len = int(spec[1:])
            start = max(0, total - suffix_len)
            end = total - 1
        elif spec.endswith("-"):
            start = int(spec[:-1])
            end = total - 1
        else:
            parts = spec.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
        if start > end or start >= total or end >= total:
            raise ValueError("unsatisfiable range")
    except (ValueError, IndexError):
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{total}"},
        )
    sliced = content[start : end + 1]
    return Response(
        content=sliced,
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(len(sliced)),
            "Accept-Ranges": "bytes",
        },
    )


def _get_icechunk_store_path_or_404(
    dataset_id: str, session: _IcechunkSession, relative_path: str, range_header: str | None = None
) -> Response | dict[str, object]:
    store = session.store
    target = _normalize_icechunk_relative_path(relative_path)
    if target == "":
        # The store root. A Zarr library never asks for it — it treats the URL as a base and
        # appends keys — but a person pasting the URL does, and FastAPI's redirect sends the
        # slashless form here too, so 404 made the root of every store look like a dead store.
        # Serve the group metadata it stands for, which is also what the appended request gets.
        target = "zarr.json"

    if _icechunk_exists(store, target):
        payload = _icechunk_get(store, target)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Zarr path '{relative_path}' not found")
        if target.endswith("zarr.json"):
            meta = json.loads(payload.decode("utf-8"))
            # Inject consolidated metadata into the root zarr.json so xarray can
            # enumerate variables when accessing the store over HTTP, without needing
            # directory listing or a separate consolidation step.  Result is cached
            # per snapshot so repeated requests within a snapshot lifetime are free.
            if target == "zarr.json" and meta.get("node_type") == "group":
                meta["consolidated_metadata"] = _build_icechunk_consolidated_metadata(session)
            return JSONResponse(content=meta)
        media_type, _ = mimetypes.guess_type(target)
        return _serve_bytes_ranged(payload, media_type or "application/octet-stream", range_header)

    raise HTTPException(status_code=404, detail=f"Zarr path '{relative_path}' not found")


def _decode_record(item: dict[str, object]) -> ArtifactRecord:
    """Rebuild a record from its stored form, applying schema and path migrations."""
    return ArtifactRecord.model_validate(decode_record_paths(_upgrade_legacy_record(item)))


def _encode_records(records: list[ArtifactRecord]) -> str:
    """Serialize records for disk, with store paths in their data-root-relative form."""
    payload = [encode_record_paths(record.model_dump(mode="json")) for record in records]
    return f"{json.dumps(payload, indent=2)}\n"


def _load_records() -> list[ArtifactRecord]:
    ensure_store()
    raw = json.loads(ARTIFACTS_INDEX_PATH.read_text(encoding="utf-8"))
    return [_decode_record(item) for item in raw]


def _save_records(records: list[ArtifactRecord]) -> None:
    ensure_store()
    ARTIFACTS_INDEX_PATH.write_text(_encode_records(records), encoding="utf-8")


def _store_artifact_record(
    record: ArtifactRecord,
    *,
    publish: bool,
) -> ArtifactRecord:
    """Persist a newly created artifact record while avoiding lost updates."""

    def mutate(records: list[ArtifactRecord]) -> ArtifactRecord:
        existing = _find_artifact_by_request_scope(
            records=records,
            dataset_id=record.dataset_id,
            request_scope=record.request_scope,
        )
        if existing is not None and existing.coverage == record.coverage:
            return existing

        records.append(record)
        return record

    return _mutate_records(mutate)


def _upsert_artifact_record(
    record: ArtifactRecord,
    *,
    publish: bool,
    overwrite: bool,
) -> ArtifactRecord:
    """Persist a new or replacement artifact record for the same logical request scope."""
    if not overwrite:
        return _store_artifact_record(record, publish=publish)

    def mutate(records: list[ArtifactRecord]) -> ArtifactRecord:
        existing = _find_artifact_by_request_scope(
            records=records,
            dataset_id=record.dataset_id,
            request_scope=record.request_scope,
        )
        if existing is None:
            records.append(record)
            return record

        replacement = record.model_copy(
            update={
                "artifact_id": existing.artifact_id,
                "publication": existing.publication,
            }
        )
        for index, current in enumerate(records):
            if current.artifact_id != existing.artifact_id:
                continue
            records[index] = replacement
            return replacement
        raise HTTPException(status_code=404, detail=f"Artifact '{existing.artifact_id}' not found")

    return _mutate_records(mutate)


def _mutate_records(mutation: Callable[[list[ArtifactRecord]], ArtifactRecord]) -> ArtifactRecord:
    """Apply a read-modify-write mutation under an exclusive file lock."""
    ensure_store()
    with ARTIFACTS_INDEX_PATH.open("a+", encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        records = [_decode_record(item) for item in json.loads(raw or "[]")]
        result = mutation(records)
        handle.seek(0)
        handle.truncate()
        handle.write(_encode_records(records))
        handle.flush()
        os.fsync(handle.fileno())
        portalocker.unlock(handle)
        return result


def _find_existing_artifact(
    *,
    dataset_id: str,
    request_scope: ArtifactRequestScope,
) -> ArtifactRecord | None:
    """Return an existing artifact for an identical logical request when possible."""
    return _find_existing_artifact_in_records(
        records=_load_records(),
        dataset_id=dataset_id,
        request_scope=request_scope,
    )


def _normalize_request_period(value: str, *, period_type: str, field_name: str) -> str:
    """Normalize a required request period or raise a clear client error."""
    try:
        return normalize_period_string(value, period_type)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} period '{value}': {exc}",
        ) from exc


def _normalize_optional_request_period(value: str | None, *, period_type: str, field_name: str) -> str | None:
    """Normalize an optional request period or raise a clear client error."""
    if value is None:
        return None
    return _normalize_request_period(value, period_type=period_type, field_name=field_name)


def _resolve_request_start(value: str | None, *, dataset: dict[str, object], period_type: str) -> str:
    """Return the normalized start period, substituting "now" for a forecast dataset.

    A future-facing dataset may omit ``start``: its periods lie ahead of now, so any fixed
    date would be stale by the next day and a scheduled re-ingest would silently drift out
    of the forecast window. Omitting it means "from now", resolved here in the dataset's
    native period format.

    Note what is deliberately *not* done: ``None`` is never passed through to
    ``plugin.periods()``. Every existing plugin's signature is ``periods(start: str, end:
    str)`` and several parse the value immediately (``date.fromisoformat(start)``), so
    handing them ``None`` would turn an omitted field into a ``TypeError`` deep inside a
    plugin. Resolving here keeps the plugin contract exactly as it is.

    A historical dataset with no start is a client error, not a defaulted one — silently
    ingesting from "now" backwards would be a guess about intent, and the whole record is
    rarely what someone wants.
    """
    if value is not None:
        return _normalize_request_period(value, period_type=period_type, field_name="start")

    if not registry_datasets.is_future_facing(dataset):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset '{dataset.get('id')}' requires a start period: its periods are historical. "
                "Only datasets declaring 'temporal_direction: future' (e.g. forecasts) may omit it."
            ),
        )
    return _current_request_period(period_type)


def _forecast_horizon(dataset: dict[str, object], period_type: str) -> str:
    """Return how far ahead to ask a forecast plugin for, when no end was requested.

    Prefers the template's declared ``extents.temporal.end``. Otherwise offers a year
    ahead — deliberately generous, because the real limit belongs to the plugin (its own
    lead-time parameter) and a bound here would silently truncate a longer forecast. The
    plugin is expected to clip to what it publishes; ``request_scope.end`` remains None so
    the coverage check does not require it to fill this whole window.

    A year rather than no bound at all keeps ``periods()`` typed as ``end: str``, so a
    plugin author never has to handle a missing end.
    """
    declared = registry_datasets.declared_temporal_end(dataset)
    if declared:
        return declared
    # 365 days rather than replace(year=+1): the latter raises "day is out of range for
    # month" on 29 February, since the following year has no such date — a crash on leap
    # day for a horizon that is approximate by design. Read the date once, so a run
    # crossing midnight cannot mix two different days.
    horizon = utc_today() + timedelta(days=365)
    if period_type == "yearly":
        return str(horizon.year)
    if period_type == "monthly":
        return f"{horizon.year:04d}-{horizon.month:02d}"
    if period_type in {"hourly", "weekly"}:
        return datetime_to_period_string(datetime.combine(horizon, datetime.min.time(), tzinfo=UTC), period_type)
    return horizon.isoformat()


def _current_request_period(period_type: str) -> str:
    """Return "now" as a dataset-native period string.

    Used for both an omitted ``end`` (sync through the latest period) and an omitted
    ``start`` on a future-facing dataset (ingest from now forward).
    """
    if period_type == "hourly":
        return datetime_to_period_string(utc_now(), period_type)
    if period_type == "daily":
        return utc_today().isoformat()
    if period_type == "dekadal":
        return dekad_start(utc_today()).isoformat()
    if period_type == "weekly":
        return datetime_to_period_string(utc_now(), period_type)
    if period_type == "monthly":
        today = utc_today()
        return f"{today.year:04d}-{today.month:02d}"
    if period_type == "yearly":
        return str(utc_today().year)
    if period_type == "climatology":
        # Day-of-year climatology: the plugin enumerates 1..366 regardless of the
        # request range, so the end value is unused — return a stable sentinel.
        return "366"
    raise HTTPException(status_code=400, detail=f"Invalid period_type '{period_type}' for request period defaulting")


def _validate_download_scope(
    *,
    start: str,
    end: str | None,
    download_start: str | None,
    download_end: str | None,
) -> None:
    """Validate optional delta download scope against the normalized request scope."""
    if (download_start is None) != (download_end is None):
        raise HTTPException(
            status_code=400,
            detail="download_start and download_end must either both be provided or both be omitted",
        )
    if download_start is None or download_end is None:
        return
    if download_start < start:
        raise HTTPException(status_code=400, detail="download_start must be greater than or equal to start")
    if download_end < download_start:
        raise HTTPException(status_code=400, detail="download_end must be greater than or equal to download_start")
    if end is not None and download_end > end:
        raise HTTPException(status_code=400, detail="download_end must be less than or equal to end")


def _find_existing_artifact_in_records(
    *,
    records: list[ArtifactRecord],
    dataset_id: str,
    request_scope: ArtifactRequestScope,
) -> ArtifactRecord | None:
    """Return an existing artifact for an identical logical request from a provided record set."""
    for record in reversed(records):
        if not _artifact_storage_exists(record):
            logger.warning(
                "Ignoring stale artifact '%s' because backing storage is missing",
                record.artifact_id,
            )
            continue
        if record.dataset_id != dataset_id:
            continue
        if record.request_scope != request_scope:
            continue
        if not _artifact_coverage_matches_request_scope(record):
            logger.warning(
                "Ignoring existing artifact '%s' because coverage %s..%s does not match request scope %s..%s",
                record.artifact_id,
                record.coverage.temporal.start,
                record.coverage.temporal.end,
                record.request_scope.start,
                record.request_scope.end,
            )
            continue
        return record
    return None


def _find_artifact_by_request_scope(
    *,
    records: list[ArtifactRecord],
    dataset_id: str,
    request_scope: ArtifactRequestScope,
) -> ArtifactRecord | None:
    """Return the latest materialized record for the same operation request.

    Unlike the API reuse lookup above, persistence must allow cumulative coverage
    to extend beyond a finite incremental request. The caller compares coverage
    when deduplicating and replaces the record explicitly for overwrite semantics.
    """
    for record in reversed(records):
        if record.dataset_id != dataset_id or record.request_scope != request_scope:
            continue
        if _artifact_storage_exists(record):
            return record
    return None


def _artifact_coverage_matches_request_scope(record: ArtifactRecord) -> bool:
    """Return whether an existing artifact is safe to reuse for its request scope."""
    return _temporal_coverage_matches_request_scope(record.coverage.temporal, record.request_scope)


def _materialized_records(records: list[ArtifactRecord]) -> list[ArtifactRecord]:
    """Return only artifact records whose backing storage still exists."""
    materialized: list[ArtifactRecord] = []
    for record in records:
        if _artifact_storage_exists(record):
            materialized.append(record)
            continue
        logger.warning("Ignoring stale artifact '%s' because backing storage is missing", record.artifact_id)
    return materialized


def _artifact_storage_exists(record: ArtifactRecord) -> bool:
    """Return whether an artifact's on-disk backing files are still present."""
    paths: list[str] = []
    if record.path is not None:
        paths.append(record.path)
    if record.asset_paths:
        paths.extend(record.asset_paths)
    if not paths:
        return False
    return all(Path(path).exists() for path in paths)


def _temporal_coverage_matches_request_scope(
    temporal: CoverageTemporal,
    request_scope: ArtifactRequestScope,
) -> bool:
    """Return whether temporal coverage exactly matches the requested temporal scope."""
    if temporal.start != request_scope.start:
        return False
    # Open-ended requests intentionally reuse the latest artifact for the same
    # logical start/scope even though the realized end is time-dependent.
    if request_scope.end is not None and temporal.end != request_scope.end:
        return False
    return True


def _temporal_coverage_matches_streaming_request_scope(
    temporal: CoverageTemporal,
    request_scope: ArtifactRequestScope,
) -> bool:
    """Return whether streaming coverage is compatible with the requested scope.

    Plugin-backed streaming ingest may legitimately clamp the realized end of an
    artifact to the source's latest available period. That should still be
    treated as a successful ingest as long as the coverage starts where
    requested and does not extend beyond the requested temporal end.

    Start handling remains strict by design. A later-than-requested start
    indicates the realized dataset does not cover the requested opening period
    at all, while a shorter realized end can be a normal consequence of source
    availability clamping.
    """
    if temporal.start != request_scope.start:
        return False
    if request_scope.end is not None and temporal.end is not None and temporal.end > request_scope.end:
        return False
    return True


def _build_dataset_record(dataset_id: str, artifacts: list[ArtifactRecord]) -> DatasetRecord:
    latest = max(artifacts, key=lambda artifact: artifact.created_at)
    source_dataset = registry_datasets.get_dataset(latest.dataset_id) or {}
    return DatasetRecord(
        dataset_id=dataset_id,
        source_dataset_id=latest.source_dataset_id or latest.dataset_id,
        dataset_name=latest.dataset_name,
        short_name=_as_optional_str(source_dataset.get("short_name")),
        variable=latest.variable,
        period_type=_as_optional_str(source_dataset.get("period_type")) or latest.period_type or "unknown",
        units=_as_optional_str(source_dataset.get("units")),
        resolution=_as_optional_str(source_dataset.get("resolution")),
        source=_as_optional_str(source_dataset.get("source")),
        source_url=_as_optional_str(source_dataset.get("source_url")),
        extent=latest.coverage,
        last_updated=latest.created_at,
        links=_dataset_links(dataset_id, latest),
        publication=DatasetPublication(
            status=latest.publication.status,
            published_at=latest.publication.published_at,
        ),
    )


def _build_ingestion_response(artifact: ArtifactRecord) -> IngestionResponse:
    """Build an operational ingestion response from one stored artifact record."""
    return IngestionResponse(
        ingestion_id=artifact.artifact_id,
        status="completed",
        dataset=get_dataset_summary_for_artifact_or_404(artifact.artifact_id),
    )


def _build_dataset_detail_record(dataset_id: str, artifacts: list[ArtifactRecord]) -> DatasetDetailRecord:
    base = _build_dataset_record(dataset_id, artifacts)
    ordered_artifacts = sorted(artifacts, key=lambda artifact: artifact.created_at, reverse=True)
    return DatasetDetailRecord(
        **base.model_dump(),
        versions=[
            DatasetVersionRecord(
                created_at=artifact.created_at,
                format=artifact.format,
                coverage=artifact.coverage,
                request_scope=artifact.request_scope,
            )
            for artifact in ordered_artifacts
        ],
    )


def _dataset_links(dataset_id: str, latest: ArtifactRecord) -> list[DatasetAccessLink]:
    links = [DatasetAccessLink(href=f"/datasets/{dataset_id}", rel="self", title="Dataset detail")]
    if latest.format == ArtifactFormat.ICECHUNK:
        links.append(DatasetAccessLink(href=f"/zarr/{dataset_id}", rel="zarr", title="Zarr store"))
    if latest.publication.status == PublicationStatus.PUBLISHED and latest.format == ArtifactFormat.ICECHUNK:
        links.append(DatasetAccessLink(href=f"/stac/collections/{dataset_id}", rel="stac", title="STAC collection"))
    if latest.format == ArtifactFormat.NETCDF:
        links.append(
            DatasetAccessLink(href=f"/datasets/{dataset_id}/download", rel="download", title="Download NetCDF")
        )
    return links


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _upgrade_legacy_record(item: dict[str, object]) -> dict[str, object]:
    """Backfill newer schema fields for records created before migrations existed."""
    if "request_scope" not in item:
        coverage = item.get("coverage")
        if isinstance(coverage, dict):
            spatial = coverage.get("spatial")
            temporal = coverage.get("temporal")
            bbox: tuple[float, float, float, float] | None = None
            if isinstance(spatial, dict):
                xmin = spatial.get("xmin")
                ymin = spatial.get("ymin")
                xmax = spatial.get("xmax")
                ymax = spatial.get("ymax")
                if (
                    isinstance(xmin, int | float)
                    and isinstance(ymin, int | float)
                    and isinstance(xmax, int | float)
                    and isinstance(ymax, int | float)
                ):
                    bbox = (float(xmin), float(ymin), float(xmax), float(ymax))

            start = ""
            end: str | None = None
            if isinstance(temporal, dict):
                raw_start = temporal.get("start")
                raw_end = temporal.get("end")
                if isinstance(raw_start, str):
                    start = raw_start
                if isinstance(raw_end, str):
                    end = raw_end

            item["request_scope"] = {
                "start": start,
                "end": end,
                "bbox": bbox,
            }
    return item
