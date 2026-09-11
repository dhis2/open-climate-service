"""Internal per-period streaming ingest orchestrator.

This module implements the Ticket 1 execution loop:
1. enumerate source-valid periods,
2. reconcile those periods with committed store state,
3. fetch missing periods with bounded concurrency,
4. infer the store grid from the first fetched period, and
5. append each fetched period into a flat Icechunk-backed Zarr v3 store.

It is intentionally lower-level than `open_climate_service.ingestions.services`.
The ingestion service decides *when* to use streaming and how to persist
artifact metadata. This module decides *how* one plugin-backed stream is
fetched and committed safely.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import xarray as xr

from open_climate_service.shared.cf import apply_cf_metadata, cf_attrs_from_template, drop_unserializable_attrs
from open_climate_service.shared.raster_contract import (
    prepare_for_publication,
    spatial_coords_match,
)
from open_climate_service.streaming.protocol import GridSpec, IngestionPlugin, close_ingestion_plugin
from open_climate_service.streaming.store import (
    committed_data_group,
    is_store_empty,
    open_or_create_repo,
    read_committed_period_ids,
    write_geozarr_attrs,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingIngestResult:
    """Structured result returned to the ingestion service layer.

    The result is intentionally small. Artifact persistence, publication, and
    public API response construction remain the responsibility of
    `open_climate_service.ingestions`.
    """

    store_path: Path
    period_type: str
    periods_written: int


def _strip_cf_encoding(ds: xr.Dataset, period_type: str, *, time_dim: str) -> None:
    """Normalize xarray encodings so repeated zarr appends remain stable.

    Source datasets frequently carry CF encoding hints that are useful for
    one-off writes but unstable across repeated append sessions. The streaming
    path strips them to keep later commits shape-compatible.
    """
    cf_keys = frozenset({"scale_factor", "add_offset", "missing_value", "_FillValue", "coordinates"})
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding.clear()
        ds[name].attrs = {key: value for key, value in ds[name].attrs.items() if key not in cf_keys}
    # Only force datetime CF encoding for an actual datetime axis. An ordinal step
    # dimension (e.g. an integer day-of-year climatology) is left as-is.
    if time_dim in ds.coords and ds[time_dim].dtype.kind == "M":
        units = "hours since 1970-01-01" if period_type == "hourly" else "days since 1970-01-01"
        ds[time_dim].encoding.update({"units": units, "dtype": "int32"})


def _grid_spec_from_dataset(ds: xr.Dataset, *, time_dim: str, x_dim: str, y_dim: str, crs: str) -> GridSpec:
    """Infer the store `GridSpec` from the first fetched period.

    Reads dtype/nodata before CF encoding is stripped, so the no-data sentinel survives into the
    store's fill value. ``crs`` comes from :func:`prepare_for_publication`, which is the single
    place a dataset's CRS is decided — a second inference here would be a second answer.
    """
    try:
        primary = next(iter(ds.data_vars))
    except StopIteration as exc:
        raise RuntimeError("Fetched dataset has no data variables to infer a GridSpec from") from exc
    da = ds[primary]
    nodata = da.encoding.get("_FillValue", da.attrs.get("_FillValue"))
    try:
        shape = (int(ds.sizes[y_dim]), int(ds.sizes[x_dim]))
    except KeyError as exc:
        raise RuntimeError(
            f"Fetched dataset is missing the expected spatial dimensions '{y_dim}' and '{x_dim}'; "
            f"got dims {tuple(ds.sizes)}. The plugin must return data on those dims (or set x_dim/y_dim)."
        ) from exc
    return GridSpec(
        shape=shape,
        crs=crs,
        dtype=da.dtype,
        nodata=float(nodata) if nodata is not None else None,
        time_dim=time_dim,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def _assert_appendable_axes(store_path: Path, ds: xr.Dataset, *, x_dim: str, y_dim: str) -> None:
    """Refuse to append a period whose spatial axes disagree with the committed store.

    Normalisation reorders rows (y descending) and columns (longitudes rolled to −180…180). An
    append writes along the time axis only, so the committed coordinate arrays stay as they were:
    a period normalised one way, appended to a store written the other, puts data under
    coordinates that no longer describe it — a silently mirrored or half-shifted raster, which is
    far worse than metadata being out of date.

    That can only happen to a store written before the contract existed (or by a plugin whose
    orientation has since changed), and the fix is a re-ingest rather than anything this function
    can do safely, so it fails with an actionable message instead of guessing.
    """
    from open_climate_service.streaming.store import read_committed_spatial_coords

    stored = read_committed_spatial_coords(store_path, x_dim=x_dim, y_dim=y_dim)
    if stored is None:
        return  # nothing committed to disagree with, or a store we cannot read
    for axis in (y_dim, x_dim):
        if axis not in ds.coords or axis not in stored:
            continue
        if spatial_coords_match(stored[axis], ds[axis], axis=axis):
            continue
        raise RuntimeError(
            f"Cannot append to {store_path.name}: its committed {axis!r} coordinates do not match "
            f"the normalised ones for this period "
            f"(store {stored[axis][0]!r}…{stored[axis][-1]!r}, "
            f"period {float(ds[axis].values[0])!r}…{float(ds[axis].values[-1])!r}). "
            "The store predates the published-store layout contract (y descending, longitudes "
            "−180…180); appending would write data under coordinates that no longer describe it. "
            "Re-ingest this dataset from scratch to rebuild it under the current contract."
        )


async def run_streaming_ingest(
    *,
    plugin: IngestionPlugin,
    params: dict[str, Any],
    dataset: dict[str, Any] | None = None,
    bbox: list[float],
    start: str,
    end: str,
    store_path: Path,
    period_type: str,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, Any]], None] | None = None,
    periods: list[str] | None = None,
    committed_periods: list[str] | None = None,
) -> StreamingIngestResult:
    """Stream one dataset into a flat Zarr v3 store one period at a time.

    Resume policy:
    - committed store state is authoritative

    This avoids replaying already-committed periods after crashes that happen
    between a store commit and a later cursor write.

    ``periods`` may be supplied by the sync planner to avoid a second
    ``plugin.periods()`` call when the list was already fetched during planning.
    """
    from open_climate_service.jobs.models import JobCancelledError

    # The grid is inferred from the first fetched period (below). Dimension names and
    # the CRS fallback come from the plugin's class attributes (defaults t/x/y, 4326).
    time_dim: str = getattr(plugin, "time_dim", "t")
    x_dim: str = getattr(plugin, "x_dim", "x")
    y_dim: str = getattr(plugin, "y_dim", "y")
    fallback_crs: int | str = getattr(plugin, "crs", 4326)
    spec: GridSpec | None = None

    all_periods = periods if periods is not None else await plugin.periods(start, end)
    if not all_periods:
        close_ingestion_plugin(plugin)
        return StreamingIngestResult(store_path=store_path, period_type=period_type, periods_written=0)

    committed = (
        set(committed_periods)
        if committed_periods is not None
        else read_committed_period_ids(store_path, period_type, time_dim=time_dim)
    )
    pending = [period for period in all_periods if period not in committed]
    if not pending:
        close_ingestion_plugin(plugin)
        return StreamingIngestResult(store_path=store_path, period_type=period_type, periods_written=0)

    if on_progress:
        on_progress(len(all_periods) - len(pending), len(all_periods), f"{len(pending)} periods pending")

    if committed:
        is_first_write = False
    elif is_store_empty(store_path):
        is_first_write = True
    else:
        close_ingestion_plugin(plugin)
        raise RuntimeError(
            f"Existing store at {store_path} holds data whose committed periods could not be read, so it is "
            "neither safe to append to nor safe to overwrite. Inspect the store, or remove it to re-ingest "
            "from scratch. A store that was created but never written to is not this case and is handled as "
            "a first write."
        )

    repo = open_or_create_repo(store_path)
    period_queue = iter(pending)
    in_flight: deque[tuple[str, asyncio.Task[xr.Dataset]]] = deque()
    max_parallel = max(1, int(getattr(plugin, "max_concurrency", 1)))
    # Ticket 1 still commits every fetched period individually. The plugin's
    # commit_batch_size currently controls cursor checkpoint cadence rather than
    # Icechunk transaction batching.
    commit_batch_size = max(1, int(getattr(plugin, "commit_batch_size", 1)))
    expected_spatial_shape: tuple[int, int] | None = None
    # The append-axes check runs once per run, on the first period appended to an existing store.
    axes_checked = False
    # Group to append into: None for a flat store, "0" for a pyramided one.
    append_group: str | None = None
    # CF attributes (units / standard_name / cell_methods) declared on the template,
    # stamped onto each written period so the store is CF-compliant on disk (#280).
    # The template is authoritative for the fields it declares, so we overwrite any
    # generic or placeholder value the source/transform left behind (e.g. GRIB's
    # standard_name="unknown", or a unit conversion's dimensionally-generic "mm" for
    # what the template declares as a rate "mm/d"). Fields the template omits are kept.
    cf_attrs = cf_attrs_from_template(dataset)

    # fetch_period may be a plain (blocking) method or an async one. Run blocking
    # implementations in a worker thread; await native coroutines directly.
    fetch_is_async = inspect.iscoroutinefunction(plugin.fetch_period)

    async def _fetch(period_id: str) -> xr.Dataset:
        if fetch_is_async:
            return await cast("Awaitable[xr.Dataset]", plugin.fetch_period(period_id, bbox, **params))
        sync_fetch = cast("Callable[..., xr.Dataset]", plugin.fetch_period)
        return await asyncio.to_thread(sync_fetch, period_id, bbox, **params)

    # Keep only a rolling fetch window in memory rather than creating one task
    # for every pending period up front. This keeps long backfills bounded.
    for _ in range(min(max_parallel, len(pending))):
        period_id = next(period_queue, None)
        if period_id is None:
            break
        in_flight.append((period_id, asyncio.create_task(_fetch(period_id))))
    written = 0
    try:
        while in_flight:
            if is_cancel_requested and is_cancel_requested():
                raise JobCancelledError("Streaming ingest cancelled")

            period_id, task = in_flight.popleft()
            ds = await task
            # Always release the fetched dataset's backing handles (open_rasterio /
            # open_dataset / remote HTTP) once it is written, so long backfills don't
            # leak file descriptors one period at a time.
            try:
                # Enforce the published-store contract on every period, so a plugin can return
                # data in whatever orientation its source delivers (CLIM-821). Deterministic, so
                # every period of a run agrees; a store written before the contract can disagree,
                # which _assert_appendable_axes catches below.
                prepared = prepare_for_publication(
                    ds, fallback_crs=fallback_crs, time_dim=time_dim, x_dim=x_dim, y_dim=y_dim
                )
                ds = prepared.dataset
                # Infer the grid from the first fetched period, before _strip_cf_encoding
                # so the nodata sentinel survives into the spec.
                if spec is None:
                    spec = _grid_spec_from_dataset(ds, time_dim=time_dim, x_dim=x_dim, y_dim=y_dim, crs=prepared.crs)
                _strip_cf_encoding(ds, period_type, time_dim=spec.time_dim)
                apply_cf_metadata(ds, cf_attrs, overwrite=True)
                # Every plugin, not just the ones that noticed: a source's own attributes may be
                # shapes Zarr accepts and NetCDF cannot, and the store is what later exports read.
                drop_unserializable_attrs(ds, context=f"{(dataset or {}).get('id', '<unknown>')} {period_id}")
                try:
                    spatial_shape = (int(ds.sizes[spec.y_dim]), int(ds.sizes[spec.x_dim]))
                except KeyError as exc:
                    raise RuntimeError(
                        f"Fetched dataset for {period_id} is missing expected spatial dimensions "
                        f"'{spec.y_dim}' and '{spec.x_dim}'"
                    ) from exc
                if expected_spatial_shape is None:
                    expected_spatial_shape = spatial_shape
                elif spatial_shape != expected_spatial_shape:
                    raise RuntimeError(
                        f"Fetched dataset for {period_id} has spatial shape {spatial_shape}, "
                        f"expected {expected_spatial_shape}"
                    )

                session = repo.writable_session("main")
                if is_first_write:
                    ds.to_zarr(session.store, mode="w", zarr_format=3)
                    is_first_write = False
                else:
                    if not axes_checked:
                        _assert_appendable_axes(store_path, ds, x_dim=x_dim, y_dim=y_dim)
                        # Where the data actually lives. A store promoted to a pyramid by an
                        # earlier sync keeps its variables in level groups, and appending at the
                        # root would create a second one-timestep array beside the pyramid,
                        # leaving the store unopenable. Resolved once per sync, since promotion
                        # only happens between syncs.
                        append_group = committed_data_group(store_path)
                        axes_checked = True
                    ds.to_zarr(
                        session.store,
                        group=append_group,
                        append_dim=spec.time_dim,
                        zarr_format=3,
                    )
                # Root attrs are rewritten on every commit so later append sessions
                # preserve GeoZarr metadata even if the underlying store layer only
                # touches array content for the new period.
                write_geozarr_attrs(session.store, spec=spec, bbox=bbox)
                session.commit(f"ingest: {period_id}")
            finally:
                ds.close()

            written += 1
            if save_cursor and (written % commit_batch_size == 0 or written == len(pending)):
                save_cursor({"last_committed": period_id})
            if on_progress:
                on_progress(
                    len(all_periods) - len(pending) + written,
                    len(all_periods),
                    f"Wrote {period_id}",
                )

            next_period = next(period_queue, None)
            if next_period is not None:
                in_flight.append((next_period, asyncio.create_task(_fetch(next_period))))
    finally:
        tasks_to_cancel = [task for _, task in in_flight if not task.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        close_ingestion_plugin(plugin)

    # Prune intermediate ingest snapshots: each period commit created one snapshot;
    # only the final HEAD state needs to be retained.  expire_snapshots marks older
    # snapshots as expired without deleting chunk data — garbage_collect would be
    # needed to reclaim manifest storage.  The "main" branch ref preserves HEAD.
    from datetime import datetime, timezone

    try:
        final_repo = open_or_create_repo(store_path)
        expired = final_repo.expire_snapshots(older_than=datetime.now(tz=timezone.utc))
        if expired:
            logger.info("Expired %d intermediate snapshots from %s", len(expired), store_path)
    except Exception:
        logger.warning("expire_snapshots failed for %s — store remains valid", store_path, exc_info=True)

    return StreamingIngestResult(store_path=store_path, period_type=period_type, periods_written=written)


def run_streaming_ingest_sync(
    *,
    plugin: IngestionPlugin,
    params: dict[str, Any],
    dataset: dict[str, Any] | None = None,
    bbox: list[float],
    start: str,
    end: str,
    store_path: Path,
    period_type: str,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, Any]], None] | None = None,
    periods: list[str] | None = None,
    committed_periods: list[str] | None = None,
) -> StreamingIngestResult:
    """Synchronous wrapper for threaded job execution.

    The jobs framework invokes callables in worker threads today, so the
    ingestion service exposes this sync wrapper and lets the orchestrator keep
    its async internal structure.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_streaming_ingest_sync cannot be called from a running event loop")

    return asyncio.run(
        run_streaming_ingest(
            plugin=plugin,
            params=params,
            dataset=dataset,
            bbox=bbox,
            start=start,
            end=end,
            store_path=store_path,
            period_type=period_type,
            on_progress=on_progress,
            is_cancel_requested=is_cancel_requested,
            save_cursor=save_cursor,
            periods=periods,
            committed_periods=committed_periods,
        )
    )
