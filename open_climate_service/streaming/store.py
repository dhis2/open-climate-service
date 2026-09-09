"""Icechunk-backed store helpers for the internal streaming engine.

Ticket 1 is deliberately flat-store-only. These helpers therefore focus on the
minimum store concerns needed for initial per-period append:
open or create an Icechunk repository, inspect committed time steps for resume,
and keep GeoZarr-compatible root attributes stable on the written Zarr v3
store.

Rechunking, multiscale/pyramid behavior, and broader publication concerns are
explicitly deferred to later tickets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geozarr_toolkit import create_geozarr_attrs

from open_climate_service.shared.crs import store_crs_attrs
from open_climate_service.shared.geozarr import check_grid_description, grid_geometry, write_gdal_geotransform
from open_climate_service.stac.media_types import is_multiscales_convention
from open_climate_service.streaming.protocol import GridSpec

logger = logging.getLogger(__name__)


def open_or_create_repo(store_path: Path) -> Any:
    """Open an existing Icechunk repository or create one at ``store_path``."""
    import icechunk

    storage = icechunk.local_filesystem_storage(str(store_path))
    if store_path.exists():
        return icechunk.Repository.open(storage)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return icechunk.Repository.create(storage)


def committed_data_group(store_path: Path) -> str | None:
    """Return the group holding the committed data variables: ``"0"``, or ``None`` for the root.

    A pyramided store keeps its data in level groups and leaves the root with only the time
    coordinate and ``spatial_ref``, so every reader and the *appender* have to descend. Getting
    this wrong is not a degraded read: appending at the root of a pyramided store creates a
    second, one-timestep ``gpp`` beside the pyramid, after which the root no longer opens at
    all — ``conflicting sizes for dimension 't'`` — and nothing can repair it in place.

    Returns None for a missing or unreadable store, which callers already treat as "nothing
    committed to compare against".
    """
    import xarray as xr

    if not store_path.exists():
        return None
    try:
        repo = open_or_create_repo(store_path)
        store = repo.readonly_session("main").store
        ds = xr.open_zarr(store)
        try:
            if ds.data_vars:
                return None
        finally:
            ds.close()
        # Confirm the level exists rather than assuming the layout.
        level0 = xr.open_zarr(store, group="0")
        level0.close()
        return "0"
    except Exception:
        logger.debug("Could not determine the committed data group for %s", store_path, exc_info=True)
        return None


def _open_committed(store_path: Path) -> Any:
    """Open the committed store at whichever group actually holds the data."""
    import xarray as xr

    repo = open_or_create_repo(store_path)
    store = repo.readonly_session("main").store
    group = committed_data_group(store_path)
    return xr.open_zarr(store) if group is None else xr.open_zarr(store, group=group)


def read_committed_period_ids(store_path: Path, period_type: str, *, time_dim: str = "t") -> set[str]:
    """Return period ids already committed in the store, or an empty set.

    Resume correctness is store-first: if the repository already contains a
    committed time step, the orchestrator treats that as authoritative even when
    a persisted cursor is stale or missing.
    """
    import pandas as pd

    from open_climate_service.shared.time import datetime_to_period_string

    if not store_path.exists():
        return set()

    try:
        # Level 0 rather than the root: the root time coordinate of a pyramided store is
        # rebuilt at the end of a sync, so mid-sync it lags the data and resume would
        # re-append periods that are already committed.
        ds = _open_committed(store_path)
        try:
            if time_dim not in ds.coords:
                return set()
            coord = ds[time_dim]
            if coord.dtype.kind != "M":
                # Non-datetime (ordinal) step dimension — e.g. an integer
                # ``dayofyear`` axis. Period ids are the coord values as plain
                # strings, matching what ``plugin.periods()`` emits; parsing them
                # as datetimes would mangle every value and leave ``committed``
                # empty, causing duplicate appends on resume.
                if coord.dtype.kind in "iu":
                    return {str(int(item)) for item in coord.values}
                return {str(item.item()) for item in coord.values}
            return {
                datetime_to_period_string(pd.Timestamp(item.item()).to_pydatetime(), period_type)
                for item in coord.values
            }
        finally:
            ds.close()
    except Exception:
        logger.debug("Could not read committed periods from %s", store_path, exc_info=True)
        return set()


def is_store_empty(store_path: Path) -> bool:
    """Return whether an Icechunk repository has no arrays, groups, or attrs.

    This is a conservative safety check for first-write detection. If the store
    cannot be inspected, we treat it as non-empty to avoid a destructive
    `mode="w"` rewrite of data that may already exist.

    A repository created but never written to is the one inspection failure that means
    the opposite. Ingest creates the repo before fetching, so any failure before the first
    period commits leaves a repo holding nothing but its initial snapshot. Reading that as
    "cannot inspect, assume data" made the emptiest possible store look occupied, and the
    orchestrator's first-write guard then refused every retry until the directory was
    deleted by hand (CLIM-898). Nothing committed is not an unknown state: it is an empty
    one, and there is nothing for a `mode="w"` write to destroy.

    That case is decided from Icechunk's own history rather than from how `zarr.open_group`
    reports a missing hierarchy, which varies by version. The `GroupNotFoundError` branch
    below then only covers a repo that has commits but no group, and cannot be the sole
    signal for a never-written store.
    """
    import itertools

    import zarr
    from zarr.errors import GroupNotFoundError

    if not store_path.exists():
        return True

    try:
        repo = open_or_create_repo(store_path)
        # Two is enough to answer "more than the initial snapshot?" — ancestry is lazy, so
        # this does not walk the history of a store with thousands of commits.
        if len(list(itertools.islice(repo.ancestry(branch="main"), 2))) < 2:
            logger.debug("%s has no commits beyond initialization; treating as empty", store_path)
            return True
        session = repo.readonly_session("main")
        try:
            root = zarr.open_group(session.store, mode="r")
        except GroupNotFoundError:
            logger.debug("%s holds no zarr group; treating as empty", store_path)
            return True
        return not list(root.array_keys()) and not list(root.group_keys()) and not bool(root.attrs.asdict())
    except Exception:
        logger.debug("Could not determine whether %s is empty", store_path, exc_info=True)
        return False


def read_committed_spatial_coords(store_path: Path, *, x_dim: str = "x", y_dim: str = "y") -> Any:
    """The spatial coordinate arrays already committed to a store, or None.

    An append writes along the time axis only, leaving these as they are — so a period whose
    rows or columns have been reordered by normalisation would land under coordinates that no
    longer describe it. The orchestrator compares against these before its first append.

    Returns None when there is nothing to compare against (new store, unreadable, no coords).

    Reads level 0 for a pyramided store. The root there carries only the time coordinate and
    ``spatial_ref``, so reading the root returned None and silently disabled the check — on
    exactly the large stores where a mirrored raster is hardest to spot by eye.
    """
    if not store_path.exists():
        return None
    try:
        ds = _open_committed(store_path)
        try:
            if x_dim not in ds.coords or y_dim not in ds.coords:
                return None
            return {x_dim: ds[x_dim].values, y_dim: ds[y_dim].values}
        finally:
            ds.close()
    except Exception:  # noqa: BLE001 — best effort; a store we cannot read cannot be checked
        logger.debug("Could not read committed spatial coordinates from %s", store_path, exc_info=True)
        return None


def _stored_grid_geometry(root: Any, spec: GridSpec) -> dict[str, Any] | None:
    """Grid geometry read from the store's own coordinate arrays, or None if unavailable.

    The coordinate arrays are the ground truth for where the data actually is. The requested
    ``bbox`` is not: a source can return less than was asked for (CHIRPS ends at 60°N, so a
    Norway request to 72.5°N yields a grid that stops at 60°N), and describing the request
    would stretch the raster over ground it does not cover.

    Returns None for a store whose coordinates aren't readable yet — the first-write path
    calls this before any data exists, and a 1-cell axis has no derivable cell size.
    """
    try:
        x_centres = root[spec.x_dim][:]
        y_centres = root[spec.y_dim][:]
    except (KeyError, TypeError):
        return None
    return grid_geometry(x_centres, y_centres)


def write_geozarr_attrs(store: Any, *, spec: GridSpec, bbox: list[float]) -> None:
    """Write root metadata for a flat Zarr v3 store.

    The attrs are rewritten after each commit rather than only on first write so
    root metadata stays stable across repeated append sessions.
    """
    import zarr

    root = zarr.open_group(store, mode="r+")
    geometry = _stored_grid_geometry(root, spec)

    attrs = create_geozarr_attrs(
        # Array order, (y, x) — `spatial:dimensions` and `spatial:shape` are read
        # positionally, so naming them x-first transposes the grid for any client that
        # trusts the convention.
        dimensions=[spec.y_dim, spec.x_dim],
        crs=f"EPSG:{spec.crs}",
        bbox=bbox,
        shape=spec.shape,
    )
    # `create_geozarr_attrs` records an EPSG input as `proj:code` and nothing else, which
    # only describes the store to a reader that can look the code up. Overwrite with the full
    # `proj:` set so the store carries its own definition — see store_crs_attrs.
    attrs.update(store_crs_attrs(f"EPSG:{spec.crs}"))
    # Native-CRS extent, in the GeoZarr `spatial:bbox` convention. Direct-Zarr clients
    # (GDAL/QGIS, zarr-layer) read the CRS from the CF grid-mapping (`crs_wkt`) / the `proj:`
    # convention above, and the extent from here or the coordinate arrays — no non-standard
    # `proj4`/`bounds` attrs required. stac/services.py publishes the same CRS fields plus
    # `proj:bbox` on the collection, for clients that read the catalogue and not the store.
    attrs["spatial:bbox"] = bbox
    if geometry is not None:
        # The affine is what a client actually places the raster with. Without it, viewers
        # fall back to inferring a grid from the coordinate arrays and assume EPSG:4326 while
        # doing so, which puts every projected store (seNorge's UTM33 metres) off the map.
        attrs["spatial:transform"] = geometry["transform"]
        attrs["spatial:shape"] = geometry["shape"]
        attrs["spatial:bbox"] = geometry["bbox"]
    attrs.update(spec.attrs)

    # Checked after `spec.attrs` is merged, not before: a plugin's own attrs land last and
    # could otherwise reintroduce a bad description past the guard. `spec.shape` is (y, x).
    check_grid_description(
        attrs,
        y_dim=spec.y_dim,
        x_dim=spec.x_dim,
        y_size=int(spec.shape[0]),
        x_size=int(spec.shape[1]),
    )

    # An append to a pyramided store must not un-declare its pyramid. `create_geozarr_attrs`
    # builds `zarr_conventions` for a *flat* store, and the `update` below replaces the list
    # wholesale — dropping the multiscales entry while the data still lives under `0/`, so the
    # store would advertise itself as flat while having no root data variables at all. That
    # then persists if the run aborts before the end-of-sync pyramid rebuild.
    existing_conventions = root.attrs.get("zarr_conventions")
    if isinstance(existing_conventions, list):
        preserved = [c for c in existing_conventions if is_multiscales_convention(c)]
        if preserved:
            declared = attrs.get("zarr_conventions")
            attrs["zarr_conventions"] = ([*declared] if isinstance(declared, list) else []) + preserved

    root.attrs.update(attrs)
    if geometry is not None:
        write_gdal_geotransform(root, geometry["transform"])
