"""Write raster datasets to Icechunk stores with GeoZarr conventions."""

import logging
from pathlib import Path
from typing import Any, cast

import xarray as xr
import xproj  # noqa: F401  # type: ignore[import-untyped]  # pyright: ignore[reportUnusedImport]
import zarr
from geozarr_toolkit import MultiscalesConventionMetadata, create_geozarr_attrs
from topozarr import CoarseningMethod
from topozarr.coarsen import create_pyramid

from open_climate_service import config as api_config
from open_climate_service.shared.geozarr import grid_geometry, write_gdal_geotransform
from open_climate_service.shared.raster_contract import prepare_for_publication

logger = logging.getLogger(__name__)


def _resolve_download_dir() -> Path:
    return api_config.get_download_root()


DOWNLOAD_DIR = _resolve_download_dir()


# A grid larger than this gets overviews. Compared against the *grid* (nx * ny), not the
# chunk shape — a pyramid appends coarser levels beside level 0 and costs ~+33% storage
# (1 + 1/4 + 1/16), so the only question is whether a map client would benefit.
#
# 1024x1024 rather than 2048x2048: at 2048² a 1949 x 1895 store (Uganda's CLMS GPP, 3.69 Mpx)
# fell 0.88x short of the bar and rendered flat, making every frame at every zoom read the
# full grid — 32 chunk requests and 14.8 MB per timestep to fill a few hundred screen pixels.
# 1024² catches that while still skipping grids small enough to send whole.
_PYRAMID_PIXEL_THRESHOLD = 1024 * 1024
_PYRAMID_MAX_LEVELS = 8
_PYRAMID_TARGET_TILE_SIZE = 512
_ROOT_TIME_COORD_MAX_CHUNK = 4096
# Memory budget per streamed level-0 region. topozarr writes region by region from a *lazy*
# source, so this — not the store size — bounds the pyramid build. Its own default is 256 MB;
# 64 MB keeps peak well within a small on-prem box, since peak is roughly
# `max_workers * 5 * region_bytes`.
_PYRAMID_MAX_REGION_BYTES = 64 * 1024 * 1024


def _uniform_chunks(ds: xr.Dataset) -> xr.Dataset:
    """Rechunk any dim whose dask chunks are non-uniform except for the final chunk.

    Zarr requires that shape; dask does not. Reversing an axis produces exactly the illegal
    case — ``raster_contract.ensure_north_up`` flips a south-up source with
    ``isel(slice(None, None, -1))``, which reverses the chunk *tuple* too, so a legal
    trailing remainder (244, 244, …, 241) becomes a leading one (241, 244, …) and
    ``to_zarr`` refuses the write.

    Rechunking to the largest existing chunk restores the source layout, since dask fills
    uniformly and leaves the remainder last.

    Only the flat path needs this. The pyramid path is unaffected because topozarr plans its
    own chunk layout per level from ``target_chunk_bytes`` rather than inheriting the source's,
    so a reversed tuple never reaches the write. (Deliberately not "because the pyramid path
    calls ``load()``" — it does today, but that materialisation is being removed, and this
    would then read as a guarantee that no longer holds.)

    Applied per dimension across the whole dataset, so a variable with a legal layout can be
    rechunked because a sibling on the same dim was illegal. Harmless, and only when at least
    one variable needs it.
    """
    targets: dict[Any, int] = {}
    for var in ds.data_vars.values():
        for dim, chunks in zip(var.dims, var.chunks or (), strict=False):
            # Legal already when every chunk but the last matches and the last is no larger.
            # Checked rather than blanket-rechunked because rechunking a full-resolution grid
            # costs a graph rewrite and a shuffle for nothing.
            if len(set(chunks[:-1])) <= 1 and chunks[-1] <= chunks[0]:
                continue
            targets[dim] = max(targets.get(dim, 0), max(chunks))
    if not targets:
        return ds
    logger.info("Rechunking to uniform Zarr-legal chunks: %s", targets)
    return ds.chunk(targets)


def _needs_pyramid(ds: xr.Dataset, x_dim: str, y_dim: str) -> bool:
    """Return True when the spatial extent is large enough to benefit from a pyramid."""
    return ds.sizes[x_dim] * ds.sizes[y_dim] > _PYRAMID_PIXEL_THRESHOLD


def _pyramid_levels(ds: xr.Dataset, x_dim: str, y_dim: str) -> int:
    """Compute the number of pyramid levels needed to reach a manageable tile size."""
    import math

    max_dim = max(ds.sizes[x_dim], ds.sizes[y_dim])
    levels = math.ceil(math.log2(max_dim / _PYRAMID_TARGET_TILE_SIZE))
    return max(2, min(levels, _PYRAMID_MAX_LEVELS))


# Coarsening methods topozarr computes itself, each level block-reduced from the one above.
# That is valid because they are composable — `nearest` included, since corner-of-corners
# equals corner-of-native. `nearest` arrived in topozarr 0.1.3 (carbonplan/topozarr#26) and
# is now delegated rather than recomputed here.
_COMPOSABLE_METHODS = frozenset({"mean", "max", "min", "sum", "nearest"})
# Methods topozarr cannot do: non-composable, so each level must be resampled from the
# native (level-0) array rather than from an already-coarsened parent, and OCS recomputes
# those levels itself. Only `mode` remains — carbonplan/topozarr#26 is still open for it, and
# upstream's own note says such methods "must reduce from native instead". Delete this path
# and the helpers below once it lands.
_NATIVE_RESAMPLE_METHODS = frozenset({"mode"})
_RESAMPLING_METHODS = _COMPOSABLE_METHODS | _NATIVE_RESAMPLE_METHODS
_DEFAULT_RESAMPLING = "mean"


def _normalize_resampling_method(raw: object) -> str:
    """Trim/lower-case a resampling method and validate it against the supported set.

    Unknown or missing values fall back to ``mean`` (the safe default for continuous
    data) with a warning, so neither a mistyped template value nor a stray direct call
    reaches topozarr as an invalid ``CoarseningMethod``.
    """
    if raw is None:
        return _DEFAULT_RESAMPLING
    method = str(raw).strip().lower()
    if method not in _RESAMPLING_METHODS:
        logger.warning(
            "Unknown resampling method %r (expected one of %s); using %r",
            raw,
            sorted(_RESAMPLING_METHODS),
            _DEFAULT_RESAMPLING,
        )
        return _DEFAULT_RESAMPLING
    return method


def resampling_method_from_template(template: dict[str, Any] | None) -> str:
    """Pyramid coarsening method from a dataset template's ``ingestion.resampling`` field.

    It lives under ``ingestion`` rather than ``display`` because it changes the *stored*
    pyramid data (how coarse levels are aggregated), not how the layer is rendered.

    Defaults to ``mean`` — correct for continuous data (temperature, precipitation, …).
    Categorical layers should declare ``mode`` (multi-class, e.g. land-cover class codes)
    or ``max`` (binary presence masks); ``mean`` averages class codes into meaningless
    values at coarse zoom. An unrecognised value falls back to ``mean`` with a warning.
    """
    ingestion = (template or {}).get("ingestion")
    raw = ingestion.get("resampling") if isinstance(ingestion, dict) else None
    return _normalize_resampling_method(raw)


def _mode_reduce(values: Any, axis: Any = None, **_kwargs: Any) -> Any:
    """Majority value over ``axis`` — the reducer xarray's ``coarsen(...).reduce`` calls.

    ``axis`` is the tuple of window axes; flatten them and take the per-cell mode so a
    coarsened cell keeps a real class code instead of an average. NaNs are ignored.
    """
    import numpy as np
    from scipy import stats  # type: ignore[import-untyped]

    if axis is None:
        axes: tuple[int, ...] = tuple(range(values.ndim))
    elif isinstance(axis, (tuple, list)):
        axes = tuple(int(a) for a in axis)
    else:
        axes = (int(axis),)
    axes = tuple(a % values.ndim for a in axes)
    keep = [a for a in range(values.ndim) if a not in axes]
    moved = np.transpose(values, keep + list(axes))
    flat = moved.reshape(*(values.shape[a] for a in keep), -1)
    result = stats.mode(flat, axis=-1, nan_policy="omit", keepdims=False)
    return result.mode


def _coarsen_native(da: xr.DataArray, x_dim: str, y_dim: str, factor: int, method: str) -> xr.DataArray:
    """Coarsen ``da`` by ``factor`` over the spatial dims using a non-composable ``method``.

    ``boundary="trim"`` drops the trailing partial window, so the result is
    ``floor(size / factor)`` per spatial dim — the same shape topozarr produces by
    repeatedly halving, so the output drops straight into the level's existing array.

    Only ``mode`` reaches here. ``nearest`` used to as well, but topozarr 0.1.3 does it
    natively and composably, so it is delegated (see ``_COMPOSABLE_METHODS``).
    """
    if method != "mode":
        raise ValueError(f"unsupported native-resample method {method!r}")
    return da.coarsen({x_dim: factor, y_dim: factor}, boundary="trim").reduce(_mode_reduce)


def _overwrite_native_resampled_levels(
    store: Any, ds: xr.Dataset, x_dim: str, y_dim: str, levels: int, method: str
) -> None:
    """Replace topozarr's coarsened levels with a native resample for categorical data.

    topozarr block-reduces each level from the level above, which is wrong for ``mode``: a
    coarse cell's majority class must be derived from the native cells it covers, not from an
    already-coarsened parent, or a locally dominant class can win at coarse zoom while being
    globally rare. topozarr has still written every level's group/array with the right shape,
    chunking and encoding, so we recompute levels 1..N-1 straight from the native (level-0)
    arrays and write the values back in place.
    """
    root = zarr.open_group(store, mode="a")
    spatial_vars = [str(name) for name, da in ds.data_vars.items() if {x_dim, y_dim} <= set(da.dims)]
    for lvl in range(1, levels):
        factor = 2**lvl
        level_group = cast(zarr.Group, root[str(lvl)])
        for name in spatial_vars:
            da = ds[name]
            coarsened = _coarsen_native(da, x_dim, y_dim, factor, method).transpose(*da.dims)
            target = cast(zarr.Array, level_group[name])
            values = coarsened.values
            if values.shape != target.shape:
                # Defensive: clip to the level array's shape if trimming disagrees by a cell.
                values = values[tuple(slice(0, s) for s in target.shape)]
            target[:] = values.astype(target.dtype, copy=False)


def _write_root_time_coordinate(zarr_store: "Path | Any", ds: xr.Dataset, *, time_dim: str) -> None:
    """Expose the time coordinate at the pyramid root with bounded chunking for browser clients.

    zarr_store may be a filesystem Path (plain Zarr) or a zarr-compatible store object
    (e.g. an Icechunk session store).
    """
    if time_dim not in ds.dims:
        logger.debug("Skipping root time coordinate write: no %s dimension found", time_dim)
        return
    if time_dim not in ds.coords or ds.sizes.get(time_dim, 0) == 0:
        logger.debug("Skipping root time coordinate write: empty or missing %s coordinate", time_dim)
        return

    store_arg = str(zarr_store) if isinstance(zarr_store, Path) else zarr_store
    root = zarr.open_group(store_arg, mode="a", zarr_format=3)
    root_attrs = dict(root.attrs)
    if time_dim in root:
        del root[time_dim]
    time_coord = xr.Dataset(coords={time_dim: ds[time_dim]})
    time_coord.to_zarr(
        store_arg,
        mode="a",
        zarr_format=3,
        consolidated=False,
        encoding={time_dim: {"chunks": (min(ds.sizes[time_dim], _ROOT_TIME_COORD_MAX_CHUNK),)}},
    )
    root = zarr.open_group(store_arg, mode="a", zarr_format=3)
    root.attrs.update(root_attrs)


def _get_cache_prefix(dataset: dict[str, Any]) -> str:
    return str(dataset["id"])


def get_cache_files(dataset: dict[str, Any]) -> list[Path]:
    """Return all NetCDF cache files matching this dataset's prefix."""
    # TODO: not bulletproof -- e.g. 2m_temperature matches 2m_temperature_modified
    prefix = _get_cache_prefix(dataset)
    return list(DOWNLOAD_DIR.glob(f"{prefix}*.nc"))


def get_zarr_path(dataset: dict[str, Any]) -> Path | None:
    """Return the optimised zarr archive path if it exists."""
    prefix = _get_cache_prefix(dataset)
    optimized = DOWNLOAD_DIR / f"{prefix}.zarr"
    if optimized.exists():
        return optimized
    return None


def get_icechunk_path(dataset: dict[str, Any]) -> Path:
    """Return the Icechunk store path for a dataset."""
    prefix = _get_cache_prefix(dataset)
    return DOWNLOAD_DIR / f"{prefix}.icechunk"


def needs_pyramid(ds: xr.Dataset) -> bool:
    """Return True when the dataset is large enough to need a multiscale pyramid."""
    return _needs_pyramid(ds, "x", "y")


def ensure_time_coordinate_chunking(store_path: Path, time_dim: str) -> bool:
    """Re-chunk an existing Icechunk store's 1-D time coordinate to a bounded chunk size.

    Streaming ingests append one period at a time, which leaves the ``time_dim`` coordinate
    chunked at size 1. A browser map client reads the whole time axis to drive its step
    control, so a chunk-per-timestep coordinate means one tiny HTTP request per step — tens
    of thousands of them for a multi-year daily store, dominating load/step latency. Rewrite
    only the coordinate array to ``<= _ROOT_TIME_COORD_MAX_CHUNK`` values per chunk (the data
    variables, chunked at one timestep for independent per-step reads, are left untouched).

    Surgical and transactional (Icechunk commit-or-nothing), and idempotent — a no-op when
    the coordinate is already coarse enough. Returns True if it re-chunked.
    """
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    read_group = zarr.open_group(repo.readonly_session("main").store, mode="r")
    if time_dim not in read_group:
        return False
    coord = read_group[time_dim]
    if not isinstance(coord, zarr.Array) or coord.ndim != 1:
        return False
    n = int(coord.shape[0])
    target = min(n, _ROOT_TIME_COORD_MAX_CHUNK)
    if int(coord.chunks[0]) >= target:
        return False  # already coarse enough

    values = coord[:]
    attrs = dict(coord.attrs)
    attrs.pop("_ChunkSizes", None)  # stale netCDF per-chunk hint from the source
    dtype = coord.dtype
    fill_value = coord.fill_value

    session = repo.writable_session("main")
    write_group = zarr.open_group(session.store, mode="r+")
    del write_group[time_dim]
    new_coord = write_group.create_array(
        time_dim,
        shape=(n,),
        chunks=(target,),
        dtype=dtype,
        fill_value=fill_value,
        dimension_names=(time_dim,),
    )
    new_coord[:] = values
    for key, value in attrs.items():
        new_coord.attrs[key] = value
    session.commit(f"Re-chunk '{time_dim}' coordinate to <= {target}/chunk for browser-friendly axis reads")
    logger.info("Re-chunked '%s' coordinate of '%s' from 1 to %d values/chunk", time_dim, store_path.name, target)
    return True


def write_to_icechunk_store(
    ds: xr.Dataset,
    store_path: Path,
    x_dim: str = "x",
    y_dim: str = "y",
    t_dim: str | None = "t",
    *,
    crs: str | None = None,
    pyramid_method: str = "mean",
    commit_message: str = "Materialized dataset",
) -> None:
    """Write *ds* to an Icechunk store, building a multiscale pyramid when needed.

    Applies GeoZarr conventions throughout. Creates the store if it does not exist;
    overwrites any existing content in the new commit.

    Three CRS calls are required: ``proj.assign_crs`` sets the xproj CRS needed by
    topozarr, ``rio.write_crs`` populates ``spatial_ref`` attrs and encodes
    ``grid_mapping`` per variable, then ``proj.assign_crs`` again because
    ``rio.write_crs`` destroys xproj CRS detection.
    """
    import icechunk
    import rioxarray as _rxr  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio accessor

    # Bring the cube up to the published-store contract, and resolve the CRS that describes it,
    # at the write boundary — rather than relying on every plugin to get it right (CLIM-821).
    prepared = prepare_for_publication(ds, fallback_crs=crs, x_dim=x_dim, y_dim=y_dim)
    ds, crs = prepared.dataset, prepared.crs
    if t_dim is not None and t_dim != "t":
        # The caller named the temporal dim as it arrived; normalisation has renamed it, so the
        # rest of this function (root time coordinate, chunking) must follow the new name.
        t_dim = "t" if "t" in ds.dims else t_dim

    # Array order, (y, x): `spatial:dimensions` and `spatial:shape` are read positionally, so
    # naming them x-first transposes the grid for any client that trusts the convention.
    dims = [y_dim, x_dim]
    geometry = grid_geometry(ds[x_dim], ds[y_dim])
    if geometry is None:
        # Degenerate grid (a single cell on an axis) — the cell size can't be derived from
        # coordinates, so fall back to the centre-based extent and omit the affine.
        bbox = [
            float(ds[x_dim].min()),
            float(ds[y_dim].min()),
            float(ds[x_dim].max()),
            float(ds[y_dim].max()),
        ]
        shape: tuple[int, ...] = (ds.sizes[y_dim], ds.sizes[x_dim])
    else:
        bbox = geometry["bbox"]
        shape = tuple(geometry["shape"])
    geozarr_attrs = create_geozarr_attrs(dimensions=dims, crs=crs, bbox=bbox, shape=shape)
    if geometry is not None:
        # The affine is what a client places the raster with; without it, viewers infer a grid
        # from the coordinate arrays and assume EPSG:4326 while doing so, which puts every
        # projected store off the map. topozarr writes the same affine for pyramid level 0,
        # and this matches it exactly (pixel registration, so the origin is the cell edge).
        geozarr_attrs["spatial:transform"] = geometry["transform"]

    ds = ds.proj.assign_crs(spatial_ref=crs)
    ds = ds.rio.write_crs(crs)
    ds = ds.proj.assign_crs(spatial_ref=crs)

    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")

    if _needs_pyramid(ds, x_dim, y_dim):
        zarr_conventions = list(geozarr_attrs.get("zarr_conventions", []))
        zarr_conventions.append(MultiscalesConventionMetadata().model_dump())
        geozarr_attrs["zarr_conventions"] = zarr_conventions

        levels = _pyramid_levels(ds, x_dim, y_dim)
        logger.info(
            "Building %d-level pyramid into Icechunk store '%s' (dims %dx%d)",
            levels,
            store_path.name,
            ds.sizes[x_dim],
            ds.sizes[y_dim],
        )

        # Deliberately *not* `ds.load()`. topozarr streams level 0 region by region and sizes
        # those regions from `max_region_bytes`, so a lazy source is what bounds peak memory —
        # its own docs say "for bounded memory on large stores, open the source lazily".
        # Materialising first defeated that: Uganda's GPP store is 3.47 GB and growing ~0.6 GB
        # per year of dekads, which is what made a lower threshold risky on a small box.
        # Validate/normalize here too, so a direct caller passing e.g. "MAX" or a mistyped
        # value never reaches topozarr as an invalid CoarseningMethod (falls back to mean).
        pyramid_method = _normalize_resampling_method(pyramid_method)
        # topozarr only offers composable coarsening (mean/max/min/sum/nearest). For `mode`
        # it writes valid structure with a placeholder here, then we overwrite the coarsened
        # levels from native below (see #293 and carbonplan/topozarr#26).
        native_resample = pyramid_method in _NATIVE_RESAMPLE_METHODS
        composable_method = "max" if native_resample else pyramid_method
        pyramid = create_pyramid(
            ds, levels=levels, x_dim=x_dim, y_dim=y_dim, method=cast(CoarseningMethod, composable_method)
        )
        # Pyramid.write() writes the root group attributes from ``pyramid.attrs``,
        # so merge the GeoZarr conventions (multiscales / proj / spatial) in here.
        pyramid.attrs.update(geozarr_attrs)

        # Keep the no-data sentinel consistent across pyramid levels so the map
        # renders the same pixels transparent at every zoom. topozarr mean-coarsens
        # an integer-coded mask (e.g. a 0/1 hotspot raster, where 0 is the no-data
        # background) into float levels that would otherwise default to a NaN fill,
        # leaving the 0.0 background opaque at overview zooms. Pin the fill_value to
        # 0 for integer-coded variables (Pyramid.write applies it across all levels);
        # float variables already use NaN as both data and fill, so leave them be.
        for var_name, var_da in ds.data_vars.items():
            if var_da.dtype.kind != "f" and pyramid.fill_values.get(str(var_name)) is None:
                pyramid.fill_values[str(var_name)] = 0

        pyramid.write(session.store, mode="w", max_region_bytes=_PYRAMID_MAX_REGION_BYTES)

        if native_resample:
            # Categorical data: recompute the coarsened levels from native so class codes /
            # masks aren't averaged (topozarr wrote them composably above; replace in place).
            logger.info("Resampling pyramid levels of '%s' from native (%s)", store_path.name, pyramid_method)
            _overwrite_native_resampled_levels(session.store, ds, x_dim, y_dim, levels, pyramid_method)

        # topozarr demotes spatial_ref from coordinate to data variable in the pyramid.
        # Patch the root and each level group: add CRS to multiscales datasets entries so
        # ZarrLayer can detect the coordinate system without guessing (it defaults to
        # EPSG:3857 when crs is absent from the OME-NGFF datasets array).
        root = zarr.open_group(session.store, mode="a")
        root_attrs = dict(root.attrs)
        ms = root_attrs.get("multiscales")
        if isinstance(ms, list) and ms and isinstance(ms[0], dict):
            datasets = ms[0].get("datasets")
            if isinstance(datasets, list):
                for ds_entry in datasets:
                    if isinstance(ds_entry, dict):
                        ds_entry.setdefault("crs", crs)
                root_attrs["multiscales"] = ms
                root.attrs.update(root_attrs)
        if geometry is not None:
            write_gdal_geotransform(root, geometry["transform"])
        for level_key in root.keys():
            level_group = root[level_key]
            if not isinstance(level_group, zarr.Group):
                continue
            lvl_attrs = dict(level_group.attrs)
            lvl_attrs["coordinates"] = "spatial_ref"
            level_group.attrs.update(lvl_attrs)
            # Level groups are named by their downsample exponent ("0" = native). Anything
            # else is not a pyramid level, so leave its georeferencing alone.
            if geometry is not None and level_key.isdigit():
                # Each level halves the resolution, so it needs its own GeoTransform — one
                # copied from level 0 would place every overview at twice its true size.
                factor = float(2 ** int(level_key))
                step_x, rot_x, origin_x, rot_y, step_y, origin_y = geometry["transform"]
                write_gdal_geotransform(
                    level_group,
                    [step_x * factor, rot_x, origin_x, rot_y, step_y * factor, origin_y],
                )

        if t_dim is not None:
            _write_root_time_coordinate(session.store, ds, time_dim=t_dim)
    else:
        logger.info(
            "Writing flat Icechunk store '%s' (dims %dx%d)",
            store_path.name,
            ds.sizes[x_dim],
            ds.sizes[y_dim],
        )
        ds = ds.assign_attrs({**ds.attrs, **geozarr_attrs})
        ds = _uniform_chunks(ds)
        # Bound the time-coordinate chunk so map clients read the axis in a few requests
        # rather than one per timestep (data variables keep their own chunking).
        flat_encoding = {}
        if t_dim is not None and t_dim in ds.coords and ds.sizes.get(t_dim, 0) > _ROOT_TIME_COORD_MAX_CHUNK:
            flat_encoding[t_dim] = {"chunks": (_ROOT_TIME_COORD_MAX_CHUNK,)}
        ds.to_zarr(session.store, mode="w", zarr_format=3, encoding=flat_encoding or None)

        # xarray demotes scalar coordinates (like spatial_ref) to data variables in zarr v3.
        # Patch the root group so rioxarray can follow the CF grid_mapping attribute.
        root_flat = zarr.open_group(session.store, mode="a")
        flat_attrs = dict(root_flat.attrs)
        flat_attrs["coordinates"] = "spatial_ref"
        root_flat.attrs.update(flat_attrs)
        if geometry is not None:
            write_gdal_geotransform(root_flat, geometry["transform"])

    session.commit(commit_message)
