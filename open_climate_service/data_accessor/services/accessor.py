"""Loading raster data from downloaded files and stores into xarray."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pyproj import Transformer

from ...data_manager.services.downloader import get_cache_files, get_icechunk_path, get_zarr_path
from ...data_manager.services.utils import get_time_dim, get_x_y_dims
from ...shared.crs import dataset_crs
from ...shared.time import numpy_datetime_to_period_string

logger = logging.getLogger(__name__)


class DatasetDataUnavailable(OSError):
    """Nothing has been ingested for this dataset, so there is no data to read or describe.

    Distinct from a store that exists but cannot be opened. A dataset template is a
    *declaration* — it exists whether or not anyone has ingested it — so "never ingested" is an
    ordinary state, while an unreadable store is a fault. Code that *describes* a dataset treats
    this as "no coverage"; code that *serves* data still fails, which is correct.

    Subclasses OSError because that is what `open_mfdataset` raised before this existed, so a
    caller already handling that keeps working.
    """


def get_data(
    dataset: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
    bbox: list[float] | None = None,
) -> xr.Dataset:
    """Load an xarray raster dataset for a given time range and bbox."""
    logger.info("Opening dataset")
    icechunk_path = get_icechunk_path(dataset)
    if icechunk_path.exists():
        logger.info("Using Icechunk-backed store: %s", icechunk_path)
        ds = open_icechunk_dataset(icechunk_path)
    else:
        zarr_path = get_zarr_path(dataset)
        if zarr_path:
            logger.info(f"Using optimized zarr file: {zarr_path}")
            ds = open_zarr_dataset(str(zarr_path))
        else:
            files = get_cache_files(dataset)
            if not files:
                # The last of the three sources, so nothing has been ingested. Raised rather
                # than left to `open_mfdataset`, whose "no files to open" gave no dataset id
                # and could not be told apart from a genuinely broken store (CLIM-897).
                raise DatasetDataUnavailable(
                    f"Dataset '{dataset['id']}' has no ingested data: no Icechunk store, no Zarr "
                    f"archive and no cached NetCDF files. Ingest it before reading it."
                )
            # Logged only once files are known to exist. Before, every never-ingested dataset
            # warned about falling back to NetCDF when there was no NetCDF either — on a fresh
            # instance that is one misleading warning per template.
            logger.warning(
                f"Could not find optimized zarr file for dataset {dataset['id']}, using slower netcdf files instead."
            )
            ds = xr.open_mfdataset(
                files,
                data_vars="minimal",
                coords="minimal",  # pyright: ignore[reportArgumentType]
                compat="override",
            )

    if start and end:
        logger.info(f"Subsetting time to {start} and {end}")
        time_dim = get_time_dim(ds)
        ds = ds.sel(**{time_dim: slice(start, end)})  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    if bbox is not None:
        logger.info(f"Subsetting xy to {bbox}")
        xmin, ymin, xmax, ymax = list(map(float, bbox))
        x_dim, y_dim = get_x_y_dims(ds)
        # TODO: this assumes y axis increases towards north and is not very stable
        # ...and also does not consider partial pixels at the edges
        # ...should probably switch to rioxarray.clip instead
        ds = ds.sel(**{x_dim: slice(xmin, xmax), y_dim: slice(ymax, ymin)})  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    return ds


def get_data_coverage(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return temporal and spatial coverage metadata for a dataset's ingested data.

    A dataset with nothing ingested reports `has_data: False` rather than raising, using the
    same shape as a store whose dimensions are empty — both mean "there is nothing to
    describe", and a caller describing a template should not have to tell them apart. Only the
    *reading* paths still fail on absent data (CLIM-897).
    """
    try:
        ds = get_data(dataset)
    except DatasetDataUnavailable:
        return _empty_coverage()
    try:
        return _coverage_from_dataset(ds=ds, period_type=str(dataset["period_type"]), native_crs=dataset_crs(ds))
    finally:
        ds.close()


def get_data_coverage_for_paths(
    dataset: dict[str, Any],
    *,
    zarr_path: str | None = None,
    icechunk_path: str | None = None,
    netcdf_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return coverage metadata for the concrete files created for one artifact."""
    provided = sum(value is not None for value in (zarr_path, icechunk_path)) + int(bool(netcdf_paths))
    if provided != 1:
        raise ValueError("Coverage calculation requires exactly one artifact source")

    if icechunk_path is not None:
        ds = open_icechunk_dataset(icechunk_path)
    elif zarr_path is not None:
        ds = open_zarr_dataset(zarr_path)
    else:
        assert netcdf_paths is not None
        ds = xr.open_mfdataset(
            netcdf_paths,
            data_vars="minimal",
            coords="minimal",  # pyright: ignore[reportArgumentType]
            compat="override",
        )

    try:
        return _coverage_from_dataset(ds=ds, period_type=str(dataset["period_type"]), native_crs=dataset_crs(ds))
    finally:
        ds.close()


def open_zarr_dataset(zarr_path: str) -> xr.Dataset:
    """Open a zarr store, handling pyramid stores by opening the base resolution level.

    When the root group has no data variables (as in a multiscale pyramid store),
    attempts to open level 0 instead. This works for any store backend — local
    paths, S3 URIs, GCS URIs, fsspec mappers — without filesystem-specific checks.
    """
    ds = _open_zarr(zarr_path)
    if not ds.data_vars:
        level0_path = zarr_path.rstrip("/") + "/0"
        try:
            level0 = _open_zarr(level0_path)
        except Exception as exc:
            ds.close()
            raise ValueError(
                f"Zarr store at {zarr_path!r} has no data variables at the root "
                f"and base pyramid level {level0_path!r} could not be opened"
            ) from exc
        ds.close()
        ds = level0
    return ds


def open_icechunk_dataset(store_path: str | Path) -> xr.Dataset:
    """Open an Icechunk-backed dataset through a readonly repository session."""
    import icechunk

    path = Path(store_path)
    if not path.exists():
        raise FileNotFoundError(f"Icechunk store not found: {path}")
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds: xr.Dataset = xr.open_zarr(session.store, zarr_format=3)
    if not ds.data_vars:
        try:
            level0: xr.Dataset = xr.open_zarr(session.store, group="0", zarr_format=3)
        except Exception as exc:
            ds.close()
            raise ValueError(
                f"Icechunk store at {path!r} has no data variables at the root "
                "and base pyramid level '0' could not be opened"
            ) from exc
        ds.close()
        ds = level0
    try:
        t_dim = get_time_dim(ds)
        ds = ds.sortby(t_dim)
    except ValueError:
        pass
    return ds


def _open_zarr(zarr_path: str) -> xr.Dataset:
    """Open a zarr store with automatic consolidated metadata detection."""
    return xr.open_zarr(zarr_path, consolidated=None)  # type: ignore[no-any-return]


def _empty_coverage() -> dict[str, Any]:
    """The coverage payload for a dataset with nothing to describe.

    Null fields rather than absent ones, so a client can tell "not ingested" from "this build
    does not report coverage". `has_data` is the flag to branch on.
    """
    return {
        "has_data": False,
        "coverage": {
            "temporal": {"start": None, "end": None},
            "spatial": {"xmin": None, "ymin": None, "xmax": None, "ymax": None},
            "spatial_wgs84": None,
        },
    }


def _coverage_from_dataset(*, ds: xr.Dataset, period_type: str, native_crs: str = "EPSG:4326") -> dict[str, Any]:
    """Summarize temporal and spatial coverage for an already opened dataset."""
    if any(size == 0 for size in ds.sizes.values()):
        return _empty_coverage()

    x_dim, y_dim = get_x_y_dims(ds)

    # A non-temporal (ordinal) dataset — e.g. a day-of-year climatology has a
    # `dayofyear` axis, not a datetime one — has no temporal coverage. Report
    # spatial coverage only, mirroring how the STAC builder treats these stores.
    start: str | None
    end: str | None
    try:
        time_dim = get_time_dim(ds)
    except ValueError:
        # No datetime axis — a non-temporal (ordinal) dataset, e.g. a day-of-year
        # climatology — so there is no temporal coverage. (Scoped to get_time_dim only,
        # so a genuine period-string conversion error below still surfaces.)
        start = end = None
    else:
        start = _period_string_scalar(numpy_datetime_to_period_string(ds[time_dim].min(), period_type))  # type: ignore[arg-type]
        end = _period_string_scalar(numpy_datetime_to_period_string(ds[time_dim].max(), period_type))  # type: ignore[arg-type]

    xmin, xmax = ds[x_dim].min().item(), ds[x_dim].max().item()
    ymin, ymax = ds[y_dim].min().item(), ds[y_dim].max().item()

    spatial_wgs84 = None
    wgs84 = "EPSG:4326"
    if native_crs.upper() != wgs84:
        transformer = Transformer.from_crs(native_crs, wgs84, always_xy=True)
        lon_min, lat_min, lon_max, lat_max = transformer.transform_bounds(xmin, ymin, xmax, ymax)
        spatial_wgs84 = {"xmin": lon_min, "ymin": lat_min, "xmax": lon_max, "ymax": lat_max}

    return {
        "has_data": True,
        "coverage": {
            "temporal": {"start": start, "end": end},
            "spatial": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "spatial_wgs84": spatial_wgs84,
        },
    }


def _period_string_scalar(value: Any) -> str:
    """Normalize a numpy scalar or 0-d array period string to plain Python str."""
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)


def xarray_to_temporary_netcdf(ds: xr.Dataset) -> str:
    """Write a dataset to a temporary NetCDF file and return the path."""
    fd = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    path = fd.name
    fd.close()
    ds.to_netcdf(path)
    return path


def cleanup_file(path: str) -> None:
    """Remove a file from disk."""
    os.remove(path)
