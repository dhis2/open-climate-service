"""Reading and writing netCDF files from threads.

netCDF4 and the HDF5 library it links are not thread-safe, and OCS runs netCDF I/O in threads:
ingest fetches (up to a plugin's ``max_concurrency`` per job, and several jobs at once), openEO
jobs saving a netCDF result, and the download endpoint. Two of them at the same time can corrupt
HDF5's state and crash the whole process with a segmentation fault, so every netCDF read and
write goes through ``NETCDF_LOCK``: a whole read or write at a time, which xarray's own
per-call lock does not give.

A stopgap: xarray's fix for this (pydata/xarray#11629) is merged but not yet released. Once a
release includes it, raise the xarray minimum to it and remove this module.
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dask.config
import xarray as xr

NETCDF_LOCK = threading.Lock()


def load_netcdf(path: str | Path) -> xr.Dataset:
    """Read a netCDF file into memory under the process-wide lock."""
    with NETCDF_LOCK, xr.open_dataset(path, engine="netcdf4") as ds:
        return ds.load()


def load_netcdfs(paths: Sequence[str | Path], **kwargs: Any) -> xr.Dataset:
    """Read several netCDF files as one dataset under the lock, without dask reading them in parallel."""
    with NETCDF_LOCK, xr.open_mfdataset(paths, engine="netcdf4", **kwargs) as ds:
        return ds.load(scheduler="synchronous")


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    """Write a dataset to a netCDF file under the process-wide lock, computing it in this thread."""
    with NETCDF_LOCK, dask.config.set(scheduler="synchronous"):
        ds.to_netcdf(path)
