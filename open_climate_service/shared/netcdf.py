"""Reading and writing netCDF files from threads.

netCDF4 and the HDF5 library it links are not thread-safe, and OCS runs netCDF I/O in threads:
ingest fetches (up to a plugin's ``max_concurrency`` per job, and several jobs at once), openEO
jobs saving a netCDF result, and the download endpoint. Two of them at the same time can corrupt
HDF5's state and crash the whole process with a segmentation fault, so every netCDF read and
write goes through ``NETCDF_LOCK``: a whole read or write at a time, which xarray's own
per-call lock does not give.

The lock is re-entrant. A write of a lazily opened netCDF dataset holds it while reading the
source files, which take it again; with dask's synchronous scheduler those reads run in the same
thread, so they enter the lock rather than wait on it forever.
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dask.config
import xarray as xr

NETCDF_LOCK = threading.RLock()


def load_netcdf(path: str | Path) -> xr.Dataset:
    """Read a netCDF file into memory under the process-wide lock."""
    with NETCDF_LOCK, xr.open_dataset(path, engine="netcdf4") as ds:
        return ds.load()


def load_netcdfs(paths: Sequence[str | Path], **kwargs: Any) -> xr.Dataset:
    """Read several netCDF files as one dataset under the lock, without dask reading them in parallel."""
    with NETCDF_LOCK, xr.open_mfdataset(paths, engine="netcdf4", **kwargs) as ds:
        return ds.load(scheduler="synchronous")


def open_netcdfs(paths: Sequence[str | Path], **kwargs: Any) -> xr.Dataset:
    """Open several netCDF files as one lazy dataset whose reads take the process-wide lock.

    For files too large to load at once. Opening reads every file's metadata, so it happens under
    ``NETCDF_LOCK``; so does each read xarray makes from the files later, whenever the dataset is
    computed, which takes this lock instead of xarray's own; and so does closing them, since
    xarray's file manager closes under a private lock of its own.
    """
    with NETCDF_LOCK:
        ds = xr.open_mfdataset(paths, lock=NETCDF_LOCK, **kwargs)
    close = ds._close

    def locked_close() -> None:
        with NETCDF_LOCK:
            if close is not None:
                close()

    ds.set_close(locked_close)
    return ds


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    """Write a dataset to a netCDF file under the process-wide lock.

    Dask's synchronous scheduler computes a lazy dataset in this thread, so a netCDF source it
    reads from takes the re-entrant lock this thread already holds, rather than deadlocking.
    """
    with NETCDF_LOCK, dask.config.set(scheduler="synchronous"):
        ds.to_netcdf(path)
