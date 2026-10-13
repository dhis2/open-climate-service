"""GloFAS river discharge reanalysis from the CEMS Early Warning Data Store (EWDS).

``cems-glofas-historical`` holds daily LISFLOOD simulations at 0.05 degrees from 1979 to a few
days ago: ``dis24`` is the mean discharge (m3/s) over the 24 hours ending at the valid time.
Discharge is only meaningful on the river network; off-river cells hold small values, so
averaging the grid over a district is not a measure of river flow. Use a per-district river
cell lookup to extract it (see the IMACS table in the request) rather than a zonal mean.

The EWDS splits the record into ``consolidated`` and ``intermediate`` products; consolidated is
requested first and intermediate fills the recent months it does not cover yet.
"""

from __future__ import annotations

import asyncio
import calendar
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _Client

from open_climate_service.plugins.rasters.era5_land_daily_stats import _MonthlyCachedCDSPlugin
from open_climate_service.streaming import daily_period_ids

_EWDS_URL = "https://ewds.climate.copernicus.eu/api"
_COLLECTION = "cems-glofas-historical"
_HALF_GRID_STEP = 0.025
_VERSIONS = {"4": "version_4_0", "5": "version_5_0"}

_BBox = tuple[float, float, float, float]


def _availability_cutoff() -> date:
    end = _Client(url=_EWDS_URL, key="").get_collection(_COLLECTION).end_datetime
    if end is None:
        raise RuntimeError(f"EWDS collection '{_COLLECTION}' returned no end_datetime")
    return end.date()


def _open_download(path: Path, workdir: Path) -> xr.Dataset:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(workdir)
        files = sorted(workdir.rglob("*.nc"))
        return xr.open_mfdataset(files, combine="by_coords", engine="netcdf4").load()
    return xr.open_dataset(path, engine="netcdf4").load()


class GloFASDailyDischargePlugin(_MonthlyCachedCDSPlugin):
    """Daily mean river discharge (m3/s) from the GloFAS reanalysis.

    Template params:
        version: GloFAS system version, ``"4"`` (default) or ``"5"``.
    """

    variable = "dis24"

    def __init__(self, version: str = "4", **_: Any) -> None:
        super().__init__()
        if str(version) not in _VERSIONS:
            raise ValueError(f"{type(self).__name__}: unsupported version {version!r}; expected {list(_VERSIONS)}")
        self.system_version = _VERSIONS[str(version)]

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff)

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        _, last_day = calendar.monthrange(year, month)
        cutoff = _availability_cutoff()
        if (cutoff.year, cutoff.month) == (year, month):
            last_day = min(last_day, cutoff.day)
        xmin, ymin, xmax, ymax = bbox
        request: dict[str, Any] = {
            "system_version": [self.system_version],
            "hydrological_model": ["lisflood"],
            "timespan": ["time_mean"],
            "variable": ["average_river_discharge_in_the_last_24_hours"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, last_day + 1)],
            "area": [ymax + _HALF_GRID_STEP, xmin - _HALF_GRID_STEP, ymin - _HALF_GRID_STEP, xmax + _HALF_GRID_STEP],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        last_error: Exception | None = None
        for product_type in ("consolidated", "intermediate"):
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "glofas.download"
                try:
                    _Client(url=_EWDS_URL).retrieve(
                        _COLLECTION, {**request, "product_type": [product_type]}, str(target)
                    )
                except Exception as exc:  # consolidated does not cover the most recent months
                    last_error = exc
                    continue
                ds = _open_download(target, Path(tmp) / "unpacked")
            return self._normalise(ds)
        raise RuntimeError(f"GloFAS {year}-{month:02d} not available from the EWDS") from last_error

    def _normalise(self, ds: xr.Dataset) -> xr.Dataset:
        # The EWDS NetCDF names the 24-hour mean discharge ``avg_dis`` (``dis24`` in GRIB).
        name = "avg_dis" if "avg_dis" in ds else "dis24"
        ds = ds[[name]].rename({name: "dis24"})
        time_dim = next(d for d in ("valid_time", "time") if d in ds.dims)
        ds = ds.rename({time_dim: "t", "longitude": "x", "latitude": "y"})
        ds = ds.drop_vars([c for c in ds.coords if c not in ("t", "x", "y")])
        # Each value is stamped 00 UTC at the end of the 24 hours it averages (requesting
        # 1 August returns 2 August 00:00), so label it with the day it describes.
        days = ds["t"].values.astype("datetime64[D]") - np.timedelta64(1, "D")
        ds = ds.assign_coords(t=days.astype("datetime64[ns]"))
        ds["dis24"] = ds["dis24"].astype("float32").assign_attrs(units="m3 s-1")
        return ds
