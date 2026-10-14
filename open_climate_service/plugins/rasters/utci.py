"""Universal Thermal Climate Index (UTCI) from ERA5-HEAT on the Copernicus CDS.

ERA5-HEAT (``derived-utci-historical``) publishes hourly UTCI at 0.25 degrees from 1940. This
plugin downloads one month of hourly UTCI per request and reduces it to a daily statistic
over the instance's local day (``utc_offset_hours``): the daily maximum is the usual heat
stress indicator, the mean is offered too.

The CDS splits the record into a ``consolidated_dataset`` (final, a few months behind) and an
``intermediate_dataset`` (preliminary, a few days behind). Consolidated data is requested
first and the intermediate product fills the months it does not yet cover.
"""

from __future__ import annotations

import asyncio
import calendar
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.plugins.rasters.era5_land_daily_stats import _MonthlyCachedCDSPlugin
from open_climate_service.shared.netcdf import load_netcdf, load_netcdfs
from open_climate_service.streaming import daily_period_ids
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

_COLLECTION = "derived-utci-historical"
_CDS_CATALOGUE_URL = "https://cds.climate.copernicus.eu/api"
_STATISTICS = ("mean", "max", "min")
_HALF_GRID_STEP = 0.125

_BBox = tuple[float, float, float, float]


def _availability_cutoff() -> datetime:
    """Latest hour published (intermediate product included)."""
    end = _CdsClient(url=_CDS_CATALOGUE_URL, key="").get_collection(_COLLECTION).end_datetime
    if end is None:
        raise RuntimeError(f"CDS collection '{_COLLECTION}' returned no end_datetime")
    return end.replace(tzinfo=None)


def _open_download(path: Path, workdir: Path) -> xr.Dataset:
    """Open a CDS download that is either a single NetCDF or a zip of per-day NetCDFs."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(workdir)
        files = sorted(p for p in workdir.rglob("*.nc"))
        return load_netcdfs(files, combine="by_coords")
    return load_netcdf(path)


class UTCIDailyPlugin(_MonthlyCachedCDSPlugin):
    """Daily UTCI statistic (degC) over the local day, from hourly ERA5-HEAT.

    Template params:
        statistic: ``max`` (default), ``mean`` or ``min``.
    """

    variable = "utci"

    def __init__(self, statistic: str = "max", **_: Any) -> None:
        super().__init__()
        if statistic not in _STATISTICS:
            raise ValueError(f"{type(self).__name__}: unsupported statistic {statistic!r}; expected {_STATISTICS}")
        self.statistic = statistic

    async def periods(self, start: str, end: str) -> list[str]:
        from open_climate_service import config as api_config

        cutoff = await asyncio.to_thread(_availability_cutoff)
        last_local = (cutoff + api_config.get_utc_offset() - timedelta(hours=23)).date()
        return daily_period_ids(start, end, cutoff=last_local)

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        from open_climate_service import config as api_config

        offset = api_config.get_utc_offset()
        _, last_day = calendar.monthrange(year, month)
        first_utc = datetime(year, month, 1) - offset
        last_utc = min(datetime(year, month, last_day, 23) - offset, _availability_cutoff())
        hourly = self._fetch_hourly(first_utc, last_utc, bbox)["utci"]
        local = hourly.assign_coords(t=hourly["t"] + np.timedelta64(int(offset.total_seconds()), "s"))
        resampled = local.resample(t="1D")
        daily = {"mean": resampled.mean, "max": resampled.max, "min": resampled.min}[self.statistic](skipna=False)
        daily = daily.sel(t=slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"))
        ds = daily.to_dataset(name="utci")
        # ERA5-HEAT publishes UTCI in kelvin.
        return kelvin_to_celsius(ds, {"variable": "utci"})

    def _fetch_hourly(self, first: datetime, last: datetime, bbox: _BBox) -> xr.Dataset:
        cubes = []
        month_start = datetime(first.year, first.month, 1)
        while month_start <= last:
            _, ndays = calendar.monthrange(month_start.year, month_start.month)
            month_days = [date(month_start.year, month_start.month, d) for d in range(1, ndays + 1)]
            days = [d.day for d in month_days if first.date() <= d <= last.date()]
            cubes.append(self._retrieve_month(month_start.year, month_start.month, days, bbox))
            month_start = (month_start + timedelta(days=32)).replace(day=1)
        ds = xr.concat(cubes, dim="t") if len(cubes) > 1 else cubes[0]
        first_ns = np.datetime64(first, "h").astype("datetime64[ns]")
        last_ns = np.datetime64(last, "h").astype("datetime64[ns]")
        return cast(xr.Dataset, ds.sortby("t").sel(t=slice(first_ns, last_ns)))

    def _retrieve_month(self, year: int, month: int, days: list[int], bbox: _BBox) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        request: dict[str, Any] = {
            "variable": ["universal_thermal_climate_index"],
            "version": "1_1",
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in days],
            "area": [ymax + _HALF_GRID_STEP, xmin - _HALF_GRID_STEP, ymin - _HALF_GRID_STEP, xmax + _HALF_GRID_STEP],
        }
        last_error: Exception | None = None
        for product_type in ("consolidated_dataset", "intermediate_dataset"):
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "utci.download"
                try:
                    _CdsClient().retrieve(_COLLECTION, {**request, "product_type": product_type}, str(target))
                except Exception as exc:  # the consolidated product does not cover recent months
                    last_error = exc
                    continue
                ds = _open_download(target, Path(tmp) / "unpacked")
            name = next(v for v in ds.data_vars if str(v).lower() in ("utci", "universal_thermal_climate_index"))
            ds = ds[[name]].rename({name: "utci"})
            time_dim = next(d for d in ("valid_time", "time") if d in ds.dims)
            x_dim = "longitude" if "longitude" in ds.dims else "lon"
            y_dim = "latitude" if "latitude" in ds.dims else "lat"
            ds = ds.rename({time_dim: "t", x_dim: "x", y_dim: "y"})
            return ds.drop_vars([c for c in ds.coords if c not in ("t", "x", "y")])
        raise RuntimeError(f"UTCI {year}-{month:02d} not available from the CDS") from last_error
