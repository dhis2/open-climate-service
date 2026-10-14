"""Daily ERA5-Land statistics from the Copernicus CDS, for variables beyond t2m and tp.

Two plugins:

- ``ERA5LandCDSDailyStatisticPlugin`` reads the CDS ``derived-era5-land-daily-statistics``
  product, which computes the daily mean, minimum or maximum of an hourly variable on the
  CDS side, in a requested time zone. One request per calendar month.
- ``ERA5LandCDSDailyWindSpeedPlugin`` computes daily mean 10 m wind speed from hourly u and
  v components. The daily statistics product cannot do this: the speed of the mean wind
  vector understates the mean speed whenever the direction changes during the day.

Both use the CDS only, so they work with a CDS key alone (no Earth Data Hub subscription).
"""

from __future__ import annotations

import asyncio
import calendar
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.plugins.rasters.era5_land import (
    _cds_fallback_area,
    _collapse_expver,
    _daily_availability_cutoff,
    _drop_auxiliary_variables,
    _hourly_availability_cutoff,
)
from open_climate_service.shared.netcdf import load_netcdf
from open_climate_service.streaming import BaseDatasetPlugin, daily_period_ids
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

# Short name (as it appears in the downloaded NetCDF) -> CDS request name.
_DAILY_STAT_VARIABLES: dict[str, str] = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sp": "surface_pressure",
}
_KELVIN_VARIABLES = {"t2m", "d2m"}
_STATISTICS = {"mean": "daily_mean", "min": "daily_minimum", "max": "daily_maximum"}

_BBox = tuple[float, float, float, float]


def _cds_time_zone(offset: timedelta) -> str:
    """Format a UTC offset the way the CDS daily statistics form expects (``utc+01:00``)."""
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"utc{sign}{hours:02d}:{mins:02d}"


def _rename_to_canonical(ds: xr.Dataset) -> xr.Dataset:
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    return ds.rename({"longitude": "x", "latitude": "y", time_dim: "t"})


class _MonthlyCachedCDSPlugin(BaseDatasetPlugin):
    """Fetch one calendar month per CDS request and serve days out of it.

    ``max_concurrency = 1`` keeps the single-month cache free of races.
    """

    max_concurrency = 1
    commit_batch_size = 31
    variable: str

    def __init__(self) -> None:
        self._cache_lock = Lock()
        self._cached_key: tuple[int, int, _BBox] | None = None
        self._cached_ds: xr.Dataset | None = None

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        raise NotImplementedError

    def _month_cube(self, day: date, bbox: list[float]) -> xr.Dataset:
        bbox_tuple = cast(_BBox, tuple(map(float, bbox)))
        key = (day.year, day.month, bbox_tuple)
        with self._cache_lock:
            if self._cached_key != key:
                self._cached_ds = self._fetch_month(day.year, day.month, bbox_tuple)
                self._cached_key = key
            assert self._cached_ds is not None
            return self._cached_ds

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = date.fromisoformat(period_id)
        monthly = self._month_cube(day, bbox)
        timestamp = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return _drop_auxiliary_variables(monthly.sel(t=slice(timestamp, timestamp)), self.variable)


class ERA5LandCDSDailyStatisticPlugin(_MonthlyCachedCDSPlugin):
    """Daily mean, minimum or maximum of an ERA5-Land variable from CDS daily statistics.

    Days run in the instance's local time zone (``utc_offset_hours`` in
    climate-service.yaml), which the CDS applies before aggregating.

    Template params:
        variable: ERA5-Land short name, one of ``_DAILY_STAT_VARIABLES``.
        statistic: ``mean`` (default), ``min`` or ``max``.
    """

    def __init__(self, variable: str, statistic: str = "mean", **_: Any) -> None:
        super().__init__()
        if variable not in _DAILY_STAT_VARIABLES:
            expected = list(_DAILY_STAT_VARIABLES)
            raise ValueError(f"{type(self).__name__}: unsupported variable {variable!r}; expected one of {expected}")
        if statistic not in _STATISTICS:
            expected = list(_STATISTICS)
            raise ValueError(f"{type(self).__name__}: unsupported statistic {statistic!r}; expected one of {expected}")
        self.variable = variable
        self.statistic = statistic

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_daily_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff)

    def _request(self, year: int, month: int, bbox: _BBox) -> dict[str, Any]:
        from open_climate_service import config as api_config

        _, last_day = calendar.monthrange(year, month)
        cutoff = _daily_availability_cutoff()
        if (cutoff.year, cutoff.month) == (year, month):
            last_day = min(last_day, cutoff.day)
        return {
            "variable": [_DAILY_STAT_VARIABLES[self.variable]],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, last_day + 1)],
            "daily_statistic": _STATISTICS[self.statistic],
            "time_zone": _cds_time_zone(api_config.get_utc_offset()),
            "frequency": "1_hourly",
            "area": _cds_fallback_area(bbox),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_daily_stat.nc"
            _CdsClient().retrieve("derived-era5-land-daily-statistics", self._request(year, month, bbox), str(target))
            ds = load_netcdf(target)
        ds = _rename_to_canonical(_collapse_expver(ds[[self.variable]]))
        if self.variable in _KELVIN_VARIABLES:
            ds = kelvin_to_celsius(ds, {"variable": self.variable})
        return ds


class ERA5LandCDSDailyWindSpeedPlugin(_MonthlyCachedCDSPlugin):
    """Daily mean 10 m wind speed (m/s) from hourly ERA5-Land u10 and v10.

    Speed is computed per hour and then averaged over the local day, so it is the mean
    speed rather than the speed of the mean wind vector. A local day can straddle a
    month boundary, so each monthly request is widened by a day on either side.
    """

    variable = "wind_speed"

    def __init__(self, **_: Any) -> None:
        super().__init__()

    async def periods(self, start: str, end: str) -> list[str]:
        from open_climate_service import config as api_config

        cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        # Only local days whose last hour has been published.
        last_local = (cutoff.replace(tzinfo=None) + api_config.get_utc_offset() - timedelta(hours=23)).date()
        return daily_period_ids(start, end, cutoff=last_local)

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        from open_climate_service import config as api_config

        offset = api_config.get_utc_offset()
        _, last_day = calendar.monthrange(year, month)
        first_utc = datetime(year, month, 1) - offset
        last_utc = datetime(year, month, last_day) + timedelta(hours=23) - offset
        # Never ask for hours the CDS has not published; the partial last day comes out NaN
        # and is not offered by periods() anyway.
        last_utc = min(last_utc, _hourly_availability_cutoff().replace(tzinfo=None))
        hourly = self._fetch_hourly(first_utc, last_utc, bbox)
        speed = cast(xr.DataArray, np.hypot(hourly["u10"], hourly["v10"]))
        # Shift timestamps into local time, then a plain calendar-day resample is a local day.
        speed = speed.assign_coords(t=speed["t"] + np.timedelta64(int(offset.total_seconds()), "s"))
        daily = speed.resample(t="1D").mean(skipna=False)
        daily = daily.sel(t=slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"))
        daily = daily.assign_attrs(units="m s-1", long_name="10 m wind speed")
        return daily.to_dataset(name=self.variable)

    def _fetch_hourly(self, first: datetime, last: datetime, bbox: _BBox) -> xr.Dataset:
        """Download hourly u10/v10 covering ``first``..``last`` (UTC), one request per month touched."""
        cubes = []
        month_start = datetime(first.year, first.month, 1)
        while month_start <= last:
            _, ndays = calendar.monthrange(month_start.year, month_start.month)
            days = [
                d
                for d in range(1, ndays + 1)
                if first.date() <= date(month_start.year, month_start.month, d) <= last.date()
            ]
            params = {
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": str(month_start.year),
                "month": f"{month_start.month:02d}",
                "day": [f"{d:02d}" for d in days],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": _cds_fallback_area(bbox),
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "era5land_wind.nc"
                _CdsClient().retrieve("reanalysis-era5-land", params, str(target))
                raw = load_netcdf(target)[["u10", "v10"]]
            cubes.append(_rename_to_canonical(_collapse_expver(raw)))
            month_start = (month_start + timedelta(days=32)).replace(day=1)
        ds = xr.concat(cubes, dim="t") if len(cubes) > 1 else cubes[0]
        first_ns = np.datetime64(first, "h").astype("datetime64[ns]")
        last_ns = np.datetime64(last, "h").astype("datetime64[ns]")
        return cast(xr.Dataset, ds.sel(t=slice(first_ns, last_ns)))
