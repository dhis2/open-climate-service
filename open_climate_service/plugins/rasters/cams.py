"""CAMS global atmospheric composition (dust AOD, PM10) from the Copernicus Atmosphere Data Store.

History comes from the EAC4 reanalysis (``cams-global-reanalysis-eac4``, 3-hourly, 0.75 degrees,
2003 to about a year ago). Days after the end of EAC4 come from the 00 UTC run of the CAMS
global forecasts (``cams-global-atmospheric-composition-forecasts``, 0.4 degrees), lead times
0 to 21 hours, i.e. the analysis-adjacent part of each day's forecast. Forecast fields are
interpolated bilinearly onto the EAC4 grid so the whole series shares one grid.

The two are different systems (EAC4 assimilates a fixed set of observations at lower
resolution), so expect a step in the series where they meet.

Both data stores take the same ECMWF personal access token as the CDS, but each has its own
site terms and dataset licences that have to be accepted once in the browser.
"""

from __future__ import annotations

import asyncio
import calendar
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _Client

from open_climate_service.plugins.rasters.era5_land_daily_stats import _MonthlyCachedCDSPlugin
from open_climate_service.shared.netcdf import load_netcdf
from open_climate_service.streaming import daily_period_ids

_ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
_EAC4 = "cams-global-reanalysis-eac4"
_FORECASTS = "cams-global-atmospheric-composition-forecasts"
_GRID_STEP = 0.75
_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# OCS variable -> (ADS request name, NetCDF short name, scale to OCS units, units)
_VARIABLES: dict[str, tuple[str, str, float, str]] = {
    "duaod550": ("dust_aerosol_optical_depth_550nm", "duaod550", 1.0, "1"),
    "pm10": ("particulate_matter_10um", "pm10", 1e9, "ug m-3"),  # kg m-3 -> ug m-3
    "pm2p5": ("particulate_matter_2.5um", "pm2p5", 1e9, "ug m-3"),
}

_BBox = tuple[float, float, float, float]


def _ads_client() -> _Client:
    """A data store client for the ADS, reusing the CDS personal access token."""
    return _Client(url=_ADS_URL)


def _collection_end(collection: str) -> date:
    end = _Client(url=_ADS_URL, key="").get_collection(collection).end_datetime
    if end is None:
        raise RuntimeError(f"ADS collection '{collection}' returned no end_datetime")
    return end.date()


def _target_axes(bbox: _BBox) -> tuple[np.ndarray, np.ndarray]:
    """EAC4 grid points (multiples of 0.75 degrees) covering ``bbox`` padded by one step."""
    xmin, ymin, xmax, ymax = bbox
    x = np.arange(np.floor(xmin / _GRID_STEP) - 1, np.ceil(xmax / _GRID_STEP) + 2) * _GRID_STEP
    y = np.arange(np.ceil(ymax / _GRID_STEP) + 1, np.floor(ymin / _GRID_STEP) - 2, -1) * _GRID_STEP
    return x, y


def _area(bbox: _BBox) -> list[float]:
    x, y = _target_axes(bbox)
    # Pad by a further step so bilinear interpolation of the forecast grid has neighbours.
    return [float(y[0]) + _GRID_STEP, float(x[0]) - _GRID_STEP, float(y[-1]) - _GRID_STEP, float(x[-1]) + _GRID_STEP]


def _open_download(path: Path, workdir: Path) -> xr.Dataset:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(workdir)
        files = sorted(workdir.rglob("*.nc"))
        return xr.merge([load_netcdf(f) for f in files], compat="override")
    return load_netcdf(path)


def _to_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """Flatten forecast reference time x lead time into one ``t`` axis of valid times."""
    if "forecast_period" in ds.dims and "forecast_reference_time" in ds.dims:
        stacked = ds.stack(t=("forecast_reference_time", "forecast_period"))
        valid = stacked["forecast_reference_time"].values + stacked["forecast_period"].values
        stacked = stacked.drop_vars(["t", "forecast_reference_time", "forecast_period"]).assign_coords(t=valid)
        return stacked.drop_vars([c for c in stacked.coords if c not in ("t", "latitude", "longitude")])
    time_dim = next(d for d in ("valid_time", "time") if d in ds.dims)
    ds = ds.rename({time_dim: "t"})
    return ds.drop_vars([c for c in ds.coords if c not in ("t", "latitude", "longitude")])


class CAMSDailyMeanPlugin(_MonthlyCachedCDSPlugin):
    """Daily mean of a CAMS surface field over the local day, EAC4 then forecasts.

    Template params:
        variable: ``duaod550`` (dust aerosol optical depth at 550 nm), ``pm10`` or ``pm2p5``.
    """

    commit_batch_size = 31

    def __init__(self, variable: str, **_: Any) -> None:
        super().__init__()
        if variable not in _VARIABLES:
            raise ValueError(f"{type(self).__name__}: unsupported variable {variable!r}; expected {list(_VARIABLES)}")
        self.variable = variable

    async def periods(self, start: str, end: str) -> list[str]:
        # The forecast record runs to today; drop today itself, whose local day is incomplete.
        cutoff = await asyncio.to_thread(_collection_end, _FORECASTS)
        return daily_period_ids(start, end, cutoff=cutoff - timedelta(days=1))

    def _fetch_month(self, year: int, month: int, bbox: _BBox) -> xr.Dataset:
        from open_climate_service import config as api_config

        offset = api_config.get_utc_offset()
        _, last_day = calendar.monthrange(year, month)
        # A local day can reach into the neighbouring UTC days.
        first = date(year, month, 1) - timedelta(days=1)
        last = date(year, month, last_day) + timedelta(days=1)
        forecast_end = _collection_end(_FORECASTS)
        last = min(last, forecast_end)
        eac4_end = _collection_end(_EAC4)

        target_x, target_y = _target_axes(bbox)
        cubes = []
        if first <= eac4_end:
            eac4 = self._retrieve(_EAC4, first, min(last, eac4_end), bbox)
            cubes.append(eac4.sel(x=target_x, y=target_y, method="nearest").assign_coords(x=target_x, y=target_y))
        if last > eac4_end:
            fc = self._retrieve(_FORECASTS, max(first, eac4_end + timedelta(days=1)), last, bbox)
            cubes.append(fc.interp(x=target_x, y=target_y, method="linear"))
        hourly = xr.concat(cubes, dim="t") if len(cubes) > 1 else cubes[0]

        local = hourly.assign_coords(t=hourly["t"] + np.timedelta64(int(offset.total_seconds()), "s"))
        daily = local.resample(t="1D").mean()
        daily = daily.sel(t=slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"))
        _, _, scale, units = _VARIABLES[self.variable]
        out = (daily[self.variable] * scale).astype("float32").assign_attrs(units=units)
        return out.to_dataset(name=self.variable)

    def _retrieve(self, collection: str, first: date, last: date, bbox: _BBox) -> xr.Dataset:
        request_name, short_name, _, _ = _VARIABLES[self.variable]
        request: dict[str, Any] = {
            "variable": [request_name],
            "date": f"{first.isoformat()}/{last.isoformat()}",
            "area": _area(bbox),
            "data_format": "netcdf_zip",
        }
        if collection == _EAC4:
            request["time"] = [f"{h:02d}:00" for h in _HOURS]
        else:
            request |= {"time": ["00:00"], "leadtime_hour": [str(h) for h in _HOURS], "type": ["forecast"]}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cams.download"
            _ads_client().retrieve(collection, request, str(target))
            ds = _open_download(target, Path(tmp) / "unpacked")
        ds = _to_valid_time(ds[[short_name]].rename({short_name: self.variable}))
        ds = ds.rename({"longitude": "x", "latitude": "y"})
        ds = ds.assign_coords(x=((ds["x"] + 180) % 360) - 180).sortby("x").sortby("y", ascending=False)
        start = np.datetime64(datetime.combine(first, datetime.min.time()), "ns")
        end = np.datetime64(datetime.combine(last, datetime.max.time()), "ns")
        return ds.sortby("t").sel(t=slice(start, end))
