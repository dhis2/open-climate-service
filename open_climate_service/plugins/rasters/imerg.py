"""NASA GPM IMERG v07 daily precipitation from GES DISC.

Two daily IMERG runs are combined: the Final run (``GPM_3IMERGDF``, gauge-calibrated, about
3.5 months behind) for as long as it exists, then the Late run (``GPM_3IMERGDL``, satellite
only, about a day and a half behind). Each day is one global 0.1 degree NetCDF; it is
downloaded through Earthdata Login and clipped to the extent. Days are UTC days, which is
how IMERG defines them.

Needs Earthdata Login credentials in ``~/.netrc`` and the "NASA GESDISC DATA ARCHIVE"
application approved in the Earthdata profile.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.plugins.rasters.earthdata import cmr_granules, data_links, download
from open_climate_service.shared.netcdf import NETCDF_LOCK
from open_climate_service.streaming import BaseDatasetPlugin, daily_period_ids

_FINAL = "GPM_3IMERGDF"
_LATE = "GPM_3IMERGDL"
_VERSION = "07"
_HALF_GRID_STEP = 0.05


def _granule_url(short_name: str, day: date) -> str | None:
    granules = cmr_granules(short_name, (f"{day}T00:00:00Z", f"{day}T23:59:59Z"), version=_VERSION)
    for granule in granules:
        links = data_links(granule, ".nc4")
        if links and granule["time_start"][:10] == day.isoformat():
            return links[0]
    return None


def _latest_day(short_name: str) -> date | None:
    today = datetime.now(UTC).date()
    granules = cmr_granules(
        short_name, (f"{today - timedelta(days=200)}T00:00:00Z", f"{today}T23:59:59Z"), version=_VERSION
    )
    return date.fromisoformat(granules[-1]["time_start"][:10]) if granules else None


class IMERGDailyPlugin(BaseDatasetPlugin):
    """Daily precipitation (mm/day) from IMERG v07, Final run then Late run."""

    max_concurrency = 4
    commit_batch_size = 31

    def __init__(self, **_: Any) -> None:
        pass

    async def periods(self, start: str, end: str) -> list[str]:
        import asyncio

        latest = await asyncio.to_thread(_latest_day, _LATE)
        if latest is None:
            return []
        return daily_period_ids(start, end, cutoff=latest)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = date.fromisoformat(period_id)
        url = _granule_url(_FINAL, day)
        if url is None:
            url = _granule_url(_LATE, day)
        if url is None:
            raise RuntimeError(f"No IMERG v07 daily granule for {period_id}")
        with tempfile.TemporaryDirectory() as tmp:
            path = download(url, Path(tmp) / Path(url).name)
            with NETCDF_LOCK, xr.open_dataset(path, engine="netcdf4") as raw:
                xmin, ymin, xmax, ymax = map(float, bbox)
                region = raw["precipitation"].sel(
                    lon=slice(xmin - _HALF_GRID_STEP, xmax + _HALF_GRID_STEP),
                    lat=slice(ymin - _HALF_GRID_STEP, ymax + _HALF_GRID_STEP),
                )
                region = region.load()
        # IMERG stores (time, lon, lat) with latitude ascending; the store wants (t, y, x), north up.
        region = (
            region.rename({"time": "t", "lon": "x", "lat": "y"}).transpose("t", "y", "x").sortby("y", ascending=False)
        )
        region = region.assign_coords(t=[np.datetime64(period_id, "ns")])
        region = region.astype("float32")
        region.attrs = {"units": "mm/d"}
        return region.to_dataset(name="precip")
