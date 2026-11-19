"""CHIRPS3 daily and monthly precipitation, read from dynamical.org's Icechunk Zarr mirror.

dynamical.org republishes the CHC final (``rnl``) and preliminary (``sat``) daily products as
Icechunk stores on AWS Open Data, updated daily, with no lag behind CHC. A year of an extent is
one chunk read per 3 degree tile, where reading CHC's GeoTIFFs took dozens of range requests per
day and ran into CHC's concurrent-request limit (CLIM-1946). See docs/built_in_datasets.md.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import numpy as np
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr

from open_geo_service.shared.raster_contract import normalize_y_direction
from open_geo_service.shared.time import daily_period_ids, monthly_period_ids, normalize_period_string
from open_geo_service.shared.zarr import DEFAULT_MAX_CACHE_BYTES, CachedTimeSeriesZarr
from open_geo_service.streaming import BaseDatasetPlugin, normalize_period

_STORE_URLS = {
    "final": "https://dynamical-ucsb-chc-chirps.s3.us-west-2.amazonaws.com/ucsb-chc-chirps-analysis-final/v0.1.0.icechunk",
    "prelim": "https://dynamical-ucsb-chc-chirps.s3.us-west-2.amazonaws.com/ucsb-chc-chirps-analysis-preliminary/v0.1.0.icechunk",
}
_SOURCE_VARIABLE = "precipitation_surface"
_SOURCE_TIME_DIM = "time"
_VARIABLE = "precip"
# The source is a rate in kg m-2 s-1, which is mm/s.
_SECONDS_PER_DAY = 86400


class CHIRPS3DailyPlugin(BaseDatasetPlugin):
    """Streaming plugin for CHIRPS3 daily precipitation, stored as mm/d.

    Args:
        stage: ``final`` or ``prelim`` CHIRPS3 release stage.
        store_url: the Icechunk store to read; defaults to dynamical.org's store for ``stage``.
        max_cache_bytes: bound on the time window kept in memory between periods.
    """

    # One chunk read covers a year of the extent, so a second worker would only re-read it.
    max_concurrency = 1
    commit_batch_size = 30

    def __init__(
        self,
        stage: str = "final",
        store_url: str | None = None,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        **_: Any,
    ) -> None:
        if stage not in _STORE_URLS:
            raise ValueError(f"stage must be 'final' or 'prelim', got {stage!r}")
        self.stage = stage
        self._source: xr.Dataset | None = None
        # The cutoff and every read come from one open store, so they see the same snapshot.
        self._zarr = CachedTimeSeriesZarr(
            store_url or _STORE_URLS[stage],
            _SOURCE_VARIABLE,
            time_dim=_SOURCE_TIME_DIM,
            opener=lambda _url: self._open_source(),
            max_cache_bytes=max_cache_bytes,
        )

    async def periods(self, start: str, end: str) -> list[str]:
        """Return ordered daily periods, clamped to the newest day in the store."""
        cutoff = await asyncio.to_thread(self._latest_day)
        return daily_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetch one day, clipped to the bbox."""
        day = np.datetime64(date.fromisoformat(period_id), "ns")
        return self._read_days(day, day + np.timedelta64(1, "D"), bbox)

    def _read_days(self, start: np.datetime64, end: np.datetime64, bbox: list[float]) -> xr.Dataset:
        """The days in ``[start, end)`` over the bbox, in mm/d, with time on ``t``."""
        ds = self._zarr.slice_time(start, end, prepare=lambda source: self._prepare(source, bbox))
        ds = ds.rename({_SOURCE_VARIABLE: _VARIABLE})
        expected = int((end - start) // np.timedelta64(1, "D"))
        if ds.sizes[_SOURCE_TIME_DIM] != expected:
            raise ValueError(
                f"CHIRPS3 {self.stage} has {ds.sizes[_SOURCE_TIME_DIM]} of the {expected} days from "
                f"{np.datetime_as_string(start, unit='D')}"
            )
        # CHIRPS has a value for every land cell, so a day without any is one dynamical.org has added
        # to the time axis but not written yet: its update first extends the axis, then fills it.
        # Storing that day would keep NaN for good, as a sync only appends after its last day.
        empty = ds[_VARIABLE].isnull().all(["y", "x"])
        if bool(empty.any()):
            days = np.datetime_as_string(ds[_SOURCE_TIME_DIM].values[empty.values], unit="D")
            raise RuntimeError(
                f"CHIRPS3 {self.stage} has no values over the extent for {', '.join(days)}; "
                "dynamical.org is probably part-way through an update, so retry later"
            )

        precip = ds[_VARIABLE] * _SECONDS_PER_DAY
        # The source attributes describe the rate in kg m-2 s-1, so they are replaced, not kept.
        precip.attrs = {"long_name": "Precipitation rate", "units": "mm/d"}
        return ds.assign({_VARIABLE: precip}).rename({_SOURCE_TIME_DIM: "t"})

    def _prepare(self, source: xr.Dataset, bbox: list[float]) -> xr.Dataset:
        ds = source.rio.write_crs("EPSG:4326")
        # Keeps the source variable's name, which the cache reads by.
        ds = normalize_period(ds, variable=_SOURCE_VARIABLE, bbox=list(bbox), time_dim=_SOURCE_TIME_DIM)
        return normalize_y_direction(ds)

    def _latest_day(self) -> date:
        latest = self._open_source()[_SOURCE_TIME_DIM].values[-1]
        return date.fromisoformat(np.datetime_as_string(latest, unit="D"))

    def _open_source(self) -> xr.Dataset:
        """The store, opened lazily and once. Without dask: the reads are small windowed slices."""
        if self._source is None:
            import icechunk

            repo = icechunk.Repository.open(icechunk.http_storage(self._zarr.url))
            # Typed Any because the open_zarr stub allows only a string (as in data_accessor).
            no_dask: Any = None
            self._source = xr.open_zarr(repo.readonly_session("main").store, chunks=no_dask)
        return self._source


class CHIRPS3MonthlyPlugin(CHIRPS3DailyPlugin):
    """Streaming plugin for CHIRPS3 monthly precipitation: the mean of the final daily values, in mm/d.

    CHC's own monthly product is the month's total, which equals the sum of the same month's
    daily values, so the mean of the dailies is that total divided by the days in the month. The
    store holds the mean daily rate rather than the total because every other monthly
    precipitation dataset here is mm/d, including ``chirps3_precipitation_monthly_normal_1991_2020``,
    the natural partner for a monthly anomaly.
    """

    commit_batch_size = 12

    def __init__(self, store_url: str | None = None, max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES, **_: Any) -> None:
        super().__init__(stage="final", store_url=store_url, max_cache_bytes=max_cache_bytes)

    async def periods(self, start: str, end: str) -> list[str]:
        """Return ordered ``YYYY-MM`` periods, up to the newest complete month in the store."""
        latest = await asyncio.to_thread(self._latest_day)
        # Step back to the end of the previous month unless the latest day closes one.
        cutoff = (latest + timedelta(days=1)).replace(day=1) - timedelta(days=1)
        return monthly_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetch one month of days, clipped to the bbox, and return their mean on the first of the month."""
        try:
            month = np.datetime64(normalize_period_string(str(period_id), "monthly"), "M")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid CHIRPS3 monthly period id {period_id!r}: expected YYYY-MM") from exc
        start = month.astype("datetime64[ns]")
        days = self._read_days(start, (month + np.timedelta64(1, "M")).astype("datetime64[ns]"), bbox)
        # Every day is present and has values (checked by _read_days), so no day is skipped.
        return days.mean("t", skipna=False, keep_attrs=True).expand_dims(t=[start])
