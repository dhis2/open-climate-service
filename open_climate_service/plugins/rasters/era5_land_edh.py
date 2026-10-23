"""Daily and monthly ERA5-Land ingestion from the Earth Data Hub daily Zarr archive."""

from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import numpy as np
import xarray as xr
from earthkit.transforms.temporal import monthly_reduce

from open_climate_service.shared.raster_contract import normalize_y_direction
from open_climate_service.shared.time import daily_period_ids, monthly_period_ids
from open_climate_service.shared.zarr import DEFAULT_MAX_CACHE_BYTES, CachedTimeSeriesZarr
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius, metres_to_mm

_EDH_DAILY_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
_EDH_TIME_DIM = "valid_time"
_UNIT_TRANSFORMS = {
    "kelvin_to_celsius": kelvin_to_celsius,
    "metres_to_mm": metres_to_mm,
}


class ERA5LandEDHDailyPlugin(BaseDatasetPlugin):
    """Read one variable from the Earth Data Hub daily ERA5-Land Zarr store."""

    max_concurrency = 1
    commit_batch_size = 30

    def __init__(
        self,
        variable: str,
        zarr_url: str = _EDH_DAILY_URL,
        unit_transform: str | None = None,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        **kwargs: Any,
    ) -> None:
        if unit_transform and unit_transform not in _UNIT_TRANSFORMS:
            raise ValueError(f"Unknown unit_transform {unit_transform!r}; expected one of {list(_UNIT_TRANSFORMS)}")

        self.variable = variable
        self.unit_transform = unit_transform

        self._zarr = CachedTimeSeriesZarr(
            zarr_url,
            variable,
            time_dim=_EDH_TIME_DIM,
            opener=_open_edh_zarr,
            max_cache_bytes=max_cache_bytes,
        )
        self._cached_cutoff: date | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(self._fetch_cutoff)
        return daily_period_ids(start=start, end=end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        start = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return self._fetch_time_range(start, start + np.timedelta64(1, "D"), bbox)

    def _fetch_time_range(self, start: np.datetime64, end: np.datetime64, bbox: list[float]) -> xr.Dataset:
        ds = self._zarr.slice_time(
            start,
            end,
            prepare=lambda source: self._prepare_zarr_dataset(source, bbox),
        )
        if _EDH_TIME_DIM in ds.dims:
            ds = ds.rename({_EDH_TIME_DIM: "t"})
        return ds

    def _prepare_zarr_dataset(self, ds: xr.Dataset, bbox: list[float]) -> xr.Dataset:
        # The store holds longitude as 0-360; move it to -180..180 so a bbox clips as given.
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
        ds = ds.rio.write_crs("EPSG:4326")
        ds = normalize_period(ds, variable=self.variable, bbox=list(bbox), time_dim=_EDH_TIME_DIM)
        ds = normalize_y_direction(ds)
        if self.unit_transform in (None, "", "none"):
            return ds
        transform = _UNIT_TRANSFORMS[self.unit_transform]
        return transform(ds, {"variable": self.variable})

    def _fetch_cutoff(self) -> date:
        if self._cached_cutoff is None:
            ds = _open_edh_zarr(self._zarr.url)
            try:
                latest = ds[_EDH_TIME_DIM].isel({_EDH_TIME_DIM: -1}).values
                self._cached_cutoff = date.fromisoformat(str(latest.astype("datetime64[D]")))
            finally:
                ds.close()
        return self._cached_cutoff


class ERA5LandEDHMonthlyFromDailyPlugin(ERA5LandEDHDailyPlugin):
    """Aggregate daily ERA5-Land values from Earth Data Hub into calendar-month outputs."""

    commit_batch_size = 12

    def __init__(self, *, temporal_aggregation: str, **kwargs: Any) -> None:
        if temporal_aggregation not in {"mean", "sum", "min", "max"}:
            raise ValueError(
                f"temporal_aggregation must be one of 'mean', 'sum', 'min', or 'max'; got {temporal_aggregation!r}"
            )
        super().__init__(**kwargs)
        self.temporal_aggregation = temporal_aggregation

    async def periods(self, start: str, end: str) -> list[str]:
        latest = await asyncio.to_thread(self._fetch_cutoff)
        # Only complete months: step back to the end of the previous month unless the latest day closes one.
        cutoff = (latest + timedelta(days=1)).replace(day=1) - timedelta(days=1)
        return monthly_period_ids(start=start, end=end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        start = np.datetime64(period_id, "M")
        end = (start + np.timedelta64(1, "M")).astype("datetime64[ns]")
        daily = self._fetch_time_range(start.astype("datetime64[ns]"), end, bbox)
        monthly = monthly_reduce(
            daily,
            how=self.temporal_aggregation,
            time_dim="t",
            remove_partial_periods=False,
        )
        return cast(xr.Dataset, monthly)


def _open_edh_zarr(url: str) -> xr.Dataset:
    """Open an Earth Data Hub store, authenticating with ``EDH_API_KEY`` or else netrc."""
    return xr.open_zarr(url, storage_options={"client_kwargs": {"trust_env": True}})
