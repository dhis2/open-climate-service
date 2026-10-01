"""ERA5-Heat streaming plugins."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import numpy as np
import xarray as xr
from earthkit.transforms.temporal import daily_reduce

from open_climate_service.shared.time import (
    daily_period_ids,
    datetime_to_period_string,
    parse_period_string_to_datetime,
)
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

logger = logging.getLogger(__name__)

# CDS dataset
_CDS_ZARR_URL = "https://arco.datastores.ecmwf.int/cadl-arco-geo-004/arco/derived_utci_historical/all/geoChunked.zarr"
_CDS_ZARR_VARIABLES = ["utci", "mrt"]


class ERA5HeatZarrHourlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for hourly ERA5-Heat zarr archive from Copernicus CDS."""

    max_concurrency = 1
    commit_batch_size = 24

    def __init__(self, variable: str, **kwargs: Any) -> None:
        if variable not in _CDS_ZARR_VARIABLES:
            raise ValueError(
                f"ERA5HeatCDSHourlyPlugin: unsupported variable {variable!r}; "
                f"expected one of {list(_CDS_ZARR_VARIABLES)}"
            )
        self.variable = variable
        self._cached_ds: xr.Dataset | None = None
        self._cached_cutoff: datetime | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        # NOTE: Does not take into account local UTC offset hour which may include the previous or next day

        cutoff = self._fetch_cutoff()
        current = parse_period_string_to_datetime(start)
        last = parse_period_string_to_datetime(end)
        last = last.replace(hour=23, minute=59)  # make sure user requested end time is at the very end of the day
        limit = min(last, cutoff)

        if current > limit:
            return []

        result: list[str] = []
        while current <= limit:
            result.append(datetime_to_period_string(current, "hourly"))
            current += timedelta(hours=1)

        return result

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        """Fetches xarray dataset for a single hour snapshot."""
        # TODO: Later we should take into account UTC offset hour which may include the previous or next day

        # Get ds
        ds = self._fetch_ds(bbox)

        # Subset to requested hour
        dt = parse_period_string_to_datetime(period_id)
        timestamp = np.datetime64(dt.replace(tzinfo=None), "h").astype("datetime64[ns]")
        ds = ds.sel(t=timestamp)

        # NOTE: Usually .fetch_period() should call .load() before returning.
        # However, doing that for every hour results in a significant slowdown,
        # and we don't except this Hourly class to be used directly
        # so it's okay that we don't call .load().

        return ds

    def _fetch_ds(self, bbox: list[float]) -> xr.Dataset:
        if self._cached_ds is None:
            # Get data
            ds = _open_cds_zarr(_CDS_ZARR_URL)

            # Add crs needed later
            ds = ds.rio.write_crs("EPSG:4326")

            # Normalize xarray dims
            ds = normalize_period(ds, variable=self.variable, bbox=list(bbox))

            # Convert Kelvin to Celsius
            ds = kelvin_to_celsius(ds, {"variable": self.variable})

            # Save to cache
            self._cached_ds = ds

        return self._cached_ds

    def _fetch_cutoff(self) -> datetime:
        if self._cached_cutoff is None:
            self._cached_cutoff = _hourly_availability_cutoff()
        return self._cached_cutoff


class ERA5HeatZarrDailyFromHourlyPlugin(ERA5HeatZarrHourlyPlugin):
    """Daily from hourly aggregation."""

    max_concurrency = 1
    commit_batch_size = 30

    def __init__(self, temporal_aggregation: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._temporal_aggregation = temporal_aggregation

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = super()._fetch_cutoff()
        periods = daily_period_ids(start=start, end=end, cutoff=cutoff)

        return periods

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetches xarray dataset for a single day snapshot, aggregated from relevant hourly snapshots.

        Downloads or reuses cache of relevant monthly CDS NetCDF file.
        """
        # get hourly periods for the day
        hour_periods = asyncio.run(super().periods(start=period_id, end=period_id))

        # load each hour dataset and merge
        # NOTE: this is probably not very efficient but should reuse code and produce correct results
        hourly_ds = xr.concat(
            [super().fetch_period(period_id=hour_period, bbox=bbox) for hour_period in hour_periods],
            dim="t",
        )

        # load into memory for efficiency
        hourly_ds = hourly_ds.load()

        # aggregate to daily
        daily_ds = daily_reduce(
            hourly_ds,
            how=self._temporal_aggregation,
            time_shift={"hours": 0},  # TODO: Later should support local UTC offset
            remove_partial_periods=False,
        )

        return cast(xr.Dataset, daily_ds)


class ERA5HeatDailyUTCIPlugin(ERA5HeatZarrDailyFromHourlyPlugin):
    """ERA5-Heat Daily UTCI plugin."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variable="utci", **kwargs)


# Helpers


def _get_cdsapi_key() -> str:
    if os.path.exists(os.path.expanduser("~/.ecmwfdatastoresrc")):
        with open(os.path.expanduser("~/.ecmwfdatastoresrc"), "r") as f:
            for line in f:
                if line.startswith("key:"):
                    cdsapi_key = line.split(":")[1].strip()
                    if cdsapi_key:
                        return cdsapi_key

        raise SystemError(
            "Unable to retrieve CDS API key, please verify that ~/.ecmwfdatastoresrc has the correct format"
        )

    raise SystemError("Missing credentials file: ~/.ecmwfdatastoresrc")


def _open_cds_zarr(url: str) -> xr.Dataset:
    cdsapi_key = _get_cdsapi_key()
    ds: xr.Dataset = xr.open_zarr(
        url, consolidated=True, storage_options={"headers": {"Authorization": f"Bearer {cdsapi_key}"}}
    )
    return ds


def _hourly_availability_cutoff() -> datetime:
    """Return the latest hour for which the zarr archive has hourly data."""
    ds = _open_cds_zarr(_CDS_ZARR_URL)
    last_hour = ds.time.values[-1].astype("datetime64[us]").tolist()
    last_hour = last_hour.replace(tzinfo=timezone.utc)

    return cast(datetime, last_hour)
