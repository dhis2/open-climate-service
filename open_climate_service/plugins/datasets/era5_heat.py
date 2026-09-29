"""ERA5-Heat streaming plugins."""

from __future__ import annotations

import asyncio
import calendar
import logging
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import xarray as xr
from earthkit.transforms.temporal import daily_reduce
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.shared.time import (
    daily_period_ids,
    datetime_to_period_string,
    parse_period_string_to_datetime,
)
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

logger = logging.getLogger(__name__)

# CDS dataset
_CDS_CATALOGUE_URL = "https://cds.climate.copernicus.eu/api"
_CDS_HOURLY_COLLECTION = "derived-utci-historical"
_CDS_PRODUCT_TYPE = "intermediate_dataset"
_CDS_VARIABLE_NAMES: dict[str, str] = {
    "utci": "universal_thermal_climate_index",
    "mrt": "mean_radiant_temperature",
}

# Scalar coordinates that carry the CRS rather than data, so they must survive the cleanup.
_GRID_MAPPING_NAMES = ("spatial_ref", "crs")


class ERA5HeatCDSHourlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for hourly ERA5-Heat variables from the Copernicus CDS.

    Fetches one full calendar month per CDS API call and caches the result so
    that consecutive hourly ``fetch_period`` calls within the same month share
    a single remote request.
    """

    max_concurrency = 1
    commit_batch_size = 24

    def __init__(self, variable: str, **kwargs: Any) -> None:
        if variable not in _CDS_VARIABLE_NAMES:
            raise ValueError(
                f"ERA5HeatCDSHourlyPlugin: unsupported variable {variable!r}; "
                f"expected one of {list(_CDS_VARIABLE_NAMES)}"
            )
        self.variable = variable
        self._cache_lock = Lock()
        self._cached_month: tuple[int, int] | None = None
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_ds: xr.Dataset | None = None
        self._cached_cutoff: datetime | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        # NOTE: Does not take into account local UTC offset hour which may include the previous or next day
        if self._cached_cutoff is None:
            self._cached_cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        cutoff = self._cached_cutoff
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

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetches xarray dataset for a single hour snapshot.

        Downloads or reuses cache of relevant monthly CDS NetCDF file.
        """
        # TODO: Later we should take into account UTC offset hour which may include the previous or next day
        dt = parse_period_string_to_datetime(period_id)
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))
        with self._cache_lock:
            if self._cached_month != (dt.year, dt.month) or self._cached_bbox != bbox_tuple:
                self._cached_ds = self._fetch_month(dt.year, dt.month, bbox_tuple)
                self._cached_month = (dt.year, dt.month)
                self._cached_bbox = bbox_tuple
            monthly_ds = self._cached_ds
        assert monthly_ds is not None

        timestamp = np.datetime64(dt.replace(tzinfo=None), "h").astype("datetime64[ns]")
        hour_ds = monthly_ds.sel(t=timestamp)

        return hour_ds

    def _fetch_month(self, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        _, last_day = calendar.monthrange(year, month)

        # Cap to availability cutoff so we don't request future days from CDS
        if self._cached_cutoff is None:
            self._cached_cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        cutoff = self._cached_cutoff
        if cutoff.year == year and cutoff.month == month:
            last_day = min(last_day, cutoff.day)

        # Submit request
        params = {
            "variable": [_CDS_VARIABLE_NAMES[self.variable]],
            "version": "1_1",
            "product_type": _CDS_PRODUCT_TYPE,
            "year": str(year),
            "month": str(month).zfill(2),
            "day": [str(d).zfill(2) for d in range(1, last_day + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
        }
        logger.info(f"Fetching and caching month data from CDS {params}")
        remote = _CdsClient().submit(_CDS_HOURLY_COLLECTION, params)

        # Download comes as zipfile with one nc file per day
        # save to temp dir, extract, and consolidate to single nc file
        with tempfile.TemporaryDirectory(delete=True) as tmpdir:
            # download zipfile
            zip_path = Path(tmpdir) / "tempzip.zip"
            remote.download(str(zip_path))

            # extract all files to same folder
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmpdir)

            # open all extracted nc files
            nc_paths = str(Path(tmpdir) / "*.nc")
            ds = xr.open_mfdataset(nc_paths)

            # load into memory and close file connections
            ds = ds.load()
            ds.close()

        # Add crs needed later
        ds = ds.rio.write_crs("EPSG:4326")

        # Normalize xarray dims
        ds = normalize_period(ds, variable=self.variable, bbox=list(bbox))

        # Convert Kelvin to Celsius
        # TODO: This should be correct for utci and mrt, but prob not if we support other vars...
        ds = kelvin_to_celsius(ds, {"variable": self.variable})

        # Return
        return ds


class ERA5HeatCDSDailyFromHourlyPlugin(ERA5HeatCDSHourlyPlugin):
    """Daily from hourly aggregation."""

    max_concurrency = 1
    commit_batch_size = 30

    def __init__(self, temporal_aggregation: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._temporal_aggregation = temporal_aggregation

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
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

        # aggregate to daily
        daily_ds = daily_reduce(
            hourly_ds,
            how=self._temporal_aggregation,
            time_shift={"hours": 0},  # TODO: Later should support local UTC offset
            remove_partial_periods=False,
        )

        return cast(xr.Dataset, daily_ds)


class ERA5HeatDailyUTCIPlugin(ERA5HeatCDSDailyFromHourlyPlugin):
    """ERA5-Heat Daily UTCI plugin."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variable="utci", **kwargs)


# Helpers


def _cds_end_datetime(collection: str) -> datetime:
    """Query the CDS catalogue for the end_datetime of a collection."""
    col = _CdsClient(url=_CDS_CATALOGUE_URL, key="").get_collection(collection)
    if col.end_datetime is None:
        raise RuntimeError(f"CDS collection '{collection}' returned no end_datetime")
    return col.end_datetime


def _hourly_availability_cutoff() -> datetime:
    """Return the latest hour for which the CDS hourly collection data are published."""
    return _cds_end_datetime(_CDS_HOURLY_COLLECTION)
