"""Daily ERA5-Land ingestion from the CDS geo-chunked ARCO Zarr archive."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from earthkit.transforms.temporal import daily_reduce

from open_climate_service.shared.raster_contract import normalize_y_direction
from open_climate_service.shared.time import (
    daily_period_ids,
    datetime_to_period_string,
    parse_period_string_to_datetime,
)
from open_climate_service.shared.zarr import DEFAULT_MAX_CACHE_BYTES, CachedTimeSeriesZarr
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius, metres_to_mm

_CDS_ZARR_GROUPS = {
    "sfc-soil-water": "005",
    "sfc-soil-temperature": "006",
    "sfc-2m-temperature": "007",
    "sfc-wind": "008",
    "sfc-pressure-precipitation": "009",
    "sfc-radiation-heat": "010",
    "sfc-snow": "030",
    "sfc-skin-temperature": "043",
}
_CDS_ZARR_VARIABLE_GROUPS = {
    **{f"swvl{layer}": "sfc-soil-water" for layer in range(1, 5)},
    **{f"stl{layer}": "sfc-soil-temperature" for layer in range(1, 5)},
    "d2m": "sfc-2m-temperature",
    "t2m": "sfc-2m-temperature",
    "u10": "sfc-wind",
    "v10": "sfc-wind",
    "sp": "sfc-pressure-precipitation",
    "tp": "sfc-pressure-precipitation",
    "ssrd": "sfc-radiation-heat",
    "strd": "sfc-radiation-heat",
    "skt": "sfc-skin-temperature",
}
_CDS_ZARR_STORES = {
    variable: (
        f"https://arco.datastores.ecmwf.int/cadl-arco-geo-{_CDS_ZARR_GROUPS[group]}"
        f"/arco/reanalysis_era5_land/{group}/geoChunked.zarr"
    )
    for variable, group in _CDS_ZARR_VARIABLE_GROUPS.items()
}
_CDS_TIME_DIM = "time"
_UNIT_TRANSFORMS = {
    "kelvin_to_celsius": kelvin_to_celsius,
    "metres_to_mm": metres_to_mm,
}


class ERA5LandZarrHourlyPlugin(BaseDatasetPlugin):
    """Read one variable from a CDS geo-chunked ERA5-Land ARCO Zarr store."""

    max_concurrency = 1
    commit_batch_size = 24

    def __init__(
        self,
        variable: str,
        zarr_url: str | None = None,
        unit_transform: str | None = None,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        **kwargs: Any,
    ) -> None:
        source_url = zarr_url or _CDS_ZARR_STORES.get(variable)
        if not source_url:
            raise ValueError(
                f"Unsupported ERA5-Land ARCO variable {variable!r}; specify its CDS geo-chunked Zarr URL with 'zarr_url'"
            )
        
        if unit_transform:
            if unit_transform not in _UNIT_TRANSFORMS.keys():
                raise ValueError(f"Unknown unit_transform {unit_transform!r}; expected one of {list(_UNIT_TRANSFORMS)}")
        
        self.variable = variable
        self.unit_transform = unit_transform

        self._zarr = CachedTimeSeriesZarr(
            source_url,
            variable,
            time_dim=_CDS_TIME_DIM,
            opener=_open_cds_zarr,
            max_cache_bytes=max_cache_bytes,
        )
        self._cached_cutoff: datetime | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(self._fetch_cutoff)
        current = parse_period_string_to_datetime(start)
        last = parse_period_string_to_datetime(end).replace(hour=23, minute=59)
        limit = min(last, cutoff)
        if current > limit:
            return []
        periods: list[str] = []
        while current <= limit:
            periods.append(datetime_to_period_string(current, "hourly"))
            current += timedelta(hours=1)
        return periods

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        dt = parse_period_string_to_datetime(period_id)
        start = np.datetime64(dt.replace(tzinfo=None), "h").astype("datetime64[ns]")
        return self._fetch_time_range(start, start + np.timedelta64(1, "h"), bbox)

    def _fetch_time_range(self, start: np.datetime64, end: np.datetime64, bbox: list[float]) -> xr.Dataset:
        ds = self._zarr.slice_time(
            start,
            end,
            prepare=lambda source: self._prepare_zarr_dataset(source, bbox),
        )
        if _CDS_TIME_DIM in ds.dims:
            ds = ds.rename({_CDS_TIME_DIM: "t"})
        return ds

    def _prepare_zarr_dataset(self, ds: xr.Dataset, bbox: list[float]) -> xr.Dataset:
        ds = ds.rio.write_crs("EPSG:4326")
        ds = normalize_period(ds, variable=self.variable, bbox=list(bbox), time_dim=_CDS_TIME_DIM)
        ds = normalize_y_direction(ds)
        if self.unit_transform in (None, "", "none"):
            return ds
        transform = _UNIT_TRANSFORMS[self.unit_transform]
        return transform(ds, {"variable": self.variable})

    def _fetch_cutoff(self) -> datetime:
        if self._cached_cutoff is None:
            ds = _open_cds_zarr(self._zarr.url)
            try:
                latest = ds[_CDS_TIME_DIM].isel({_CDS_TIME_DIM: -1}).values
                last_hour = datetime.fromisoformat(str(latest))
            finally:
                ds.close()
            self._cached_cutoff = last_hour.replace(tzinfo=timezone.utc)
        return self._cached_cutoff


class ERA5LandZarrDailyFromHourlyPlugin(ERA5LandZarrHourlyPlugin):
    """Aggregate hourly ERA5-Land ARCO values into UTC calendar-day outputs."""

    commit_batch_size = 30

    def __init__(self, *, temporal_aggregation: str, **kwargs: Any) -> None:
        if temporal_aggregation not in {"mean", "sum", "min", "max"}:
            raise ValueError(
                f"temporal_aggregation must be one of 'mean', 'sum', 'min', or 'max'; got {temporal_aggregation!r}"
            )
        super().__init__(**kwargs)
        self.temporal_aggregation = temporal_aggregation

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = super()._fetch_cutoff()
        periods = daily_period_ids(start=start, end=end, cutoff=cutoff)

        return periods

    def fetch_period(self, period_id: str, bbox: list[float], **kwargs: Any) -> xr.Dataset:
        start = np.datetime64(period_id, "D").astype("datetime64[ns]")
        hourly = self._fetch_time_range(start, start + np.timedelta64(1, "D"), bbox)
        daily = daily_reduce(
            hourly,
            how=self.temporal_aggregation,
            time_shift={"hours": 0},
            remove_partial_periods=False,
        )
        return cast(xr.Dataset, daily)


def _open_cds_zarr(url: str) -> xr.Dataset:
    """Open a CDS ARCO store using the configured CDS API token."""
    token = _get_cdsapi_key()
    return xr.open_zarr(
        url,
        consolidated=True,
        storage_options={"headers": {"Authorization": f"Bearer {token}"}},
    )


def _get_cdsapi_key() -> str:
    token = os.environ.get("CDS_API_KEY") or os.environ.get("CDSAPI_KEY")
    if token:
        return token

    credentials_path = Path.home() / ".ecmwfdatastoresrc"
    if not credentials_path.exists():
        raise SystemError("Missing CDS API credentials: set CDS_API_KEY or configure ~/.ecmwfdatastoresrc")
    for line in credentials_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("key:"):
            token = line.partition(":")[2].strip()
            if token:
                return token
    raise SystemError(f"Unable to retrieve CDS API key from {credentials_path}")
