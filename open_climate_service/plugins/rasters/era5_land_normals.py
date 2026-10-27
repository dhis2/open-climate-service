"""ERA5-Land climate-normal ingestion."""

from __future__ import annotations

import asyncio
import base64
import calendar
import json
import logging
import netrc
import os
from threading import Lock
from time import monotonic
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import earthkit.transforms.climatology as ek_climatology
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, cell_pad
from open_climate_service.transforms.climatology import circular_rolling_mean
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius, metres_to_mm

logger = logging.getLogger(__name__)

_EDH_DAILY_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
_EDH_API_KEY_ENV = "EDH_API_KEY"
_EDH_UNKNOWN_TTL_SECONDS = 300.0
_edh_is_v3_cache: dict[str, bool] = {}
_edh_unknown_until: dict[str, float] = {}

_GRID_MAPPING_NAMES = ("spatial_ref", "crs")
_NORMALS_DAYOFYEAR_DIM = "dayofyear"
_NORMALS_DEFAULT_PERIOD = (1991, 2020)
_NORMALS_DEFAULT_SMOOTHING = 31


def _edh_is_zarr_v3(url: str) -> bool | None:
    """Whether an EDH store is Zarr v3; an unreachable store has an unknown result."""
    cached = _edh_is_v3_cache.get(url)
    if cached is not None:
        return cached
    if monotonic() < _edh_unknown_until.get(url, 0.0):
        return None

    verdict: bool | None = None
    try:
        request = Request(f"{url}/zarr.json", headers=_edh_auth_header(url))
        with urlopen(request, timeout=15) as response:  # noqa: S310  # fixed https EDH host
            document = json.loads(response.read())
        declared = document.get("zarr_format") if isinstance(document, dict) else None
        verdict = declared >= 3 if isinstance(declared, int) else None
    except HTTPError as error:
        verdict = False if error.code == 404 else None
    except Exception:  # noqa: BLE001 — an unreachable store is unknown, not non-v3
        verdict = None

    if verdict is None:
        _edh_unknown_until[url] = monotonic() + _EDH_UNKNOWN_TTL_SECONDS
        return None
    _edh_is_v3_cache[url] = verdict
    _edh_unknown_until.pop(url, None)
    return verdict


def _require_zarr_v3(url: str, ds: xr.Dataset) -> None:
    """Refuse a confirmed superseded EDH store that no longer receives updates."""
    if _edh_is_zarr_v3(url) is not False:
        return
    latest = ""
    time_dim = next((name for name in ("valid_time", "time") if name in ds.coords), None)
    if time_dim and ds.sizes.get(time_dim):
        try:
            latest = f" Its latest timestamp is {str(ds[time_dim].isel({time_dim: -1}).values)[:16]}."
        except Exception:  # noqa: BLE001 — enrichment is optional, the refusal is not
            logger.debug("Could not read the latest timestamp of %s", url, exc_info=True)
    ds.close()
    raise RuntimeError(
        f"Earth Data Hub store {url} is not Zarr v3, and OCS reads only v3 stores. EDH applies "
        f"updates only to its Zarr v3 stores, so this one no longer advances.{latest} "
        "Repoint it at the v3 store for this dataset."
    )


def _edh_open_zarr(url: str) -> xr.Dataset:
    """Open an EDH Zarr store with its documented Basic Auth options."""
    opened = xr.open_zarr(
        _edh_authenticated_url(url),
        storage_options={"client_kwargs": {"trust_env": True}},
    )
    _require_zarr_v3(url, opened)
    return opened  # type: ignore[no-any-return]


def _edh_authenticated_url(url: str) -> str:
    token = os.environ.get(_EDH_API_KEY_ENV, "")
    if not token:
        return url
    parts = urlparse(url)
    return urlunparse(parts._replace(netloc=f"edh:{token}@{parts.netloc}"))


def _edh_auth_header(url: str) -> dict[str, str]:
    token = os.environ.get(_EDH_API_KEY_ENV, "")
    if token:
        return _basic_auth("edh", token)
    try:
        host = urlparse(url).hostname or ""
        credentials = netrc.netrc().authenticators(host)
    except (OSError, netrc.NetrcParseError):
        return {}
    if not credentials:
        return {}
    login, _, password = credentials
    return _basic_auth(login or "edh", password or "")


def _basic_auth(login: str, password: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{login}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _drop_auxiliary_variables(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Drop scalar source metadata while preserving dimensions and CRS coordinates."""
    keep = {variable, *ds.dims, *_GRID_MAPPING_NAMES}
    declared = ds[variable].attrs.get("grid_mapping") if variable in ds else None
    if isinstance(declared, str) and declared:
        keep.add(declared)
    extras = [name for name in ds.variables if name not in keep]
    return ds.drop_vars(extras, errors="ignore")


class ERA5LandNormalsPlugin(BaseDatasetPlugin):
    """Compute smoothed day-of-year normals from the EDH ERA5-Land daily archive."""

    max_concurrency = 1
    commit_batch_size = 30
    time_dim = _NORMALS_DAYOFYEAR_DIM
    crs = 4326

    def __init__(
        self,
        *,
        variable: str,
        edh_variable: str,
        period: list[int] | tuple[int, int] = _NORMALS_DEFAULT_PERIOD,
        smoothing_window: int = _NORMALS_DEFAULT_SMOOTHING,
        unit_transform: str | None = None,
        **_: Any,
    ) -> None:
        if not edh_variable:
            raise ValueError("ERA5LandNormalsPlugin requires 'edh_variable' in params")
        if not variable:
            raise ValueError("ERA5LandNormalsPlugin requires a non-empty 'variable'")
        try:
            start_year, end_year = int(period[0]), int(period[1])
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"period must be [start_year, end_year], got {period!r}") from exc
        if start_year > end_year:
            raise ValueError(f"period start {start_year} is after end {end_year}")
        self.variable = variable
        self.edh_variable = edh_variable
        self.period = (start_year, end_year)
        self.smoothing_window = int(smoothing_window)
        if self.smoothing_window < 0:
            raise ValueError(f"smoothing_window must be >= 0, got {self.smoothing_window}")
        if self.smoothing_window % 2 == 0 and self.smoothing_window != 0:
            raise ValueError(f"smoothing_window must be odd (a centred window), got {self.smoothing_window}")
        self.unit_transform = unit_transform
        self._lock = Lock()
        self._climatology: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        _ = start, end
        if self._climatology is not None:
            return [str(int(day)) for day in self._climatology[_NORMALS_DAYOFYEAR_DIM].values]
        start_year, end_year = self.period
        has_leap = any(calendar.isleap(year) for year in range(start_year, end_year + 1))
        return [str(day) for day in range(1, (366 if has_leap else 365) + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_sync, period_id, bbox)

    def _fetch_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        climatology = self._ensure_climatology(bbox)
        selected = climatology.sel({_NORMALS_DAYOFYEAR_DIM: [int(period_id)]})
        return _drop_auxiliary_variables(selected, self.variable)

    def _ensure_climatology(self, bbox: list[float]) -> xr.Dataset:
        with self._lock:
            if self._climatology is None:
                self._climatology = self._compute_climatology(bbox)
            return self._climatology

    def _compute_climatology(self, bbox: list[float]) -> xr.Dataset:
        region = self._load_reference(bbox)
        normals = cast(xr.Dataset, ek_climatology.daily_mean(region, time_dim="valid_time"))
        if self.smoothing_window > 0:
            normals[self.variable] = circular_rolling_mean(
                normals[self.variable], self.smoothing_window, _NORMALS_DAYOFYEAR_DIM
            )
        return self._apply_unit_transform(normals).load()

    def _apply_unit_transform(self, ds: xr.Dataset) -> xr.Dataset:
        if self.unit_transform in (None, "", "none"):
            return ds
        if self.unit_transform == "kelvin_to_celsius":
            return kelvin_to_celsius(ds, {"variable": self.variable})
        if self.unit_transform == "metres_to_mm":
            return metres_to_mm(ds, {"variable": self.variable})
        raise ValueError(f"Unknown unit_transform {self.unit_transform!r}")

    def _load_reference(self, bbox: list[float]) -> xr.Dataset:
        start_year, end_year = self.period
        xmin, ymin, xmax, ymax = map(float, bbox)
        ds = _edh_open_zarr(_EDH_DAILY_URL)
        try:
            longitude_pad = cell_pad(ds["longitude"])
            latitude_pad = cell_pad(ds["latitude"])
            base = ds[[self.edh_variable]].sel(
                latitude=slice(ymax + latitude_pad, ymin - latitude_pad),
                valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
            )
            if xmax - xmin >= 360 - longitude_pad:
                region = base
            else:
                lon_min, lon_max = xmin % 360, xmax % 360
                if lon_min <= lon_max:
                    region = base.sel(longitude=slice(lon_min - longitude_pad, lon_max + longitude_pad))
                else:
                    region = xr.concat(
                        [
                            base.sel(longitude=slice(lon_min - longitude_pad, 360.0)),
                            base.sel(longitude=slice(0.0, lon_max + longitude_pad)),
                        ],
                        dim="longitude",
                    )
            region = region.load()
        finally:
            ds.close()
        region = region.assign_coords(longitude=((region.longitude + 180) % 360) - 180)
        region = region.sortby("longitude").rename({"longitude": "x", "latitude": "y"})
        if self.edh_variable != self.variable:
            region = region.rename({self.edh_variable: self.variable})
        return region
