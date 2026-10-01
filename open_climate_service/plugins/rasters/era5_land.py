"""ERA5-Land streaming plugins."""

from __future__ import annotations

import asyncio
import base64
import calendar
import json
import logging
import netrc
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr
from earthkit.transforms import climatology as ek_climatology
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.shared.time import (
    daily_period_ids,
    datetime_to_period_string,
    parse_period_string_to_datetime,
)
from open_climate_service.streaming import BaseDatasetPlugin, monthly_period_ids
from open_climate_service.transforms.climatology import circular_rolling_mean
from open_climate_service.transforms.unit_conversion import kelvin_to_celsius, metres_to_mm

logger = logging.getLogger(__name__)

# CDS API long-name variables for reanalysis-era5-land-monthly-means
_CDS_VARIABLE_NAMES: dict[str, str] = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
}

# The grid (0.1° resolution, EPSG:4326, float32) is inferred by the orchestrator
# from the first fetched period.


class ERA5LandCDSHourlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for hourly ERA5-Land variables from the Copernicus CDS.

    Fetches one full calendar month per CDS API call and caches the result so
    that consecutive hourly ``fetch_period`` calls within the same month share
    a single remote request.
    """

    max_concurrency = 1
    commit_batch_size = 24

    def __init__(self, variable: str, **_: Any) -> None:
        if variable not in _CDS_VARIABLE_NAMES:
            raise ValueError(
                f"ERA5LandCDSHourlyPlugin: unsupported variable {variable!r}; "
                f"expected one of {list(_CDS_VARIABLE_NAMES)}"
            )
        self.variable = variable
        self._cache_lock = Lock()
        self._cached_month: tuple[int, int] | None = None
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_ds: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        current = parse_period_string_to_datetime(start)
        limit = min(parse_period_string_to_datetime(end), cutoff)
        if current > limit:
            return []
        result: list[str] = []
        while current <= limit:
            result.append(datetime_to_period_string(current, "hourly"))
            current += timedelta(hours=1)
        return result

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
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
        return monthly_ds.sel(t=timestamp)

    def _fetch_month(self, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        _, last_day = calendar.monthrange(year, month)
        # Cap to availability cutoff so we don't request future days from CDS
        cutoff = _hourly_availability_cutoff()
        if cutoff.year == year and cutoff.month == month:
            last_day = min(last_day, cutoff.day)
        params = {
            "variable": [_CDS_VARIABLE_NAMES[self.variable]],
            "year": str(year),
            "month": str(month).zfill(2),
            "day": [str(d).zfill(2) for d in range(1, last_day + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_hourly.nc"
            remote = _CdsClient().submit("reanalysis-era5-land", params)
            remote.download(str(target))
            ds = xr.open_dataset(target, engine="netcdf4").load()
        ds = _collapse_expver(ds[[self.variable]])
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds = ds.rename({"longitude": "x", "latitude": "y", time_dim: "t"})
        if self.variable == "t2m":
            ds = kelvin_to_celsius(ds, {"variable": self.variable})
        # Cleaned on the cached monthly cube, while `t` is still a dimension. Doing it after
        # `fetch_period`'s scalar `sel(t=...)` would take the demoted `t` coordinate with it.
        return _drop_auxiliary_variables(ds, self.variable)


class ERA5LandPrecipitationPlugin(ERA5LandCDSHourlyPlugin):
    """ERA5-Land hourly precipitation plugin (metres → millimetres)."""

    def __init__(self, **_: Any) -> None:
        super().__init__(variable="tp")

    def _fetch_month(self, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        ds = super()._fetch_month(year, month, bbox)
        ds = ds.assign(tp=_deaccumulate_tp(ds["tp"], time_dim="t"))
        return metres_to_mm(ds, {"variable": self.variable})


class ERA5LandDailyTemperaturePlugin(BaseDatasetPlugin):
    """Streaming plugin for daily ERA5-Land 2m temperature from the Copernicus CDS.

    Uses the ``derived-era5-land-daily-statistics`` dataset. Fetches one full
    calendar month per CDS API call and caches the result in memory so that
    consecutive daily ``fetch_period`` calls within the same month only submit
    one remote request.  ``max_concurrency = 1`` ensures the cache is never
    accessed by concurrent fetches.
    """

    max_concurrency = 1
    commit_batch_size = 31

    def __init__(self, **_: Any) -> None:
        self._cache_lock = Lock()
        self._cached_month: tuple[int, int] | None = None
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_ds: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_daily_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = date.fromisoformat(period_id)
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))

        with self._cache_lock:
            if self._cached_month != (day.year, day.month) or self._cached_bbox != bbox_tuple:
                self._cached_ds = self._fetch_month(day.year, day.month, bbox_tuple)
                self._cached_month = (day.year, day.month)
                self._cached_bbox = bbox_tuple
            monthly_ds = self._cached_ds

        assert monthly_ds is not None
        timestamp = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return _drop_auxiliary_variables(monthly_ds.sel(t=slice(timestamp, timestamp)), "t2m")

    def _fetch_month(self, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        _, last_day = calendar.monthrange(year, month)
        cutoff = _daily_availability_cutoff()
        if cutoff.year == year and cutoff.month == month:
            last_day = min(last_day, cutoff.day)
        params = {
            "variable": ["2m_temperature"],
            "year": str(year),
            "month": str(month).zfill(2),
            "day": [str(d).zfill(2) for d in range(1, last_day + 1)],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_daily.nc"
            remote = _CdsClient().submit("derived-era5-land-daily-statistics", params)
            remote.download(str(target))
            ds = xr.open_dataset(target, engine="netcdf4").load()

        ds = _collapse_expver(ds[["t2m"]])
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds = ds.rename({"longitude": "x", "latitude": "y", time_dim: "t"})
        return kelvin_to_celsius(ds, {"variable": "t2m"})


class ERA5LandMonthlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for monthly ERA5-Land means from the Copernicus CDS.

    Fetches a **calendar year** per CDS API call from ``reanalysis-era5-land-monthly-means``
    and caches it, so the twelve ``fetch_period`` calls for that year share one remote
    request. Credentials are read from ``~/.cdsapirc`` or the ``CDSAPI_URL`` / ``CDSAPI_KEY``
    environment variables.

    A CDS request costs almost entirely queue and dispatch rather than data — one month over
    a country extent is about 30 KB — so twelve months cost barely more than one. Measured
    against the live API: 28.7 s for a single month against 34.4 s for a year, which is 2.9 s
    per month rather than 28.7 (CLIM-956). Over a 30-year ingest that is roughly 21 minutes
    instead of 204.

    ``max_concurrency = 1`` is what keeps the cache safe from concurrent fetches, the same
    reason it is pinned on :class:`ERA5LandCDSHourlyPlugin`. It is not a throughput limit to
    raise: batching cuts the request count outright, which is both faster and lighter on CDS
    than running the same number of requests in parallel.
    """

    max_concurrency = 1
    commit_batch_size = 12

    def __init__(self, variable: str, **_: Any) -> None:
        if variable not in _CDS_VARIABLE_NAMES:
            raise ValueError(
                f"ERA5LandMonthlyPlugin: unsupported variable {variable!r}; expected one of {list(_CDS_VARIABLE_NAMES)}"
            )
        self.variable = variable
        self._cache_lock = Lock()
        self._cached_year: int | None = None
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_ds: xr.Dataset | None = None
        # Resolved once per plugin, not once per year: it is a catalogue round-trip, and the
        # answer cannot move backwards during a single ingest.
        self._cutoff: date | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = await asyncio.to_thread(_monthly_availability_cutoff)
        return monthly_period_ids(_parse_monthly(start), _parse_monthly(end), cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        year, month = int(period_id[:4]), int(period_id[5:7])
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))

        with self._cache_lock:
            if self._cached_year != year or self._cached_bbox != bbox_tuple:
                self._cached_ds = self._fetch_year(year, bbox_tuple)
                self._cached_year = year
                self._cached_bbox = bbox_tuple
            yearly = self._cached_ds

        assert yearly is not None
        # Selected by calendar month rather than by an exact timestamp: the store's own
        # convention for where in the month the stamp sits is CDS's to decide, and matching on
        # it would break quietly if that ever changed.
        stamps = yearly["t"].dt
        wanted = np.flatnonzero(((stamps.year == year) & (stamps.month == month)).values)
        if wanted.size == 0:
            raise ValueError(f"CDS returned no ERA5-Land monthly mean for {period_id}")
        return _drop_auxiliary_variables(yearly.isel(t=wanted), self.variable)

    def _fetch_year(self, year: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        """Fetch every published month of *year* in as few CDS requests as possible.

        Usually one. Months are grouped by product type first, because
        :func:`_era5land_monthly_product_type` switches product mid-year for accumulated
        variables between September 2022 and February 2024 — asking for a whole year in one
        request there would silently return the buggy product for part of it, which is the
        very thing that workaround exists to avoid.
        """
        if self._cutoff is None:
            self._cutoff = _monthly_availability_cutoff()
        cutoff = self._cutoff
        last_month = cutoff.month if year == cutoff.year else 12
        if year > cutoff.year:
            raise ValueError(f"ERA5-Land monthly means are not published for {year}")

        by_product: dict[str, list[int]] = {}
        for month in range(1, last_month + 1):
            product = _era5land_monthly_product_type(year, month, self.variable)
            by_product.setdefault(product, []).append(month)

        parts = [self._fetch_months(year, months, product, bbox) for product, months in by_product.items()]
        if len(parts) == 1:
            return parts[0]
        return xr.concat(parts, dim="t").sortby("t")

    def _fetch_months(
        self, year: int, months: list[int], product_type: str, bbox: tuple[float, float, float, float]
    ) -> xr.Dataset:
        xmin, ymin, xmax, ymax = bbox
        params = {
            "product_type": [product_type],
            "variable": [_CDS_VARIABLE_NAMES[self.variable]],
            "year": [str(year)],
            "month": [str(month).zfill(2) for month in months],
            "time": ["00:00"],
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_monthly.nc"
            remote = _CdsClient().submit("reanalysis-era5-land-monthly-means", params)
            remote.download(str(target))
            # Loaded, not lazy: the framework closes what `fetch_period` returns, and every
            # month of the year is served from this one object.
            ds = xr.open_dataset(target, engine="netcdf4").load()

        ds = _collapse_expver(ds[[self.variable]])
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        rename_map = {"longitude": "x", "latitude": "y", time_dim: "t"}
        ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})

        if self.variable == "t2m":
            ds = kelvin_to_celsius(ds, {"variable": self.variable})
        return ds


class ERA5LandMonthlyPrecipitationPlugin(ERA5LandMonthlyPlugin):
    """ERA5-Land monthly precipitation plugin (mean daily rate, mm/day).

    CDS returns the monthly mean of daily totals (m/day). We convert m -> mm to
    store the mean daily precipitation *rate* (mm/day) — the dimensionally-correct
    precipitation flux that xclim's indices (SPI/SPEI, …) consume directly. We
    deliberately do not pre-multiply by days-in-month: the calendar-month total is
    a trivial ``rate * days_in_month`` derivation, while keeping the rate lets xclim
    integrate over the exact period length itself.
    """

    def __init__(self, **_: Any) -> None:
        super().__init__(variable="tp")

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        ds = super().fetch_period(period_id, bbox)
        return metres_to_mm(ds, {"variable": self.variable})


def _parse_monthly(period: str) -> datetime:
    """Parse a monthly period string (YYYY-MM or full ISO) to the first of the month."""
    if len(period) == 7:
        return datetime(int(period[:4]), int(period[5:7]), 1)
    dt = parse_period_string_to_datetime(period)
    return datetime(dt.year, dt.month, 1)


_CDS_CATALOGUE_URL = "https://cds.climate.copernicus.eu/api"
_CDS_HOURLY_COLLECTION = "reanalysis-era5-land"
_CDS_DAILY_COLLECTION = "derived-era5-land-daily-statistics"
_CDS_MONTHLY_COLLECTION = "reanalysis-era5-land-monthly-means"


def _era5land_monthly_product_type(year: int, month: int, variable: str) -> str:
    """Return the correct CDS product_type for ERA5-Land monthly means.

    The standard ``monthly_averaged_reanalysis`` product contains incorrect
    data for **accumulated variables** (tp) for September 2022 – February 2024.
    For that range, ``monthly_averaged_reanalysis_by_hour_of_day`` at
    time=00:00 returns the monthly mean of daily step-24 values, which is
    unaffected by the bug and gives the correct monthly totals.

    For instantaneous variables (t2m), switching product type would change the
    semantics from "monthly mean" to "mean at 00:00 UTC" — so the workaround
    is intentionally skipped for non-accumulated variables.

    Reference: https://forum.ecmwf.int/t/2370
    """
    accumulated = {"tp"}
    if variable in accumulated and (2022, 9) <= (year, month) <= (2024, 2):
        return "monthly_averaged_reanalysis_by_hour_of_day"
    return "monthly_averaged_reanalysis"


def _cds_end_datetime(collection: str) -> datetime:
    """Query the CDS catalogue for the end_datetime of a collection."""
    col = _CdsClient(url=_CDS_CATALOGUE_URL, key="").get_collection(collection)
    if col.end_datetime is None:
        raise RuntimeError(f"CDS collection '{collection}' returned no end_datetime")
    return col.end_datetime


def _hourly_availability_cutoff() -> datetime:
    """Return the latest hour for which CDS ERA5-Land hourly data are published."""
    return _cds_end_datetime(_CDS_HOURLY_COLLECTION)


def _daily_availability_cutoff() -> date:
    """Return the latest date for which CDS ERA5-Land daily statistics are published."""
    return _cds_end_datetime(_CDS_DAILY_COLLECTION).date()


def _monthly_availability_cutoff() -> date:
    """Return the latest month for which CDS ERA5-Land monthly means are published."""
    return _cds_end_datetime(_CDS_MONTHLY_COLLECTION).date().replace(day=1)


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """Normalize longitude coordinates from 0–360 to –180/180."""
    return ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)


def _deaccumulate_tp(tp: xr.DataArray, time_dim: str = "valid_time") -> xr.DataArray:
    """Deaccumulate ERA5-Land total precipitation to hourly rates.

    ERA5-Land ``tp`` accumulates continuously within each calendar day,
    resetting to zero at 01:00 UTC.  At the reset boundary the raw diff is
    negative; the correct hourly rate for that step is the current ``tp``
    value itself (accumulated since the new period start), so we substitute
    the current value rather than clipping to zero.  This recovers all
    precipitation with no lost hours.
    """
    diff = tp.diff(time_dim)
    # At the reset boundary diff is negative; the current tp IS the 1-hour rate
    hourly = diff.where(diff >= 0, tp.isel({time_dim: slice(1, None)}))
    return xr.concat([tp.isel({time_dim: slice(0, 1)}), hourly], dim=time_dim)


# ---------------------------------------------------------------------------
# Earth Data Hub (EDH) plugins — lazy Zarr access, no per-period downloads
# ---------------------------------------------------------------------------

# Hourly ERA5-Land, Zarr v3. The store this replaced stopped advancing at 2026-05-31 while the
# daily store kept moving, because EDH applies updates only to its v3 stores (CLIM-955).
# Verified identical in every respect that matters to the plugins below: 0-360 longitude, the
# same latitude range, a `valid_time` axis and the same 50 variables including t2m and tp.
_EDH_HOURLY_URL = "https://api.earthdatahub.destine.eu/era5/era5-land-v0.zarr"
# Daily ERA5-Land: new DestinE API, Zarr v3, requires an additional subscription
_EDH_DAILY_URL = "https://api.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
_EDH_API_KEY_ENV = "EDH_API_KEY"


# A confirmed verdict never changes for a given URL, so it is cached for the process.
_edh_is_v3_cache: dict[str, bool] = {}
# An *undeterminable* verdict is cached only briefly — a deliberate middle course. Memoising it
# permanently would let one transient timeout disable the check for the process. Not caching it
# at all is worse: `_latest_available` reopens the store for every period, so a store fsspec can
# read while this separate request times out would cost a 15 s probe *per period*. A short TTL
# bounds the cost and still recovers on its own.
_EDH_UNKNOWN_TTL_SECONDS = 300.0
_edh_unknown_until: dict[str, float] = {}


def _edh_is_zarr_v3(url: str) -> bool | None:
    """Whether an EDH store is Zarr v3. None when the question cannot be answered.

    A v3 store answers ``zarr.json``; anything else does not. That single probe is the whole
    test — OCS reads only v3, so there is nothing to learn from identifying *which* older
    format a store uses.

    The three-way result matters. A definite 404 means the store is not v3 and the caller
    should refuse. A timeout or a 5xx means we do not know, and a transient network fault must
    never be turned into a refusal of a store that reads perfectly well.
    """
    cached = _edh_is_v3_cache.get(url)
    if cached is not None:
        return cached
    if monotonic() < _edh_unknown_until.get(url, 0.0):
        return None  # probed recently and could not tell; do not re-probe per period
    verdict: bool | None = None
    try:
        # The credential-free URL plus an explicit header: `urlopen` does not turn
        # `user:pass@host` userinfo into Basic Auth the way fsspec does, so probing the
        # authenticated URL fails for anyone using EDH_API_KEY.
        request = Request(f"{url}/zarr.json", headers=_edh_auth_header(url))
        with urlopen(request, timeout=15) as response:  # noqa: S310  # fixed https EDH host
            document = json.loads(response.read())
        # Only a document that actually states its format is evidence. A 200 of an unexpected
        # shape — `[]`, or a proxy's JSON error page — says nothing about the store, so it is
        # unknown rather than non-v3: refusing on it would break a store that reads fine.
        declared = document.get("zarr_format") if isinstance(document, dict) else None
        verdict = declared >= 3 if isinstance(declared, int) else None
    except HTTPError as error:
        # 404 is an answer: there is no v3 metadata document here. Any other status is not.
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
    """Refuse a store that is not Zarr v3, closing it first.

    OCS reads only Zarr v3. EDH keeps superseded stores readable and applies updates only to
    the v3 ones, so an older store does not fail — it silently stops advancing, and an ingest
    against it reports no new periods, which is indistinguishable from "nothing has been
    published yet" (the same shape as CLIM-952). Refusing is what makes that visible.

    Only a *confirmed* non-v3 store is refused. An undeterminable probe leaves the store
    usable, because a network fault is not evidence about the store's format.
    """
    if _edh_is_zarr_v3(url) is not False:
        return
    latest = ""
    time_dim = next((name for name in ("valid_time", "time") if name in ds.coords), None)
    if time_dim and ds.sizes.get(time_dim):
        try:
            # A lazy remote read, which can fail on its own. It only sharpens the message.
            latest = f" Its latest timestamp is {str(ds[time_dim].isel({time_dim: -1}).values)[:16]}."
        except Exception:  # noqa: BLE001 — enrichment is optional, the refusal is not
            logger.debug("Could not read the latest timestamp of %s", url, exc_info=True)
    ds.close()
    raise RuntimeError(
        f"Earth Data Hub store {url} is not Zarr v3, and OCS reads only v3 stores. EDH applies "
        f"updates only to its Zarr v3 stores, so this one no longer advances.{latest} "
        f"Repoint it at the v3 store for this dataset. See CLIM-955."
    )


def _edh_open_zarr(url: str) -> xr.Dataset:
    """Open an Earth Data Hub Zarr store.

    Injects the ``EDH_API_KEY`` environment variable as HTTP Basic Auth
    (username ``edh``, password = key).  Falls back to netrc when the
    variable is not set.

    Every store OCS pins is Zarr v3, which carries its metadata inline, so nothing here has to
    say how to find it — the `consolidated` switch that older stores needed is gone with them.
    `_require_zarr_v3` is what keeps that assumption honest: EDH leaves superseded stores
    readable, and they fail by going quiet rather than by erroring (CLIM-955).
    """
    opened = xr.open_zarr(
        _edh_authenticated_url(url),
        storage_options={"client_kwargs": {"trust_env": True}},
    )
    _require_zarr_v3(url, opened)
    return opened  # type: ignore[no-any-return]


def _edh_authenticated_url(url: str) -> str:
    """Inject ``EDH_API_KEY`` as HTTP Basic Auth, leaving netrc to handle it when unset."""
    token = os.environ.get(_EDH_API_KEY_ENV, "")
    if not token:
        return url
    parts = urlparse(url)
    return urlunparse(parts._replace(netloc=f"edh:{token}@{parts.netloc}"))


def _edh_auth_header(url: str) -> dict[str, str]:
    """Basic Auth for the plain ``urlopen`` probe, from either credential source.

    Needed because ``urlopen`` authenticates like neither of the paths that work elsewhere:
    it does not read netrc (fsspec does, via ``trust_env=True``) and it does not turn
    ``user:pass@host`` userinfo into an Authorization header (fsspec does that too). So both
    of the documented ways to give OCS an EDH credential leave the probe unauthenticated, it
    takes a 401, returns None, and the superseded-store warning silently never fires — the
    exact failure it exists to prevent.
    """
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
    # `or "edh"`: the netrc entry EDH documents carries only a password, and Python then
    # reports an empty login — which would build `:key` here while the EDH_API_KEY path builds
    # `edh:key`. EDH happens to accept both today, so this is not a live failure, but two auth
    # paths disagreeing means one of them works only by the service's tolerance.
    return _basic_auth(login or "edh", password or "")


def _basic_auth(login: str, password: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{login}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


class _ERA5LandEDHBase(BaseDatasetPlugin):
    """Shared cache and region logic for EDH Zarr plugins."""

    _edh_url: str  # set by subclass
    _edh_lon_360: bool = False  # True when longitude is stored 0–360

    def __init__(self, variable: str) -> None:
        if variable not in _CDS_VARIABLE_NAMES:
            raise ValueError(
                f"{type(self).__name__}: unsupported variable {variable!r}; expected one of {list(_CDS_VARIABLE_NAMES)}"
            )
        self.variable = variable
        self._cache_lock = Lock()
        self._cached_bbox: tuple[float, float, float, float] | None = None
        self._cached_region: xr.Dataset | None = None

    def _region_for_bbox(self, bbox: list[float]) -> xr.Dataset:
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))
        with self._cache_lock:
            if self._cached_region is not None and self._cached_bbox == bbox_tuple:
                return self._cached_region
            self._close_cached_locked()
            xmin, ymin, xmax, ymax = bbox_tuple
            ds = _edh_open_zarr(self._edh_url)
            # Extend bbox by half a grid step (0.05°) to avoid floating-point
            # boundary exclusion (e.g. 360 - 10.1 = 349.8999... misses the 349.9
            # grid point). CDS API is inclusive of boundary points; this aligns
            # the EDH selection so both sources return the same spatial grid.
            _eps = 0.05
            if self._edh_lon_360:
                xmin_sel = (xmin % 360) - _eps
                xmax_sel = (xmax % 360) + _eps
            else:
                xmin_sel, xmax_sel = xmin - _eps, xmax + _eps
            self._cached_region = ds[[self.variable]].sel(
                latitude=slice(ymax + _eps, ymin - _eps),
                longitude=slice(xmin_sel, xmax_sel),
            )
            self._cached_bbox = bbox_tuple
            return self._cached_region

    def _apply_transforms(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable == "t2m":
            return kelvin_to_celsius(ds, {"variable": self.variable})
        if self.variable == "tp":
            return metres_to_mm(ds, {"variable": self.variable})
        return ds

    def close(self) -> None:
        with self._cache_lock:
            self._close_cached_locked()

    def _close_cached_locked(self) -> None:
        if self._cached_region is not None:
            self._cached_region.close()
        self._cached_region = None
        self._cached_bbox = None

    def _latest_available(self) -> str:
        """Return the latest available timestamp from the EDH Zarr store."""
        ds = _edh_open_zarr(self._edh_url)
        try:
            return str(np.datetime64(ds.valid_time.isel(valid_time=-1).values, "h"))
        finally:
            ds.close()


_GRID_MAPPING_NAMES = ("spatial_ref", "crs")
"""Scalar coordinates that carry the CRS rather than data, so they must survive the cleanup."""


_EXPVER_DIM = "expver"


def _collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Merge ERA5's ``expver`` dimension into one series, preferring final over preliminary.

    A CDS request spanning the boundary between final ERA5 (``expver=1``) and preliminary
    ERA5T (``expver=5``) returns both along an ``expver`` dimension, each slice NaN wherever
    the other supplies data. Left in place it doubles the array along a dimension the rest of
    OCS does not model: aggregation then failed with ``boolean index did not match indexed
    array`` on a mask half the size of the data (CLIM-923).

    ``_drop_auxiliary_variables`` does not remove it — that keeps every dimension by design and
    only strips scalar coordinates that became phantom variables. Both guards are needed.

    Collapsed at the download boundary rather than on the way out, because a reduction applied
    first corrupts the result: ``sum("t")`` over the all-NaN slice yields 0.0 rather than
    propagating the gap.

    Ascending ``expver`` order gives final data precedence without hard-coding 5 as the
    preliminary value.
    """
    if _EXPVER_DIM not in ds.dims:
        # Also covers the scalar case, where it is a coordinate rather than a dimension.
        return ds.drop_vars(_EXPVER_DIM, errors="ignore")
    versions = sorted(ds[_EXPVER_DIM].values.tolist())
    merged = ds.sel({_EXPVER_DIM: versions[0]}, drop=True)
    for version in versions[1:]:
        merged = merged.combine_first(ds.sel({_EXPVER_DIM: version}, drop=True))
    return merged.astype({name: ds[name].dtype for name in ds.data_vars})


def _drop_auxiliary_variables(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Keep ``variable``, the dimension coordinates and the CRS, dropping everything else.

    ERA5 sources attach scalar ensemble-member coordinates (``number``, value 0 for
    ERA5-Land reanalysis; sometimes ``expver``) that otherwise persist as phantom data
    variables — which then load as a spurious ``bands`` dimension downstream (e.g. anomalies
    came out with a stray ``number`` band). Shared by the daily ERA5-Land fetch paths.

    The grid mapping is exempt. "Everything that is not a dimension coordinate" would also
    take rioxarray's ``spatial_ref``, which is where the CRS lives — and since grid inference
    falls back to EPSG:4326 when a cube carries none, ERA5-Land would still be recorded
    correctly, by luck rather than because the data said so. A projected source cleaned the
    same way would be silently mislabelled.
    """
    keep = {variable, *ds.dims, *_GRID_MAPPING_NAMES}
    declared = ds[variable].attrs.get("grid_mapping") if variable in ds else None
    if isinstance(declared, str) and declared:
        keep.add(declared)
    extras = [v for v in ds.variables if v not in keep]
    return ds.drop_vars(extras, errors="ignore")


class ERA5LandEDHDailyPlugin(_ERA5LandEDHBase):
    """Streaming plugin for daily ERA5-Land from the Earth Data Hub Zarr store.

    Unlike the CDS daily statistics dataset, the EDH daily store includes
    ``total_precipitation`` (``tp``), enabling daily precipitation ingestion.

    Requires a Standard API key configured in netrc for ``api.earthdatahub.destine.eu``
    or the ``EDH_API_KEY`` environment variable.
    """

    _edh_url = _EDH_DAILY_URL
    _edh_lon_360 = True  # longitude stored as 0–360
    max_concurrency = 4
    commit_batch_size = 30

    def __init__(self, variable: str, **_: Any) -> None:
        super().__init__(variable)

    async def periods(self, start: str, end: str) -> list[str]:
        latest = await asyncio.to_thread(self._latest_available)
        return daily_period_ids(start, end, cutoff=latest)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        region = self._region_for_bbox(bbox)
        timestamp = np.datetime64(period_id, "D").astype("datetime64[ns]")
        ds = region.sel(valid_time=slice(timestamp, timestamp))[[self.variable]].load()
        if self._edh_lon_360:
            ds = _normalize_lon(ds)
        ds = ds.rename({"longitude": "x", "latitude": "y", "valid_time": "t"})
        return _drop_auxiliary_variables(self._apply_transforms(ds), self.variable)


class ERA5LandTempDailyPlugin(ERA5LandEDHDailyPlugin):
    """Canonical daily temperature plugin: EDH daily store for history, CDS for the recent tail.

    Uses pre-computed daily means from the EDH daily store.  Falls back to
    the CDS ``derived-era5-land-daily-statistics`` dataset for the ~4-week
    tail not yet published by EDH.  Only supports ``t2m``; precipitation is
    not available from the CDS daily statistics product.
    """

    def __init__(self, variable: str, **_: Any) -> None:
        super().__init__(variable=variable)
        self._cds_plugin = ERA5LandDailyTemperaturePlugin() if variable == "t2m" else None

    async def periods(self, start: str, end: str) -> list[str]:
        if self._cds_plugin is None:
            # tp: EDH only
            return await super().periods(start, end)
        cds_cutoff = await asyncio.to_thread(_daily_availability_cutoff)
        return daily_period_ids(start, end, cutoff=cds_cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        edh_latest = self._latest_available()
        if self._cds_plugin is None or period_id <= edh_latest:
            ds = super().fetch_period(period_id, bbox)
        else:
            ds = self._cds_plugin.fetch_period(period_id, bbox)
        # The EDH path is already cleaned by the base; this also covers the CDS-tail source.
        return _drop_auxiliary_variables(ds, self.variable)


class ERA5LandEDHPrecipitationDailyPlugin(_ERA5LandEDHBase):
    """Daily total precipitation from the EDH hourly store with proper deaccumulation.

    Fetches 25 consecutive hourly ``tp`` values (the target day plus the
    preceding hour for deaccumulation context), deaccumulates by differencing
    and clipping negatives (which arise at the 12 UTC forecast reset), then
    sums the 24 target-day hours to produce a daily total in mm.

    This avoids the EDH daily Zarr store, whose ``tp`` field is a daily mean
    of accumulated values — a quantity with no physical interpretation.

    Requires a Standard API key configured in netrc for ``api.earthdatahub.destine.eu``
    or the ``EDH_API_KEY`` environment variable.
    """

    _edh_url = _EDH_HOURLY_URL
    _edh_lon_360 = True
    max_concurrency = 4
    commit_batch_size = 30

    def __init__(self, **_: Any) -> None:
        super().__init__("tp")

    async def periods(self, start: str, end: str) -> list[str]:
        latest_str = await asyncio.to_thread(self._latest_available)
        return daily_period_ids(start, end, cutoff=latest_str)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return _drop_auxiliary_variables(self._fetch_daily_sync(period_id, bbox), self.variable)

    def _fetch_daily_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        from open_climate_service import config as api_config

        region = self._region_for_bbox(bbox)
        day = date.fromisoformat(period_id)
        utc_offset = api_config.get_utc_offset()

        if utc_offset == 0:
            # ERA5-Land tp accumulates from 00:00 UTC each day. The full 24h daily
            # total for day D is stored at 00:00 UTC of day D+1 (step 24). Using
            # this avoids missing the 23:00–midnight precipitation hour.
            next_midnight = np.datetime64(f"{day + timedelta(days=1)}T00", "h").astype("datetime64[ns]")
            ds = region.sel(valid_time=next_midnight)[["tp"]].load()
            if self._edh_lon_360:
                ds = _normalize_lon(ds)
            ds = metres_to_mm(ds, {"variable": "tp"})
        else:
            # For non-UTC instances: aggregate deaccumulated hourly values within
            # the local-day window (e.g. UTC+3 → 21:00 UTC D-1 to 20:00 UTC D).
            local_start = datetime(day.year, day.month, day.day) - utc_offset
            local_end = local_start + timedelta(hours=23)
            window_start = local_start - timedelta(hours=1)
            start_np = np.datetime64(local_start, "h").astype("datetime64[ns]")
            end_np = np.datetime64(local_end, "h").astype("datetime64[ns]")
            window_np = np.datetime64(window_start, "h").astype("datetime64[ns]")

            window = region.sel(valid_time=slice(window_np, end_np))[["tp"]].load()
            if self._edh_lon_360:
                window = _normalize_lon(window)
            deacc = _deaccumulate_tp(window["tp"])
            daily_total = deacc.sel(valid_time=slice(start_np, end_np)).sum("valid_time")
            ds = metres_to_mm(xr.Dataset({"tp": daily_total}), {"variable": "tp"})

        ts = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return ds.expand_dims({"valid_time": [ts]}).rename({"longitude": "x", "latitude": "y", "valid_time": "t"})


class ERA5LandPrecipDailyPlugin(ERA5LandEDHPrecipitationDailyPlugin):
    """Canonical daily precipitation plugin: EDH hourly for history, CDS hourly for the recent tail.

    Extends ``ERA5LandEDHPrecipitationDailyPlugin`` with a CDS fallback so
    periods beyond EDH's ~4-week publishing lag are filled automatically.
    Timezone-aware: the local-day window respects ``utc_offset_hours`` in
    climate-service.yaml on both the EDH and CDS paths.  Serves both
    ``era5land_precipitation_daily`` and ``era5land_precipitation_daily_from_hourly``.
    """

    max_concurrency = 1

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self._cds_hourly = ERA5LandPrecipitationPlugin()

    async def periods(self, start: str, end: str) -> list[str]:
        cds_cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        from open_climate_service import config as api_config

        utc_offset = api_config.get_utc_offset()
        # Subtract the UTC offset so we only include local days whose final
        # UTC hour is already published (e.g. UTC+3 needs data until 20:00 UTC).
        cutoff_local_dt = cds_cutoff - utc_offset if hasattr(cds_cutoff, "__sub__") else cds_cutoff
        cutoff_local = cutoff_local_dt.date() if hasattr(cutoff_local_dt, "date") else cutoff_local_dt
        return daily_period_ids(start, end, cutoff=cutoff_local)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        edh_latest = self._latest_available()
        if period_id <= edh_latest[:10]:
            # Delegate rather than repeat the parent's cleaning, so a future override of this
            # method cannot silently shed it the way this one did (CLIM-923).
            return super().fetch_period(period_id, bbox)
        return _drop_auxiliary_variables(self._fetch_cds_daily_sync(period_id, bbox), self.variable)

    def _fetch_cds_daily_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        from open_climate_service import config as api_config

        day = date.fromisoformat(period_id)
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))
        utc_offset = api_config.get_utc_offset()

        with self._cds_hourly._cache_lock:
            if self._cds_hourly._cached_month != (day.year, day.month) or self._cds_hourly._cached_bbox != bbox_tuple:
                self._cds_hourly._cached_ds = self._cds_hourly._fetch_month(day.year, day.month, bbox_tuple)
                self._cds_hourly._cached_month = (day.year, day.month)
                self._cds_hourly._cached_bbox = bbox_tuple
            monthly_ds = self._cds_hourly._cached_ds

        assert monthly_ds is not None

        if utc_offset == 0:
            # Sum T01..T00-next (24 deaccumulated hours) to match the EDH step-24 value
            next_day = day + timedelta(days=1)
            day_start = np.datetime64(f"{day}T01", "h").astype("datetime64[ns]")
            day_end = np.datetime64(f"{next_day}T00", "h").astype("datetime64[ns]")
        else:
            local_start = datetime(day.year, day.month, day.day) - utc_offset
            local_end = local_start + timedelta(hours=24)
            day_start = np.datetime64(local_start, "h").astype("datetime64[ns]")
            day_end = np.datetime64(local_end, "h").astype("datetime64[ns]")

        daily_total = monthly_ds.sel(t=slice(day_start, day_end))["tp"].sum("t")
        ds = xr.Dataset({"tp": daily_total})
        ts = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return ds.expand_dims({"valid_time": [ts]}).rename({"valid_time": "t"})


class ERA5LandTempDailyFromHourlyPlugin(_ERA5LandEDHBase):
    """Daily 2m temperature: EDH hourly for history, CDS hourly for the recent tail.

    Computes the daily mean from 24 individual hourly values, respecting
    ``utc_offset_hours`` from climate-service.yaml.  EDH covers history
    efficiently via lazy Zarr access; the CDS ERA5-Land hourly dataset fills
    the ~4-week recent tail not yet published by EDH.
    """

    _edh_url = _EDH_HOURLY_URL
    _edh_lon_360 = True
    max_concurrency = 4
    commit_batch_size = 30

    def __init__(self, **_: Any) -> None:
        super().__init__("t2m")
        self._cds_hourly = ERA5LandCDSHourlyPlugin(variable="t2m")

    async def periods(self, start: str, end: str) -> list[str]:
        cds_cutoff = await asyncio.to_thread(_hourly_availability_cutoff)
        from open_climate_service import config as api_config

        utc_offset = api_config.get_utc_offset()
        # Subtract the UTC offset so we only include local days whose final
        # UTC hour is already published (e.g. UTC+3 needs data until 20:00 UTC).
        cutoff_local_dt = cds_cutoff - utc_offset if hasattr(cds_cutoff, "__sub__") else cds_cutoff
        cutoff_local = cutoff_local_dt.date() if hasattr(cutoff_local_dt, "date") else cutoff_local_dt
        return daily_period_ids(start, end, cutoff=cutoff_local)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        edh_latest = self._latest_available()
        if period_id <= edh_latest[:10]:
            cube = self._fetch_daily_sync(period_id, bbox)
        else:
            cube = self._fetch_cds_daily_sync(period_id, bbox)
        return _drop_auxiliary_variables(cube, self.variable)

    def _fetch_daily_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        from open_climate_service import config as api_config

        region = self._region_for_bbox(bbox)
        day = date.fromisoformat(period_id)
        utc_offset = api_config.get_utc_offset()

        local_start = datetime(day.year, day.month, day.day) - utc_offset
        local_end = local_start + timedelta(hours=23)
        start_np = np.datetime64(local_start, "h").astype("datetime64[ns]")
        end_np = np.datetime64(local_end, "h").astype("datetime64[ns]")

        window = region.sel(valid_time=slice(start_np, end_np))[["t2m"]].load()
        if self._edh_lon_360:
            window = _normalize_lon(window)

        daily_mean = window["t2m"].mean("valid_time")
        ds = kelvin_to_celsius(xr.Dataset({"t2m": daily_mean}), {"variable": "t2m"})
        ts = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return ds.expand_dims({"valid_time": [ts]}).rename({"longitude": "x", "latitude": "y", "valid_time": "t"})

    def _fetch_cds_daily_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        from open_climate_service import config as api_config

        day = date.fromisoformat(period_id)
        bbox_tuple = cast(tuple[float, float, float, float], tuple(map(float, bbox)))
        utc_offset = api_config.get_utc_offset()

        with self._cds_hourly._cache_lock:
            if self._cds_hourly._cached_month != (day.year, day.month) or self._cds_hourly._cached_bbox != bbox_tuple:
                self._cds_hourly._cached_ds = self._cds_hourly._fetch_month(day.year, day.month, bbox_tuple)
                self._cds_hourly._cached_month = (day.year, day.month)
                self._cds_hourly._cached_bbox = bbox_tuple
            monthly_ds = self._cds_hourly._cached_ds

        assert monthly_ds is not None
        local_start = datetime(day.year, day.month, day.day) - utc_offset
        local_end = local_start + timedelta(hours=23)
        start_np = np.datetime64(local_start, "h").astype("datetime64[ns]")
        end_np = np.datetime64(local_end, "h").astype("datetime64[ns]")

        daily_mean = monthly_ds.sel(t=slice(start_np, end_np))["t2m"].mean("t")
        ds = xr.Dataset({"t2m": daily_mean})
        ts = np.datetime64(period_id, "D").astype("datetime64[ns]")
        return ds.expand_dims({"valid_time": [ts]}).rename({"valid_time": "t"})


# ---------------------------------------------------------------------------
# ERA5-Land day-of-year climate normals (WMO climatology)
# ---------------------------------------------------------------------------

_NORMALS_DAYOFYEAR_DIM = "dayofyear"
_NORMALS_DEFAULT_PERIOD = (1991, 2020)
_NORMALS_DEFAULT_SMOOTHING = 31


class ERA5LandNormalsPlugin(BaseDatasetPlugin):
    """Streaming plugin that computes WMO day-of-year climate normals from ERA5-Land.

    Reads the reference period directly from the Earth Data Hub daily ERA5-Land Zarr
    store so no prior 30-year ingestion is required. The result is stored with a real
    ``dayofyear`` dimension (1..366) — a non-temporal stepping axis rendered by the
    map viewer's generic dimension slider.

    Args (from ``ingestion.params``):
        variable: output variable name written into the store.
        edh_variable: EDH variable name to read (e.g. ``t2m``, ``tp``).
        period: ``[start_year, end_year]`` reference period (default 1991–2020).
        smoothing_window: circular rolling-mean window in days (default 31; 0 disables).
        unit_transform: ``kelvin_to_celsius`` | ``metres_to_mm`` | ``none``.
    """

    max_concurrency = 1
    commit_batch_size = 30
    # Non-temporal stepping axis: the orchestrator appends along dayofyear (1..366),
    # not a datetime ``t``. Declared as a class attribute since the contract no longer
    # has a ``probe()`` to report it; the grid (shape/dtype/CRS) is inferred from the
    # first fetched period.
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
        # circular_rolling_mean assumes a centred, odd window; 0 disables smoothing.
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
            return [str(int(d)) for d in self._climatology[_NORMALS_DAYOFYEAR_DIM].values]
        # periods() runs before the first fetch (without a bbox), so the climatology
        # isn't computed yet. Derive the day count from the reference period's leap
        # status: a period that spans no leap year has no day 366, and enumerating it
        # would KeyError when fetch_period selects that day-of-year off the climatology.
        start_year, end_year = self.period
        has_leap = any(calendar.isleap(year) for year in range(start_year, end_year + 1))
        return [str(d) for d in range(1, (366 if has_leap else 365) + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_sync, period_id, bbox)

    def _fetch_sync(self, period_id: str, bbox: list[float]) -> xr.Dataset:
        clim = self._ensure_climatology(bbox)
        selected = clim.sel({_NORMALS_DAYOFYEAR_DIM: [int(period_id)]})
        return _drop_auxiliary_variables(selected, self.variable)

    def _ensure_climatology(self, bbox: list[float]) -> xr.Dataset:
        with self._lock:
            if self._climatology is None:
                self._climatology = self._compute_climatology(bbox)
            return self._climatology

    def _compute_climatology(self, bbox: list[float]) -> xr.Dataset:
        region = self._load_reference(bbox)
        # earthkit computes the day-of-year mean (1..366), handling the calendar/leap-year
        # binning and preserving dask laziness. WMO circular smoothing stays a post-step:
        # earthkit's rolling reduction lives in `temporal` and does not wrap the year
        # boundary (ecmwf/earthkit-transforms#103), which is the only part we supply.
        normals = cast(xr.Dataset, ek_climatology.daily_mean(region, time_dim="valid_time"))
        if self.smoothing_window > 0:
            normals[self.variable] = circular_rolling_mean(
                normals[self.variable], self.smoothing_window, _NORMALS_DAYOFYEAR_DIM
            )
        normals = self._apply_unit_transform(normals)
        return normals.load()

    def _apply_unit_transform(self, ds: xr.Dataset) -> xr.Dataset:
        if self.unit_transform in (None, "", "none"):
            return ds
        if self.unit_transform == "kelvin_to_celsius":
            return kelvin_to_celsius(ds, {"variable": self.variable})
        if self.unit_transform == "metres_to_mm":
            return metres_to_mm(ds, {"variable": self.variable})
        raise ValueError(f"Unknown unit_transform {self.unit_transform!r}")

    def _load_reference(self, bbox: list[float]) -> xr.Dataset:
        """Load the reference-period ERA5-Land data from EDH as a (valid_time, y, x) dataset."""
        start_year, end_year = self.period
        eps = 0.05
        xmin, ymin, xmax, ymax = map(float, bbox)
        ds = _edh_open_zarr(_EDH_DAILY_URL)
        try:
            base = ds[[self.edh_variable]].sel(
                latitude=slice(ymax + eps, ymin - eps),
                valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
            )
            # EDH stores longitude in [0, 360). Map the WGS84 (-180/180) bbox onto it.
            # A bbox that straddles the 0°/360° seam (e.g. spans the prime meridian)
            # gives lon_min > lon_max, so select the two pieces and concatenate; a
            # (near-)global span takes the whole axis. A single `slice(xmin%360, xmax%360)`
            # would silently return empty/reversed results for these cases.
            if xmax - xmin >= 360 - eps:
                region = base
            else:
                lon_min, lon_max = xmin % 360, xmax % 360
                if lon_min <= lon_max:
                    region = base.sel(longitude=slice(lon_min - eps, lon_max + eps))
                else:
                    region = xr.concat(
                        [
                            base.sel(longitude=slice(lon_min - eps, 360.0)),
                            base.sel(longitude=slice(0.0, lon_max + eps)),
                        ],
                        dim="longitude",
                    )
            # Eagerly load the whole reference-period region (≈30 years daily for the
            # bbox) before reducing. Fine for a country-sized instance extent; a very
            # large/continental extent would want a chunked (dask) groupby-mean instead
            # of materialising the full period in memory.
            region = region.load()
        finally:
            ds.close()
        # Back to WGS84 (-180/180), ascending, with canonical dim names.
        region = region.assign_coords(longitude=((region.longitude + 180) % 360) - 180)
        region = region.sortby("longitude").rename({"longitude": "x", "latitude": "y"})
        if self.edh_variable != self.variable:
            region = region.rename({self.edh_variable: self.variable})
        return region
