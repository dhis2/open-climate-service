"""CHIRPS3 plugins for per-period streaming ingest.

CHIRPS3 keeps the first vertical slice small:
- daily and monthly periods
- grid inferred from the first fetched period
- one remote raster per period
- straightforward bbox clipping into the requested extent

These plugins stay intentionally source-focused. They know how to derive the
available periods and fetch one period, but they do not know anything about job
state, artifact records, or store mutation beyond returning one dataset.
"""

from __future__ import annotations

import asyncio
import calendar
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, TypeVar

import xarray as xr

from open_climate_service.shared.time import normalize_period_string
from open_climate_service.streaming import (
    BaseDatasetPlugin,
    daily_period_ids,
    monthly_period_ids,
    normalize_period,
)

_CHIRPS3_NODATA = -9999.0

_T = TypeVar("_T")


def _latest_published(
    candidates: Callable[[int], _T],
    url_for: Callable[[_T], str],
    *,
    lookback: int,
    description: str,
) -> _T:
    """Return the most recent candidate period whose raster is actually published.

    Issues a HEAD request per candidate, walking backwards from the present, so the
    availability cutoff reflects real CDN state rather than a hardcoded lag assumption —
    CHIRPS3's publication delay varies. Shared by the daily and monthly plugins, which
    differ only in what a candidate is and how its URL is built.

    Args:
        candidates: maps a step count (0 = most recent candidate) to a candidate period.
        url_for: builds the raster URL to probe for a candidate.
        lookback: how many steps back to try before giving up.
        description: used in the error message when nothing is published.

    Raises:
        RuntimeError: if no candidate within ``lookback`` steps is published.
    """
    import httpx

    for step in range(lookback):
        candidate = candidates(step)
        response = httpx.head(url_for(candidate), timeout=10, follow_redirects=True)
        if response.status_code == 200:
            return candidate
    raise RuntimeError(f"No published CHIRPS3 {description} found in the last {lookback} steps")


class CHIRPS3DailyPlugin(BaseDatasetPlugin):
    """Streaming plugin for CHIRPS3 daily precipitation.

    Args:
        stage: ``final`` or ``prelim`` CHIRPS3 release stage.
        flavor: delivery flavor for final products. ``prelim`` currently only
            supports the ``sat`` path.
    """

    max_concurrency = 4
    commit_batch_size = 30

    def __init__(self, stage: str = "final", flavor: str = "rnl", **_: Any) -> None:
        if stage not in {"final", "prelim"}:
            raise ValueError(f"stage must be 'final' or 'prelim', got {stage!r}")
        if stage == "final" and flavor not in {"rnl", "sat"}:
            raise ValueError(f"For stage='final', flavor must be 'rnl' or 'sat', got {flavor!r}")
        if stage == "prelim" and flavor != "sat":
            raise ValueError(f"For stage='prelim', flavor must be 'sat', got {flavor!r}")
        self.stage = stage
        self.flavor = flavor

    async def periods(self, start: str, end: str) -> list[str]:
        """Return ordered daily periods, clamped to the latest complete month."""
        cutoff = await asyncio.to_thread(self._availability_cutoff)
        return daily_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetch one day, clip it to the requested bbox, and return a one-step dataset.

        A regular (blocking) method — the framework runs it in a worker thread.
        """
        import rioxarray

        day = date.fromisoformat(period_id)
        da = rioxarray.open_rasterio(self._url_for_day(day), chunks=None, masked=True)
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Expected DataArray from CHIRPS3 raster read, got {type(da).__name__}")
        return normalize_period(da, variable="precip", period=period_id, nodata=_CHIRPS3_NODATA, bbox=bbox)

    def _availability_cutoff(self) -> date:
        """Return the last day of the most recently published complete month."""

        def candidate(step: int) -> date:
            today = datetime.now(UTC).date()
            months_back = step + 1  # the current month is never complete
            year, month = divmod(today.year * 12 + today.month - 1 - months_back, 12)
            month += 1
            return date(year, month, calendar.monthrange(year, month)[1])

        return _latest_published(candidate, self._url_for_day, lookback=6, description="daily month")

    def _url_for_day(self, day: date) -> str:
        """Build the remote CHIRPS3 raster URL for one day."""
        if self.stage == "final":
            return (
                f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{self.flavor}/cogs/{day.year}/"
                f"chirps-v3.0.{self.flavor}.{day.year}.{day.month:02d}.{day.day:02d}.cog"
            )
        return (
            f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/{day.year}/"
            f"chirps-v3.0.prelim.{day.year}.{day.month:02d}.{day.day:02d}.tif"
        )


class CHIRPS3MonthlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for CHIRPS3 monthly precipitation.

    One global COG per calendar month, so no ``stage``/``flavor`` variants and no
    constructor arguments. The archive runs from 1981-01.

    Stored as **mm/day**, not the raw monthly total. That is a real conversion rather than
    the relabel the daily plugin gets away with: the source raster is a monthly *total* in
    mm — verified against the sum of the same month's dailies, which matches to four decimal
    places — so the rate is ``total / days_in_month``. It matters because every other monthly
    precipitation dataset here is mm/d, including ``chirps3_precipitation_monthly_normal_1991_2020``,
    which is the natural partner for a monthly anomaly. Storing the total under the same
    units would make that pairing silently wrong by a factor of ~30.
    """

    max_concurrency = 4
    commit_batch_size = 12

    async def periods(self, start: str, end: str) -> list[str]:
        """Return ordered ``YYYY-MM`` periods, clamped to the latest published month."""
        cutoff = await asyncio.to_thread(self._availability_cutoff)
        return monthly_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Fetch one month, clip it to the bbox, then convert the total to a mean daily rate."""
        import rioxarray

        year, month = self._parse_month(period_id)
        da = rioxarray.open_rasterio(self._url_for_month(year, month), chunks=None, masked=True)
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Expected DataArray from CHIRPS3 monthly raster read, got {type(da).__name__}")

        # Clip first, scale second — the order is the whole point (CLIM-951). The source is a
        # global COG (7200 x 2400, tiled 512 x 512), so `normalize_period`'s clip can push down
        # into a windowed read over HTTP range requests, but only while the array is still
        # lazy. Any arithmetic here would instead touch every cell on the globe to keep the
        # handful the bbox asks for: measured at 13 s and 572 MB per period, against 3.5 s and
        # 205 MB when the clip goes first. Over a 439-month ingest that was 81 minutes and a
        # 41.6 GB peak to produce a 41 MB store.
        #
        # Stamp the first of the month as the time coordinate, matching the monthly
        # convention used by the ERA5-Land monthly datasets.
        ds = normalize_period(
            da,
            variable="precip",
            period=f"{year:04d}-{month:02d}-01",
            nodata=_CHIRPS3_NODATA,
            bbox=bbox,
        )

        # The source raster is a monthly *total* in mm, and the store holds mm/day — a real
        # conversion, verified against the sum of the same month's dailies (see the class
        # docstring). Scaling after `normalize_period` needs no guard for the -9999 sentinel:
        # it has already been masked to NaN by then, and NaN / days stays NaN. Attributes and
        # encoding are carried over explicitly because an xarray binary op drops both, and the
        # store's nodata inference reads `_FillValue` off the encoding.
        days_in_month = calendar.monthrange(year, month)[1]
        precip = ds["precip"]
        scaled = precip / days_in_month
        scaled.attrs = dict(precip.attrs)
        scaled.encoding = dict(precip.encoding)
        ds["precip"] = scaled
        return ds

    def _availability_cutoff(self) -> str:
        """Return the ``YYYY-MM`` of the most recently published monthly COG."""

        def candidate(step: int) -> tuple[int, int]:
            today = datetime.now(UTC).date()
            year, month = divmod(today.year * 12 + today.month - 1 - step, 12)
            return year, month + 1

        year, month = _latest_published(
            candidate,
            lambda ym: self._url_for_month(*ym),
            # The monthly final product trails the calendar by a month or two; 24 is ample
            # headroom for a longer-than-usual upstream delay.
            lookback=24,
            description="monthly COG",
        )
        return f"{year:04d}-{month:02d}"

    @staticmethod
    def _parse_month(period_id: str) -> tuple[int, int]:
        """Accept ``YYYY-MM`` or any ``YYYY-MM-DD`` within the month.

        Delegates to the canonical monthly parser so an out-of-range month is rejected here,
        with the plugin's own contract in the message, rather than surfacing further in as a
        ``calendar.IllegalMonthError`` from ``monthrange``.
        """
        try:
            canonical = normalize_period_string(str(period_id), "monthly")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid CHIRPS3 monthly period id {period_id!r}: expected YYYY-MM") from exc
        return int(canonical[:4]), int(canonical[5:7])

    @staticmethod
    def _url_for_month(year: int, month: int) -> str:
        """Build the remote CHIRPS3 monthly COG URL for one month."""
        return f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs/chirps-v3.0.{year}.{month:02d}.cog"
