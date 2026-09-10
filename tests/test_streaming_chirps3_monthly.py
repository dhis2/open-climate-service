"""Tests for the CHIRPS3 monthly plugin and the shared monthly period helper (CLIM-867)."""

from __future__ import annotations

import asyncio
import calendar
from datetime import date
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.plugins.datasets import chirps3
from open_climate_service.plugins.datasets.chirps3 import CHIRPS3MonthlyPlugin
from open_climate_service.shared.time import monthly_period_ids

_NODATA = -9999.0


def _source_raster(total_mm: float = 310.0, with_nodata: bool = False) -> xr.DataArray:
    """A CHIRPS3-shaped monthly raster: a total in mm, with a raw -9999 sentinel.

    Carries an explicit CRS because the real COGs do, and ``normalize_period`` needs one to
    reproject the request bbox for the spatial clip.
    """
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

    values = np.full((1, 2, 2), total_mm, dtype="float32")
    if with_nodata:
        values[0, 0, 1] = _NODATA
    da = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={"band": [1], "y": [-13.0, -14.0], "x": [33.0, 34.0]},
    )
    return da.rio.write_crs(4326)


@pytest.fixture
def stub_raster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a synthetic raster instead of reaching the CHC CDN."""

    def fake_open(url: str, **_: Any) -> xr.DataArray:
        return _source_raster(with_nodata="ocean" in url)

    import rioxarray

    monkeypatch.setattr(rioxarray, "open_rasterio", fake_open)


# --- monthly_period_ids ---------------------------------------------------------------


def test_enumerates_months_across_a_year_boundary() -> None:
    assert monthly_period_ids("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_single_month() -> None:
    assert monthly_period_ids("2025-01", "2025-01") == ["2025-01"]


def test_reversed_range_is_empty() -> None:
    assert monthly_period_ids("2025-03", "2025-01") == []


def test_cutoff_caps_the_end() -> None:
    assert monthly_period_ids("2024-11", "2025-06", cutoff="2024-12") == ["2024-11", "2024-12"]


def test_cutoff_beyond_the_end_is_ignored() -> None:
    assert monthly_period_ids("2025-01", "2025-02", cutoff="2030-01") == ["2025-01", "2025-02"]


@pytest.mark.parametrize("value", ["2024-11-17", "2024-11-01"])
def test_any_day_within_a_month_selects_that_month(value: str) -> None:
    assert monthly_period_ids(value, "2024-12") == ["2024-11", "2024-12"]


def test_accepts_date_objects() -> None:
    assert monthly_period_ids(date(2024, 11, 17), date(2025, 1, 3)) == ["2024-11", "2024-12", "2025-01"]


def test_an_out_of_range_month_does_not_loop_forever() -> None:
    """Regression: month 13 used to hang, not merely return a wrong answer.

    The increment only wraps at exactly 12, so a month of 13 climbed indefinitely while the
    ``(year, month) <= (last_year, last_month)`` guard stayed true — an unbounded loop
    appending to a list. Validation is what makes it unreachable.
    """
    import signal

    def _bail(*_: object) -> None:
        raise TimeoutError("monthly_period_ids did not terminate")

    original = signal.signal(signal.SIGALRM, _bail)
    signal.alarm(5)
    try:
        with pytest.raises(ValueError, match="Invalid monthly period"):
            monthly_period_ids("2025-13", "2026-01")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)


@pytest.mark.parametrize("value", ["2025-13", "2025-00", "2025-1", "2025", "garbage", ""])
def test_rejects_malformed_months(value: str) -> None:
    """Fails fast like ``daily_period_ids`` does via ``date.fromisoformat``."""
    with pytest.raises(ValueError, match="Invalid monthly period"):
        monthly_period_ids(value, "2026-01")


def test_rejects_a_malformed_end_too() -> None:
    with pytest.raises(ValueError, match="Invalid monthly period"):
        monthly_period_ids("2025-01", "2025-13")


def test_era5_land_monthly_uses_the_shared_helper() -> None:
    """It hand-rolled the same month loop before; the third copy is what prompted this."""
    from open_climate_service.plugins.datasets import era5_land

    source = __import__("pathlib").Path(era5_land.__file__).read_text(encoding="utf-8")
    assert "monthly_period_ids(" in source


# --- the mm/d conversion --------------------------------------------------------------


def test_monthly_total_becomes_a_mean_daily_rate(stub_raster: None) -> None:
    """310 mm over a 31-day month is 10 mm/d. The core of CLIM-867's units decision."""
    ds = CHIRPS3MonthlyPlugin().fetch_period("2025-01", [33.0, -14.0, 34.0, -13.0])
    assert float(ds["precip"].mean()) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("period", "days"),
    [("2025-01", 31), ("2025-02", 28), ("2024-02", 29), ("2025-04", 30)],
)
def test_divides_by_the_actual_days_in_month(stub_raster: None, period: str, days: int) -> None:
    """February differs between years, so a fixed 30 or 31 would be wrong twice over."""
    ds = CHIRPS3MonthlyPlugin().fetch_period(period, [33.0, -14.0, 34.0, -13.0])
    assert float(ds["precip"].mean()) == pytest.approx(310.0 / days)
    assert days == calendar.monthrange(int(period[:4]), int(period[5:7]))[1]


def test_nodata_is_masked_not_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the conversion guards the sentinel.

    The monthly COGs declare no nodata, so a masked read leaves a raw -9999 in the array.
    Dividing it by the day count would give ~-322.5, which no longer matches the sentinel
    normalize_period masks on — so it would be stored as a plausible negative rainfall rate.
    """
    import rioxarray

    monkeypatch.setattr(rioxarray, "open_rasterio", lambda url, **_: _source_raster(with_nodata=True))
    ds = CHIRPS3MonthlyPlugin().fetch_period("2025-01", [33.0, -14.0, 34.0, -13.0])
    values = ds["precip"].values
    assert np.isnan(values).sum() == 1, "the sentinel cell should be NaN"
    finite = values[np.isfinite(values)]
    assert finite.min() > 0, "no negative rate should survive"
    assert not np.any(np.isclose(finite, _NODATA / 31, atol=1.0)), "the sentinel was scaled instead of masked"


def test_the_clip_runs_before_the_conversion(stub_raster: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordering is the fix, so assert the mechanism and not just the result (CLIM-951).

    The source is a global COG. `normalize_period` clips it, and that clip only becomes a
    windowed read while the array is still lazy — any arithmetic beforehand materialises the
    whole globe to keep the handful of cells the bbox asks for. Guarding the *result* alone
    would not catch a regression here, because both orderings produce identical values.
    """
    seen: list[xr.DataArray] = []
    original = chirps3.normalize_period

    def spy(obj: xr.DataArray, **kwargs: Any) -> xr.Dataset:
        seen.append(obj)
        return original(obj, **kwargs)

    monkeypatch.setattr(chirps3, "normalize_period", spy)
    CHIRPS3MonthlyPlugin().fetch_period("2025-01", [33.0, -14.0, 34.0, -13.0])

    assert len(seen) == 1
    # Still the raw monthly total, not 310/31 — nothing has scaled it before the clip.
    assert float(np.nanmax(seen[0].values)) == pytest.approx(310.0)


def test_reordering_preserves_the_previous_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip-then-scale must equal scale-then-clip, sentinel included.

    Reproduces the superseded ordering inline and compares. Against the live COGs the two
    were bit-identical for the same month (max abs diff 0.0); this pins that here so the
    optimisation cannot quietly change stored values.
    """
    import rioxarray

    monkeypatch.setattr(rioxarray, "open_rasterio", lambda url, **_: _source_raster(with_nodata=True))
    bbox = [33.0, -14.0, 34.0, -13.0]
    actual = CHIRPS3MonthlyPlugin().fetch_period("2025-01", bbox)["precip"].values

    # The previous implementation: scale everything except the sentinel, then clip and mask.
    days = calendar.monthrange(2025, 1)[1]
    raw = _source_raster(with_nodata=True)
    scaled_first = raw.where(raw == _NODATA, raw / days)
    expected = chirps3.normalize_period(
        scaled_first, variable="precip", period="2025-01-01", nodata=_NODATA, bbox=bbox
    )["precip"].values

    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), "NaN pattern differs"
    assert np.array_equal(actual, expected, equal_nan=True), "values differ"


def test_time_coordinate_is_the_first_of_the_month(stub_raster: None) -> None:
    ds = CHIRPS3MonthlyPlugin().fetch_period("2025-03", [33.0, -14.0, 34.0, -13.0])
    assert np.datetime_as_string(ds["t"].values[0], unit="D") == "2025-03-01"


def test_accepts_a_full_date_period_id(stub_raster: None) -> None:
    ds = CHIRPS3MonthlyPlugin().fetch_period("2025-03-01", [33.0, -14.0, 34.0, -13.0])
    assert np.datetime_as_string(ds["t"].values[0], unit="D") == "2025-03-01"


@pytest.mark.parametrize("value", ["not-a-month", "2025-13", "2025-00", "2025-1", "2025"])
def test_rejects_an_unparseable_period_id(stub_raster: None, value: str) -> None:
    """An out-of-range month should name the plugin's contract, not surface as
    ``calendar.IllegalMonthError`` from ``monthrange`` further in."""
    with pytest.raises(ValueError, match="expected YYYY-MM"):
        CHIRPS3MonthlyPlugin().fetch_period(value, [33.0, -14.0, 34.0, -13.0])


# --- URLs and availability ------------------------------------------------------------


def test_url_matches_the_published_layout() -> None:
    assert CHIRPS3MonthlyPlugin._url_for_month(2025, 1) == (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs/chirps-v3.0.2025.01.cog"
    )


def test_periods_are_clamped_to_the_published_month(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CHIRPS3MonthlyPlugin, "_availability_cutoff", lambda self: "2025-02")
    periods = asyncio.run(CHIRPS3MonthlyPlugin().periods("2024-12", "2025-06"))
    assert periods == ["2024-12", "2025-01", "2025-02"]


def test_availability_probe_walks_back_to_the_newest_published_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """The current month and the one before are typically unpublished."""
    published = "chirps-v3.0.2025.03.cog"
    calls: list[str] = []

    class _Response:
        def __init__(self, url: str) -> None:
            self.status_code = 200 if published in url else 404

    def fake_head(url: str, **_: Any) -> _Response:
        calls.append(url)
        return _Response(url)

    import httpx

    monkeypatch.setattr(httpx, "head", fake_head)
    monkeypatch.setattr(chirps3, "datetime", _FrozenDatetime)
    assert CHIRPS3MonthlyPlugin()._availability_cutoff() == "2025-03"
    assert len(calls) == 3, "should stop at the first published month (May, April, March)"


def test_availability_probe_raises_when_nothing_is_published(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Missing:
        status_code = 404

    import httpx

    monkeypatch.setattr(httpx, "head", lambda url, **_: _Missing())
    monkeypatch.setattr(chirps3, "datetime", _FrozenDatetime)
    with pytest.raises(RuntimeError, match="No published CHIRPS3 monthly COG"):
        CHIRPS3MonthlyPlugin()._availability_cutoff()


class _FrozenDatetime:
    """Pin "now" so the probe's candidate months are deterministic."""

    @staticmethod
    def now(_tz: Any = None) -> Any:
        class _Now:
            @staticmethod
            def date() -> date:
                return date(2025, 5, 20)

        return _Now()


# --- the template ---------------------------------------------------------------------


def test_template_is_registered_with_the_monthly_plugin() -> None:
    template = {d["id"]: d for d in registry.list_datasets()}["chirps3_precipitation_monthly"]
    assert template["period_type"] == "monthly"
    assert template["ingestion"]["plugin"].endswith("CHIRPS3MonthlyPlugin")


def test_template_units_match_the_monthly_normal() -> None:
    """The whole point of converting: an anomaly pairs these two, so units must agree."""
    loaded = {d["id"]: d for d in registry.list_datasets()}
    assert loaded["chirps3_precipitation_monthly"]["units"] == "mm/d"
    assert loaded["chirps3_precipitation_monthly_normal_1991_2020"]["units"] == "mm/d"
    assert loaded["era5land_precipitation_monthly"]["units"] == "mm/d"
