import asyncio
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.rasters.era5_heat import (
    ERA5HeatDailyUTCIPlugin,
    ERA5HeatZarrDailyFromHourlyPlugin,
    ERA5HeatZarrHourlyPlugin,
)
from open_climate_service.shared.zarr import CachedTimeSeriesZarr

_TEST_BBOX = [28.0, -3.0, 29.0, -2.0]


def test_hourly_periods(monkeypatch: pytest.MonkeyPatch):
    plugin = ERA5HeatZarrHourlyPlugin(variable="utci")

    # monkeypatch the cutoff to avoid hitting CDS servers
    fake_cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(plugin, "_cached_cutoff", fake_cutoff)

    # test correct fetching of hours in a day
    start = end = "2020-01-01"
    hours = asyncio.run(plugin.periods(start, end))
    assert len(hours) == 24
    assert hours[0][:13] == "2020-01-01T00"
    assert hours[-1][:13] == "2020-01-01T23"

    # test correct fetching of hours across 2 days
    start = "2020-01-01"
    end = "2020-01-02"
    hours = asyncio.run(plugin.periods(start, end))
    assert len(hours) == 48
    assert hours[0][:13] == "2020-01-01T00"
    assert hours[-1][:13] == "2020-01-02T23"


def test_daily_periods(monkeypatch: pytest.MonkeyPatch):
    plugin = ERA5HeatDailyUTCIPlugin(temporal_aggregation="mean")

    # monkeypatch the cutoff to avoid hitting data servers
    fake_cutoff = datetime.today()
    monkeypatch.setattr(plugin, "_cached_cutoff", fake_cutoff)

    # future invalid dates
    assert asyncio.run(plugin.periods("2050-01-01", "2050-12-31")) == []

    # inside valid date range
    start, end = "2020-01-01", "2020-12-31"
    days = asyncio.run(plugin.periods(start, end))
    assert len(days) == 366
    assert days[0] == start
    assert days[-1] == end

    # respects end cutoff
    # starting ~6 months in the past should be valid
    start = (date.today() - timedelta(days=30 * 6)).isoformat()
    # ending ~6 months in the future should be restricted by cutoff
    end = (date.today() + timedelta(days=30 * 6)).isoformat()
    # request and validate days
    days = asyncio.run(plugin.periods(start, end))
    assert days[0] == start
    assert days[-1] < end


def test_daily_fetch_reuses_cached_time_window(monkeypatch: pytest.MonkeyPatch):
    plugin = ERA5HeatZarrDailyFromHourlyPlugin(variable="utci", temporal_aggregation="mean")
    times = np.arange("2020-01-01", "2020-01-03", dtype="datetime64[h]").astype("datetime64[ns]")
    values = np.arange(48, dtype=float).reshape(48, 1, 1)
    ds = xr.Dataset({"utci": (("t", "y", "x"), values)}, coords={"t": times})
    plugin._zarr = CachedTimeSeriesZarr("unused", "utci", time_dim="t", opener=lambda _: ds)
    monkeypatch.setattr(plugin, "_prepare_zarr_dataset", lambda ds, bbox: ds)

    first_day = plugin.fetch_period("2020-01-01", _TEST_BBOX)
    cached_chunk = plugin._zarr._cache
    second_day = plugin.fetch_period("2020-01-02", _TEST_BBOX)

    assert plugin._zarr._cache is cached_chunk
    assert first_day.utci.item() == pytest.approx(11.5)
    assert second_day.utci.item() == pytest.approx(35.5)


def test_daily_fetch_reuses_bounded_cache_while_progressing_forward(monkeypatch: pytest.MonkeyPatch):
    plugin = ERA5HeatZarrDailyFromHourlyPlugin(variable="utci", temporal_aggregation="mean")
    times = np.arange("2020-01-01", "2020-01-09", dtype="datetime64[h]").astype("datetime64[ns]")
    values = np.arange(8 * 24, dtype=float).reshape(8 * 24, 1, 1)
    ds = xr.Dataset({"utci": (("t", "y", "x"), values)}, coords={"t": times})
    plugin._zarr = CachedTimeSeriesZarr("unused", "utci", time_dim="t", opener=lambda _: ds, max_cache_bytes=72 * 8)
    monkeypatch.setattr(plugin, "_prepare_zarr_dataset", lambda ds, bbox: ds)

    previous_cache = None
    for day in range(8):
        period_id = (date(2020, 1, 1) + timedelta(days=day)).isoformat()
        daily_ds = plugin.fetch_period(period_id, _TEST_BBOX)

        assert daily_ds.utci.item() == pytest.approx(day * 24 + 11.5)
        assert plugin._zarr._cache is not None
        assert plugin._zarr._cache.sizes["t"] == 72

        if day in (0, 3, 6):
            if previous_cache is not None:
                assert plugin._zarr._cache is not previous_cache
            previous_cache = plugin._zarr._cache
            expected_cache_start = {0: 0, 3: 72, 6: 120}[day]
            assert plugin._zarr._cache_start == expected_cache_start
        else:
            assert plugin._zarr._cache is previous_cache


def test_cached_time_series_zarr_uses_explicit_time_dimension():
    times = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    values = np.arange(8, dtype=float).reshape(2, 4, 1)
    ds = xr.Dataset(
        {"value": (("y", "valid_at", "x"), values)},
        coords={"valid_at": times},
    )
    zarr = CachedTimeSeriesZarr("unused", "value", time_dim="valid_at", opener=lambda _: ds, max_cache_bytes=32)

    result = zarr.slice_time(np.datetime64("2020-01-02"), np.datetime64("2020-01-04"))

    assert result.sizes["valid_at"] == 2
    assert result.value.values[:, :, 0].tolist() == [[1, 2], [5, 6]]


def test_cached_time_series_zarr_rejects_ranges_over_cache_limit():
    times = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
    ds = xr.Dataset({"value": (("time", "x"), np.ones((3, 2), dtype="float32"))}, coords={"time": times})
    zarr = CachedTimeSeriesZarr(
        "unused",
        "value",
        time_dim="time",
        opener=lambda _: ds,
        max_cache_bytes=8,
    )

    with pytest.raises(ValueError, match="use xarray directly for large ranges"):
        zarr.slice_time(np.datetime64("2020-01-01"), np.datetime64("2020-01-03"))


# Functional integrations tests


@pytest.fixture
def daily_utci_data_func():
    """Returns adjustable function to easily fetch data for a single day of UTCI heat index data"""
    # hacky check for TEST_INTEGRATIONS flag for integration tests that should only be run manually
    if os.getenv("TEST_INTEGRATIONS") != "1":
        pytest.skip("Set TEST_INTEGRATIONS=1 to run remote data tests")

    def func(period_id: str, temporal_aggregation: str):
        plugin = ERA5HeatDailyUTCIPlugin(temporal_aggregation=temporal_aggregation)
        ds = plugin.fetch_period(period_id=period_id, bbox=_TEST_BBOX)
        return ds

    return func


def test_daily_from_hourly_periods_efficiency():
    """Check that daily-from-hourly period generation stays fast across a multi-year daily range.

    This generates the list of daily periods, checks that the hourly plugin reuses rather than
    re-fetching the cutoff cache, and bounds the full loop's runtime.
    """
    if os.getenv("TEST_INTEGRATIONS") != "1":
        pytest.skip("Set TEST_INTEGRATIONS=1 to run remote data tests")

    hourly_plugin = ERA5HeatZarrHourlyPlugin(variable="utci")
    daily_plugin = ERA5HeatZarrDailyFromHourlyPlugin(variable="utci", temporal_aggregation="mean")

    # Generate daily IDs, then ensure each day's hourly IDs are produced quickly.
    start = "2020-01-01"
    end = "2025-12-31"
    days = asyncio.run(daily_plugin.periods(start, end))
    assert hourly_plugin._cached_cutoff is None
    t = time.monotonic()
    for day in days:
        hours = asyncio.run(hourly_plugin.periods(start=day, end=day))
        assert len(hours) == 24
        assert hours[0][:10] == day
        assert hourly_plugin._cached_cutoff is not None
    duration = time.monotonic() - t
    max_duration = 10  # seconds for the full multi-year loop
    assert duration < max_duration


def test_utci_dims_and_values(daily_utci_data_func: Any):
    # run the data function
    period_id = "2023-01-01"
    daily_utci_data = daily_utci_data_func(period_id=period_id, temporal_aggregation="mean")

    assert isinstance(daily_utci_data, xr.Dataset)

    assert set(("t", "y", "x")).issubset(daily_utci_data.dims)

    assert daily_utci_data.sizes["x"] > 1
    assert daily_utci_data.sizes["y"] > 1
    assert daily_utci_data.sizes["t"] == 1
    assert str(daily_utci_data.t.values[0])[:10] == period_id

    assert "utci" in daily_utci_data.data_vars
    assert daily_utci_data["utci"].size > 0


def test_utci_temporal_aggregation(daily_utci_data_func: Any):
    # run the data function
    period_id = "2023-01-01"
    min_data = daily_utci_data_func(
        period_id=period_id,
        temporal_aggregation="min",
    )
    mean_data = daily_utci_data_func(
        period_id=period_id,
        temporal_aggregation="mean",
    )
    max_data = daily_utci_data_func(
        period_id=period_id,
        temporal_aggregation="max",
    )

    # ensure stats values are indeed lower or higher than each other
    assert (min_data.utci < mean_data.utci).all()
    assert (mean_data.utci < max_data.utci).all()
