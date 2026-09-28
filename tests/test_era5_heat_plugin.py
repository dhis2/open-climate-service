import asyncio
import os
import time
from datetime import date, timedelta
from typing import Any

import pytest
import xarray as xr

from open_climate_service.plugins.datasets.era5_heat import (
    ERA5HeatCDSDailyFromHourlyPlugin,
    ERA5HeatCDSHourlyPlugin,
    ERA5HeatDailyUTCIPlugin,
)

_TEST_BBOX = [28, -3, 29, -2]


def test_hourly_periods():
    plugin = ERA5HeatCDSHourlyPlugin(variable="utci")

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


def test_daily_from_hourly_periods_efficiency():
    """Background: Daily aggregation is done by reusing the hourly .periods() function to provide
    the hours of each day, and then fetching and merging each hour. This will result in many periods() calls,
    so we check that it finishes in reasonable time and uses the internal cutoff cache.
    """
    hourly_plugin = ERA5HeatCDSHourlyPlugin(variable="utci")
    daily_plugin = ERA5HeatCDSDailyFromHourlyPlugin(variable="utci", temporal_aggregation="mean")

    # test that calling hourly periods for many days (5 years worth) finishes in reasonable time
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
    max_duration = 10  # seconds
    assert duration < max_duration


def test_daily_periods():
    plugin = ERA5HeatDailyUTCIPlugin(temporal_aggregation="mean")

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


@pytest.fixture
def daily_utci_data_func():
    """Returns adjustable function to easily fetch data for a single day of UTCI heat index data"""
    # hacky check for TEST_INTEGRATIONS flag for integration tests that should only be run manually
    if not os.getenv("TEST_INTEGRATIONS"):
        pytest.skip("Set TEST_INTEGRATIONS=1 to run remote data tests")

    def func(period_id: str, temporal_aggregation: str):
        plugin = ERA5HeatDailyUTCIPlugin(temporal_aggregation=temporal_aggregation)
        ds = plugin.fetch_period(period_id=period_id, bbox=_TEST_BBOX)
        return ds

    return func


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
