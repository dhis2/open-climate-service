import asyncio
import time
from datetime import date, timedelta

from open_climate_service.plugins.datasets.era5_heat import (
    ERA5HeatCDSDailyFromHourlyPlugin,
    ERA5HeatCDSHourlyPlugin,
    ERA5HeatDailyUTCIPlugin,
)

_TEST_BBOX = [28.7, -2.9, 28.8, -2.8]


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
