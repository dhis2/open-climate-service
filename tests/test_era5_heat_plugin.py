import asyncio
from datetime import date, timedelta

import pytest
import xarray as xr

from open_climate_service.plugins.datasets.era5_heat import ERA5HeatDailyUTCIPlugin

_TEST_BBOX = [28.7, -2.9, 28.8, -2.8]


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
