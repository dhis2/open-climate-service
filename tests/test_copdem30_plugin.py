import asyncio
import os

import pytest
import xarray as xr

from open_climate_service.plugins.datasets.copdem30 import CopDEM30Plugin

_TEST_BBOX = [28.7, -2.9, 28.8, -2.8]


def test_periods():
    plugin = CopDEM30Plugin()

    assert asyncio.run(plugin.periods("2000", "2009")) == []
    assert asyncio.run(plugin.periods("2009", "2010")) == ["2010"]
    assert asyncio.run(plugin.periods("2009", "2011")) == ["2010"]
    assert asyncio.run(plugin.periods("2010", "2010")) == ["2010"]
    assert asyncio.run(plugin.periods("2010", "2020")) == ["2010"]
    assert asyncio.run(plugin.periods("2011", "2020")) == []

    assert asyncio.run(plugin.periods("2009-01-01", "2011-01-01")) == ["2010"]
    assert asyncio.run(plugin.periods("2010-01-01", "2010-01-01")) == ["2010"]
    assert asyncio.run(plugin.periods("2009-01-01", "2009-01-01")) == []


@pytest.fixture
def elevation_data():
    # hacky check for TEST_INTEGRATIONS flag for integration tests that should only be run manually
    if not os.getenv("TEST_INTEGRATIONS"):
        pytest.skip("Set TEST_INTEGRATIONS=1 to run remote data tests")

    plugin = CopDEM30Plugin()
    ds = plugin.fetch_period(period_id="2010", bbox=_TEST_BBOX)
    return ds


def test_copdem30_dims_and_values(elevation_data: xr.Dataset):
    assert isinstance(elevation_data, xr.Dataset)

    assert set(("t", "y", "x")).issubset(elevation_data.dims)

    assert elevation_data.sizes["x"] > 1
    assert elevation_data.sizes["y"] > 1
    assert elevation_data.sizes["t"] == 1
    assert elevation_data.t.dt.year.item() == 2010

    assert "elevation" in elevation_data.data_vars
    assert elevation_data["elevation"].size > 0
