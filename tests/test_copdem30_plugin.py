import os

import pytest
import xarray as xr

from open_climate_service.plugins.rasters import copdem30
from open_climate_service.plugins.rasters.copdem30 import CopDEM30Plugin

_TEST_BBOX = [28.7, -2.9, 28.8, -2.8]


def test_fetch_returns_elevation_without_a_time_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A static dataset (CLIM-1448): the source grid comes back as (y, x), clipped and renamed."""
    import numpy as np

    lon = np.arange(28.0, 29.5, 0.05)
    lat = np.arange(-2.0, -3.5, -0.05)
    source = xr.Dataset(
        {"dsm": (("lat", "lon"), np.ones((lat.size, lon.size), dtype="float32"))}, {"lat": lat, "lon": lon}
    )
    monkeypatch.setattr(copdem30.xr, "open_dataset", lambda *args, **kwargs: source)

    ds = CopDEM30Plugin().fetch(bbox=_TEST_BBOX)

    assert set(ds["elevation"].dims) == {"y", "x"} and "t" not in ds.coords
    assert float(ds.x.min()) >= _TEST_BBOX[0] - 0.05 and float(ds.x.max()) <= _TEST_BBOX[2] + 0.05


@pytest.fixture
def elevation_data():
    # hacky check for TEST_INTEGRATIONS flag for integration tests that should only be run manually
    if not os.getenv("TEST_INTEGRATIONS"):
        pytest.skip("Set TEST_INTEGRATIONS=1 to run remote data tests")

    plugin = CopDEM30Plugin()
    ds = plugin.fetch(bbox=_TEST_BBOX)
    return ds


def test_copdem30_dims_and_values(elevation_data: xr.Dataset):
    assert isinstance(elevation_data, xr.Dataset)

    assert set(elevation_data.dims) == {"y", "x"}

    assert elevation_data.sizes["x"] > 1
    assert elevation_data.sizes["y"] > 1

    assert "elevation" in elevation_data.data_vars
    assert elevation_data["elevation"].size > 0
