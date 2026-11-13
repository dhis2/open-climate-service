import pytest
import xarray as xr

from open_climate_service.plugins.rasters import copdem
from open_climate_service.plugins.rasters.copdem import CopDEMPlugin

_TEST_BBOX = [28.7, -2.9, 28.8, -2.8]


@pytest.mark.parametrize("product", ["GLO-30", "GLO-90"])
def test_fetch_returns_elevation_without_a_time_axis(monkeypatch: pytest.MonkeyPatch, product: str) -> None:
    """A static dataset (CLIM-1448): the source grid comes back as (y, x), clipped and renamed."""
    import numpy as np

    lon = np.arange(28.0, 29.5, 0.05)
    lat = np.arange(-2.0, -3.5, -0.05)
    source = xr.Dataset(
        {"dsm": (("lat", "lon"), np.ones((lat.size, lon.size), dtype="float32"))}, {"lat": lat, "lon": lon}
    )
    opened: list[str] = []

    def open_dataset(url: str, **_: object) -> xr.Dataset:
        opened.append(url)
        return source

    monkeypatch.setattr(copdem.xr, "open_dataset", open_dataset)
    monkeypatch.setenv("EDH_API_KEY", "mytoken")

    ds = CopDEMPlugin(product=product).fetch(bbox=_TEST_BBOX)

    assert opened == [f"https://data.earthdatahub.destine.eu/copernicus-dem/{product}-v1.zarr"]
    assert set(ds["elevation"].dims) == {"y", "x"} and "t" not in ds.coords
    assert float(ds.x.min()) >= _TEST_BBOX[0] - 0.05 and float(ds.x.max()) <= _TEST_BBOX[2] + 0.05


def test_an_unknown_product_is_refused() -> None:
    with pytest.raises(ValueError, match="GLO-60"):
        CopDEMPlugin(product="GLO-60")
