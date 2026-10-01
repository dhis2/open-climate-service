import asyncio
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers the .rio accessor for the fakes
import xarray as xr

from open_climate_service.plugins.rasters.worldpop import (
    WorldPopYearlyPlugin,
    _population_url,
    _resolve_variant,
    _worldpop_cache_dir,
)


def _fake_geotiff(values: list) -> xr.DataArray:
    """A WorldPop-like ``(band, y, x)`` DataArray with a WGS84 CRS so it can be clipped."""
    arr = np.array(values, dtype="float32")  # shape (1, ny, nx)
    ny, nx = arr.shape[1], arr.shape[2]
    da = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={"band": [1], "y": list(range(ny, 0, -1)), "x": list(range(nx))},
    )
    return da.rio.write_crs("EPSG:4326")


def test_worldpop_plugin_periods_enumerates_years() -> None:
    plugin = WorldPopYearlyPlugin()

    periods = asyncio.run(plugin.periods("2020", "2023"))

    assert periods == ["2020", "2021", "2022", "2023"]


def test_worldpop_periods_clamped_to_global2_window() -> None:
    from open_climate_service.plugins.rasters.worldpop import WorldPopAgeSexYearlyPlugin

    plugin = WorldPopYearlyPlugin()
    # Global2 only publishes 2015–2030, so out-of-range years are dropped (no 404s).
    assert asyncio.run(plugin.periods("2010", "2017")) == ["2015", "2016", "2017"]
    assert asyncio.run(plugin.periods("2029", "2035")) == ["2029", "2030"]
    assert asyncio.run(plugin.periods("2031", "2040")) == []
    # age/sex shares the same window
    assert asyncio.run(WorldPopAgeSexYearlyPlugin().periods("2010", "2016")) == ["2015", "2016"]


def test_population_url_100m_and_1km_and_constrained_flavours() -> None:
    # 100 m constrained (the default)
    u100 = _population_url(2022, "sle", "R2025A", "100m", True)
    assert "Global_2015_2030/R2025A/2022/SLE/v1/100m/constrained/" in u100
    assert u100.endswith("sle_pop_2022_CN_100m_R2025A_v1.tif")

    # 1 km uses the 1km_ua dir and the _UA_ filename suffix
    u1km = _population_url(2022, "SLE", "R2025A", "1km", True)
    assert "/v1/1km_ua/constrained/" in u1km
    assert u1km.endswith("sle_pop_2022_CN_1km_R2025A_UA_v1.tif")

    # unconstrained swaps the subdir + CN->UC token
    uunc = _population_url(2022, "sle", "R2025A", "100m", False)
    assert "/100m/unconstrained/" in uunc and "_UC_100m_" in uunc

    with pytest.raises(ValueError, match="Unsupported WorldPop resolution"):
        _population_url(2022, "SLE", "R2025A", "250m", True)


def test_worldpop_plugin_fetch_period_reads_country_url(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = WorldPopYearlyPlugin(version="global2")
    captured: dict[str, object] = {}

    def fake_open(url: str, **_: object) -> xr.DataArray:
        captured["url"] = url
        return _fake_geotiff([[[1.0, 2.0], [3.0, 4.0]]])

    monkeypatch.setattr("open_climate_service.plugins.rasters.worldpop._open_worldpop_raster", fake_open)

    dataset = plugin.fetch_period("2022", [-1.0, -1.0, 5.0, 5.0], country_code="SLE")

    assert "/2022/SLE/" in str(captured["url"])
    assert list(dataset.data_vars) == ["pop_total"]


def test_worldpop_cache_dir_uses_data_dir_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nextent:\n  id: test\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    assert _worldpop_cache_dir() == tmp_path / "data" / "cache" / "worldpop"


def test_worldpop_cache_dir_falls_back_to_xdg_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert _worldpop_cache_dir() == tmp_path / "climate-service" / "cache" / "worldpop"


def test_worldpop_plugin_requires_country_code() -> None:
    plugin = WorldPopYearlyPlugin()

    with pytest.raises(ValueError, match="country_code"):
        plugin.fetch_period("2022", [1.0, 2.0, 3.0, 4.0])


def test_worldpop_plugin_variant_resolver_supports_total_product() -> None:
    variant = _resolve_variant(product="total", variable="pop_total")

    assert variant.product == "total"
    assert variant.output_variable == "pop_total"


def test_worldpop_plugin_variant_resolver_rejects_unknown_product() -> None:
    with pytest.raises(ValueError, match="Unsupported WorldPop product"):
        _resolve_variant(product="female", variable="pop_female")


def test_worldpop_plugin_renames_output_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = WorldPopYearlyPlugin(product="total", variable="population_total")
    monkeypatch.setattr(
        "open_climate_service.plugins.rasters.worldpop._open_worldpop_raster",
        lambda url, **_: _fake_geotiff([[[1.0, 2.0], [3.0, 4.0]]]),
    )

    dataset = plugin.fetch_period("2022", [-1.0, -1.0, 5.0, 5.0], country_code="SLE")

    assert list(dataset.data_vars) == ["population_total"]
    assert "x" in dataset.dims and "y" in dataset.dims


def test_worldpop_plugin_masks_nodata_sentinel_to_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values equal to the WorldPop -99999 sentinel must become NaN, not be stored."""
    plugin = WorldPopYearlyPlugin()
    monkeypatch.setattr(
        "open_climate_service.plugins.rasters.worldpop._open_worldpop_raster",
        lambda url, **_: _fake_geotiff([[[5.0, -99999.0], [12.0, 7.0]]]),
    )

    dataset = plugin.fetch_period("2020", [-1.0, -1.0, 5.0, 5.0], country_code="SLE")

    values = dataset["pop_total"].values.flatten()
    assert np.isnan(values).sum() == 1, "the single -99999 sentinel must be masked to NaN"
    finite = sorted(int(round(v)) for v in values if np.isfinite(v))
    assert finite == [5, 7, 12], "valid values must be preserved"


def test_worldpop_plugin_accepts_extra_kwargs() -> None:
    plugin = WorldPopYearlyPlugin(
        version="global2", product="total", variable="pop_total", unknown_future_field="ignored"
    )
    assert plugin.version == "global2"


# ---------------------------------------------------------------------------
# Age/sex structures (hub id=8)
# ---------------------------------------------------------------------------


def test_agesex_url_encodes_country_year_sex_age() -> None:
    from open_climate_service.plugins.rasters.worldpop import _agesex_url

    u = _agesex_url(2020, "sle", "R2025A", "f", "05", True)
    assert "/AgeSex_structures/Global_2015_2030/R2025A/2020/SLE/v1/100m/constrained/" in u
    assert u.endswith("sle_f_05_2020_CN_100m_R2025A_v1.tif")


def test_agesex_plugin_rejects_non_100m() -> None:
    from open_climate_service.plugins.rasters.worldpop import WorldPopAgeSexYearlyPlugin

    with pytest.raises(ValueError, match="only resolution '100m'"):
        WorldPopAgeSexYearlyPlugin(resolution="1km")


def test_agesex_fetch_builds_population_sex_age_cube(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.rasters.worldpop import _AGE_BANDS, WorldPopAgeSexYearlyPlugin

    plugin = WorldPopAgeSexYearlyPlugin()
    monkeypatch.setattr(
        "open_climate_service.plugins.rasters.worldpop._open_worldpop_raster",
        lambda url, **_: _fake_geotiff([[[1.0, 2.0], [3.0, 4.0]]]),
    )

    ds = plugin.fetch_period("2020", [-1.0, -1.0, 5.0, 5.0], country_code="SLE")

    # Single quantity (population); sex and age are both disaggregation dimensions.
    assert set(ds.data_vars) == {"population"}
    assert ds.sizes["sex"] == 2
    assert ds.sizes["age_group"] == len(_AGE_BANDS)
    assert {"t", "sex", "age_group", "y", "x"} <= set(ds.dims)
    assert list(ds["sex"].values) == ["female", "male"]
    assert list(ds["age_group"].values[:3]) == [0, 1, 5]
