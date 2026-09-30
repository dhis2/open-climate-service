"""Tests for the ERA5-Land climate normals plugin (ERA5LandNormalsPlugin)."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_climate_service.plugins.rasters import era5_land as cn
from open_climate_service.plugins.rasters.era5_land import ERA5LandNormalsPlugin


def _synthetic_region(*, var: str = "t2m", time_dim: str = "valid_time", years: tuple[str, str] = ("1991", "1992")):
    """Daily data where each step's value equals its day-of-year, so the day-of-year
    climatology is exactly the day-of-year (handy for assertions)."""
    times = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    doy = times.dayofyear.to_numpy().astype("float32")
    data = np.broadcast_to(doy[:, None, None], (len(times), 2, 2)).copy()
    return xr.Dataset(
        {var: ((time_dim, "y", "x"), data)},
        coords={time_dim: times, "y": [1.0, 2.0], "x": [10.0, 11.0]},
    )


def _edh_plugin(**overrides: object) -> ERA5LandNormalsPlugin:
    kwargs: dict[str, object] = dict(edh_variable="t2m", variable="t2m", smoothing_window=0, unit_transform=None)
    kwargs.update(overrides)
    return ERA5LandNormalsPlugin(**kwargs)  # type: ignore[arg-type]


def test_periods_are_dayofyear_ids() -> None:
    periods = asyncio.run(_edh_plugin().periods("ignored", "ignored"))
    assert len(periods) == 366
    assert periods[0] == "1"
    assert periods[-1] == "366"


def test_periods_omit_day_366_for_a_non_leap_reference_period() -> None:
    # A reference period that spans no leap year has no day-of-year 366; periods()
    # must not enumerate it (fetch_period would KeyError selecting it off the cube).
    periods = asyncio.run(_edh_plugin(period=[2021, 2023]).periods("ignored", "ignored"))
    assert len(periods) == 365
    assert periods[-1] == "365"


def test_periods_include_day_366_when_reference_period_has_a_leap_year() -> None:
    periods = asyncio.run(_edh_plugin(period=[2019, 2021]).periods("ignored", "ignored"))  # 2020 is leap
    assert len(periods) == 366
    assert periods[-1] == "366"


def test_compute_climatology_groups_by_dayofyear(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _edh_plugin()
    monkeypatch.setattr(plugin, "_load_reference", lambda bbox: _synthetic_region())

    clim = plugin._compute_climatology([10.0, 1.0, 11.0, 2.0])

    assert "dayofyear" in clim.dims
    assert clim.sizes["dayofyear"] == 366  # leap day present (1992)
    assert float(clim["t2m"].sel(dayofyear=100).isel(y=0, x=0)) == pytest.approx(100.0)
    assert float(clim["t2m"].sel(dayofyear=366).isel(y=0, x=0)) == pytest.approx(366.0)


def test_fetch_period_returns_single_dayofyear(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _edh_plugin()
    monkeypatch.setattr(plugin, "_load_reference", lambda bbox: _synthetic_region())

    ds = plugin._fetch_sync("100", [10.0, 1.0, 11.0, 2.0])

    assert ds.sizes["dayofyear"] == 1  # one step, appended along the dayofyear axis
    assert int(ds["dayofyear"].values[0]) == 100
    assert float(ds["t2m"].isel(dayofyear=0, y=0, x=0)) == pytest.approx(100.0)


def test_unit_transform_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _edh_plugin(unit_transform="kelvin_to_celsius")
    monkeypatch.setattr(plugin, "_load_reference", lambda bbox: _synthetic_region())

    clim = plugin._compute_climatology([10.0, 1.0, 11.0, 2.0])

    assert float(clim["t2m"].sel(dayofyear=300).isel(y=0, x=0)) == pytest.approx(300.0 - 273.15)


def test_smoothing_window_changes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _edh_plugin(smoothing_window=31)
    monkeypatch.setattr(plugin, "_load_reference", lambda bbox: _synthetic_region())

    clim = plugin._compute_climatology([10.0, 1.0, 11.0, 2.0])

    # A 31-day circular mean around a linear ramp shifts mid-year values only slightly,
    # but the wrap-around makes day-of-year 1 differ markedly from its raw value (1.0).
    assert float(clim["t2m"].sel(dayofyear=1).isel(y=0, x=0)) != pytest.approx(1.0)


def test_load_reference_edh_normalises_longitude_and_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    times = pd.date_range("1991-01-01", "1991-12-31", freq="D")
    edh = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.ones((len(times), 2, 2), dtype="float32"))},
        coords={"valid_time": times, "latitude": [2.0, 1.0], "longitude": [350.0, 351.0]},  # 0–360
    )
    monkeypatch.setattr(cn, "_edh_open_zarr", lambda *args, **kwargs: edh)
    plugin = _edh_plugin(period=[1991, 1991])

    region = plugin._load_reference([-10.0, 1.0, -9.0, 2.0])  # bbox in -180/180; lon 350 -> -10

    assert "x" in region.dims and "y" in region.dims
    assert float(region["x"].min()) < 0  # 0–360 normalised to -180/180


def test_param_validation() -> None:
    with pytest.raises(ValueError, match="edh_variable"):
        ERA5LandNormalsPlugin(variable="t2m", edh_variable="")


def test_param_validation_rejects_empty_variable() -> None:
    with pytest.raises(ValueError, match="variable"):
        ERA5LandNormalsPlugin(variable="", edh_variable="t2m")


def test_param_validation_rejects_malformed_period() -> None:
    with pytest.raises(ValueError, match="period"):
        ERA5LandNormalsPlugin(variable="t2m", edh_variable="t2m", period=[1991])  # type: ignore[arg-type]


def test_plugin_declares_dayofyear_stepping_dimension() -> None:
    # After the BaseDatasetPlugin migration the orchestrator reads time_dim from the
    # class attribute (probe() is gone), so it must point at the dayofyear axis.
    from open_climate_service.streaming import BaseDatasetPlugin

    plugin = _edh_plugin()
    assert isinstance(plugin, BaseDatasetPlugin)
    assert plugin.time_dim == "dayofyear"


@pytest.mark.parametrize("bad", [-1, 30])
def test_smoothing_window_must_be_nonnegative_and_odd(bad: int) -> None:
    with pytest.raises(ValueError, match="smoothing_window"):
        _edh_plugin(smoothing_window=bad)


def test_load_reference_handles_bbox_crossing_zero_meridian(monkeypatch: pytest.MonkeyPatch) -> None:
    times = pd.date_range("1991-01-01", "1991-12-31", freq="D")
    # Monotonic-ascending [0, 360) store (as EDH publishes it); the seam pieces the
    # bbox needs are the low (0,1,2) and high (358,359) ends.
    edh = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.ones((len(times), 2, 5), dtype="float32"))},
        coords={"valid_time": times, "latitude": [2.0, 1.0], "longitude": [0.0, 1.0, 2.0, 358.0, 359.0]},
    )
    monkeypatch.setattr(cn, "_edh_open_zarr", lambda *args, **kwargs: edh)
    plugin = _edh_plugin(period=[1991, 1991])

    region = plugin._load_reference([-2.0, 1.0, 2.0, 2.0])  # bbox crosses the prime meridian

    xs = [float(v) for v in region["x"].values]
    assert xs == sorted(xs)  # ascending after the 0–360 → -180/180 conversion
    assert min(xs) < 0 < max(xs)  # genuinely spans the meridian, not an empty/reversed strip
    assert len(xs) == 5  # all five source longitudes selected (358→-2 … 2→2)
