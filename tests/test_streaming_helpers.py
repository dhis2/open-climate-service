"""Tests for the shared streaming plugin helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_geo_service.streaming import normalize_period


def test_normalize_period_renames_dims_masks_nodata_and_stamps_time() -> None:
    da = xr.DataArray(
        np.array([[[5.0, -9999.0], [12.0, 7.0]]], dtype="float32"),
        dims=("band", "lat", "lon"),
        coords={"band": [1], "lat": [1.0, 2.0], "lon": [10.0, 11.0]},
    )
    ds = normalize_period(da, variable="precip", period="2024-01-01", nodata=-9999.0)

    assert list(ds.data_vars) == ["precip"]
    assert set(ds.dims) == {"t", "y", "x"}  # band squeezed, lat/lon renamed, t added
    assert ds.sizes["t"] == 1
    assert str(ds["t"].values[0])[:10] == "2024-01-01"
    assert bool(np.isnan(ds["precip"].values).any())  # sentinel masked to NaN


def test_normalize_period_renames_source_variable_on_a_dataset() -> None:
    ds_in = xr.Dataset(
        {"pop_total": (("time", "lon", "lat"), np.array([[[3.0]]], dtype="float32"))},
        coords={"time": [np.datetime64("2020-01-01")], "lon": [1.0], "lat": [2.0]},
    )
    ds = normalize_period(ds_in, variable="population", source_variable="pop_total")

    assert list(ds.data_vars) == ["population"]
    assert "t" in ds.dims and "x" in ds.dims and "y" in ds.dims


def test_normalize_period_keeps_existing_time_dim() -> None:
    """When the source already carries a time dim, period is not double-stamped."""
    ds_in = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((2, 1, 1), dtype="float32"))},
        coords={"time": [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")], "lat": [1.0], "lon": [1.0]},
    )
    ds = normalize_period(ds_in, variable="v", period="2020-01-01")
    assert ds.sizes["t"] == 2  # not collapsed or re-expanded


def test_normalize_period_renames_projected_axes_and_drops_2d_aux_coords() -> None:
    """A projected grid (uppercase X/Y dims + curvilinear 2-D lon/lat helper coords)
    is normalized to x/y without any plugin-side rename: the 2-D aux coords are
    dropped and X/Y are renamed (so they don't collide with the lon/lat names)."""
    da = xr.DataArray(
        np.arange(6, dtype="float32").reshape(2, 3),
        dims=("Y", "X"),
        coords={
            "X": [0.0, 1000.0, 2000.0],
            "Y": [7000000.0, 6999000.0],
            "longitude": (("Y", "X"), np.zeros((2, 3))),
            "latitude": (("Y", "X"), np.zeros((2, 3))),
        },
        name="v",
    )
    ds = normalize_period(da, variable="v")

    assert set(ds.dims) == {"x", "y"}  # X/Y renamed
    assert "longitude" not in ds.coords and "latitude" not in ds.coords  # 2-D aux dropped
    assert ds["v"].shape == (2, 3)


@pytest.mark.parametrize("crs_in,expected", [(4326, 4326), ("EPSG:32633", 32633), ("epsg:4326", 4326), ("CRS84", 4326)])
def test_to_epsg_int_accepts_int_and_string_forms(crs_in: int | str, expected: int) -> None:
    from open_geo_service.streaming.protocol import to_epsg_int

    assert to_epsg_int(crs_in) == expected
