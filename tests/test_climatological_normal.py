"""Tests for the built-in `climatological_normal` openEO process."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_geo_service.plugins.processes.climatological_normal import climatological_normal


def _daily_cube() -> xr.DataArray:
    """Two years of daily data where each step equals its day-of-year."""
    times = pd.date_range("1991-01-01", "1992-12-31", freq="D")
    doy = times.dayofyear.to_numpy().astype("float32")
    data = np.broadcast_to(doy[:, None, None], (len(times), 2, 2)).copy()
    return xr.DataArray(data, dims=("t", "y", "x"), coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]})


def test_climatological_normal_reduces_time_to_dayofyear() -> None:
    out = climatological_normal(_daily_cube(), smoothing_window=0)

    assert "dayofyear" in out.dims
    assert "t" not in out.dims
    assert out.sizes["dayofyear"] == 366  # leap day present (1992)
    assert float(out.sel(dayofyear=100).isel(y=0, x=0)) == pytest.approx(100.0)
    assert float(out.sel(dayofyear=366).isel(y=0, x=0)) == pytest.approx(366.0)


def test_climatological_normal_applies_smoothing() -> None:
    out = climatological_normal(_daily_cube(), smoothing_window=31)
    # Circular wrap makes day-of-year 1 differ markedly from its raw value (1.0).
    assert float(out.sel(dayofyear=1).isel(y=0, x=0)) != pytest.approx(1.0)


def test_climatological_normal_requires_temporal_dimension() -> None:
    flat = xr.DataArray(np.zeros((2, 2), dtype="float32"), dims=("y", "x"), coords={"y": [1.0, 2.0], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="temporal dimension"):
        climatological_normal(flat)


def test_climatological_normal_monthly_reduces_time_to_month() -> None:
    # A cube where each step equals its calendar month → the month-of-year mean is the
    # month itself (1..12).
    times = pd.date_range("1991-01-01", "1992-12-31", freq="D")
    month = times.month.to_numpy().astype("float32")
    data = np.broadcast_to(month[:, None, None], (len(times), 2, 2)).copy()
    cube = xr.DataArray(data, dims=("t", "y", "x"), coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]})

    out = climatological_normal(cube, frequency="month")

    assert "month" in out.dims and "dayofyear" not in out.dims and "t" not in out.dims
    assert out.sizes["month"] == 12
    assert float(out.sel(month=1).isel(y=0, x=0)) == pytest.approx(1.0)
    assert float(out.sel(month=12).isel(y=0, x=0)) == pytest.approx(12.0)


def test_climatological_normal_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        climatological_normal(_daily_cube(), frequency="weekly")


def test_climatological_normal_requires_datetime_temporal_dim() -> None:
    # A dim named "t" with non-datetime coords must raise clearly, not fall through to
    # groupby("t.dayofyear") on integer values.
    arr = xr.DataArray(
        np.zeros((3, 2, 2), dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": [0, 1, 2], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="datetime"):
        climatological_normal(arr)


@pytest.mark.parametrize("bad", [-1, 30, 999])
def test_climatological_normal_validates_smoothing_window(bad: int) -> None:
    # negative (< 0), even (30), and too-large (> 366 days) all raise.
    with pytest.raises(ValueError, match="smoothing_window"):
        climatological_normal(_daily_cube(), smoothing_window=bad)


def test_climatological_normal_smoothing_preserves_dask_laziness() -> None:
    cube = _daily_cube().chunk({"y": 1, "x": 1})
    out = climatological_normal(cube, smoothing_window=31)
    assert out.chunks is not None  # smoothing stays lazy — not materialised via .values
