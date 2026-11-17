"""Tests for the built-in `compute_anomaly` openEO process."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_geo_service.plugins.processes.compute_anomaly import compute_anomaly


def _observed_doy(scale: float = 1.0, offset: float = 0.0) -> xr.DataArray:
    """Daily 2024 cube where each day's value is ``scale*dayofyear + offset``."""
    times = pd.date_range("2024-01-01", "2024-12-31", freq="D")  # leap year → 366 days
    doy = times.dayofyear.to_numpy().astype("float32")
    data = np.broadcast_to((scale * doy + offset)[:, None, None], (len(times), 2, 2)).copy()
    return xr.DataArray(data, dims=("t", "y", "x"), coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]})


def _normal_doy() -> xr.DataArray:
    """Day-of-year normal where each value equals its day-of-year (1..366)."""
    doy = np.arange(1, 367, dtype="float32")
    data = np.broadcast_to(doy[:, None, None], (366, 2, 2)).copy()
    return xr.DataArray(
        data,
        dims=("dayofyear", "y", "x"),
        coords={"dayofyear": list(range(1, 367)), "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )


def test_absolute_anomaly_aligns_dayofyear_and_subtracts() -> None:
    out = compute_anomaly(_observed_doy(offset=5.0), _normal_doy(), method="absolute")

    assert "dayofyear" not in out.dims and set(out.dims) >= {"t", "y", "x"}
    assert out.sizes["t"] == 366
    # observed(d) = dayofyear+5, normal(d) = dayofyear  →  anomaly = 5 everywhere
    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0)
    assert float(out.max()) == pytest.approx(5.0) and float(out.min()) == pytest.approx(5.0)


def test_relative_anomaly_is_percent_of_normal() -> None:
    # observed = 2 × normal  →  relative = 100·(2n − n)/n = 100 %
    out = compute_anomaly(_observed_doy(scale=2.0), _normal_doy(), method="relative")
    assert float(out.isel(t=180, y=0, x=0)) == pytest.approx(100.0)


def test_month_aligned_anomaly() -> None:
    # Monthly observed + monthly normal: each month's value is month+1 and the normal is
    # month, so the anomaly is 1 everywhere. earthkit aligns on the normal's `month` axis
    # and keeps the (monthly) observed time axis.
    times = pd.date_range("2024-01-01", "2024-12-01", freq="MS")  # 12 monthly steps
    month = times.month.to_numpy().astype("float32")
    obs = xr.DataArray(
        np.broadcast_to((month + 1.0)[:, None, None], (len(times), 2, 2)).copy(),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    normal = xr.DataArray(
        np.broadcast_to(np.arange(1, 13, dtype="float32")[:, None, None], (12, 2, 2)).copy(),
        dims=("month", "y", "x"),
        coords={"month": list(range(1, 13)), "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )

    out = compute_anomaly(obs, normal, method="absolute")

    assert "month" not in out.dims and out.sizes["t"] == len(times)
    assert float(out.max()) == pytest.approx(1.0) and float(out.min()) == pytest.approx(1.0)


def test_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError, match="method"):
        compute_anomaly(_observed_doy(), _normal_doy(), method="bogus")


def test_standardised_method_points_to_std_normal() -> None:
    with pytest.raises(ValueError, match="standard-deviation|standardised"):
        compute_anomaly(_observed_doy(), _normal_doy(), method="standardised")


def test_requires_datetime_observed_axis() -> None:
    obs = xr.DataArray(
        np.zeros((3, 2, 2), dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": [0, 1, 2], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="datetime"):
        compute_anomaly(obs, _normal_doy())


def test_requires_ordinal_normal_axis() -> None:
    bad_normal = xr.DataArray(
        np.zeros((2, 2), dtype="float32"), dims=("y", "x"), coords={"y": [1.0, 2.0], "x": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="dayofyear|month"):
        compute_anomaly(_observed_doy(), bad_normal)


def test_aligns_grids_differing_by_float_noise() -> None:
    # Observed and normal can be produced independently (e.g. CDS-derived observed vs an
    # EDH normal whose lon was remapped from [0, 360)), leaving spatial coords off by
    # ~1e-11. Exact-equality alignment would intersect to an empty grid; the process must
    # snap the normal onto the observed grid so the subtraction survives.
    normal = _normal_doy()
    noisy = normal.assign_coords(x=normal.x.values - 1e-11, y=normal.y.values + 1e-11)

    out = compute_anomaly(_observed_doy(offset=5.0), noisy, method="absolute")

    assert out.sizes["x"] == 2 and out.sizes["y"] == 2  # not collapsed to an empty grid
    assert bool(np.isfinite(out).all())
    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0)


def test_grid_offset_larger_than_float_noise_is_rejected() -> None:
    """A real offset must raise, not snap to whichever cell happens to be nearest.

    The earlier implementation reindexed by nearest neighbour within *half a grid step*, so a
    normal offset by a whole cell came back shifted one cell with a single NaN — the anomaly
    computed against the neighbouring cell for almost every cell, and nothing said so. A
    cell-centre vs cell-edge convention difference between two products is exactly that case.
    """
    normal = _normal_doy()  # x/y are [1.0, 2.0], so the grid step is 1.0
    for offset in (0.4, 0.5, 1.0):
        shifted = normal.assign_coords(x=normal.x.values + offset)
        with pytest.raises(ValueError, match="describe different cells"):
            compute_anomaly(_observed_doy(offset=5.0), shifted, method="absolute")


def test_grid_size_mismatch_is_rejected() -> None:
    normal = _normal_doy().isel(x=slice(0, 1))
    with pytest.raises(ValueError, match="disagree on the size of 'x'"):
        compute_anomaly(_observed_doy(offset=5.0), normal, method="absolute")


def test_grid_spacing_mismatch_is_rejected() -> None:
    """Same cell count, different resolution — a regrid, which must be explicit."""
    normal = _normal_doy().assign_coords(x=[1.0, 3.0])
    with pytest.raises(ValueError, match="different 'x' spacing"):
        compute_anomaly(_observed_doy(offset=5.0), normal, method="absolute")


def test_float_noise_relabels_rather_than_resamples() -> None:
    """The noise fix must copy coordinates across, leaving the values untouched."""
    normal = _normal_doy()
    noisy = normal.assign_coords(x=normal.x.values - 1e-11)
    observed = _observed_doy(offset=5.0)

    out = compute_anomaly(observed, noisy, method="absolute")

    np.testing.assert_array_equal(out.x.values, observed.x.values)
    assert bool(np.isfinite(out).all())


def test_anomaly_preserves_dask_laziness() -> None:
    out = compute_anomaly(_observed_doy(offset=5.0).chunk({"y": 1, "x": 1}), _normal_doy().chunk({"y": 1, "x": 1}))
    assert out.chunks is not None


def _month_normal() -> xr.DataArray:
    """Month-of-year normal where each value equals its month (1..12)."""
    data = np.broadcast_to(np.arange(1, 13, dtype="float32")[:, None, None], (12, 2, 2)).copy()
    return xr.DataArray(
        data, dims=("month", "y", "x"), coords={"month": list(range(1, 13)), "y": [1.0, 2.0], "x": [1.0, 2.0]}
    )


def test_relative_rejected_for_temperature() -> None:
    # 'relative' divides by the normal — meaningless for an interval scale (temperature),
    # which it detects from the variable's units/standard_name.
    obs = _observed_doy(scale=2.0)
    obs.attrs["units"] = "degC"
    with pytest.raises(ValueError, match="temperature"):
        compute_anomaly(obs, _normal_doy(), method="relative")

    obs2 = _observed_doy(scale=2.0)
    obs2.attrs["standard_name"] = "air_temperature"
    with pytest.raises(ValueError, match="temperature"):
        compute_anomaly(obs2, _normal_doy(), method="relative")


def test_rejects_daily_observed_with_monthly_normal() -> None:
    # A month normal paired with a daily observed would be silently resampled by earthkit.
    with pytest.raises(ValueError, match="monthly observed"):
        compute_anomaly(_observed_doy(), _month_normal())


def test_rejects_monthly_observed_with_dayofyear_normal() -> None:
    times = pd.date_range("2024-01-01", "2024-12-01", freq="MS")  # 12 monthly steps
    obs = xr.DataArray(
        np.zeros((len(times), 2, 2), "float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="daily observed"):
        compute_anomaly(obs, _normal_doy())


# --- non-spatial axes ---------------------------------------------------------------------


def test_a_bands_axis_does_not_reach_the_grid_check() -> None:
    """Only x/y are grid axes; `bands` labels are strings and must be left to xarray.

    Treating every shared non-temporal dimension as spatial made the spacing check call
    `astype(float)` on band labels, so a standard multi-band openEO cube failed with
    `could not convert string to float` before earthkit ever ran.
    """
    observed = _observed_doy().expand_dims(bands=["tp"]).transpose("t", "bands", "y", "x")
    normal = _normal_doy().expand_dims(bands=["tp"]).transpose("dayofyear", "bands", "y", "x")

    out = compute_anomaly(observed, normal, method="absolute")

    assert "bands" in out.dims
    assert float(out.isel(t=99, bands=0, y=0, x=0)) == pytest.approx(0.0)


# --- units -------------------------------------------------------------------------------


def _with_units(da: xr.DataArray, units: str) -> xr.DataArray:
    out = da.copy()
    out.attrs = {"units": units, "standard_name": "precipitation_amount"}
    return out


def test_a_normal_in_a_convertible_unit_is_converted_not_subtracted_raw() -> None:
    """earthkit subtracts with plain xarray arithmetic, which ignores units entirely.

    An observed cube in mm/d against a normal in m/d would otherwise come back as
    observed − 0.001·observed, a plausible number that is wrong by three orders of magnitude.
    """
    observed = _with_units(_observed_doy(offset=5.0), "mm/d")
    normal_m = _with_units(_normal_doy() / 1000.0, "m/d")

    out = compute_anomaly(observed, normal_m, method="absolute")

    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0, abs=1e-3)


def test_an_alias_spelling_needs_no_conversion() -> None:
    observed = _with_units(_observed_doy(offset=5.0), "mm/d")
    normal = _with_units(_normal_doy(), "mm/day")

    assert float(compute_anomaly(observed, normal, method="absolute").isel(t=99, y=0, x=0)) == pytest.approx(5.0)


def test_a_normal_measuring_a_different_quantity_is_refused() -> None:
    """A percentage normal against a rate observed is not an anomaly, at any scale."""
    observed = _with_units(_observed_doy(), "mm/d")
    normal = _with_units(_normal_doy(), "%")

    with pytest.raises(ValueError, match="measures a different quantity"):
        compute_anomaly(observed, normal, method="absolute")


def test_an_unlabelled_cube_is_left_alone() -> None:
    """The guard adds a refusal, not a requirement that every cube declare units."""
    observed = _with_units(_observed_doy(offset=5.0), "mm/d")
    normal = _normal_doy()  # no units attribute at all

    assert float(compute_anomaly(observed, normal, method="absolute").isel(t=99, y=0, x=0)) == pytest.approx(5.0)


# --- grid identity: names and CRS --------------------------------------------------------


def _normal_named(y_name: str, x_name: str) -> xr.DataArray:
    """A day-of-year normal on the same grid as `_normal_doy`, under different axis names."""
    return _normal_doy().rename({"y": y_name, "x": x_name})


def test_a_normal_naming_its_axes_lat_lon_is_renamed_not_broadcast() -> None:
    """Sharing no dimension name meant nothing was compared and xarray multiplied out.

    A (366, 2, 2) normal against a (366, 2, 2) observed came back (366, 2, 2, 2, 2) — on a
    country grid that is thousands of times the memory, for a meaningless result.
    """
    observed = _observed_doy(offset=5.0)

    out = compute_anomaly(observed, _normal_named("latitude", "longitude"), method="absolute")

    assert out.dims == observed.dims, f"broadcast instead of aligned: {out.dims}"
    assert out.shape == observed.shape
    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0)


def test_a_normal_with_unrecognised_extra_axes_is_refused() -> None:
    """Axis names that cannot be mapped onto x/y would broadcast just the same."""
    normal = _normal_named("northing", "easting")

    with pytest.raises(ValueError, match="broadcast rather than align"):
        compute_anomaly(_observed_doy(), normal, method="absolute")


def _with_crs(da: xr.DataArray, epsg: int) -> xr.DataArray:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

    return da.rio.write_crs(epsg)


def test_a_normal_in_a_different_projection_is_refused() -> None:
    """Coordinates only compare within one projection.

    Neighbouring UTM zones carry overlapping eastings and northings, so identical numbers can
    describe places hundreds of kilometres apart and every numeric check still passes.
    """
    observed = _with_crs(_observed_doy(), 32645)
    normal = _with_crs(_normal_doy(), 32644)

    with pytest.raises(ValueError, match="coordinates are not comparable"):
        compute_anomaly(observed, normal, method="absolute")


def test_the_same_projection_on_both_sides_is_fine() -> None:
    observed = _with_crs(_observed_doy(offset=5.0), 32645)
    normal = _with_crs(_normal_doy(), 32645)

    out = compute_anomaly(observed, normal, method="absolute")

    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0)


def test_a_crs_on_only_one_side_is_not_an_error() -> None:
    """Nothing to compare against, so the numeric checks stand alone."""
    observed = _with_crs(_observed_doy(offset=5.0), 4326)

    out = compute_anomaly(observed, _normal_doy(), method="absolute")

    assert float(out.isel(t=99, y=0, x=0)) == pytest.approx(5.0)


def test_anomaly_keeps_the_crs_coordinate_scalar() -> None:
    """earthkit's per-timestep indexing broadcasts the normal's scalar coords along time.

    A `spatial_ref` with a time dimension cannot be published — `xproj.assign_crs` raises
    "can only create a CRSIndex from one scalar variable" — and a per-timestep CRS would be
    meaningless if it wrote. Found by publishing a real monthly temperature anomaly.
    """
    times = pd.date_range("2026-01-01", periods=3, freq="MS")
    observed = xr.DataArray(
        np.ones((3, 2, 2), dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 0.0], "x": [0.0, 1.0], "spatial_ref": 0},
        attrs={"units": "degC"},
    )
    normal = xr.DataArray(
        np.zeros((12, 2, 2), dtype="float32"),
        dims=("month", "y", "x"),
        coords={"month": range(1, 13), "y": [1.0, 0.0], "x": [0.0, 1.0], "spatial_ref": 0},
        attrs={"units": "degC"},
    )

    result = compute_anomaly(observed, normal, method="absolute")

    assert result["spatial_ref"].dims == (), f"spatial_ref gained dims {result['spatial_ref'].dims}"
    assert "month" not in result.coords, "the normal's ordinal coord leaked into the anomaly"
    assert result.dims == ("t", "y", "x")


def test_relative_anomaly_masks_cells_with_a_zero_normal() -> None:
    """A zero normal is ordinary in a dry cell, and dividing by it has no answer.

    Unmasked, `100 * (observed - 0) / 0` published ±inf into the store as if it were a real
    percentage — and one inf poisons any spatial mean taken over the field.
    """
    times = pd.date_range("2026-01-01", periods=2, freq="MS")
    observed = xr.DataArray(
        np.array([[[2.0, 2.0]], [[2.0, 2.0]]], dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [0.0], "x": [0.0, 1.0]},
        attrs={"units": "mm/d"},
    )
    # Second cell is dry in every month: a legitimate zero normal.
    normal = xr.DataArray(
        np.tile(np.array([[1.0, 0.0]], dtype="float32"), (12, 1, 1)),
        dims=("month", "y", "x"),
        coords={"month": range(1, 13), "y": [0.0], "x": [0.0, 1.0]},
        attrs={"units": "mm/d"},
    )

    result = compute_anomaly(observed, normal, method="relative").compute()

    assert np.isfinite(result.isel(x=0)).all(), "the wet cell must keep its percentage"
    assert np.isnan(result.isel(x=1)).all(), "the zero-normal cell must be NaN, not inf"


def test_month_normal_refuses_a_quarterly_observed_cube() -> None:
    """`>= 20 days` alone accepted bimonthly, quarterly and yearly series as monthly."""
    times = pd.date_range("2026-01-01", periods=4, freq="QS")
    observed = xr.DataArray(
        np.ones((4, 1, 1), dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [0.0], "x": [0.0]},
        attrs={"units": "mm/d"},
    )
    normal = xr.DataArray(
        np.zeros((12, 1, 1), dtype="float32"),
        dims=("month", "y", "x"),
        coords={"month": range(1, 13), "y": [0.0], "x": [0.0]},
        attrs={"units": "mm/d"},
    )

    with pytest.raises(ValueError, match="expects a monthly observed dataset"):
        compute_anomaly(observed, normal, method="absolute")


def test_broadcasting_is_refused_even_when_the_observed_axes_are_unrecognised() -> None:
    """The guard used to sit behind the unknown-axes early return, so this broadcast to 5-D."""
    times = pd.date_range("2026-01-01", periods=2, freq="D")
    observed = xr.DataArray(
        np.ones((2, 2, 2), dtype="float32"),
        dims=("t", "northing", "easting"),
        coords={"t": times, "northing": [1.0, 0.0], "easting": [0.0, 1.0]},
        attrs={"units": "degC"},
    )
    normal = xr.DataArray(
        np.zeros((366, 2, 2), dtype="float32"),
        dims=("dayofyear", "y", "x"),
        coords={"dayofyear": range(1, 367), "y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"units": "degC"},
    )

    with pytest.raises(ValueError):
        compute_anomaly(observed, normal, method="absolute")
