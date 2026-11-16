"""Tests for the shared circular rolling mean (CLIM-859)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_geo_service.transforms.climatology import circular_rolling_mean


def _cube(n: int, dim: str = "dayofyear", seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(n, 3, 4)),
        dims=(dim, "y", "x"),
        coords={dim: np.arange(1, n + 1), "y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]},
        name="v",
    )


def _reference(da: xr.DataArray, window: int, dim: str = "dayofyear") -> np.ndarray:
    """Independent numpy implementation: triple the axis and average a centred window.

    Deliberately not the implementation under test — this is the eager formulation the
    era5_land plugin used before consolidating onto earthkit, kept as the oracle.
    """
    values = np.concatenate([da.values, da.values, da.values], axis=0)
    out = np.empty_like(da.values)
    half, n = window // 2, da.sizes[dim]
    for i in range(n):
        centre = n + i
        out[i] = values[centre - half : centre + half + 1].mean(axis=0)
    return out


@pytest.mark.parametrize(("n", "window"), [(366, 31), (366, 5), (365, 31), (12, 3), (31, 31), (7, 1)])
def test_matches_an_independent_implementation(n: int, window: int) -> None:
    da = _cube(n)
    result = circular_rolling_mean(da, window)
    np.testing.assert_allclose(result.values, _reference(da, window))


def test_wraps_the_year_boundary() -> None:
    """The point of the helper: earthkit's rolling reduction alone leaves NaN at both ends."""
    da = _cube(366)
    result = circular_rolling_mean(da, 31)
    assert np.isfinite(result.values).all()


def test_first_step_averages_across_the_boundary() -> None:
    """Day 1 of a 3-day window must be the mean of the last day, day 1 and day 2."""
    da = _cube(10)
    result = circular_rolling_mean(da, 3)
    expected = (da.isel(dayofyear=-1) + da.isel(dayofyear=0) + da.isel(dayofyear=1)) / 3
    np.testing.assert_allclose(result.isel(dayofyear=0).values, expected.values)


def test_preserves_shape_and_coordinates() -> None:
    da = _cube(366)
    result = circular_rolling_mean(da, 31)
    assert result.sizes == da.sizes
    assert result.dims == da.dims
    np.testing.assert_array_equal(result["dayofyear"].values, da["dayofyear"].values)


def test_window_of_one_is_the_identity() -> None:
    da = _cube(12)
    xr.testing.assert_identical(circular_rolling_mean(da, 1), da)


def test_preserves_dask_laziness() -> None:
    """A (dim, y, x) climatology must not be materialised on the way to the store."""
    da = _cube(366).chunk({"dayofyear": 30})
    result = circular_rolling_mean(da, 31)
    assert result.chunks is not None


@pytest.mark.parametrize("chunk", [30, 100, 122, 366])
def test_result_keeps_the_input_chunking_so_it_stays_writable(chunk: int) -> None:
    """Zarr takes uniform chunks with a smaller final one, and nothing else.

    Padding and slicing the cyclic axis fragments it — chunking at 30 came back as
    ``(15, 30, …, 36, 15)`` — which no Zarr write would accept. Laziness alone is not enough
    to assert, since a lazy result can still be unwritable.
    """
    da = _cube(366).chunk({"dayofyear": chunk})
    assert da.chunks is not None  # precondition: the input really is chunked
    expected = da.chunks[0]

    result = circular_rolling_mean(da, 31)

    assert result.chunks is not None
    sizes = result.chunks[0]
    assert sizes == expected, f"chunking changed: {expected} -> {sizes}"
    assert len(set(sizes[:-1])) <= 1 and sizes[-1] <= sizes[0], f"not writable: {sizes}"


def test_constant_field_is_unchanged() -> None:
    da = xr.DataArray(
        np.full((366, 2, 2), 7.5),
        dims=("dayofyear", "y", "x"),
        coords={"dayofyear": np.arange(1, 367), "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    np.testing.assert_allclose(circular_rolling_mean(da, 31).values, 7.5)


def test_works_on_a_month_axis() -> None:
    """The helper is not hard-wired to dayofyear."""
    da = _cube(12, dim="month")
    result = circular_rolling_mean(da, 3, dim="month")
    np.testing.assert_allclose(result.values, _reference(da, 3, dim="month"))


@pytest.mark.parametrize(
    ("window", "match"),
    [
        (0, "must be >= 1"),
        (-1, "must be >= 1"),
        (4, "must be odd"),
        (401, "must be <="),  # odd, so it reaches the length check rather than the parity one
    ],
)
def test_rejects_invalid_windows(window: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        circular_rolling_mean(_cube(366), window)
