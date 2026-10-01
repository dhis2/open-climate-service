"""Bbox selection must keep every cell the extent touches, not every cell centre it contains.

A plain `slice(low, high)` selects on cell *centres*, so the cells straddling each bbox edge are
dropped and the store covers less than the configured extent — up to half a cell short per side.
On the 0.25° GEFS grid that left eastern Nepal visibly uncovered on the map, and border districts
aggregating from partial data.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_climate_service.streaming import bbox_slice, bbox_slices, cell_pad

# Nepal's configured extent, which aligns to no regular grid.
NEPAL = (80.05, 26.35, 88.20, 30.45)


def _grid(step: float, *, descending: bool) -> xr.DataArray:
    """A global axis on `step` spacing, as a geographic source would publish it."""
    values = np.arange(-90.0, 90.0 + step, step)
    if descending:
        values = values[::-1]
    return xr.DataArray(values, dims="lat", coords={"lat": values})


def _covered(coord: xr.DataArray, selector: slice) -> tuple[float, float]:
    """The selection's real coverage — cell edges, not centres."""
    picked = np.asarray(coord.sel({str(coord.dims[0]): selector}).values, dtype="float64")
    pad = cell_pad(coord)
    return float(picked.min() - pad), float(picked.max() + pad)


@pytest.mark.parametrize("step", [0.25, 0.1, 1.0])
@pytest.mark.parametrize("descending", [True, False])
def test_selection_covers_the_whole_requested_span(step: float, descending: bool) -> None:
    coord = _grid(step, descending=descending)
    low, high = NEPAL[1], NEPAL[3]

    south, north = _covered(coord, bbox_slice(coord, low, high))

    assert south <= low, f"{south} > {low}: south edge uncovered on a {step}° grid"
    assert north >= high, f"{north} < {high}: north edge uncovered on a {step}° grid"


@pytest.mark.parametrize("descending", [True, False])
def test_a_plain_label_slice_is_what_leaves_the_gap(descending: bool) -> None:
    """The regression this guards against, stated as a fact about the naive approach."""
    coord = _grid(0.25, descending=descending)
    low, high = NEPAL[1], NEPAL[3]
    naive = slice(high, low) if descending else slice(low, high)

    south, north = _covered(coord, naive)

    assert south > low and north < high, "precondition: the naive slice under-covers"
    assert pytest.approx(south - low, abs=1e-9) == 0.025
    assert pytest.approx(high - north, abs=1e-9) == 0.075


def test_selection_adds_at_most_one_cell_per_side() -> None:
    """Over-selection is safe but must stay bounded — normalize_period trims the rest."""
    coord = _grid(0.25, descending=True)
    low, high = NEPAL[1], NEPAL[3]

    picked = coord.sel(lat=bbox_slice(coord, low, high)).size
    naive = coord.sel(lat=slice(high, low)).size

    assert picked - naive <= 2, f"grew by {picked - naive} cells; expected at most one per side"


def test_bbox_slices_returns_both_axes_in_their_own_direction() -> None:
    lat = np.arange(30.75, 26.0, -0.25)
    lon = np.arange(80.0, 88.5, 0.25)
    ds = xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.ones((lat.size, lon.size), "float32"))},
        coords={"latitude": lat, "longitude": lon},
    )

    selectors = bbox_slices(ds, list(NEPAL), x_dim="longitude", y_dim="latitude")
    picked = ds.sel(selectors)

    y = np.asarray(picked["latitude"].values)
    x = np.asarray(picked["longitude"].values)
    assert y.min() - 0.125 <= NEPAL[1] and y.max() + 0.125 >= NEPAL[3]
    assert x.min() - 0.125 <= NEPAL[0] and x.max() + 0.125 >= NEPAL[2]


def test_cell_pad_uses_the_widest_spacing_on_an_irregular_axis() -> None:
    """Sufficiency matters more than tightness: an extra cell is trimmed, a missing one is not."""
    coord = xr.DataArray([0.0, 0.1, 0.5, 0.6], dims="lat")

    assert cell_pad(coord) == pytest.approx(0.2)


def test_cell_pad_on_a_single_cell_axis_is_zero() -> None:
    """No spacing to infer, and nothing to pad — must not raise."""
    assert cell_pad(xr.DataArray([42.0], dims="lat")) == 0.0
