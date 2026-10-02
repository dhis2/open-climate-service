from datetime import date

import pytest
import numpy as np
import xarray as xr

from open_climate_service.plugins.processes.aggregate_spatial_weighted import (
    aggregate_spatial_weighted,
)

def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }

def _grid(y_ascending: bool) -> xr.DataArray:
    """4x4 grid of unit cells centred on 0..3, where each cell value encodes y*10 + x."""
    xv = np.array([0.0, 1.0, 2.0, 3.0])
    yv = xv.copy() if y_ascending else xv[::-1].copy()
    data = np.array([[yc * 10 + xc for xc in xv] for yc in yv])
    return xr.DataArray(data, dims=("y", "x"), coords={"y": yv, "x": xv}, name="v")


def test_aggregate_spatial_dataarray_lower_left_box() -> None:
    """A box symmetric over cells (0,0),(1,0),(0,1),(1,1) weights them equally."""
    da = _grid(y_ascending=True)
    out = aggregate_spatial_weighted(da, _box(-0.4, -0.4, 1.4, 1.4), "mean")
    # values {0, 1, 10, 11}, each 0.9 x 0.9 covered -> mean 5.5
    assert float(out.isel(geometry=0)) == pytest.approx(5.5)

def test_aggregate_spatial_with_time_dimension() -> None:
    base = _grid(y_ascending=True)
    years = [2025, 2026]
    t_coords = [date(year=y, month=1, day=1) for y in years]
    ds = xr.concat([base, base + 100], dim="t").assign_coords(t=t_coords).to_dataset(name="v")
    out = aggregate_spatial_weighted(ds, _box(-0.4, -0.4, 1.4, 1.4), "mean")
    assert list(out["t"].values) == t_coords
    np.testing.assert_allclose(out.isel(geometry=0).values, [5.5, 105.5])
