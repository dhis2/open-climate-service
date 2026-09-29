"""Sampling the surface at point geometries."""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.zonal.grid import Grid, VarResult
from open_climate_service.zonal.reducers import make_reducer_caller

logger = logging.getLogger(__name__)

# openEO `resample_spatial` vocabulary, mapped to xarray interpolation methods.
POINT_METHODS = {"near": "nearest", "bilinear": "linear", "cubic": "cubic"}


def point_method(requested: str | None, categorical: bool) -> str:
    """The xarray interpolation method for points, from openEO's resampling vocabulary."""
    if requested is None:
        return "nearest" if categorical else "linear"
    if requested not in POINT_METHODS:
        raise ValueError(
            f"aggregate_spatial: method '{requested}' is not supported; expected one of {sorted(POINT_METHODS)}"
        )
    if categorical and requested != "near":
        logger.warning(
            "aggregate_spatial: '%s' interpolation requested over categorical data; class codes "
            "between cells are interpolated as if ordinal",
            requested,
        )
    return POINT_METHODS[requested]


def _sample_points(data: xr.Dataset, grid: Grid, points: Sequence[Any], method: str) -> xr.Dataset:
    """Sample every variable at each point, as a cube with a leading ``__point__`` dimension.

    Where interpolation returns NaN — a point within half a cell of the grid edge, or next to
    a missing cell such as sea beside a coastal facility — the containing cell's value is used.
    """
    xs = xr.DataArray([p.x for p in points], dims="__point__")
    ys = xr.DataArray([p.y for p in points], dims="__point__")
    nearest = data.sel({grid.x_dim: xs, grid.y_dim: ys}, method="nearest")
    inside = (xs >= grid.xmin) & (xs <= grid.xmax) & (ys >= grid.ymin) & (ys <= grid.ymax)
    nearest = nearest.where(inside)
    if method == "nearest":
        return nearest.drop_vars([grid.x_dim, grid.y_dim], errors="ignore")
    interpolated = data.interp({grid.x_dim: xs, grid.y_dim: ys}, method=method)  # type: ignore[arg-type]
    interpolated = interpolated.drop_vars([grid.x_dim, grid.y_dim], errors="ignore")
    return interpolated.fillna(nearest.drop_vars([grid.x_dim, grid.y_dim], errors="ignore"))


def sampled_points(
    data: xr.Dataset,
    grid: Grid,
    points: Sequence[Any],
    methods: dict[str, str],
    reducer: Callable | None,
    context: Any,
) -> dict[str, VarResult]:
    """Per variable, sampled by its own method; passed through *reducer* when that is unknown."""
    if "cubic" in methods.values() and (grid.width < 4 or grid.height < 4):
        raise ValueError(
            f"aggregate_spatial: cubic interpolation needs at least 4 cells along each axis; this grid "
            f"is {grid.width} x {grid.height}, so use 'bilinear' or 'near'"
        )
    sampled = xr.merge(
        [
            _sample_points(data[[v for v, m in methods.items() if m == method]], grid, points, method)
            for method in sorted(set(methods.values()))
        ],
        compat="override",
    )
    reduce = make_reducer_caller(reducer, context) if reducer is not None else None
    out: dict[str, VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        da = sampled[vname]
        other: list[Hashable] = [d for d in da.dims if d != "__point__"]
        arr = np.asarray(da.transpose("__point__", *other).values, dtype="float64")
        if reduce is not None:
            # An unknown reducer sees the one sampled value, as it would see a zone's pixels.
            flat = arr.reshape(arr.shape[0], -1)
            reduced = [[reduce(np.array([v])) if not np.isnan(v) else np.nan for v in row] for row in flat]
            arr = np.asarray(reduced, dtype="float64").reshape(arr.shape)
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
        out[vname] = (arr, other, coords)
    return out
