"""The openEO specification's pixel-centre rule, for reducers the weighted path cannot compute."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.zonal.grid import Grid, VarResult, blocks
from open_climate_service.zonal.reducers import is_graph_callback, make_reducer_caller


def pixel_centre_polygons(
    data: xr.Dataset, grid: Grid, polygons: Sequence[Any], reducer: Callable, context: Any
) -> dict[str, VarResult]:
    """The openEO specification's rule, for reducers the weighted path does not know.

    A cell counts when its centre lies inside the polygon; the reducer runs on those values.
    An openEO graph gets the missing ones too, as the specification passes them, so its own
    ``ignore_nodata`` decides; a plain Python reducer gets only the valid ones.
    """
    import rasterio.features
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    transform = from_bounds(grid.xmin, grid.ymin, grid.xmax, grid.ymax, grid.width, grid.height)
    reduce = make_reducer_caller(reducer, context)
    keep_nodata = is_graph_callback(reducer)
    out: dict[str, VarResult] = {}
    masks = []
    for geom in polygons:
        mask = rasterio.features.geometry_mask(
            [mapping(geom)], out_shape=(grid.height, grid.width), transform=transform, invert=True
        )
        # rasterio builds the mask north row first; flip when the cube's y ascends.
        if grid.y_ascending:
            mask = mask[::-1]
        if not grid.x_ascending:
            mask = mask[:, ::-1]
        masks.append(mask.ravel())
    for name in data.data_vars:
        vname = str(name)
        da = data[vname]
        other: list[Hashable] = [d for d in da.dims if d not in {grid.y_dim, grid.x_dim}]
        pieces: list[list[np.ndarray]] = [[] for _ in masks]
        concat_axis = 0
        # A block at a time, as on the weighted path, so memory does not grow with the series.
        for axis, block in blocks(da, other, grid):
            concat_axis = axis or 0
            flat = block.reshape((-1, grid.height * grid.width))
            for mask, mask_pieces in zip(masks, pieces, strict=True):
                selected = (pixels[mask] for pixels in flat)
                reduced = [reduce(v if keep_nodata else v[~np.isnan(v)]) for v in selected]
                mask_pieces.append(np.asarray(reduced, dtype="float64").reshape(block.shape[:-2]))
        rows = [np.concatenate(p, axis=concat_axis) if other else p[0] for p in pieces]
        shape = tuple(int(da.sizes[d]) for d in other)
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
        out[vname] = (np.stack(rows) if rows else np.empty((0, *shape)), other, coords)
    return out
