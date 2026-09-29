"""aggregate_spatial — zonal statistics plugin process.

Semantics deliberately diverge from the openEO specification (CLIM-785):

* **Polygons are area-weighted.** Each cell counts by the fraction of it the polygon covers,
  computed by exactextract. The specification's pixel-centre rule returns NaN for any zone
  smaller than a cell, and misplaces every zone edge by up to half a cell.
* **Points are interpolated.** A point samples the surface at its location (bilinear by
  default) rather than taking whichever cell contains it.
* **Categorical data is never averaged.** A cube whose dataset declares
  ``ingestion.resampling: mode`` aggregates polygons by area-weighted majority and samples
  points from the nearest cell.

The weighted path applies to reducers that are exactly one known statistic (mean, sum, min,
max, median, and the categorical majority and fractions), recognised by the structure of their
graph. Any other reducer is an arbitrary process graph that cannot be weighted, so it falls back
to the specification's pixel-centre selection and runs as given.

Everything but the process itself lives in ``open_climate_service.zonal``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.process import process
from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD
from open_climate_service.zonal.categorical import CATEGORICAL_REPLACED, categorical_variables
from open_climate_service.zonal.geometries import POINT_TYPES, POLYGON_TYPES, parse_geometries, to_cube_crs
from open_climate_service.zonal.grid import Grid, VarResult, find_dim, polygon_zones
from open_climate_service.zonal.pixel_centre import pixel_centre_polygons
from open_climate_service.zonal.points import POINT_METHODS, point_method, sampled_points
from open_climate_service.zonal.reducers import identify_reducer
from open_climate_service.zonal.reducers import (
    reduce_by_method as reduce_by_method,  # a process too; the loader registers it from here
)
from open_climate_service.zonal.weighting import weighted_polygons

logger = logging.getLogger(__name__)


def _warn_empty_zones(result: xr.Dataset, geom_dim: str, labels: list[str]) -> None:
    """Log the geometries that returned no value at all, which otherwise arrive as a silent NaN."""
    empty = np.ones(len(labels), dtype=bool)
    for da in result.data_vars.values():
        flat = da.transpose(geom_dim, ...).values.reshape(len(labels), -1)
        if flat.shape[1]:
            empty &= np.isnan(flat).all(axis=1)
    missing = [label for label, is_empty in zip(labels, empty, strict=True) if is_empty]
    if missing:
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10 else "")
        logger.warning(
            "aggregate_spatial: %d of %d geometries captured no data and are NaN: %s",
            len(missing),
            len(labels),
            shown,
        )


@process(
    summary="Aggregate spatial data within geometries",
    parameters={
        "data": {"description": "A raster data cube."},
        "geometries": {"description": "GeoJSON FeatureCollection, Feature, or geometry (polygons or points)."},
        "reducer": {"description": "A reducer to apply on the pixel values."},
        "target_dimension": {"description": "Name for the new geometry dimension (default: 'geometry')."},
        "context": {"description": "Optional context passed to the reducer."},
        "method": {
            "description": (
                "OCS extension. How points sample the surface, in openEO's resample_spatial "
                "vocabulary: near, bilinear or cubic. Defaults to bilinear, or near for a "
                "categorical dataset. Polygons are always area-weighted."
            )
        },
    },
)
def aggregate_spatial(
    data: Any,
    geometries: Any,
    reducer: Callable,
    target_dimension: str | None = None,
    context: Any = None,
    method: str | None = None,
) -> xr.Dataset:
    """Aggregate raster values within each polygon, or sample them at each point."""
    if method is not None and method not in POINT_METHODS:
        # Checked before anything else, so a typo fails whether or not the geometries have points.
        raise ValueError(
            f"aggregate_spatial: method '{method}' is not supported; expected one of {sorted(POINT_METHODS)}"
        )
    geom_shapes, geom_labels = parse_geometries(geometries)
    if not geom_shapes:
        raise ValueError("aggregate_spatial: geometries contains no shapes")

    if isinstance(data, xr.DataArray):
        # The variable keeps the DataArray's attributes, including the categorical marker.
        data = data.to_dataset(name=data.name or "data")

    x_dim = find_dim(data, ["x", "longitude", "lon"])
    y_dim = find_dim(data, ["y", "latitude", "lat"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial: cannot identify x/y dimensions in {list(data.dims)}")
    grid = Grid.of(data, x_dim, y_dim)
    # Computed in the cube's CRS; the result keeps the shapes as supplied, which the vector
    # writers read as the request's GeoJSON (WGS 84).
    geom_shapes, wgs84_shapes = to_cube_crs(geom_shapes, data)

    categorical = categorical_variables(data)
    named = identify_reducer(reducer)
    variables = [str(v) for v in data.data_vars]
    # Per variable: a categorical one takes the majority where the reducer would average codes.
    effective: dict[str, str] = {}
    if named is not None:
        effective = {v: "majority" if v in categorical and named in CATEGORICAL_REPLACED else named for v in variables}
        replaced = sorted(v for v in variables if effective[v] != named)
        if replaced:
            logger.warning(
                "aggregate_spatial: '%s' requested over categorical data (%s); using the area-weighted "
                "majority class instead of averaging class codes",
                named,
                ", ".join(replaced),
            )
    else:
        logger.info(
            "aggregate_spatial: reducer is not one the area-weighted path recognises; polygons "
            "use pixel-centre selection"
        )

    polygon_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in POLYGON_TYPES]
    point_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in POINT_TYPES]
    if point_idx and named == "fractions":
        raise ValueError("aggregate_spatial: fractions apply to polygons; a point has one class, not a share")

    from open_climate_service.shared.provenance import (
        observe_spatial_aggregation,
        record_spatial_reduction,
        unattributed_spatial_reduction,
    )

    parts: list[tuple[list[int], dict[str, VarResult]]] = []
    with observe_spatial_aggregation():
        # The reducer is not called on the weighted path, so what ran is recorded here. Two
        # different methods (a merged categorical and continuous cube) record as unattributable,
        # and so does any point: a sampled value is not the named reduction.
        if not point_idx:
            for used in sorted(set(effective.values())):
                record_spatial_reduction(used)
        if polygon_idx:
            polygons = [geom_shapes[i] for i in polygon_idx]
            if effective:
                values = weighted_polygons(data, grid, polygon_zones(grid, polygons), effective)
            else:
                with unattributed_spatial_reduction():
                    values = pixel_centre_polygons(data, grid, polygons, reducer, context)
            parts.append((polygon_idx, values))
        if point_idx:
            points = [geom_shapes[i] for i in point_idx]
            point_methods = {
                v: point_method(method, v in categorical or effective.get(v) == "majority") for v in variables
            }
            with unattributed_spatial_reduction():
                values = sampled_points(data, grid, points, point_methods, None if effective else reducer, context)
            parts.append((point_idx, values))

    geom_dim = target_dimension or "geometry"
    combined = _assemble(parts, len(geom_shapes), geom_dim)
    _warn_empty_zones(combined, geom_dim, geom_labels)
    combined[geom_dim] = geom_labels
    # Carry the shapes as well as the labels, so the result is a vector datacube rather than a
    # table that has forgotten where it came from. A companion coordinate rather than replacing
    # the labels on `geom_dim`: the label is the feature id, which the DHIS2 and CHAP exports key
    # their location column on. WKT strings, because a string coordinate is inert on every path
    # the cube can take, where an object-dtype one makes `to_zarr` fail. See CLIM-836.
    return combined.assign_coords({GEOMETRY_WKT_COORD: (geom_dim, [geom.wkt for geom in wgs84_shapes])})


def _assemble(
    parts: list[tuple[list[int], dict[str, VarResult]]],
    n_geometries: int,
    geom_dim: str,
) -> xr.Dataset:
    """Put polygon and point results back in the input order along *geom_dim*."""
    variables: dict[str, xr.DataArray] = {}
    for name in parts[0][1]:
        first, dims, coords = parts[0][1][name]
        full = np.full((n_geometries, *first.shape[1:]), np.nan)
        for idx, values in parts:
            arr = values[name][0]
            if arr.shape[1:] != first.shape[1:]:
                raise ValueError(f"aggregate_spatial: points and polygons produced different shapes for '{name}'")
            full[idx] = arr
        variables[name] = xr.DataArray(full, dims=[geom_dim, *dims], coords=coords)
    return xr.Dataset(variables)
