"""aggregate_spatial_weighted — weighted zonal statistics plugin process."""

from typing import Any, Callable

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

from open_climate_service.process import process

REDUCERS = ("mean", "sum", "min", "max", "median")
"""The statistics the weighted aggregation offers: the methods a DHIS2 export can declare."""


@process(
    summary="Aggregate a raster data cube over vector geometries using spatially weighted statistics.",
    description="For each geometry, pixels overlapping the geometry are spatially "
    "aggregated using their fractional spatial overlap as weights.",
    parameters={
        "data": {
            "description": "A raster data cube.",
            "schema": {
                "type": "object",
                "subtype": "datacube",
                "dimensions": [{"type": "spatial", "axis": ["x", "y"]}],
            },
        },
        "geometries": {
            "description": (
                "A vector data cube (a GeoDataFrame or xvec cube), or GeoJSON: a FeatureCollection, "
                "Feature or geometry. Each feature's id is kept as the result's `feature_id`."
            ),
            "schema": [
                {"type": "object", "subtype": "datacube", "dimensions": [{"type": "geometry"}]},
                {"type": "object", "subtype": "geojson"},
            ],
        },
        "reducer": {
            "description": (
                "The statistic, by name. The pixels are weighted by their overlap with the geometry, "
                "which exactextract does for these named statistics only, not for a reducer process."
            ),
            "schema": {"type": "string", "enum": list(REDUCERS)},
        },
    },
)
def aggregate_spatial_weighted(
    data: xr.Dataset | xr.DataArray,
    geometries: Any,
    reducer: str | Callable,
) -> xr.DataArray:
    """Spatially aggregate raster values over vector geometries using fractional pixel overlap as weights.

    For each geometry, only the portion of each pixel covered by the
    geometry contributes to the aggregation. The specified reducer
    determines how the values are aggregated.

    Parameters
    ----------
    data
        Raster data cube to aggregate (either xr.DataArray or single-variable xr.Dataset).
    geometries
        Vector geometries over which to aggregate the raster values.
    reducer
        Statistic to calculate, or a reducer callable.

    Returns:
    -------
    VectorCube
        Vector data cube (xr.DataArray named after the input variable) containing one aggregated
        value per geometry, with the geometries on `geometry`, each feature's id as `feature_id`
        and the non-spatial dimensions of the input cube preserved.
    """
    # NOTE: adapted from openeo_processes_dask.processes.aggregate_spatial to support exactextract
    from open_climate_service.shared.provenance import record_features, record_spatial_aggregation
    from open_climate_service.shared.vectors import features_in_crs, single_raster, vector_result

    if not isinstance(reducer, str) or reducer not in REDUCERS:
        raise ValueError(
            f"aggregate_spatial_weighted: reducer must be one of {', '.join(REDUCERS)}, by name; "
            "to aggregate with an openEO reducer process, use aggregate_spatial instead"
        )
    record_features(geometries)
    raster = single_raster(data)
    frame = features_in_crs(geometries, raster.rio.crs)

    # Run xvec zonal stats with exactextract backend.
    vec_cube: xr.DataArray = raster.xvec.zonal_stats(
        frame.geometry,
        x_coords="x",
        y_coords="y",
        method="exactextract",
        stats=reducer,
    )
    # The named method, so a DHIS2 export can check the aggregation it declares against this one.
    record_spatial_aggregation(reducer)
    return vector_result(vec_cube, raster, frame.index)
