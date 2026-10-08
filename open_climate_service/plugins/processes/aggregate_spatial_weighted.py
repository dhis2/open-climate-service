"""aggregate_spatial_weighted — weighted zonal statistics plugin process."""

from typing import Any, Callable

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

from open_climate_service.process import process


@process(
    summary="Aggregate a raster data cube over vector geometries using spatially weighted statistics.",
    description="For each geometry, pixels overlapping the geometry are spatially "
    "aggregated using their fractional spatial overlap as weights.",
    parameters={
        "data": {"description": "A raster data cube."},
        "geometries": {
            "description": (
                "A vector data cube (a GeoDataFrame or xvec cube), or GeoJSON: a FeatureCollection, "
                "Feature or geometry. Each feature's id is kept as the result's `feature_id`."
            )
        },
        "reducer": {"description": "A reducer to apply on the pixel values."},
    },
)
def aggregate_spatial_weighted(
    data: xr.Dataset | xr.DataArray,
    geometries: Any,
    reducer: str | Callable,
) -> xr.Dataset:
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
        Vector data cube (xr.Dataset named after the input variable) containing one aggregated
        value per geometry, with each feature's id as `feature_id` and the non-spatial dimensions
        of the input cube preserved.
    """
    # NOTE: adapted from openeo_processes_dask.processes.aggregate_spatial to support exactextract
    from open_climate_service.shared.provenance import observe_spatial_aggregation, record_features, record_reduction
    from open_climate_service.shared.vectors import raster_and_features, vector_result

    record_features(geometries)
    raster, frame = raster_and_features(data, geometries)

    # Run xvec zonal stats with exactextract backend. A named method is recorded, so a DHIS2
    # export can check the aggregation it declares against the one that ran.
    with observe_spatial_aggregation():
        if isinstance(reducer, str):
            record_reduction(reducer)
        vec_cube: xr.DataArray = raster.xvec.zonal_stats(
            frame.geometry,
            x_coords="x",
            y_coords="y",
            method="exactextract",
            stats=reducer,
        )
    return vector_result(vec_cube, raster, frame.index)
