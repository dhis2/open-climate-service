"""aggregate_spatial_weighted — weighted zonal statistics plugin process."""

import logging
from typing import Any, Callable

import geopandas as gpd
import shapely
import xarray as xr

from open_climate_service.process import process

logger = logging.getLogger(__name__)


@process(
    summary="Spatially aggregate a raster data cube over vector geometries using spatially weighted statistics.",
    description="For each geometry, pixels overlapping the geometry are spatially "
    "aggregated using their fractional spatial overlap as weights. "
    "Multiple statistics can be calculated in a single operation.",
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
        Vector data cube (xr.DataArray) containing one aggregated value per geometry,
        with non-spatial dimensions of the input cube preserved.
    """
    # NOTE: adapted from openeo_processes_dask.processes.aggregate_spatial to support exactextract

    x_dim = "x"
    y_dim = "y"
    default_crs = "EPSG:4326"

    # Ensure raster cube is a single-variable DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 1:
            data = data[list(data.data_vars.keys())[0]]

        else:
            raise ValueError(
                "The data parameter needs to be a raster cube in the form of an xarray DataArray "
                "or a single-variable xarray Dataset, received: \n{data}"
            )

        assert isinstance(data, xr.DataArray)

    # Ensure raster cube has crs
    if data.rio.crs is None:
        data = data.rio.set_crs(default_crs)

    # Allow importing geometries from url (e.g. github raw)
    if isinstance(geometries, str):
        import json
        from urllib.request import urlopen

        response = urlopen(geometries)
        geometries = json.loads(response.read())

    # Convert GeoJSON dict to GeoDataFrame
    if isinstance(geometries, dict):
        # Get crs from geometries
        if "features" in geometries:
            for feature in geometries["features"]:
                if "properties" not in feature:
                    feature["properties"] = {}
                elif feature["properties"] is None:
                    feature["properties"] = {}
            if isinstance(geometries.get("crs", {}), dict):
                default_crs = geometries.get("crs", {}).get("properties", {}).get("name", default_crs)
            else:
                default_crs = str(geometries.get("crs", {}))
            logger.info(f"CRS in geometries: {default_crs}.")

        if "type" in geometries and geometries["type"] == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=default_crs)
        elif "type" in geometries and geometries["type"] in ["Polygon"]:
            polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
            gdf = gpd.GeoDataFrame(geometry=[polygon])
            gdf.crs = default_crs

    # Convert xarray vector cube to GeoDataFrame
    if isinstance(geometries, xr.Dataset):
        if hasattr(geometries, "xvec"):
            gdf = geometries.xvec.to_geodataframe()

    if isinstance(geometries, gpd.GeoDataFrame):
        gdf = geometries

    else:
        raise TypeError(f"Failed to convert geometries input value to GeoDataFrame: {geometries}")

    # Reproject geometries to same crs as raster cube
    gdf = gdf.to_crs(data.rio.crs)

    # Convert to geopandas geometries Series
    geometries_series = gdf.geometry

    # Run xvec zonal stats with exactextract backend
    vec_cube: xr.DataArray = data.xvec.zonal_stats(
        geometries_series,
        x_coords=x_dim,
        y_coords=y_dim,
        method="exactextract",
        stats=reducer,
    )

    return vec_cube
