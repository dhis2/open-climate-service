import logging
from typing import Any, Callable

import geopandas as gpd
import xarray as xr
import shapely
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from open_climate_service.process import process


logger = logging.getLogger(__name__)


@process(
  summary="Spatially aggregate a raster data cube over vector geometries using spatially weighted statistics.",
  description="For each geometry, pixels overlapping the geometry are spatially "
    "aggregated using their fractional spatial overlap as weights. "
    "Multiple statistics can be calculated in a single operation.",
)
def aggregate_spatial_weighted(
    data: RasterCube,
    geometries: Any,
    reducer: str | Callable,
) -> VectorCube:
    # NOTE: adapted from openeo_processes_dask.processes.aggregate_spatial to support exactextract

    x_dim = "x"
    y_dim = "y"
    DEFAULT_CRS = "EPSG:4326"

    # Ensure raster cube is a single-variable DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 1:
            data = data[list(data.data_vars.keys())[0]]

        else:
            raise ValueError(f"The data parameter needs to be a raster cube in the form of an xarray DataArray or a single-variable xarray Dataset, received: \n{data}")

        assert isinstance(data, xr.DataArray)

    # Ensure raster cube has crs
    if data.rio.crs is None:
        data = data.rio.set_crs(DEFAULT_CRS)

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
                DEFAULT_CRS = (
                    geometries.get("crs", {})
                    .get("properties", {})
                    .get("name", DEFAULT_CRS)
                )
            else:
                DEFAULT_CRS = int(geometries.get("crs", {}))
            logger.info(f"CRS in geometries: {DEFAULT_CRS}.")

        if "type" in geometries and geometries["type"] == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=DEFAULT_CRS)
        elif "type" in geometries and geometries["type"] in ["Polygon"]:
            polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
            gdf = gpd.GeoDataFrame(geometry=[polygon])
            gdf.crs = DEFAULT_CRS

    # Convert xarray vector cube to GeoDataFrame
    if isinstance(geometries, xr.Dataset):
        if hasattr(geometries, "xvec"):
            gdf = geometries.xvec.to_geodataframe()

    if isinstance(geometries, gpd.GeoDataFrame):
        gdf = geometries

    # Reproject geometries to same crs as raster cube
    gdf = gdf.to_crs(data.rio.crs)

    # Convert to geopandas geometries Series
    geometries_series = gdf.geometry

    # Run xvec zonal stats with exactextract backend
    vec_cube = data.xvec.zonal_stats(
        geometries_series,
        x_coords=x_dim,
        y_coords=y_dim,
        method="exactextract",
        stats=reducer,
    )

    return vec_cube
