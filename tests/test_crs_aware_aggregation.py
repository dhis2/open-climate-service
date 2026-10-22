"""Aggregation over projected rasters: the geometries move into the raster's CRS, never the
raster into theirs, and geometries that all miss the raster get no value and a warning.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box

from open_climate_service.openeo import execution
from open_climate_service.plugins.processes.aggregate_spatial_weighted import aggregate_spatial_weighted


def _lonlat_grid() -> xr.DataArray:
    """0.05 degree cells over Nepal, each holding its latitude."""
    lon = np.arange(80.0, 88.0, 0.05) + 0.025
    lat = np.arange(30.5, 26.0, -0.05) - 0.025
    values = np.broadcast_to(lat[:, None], (lat.size, lon.size)).astype("float32")
    return xr.DataArray(values, dims=("y", "x"), coords={"y": lat, "x": lon}, name="v").rio.write_crs("EPSG:4326")


_DISTRICTS = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "N", "properties": {}, "geometry": box(83, 28.5, 86, 29.5).__geo_interface__},
        {"type": "Feature", "id": "S", "properties": {}, "geometry": box(83, 26.5, 86, 27.5).__geo_interface__},
    ],
}


def _builtin(raster: xr.DataArray, geometries: Any) -> xr.DataArray:
    def mean(data: Any, axis: Any = None, **_: Any) -> Any:
        return np.nanmean(data, axis=axis)

    aggregate = execution._build_process_registry()["aggregate_spatial"].implementation
    result: xr.DataArray = aggregate(data=raster, geometries=geometries, reducer=mean)
    return result


def _weighted(raster: xr.DataArray, geometries: Any) -> xr.DataArray:
    return aggregate_spatial_weighted(raster, geometries, "mean")


def _by_id(result: xr.DataArray) -> dict[str, float]:
    return dict(zip(result.feature_id.values.tolist(), result.values.ravel().tolist(), strict=True))


@pytest.mark.parametrize("aggregate", [_builtin, _weighted])
@pytest.mark.parametrize("crs", ["EPSG:3857", "ESRI:54009"])  # Web Mercator, Mollweide
def test_a_projected_raster_gives_the_values_of_its_lonlat_original(aggregate: Any, crs: str) -> None:
    lonlat = _lonlat_grid()
    projected = lonlat.rio.reproject(crs, resampling=Resampling.nearest, nodata=np.nan)

    expected = _by_id(aggregate(lonlat, _DISTRICTS))
    assert expected == pytest.approx({"N": 29.0, "S": 27.0}, abs=0.05)
    assert _by_id(aggregate(projected, _DISTRICTS)) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("aggregate", [_builtin, _weighted])
def test_a_raster_in_metres_read_as_degrees_warns(aggregate: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Ingestion records a CRS on every store, so this takes a cube that lost it. Read as WGS 84,
    its metres overlap no district: each gets no value, and the server log says why."""
    untagged = _lonlat_grid().rio.reproject("EPSG:3857", nodata=np.nan).drop_vars("spatial_ref")
    assert untagged.rio.crs is None

    with caplog.at_level("WARNING", logger="open_climate_service.shared.vectors"):
        result = aggregate(untagged, _DISTRICTS)
    assert all(np.isnan(value) for value in _by_id(result).values())
    assert "None of the 2 geometries overlaps the raster" in caplog.text


@pytest.mark.parametrize("aggregate", [_builtin, _weighted])
def test_geometries_that_miss_the_raster_warn_naming_both_crses(
    aggregate: Any, caplog: pytest.LogCaptureFixture
) -> None:
    projected = _lonlat_grid().rio.reproject("EPSG:3857", nodata=np.nan)
    elsewhere = {"type": "Feature", "id": "far", "properties": {}, "geometry": box(10, 50, 11, 51).__geo_interface__}

    with caplog.at_level("WARNING", logger="open_climate_service.shared.vectors"):
        result = aggregate(projected, elsewhere)
    assert np.isnan(_by_id(result)["far"])
    assert re.search(r"None of the 1 geometries overlaps the raster.*EPSG:4326.*EPSG:3857", caplog.text)
