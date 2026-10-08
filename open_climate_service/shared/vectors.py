"""Shared conventions for vector datacubes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr

_Cube = TypeVar("_Cube", "xr.Dataset", "xr.DataArray")

GEOMETRY_WKT_COORD = "geometry_wkt"
"""Coordinate containing each feature's geometry as WKT.

The vector cube's geometry dimension carries feature labels/IDs; this coordinate
stores the corresponding geometry without replacing those labels.
"""

FEATURE_ID_COORD = "feature_id"
"""Companion coordinate on a vector cube's geometry dimension, holding each feature's id.

The id is what a DHIS2 or CHAP export keys its location on. openEO labels the geometry dimension
with the geometries themselves and leaves the feature id to the backend
(openeo-processes#466); this is where OCS keeps it, so it survives an aggregation whatever that
aggregation puts in the labels. A `location_field` or `org_unit_field` of `geometry` reads it
(:func:`feature_id_field`).
"""

GEOMETRY_FIELD = "geometry"
"""The default location field of the exports, and the default geometry dimension name."""


def geometries_to_frame(geometries: Any) -> gpd.GeoDataFrame:
    """The features of an aggregation's `geometries` argument, indexed by feature id.

    Accepts what openEO allows there: GeoJSON (a FeatureCollection, a Feature or a bare
    geometry), a GeoDataFrame or an xvec vector cube, plus a plain sequence of shapes or GeoJSON
    geometries. The id is the GeoJSON Feature's `id`, the GeoDataFrame's index, or an xvec cube's
    `feature_id` coordinate; a feature without one gets its position, as a string. GeoJSON is
    WGS 84 by RFC 7946; the other inputs keep the CRS they declare.
    """
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import shape

    if isinstance(geometries, gpd.GeoDataFrame):
        return _indexed(geometries.geometry.tolist(), [str(label) for label in geometries.index], geometries.crs)
    if isinstance(geometries, (xr.Dataset, xr.DataArray)):
        return _frame_from_vector_cube(geometries)
    if isinstance(geometries, dict):
        kind = geometries.get("type", "")
        if kind == "FeatureCollection":
            features = geometries.get("features", [])
            shapes = [shape(feature["geometry"]) for feature in features]
            ids = [_feature_id(feature, position) for position, feature in enumerate(features)]
        elif kind == "Feature":
            shapes, ids = [shape(geometries["geometry"])], [_feature_id(geometries, 0)]
        else:
            shapes, ids = [shape(geometries)], ["0"]
        return _indexed(shapes, ids, "EPSG:4326")
    if isinstance(geometries, Sequence) and not isinstance(geometries, (str, bytes)):
        shapes = [shape(item) if isinstance(item, dict) else item for item in geometries]
        return _indexed(shapes, [str(position) for position in range(len(shapes))], "EPSG:4326")
    raise ValueError(f"geometries must be GeoJSON, a GeoDataFrame or a vector cube, got {type(geometries).__name__}")


def _feature_id(feature: dict[str, Any], position: int) -> str:
    """A GeoJSON Feature's `id`, or its position when it has none: a null `id` is no id."""
    value = feature.get("id")
    return str(position) if value is None else str(value)


def _frame_from_vector_cube(cube: xr.Dataset | xr.DataArray) -> gpd.GeoDataFrame:
    """The features of an xvec cube: the shapes on its geometry dimension, ids beside them."""
    for name in cube.dims:
        values = cube[name].values
        if len(values) and all(hasattr(value, "geom_type") for value in values):
            index = cube.xindexes.get(name)
            if FEATURE_ID_COORD in cube.coords and cube[FEATURE_ID_COORD].dims == (name,):
                ids = [str(value) for value in cube[FEATURE_ID_COORD].values]
            else:
                ids = [str(position) for position in range(len(values))]
            return _indexed(list(values), ids, getattr(index, "crs", None))
    raise ValueError("geometries is a data cube without a dimension of geometries")


def _indexed(shapes: list[Any], ids: list[str], crs: Any) -> gpd.GeoDataFrame:
    import geopandas as gpd
    import pandas as pd

    return gpd.GeoDataFrame(geometry=shapes, index=pd.Index(ids, name=FEATURE_ID_COORD), crs=crs)


def attach_feature_ids(result: _Cube, ids: Iterable[Any], dim: str) -> _Cube:
    """*result* with each feature's id as the `feature_id` coordinate on its geometry dimension."""
    return result.assign_coords({FEATURE_ID_COORD: (dim, [str(value) for value in ids])})


def feature_id_field(columns: Iterable[Any], requested: str) -> str:
    """The column an export reads feature ids from, when it asked for *requested*.

    `geometry` is the exports' default location field, and means the feature's identity. When the
    result carries `feature_id`, that is where the identity is, whatever the geometry labels
    hold: an aggregation that labels its dimension with shapes (as openEO does) still exports
    org-unit ids. Any other field name is taken as given.
    """
    if requested == GEOMETRY_FIELD and FEATURE_ID_COORD in {str(column) for column in columns}:
        return FEATURE_ID_COORD
    return requested


def raster_and_features(data: Any, geometries: Any) -> tuple[xr.DataArray, gpd.GeoDataFrame]:
    """The inputs of a spatial aggregation, ready for xvec: one named raster, features in its CRS.

    A single-variable Dataset becomes its variable. A raster without a CRS is taken as WGS 84,
    the CRS of GeoJSON, which is what a grid without one has always been compared against. The
    features are indexed by feature id (:func:`geometries_to_frame`), and a frame without a CRS
    is WGS 84 too.
    """
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
    import xarray as xr

    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(f"spatial aggregation needs a single-variable raster, got {list(data.data_vars)}")
        data = data[next(iter(data.data_vars))]
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326")
    frame = geometries_to_frame(geometries)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return data, frame.to_crs(data.rio.crs)


def vector_result(
    result: xr.DataArray, data: xr.DataArray, ids: Iterable[Any], dim: str = GEOMETRY_FIELD
) -> xr.Dataset:
    """An aggregation's output in OCS's vector cube form: named, labelled by feature id.

    xvec returns an unnamed DataArray whose geometry dimension holds shapely objects, which Zarr
    and NetCDF cannot encode. This relabels the dimension with the feature ids, keeps the shapes
    as WKT in `geometry_wkt` and the ids in `feature_id`, names the result after the input
    variable so tabular exports get a value column, and keeps the input's cadence, since a
    spatial aggregation changes nothing about time.
    """
    from open_climate_service.shared.time import cadence_of, stamp_cadence

    labels = [str(value) for value in ids]
    wkt = [shape.wkt for shape in result[dim].values]
    plain = result.drop_vars(dim).assign_coords({dim: labels, GEOMETRY_WKT_COORD: (dim, wkt)})
    named = attach_feature_ids(plain, labels, dim).to_dataset(name=str(data.name or "data"))
    stamp_cadence(named, cadence_of(data))
    return named
