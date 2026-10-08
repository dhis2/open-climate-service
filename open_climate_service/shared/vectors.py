"""Shared conventions for vector datacubes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr

_Cube = TypeVar("_Cube", "xr.Dataset", "xr.DataArray")

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


def explicit_feature_collection(geometries: Any) -> Any:
    """*geometries* as a GeoJSON FeatureCollection when it carries explicit feature ids.

    A GeoDataFrame with an index of its own (not pandas' default 0, 1, 2...) or a vector cube
    with `feature_id` names its features as clearly as GeoJSON `id`s do; this lets the GeoJSON
    validators and fingerprints apply to them. Anything else, GeoJSON included, comes back as
    it is, so positional ids are still refused where ids are required.
    """
    import geopandas as gpd
    import pandas as pd
    import xarray as xr
    from shapely.geometry import mapping

    frame: gpd.GeoDataFrame | None = None
    if isinstance(geometries, gpd.GeoDataFrame) and not isinstance(geometries.index, pd.RangeIndex):
        frame = geometries
    elif isinstance(geometries, (xr.Dataset, xr.DataArray)) and FEATURE_ID_COORD in geometries.coords:
        frame = _frame_from_vector_cube(geometries)
    if frame is None:
        return geometries
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": str(label), "geometry": mapping(shape), "properties": {}}
            for label, shape in zip(frame.index, frame.geometry, strict=True)
        ],
    }


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


def feature_id_field(frame: Any, requested: str) -> str:
    """The column of *frame* an export reads feature ids from, when it asked for *requested*.

    `geometry` is the exports' default location field, and means the feature's identity, as does
    any column of shapes: the features' dimension after an aggregation, whatever openEO's
    `target_dimension` named it. When the result carries `feature_id`, that is where the identity
    is. Any other field name is taken as given.
    """
    columns = {str(column) for column in frame.columns}
    if FEATURE_ID_COORD not in columns:
        return requested
    if requested == GEOMETRY_FIELD:
        return FEATURE_ID_COORD
    if requested in columns:
        values = frame[requested]
        if len(values) and all(hasattr(value, "geom_type") for value in values):
            return FEATURE_ID_COORD
    return requested


def single_raster(data: Any) -> xr.DataArray:
    """*data* as one raster DataArray with a CRS, for a process that works on a single variable.

    A single-variable Dataset becomes its variable; more than one is refused. A raster without
    a CRS is taken as WGS 84, the CRS of GeoJSON, which is what a grid without one has always
    been compared against.
    """
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
    import xarray as xr

    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(f"expected a single-variable raster, got {list(data.data_vars)}")
        data = data[next(iter(data.data_vars))]
    raster: xr.DataArray = data
    if raster.rio.crs is None:
        raster = raster.rio.write_crs("EPSG:4326")
    return raster


def features_in_crs(geometries: Any, crs: Any) -> gpd.GeoDataFrame:
    """The features of a `geometries` argument, indexed by feature id, reprojected to *crs*.

    See :func:`geometries_to_frame` for what is accepted. Features without a CRS are WGS 84.
    """
    frame = geometries_to_frame(geometries)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.to_crs(crs)


def vector_result(
    result: xr.DataArray, data: xr.DataArray, ids: Iterable[Any], dim: str = GEOMETRY_FIELD
) -> xr.DataArray:
    """An aggregation's output as a vector cube: the shapes on *dim*, each feature's id beside them.

    *result* is xvec's, with the shapes on `geometry` under xvec's geometry index (which carries
    their CRS), as openEO describes a vector cube. Three things are added: the dimension takes
    *dim*'s name (openEO's `target_dimension`), the feature ids ride along as `feature_id`, which
    the DHIS2 and CHAP exports key on, and the result is named after the input variable, so a
    tabular export has a value column. The input's cadence is kept, since a spatial aggregation
    changes nothing about time. Writers encode the shapes for formats that cannot hold them
    (:func:`encode_vector_cube`).
    """
    from open_climate_service.shared.time import cadence_of, stamp_cadence

    if dim != GEOMETRY_FIELD:
        result = result.rename({GEOMETRY_FIELD: dim})
    index = result.xindexes.get(dim)
    if type(index).__name__ != "GeometryIndex" or getattr(index, "crs", None) is None:
        # The shapes are in the raster's CRS, which the features were reprojected to. openEO's
        # built-in hands xvec a plain list of shapes, so its index has no CRS to carry.
        import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

        result = result.xvec.set_geom_indexes(dim, crs=data.rio.crs)
    # Scalar coordinates describe the raster's grid (its grid mapping, `spatial_ref`), not the
    # features; left on, a table gets a column per grid mapping and the exports, which find
    # their value column by elimination, take it for a second value.
    result = result.drop_vars([name for name, coord in result.coords.items() if coord.ndim == 0])
    named = attach_feature_ids(result, ids, dim).rename(str(data.name or "data"))
    stamp_cadence(named, cadence_of(data))
    return named


def vector_dim(cube: Any) -> str | None:
    """The dimension a vector cube's features live on, or None for a raster cube.

    The dimension under xvec's geometry index, whatever it is called (openEO's
    `target_dimension` can rename it); otherwise one whose labels are all shapes; otherwise a
    dimension named `geometry`, so a cube that says it is vector but carries no usable shapes
    is refused as such rather than written as a raster.
    """
    indexes = getattr(cube, "xindexes", {})
    dims = getattr(cube, "dims", ())
    for name in dims:
        if type(indexes.get(name)).__name__ == "GeometryIndex":
            return str(name)
    for name in dims:
        if holds_shapes(cube, str(name)):
            return str(name)
    return GEOMETRY_FIELD if GEOMETRY_FIELD in dims else None


def holds_shapes(cube: Any, dim: str) -> bool:
    """Whether *dim*'s labels are all geometries, as on an xvec cube."""
    if dim not in getattr(cube, "coords", {}):
        return False
    values = cube[dim].values
    return bool(len(values)) and all(hasattr(value, "geom_type") for value in values)


def encode_vector_cube(ds: xr.Dataset) -> xr.Dataset:
    """*ds* with its shapes encoded as CF geometry, for writers that cannot hold shapely objects.

    Zarr and NetCDF store arrays of numbers and strings, not geometry objects. xvec's CF encoding
    turns the shapes into the CF conventions' geometry container, which xvec, and any CF-aware
    reader, decodes back; the feature ids stay as `feature_id`. A cube with no geometry index is
    returned unchanged.
    """
    dim = vector_dim(ds)
    if dim is None or type(ds.xindexes.get(dim)).__name__ != "GeometryIndex":
        return ds
    import xvec  # noqa: F401  # pyright: ignore[reportUnusedImport]

    encoded: xr.Dataset = ds.xvec.encode_cf()
    return encoded
