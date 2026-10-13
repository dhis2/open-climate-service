"""aggregate_spatial_classes — zonal statistics for a categorical raster (CLIM-895).

A categorical dataset holds class codes, so its zonal statistics are per class: how much of each
geometry each class covers, which class covers most of it, or how much of a weight (people,
usually) lives in each class. The result has a `class` dimension, one entry per declared class,
labelled by `class_label`.
"""

from typing import Any

import numpy as np
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

from open_climate_service.process import process

STATISTICS = ("total", "share", "majority")
CLASS_DIM = "class"


def _xy(raster: xr.DataArray) -> xr.DataArray:
    """The raster with its spatial axes named x and y, which exactextract needs."""
    from open_climate_service.data_manager.services.utils import get_x_y_dims

    x_dim, y_dim = get_x_y_dims(raster)
    if (x_dim, y_dim) != ("x", "y"):
        raster = raster.rename({x_dim: "x", y_dim: "y"}).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return raster


def _zonal(raster: xr.DataArray, frame: Any, stat: str) -> xr.DataArray:
    result: xr.DataArray = raster.load().xvec.zonal_stats(
        frame.geometry, x_coords="x", y_coords="y", method="exactextract", stats=stat
    )
    return result


def _codes_only(raster: xr.DataArray) -> xr.DataArray:
    """The raster as floats, with the nodata code and missing values as NaN."""
    from open_climate_service.shared.classes import NODATA_ATTR

    nodata = raster.attrs.get(NODATA_ATTR)
    values = raster.astype("float64")
    if nodata is not None:
        values = values.where(raster != nodata)
    masked: xr.DataArray = values.rio.write_nodata(np.nan)
    return masked


def _single_step(weights: xr.DataArray) -> xr.DataArray:
    """A weight raster as one 2-D grid: a population for one year, say."""
    other = [dim for dim in weights.dims if dim not in ("x", "y")]
    extra = {dim: weights.sizes[dim] for dim in other if weights.sizes[dim] != 1}
    if extra:
        raise ValueError(
            f"aggregate_spatial_classes: weights must be one grid, but have {extra}; select one step "
            "first, for example one year of population with filter_temporal"
        )
    return weights.squeeze([dim for dim in other], drop=True)


@process(
    summary="Aggregate a categorical raster over vector geometries, per class.",
    description=(
        "For a dataset whose template declares classes. `total` sums a weight raster, such as "
        "population, per class in each geometry: the class raster is first resampled onto the weight "
        "grid by nearest neighbour, since a weight is only meaningful on its own grid. `share` gives "
        "the fraction of each geometry's area in each class, and `majority` the class covering most "
        "of it. Pixels are weighted by their overlap with the geometry. The nodata code, if declared, "
        "is not a class and counts towards none."
    ),
    parameters={
        "data": {
            "description": "A categorical raster data cube: one whose dataset declares classes.",
            "schema": {
                "type": "object",
                "subtype": "datacube",
                "dimensions": [{"type": "spatial", "axis": ["x", "y"]}],
            },
        },
        "geometries": {
            "description": "A vector data cube or GeoJSON, as for aggregate_spatial_weighted.",
            "schema": [
                {"type": "object", "subtype": "datacube", "dimensions": [{"type": "geometry"}]},
                {"type": "object", "subtype": "geojson"},
            ],
        },
        "statistic": {
            "description": "`total` (needs `weights`), `share` or `majority`.",
            "schema": {"type": "string", "enum": list(STATISTICS)},
        },
        "weights": {
            "description": "For `total`: a raster to sum per class, such as population, on its own grid.",
            "schema": [
                {"type": "object", "subtype": "datacube"},
                {"type": "null"},
            ],
            "default": None,
            "optional": True,
        },
    },
)
def aggregate_spatial_classes(
    data: xr.Dataset | xr.DataArray,
    geometries: Any,
    statistic: str,
    weights: xr.Dataset | xr.DataArray | None = None,
) -> xr.DataArray:
    """Per-class zonal statistics of a categorical raster.

    Returns:
    -------
    VectorCube
        For `total` and `share`, one value per class and geometry, with the classes on `class`
        (their codes) and `class_label`; for `majority`, the majority class code per geometry.
        The non-spatial dimensions of the input are preserved, as for aggregate_spatial_weighted.
    """
    from open_climate_service.shared.classes import cube_classes, cube_name
    from open_climate_service.shared.provenance import record_features, record_spatial_aggregation
    from open_climate_service.shared.vectors import features_in_crs, single_raster, vector_result

    if statistic not in STATISTICS:
        raise ValueError(f"aggregate_spatial_classes: statistic must be one of {', '.join(STATISTICS)}")
    raster = single_raster(data)
    declared = cube_classes(raster)
    if declared is None:
        raise ValueError(
            f"aggregate_spatial_classes: '{cube_name(raster)}' declares no classes; it is not a categorical "
            "dataset. Declare `classes` in its template, or use aggregate_spatial_weighted for continuous data"
        )
    codes, labels = declared
    record_features(geometries)

    if statistic == "total":
        if weights is None:
            raise ValueError("aggregate_spatial_classes: statistic 'total' needs `weights`, such as population")
        weight = _xy(_single_step(single_raster(weights)))
        classes = _codes_only(_xy(raster))
        # Onto the weight grid: a weight, people in a 100 m cell say, belongs to the class of the
        # cell it is in, and exactextract weights only by an aligned grid.
        classes = classes.rio.reproject_match(weight, resampling=_nearest())
        classes = classes.assign_coords(x=weight.x, y=weight.y)
        frame = features_in_crs(geometries, weight.rio.crs)
        per_class = [_zonal(weight.where(classes == code), frame, "sum") for code in codes]
        # The shapes are in the weight grid's CRS; the cadence and the name are the class raster's.
        reference = raster.rio.write_crs(weight.rio.crs)
    else:
        classes = _codes_only(_xy(raster))
        frame = features_in_crs(geometries, classes.rio.crs)
        reference = raster
        if statistic == "majority":
            result = _zonal(classes, frame, "majority")
            record_spatial_aggregation("majority")
            return vector_result(result, raster, frame.index)
        valid = classes.notnull()
        per_class = [_zonal((classes == code).astype("float64").where(valid), frame, "mean") for code in codes]

    stacked = xr.concat(per_class, dim=xr.Variable(CLASS_DIM, codes))
    stacked = stacked.assign_coords(class_label=(CLASS_DIM, labels))
    record_spatial_aggregation(statistic)
    return vector_result(stacked, reference, frame.index)


def _nearest() -> Any:
    from rasterio.enums import Resampling

    return Resampling.nearest
