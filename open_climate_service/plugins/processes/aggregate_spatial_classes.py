"""aggregate_spatial_classes — zonal statistics per class for a categorical raster (CLIM-895).

A categorical dataset holds class codes, not measurements, so its zonal statistics are per class
rather than a mean or sum of the codes. Three reducers, all computed by exactextract, which
weights each cell by the share of it the geometry covers:

- `share`: the fraction of each geometry's area in each class.
- `majority`: the class covering most of each geometry.
- `total`: the sum of a weight raster, usually population, in each class.

`share` and `total` return one value per class and geometry, on a `class` dimension labelled by
`class_label`; `majority` returns one class code per geometry. A missing (NaN) cell is not a class
and counts towards none.
"""

from typing import Any

import numpy as np
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

from open_climate_service.process import process

REDUCERS = ("total", "share", "majority")
"""The per-class statistics: a weight per class, the area share per class, the majority class."""

CLASS_DIM = "class"
"""The result dimension holding the class codes, with their labels as the `class_label` coordinate."""


def _ensure_xy(raster: xr.DataArray) -> xr.DataArray:
    """The raster with its spatial axes named x and y, which exactextract needs.

    As in aggregate_spatial_weighted: a cube's axes may be x/y, lon/lat or longitude/latitude.
    """
    from open_climate_service.data_manager.services.utils import get_x_y_dims

    x_dim, y_dim = get_x_y_dims(raster)
    if (x_dim, y_dim) != ("x", "y"):
        raster = raster.rename({x_dim: "x", y_dim: "y"}).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return raster


def _apply_missing(raster: xr.DataArray) -> xr.DataArray:
    """The class raster as floats, with missing cells as NaN.

    exactextract skips NaN cells, so a cell that is no class counts towards no class, and the
    share of a class is of the geometry's area that has a class.
    """
    masked: xr.DataArray = raster.astype("float64").rio.write_nodata(np.nan)
    return masked


def _ensure_single_grid(weights: xr.DataArray) -> xr.DataArray:
    """Ensure a raster - with axes already named x and y - is as a single 2-D grid for a single time period.

    E.g. a class series is summed against a single weight grid; a weight series would leave it unclear
    which year of people belongs to which day of classes, so it is refused.
    """
    other = [dim for dim in weights.dims if dim not in ("x", "y")]
    extra = {dim: weights.sizes[dim] for dim in other if weights.sizes[dim] != 1}
    if extra:
        raise ValueError(
            f"aggregate_spatial_classes: weights must be one grid, but have {extra}; select one step "
            "first, for example one year of population with filter_temporal"
        )
    return weights.squeeze(other, drop=True)


def _read_chunks(classes: xr.DataArray, grid_cells: int) -> tuple[str | None, list[slice]]:
    """Slices of the longest non-spatial axis, each within `READ_CHUNK_BYTES` once in memory.

    The same bound as aggregate_spatial_weighted, so a daily series over a large country is read a
    slice at a time. The size of a step is that of the grid the statistic computes on,
    *grid_cells*: the class grid for `share` and `majority`, the weight grid for `total`, which
    each step of classes is resampled onto, as float64.
    """
    from open_climate_service.plugins.processes.aggregate_spatial_weighted import READ_CHUNK_BYTES

    other = [dim for dim in classes.dims if dim not in ("x", "y")]
    if not other:
        return None, [slice(None)]

    dim = str(max(other, key=lambda name: classes.sizes[name]))
    grids_per_step = classes.size // (classes.sizes[dim] * classes.sizes["x"] * classes.sizes["y"])
    step_bytes = 8 * grid_cells * grids_per_step
    per_read = max(1, READ_CHUNK_BYTES // max(1, step_bytes))
    steps = classes.sizes[dim]
    slices = [slice(start, min(steps, start + per_read)) for start in range(0, steps, per_read)]

    return dim, slices


def _zonal(raster: xr.DataArray, frame: Any, stat: str) -> xr.DataArray:
    """One exactextract statistic of *raster* per geometry in *frame*."""
    result: xr.DataArray = raster.xvec.zonal_stats(
        frame.geometry, x_coords="x", y_coords="y", method="exactextract", stats=stat
    )
    return result


def _per_class(stats: list[xr.DataArray], codes: list[int], labels: list[str]) -> xr.DataArray:
    """One statistic per class, stacked on the `class` dimension with its labels."""
    stacked = xr.concat(stats, dim=xr.Variable(CLASS_DIM, codes))
    return stacked.assign_coords(class_label=(CLASS_DIM, labels))


def _share(classes: xr.DataArray, frame: Any, codes: list[int], labels: list[str]) -> xr.DataArray:
    """The fraction of each geometry's classed area in each class.

    The coverage-weighted mean of a 0/1 indicator, "is this cell class c", is the share of the
    geometry's area in class c. Cells with no class stay NaN, so they are left out of the area.
    """
    has_class = classes.notnull()
    stats = [_zonal((classes == code).astype("float64").where(has_class), frame, "mean") for code in codes]
    return _per_class(stats, codes, labels)


def _majority(classes: xr.DataArray, frame: Any) -> xr.DataArray:
    """The class code covering most of each geometry."""
    return _zonal(classes, frame, "majority")


def _total(
    classes: xr.DataArray, weight: xr.DataArray, frame: Any, codes: list[int], labels: list[str]
) -> xr.DataArray:
    """The sum of *weight*, people say, within each geometry in each class.

    The classes are resampled onto the weight grid by nearest neighbour, not the other way round:
    a weight, the people in a 100 m cell, belongs to the class of the cell it lies in, and a
    count resampled onto a coarser grid would no longer add up. exactextract also needs the two
    grids aligned, hence the weight's coordinates on the result.
    """
    from rasterio.enums import Resampling

    on_grid = classes.rio.reproject_match(weight, resampling=Resampling.nearest).assign_coords(x=weight.x, y=weight.y)
    stats = [_zonal(weight.where(on_grid == code), frame, "sum") for code in codes]
    return _per_class(stats, codes, labels)


@process(
    summary="Aggregate a categorical raster over vector geometries, per class.",
    description=(
        "For a dataset whose template declares classes. `total` sums a weight raster, such as "
        "population, per class in each geometry: the class raster is first resampled onto the weight "
        "grid by nearest neighbour, since a weight is only meaningful on its own grid. `share` gives "
        "the fraction of each geometry's area in each class, and `majority` the class covering most "
        "of it. Pixels are weighted by their overlap with the geometry. A missing cell is not a class "
        "and counts towards none."
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
        "reducer": {
            "description": "`total` (needs `weights`), `share` or `majority`.",
            "schema": {"type": "string", "enum": list(REDUCERS)},
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
    reducer: str,
    weights: xr.Dataset | xr.DataArray | None = None,
) -> xr.DataArray:
    """Per-class zonal statistics of a categorical raster.

    Parameters
    ----------
    data
        Categorical raster data cube (xr.DataArray or single-variable xr.Dataset) carrying its
        classes as CF `flag_values` and `flag_meanings`, as load_collection stamps them.
    geometries
        Vector geometries over which to aggregate, as for aggregate_spatial_weighted.
    reducer
        `total`, `share` or `majority`.
    weights
        For `total` only: one grid to sum per class, such as one year of population.

    Returns:
    -------
    VectorCube
        For `total` and `share`, one value per class and geometry, with the classes on `class`
        (their codes) and `class_label`; for `majority`, the majority class code per geometry.
        The non-spatial dimensions of the input are preserved, as for aggregate_spatial_weighted.
    """
    from open_climate_service.shared.classes import cube_classes
    from open_climate_service.shared.provenance import record_features, record_spatial_aggregation
    from open_climate_service.shared.vectors import features_in_crs, single_raster, vector_result

    if reducer not in REDUCERS:
        raise ValueError(f"aggregate_spatial_classes: reducer must be one of {', '.join(REDUCERS)}")

    # Preprocess the geometries
    record_features(geometries)

    # Preprocess the categorical raster
    raster = single_raster(data)
    raster = _apply_missing(_ensure_xy(raster))

    # Extract the categorical classification
    classification = cube_classes(raster)
    if classification is None:
        raise ValueError(
            f"aggregate_spatial_classes: '{raster.name or 'the input'}' declares no classes; it is not a categorical "
            "dataset. Declare `classes` in its template, or use aggregate_spatial_weighted for continuous data"
        )
    codes, labels = classification

    # Prepare data for the requested reducer
    if reducer == "total":
        if weights is None:
            raise ValueError("aggregate_spatial_classes: reducer 'total' needs `weights`, such as population")

        # Axes renamed before the one-grid check, so a weight on lon/lat is one grid, not a series.
        weight = _ensure_single_grid(_ensure_xy(single_raster(weights))).load()

        # The sums are on the weight grid, so the geometries are put in its CRS.
        frame = features_in_crs(geometries, weight.rio.crs)
        reference = raster.rio.write_crs(weight.rio.crs)
        grid_cells = weight.size

        def compute(part: xr.DataArray) -> xr.DataArray:
            return _total(part, weight, frame, codes, labels)

    else:
        frame = features_in_crs(geometries, raster.rio.crs)
        reference = raster
        grid_cells = raster.sizes["x"] * raster.sizes["y"]

        if reducer == "share":

            def compute(part: xr.DataArray) -> xr.DataArray:
                return _share(part, frame, codes, labels)

        elif reducer == "majority":

            def compute(part: xr.DataArray) -> xr.DataArray:
                return _majority(part, frame)

    # Read and aggregate a bounded slice of the series at a time and concatenate the results
    dim, reads = _read_chunks(raster, grid_cells)
    parts = [compute((raster if dim is None else raster.isel({dim: read})).load()) for read in reads]
    result: xr.DataArray = parts[0] if len(parts) == 1 else xr.concat(parts, dim=str(dim))

    # Record aggregation method and return the results
    record_spatial_aggregation(reducer)
    return vector_result(result, reference, frame.index)
