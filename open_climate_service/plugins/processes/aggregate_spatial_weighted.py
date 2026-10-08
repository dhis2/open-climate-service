"""aggregate_spatial_weighted — weighted zonal statistics plugin process."""

from typing import Any, Callable, cast

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
import xvec  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

from open_climate_service.process import process

REDUCERS = ("mean", "sum", "min", "max", "median")
"""The statistics the weighted aggregation offers: the methods a DHIS2 export can declare."""

READ_BLOCK_BYTES = 512 * 2**20
"""The most raster data read into memory at once, the same bound as the reference implementation.

exactextract reads a cube one step of its non-spatial axis at a time. Handed a lazy cube, each of
those reads goes back to the store and decompresses whole chunks for one step, so a store chunked
61 days deep had each chunk decompressed 61 times: a year of daily seNorge over Norway's 357
municipalities took 300 s. Reading a block of steps at once, aligned to the store's chunks, reads
each chunk once (20 s), and the bound keeps a long series from being loaded whole.
"""


def _blocks(raster: xr.DataArray) -> tuple[str | None, list[slice]]:
    """Slices of the longest non-spatial axis, each within `READ_BLOCK_BYTES`, along the chunks.

    Consecutive chunks are grouped while they fit, so a block never splits a chunk unless that
    chunk alone is over the bound; then it is split into steps that fit. A 2-D raster is one block.
    """
    other = [dim for dim in raster.dims if dim not in ("x", "y")]
    if not other:
        return None, [slice(None)]
    dim = str(max(other, key=lambda name: raster.sizes[name]))
    step_bytes = raster.dtype.itemsize * raster.size // raster.sizes[dim]
    per_block = max(1, READ_BLOCK_BYTES // max(1, step_bytes))
    chunks = raster.chunksizes.get(dim) if raster.chunks else None
    sizes = list(chunks) if chunks else [raster.sizes[dim]]

    blocks: list[slice] = []
    start = end = 0
    for size in sizes:
        if end > start and end - start + size > per_block:
            blocks.append(slice(start, end))
            start = end
        end += size
        while end - start > per_block:
            blocks.append(slice(start, start + per_block))
            start += per_block
    if end > start:
        blocks.append(slice(start, end))
    return dim, blocks


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
        The statistic to calculate, by name: one of mean, sum, min, max or median. A reducer
        process is refused, since exactextract weights by overlap only for named statistics.

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
    # The cube's own spatial axes may be x/y, lon/lat or longitude/latitude. exactextract finds
    # only x and y, so they are renamed; the aggregation consumes them, so no name reaches the result.
    from open_climate_service.data_manager.services.utils import get_x_y_dims

    x_dim, y_dim = get_x_y_dims(raster)
    if (x_dim, y_dim) != ("x", "y"):
        raster = raster.rename({x_dim: "x", y_dim: "y"}).rio.set_spatial_dims(x_dim="x", y_dim="y")
    dim, blocks = _blocks(raster)
    parts = [
        (raster if dim is None else raster.isel({dim: block}))
        .load()
        .xvec.zonal_stats(frame.geometry, x_coords="x", y_coords="y", method="exactextract", stats=reducer)
        for block in blocks
    ]
    vec_cube: xr.DataArray = parts[0] if len(parts) == 1 else cast(xr.DataArray, xr.concat(parts, dim=str(dim)))
    # The named method, so a DHIS2 export can check the aggregation it declares against this one.
    record_spatial_aggregation(reducer)
    return vector_result(vec_cube, raster, frame.index)
