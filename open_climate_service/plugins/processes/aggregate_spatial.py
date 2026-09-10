"""aggregate_spatial — zonal statistics plugin process."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr

from open_climate_service.process import process
from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD


def _parse_geometries(geometries: Any) -> tuple[list[Any], list[str]]:
    """Extract Shapely geometries and stable string labels from GeoJSON input.

    Labels use the feature ``id`` when present, otherwise a sequential integer.
    """
    from shapely.geometry import shape

    if isinstance(geometries, dict):
        gtype = geometries.get("type", "")
        if gtype == "FeatureCollection":
            features = geometries.get("features", [])
            geoms = [shape(f["geometry"]) for f in features]
            labels = [str(f.get("id", i)) for i, f in enumerate(features)]
            return geoms, labels
        if gtype == "Feature":
            return [shape(geometries["geometry"])], [str(geometries.get("id", 0))]
        return [shape(geometries)], ["0"]
    items = [shape(g) if isinstance(g, dict) else g for g in geometries]
    return items, [str(i) for i in range(len(items))]


def _find_dim(data: xr.Dataset | xr.DataArray, candidates: list[str]) -> str | None:
    dims = data.dims if isinstance(data, xr.DataArray) else set(data.dims)
    for c in candidates:
        if c in dims:
            return c
    return None


def _make_reducer_caller(reducer: Callable, context: Any) -> Callable[[np.ndarray], float]:
    """Return a function that applies the reducer, forwarding ``context`` when supported.

    openEO reducers may or may not accept a ``context`` keyword; we inspect the
    signature once so context-aware reducers receive it without breaking the
    common array-only reducers (mean, median, ...).
    """
    import inspect

    pass_context = False
    if context is not None:
        try:
            params = inspect.signature(reducer).parameters
            pass_context = "context" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            pass_context = False

    def _call(pixels: np.ndarray) -> float:
        if not pixels.size:
            return float("nan")
        return float(reducer(data=pixels, context=context) if pass_context else reducer(data=pixels))

    return _call


def _dataset_reduce_spatial(
    ds: xr.Dataset,
    mask: np.ndarray,
    reducer: Callable,
    y_dim: str,
    x_dim: str,
    context: Any = None,
) -> xr.Dataset:
    """Apply spatial mask and reducer to each variable in a Dataset.

    Reduces only over the spatial dimensions while preserving any other dimensions
    (e.g. time or ``merge_cubes``' synthetic ``__cubes__`` dimension).

    Pixels are selected by boolean-indexing the raw arrays with ``mask`` directly
    rather than materialising ``ds.where(mask)`` — the latter allocates a full
    NaN-filled copy of the cube just to throw most of it away.
    """
    reduce = _make_reducer_caller(reducer, context)

    def _drop_nan(pixels: np.ndarray) -> np.ndarray:
        # Only floating arrays can carry NaN; np.isnan raises on integer dtypes.
        return pixels[~np.isnan(pixels)] if np.issubdtype(pixels.dtype, np.floating) else pixels

    mask_flat = mask.ravel()
    result_vars: dict[str, Any] = {}
    for var in ds.data_vars:
        vname = str(var)
        da = ds[vname]
        remaining_dims = [d for d in da.dims if d not in {y_dim, x_dim}]
        arr = da.transpose(*remaining_dims, y_dim, x_dim).values
        arr = arr.reshape((-1, mask_flat.size))
        reduced = [reduce(_drop_nan(pixels[mask_flat])) for pixels in arr]
        if remaining_dims:
            coords = {d: da.coords[d].values for d in remaining_dims if d in da.coords}
            shape = tuple(int(da.sizes[d]) for d in remaining_dims)
            result_vars[vname] = xr.DataArray(np.array(reduced).reshape(shape), coords=coords, dims=remaining_dims)
        else:
            result_vars[vname] = xr.DataArray(reduced[0])
    return xr.Dataset(result_vars)


@process(
    summary="Aggregate spatial data within geometries",
    parameters={
        "data": {"description": "A raster data cube."},
        "geometries": {"description": "GeoJSON FeatureCollection, Feature, or geometry."},
        "reducer": {"description": "A reducer to apply on the pixel values."},
        "target_dimension": {"description": "Name for the new geometry dimension (default: 'geometry')."},
        "context": {"description": "Optional context passed to the reducer."},
    },
)
def aggregate_spatial(
    data: Any,
    geometries: Any,
    reducer: Callable,
    target_dimension: str | None = None,
    context: Any = None,
) -> xr.Dataset:
    """Aggregate raster values within each polygon using the supplied reducer."""
    import rasterio.features
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    geom_shapes, geom_labels = _parse_geometries(geometries)
    if not geom_shapes:
        raise ValueError("aggregate_spatial: geometries contains no shapes")

    # Promote DataArray to Dataset so we can handle both uniformly
    if isinstance(data, xr.DataArray):
        name = data.name or "data"
        data = data.to_dataset(name=name)

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial: cannot identify x/y dimensions in {list(data.dims)}")

    x_coords = data[x_dim].values.astype(float)
    y_coords = data[y_dim].values.astype(float)
    height = len(y_coords)
    width = len(x_coords)

    dx = float(abs(x_coords[1] - x_coords[0])) if width > 1 else 1.0
    dy = float(abs(y_coords[1] - y_coords[0])) if height > 1 else 1.0
    xmin = float(x_coords.min()) - dx / 2
    xmax = float(x_coords.max()) + dx / 2
    ymin = float(y_coords.min()) - dy / 2
    ymax = float(y_coords.max()) + dy / 2
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    geom_dim = target_dimension or "geometry"
    results: list[xr.Dataset] = []

    for geom in geom_shapes:
        mask = rasterio.features.geometry_mask(
            [mapping(geom)],
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
        # rasterio builds the mask top-row first (y descending); flip when y is ascending
        if height > 1 and float(y_coords[1]) > float(y_coords[0]):
            mask = mask[::-1]

        geom_ds = _dataset_reduce_spatial(data, mask, reducer, y_dim, x_dim, context)
        results.append(geom_ds)

    combined = xr.concat(results, dim=geom_dim)
    combined[geom_dim] = geom_labels
    # Carry the shapes as well as the labels, so the result is a vector datacube rather than a
    # table that has forgotten where it came from — `save_result(format="GeoParquet")` writes
    # them straight out, and a consumer can map the result without joining back to a boundary
    # file. See CLIM-836.
    #
    # A companion coordinate rather than replacing the labels on `geom_dim`: the label is the
    # feature id, and the DHIS2 and CHAP exports key their location column on it
    # (`location_field` defaults to "geometry"). Putting shapes there would break both.
    #
    # WKT strings rather than shapely objects, which is what openEO's vector datacube model uses
    # (xvec puts shapely geometries in an object-dtype coordinate with a CRS-carrying
    # GeometryIndex). This process does not build an xvec cube in the first place — the geometry
    # dimension carries string feature ids, because that is the label the exports need — so the
    # carrier is chosen for robustness instead: a string coordinate is inert on every path the
    # cube can take, whereas an object-dtype one makes `to_zarr` fail outright, and the drop that
    # currently prevents that lives in a single place in the job writer.
    #
    # xvec's own encodings are not a better option here: `encode_wkb` round-trips through Zarr as
    # null-terminated bytes and comes back truncated, and `encode_cf` restructures the cube into
    # several CF geometry variables — right for an archival file, wrong for a carrier that only
    # has to survive as far as the vector writer.
    # Cost worth knowing: numpy stores this as fixed-width unicode (`<U{longest}`) at 4 bytes per
    # character, sized by the *longest* WKT, and it is attached on every aggregation regardless of
    # the output format. For simple boundaries that is nothing; for detailed ones it is roughly
    # 4 x longest-WKT x n_features. Hex-encoded WKB in a bytes array would be ~4x smaller and is
    # still null-free, so still Zarr-safe -- worth doing if a real boundary set makes this hurt.
    combined = combined.assign_coords({GEOMETRY_WKT_COORD: (geom_dim, [geom.wkt for geom in geom_shapes])})
    return combined


_REDUCE_METHODS: dict[str, Callable[..., Any]] = {
    "mean": np.mean,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "median": np.median,
}


@process(
    summary="Reduce pixel values by a named method",
    parameters={
        "data": {"description": "The array of values to reduce."},
        "method": {"description": "Reduction method: mean (default), sum, min, max or median."},
    },
)
def reduce_by_method(data: Any, method: str = "mean") -> float:
    """Reduce an array of values by a named method.

    A spec-compliant alternative to parameterising a reducer's ``process_id``: workflows
    pass ``method`` as an ordinary string argument (mean/sum/min/max/median) while
    ``process_id`` stays the literal ``"reduce_by_method"``, so standard openEO tooling
    can validate the graph. Used as the reducer in the ``aggregate_*`` workflows, where
    it receives the (already NaN-dropped, 1-D) pixel values within each geometry.
    """
    if method not in _REDUCE_METHODS:
        raise ValueError(f"Unknown reduce method '{method}'; expected one of {sorted(_REDUCE_METHODS)}")
    arr = np.asarray(data, dtype="float64").ravel()
    if arr.size == 0:
        return float("nan")
    return float(_REDUCE_METHODS[method](arr))
