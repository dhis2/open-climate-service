"""aggregate_spatial — zonal statistics plugin process.

Semantics deliberately diverge from the openEO specification (CLIM-785):

* **Polygons are area-weighted.** Each cell counts by the fraction of it the polygon covers,
  computed by exactextract. The specification's pixel-centre rule returns NaN for any zone
  smaller than a cell, and misplaces every zone edge by up to half a cell.
* **Points are interpolated.** A point samples the surface at its location (bilinear by
  default) rather than taking whichever cell contains it.
* **Categorical data is never averaged.** A cube whose dataset declares
  ``ingestion.resampling`` of ``mode``, ``max`` or ``nearest`` aggregates polygons by area-weighted majority and
  samples points from the nearest cell.

The weighted path applies to the reducers it recognises (mean, sum, min, max, median, and the
categorical majority and fractions). Any other reducer is an arbitrary process graph that
cannot be weighted, so it falls back to the specification's pixel-centre selection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.process import process
from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD, RESAMPLING_ATTR

logger = logging.getLogger(__name__)

_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
_POINT_TYPES = frozenset({"Point"})

_CATEGORICAL_RESAMPLING = frozenset({"mode", "max", "nearest"})

# openEO `resample_spatial` vocabulary, mapped to xarray interpolation methods.
_POINT_METHODS = {"near": "nearest", "bilinear": "linear", "cubic": "cubic"}

# Numeric reducers averaging class codes would be meaningless; on categorical data they become
# a majority. min and max keep their meaning (max over a presence mask is "any present").
_CATEGORICAL_REPLACED = frozenset({"mean", "median", "sum"})

_WEIGHTED_METHODS = frozenset({"mean", "sum", "min", "max", "median", "majority", "fractions"})

# Coverage below this is floating-point residue at a polygon edge, not a covered cell.
_MIN_COVERAGE = 1e-9

_VarResult = tuple[np.ndarray, list[Hashable], dict[Hashable, Any]]
"""One variable's result: an array led by the geometry axis, and its other dims and coords."""

FRACTIONS_DIM = "class"
"""Dimension a ``fractions`` aggregation adds: one entry per class value found in the zones."""


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------


def _parse_geometries(geometries: Any) -> tuple[list[Any], list[str]]:
    """Extract Shapely geometries and stable string labels from GeoJSON input.

    Labels use the feature ``id`` when present, otherwise a sequential integer.
    """
    from shapely.geometry import shape

    from open_climate_service.shared.provenance import record_features

    record_features(geometries)

    if isinstance(geometries, dict):
        gtype = geometries.get("type", "")
        if gtype == "FeatureCollection":
            features = geometries.get("features", [])
            geoms = [shape(f["geometry"]) for f in features]
            labels = [str(f.get("id", i)) for i, f in enumerate(features)]
        elif gtype == "Feature":
            geoms, labels = [shape(geometries["geometry"])], [str(geometries.get("id", 0))]
        else:
            geoms, labels = [shape(geometries)], ["0"]
    else:
        geoms = [shape(g) if isinstance(g, dict) else g for g in geometries]
        labels = [str(i) for i in range(len(geoms))]
    _require_supported_geometry_types(geoms, labels)
    return geoms, labels


def _require_supported_geometry_types(geoms: list[Any], labels: list[str]) -> None:
    """Refuse a geometry type this process has no defined sampling rule for.

    Polygons are area-weighted and points are interpolated. A line or a multipoint would need
    a rule of its own (length weighting, or one value per member), so it is refused by name
    rather than routed through a path that would return a number with no defined meaning.
    """
    for geom, label in zip(geoms, labels, strict=True):
        if geom.geom_type not in _POLYGON_TYPES | _POINT_TYPES:
            raise ValueError(
                f"aggregate_spatial: geometry '{label}' is a {geom.geom_type}, which is not "
                "supported; only Polygon, MultiPolygon and Point are"
            )


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


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


_probe: ContextVar[list[str] | None] = ContextVar("aggregate_spatial_reducer_probe", default=None)

# Two arrays whose mean, sum, min, max and median are pairwise distinct, so a reducer that
# matches one statistic on both is that statistic.
_FINGERPRINT_INPUTS = (np.array([1.0, 2.0, 4.0, 8.0]), np.array([0.5, 3.0, 7.0]))
_FINGERPRINTS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(np.mean(a)),
    "sum": lambda a: float(np.sum(a)),
    "min": lambda a: float(np.min(a)),
    "max": lambda a: float(np.max(a)),
    "median": lambda a: float(np.median(a)),
}


def _identify_reducer(reducer: Callable, context: Any) -> str | None:
    """Name the statistic *reducer* computes, or None when it is not one the weighted path knows.

    An openEO reducer arrives as an opaque callable over a process graph, so it is identified
    by calling it. ``reduce_by_method`` names itself through the probe; any other reducer is
    matched on two fixed inputs against the statistics above. A reducer that fails, or matches
    none, is treated as unknown and takes the pixel-centre fallback.
    """
    call = _make_reducer_caller(reducer, context)
    named: list[str] = []
    token = _probe.set(named)
    try:
        results = [call(values) for values in _FINGERPRINT_INPUTS]
    except Exception:
        results = None
    finally:
        _probe.reset(token)
    # One entry per probe call; a reducer that named two different methods is a composite.
    if len(set(named)) == 1:
        return named[0]
    if named or results is None:
        return None
    for name, statistic in _FINGERPRINTS.items():
        if all(np.isclose(result, statistic(values)) for result, values in zip(results, _FINGERPRINT_INPUTS)):
            return name
    return None


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Grid:
    """The cube's cell layout, as exactextract sees it (row 0 north, column 0 west)."""

    x_dim: str
    y_dim: str
    width: int
    height: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    x_ascending: bool
    y_ascending: bool

    @classmethod
    def of(cls, data: xr.Dataset, x_dim: str, y_dim: str) -> _Grid:
        x = data[x_dim].values.astype(float)
        y = data[y_dim].values.astype(float)
        dx = float(abs(x[1] - x[0])) if x.size > 1 else 1.0
        dy = float(abs(y[1] - y[0])) if y.size > 1 else 1.0
        return cls(
            x_dim=x_dim,
            y_dim=y_dim,
            width=int(x.size),
            height=int(y.size),
            xmin=float(x.min()) - dx / 2,
            xmax=float(x.max()) + dx / 2,
            ymin=float(y.min()) - dy / 2,
            ymax=float(y.max()) + dy / 2,
            x_ascending=not (x.size > 1 and x[1] < x[0]),
            y_ascending=bool(y.size > 1 and y[1] > y[0]),
        )

    def array_indices(self, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map exactextract cell ids to (row, column) indices into the cube's own arrays."""
        rows, cols = np.divmod(cell_ids.astype(np.int64), self.width)
        if self.y_ascending:
            rows = self.height - 1 - rows
        if not self.x_ascending:
            cols = self.width - 1 - cols
        return rows, cols


@dataclass
class _Zone:
    """The cells a polygon covers, as indices into the cube, with the fraction of each covered."""

    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray


def _polygon_zones(grid: _Grid, polygons: Sequence[Any]) -> list[_Zone]:
    """Coverage fractions for each polygon, in one exactextract call."""
    from exactextract import exact_extract
    from exactextract.raster import NumPyRasterSource
    from shapely.geometry import mapping

    if not polygons:
        return []
    # A zero-stride array: exactextract needs the grid, not the values, so nothing is allocated.
    source = NumPyRasterSource(
        np.broadcast_to(np.float32(0), (grid.height, grid.width)),
        xmin=grid.xmin,
        ymin=grid.ymin,
        xmax=grid.xmax,
        ymax=grid.ymax,
    )
    features = [{"type": "Feature", "properties": {}, "geometry": mapping(geom)} for geom in polygons]
    # A DataFrame with output="pandas"; the untyped return is annotated so pyright accepts it.
    table: Any = exact_extract(source, features, ["cell_id", "coverage"], output="pandas")
    zones: list[_Zone] = []
    for cell_ids, coverage in zip(table["cell_id"], table["coverage"], strict=True):
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        weights = np.asarray(coverage, dtype="float64")
        keep = weights > _MIN_COVERAGE
        rows, cols = grid.array_indices(cell_ids[keep])
        zones.append(_Zone(rows=rows, cols=cols, weights=weights[keep]))
    return zones


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


def _weighted(values: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    """Reduce the last axis of *values* (…, cells) by *method*, weighting each cell.

    NaN cells are left out, as the unweighted path drops them. A zone with no valid cell is NaN.
    """
    valid = ~np.isnan(values)
    w = np.where(valid, weights, 0.0)
    total = w.sum(axis=-1)
    filled = np.where(valid, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "mean":
            out = (filled * w).sum(axis=-1) / total
        elif method == "sum":
            out = (filled * w).sum(axis=-1)
            out = np.where(total > 0, out, np.nan)
        elif method == "min":
            out = np.where(valid, values, np.inf).min(axis=-1, initial=np.inf)
            out = np.where(np.isfinite(out), out, np.nan)
        elif method == "max":
            out = np.where(valid, values, -np.inf).max(axis=-1, initial=-np.inf)
            out = np.where(np.isfinite(out), out, np.nan)
        elif method == "median":
            out = _weighted_quantile(values, w, 0.5)
        elif method == "majority":
            out = _weighted_majority(values, w)
        else:
            raise ValueError(f"no weighted implementation for '{method}'")
    return np.asarray(out, dtype="float64")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """The smallest value whose cumulative weight reaches *q* of the zone's total."""
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full(flat_v.shape[0], np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        keep = w > 0
        if not keep.any():
            continue
        order = np.argsort(v[keep])
        cumulative = np.cumsum(w[keep][order])
        out[i] = v[keep][order][np.searchsorted(cumulative, q * cumulative[-1])]
    return out.reshape(values.shape[:-1])


def _weighted_majority(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The class covering the most area; ties go to the smallest class value."""
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full(flat_v.shape[0], np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        keep = w > 0
        if not keep.any():
            continue
        classes, inverse = np.unique(v[keep], return_inverse=True)
        out[i] = classes[np.argmax(np.bincount(inverse, weights=w[keep]))]
    return out.reshape(values.shape[:-1])


def _weighted_fractions(values: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Share of each zone's valid area in each class, as (…, class)."""
    valid = ~np.isnan(values)
    w = np.where(valid, weights, 0.0)
    total = w.sum(axis=-1, keepdims=True)
    per_class = np.stack([np.where(values == c, w, 0.0).sum(axis=-1) for c in classes], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(per_class / total, dtype="float64")


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


def _is_categorical(data: xr.Dataset) -> bool:
    """True when the cube's dataset declares a categorical ``ingestion.resampling``."""
    declared = {str(data[v].attrs.get(RESAMPLING_ATTR, "")) for v in data.data_vars}
    declared.add(str(data.attrs.get(RESAMPLING_ATTR, "")))
    return bool(declared & _CATEGORICAL_RESAMPLING)


def _point_method(requested: str | None, categorical: bool) -> str:
    """The xarray interpolation method for points, from openEO's resampling vocabulary."""
    if requested is None:
        return "nearest" if categorical else "linear"
    if requested not in _POINT_METHODS:
        raise ValueError(
            f"aggregate_spatial: method '{requested}' is not supported; expected one of {sorted(_POINT_METHODS)}"
        )
    if categorical and requested != "near":
        logger.warning(
            "aggregate_spatial: '%s' interpolation requested over categorical data; class codes "
            "between cells are interpolated as if ordinal",
            requested,
        )
    return _POINT_METHODS[requested]


def _sample_points(data: xr.Dataset, grid: _Grid, points: Sequence[Any], method: str) -> xr.Dataset:
    """Sample every variable at each point, as a cube with a leading ``__point__`` dimension.

    Where interpolation returns NaN — a point within half a cell of the grid edge, or next to
    a missing cell such as sea beside a coastal facility — the containing cell's value is used.
    """
    xs = xr.DataArray([p.x for p in points], dims="__point__")
    ys = xr.DataArray([p.y for p in points], dims="__point__")
    nearest = data.sel({grid.x_dim: xs, grid.y_dim: ys}, method="nearest")
    inside = (xs >= grid.xmin) & (xs <= grid.xmax) & (ys >= grid.ymin) & (ys <= grid.ymax)
    nearest = nearest.where(inside)
    if method == "nearest":
        return nearest.drop_vars([grid.x_dim, grid.y_dim], errors="ignore")
    interpolated = data.interp({grid.x_dim: xs, grid.y_dim: ys}, method=method)  # type: ignore[arg-type]
    interpolated = interpolated.drop_vars([grid.x_dim, grid.y_dim], errors="ignore")
    return interpolated.fillna(nearest.drop_vars([grid.x_dim, grid.y_dim], errors="ignore"))


def _crop(data: xr.Dataset, grid: _Grid, zones: list[_Zone]) -> tuple[xr.Dataset, int, int]:
    """Load only the window the zones cover, returning it with its row and column offsets."""
    covered = [z for z in zones if z.rows.size]
    if not covered:
        return data.isel({grid.y_dim: slice(0, 0), grid.x_dim: slice(0, 0)}), 0, 0
    r0 = int(min(z.rows.min() for z in covered))
    r1 = int(max(z.rows.max() for z in covered))
    c0 = int(min(z.cols.min() for z in covered))
    c1 = int(max(z.cols.max() for z in covered))
    return data.isel({grid.y_dim: slice(r0, r1 + 1), grid.x_dim: slice(c0, c1 + 1)}), r0, c0


def _weighted_polygons(data: xr.Dataset, grid: _Grid, zones: list[_Zone], method: str) -> dict[str, _VarResult]:
    """Per variable: an array (zone, …other dims) and its dims and coords."""
    window, r0, c0 = _crop(data, grid, zones)
    out: dict[str, _VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        da = window[vname]
        other = [d for d in da.dims if d not in {grid.y_dim, grid.x_dim}]
        arr = np.asarray(da.transpose(*other, grid.y_dim, grid.x_dim).values, dtype="float64")
        coords = {d: da.coords[d].values for d in other if d in da.coords}
        classes = np.array([])
        if method == "fractions":
            picked = [arr[..., z.rows - r0, z.cols - c0] for z in zones if z.rows.size]
            found = np.concatenate([p.ravel() for p in picked]) if picked else np.array([])
            classes = np.unique(found[~np.isnan(found)])
        rows = []
        for zone in zones:
            shape = arr.shape[:-2]
            if not zone.rows.size:
                rows.append(np.full(shape + ((classes.size,) if method == "fractions" else ()), np.nan))
                continue
            values = arr[..., zone.rows - r0, zone.cols - c0]
            weights = np.broadcast_to(zone.weights, values.shape)
            if method == "fractions":
                rows.append(_weighted_fractions(values, weights, classes))
            else:
                rows.append(_weighted(values, weights, method))
        dims = list(other)
        if method == "fractions":
            dims.append(FRACTIONS_DIM)
            coords[FRACTIONS_DIM] = classes
        out[vname] = (np.stack(rows), dims, coords)
    return out


def _pixel_centre_polygons(
    data: xr.Dataset, grid: _Grid, polygons: Sequence[Any], reducer: Callable, context: Any
) -> dict[str, _VarResult]:
    """The openEO specification's rule, for reducers the weighted path does not know.

    A cell counts when its centre lies inside the polygon; the reducer runs on those values.
    """
    import rasterio.features
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    transform = from_bounds(grid.xmin, grid.ymin, grid.xmax, grid.ymax, grid.width, grid.height)
    reduce = _make_reducer_caller(reducer, context)
    out: dict[str, _VarResult] = {}
    masks = []
    for geom in polygons:
        mask = rasterio.features.geometry_mask(
            [mapping(geom)], out_shape=(grid.height, grid.width), transform=transform, invert=True
        )
        # rasterio builds the mask north row first; flip when the cube's y ascends.
        if grid.y_ascending:
            mask = mask[::-1]
        if not grid.x_ascending:
            mask = mask[:, ::-1]
        masks.append(mask.ravel())
    for name in data.data_vars:
        vname = str(name)
        da = data[vname]
        other = [d for d in da.dims if d not in {grid.y_dim, grid.x_dim}]
        arr = da.transpose(*other, grid.y_dim, grid.x_dim).values
        flat = arr.reshape((-1, grid.height * grid.width))
        shape = arr.shape[:-2]
        rows = []
        for mask in masks:
            reduced = []
            for pixels in flat:
                selected = pixels[mask]
                if np.issubdtype(selected.dtype, np.floating):
                    selected = selected[~np.isnan(selected)]
                reduced.append(reduce(selected))
            rows.append(np.asarray(reduced, dtype="float64").reshape(shape))
        coords = {d: da.coords[d].values for d in other if d in da.coords}
        out[vname] = (np.stack(rows) if rows else np.empty((0, *shape)), other, coords)
    return out


def _sampled_points(
    data: xr.Dataset, grid: _Grid, points: Sequence[Any], method: str, reducer: Callable | None, context: Any
) -> dict[str, _VarResult]:
    """Per variable: the sampled value at each point, passed through *reducer* when one is unknown."""
    sampled = _sample_points(data, grid, points, method)
    reduce = _make_reducer_caller(reducer, context) if reducer is not None else None
    out: dict[str, _VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        da = sampled[vname]
        other = [d for d in da.dims if d != "__point__"]
        arr = np.asarray(da.transpose("__point__", *other).values, dtype="float64")
        if reduce is not None:
            # An unknown reducer sees the one sampled value, as it would see a zone's pixels.
            flat = arr.reshape(arr.shape[0], -1)
            reduced = [[reduce(np.array([v])) if not np.isnan(v) else np.nan for v in row] for row in flat]
            arr = np.asarray(reduced, dtype="float64").reshape(arr.shape)
        coords = {d: da.coords[d].values for d in other if d in da.coords}
        out[vname] = (arr, other, coords)
    return out


def _warn_empty_zones(result: xr.Dataset, geom_dim: str, labels: list[str]) -> None:
    """Log the geometries that returned no value at all, which otherwise arrive as a silent NaN."""
    empty = np.ones(len(labels), dtype=bool)
    for da in result.data_vars.values():
        flat = da.transpose(geom_dim, ...).values.reshape(len(labels), -1)
        if flat.shape[1]:
            empty &= np.isnan(flat).all(axis=1)
    missing = [label for label, is_empty in zip(labels, empty, strict=True) if is_empty]
    if missing:
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10 else "")
        logger.warning(
            "aggregate_spatial: %d of %d geometries captured no data and are NaN: %s",
            len(missing),
            len(labels),
            shown,
        )


@process(
    summary="Aggregate spatial data within geometries",
    parameters={
        "data": {"description": "A raster data cube."},
        "geometries": {"description": "GeoJSON FeatureCollection, Feature, or geometry (polygons or points)."},
        "reducer": {"description": "A reducer to apply on the pixel values."},
        "target_dimension": {"description": "Name for the new geometry dimension (default: 'geometry')."},
        "context": {"description": "Optional context passed to the reducer."},
        "method": {
            "description": (
                "OCS extension. How points sample the surface, in openEO's resample_spatial "
                "vocabulary: near, bilinear or cubic. Defaults to bilinear, or near for a "
                "categorical dataset. Polygons are always area-weighted."
            )
        },
    },
)
def aggregate_spatial(
    data: Any,
    geometries: Any,
    reducer: Callable,
    target_dimension: str | None = None,
    context: Any = None,
    method: str | None = None,
) -> xr.Dataset:
    """Aggregate raster values within each polygon, or sample them at each point."""
    if method is not None and method not in _POINT_METHODS:
        # Checked before anything else, so a typo fails whether or not the geometries have points.
        raise ValueError(
            f"aggregate_spatial: method '{method}' is not supported; expected one of {sorted(_POINT_METHODS)}"
        )
    geom_shapes, geom_labels = _parse_geometries(geometries)
    if not geom_shapes:
        raise ValueError("aggregate_spatial: geometries contains no shapes")

    if isinstance(data, xr.DataArray):
        attrs = dict(data.attrs)
        data = data.to_dataset(name=data.name or "data")
        data.attrs.update(attrs)

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial: cannot identify x/y dimensions in {list(data.dims)}")
    grid = _Grid.of(data, x_dim, y_dim)

    categorical = _is_categorical(data)
    named = _identify_reducer(reducer, context)
    effective = named
    if categorical and named in _CATEGORICAL_REPLACED:
        logger.warning(
            "aggregate_spatial: '%s' requested over categorical data; using the area-weighted "
            "majority class instead of averaging class codes",
            named,
        )
        effective = "majority"
    if effective is None:
        logger.info(
            "aggregate_spatial: reducer is not one the area-weighted path recognises; polygons "
            "use pixel-centre selection"
        )

    polygon_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in _POLYGON_TYPES]
    point_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in _POINT_TYPES]
    if point_idx and effective == "fractions":
        raise ValueError("aggregate_spatial: fractions apply to polygons; a point has one class, not a share")

    from open_climate_service.shared.provenance import observe_spatial_aggregation, record_spatial_reduction

    parts: list[tuple[list[int], dict[str, _VarResult]]] = []
    with observe_spatial_aggregation():
        if effective is not None:
            # The reducer is not called on this path, so the method is recorded here.
            record_spatial_reduction(effective)
        if polygon_idx:
            polygons = [geom_shapes[i] for i in polygon_idx]
            if effective in _WEIGHTED_METHODS:
                values = _weighted_polygons(data, grid, _polygon_zones(grid, polygons), str(effective))
            else:
                values = _pixel_centre_polygons(data, grid, polygons, reducer, context)
            parts.append((polygon_idx, values))
        if point_idx:
            points = [geom_shapes[i] for i in point_idx]
            point_method = _point_method(method, categorical or effective == "majority")
            values = _sampled_points(
                data, grid, points, point_method, None if effective in _WEIGHTED_METHODS else reducer, context
            )
            parts.append((point_idx, values))

    geom_dim = target_dimension or "geometry"
    combined = _assemble(parts, len(geom_shapes), geom_dim)
    _warn_empty_zones(combined, geom_dim, geom_labels)
    combined[geom_dim] = geom_labels
    # Carry the shapes as well as the labels, so the result is a vector datacube rather than a
    # table that has forgotten where it came from. A companion coordinate rather than replacing
    # the labels on `geom_dim`: the label is the feature id, which the DHIS2 and CHAP exports key
    # their location column on. WKT strings, because a string coordinate is inert on every path
    # the cube can take, where an object-dtype one makes `to_zarr` fail. See CLIM-836.
    return combined.assign_coords({GEOMETRY_WKT_COORD: (geom_dim, [geom.wkt for geom in geom_shapes])})


def _assemble(
    parts: list[tuple[list[int], dict[str, _VarResult]]],
    n_geometries: int,
    geom_dim: str,
) -> xr.Dataset:
    """Put polygon and point results back in the input order along *geom_dim*."""
    variables: dict[str, xr.DataArray] = {}
    for name in parts[0][1]:
        first, dims, coords = parts[0][1][name]
        full = np.full((n_geometries, *first.shape[1:]), np.nan)
        for idx, values in parts:
            arr = values[name][0]
            if arr.shape[1:] != first.shape[1:]:
                raise ValueError(f"aggregate_spatial: points and polygons produced different shapes for '{name}'")
            full[idx] = arr
        variables[name] = xr.DataArray(full, dims=[geom_dim, *dims], coords=coords)
    return xr.Dataset(variables)


_REDUCE_METHODS: dict[str, Callable[..., Any]] = {
    "mean": np.mean,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "median": np.median,
}


def _majority(arr: np.ndarray) -> float:
    classes, counts = np.unique(arr, return_counts=True)
    return float(classes[np.argmax(counts)])


@process(
    summary="Reduce pixel values by a named method",
    parameters={
        "data": {"description": "The array of values to reduce."},
        "method": {
            "description": (
                "Reduction method: mean (default), sum, min, max, median, majority, or fractions "
                "(per-class area shares, only inside aggregate_spatial)."
            )
        },
    },
)
def reduce_by_method(data: Any, method: str = "mean") -> float:
    """Reduce an array of values by a named method.

    A spec-compliant alternative to parameterising a reducer's ``process_id``: workflows
    pass ``method`` as an ordinary string argument while ``process_id`` stays the literal
    ``"reduce_by_method"``, so standard openEO tooling can validate the graph. Inside
    ``aggregate_spatial`` the method is read by name and computed area-weighted, so this body
    only runs when the reducer is used elsewhere.
    """
    if method not in _REDUCE_METHODS and method not in {"majority", "fractions"}:
        raise ValueError(
            f"Unknown reduce method '{method}'; expected one of {sorted([*_REDUCE_METHODS, 'majority', 'fractions'])}"
        )
    probing = _probe.get()
    if probing is not None:
        probing.append(method)
        return float("nan")
    if method == "fractions":
        raise ValueError(
            "reduce_by_method: 'fractions' produces one value per class, so it only works in aggregate_spatial"
        )
    from open_climate_service.shared.provenance import record_spatial_reduction

    record_spatial_reduction(method)
    arr = np.asarray(data, dtype="float64").ravel()
    if arr.size == 0:
        return float("nan")
    if method == "majority":
        return _majority(arr)
    return float(_REDUCE_METHODS[method](arr))
