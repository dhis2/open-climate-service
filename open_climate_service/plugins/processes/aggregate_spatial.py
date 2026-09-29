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

The weighted path applies to reducers that are exactly one known statistic (mean, sum, min,
max, median, and the categorical majority and fractions), recognised by the structure of their
graph. Any other reducer is an arbitrary process graph that cannot be weighted, so it falls back
to the specification's pixel-centre selection and runs as given.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Iterator, Sequence
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
    import functools
    import inspect

    if isinstance(reducer, functools.partial) and getattr(reducer.func, "__name__", "") == "node_callable":
        # An openEO callback, called the way openeo-processes-dask calls one: the values
        # positionally, named `data` for the graph, so every node can reference the parameter.
        # Passing `data=` as a keyword instead reaches the graph's result node, which in a
        # composite reducer (`mean` then `multiply`) is not a process that takes `data`.
        def _call_graph(pixels: np.ndarray) -> float:
            if not pixels.size:
                return float("nan")
            return float(reducer(pixels, positional_parameters={"data": 0}, named_parameters={"context": context}))

        return _call_graph

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


# openEO processes the weighted path computes when a reducer graph is exactly one of them.
_GRAPH_STATISTICS = frozenset({"mean", "sum", "min", "max", "median"})

# The same statistics as plain Python reducers, recognised by identity (NaN is dropped before a
# reducer runs, so the nan-variants mean the same thing here).
_NUMPY_STATISTICS: dict[Any, str] = {
    np.mean: "mean",
    np.nanmean: "mean",
    np.sum: "sum",
    np.nansum: "sum",
    np.min: "min",
    np.nanmin: "min",
    np.max: "max",
    np.nanmax: "max",
    np.median: "median",
    np.nanmedian: "median",
}


def _identify_reducer(reducer: Callable) -> str | None:
    """Name the statistic *reducer* is, or None when the weighted path cannot stand in for it.

    Identified by structure, never by calling it: a graph that computes a named statistic and
    then does anything more (``mean`` times 2, the smaller of ``mean`` and 10) is not that
    statistic, and would silently lose the rest of its work if it were treated as one. So a
    reducer counts only when it is exactly one known step applied to the ``data`` parameter:

    * an openEO graph with a single node, ``mean``/``sum``/``min``/``max``/``median`` (without
      ``ignore_nodata: false``, which changes how missing values count) or ``reduce_by_method``
      with a literal method;
    * ``reduce_by_method`` bound to a method with ``functools.partial``, or a numpy statistic.

    Everything else takes the pixel-centre fallback, which runs the reducer as given.
    """
    import functools
    import inspect

    known = _NUMPY_STATISTICS.get(reducer) if _hashable(reducer) else None
    if known is not None:
        return known
    if not isinstance(reducer, functools.partial):
        return None
    if reducer.func is reduce_by_method:
        method = reducer.keywords.get("method", "mean")
        if reducer.args or set(reducer.keywords) - {"method"} or method not in _WEIGHTED_METHODS:
            return None
        return str(method)
    # An openEO callback: `partial(node_callable, parent_callables=[...])` from
    # openeo-pg-parser-networkx, whose closure holds the node it runs.
    if getattr(reducer.func, "__name__", "") != "node_callable" or reducer.keywords.get("parent_callables"):
        return None
    try:
        node = inspect.getclosurevars(reducer.func).nonlocals["node_with_data"]
        process_id = node["process_id"]
        arguments = dict(node["resolved_kwargs"])
    except (KeyError, TypeError, ValueError):
        return None
    if getattr(arguments.pop("data", None), "from_parameter", None) != "data":
        return None
    if process_id == "reduce_by_method":
        method = arguments.pop("method", "mean")
        return method if not arguments and isinstance(method, str) and method in _WEIGHTED_METHODS else None
    if process_id in _GRAPH_STATISTICS:
        ignore_nodata = arguments.pop("ignore_nodata", True)
        return str(process_id) if ignore_nodata is True and not arguments else None
    return None


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


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
    """The value at which cumulative weight reaches *q* of the zone's total.

    When the cumulative weight lands exactly on *q*, the midpoint of that value and the next is
    taken, so equal weights give what ``np.median`` gives: two whole cells of 0 and 10 are 5.
    """
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full(flat_v.shape[0], np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        keep = w > 0
        if not keep.any():
            continue
        order = np.argsort(v[keep])
        ordered = v[keep][order]
        cumulative = np.cumsum(w[keep][order])
        target = q * cumulative[-1]
        idx = int(np.searchsorted(cumulative, target))
        if idx + 1 < ordered.size and np.isclose(cumulative[idx], target):
            out[i] = (ordered[idx] + ordered[idx + 1]) / 2
        else:
            out[i] = ordered[min(idx, ordered.size - 1)]
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


MAX_FRACTION_CLASSES = 256
"""Most distinct values ``fractions`` will return shares for.

Land-cover schemes have tens of classes. A continuous layer has as many values as cells, which
would make the output as large as the input; that is refused rather than computed.
"""


def _weighted_fractions(values: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Share of each zone's valid area in each class, as (…, class).

    One pass over the cells: each is binned by its class index, so the work does not grow
    with the number of classes. A zone whose covered cells are all missing is NaN in every class.
    """
    shape = values.shape[:-1]
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    out = np.full((flat_v.shape[0], classes.size), np.nan)
    for i, (v, w) in enumerate(zip(flat_v, flat_w)):
        valid = ~np.isnan(v) & (w > 0)
        total = w[valid].sum()
        if not classes.size or total <= 0:
            continue
        index = np.searchsorted(classes, v[valid])
        out[i] = np.bincount(index, weights=w[valid], minlength=classes.size) / total
    return out.reshape(*shape, classes.size)


def _fraction_classes(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """The class axis shared by every variable, so each variable's shares line up."""
    found = [np.unique(a[~np.isnan(a)]) for a in arrays if a.size]
    classes = np.unique(np.concatenate(found)) if found else np.array([], dtype="float64")
    if classes.size > MAX_FRACTION_CLASSES:
        raise ValueError(
            f"aggregate_spatial: fractions found {classes.size} distinct values in the zones, more than "
            f"{MAX_FRACTION_CLASSES}; fractions are for class codes, not continuous data"
        )
    return classes


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


def _categorical_variables(data: xr.Dataset) -> set[str]:
    """The variables whose dataset declares a categorical ``ingestion.resampling``.

    Decided per variable, from the marker ``load_collection`` sets on the variable itself, so a
    cube merging land cover with temperature averages the temperature and takes the majority of
    the land cover. A cube-level attribute is not consulted: merging copies one variable's
    attributes up to the cube, which would mark every variable as categorical.
    """
    return {
        str(name)
        for name, da in data.data_vars.items()
        if str(da.attrs.get(RESAMPLING_ATTR, "")) in _CATEGORICAL_RESAMPLING
    }


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


READ_BLOCK_BYTES = 512 * 2**20
"""Most bytes of one variable read into memory at once, as float64.

A cube is reduced block by block along its largest non-spatial dimension, so memory stays
bounded however long the series is: 30 years of seNorge daily temperature over Norway is about
160 GB as float64, and was measured at 39 GiB peak for 5 years before blocking.
"""


def _blocks(da: xr.DataArray, other: list[Hashable], grid: _Grid) -> Iterator[tuple[int | None, np.ndarray]]:
    """Yield *da* as float64 (…other, y, x) pieces along its largest non-spatial dimension.

    Each piece is paired with the position of that dimension in *other*, or None when there is
    no non-spatial dimension and the whole array is one piece.
    """
    ordered = da.transpose(*other, grid.y_dim, grid.x_dim)
    if not other:
        yield None, np.asarray(ordered.values, dtype="float64")
        return
    axis = max(range(len(other)), key=lambda i: int(ordered.sizes[other[i]]))
    dim = other[axis]
    per_step = 8 * int(np.prod([int(ordered.sizes[d]) for d in ordered.dims if d != dim]))
    step = max(1, READ_BLOCK_BYTES // max(per_step, 1))
    for start in range(0, int(ordered.sizes[dim]), step):
        yield axis, np.asarray(ordered.isel({dim: slice(start, start + step)}).values, dtype="float64")


def _weighted_polygons(
    data: xr.Dataset, grid: _Grid, zones: list[_Zone], methods: dict[str, str]
) -> dict[str, _VarResult]:
    """Per variable, reduced by its own method: an array (zone, …other dims) and its dims and coords.

    Read and reduced a block at a time (``READ_BLOCK_BYTES``), so memory does not grow with
    the length of the series.
    """
    window, r0, c0 = _crop(data, grid, zones)
    covered = [z for z in zones if z.rows.size]

    def other_dims(vname: str) -> list[Hashable]:
        return [d for d in window[vname].dims if d not in {grid.y_dim, grid.x_dim}]

    # One class axis for every variable reduced to fractions, found only in covered cells. It
    # must be known before any block is reduced, so fractions take one extra pass over the data.
    classes = _fraction_classes(
        [
            np.unique(block[..., z.rows - r0, z.cols - c0])
            for v, m in methods.items()
            if m == "fractions"
            for _axis, block in _blocks(window[v], other_dims(v), grid)
            for z in covered
        ]
    )
    out: dict[str, _VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        method = methods[vname]
        da = window[vname]
        other = other_dims(vname)
        pieces: list[list[np.ndarray]] = [[] for _ in zones]
        concat_axis = 0
        for axis, block in _blocks(da, other, grid):
            concat_axis = axis or 0
            for zone, zone_pieces in zip(zones, pieces, strict=True):
                if not zone.rows.size:
                    extra = (classes.size,) if method == "fractions" else ()
                    zone_pieces.append(np.full(block.shape[:-2] + extra, np.nan))
                    continue
                values = block[..., zone.rows - r0, zone.cols - c0]
                weights = np.broadcast_to(zone.weights, values.shape)
                if method == "fractions":
                    zone_pieces.append(_weighted_fractions(values, weights, classes))
                else:
                    zone_pieces.append(_weighted(values, weights, method))
        rows = [np.concatenate(p, axis=concat_axis) if other else p[0] for p in pieces]
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
        dims: list[Hashable] = list(other)
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
        other: list[Hashable] = [d for d in da.dims if d not in {grid.y_dim, grid.x_dim}]
        pieces: list[list[np.ndarray]] = [[] for _ in masks]
        concat_axis = 0
        # A block at a time, as on the weighted path, so memory does not grow with the series.
        for axis, block in _blocks(da, other, grid):
            concat_axis = axis or 0
            flat = block.reshape((-1, grid.height * grid.width))
            for mask, mask_pieces in zip(masks, pieces, strict=True):
                reduced = [reduce(pixels[mask][~np.isnan(pixels[mask])]) for pixels in flat]
                mask_pieces.append(np.asarray(reduced, dtype="float64").reshape(block.shape[:-2]))
        rows = [np.concatenate(p, axis=concat_axis) if other else p[0] for p in pieces]
        shape = tuple(int(da.sizes[d]) for d in other)
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
        out[vname] = (np.stack(rows) if rows else np.empty((0, *shape)), other, coords)
    return out


def _sampled_points(
    data: xr.Dataset,
    grid: _Grid,
    points: Sequence[Any],
    methods: dict[str, str],
    reducer: Callable | None,
    context: Any,
) -> dict[str, _VarResult]:
    """Per variable, sampled by its own method; passed through *reducer* when that is unknown."""
    if "cubic" in methods.values() and (grid.width < 4 or grid.height < 4):
        raise ValueError(
            f"aggregate_spatial: cubic interpolation needs at least 4 cells along each axis; this grid "
            f"is {grid.width} x {grid.height}, so use 'bilinear' or 'near'"
        )
    sampled = xr.merge(
        [
            _sample_points(data[[v for v, m in methods.items() if m == method]], grid, points, method)
            for method in sorted(set(methods.values()))
        ],
        compat="override",
    )
    reduce = _make_reducer_caller(reducer, context) if reducer is not None else None
    out: dict[str, _VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        da = sampled[vname]
        other: list[Hashable] = [d for d in da.dims if d != "__point__"]
        arr = np.asarray(da.transpose("__point__", *other).values, dtype="float64")
        if reduce is not None:
            # An unknown reducer sees the one sampled value, as it would see a zone's pixels.
            flat = arr.reshape(arr.shape[0], -1)
            reduced = [[reduce(np.array([v])) if not np.isnan(v) else np.nan for v in row] for row in flat]
            arr = np.asarray(reduced, dtype="float64").reshape(arr.shape)
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
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
        # The variable keeps the DataArray's attributes, including the categorical marker.
        data = data.to_dataset(name=data.name or "data")

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial: cannot identify x/y dimensions in {list(data.dims)}")
    grid = _Grid.of(data, x_dim, y_dim)

    categorical = _categorical_variables(data)
    named = _identify_reducer(reducer)
    variables = [str(v) for v in data.data_vars]
    # Per variable: a categorical one takes the majority where the reducer would average codes.
    effective: dict[str, str] = {}
    if named is not None:
        effective = {v: "majority" if v in categorical and named in _CATEGORICAL_REPLACED else named for v in variables}
        replaced = sorted(v for v in variables if effective[v] != named)
        if replaced:
            logger.warning(
                "aggregate_spatial: '%s' requested over categorical data (%s); using the area-weighted "
                "majority class instead of averaging class codes",
                named,
                ", ".join(replaced),
            )
    else:
        logger.info(
            "aggregate_spatial: reducer is not one the area-weighted path recognises; polygons "
            "use pixel-centre selection"
        )

    polygon_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in _POLYGON_TYPES]
    point_idx = [i for i, g in enumerate(geom_shapes) if g.geom_type in _POINT_TYPES]
    if point_idx and named == "fractions":
        raise ValueError("aggregate_spatial: fractions apply to polygons; a point has one class, not a share")

    from open_climate_service.shared.provenance import (
        observe_spatial_aggregation,
        record_spatial_reduction,
        unattributed_spatial_reduction,
    )

    parts: list[tuple[list[int], dict[str, _VarResult]]] = []
    with observe_spatial_aggregation():
        # The reducer is not called on the weighted path, so what ran is recorded here. Two
        # different methods (a merged categorical and continuous cube) record as unattributable.
        for used in sorted(set(effective.values())):
            record_spatial_reduction(used)
        if polygon_idx:
            polygons = [geom_shapes[i] for i in polygon_idx]
            if effective:
                values = _weighted_polygons(data, grid, _polygon_zones(grid, polygons), effective)
            else:
                with unattributed_spatial_reduction():
                    values = _pixel_centre_polygons(data, grid, polygons, reducer, context)
            parts.append((polygon_idx, values))
        if point_idx:
            points = [geom_shapes[i] for i in point_idx]
            point_methods = {
                v: _point_method(method, v in categorical or effective.get(v) == "majority") for v in variables
            }
            with unattributed_spatial_reduction():
                values = _sampled_points(data, grid, points, point_methods, None if effective else reducer, context)
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
    """The most frequent value, leaving missing cells out as the weighted path does."""
    valid = arr[~np.isnan(arr)]
    if not valid.size:
        return float("nan")
    classes, counts = np.unique(valid, return_counts=True)
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
