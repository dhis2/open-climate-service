"""Recognising the reducers the weighted path can compute, and calling the ones it cannot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from open_climate_service.process import process

WEIGHTED_METHODS = frozenset({"mean", "sum", "min", "max", "median", "majority", "fractions"})


def is_graph_callback(reducer: Callable) -> bool:
    """Whether *reducer* is an openEO callback graph rather than a plain Python function."""
    import functools

    return isinstance(reducer, functools.partial) and getattr(reducer.func, "__name__", "") == "node_callable"


def make_reducer_caller(reducer: Callable, context: Any) -> Callable[[np.ndarray], float]:
    """Return a function that applies the reducer, forwarding ``context`` when supported.

    openEO reducers may or may not accept a ``context`` keyword; we inspect the
    signature once so context-aware reducers receive it without breaking the
    common array-only reducers (mean, median, ...).
    """
    import inspect

    if is_graph_callback(reducer):
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


def identify_reducer(reducer: Callable) -> str | None:
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
        if reducer.args or set(reducer.keywords) - {"method"} or method not in WEIGHTED_METHODS:
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
        return method if not arguments and isinstance(method, str) and method in WEIGHTED_METHODS else None
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
