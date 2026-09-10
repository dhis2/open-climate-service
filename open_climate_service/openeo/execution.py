"""Process graph execution for openEO jobs.

Uses openeo-pg-parser-networkx for process graph parsing and
openeo-processes-dask for the standard openEO process implementations.
"""

from __future__ import annotations

import datetime
import functools
import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr
from fastapi import HTTPException, Request

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset, open_zarr_dataset
from open_climate_service.data_manager.services.utils import get_time_dim
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat

logger = logging.getLogger(__name__)

_registry: Any = None  # lazy singleton


@xr.register_dataset_accessor("openeo")
class _DatasetOpenEOAccessor:  # pyright: ignore[reportUnusedClass]
    """Mirrors the DataArray openeo accessor for xr.Dataset.

    openeo-processes-dask only registers the accessor for DataArray.
    Several standard processes (e.g. rename_labels) call data.openeo.temporal_dims
    on the result of aggregate_spatial which returns a Dataset, so we register a
    minimal compatible accessor here.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        from openeo_processes_dask.process_implementations.cubes._xr_interop import (
            BANDS_GUESSES,
            TEMPORAL_GUESSES,
            X_GUESSES,
            Y_GUESSES,
        )

        dim_names = [str(d) for d in ds.dims]
        lower_dims = [d.casefold() for d in dim_names]
        self._ds = ds
        self._temporal = [dim_names[lower_dims.index(g)] for g in TEMPORAL_GUESSES if g in lower_dims]
        self._spatial = [dim_names[lower_dims.index(g)] for g in X_GUESSES + Y_GUESSES if g in lower_dims]
        self._bands = [dim_names[lower_dims.index(g)] for g in BANDS_GUESSES if g in lower_dims]
        self._other = [d for d in dim_names if d not in self._temporal + self._spatial + self._bands]

    @property
    def temporal_dims(self) -> tuple[str, ...]:
        return tuple(d for d in self._temporal if d in self._ds.dims)

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        return tuple(d for d in self._spatial if d in self._ds.dims)

    @property
    def band_dims(self) -> tuple[str, ...]:
        return tuple(d for d in self._bands if d in self._ds.dims)

    @property
    def other_dims(self) -> tuple[str, ...]:
        return tuple(d for d in self._other if d in self._ds.dims)


def _make_sorted_atp(original_fn: Any) -> Any:
    """Wrap aggregate_temporal_period to sort the time axis before resampling.

    Streaming ingests can produce non-monotonic time coordinates when append
    sessions overlap or are replayed out of order. xarray's resample() requires
    a monotonic index, so we sort defensively before delegating.
    """

    def _sorted_atp(data: Any, reducer: Any, period: str, dimension: Any = None, **kwargs: Any) -> Any:
        temporal_dims = data.openeo.temporal_dims
        t_dim = dimension or (temporal_dims[0] if temporal_dims else None)
        if t_dim is not None:
            data = data.sortby(t_dim)
        return original_fn(data=data, reducer=reducer, period=period, dimension=dimension, **kwargs)

    return _sorted_atp


def _make_named_merge_cubes(original_fn: Any) -> Any:
    """Wrap merge_cubes to preserve and extend named predictor cubes.

    The upstream implementation returns a DataArray stacked on a synthetic
    ``__cubes__`` dimension with labels like ``cube1`` / ``cube2``. That loses
    semantic source names like ``tp`` and ``t2m``. Keep the upstream return type
    unchanged for the initial merge, but relabel ``__cubes__`` to the original
    DataArray names when both inputs are named and distinct.

    A later ``merge_cubes`` in the same graph receives that already-stacked
    DataArray plus another predictor. Upstream treats their shared dimensions as
    an unresolved overlap and raises ``OverlapResolverMissing``. Append predictors
    with disjoint ``__cubes__`` labels directly when no resolver is supplied.
    Match upstream's coordinate tolerance and align label order before requiring
    equal indexes, so temporal or location misalignment is never hidden.

    ``aggregate_spatial`` returns an ``xr.Dataset`` even for one input variable.
    Distinct single-variable datasets are normalised only when starting a named
    predictor stack; ordinary Dataset merges retain upstream types and attrs.
    """
    from openeo_processes_dask.process_implementations.cubes.merge import NEW_DIM_NAME

    cube_axis = NEW_DIM_NAME

    def _as_named_array(cube: Any) -> xr.DataArray | None:
        if isinstance(cube, xr.DataArray):
            return cube
        if isinstance(cube, xr.Dataset) and len(cube.data_vars) == 1:
            variable = str(next(iter(cube.data_vars)))
            return cube[variable]
        return None

    def _with_cube_axis(cube: xr.DataArray) -> xr.DataArray | None:
        if cube_axis in cube.dims:
            return cube if cube_axis in cube.coords else None
        if not cube.name:
            return None
        return cube.expand_dims({cube_axis: [str(cube.name)]})

    def _append_disjoint_predictors(cube1: xr.DataArray, cube2: xr.DataArray) -> xr.DataArray:
        left = _with_cube_axis(cube1)
        right = _with_cube_axis(cube2)
        if left is None or right is None:
            raise ValueError("Every predictor in a merged group must have a distinct name")

        left_labels = {str(value) for value in left[cube_axis].values.tolist()}
        right_labels = {str(value) for value in right[cube_axis].values.tolist()}
        if left_labels & right_labels:
            raise ValueError("Named predictor labels must be distinct when extending a merged group")

        if set(left.dims) != set(right.dims):
            raise ValueError("Named predictors must have the same dimensions before merging")
        if set(left.indexes) != set(right.indexes):
            raise ValueError("Named predictors must have the same coordinate indexes before merging")

        from openeo_processes_dask.process_implementations.cubes.merge import FLOAT_TOLERANCE

        # Align index objects rather than sorting cube data. The common case of
        # identical indexes, including unsorted duplicates, stays untouched.
        dimensions = [dim for dim in left.dims if dim != cube_axis and dim in left.indexes]
        for dim in dimensions:
            left_index = left.indexes[dim]
            right_index = right.indexes[dim]
            if len(left_index) != len(right_index):
                raise ValueError(f"Named predictors have different labels on index '{dim}'")
            if left_index.equals(right_index):
                continue
            if not right_index.is_unique:
                raise ValueError(f"Named predictor index '{dim}' cannot be reordered because it has duplicate labels")
            positions = right_index.get_indexer(left_index)
            if (positions < 0).any():
                left_values = left_index.to_numpy()
                right_values = right_index.to_numpy()
                if (
                    left_values.shape == right_values.shape
                    and left_values.dtype.kind == "f"
                    and right_values.dtype.kind == "f"
                ):
                    if (abs(left_values - right_values) < FLOAT_TOLERANCE).all():
                        positions = np.arange(len(left_index))
                    else:
                        order = np.argsort(right_values)
                        sorted_right = right_index.take(order)
                        nearest = sorted_right.get_indexer(
                            left_index,
                            method="nearest",
                            tolerance=FLOAT_TOLERANCE,
                        )
                        if (nearest < 0).any():
                            raise ValueError(f"Named predictors have different labels on index '{dim}'")
                        positions = order[nearest]
                else:
                    raise ValueError(f"Named predictors have different labels on index '{dim}'")
            if len(set(positions.tolist())) != len(left_index):
                raise ValueError(f"Named predictors have different labels on index '{dim}'")
            right = right.isel({dim: positions})
            # Equal timestamps with different datetime64 units and tolerated
            # floating-point noise should use the left cube's canonical labels.
            right = right.assign_coords({dim: left[dim]})

        # Right-only auxiliary coordinates cannot describe the merged group and
        # make concat reject otherwise aligned predictors. Shared auxiliaries,
        # including dimensioned coordinates, keep the left value on conflict.
        right_only_auxiliary = [
            name for name in right.coords if name not in right.indexes and name != cube_axis and name not in left.coords
        ]
        right = right.drop_vars(right_only_auxiliary)
        appended: xr.DataArray = xr.concat(
            [left, right],
            dim=cube_axis,
            join="exact",
            coords="minimal",
            compat="override",
        )
        return appended.chunk({dim: -1 if dim == cube_axis else "auto" for dim in appended.dims})

    def _named_merge_cubes(
        cube1: Any,
        cube2: Any,
        overlap_resolver: Any = None,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        array1 = _as_named_array(cube1)
        array2 = _as_named_array(cube2)
        if array1 is not None and array2 is not None and (cube_axis in array1.dims or cube_axis in array2.dims):
            if overlap_resolver is not None:
                raise ValueError("An overlap resolver is only supported on the initial named predictor merge")
            return _append_disjoint_predictors(array1, array2)

        # Distinct single-variable Datasets are the aggregate_spatial predictor
        # case. Do not promote same-variable Datasets or any ordinary merge.
        promote_datasets = (
            isinstance(cube1, xr.Dataset)
            and isinstance(cube2, xr.Dataset)
            and array1 is not None
            and array2 is not None
            and array1.name != array2.name
        )
        original_cube1 = array1 if promote_datasets else cube1
        original_cube2 = array2 if promote_datasets else cube2
        merged = original_fn(
            cube1=original_cube1,
            cube2=original_cube2,
            overlap_resolver=overlap_resolver,
            context=context,
            **kwargs,
        )
        if (
            array1 is not None
            and array2 is not None
            and array1.name
            and array2.name
            and array1.name != array2.name
            and isinstance(merged, xr.DataArray)
            and cube_axis in merged.dims
        ):
            try:
                return merged.assign_coords({cube_axis: [str(array1.name), str(array2.name)]})
            except Exception as exc:
                logger.debug("Falling back to default __cubes__ labels after merge_cubes", exc_info=exc)
        return merged

    return _named_merge_cubes


def _build_process_registry() -> Any:
    """Build the base process registry (lazy-initialised singleton).

    Contains: all openeo-processes-dask standard processes, backend load_collection /
    save_result, and any exposed native YAML-configured processing plugins.
    UDPs are NOT included here — they are added per-execution by _augment_with_workflows.
    """
    global _registry
    if _registry is not None:
        return _registry

    from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry

    # openeo-processes-dask eagerly imports its whole implementation stack (xvec, odc,
    # dask_geopandas, planetary_computer, …) at module load, so the open-climate-service
    # [server] extra must provide all of it. If any piece is missing, surface one clear,
    # actionable message instead of a deep ModuleNotFoundError mid-job (#289).
    try:
        from openeo_processes_dask.process_implementations.core import process as wrap_fn

        impls_module = importlib.import_module("openeo_processes_dask.process_implementations")
        specs_module = importlib.import_module("openeo_processes_dask.specs")
    except ModuleNotFoundError as exc:
        missing = exc.name or "an openeo-processes-dask dependency"
        raise RuntimeError(
            f"Could not build the openEO process registry: missing server dependency '{missing}'. "
            "Your environment is missing part of the open-climate-service [server] extra "
            "(openeo-processes-dask eagerly imports its full implementation stack). "
            "Re-sync the environment with `uv sync` (run `uv lock --upgrade-package open-climate-service` "
            "first if you just upgraded). Don't `pip install` it by hand — uv sync prunes that. "
            "See the instance guide's 'Upgrading and troubleshooting' section."
        ) from exc

    # Order matters: wrap_funcs are applied left to right, so the first entry ends up
    # innermost. _normalise_temporal_arguments must sit inside wrap_fn to see arguments
    # after ParameterReference resolution — see its docstring.
    registry = ProcessRegistry(wrap_funcs=[_normalise_temporal_arguments, wrap_fn])

    for name, func in inspect.getmembers(impls_module, inspect.isfunction):
        spec: dict[str, Any] = getattr(specs_module, name, None) or {}
        registry[name] = Process(spec=spec, implementation=func)

    # Backend-specific processes override any stub from the standard library.
    registry["load_collection"] = Process(spec={}, implementation=_load_collection_impl)
    registry["save_result"] = Process(spec={}, implementation=_save_result_impl)
    registry["aggregate_temporal_period"] = Process(
        spec=getattr(specs_module, "aggregate_temporal_period", {}),
        implementation=_make_sorted_atp(impls_module.aggregate_temporal_period),
    )
    registry["merge_cubes"] = Process(
        spec=getattr(specs_module, "merge_cubes", {}),
        implementation=_make_named_merge_cubes(impls_module.merge_cubes),
    )

    # @process-decorated plugin functions — override standard processes with same id.
    from open_climate_service.openeo.plugin_processes import load_plugin_processes, to_openeo_descriptor

    for process_id, func in load_plugin_processes():
        registry[process_id] = Process(spec=to_openeo_descriptor(func), implementation=func)

    _registry = registry
    return registry


class _RegistryOverlay:
    """Thin read-only overlay that resolves UDPs before falling back to the base registry.

    pg-parser only calls __getitem__ on the registry, so we only need to implement that.
    """

    def __init__(self, base: Any, udp_map: dict[str, Any]) -> None:
        self._base = base
        self._udps = udp_map

    def __getitem__(self, key: Any) -> Any:
        name = key[1] if isinstance(key, tuple) else key
        if name in self._udps:
            return self._udps[name]
        return self._base[key]


def _resolve_workflow_parameters(node: Any, params: dict[str, Any]) -> Any:
    """Replace ``{"from_parameter": name}`` refs matching a workflow parameter with its value.

    References that are not workflow parameters — e.g. a reducer's ``data`` callback
    parameter — are left untouched so the engine resolves them at call time.
    """
    if isinstance(node, dict):
        if set(node) == {"from_parameter"} and node["from_parameter"] in params:
            return params[node["from_parameter"]]
        return {key: _resolve_workflow_parameters(value, params) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_workflow_parameters(item, params) for item in node]
    return node


def _augment_with_workflows(base_registry: Any) -> Any:
    """Return a registry overlay that adds currently stored workflows to the base registry.

    Workflows are loaded fresh on every call so that PUT /process_graphs changes take
    effect without restarting the server. Returns the base registry unchanged when none exist.
    """
    from openeo_pg_parser_networkx.process_registry import Process

    from open_climate_service.openeo import workflows as workflow_store

    workflow_records = workflow_store.list_workflows().processes
    if not workflow_records:
        return base_registry

    workflow_map: dict[str, Any] = {}
    overlay = _RegistryOverlay(base_registry, workflow_map)

    for wf in workflow_records:
        if not wf.process_graph:
            continue
        pg_dict: dict[str, Any] = dict(wf.process_graph)
        param_defaults: dict[str, Any] = {
            p["name"]: p["default"] for p in (wf.parameters or []) if "name" in p and "default" in p
        }

        def _make_workflow_impl(pg: dict[str, Any], wf_id: str, defaults: dict[str, Any]) -> Any:
            _executing: set[str] = set()

            def _workflow_impl(**kwargs: Any) -> Any:
                from openeo_pg_parser_networkx import OpenEOProcessGraph

                if wf_id in _executing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Recursive workflow call detected: '{wf_id}' called itself",
                    )
                _executing.add(wf_id)
                try:
                    # Substitute the workflow's declared parameters into the graph before
                    # execution. The named_parameters mechanism only resolves callback
                    # parameters (e.g. a reducer's `data`), not top-level UDP arguments, so
                    # without this the workflow's from_parameter refs raise
                    # ProcessParameterMissing. Declared defaults fill in omitted arguments.
                    resolved_pg = _resolve_workflow_parameters(pg, {**defaults, **kwargs})
                    return OpenEOProcessGraph(resolved_pg).to_callable(overlay)()  # pyright: ignore[reportArgumentType]
                finally:
                    _executing.discard(wf_id)

            return _workflow_impl

        workflow_map[wf.id] = Process(spec={}, implementation=_make_workflow_impl(pg_dict, wf.id, param_defaults))

    return overlay


# ---------------------------------------------------------------------------
# load_collection — returns DataArray (openEO data cube)
# ---------------------------------------------------------------------------


def _bbox_to_dict(extent: Any) -> dict[str, float] | None:
    """Normalise a BoundingBox object or plain dict to a lowercase-keyed dict."""
    if extent is None:
        return None
    if isinstance(extent, dict):
        _coord_keys = {"west", "south", "east", "north"}
        return {k.lower(): float(v) for k, v in extent.items() if k.lower() in _coord_keys and v is not None}
    # openeo_pg_parser_networkx BoundingBox pydantic model
    return {
        "west": float(extent.west),
        "south": float(extent.south),
        "east": float(extent.east),
        "north": float(extent.north),
    }


def _temporal_to_list(extent: Any) -> list[str | None] | None:
    """Normalise a TemporalInterval object or plain list to [start, end] strings."""
    if extent is None:
        return None
    items: list[Any]
    if isinstance(extent, list):
        items = extent
    else:
        # TemporalInterval is a pydantic RootModel wrapping a list of Date|None
        items = list(getattr(extent, "root", None) or [])

    result: list[str | None] = []
    for v in items:
        if v is None:
            result.append(None)
        elif hasattr(v, "root"):
            # Date/DateTime wrapper — inner value is a pendulum DateTime
            result.append(_strip_tz(v.root.isoformat()))
        elif isinstance(v, str):
            result.append(_strip_tz(v))
        else:
            result.append(_strip_tz(str(v)))
    return result


def _load_collection_impl(
    id: str,
    spatial_extent: Any = None,
    temporal_extent: Any = None,
    bands: Any = None,
) -> xr.DataArray:
    """Load a published dataset as an openEO data cube (xr.DataArray)."""
    artifact = _get_published_artifact(id)
    ds = _ensure_crs(_open_artifact(artifact))

    bbox = _bbox_to_dict(spatial_extent)
    t_extent = _temporal_to_list(temporal_extent)

    if t_extent is not None:
        start = t_extent[0] if len(t_extent) > 0 else None
        end = t_extent[1] if len(t_extent) > 1 else None
        try:
            t_dim = get_time_dim(ds)
            # Keep a lazy handle to the original time coordinate; only materialise
            # it (in the empty-selection branch below) so valid requests don't pay
            # to load the full coordinate on every call.
            available_coord = ds[t_dim]
            ds = ds.sel({t_dim: slice(start, end)})
        except (ValueError, KeyError):
            # ValueError: no recognisable time dimension
            # KeyError: coordinate exists but is not an indexable dimension
            pass
        else:
            # An empty temporal selection would otherwise flow into downstream
            # reducers (reduce_dimension, aggregate_*) as a zero-length cube and
            # surface as a cryptic "ndim=0" / "zero-size array" / broadcast error.
            # Fail early here, where we can name the dataset and its coverage.
            if ds.sizes.get(t_dim, 0) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"load_collection: dataset '{id}' has no data in the requested "
                        f"temporal_extent [{start}, {end}]. Available time coverage: "
                        f"{_format_time_coverage(available_coord.values)}."
                    ),
                )

    if bbox is not None:
        x_dim = next((d for d in ("x", "longitude") if d in ds.dims), None)
        y_dim = next((d for d in ("y", "latitude") if d in ds.dims), None)
        west, east = bbox.get("west"), bbox.get("east")
        south, north = bbox.get("south"), bbox.get("north")
        if x_dim and west is not None and east is not None:
            ds = ds.sel({x_dim: slice(west, east)})
        if y_dim and south is not None and north is not None:
            coord = ds[y_dim].values
            # Descending latitude axis (e.g. ERA5, WorldPop)
            if len(coord) > 1 and coord[0] > coord[-1]:
                ds = ds.sel({y_dim: slice(north, south)})
            else:
                ds = ds.sel({y_dim: slice(south, north)})

    available_vars = list(ds.data_vars)
    if isinstance(bands, list) and bands:
        available_vars = [b for b in bands if b in ds]
        if not available_vars:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"None of the requested bands {bands!r} exist in collection '{id}'. Available: {list(ds.data_vars)}"
                ),
            )

    if len(available_vars) == 1:
        return ds[available_vars[0]]

    # Multi-band: stack variables on a new "bands" dimension
    import pandas as pd

    return xr.concat(
        [ds[b] for b in available_vars],
        dim=pd.Index(available_vars, name="bands"),
    )


class SaveResultEnvelope:
    """Wraps the process graph result with the requested output format and options."""

    def __init__(self, data: Any, format: str, options: dict[str, Any] | None = None) -> None:
        self.data = data
        self.format = format.upper()
        self.options: dict[str, Any] = options or {}


def _save_result_impl(data: Any, format: str = "Zarr", options: dict[str, Any] | None = None) -> Any:
    return SaveResultEnvelope(data, format, options)


# ---------------------------------------------------------------------------
# Internal helpers shared with _load_collection_impl
# ---------------------------------------------------------------------------


def _format_time_coverage(values: Any) -> str:
    """Human-readable summary of a time coordinate's coverage for error messages."""
    import numpy as np

    vals = np.asarray(values).ravel()
    n = int(vals.size)
    if n == 0:
        return "no timesteps"

    def _fmt(v: Any) -> str:
        try:
            return str(np.datetime_as_string(v, unit="D"))
        except (TypeError, ValueError):
            return str(v)

    step_word = "step" if n == 1 else "steps"
    if n == 1:
        return f"{_fmt(vals[0])} ({n} {step_word})"
    return f"{_fmt(vals.min())} to {_fmt(vals.max())} ({n} {step_word})"


def _get_published_artifact(collection_id: str) -> Any:
    eligible = _eligible_artifacts()
    artifact = eligible.get(collection_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found")
    return artifact


def _eligible_artifacts() -> dict[str, Any]:
    return ingestion_services.latest_published_zarr_artifacts_by_dataset()


def _open_artifact(artifact: Any) -> xr.Dataset:
    path = _artifact_store_path(artifact)
    if artifact.format == ArtifactFormat.ICECHUNK:
        return open_icechunk_dataset(path)
    return open_zarr_dataset(path)


def _ensure_crs(ds: xr.Dataset) -> xr.Dataset:
    """Ensure the cube carries a CF ``spatial_ref`` coordinate that odc.geo can read.

    Streaming-ingested IceChunk stores record their CRS only as GeoZarr root-group
    attributes (``proj:code``), which odc.geo does not interpret. odc-based processes
    such as ``resample_cube_spatial`` then fail with "Can not reproject non-georegistered
    array". Writing the CRS here as a proper CF grid_mapping makes every collection
    consistently georegistered regardless of storage format. Stores written by the
    downloader/pyramid path already carry ``spatial_ref``; ``write_crs`` is a no-op for
    them, so this is safe and idempotent.

    The CRS is taken from the store's declared ``proj:code``, else WGS84 — never the instance
    config CRS, which would tag an untagged cube with a projection its coordinates are not in
    and carry that wrong CRS into anything the job publishes (CLIM-821). Tagging failures are
    logged and the dataset returned untouched rather than failing the load.
    """
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio accessor

    from open_climate_service.shared.crs import dataset_crs

    try:
        if ds.rio.crs is not None:
            return ds
        crs = dataset_crs(ds)
        tagged: xr.Dataset = ds.rio.write_crs(crs)
        return tagged
    except Exception:  # pragma: no cover - defensive; resampling will surface a clearer error
        logger.warning("Could not attach CRS to collection; leaving untagged", exc_info=True)
        return ds


def _artifact_store_path(artifact: Any) -> str:
    if artifact.path:
        return str(artifact.path)
    if artifact.asset_paths:
        return str(artifact.asset_paths[0])
    raise HTTPException(
        status_code=500,
        detail=f"Artifact '{artifact.artifact_id}' has no readable storage path",
    )


def _strip_tz(value: str | None) -> str | None:
    """Strip timezone suffix from an ISO datetime string for naive xarray coords."""
    if value is None:
        return None
    for suffix in ("Z", "+00:00"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _naive_temporal_scalar(value: Any) -> Any:
    """Convert a scalar openEO temporal argument to a timezone-naive string.

    The pg-parser validates any date-typed process argument into a pydantic ``RootModel``
    (``Date``, ``DateTime``, ``Year``, ``Time``) wrapping a **timezone-aware** pendulum
    ``DateTime``. Our published stores carry timezone-naive time coordinates, so a process
    that slices with such a value fails inside pandas with

        cannot do slice indexing on DatetimeIndex with these indexers
        [root=DateTime(1991, 1, 1, 0, 0, 0, tzinfo=Timezone('UTC'))] of type Date

    Anything that is not one of those wrappers is returned untouched. In particular
    ``TemporalInterval`` is left alone — its ``root`` is a *list*, it is normalised
    separately by ``_temporal_to_list`` at the ``load_collection`` boundary, and rewriting
    it to a string here would break that.

    A ``Date`` becomes ``YYYY-MM-DD`` rather than a midnight timestamp. It is the more
    faithful reading of a date-typed argument, and it behaves better as a slice bound:
    pandas partial-string indexing treats ``"2020-12-31"`` as the whole day, whereas
    ``"2020-12-31T00:00:00"`` would exclude everything after midnight on a sub-daily
    series. Other wrappers keep their time component, minus the zone.
    """
    # A timezone-qualified ISO *string* reaches the process raw, because the pg-parser
    # only coerces plain forms like "1991-01-01" into a Date — "1991-01-01T00:00:00Z"
    # fails that validation and stays a string. It then fails just as hard, with
    # "ValueError: Both dates must have the same UTC offset", so strip the offset here.
    # Narrow on purpose: the string must parse as ISO 8601 *and* carry an offset, so
    # ordinary string arguments ("MS", "1991", "07-01", "mm/d", "degC") are untouched,
    # as is a naive ISO string, which is returned unchanged rather than reformatted.
    if isinstance(value, str):
        # Rewrite a trailing "Z" only. A blanket replace() would touch any "Z" in the
        # string, which is harder to reason about than the one position ISO 8601 uses it in.
        candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.datetime.fromisoformat(candidate)
        except ValueError:
            return value
        return value if parsed.tzinfo is None else parsed.replace(tzinfo=None).isoformat()

    root = getattr(value, "root", None)
    # pendulum's DateTime subclasses datetime.date, so one check covers every scalar
    # wrapper while structurally excluding TemporalInterval, whose root is a list.
    if not isinstance(root, datetime.date):
        return value

    # Imported here rather than at module scope to match the rest of this module, which
    # defers every openeo_pg_parser_networkx import so the package is only required when a
    # process graph actually runs. After the first call this is a sys.modules lookup.
    from openeo_pg_parser_networkx.pg_schema import Date

    text = _strip_tz(root.isoformat()) or ""
    if isinstance(value, Date):
        return text.split("T", 1)[0]
    return text


def _normalise_temporal_arguments(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a process so scalar date arguments arrive timezone-naive.

    Registered as the **innermost** wrap function, so it runs after
    openeo-processes-dask's wrapper has resolved ``ParameterReference`` arguments. A date
    supplied through ``{"from_parameter": ...}`` — as workflows do — is still a reference
    when the outer wrapper is entered and only becomes a ``Date`` further in, so
    normalising on the outside would miss exactly that case.

    ``functools.wraps`` matters here beyond cosmetics: the outer wrapper decides whether to
    forward ``axis``/``keepdims``/``context`` by looking at ``inspect.signature`` of what it
    wraps, and that follows ``__wrapped__``. Without it every process would appear to accept
    those arguments and they would be passed on to implementations that reject them.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(
            *(_naive_temporal_scalar(a) for a in args),
            **{key: _naive_temporal_scalar(value) for key, value in kwargs.items()},
        )

    return wrapper


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_process_graph(
    process: dict[str, Any],
    request: Request | None = None,
    result_dir: str | None = None,
) -> Any:
    """Execute an openEO process graph and return the result value."""
    from openeo_pg_parser_networkx import OpenEOProcessGraph

    process_graph = process.get("process_graph")
    if not isinstance(process_graph, dict):
        raise HTTPException(status_code=422, detail="process.process_graph must be an object")

    registry = _augment_with_workflows(_build_process_registry())
    try:
        graph = OpenEOProcessGraph(process_graph)
        return graph.to_callable(registry)()
    except HTTPException:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid process graph: {exc}") from exc
    except Exception as e:
        logger.exception("Process graph execution failed")
        raise HTTPException(status_code=500, detail=f"Process graph execution failed: {e}") from e
