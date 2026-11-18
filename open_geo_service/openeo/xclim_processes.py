"""Auto-registration of xclim indicators as openEO process callables."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from open_geo_service.process import _GEOSERVICE_PROCESS_ATTR
from open_geo_service.shared.import_lock import IMPORT_LOCK

logger = logging.getLogger(__name__)

_cache: list[Any] | None = None

# The xclim realms registered as processes. `convert` (unit and variable conversions) and
# `generic` are left out.
_REALMS = ("atmos", "land", "seaIce")


def call_with_ogs_time_dim(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call an xclim function, mapping the OGS ``t`` dimension to/from ``time``.

    Our GeoZarr stores name the temporal dimension ``t``, but xclim indicators and
    indices require a dimension literally named ``time`` (they index ``da.time``
    directly). This renames ``t`` -> ``time`` on every xarray argument on the way
    in, and renames ``time`` -> ``t`` back on the result, so xclim works against our
    cubes without the caller adding ``rename_dimension`` to every process graph.
    """
    import xarray as xr

    renamed = False

    def _to_time(value: Any) -> Any:
        nonlocal renamed
        if isinstance(value, (xr.DataArray, xr.Dataset)) and "t" in value.dims and "time" not in value.dims:
            renamed = True
            return value.rename({"t": "time"})
        return value

    args = tuple(_to_time(a) for a in args)
    kwargs = {key: _to_time(value) for key, value in kwargs.items()}
    result = func(*args, **kwargs)

    def _to_t(value: Any) -> Any:
        if isinstance(value, (xr.DataArray, xr.Dataset)) and "time" in value.dims and "t" not in value.dims:
            return value.rename({"time": "t"})
        return value

    if not renamed:
        return result
    # Restore the OGS `t` name on the output(s). Some indicators return a tuple of arrays.
    if isinstance(result, tuple):
        return tuple(_to_t(item) for item in result)
    return _to_t(result)


def scan() -> list[Any]:
    """Return all xclim indicators as process callables (cached after first call).

    Holds `IMPORT_LOCK` itself, since `_collect` is where xclim's indicator modules are imported,
    so no caller has to reach this under the lock (CLIM-1549). It is re-entrant, so the callers
    that already hold it pay nothing, and the cache is filled once.
    """
    global _cache
    with IMPORT_LOCK:
        if _cache is not None:
            return _cache
        try:
            _cache = _collect()
            return _cache
        except Exception:
            logger.warning("Failed to load xclim indicators", exc_info=True)
            return []


def _collect() -> list[Any]:
    import xclim.indicators  # noqa: F401  # pyright: ignore[reportUnusedImport]  # fills the registry
    from xclim.core import InputKind
    from xclim.core.calendar import select_time
    from xclim.core.indicator import registry

    _SKIP_KINDS = {InputKind.DATASET, InputKind.KWARGS}
    _VARIABLE_KINDS = {InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE}
    kind_schema: dict[InputKind, dict[str, str]] = {
        InputKind.VARIABLE: {"type": "object", "subtype": "datacube"},
        InputKind.OPTIONAL_VARIABLE: {"type": "object", "subtype": "datacube"},
        InputKind.QUANTIFIED: {"type": "string"},
        InputKind.FREQ_STR: {"type": "string"},
        InputKind.NUMBER: {"type": "number"},
        InputKind.STRING: {"type": "string"},
        InputKind.DAY_OF_YEAR: {"type": "string"},
        InputKind.DATE: {"type": "string"},
        InputKind.NUMBER_SEQUENCE: {"type": "array"},
        InputKind.BOOL: {"type": "boolean"},
        InputKind.DICT: {"type": "object"},
    }

    # An `indexer` parameter passes extra keywords on to select_time (month, season, ...).
    time_selection = set(inspect.signature(select_time).parameters) - {"da"}

    funcs: list[Any] = []

    # Registry keys are lower-cased identifiers, used as process ids. Dotted keys are the anuclim,
    # cf and icclim collections, which repeat the core indicators under their own names.
    for indicator_id, obj in registry.items():
        if "." in indicator_id or obj.realm not in _REALMS:
            continue

        # Every spec below declares a single datacube return, so an indicator producing
        # several cubes cannot be honoured: a client composing a graph from GET /processes
        # would feed a tuple into save_result or a reducer and get a confusing failure
        # rather than "not supported". Five of xclim 1.0's registered indicators are
        # multi-output — cffwis (6), rain_season (3), jetstream_metric_woollings (2),
        # base_flow_index_seasonal_ratio and sen_slope — and no OGS workflow uses them, so
        # we advertise only the processes we honour (CLIM-860). Registering one process per
        # named output is the richer alternative, worth doing if someone asks for these.
        outputs = len(obj.outputs)
        if outputs > 1:
            logger.debug("Skipping multi-output xclim indicator '%s' (%d outputs)", indicator_id, outputs)
            continue

        params: list[dict[str, Any]] = []
        for name, p in obj.parameters.items():
            if p.kind in _SKIP_KINDS:
                continue
            param_meta: dict[str, Any] = {"name": name, "schema": kind_schema.get(p.kind, {})}
            if p.description:
                param_meta["description"] = p.description
            # Only expose JSON-serializable scalar defaults.
            # VARIABLE/OPTIONAL_VARIABLE defaults are dataset-mode variable name
            # strings (e.g. "pr") — not meaningful as API defaults.
            if p.kind not in _VARIABLE_KINDS:
                default = p.default
                if default is not inspect.Parameter.empty and isinstance(default, (str, int, float, bool, type(None))):
                    param_meta["optional"] = True
                    param_meta["default"] = default
            elif p.kind is InputKind.OPTIONAL_VARIABLE:
                param_meta["optional"] = True
            params.append(param_meta)

        meta: dict[str, Any] = {
            "id": indicator_id,
            "summary": obj.title,
            "description": obj.abstract,
            "parameters": params,
            "returns": {"schema": {"type": "object", "subtype": "datacube"}},
        }
        extra = {name for name, p in obj.parameters.items() if p.kind is InputKind.KWARGS}
        accepted: set[str] | None = None
        # Other keyword parameters, such as the fire-model options `dc` and `dmc` take as `params`
        # (dc_start, snow_thresh, ...), have no list to check against, so those calls are not checked.
        if extra <= {"indexer"}:
            accepted = set(obj.parameters) - extra
            if extra:
                accepted |= time_selection
        funcs.append(_make_callable(obj, meta, accepted))

    return funcs


def _make_callable(indicator: Any, meta: dict[str, Any], accepted: set[str] | None) -> Any:
    """Wrap an xclim indicator as a bare callable with process metadata attached.

    An unknown argument is refused by name. xclim would pass it on to select_time and fail with
    "unexpected keyword argument", which reads as an internal error; xclim 1.0 renamed `op` to
    `condition` in most indicators, so graphs written for 0.x hit this (CLIM-1767). With
    ``accepted`` None, every argument is passed on.
    """

    def _call(**kwargs: Any) -> Any:
        unknown = sorted(set(kwargs) - accepted) if accepted is not None else []
        if unknown:
            known = ", ".join(p["name"] for p in meta["parameters"])
            raise TypeError(f"{meta['id']} has no parameter {', '.join(unknown)}; its parameters are {known}")
        return _single_output(call_with_ogs_time_dim(indicator, **kwargs))

    setattr(_call, _GEOSERVICE_PROCESS_ATTR, meta)
    return _call


def _single_output(result: Any) -> Any:
    """Return the one cube of a single-output indicator.

    xclim 1.0 indicators return a Dataset by default. Its ``as_dataset`` option is a process-wide
    global with no per-call override, and jobs run in threads, so we unwrap the result here
    rather than switch the option off around the call (CLIM-1767).
    """
    import xarray as xr

    if isinstance(result, xr.Dataset):
        (name,) = result.data_vars
        return result[name]
    return result
