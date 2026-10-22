"""Auto-registration of xclim indicators as openEO process callables."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from open_climate_service.process import _OCS_PROCESS_ATTR
from open_climate_service.shared.import_lock import IMPORT_LOCK

logger = logging.getLogger(__name__)

_cache: list[Any] | None = None


def call_with_ocs_time_dim(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call an xclim function, mapping the OCS ``t`` dimension to/from ``time``.

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
    # Restore the OCS `t` name on the output(s). Some indicators return a tuple of arrays.
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
    import xclim.indicators.atmos as atmos
    import xclim.indicators.land as land
    import xclim.indicators.seaIce as seaice
    from xclim.core.indicator import InputKind

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
    }

    funcs: list[Any] = []
    seen: set[str] = set()

    for mod in (atmos, land, seaice):
        for obj in vars(mod).values():
            if not (hasattr(obj, "identifier") and hasattr(obj, "parameters") and hasattr(obj, "title")):
                continue
            indicator_id: str = obj.identifier
            if indicator_id in seen:
                continue
            seen.add(indicator_id)

            # Every spec below declares a single datacube return, so an indicator producing
            # several cubes cannot be honoured: a client composing a graph from GET /processes
            # would feed a tuple into save_result or a reducer and get a confusing failure
            # rather than "not supported". Three of xclim 0.61's registered indicators are
            # multi-output — cffwis (6), rain_season (3), jetstream_metric_woollings (2) — and
            # no OCS workflow uses them, so advertising 176 processes we honour beats 179 where
            # three lie (CLIM-860). Registering one process per named output is the richer
            # alternative, worth doing if someone asks for these specifically.
            outputs = _declared_output_count(obj)
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
                    if default is not inspect.Parameter.empty and isinstance(
                        default, (str, int, float, bool, type(None))
                    ):
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
            funcs.append(_make_callable(obj, meta))

    return funcs


def _declared_output_count(indicator: Any) -> int:
    """How many cubes an xclim indicator returns, per its own CF metadata.

    ``cf_attrs`` carries one entry per output, so its length is the indicator's arity.
    Anything unreadable counts as one: a wrong guess here would silently drop a
    single-output indicator, and losing a process is worse than the mismatch this guards.
    """
    try:
        return len(indicator.cf_attrs)
    except Exception:  # noqa: BLE001 — metadata shape is xclim's business, not ours
        logger.debug("Could not read cf_attrs for %r; assuming single output", indicator, exc_info=True)
        return 1


def _make_callable(indicator: Any, meta: dict[str, Any]) -> Any:
    """Wrap an xclim indicator as a bare callable with process metadata attached."""

    def _call(**kwargs: Any) -> Any:
        return call_with_ocs_time_dim(indicator, **kwargs)

    setattr(_call, _OCS_PROCESS_ATTR, meta)
    return _call
