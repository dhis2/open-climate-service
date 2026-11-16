"""Auto-registration of earthkit-meteo functions as openEO process callables.

earthkit-meteo's thermodynamics live behind a dispatching front end that already accepts
``xarray.DataArray``, preserves dims/coords/laziness and stamps CF ``units`` and
``standard_name`` on its output — so unlike xclim's indicators (see ``xclim_processes``)
these need no dimension renaming and no ``apply_ufunc`` wrapper.

What they *do* need is unit enforcement. Every function assumes ECMWF's native units
(temperature in K, pressure in Pa) and does no checking of its own, while our stores are
CF-stamped in whatever the dataset template declares — ERA5-Land temperature, for instance,
is converted to degC on ingest. Feeding degC to a function expecting K produces a plausible
number that is silently wrong. Each function documents the unit it wants per parameter, so
this module reads those declarations and converts (or refuses) on the way in.

Only ``earthkit.meteo`` is auto-registered. earthkit-transforms deliberately is not: its
public functions are decorator-wrapped down to ``(*_args, **_kwargs)``, so there is no
signature left to derive a process spec from, and most of what it offers duplicates
standard openEO processes (``aggregate_temporal``, ``reduce_dimension``, …). The rule of
thumb is in ``docs/extensibility.md``: if openEO already names the operation, use the
standard process; earthkit is for what openEO does not cover.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any

from open_geo_service.process import _GEOSERVICE_PROCESS_ATTR

logger = logging.getLogger(__name__)

_cache: list[Any] | None = None

# Curated — extend as processes need it, rather than exposing the whole library. `wind`
# and `solar` are the obvious next candidates; `score` is forecast verification, which
# belongs with model evaluation rather than in the data-cube process catalogue.
_MODULES: tuple[str, ...] = ("thermo",)

# The units earthkit-meteo declares in its numpydoc parameter descriptions. Only these are
# read as unit declarations — other trailing parentheticals are prose (`method`, `eps`).
_KNOWN_UNITS: frozenset[str] = frozenset({"K", "Pa", "%", "kg/kg", "Pa/K"})

# Exact conversions into each expected unit, keyed by the normalised CF `units` attribute.
# An identity entry means "already correct"; anything absent is refused rather than guessed.
_CONVERSIONS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "K": {
        "k": lambda a: a,
        "kelvin": lambda a: a,
        "degree_kelvin": lambda a: a,
        "degrees_kelvin": lambda a: a,
        "degc": lambda a: a + 273.15,
        "°c": lambda a: a + 273.15,
        "c": lambda a: a + 273.15,
        "celsius": lambda a: a + 273.15,
        "degree_celsius": lambda a: a + 273.15,
        "degrees_celsius": lambda a: a + 273.15,
    },
    "Pa": {
        "pa": lambda a: a,
        "pascal": lambda a: a,
        "hpa": lambda a: a * 100.0,
        "hectopascal": lambda a: a * 100.0,
        "mbar": lambda a: a * 100.0,
        "millibar": lambda a: a * 100.0,
    },
    "%": {
        "%": lambda a: a,
        "percent": lambda a: a,
        # CF dimensionless: relative humidity as a 0..1 fraction.
        "1": lambda a: a * 100.0,
    },
    # d(saturation vapour pressure)/dT — the `es_slope` pre-compute on the *_slope functions.
    "Pa/K": {
        "pa/k": lambda a: a,
        "pa k-1": lambda a: a,
        "pa k**-1": lambda a: a,
        "pa/k**1": lambda a: a,
        "hpa/k": lambda a: a * 100.0,
        "hpa k-1": lambda a: a * 100.0,
    },
    "kg/kg": {
        "kg/kg": lambda a: a,
        "kg kg-1": lambda a: a,
        "kg kg**-1": lambda a: a,
        "kg/kg**1": lambda a: a,
        # Dimensionless mass ratio — numerically identical to kg/kg.
        "1": lambda a: a,
    },
}

_PARAM_LINE = re.compile(r"^(\w+)\s*:")
_TRAILING_PAREN = re.compile(r"\(([^()]*)\)\s*$")
# Parameter *types* that take a cube rather than a scalar. Used to catch a physical input
# whose unit the docstring scan missed: without a unit it would otherwise be advertised as a
# plain number and passed through unconverted, which is the silent-wrong-answer case.
_ARRAY_LIKE_TYPE = re.compile(r"array-like|DataArray|FieldList|\bField\b|ndarray")

# Units the upstream docstring fails to declare in a form `_documented_units` can read.
# Keyed by function name, then parameter.
#
# earthkit-meteo 1.0.0 documents `virtual_temperature.t` as "Temperature (K)s" — a stray
# trailing "s" that defeats the end-of-line match. Without this the parameter gets no unit,
# so a degC cube would be handed to earthkit as if it were Kelvin: a plausible number,
# labelled Kelvin, wrong by 273.15. Drop entries here as upstream fixes them.
_UNIT_OVERRIDES: dict[str, dict[str, str]] = {
    "virtual_temperature": {"t": "K"},
}


def _normalise_unit(units: str) -> str:
    """Normalise a CF ``units`` string for lookup in ``_CONVERSIONS``.

    Collapses case and repeated whitespace only. Deliberately not a full unit parser:
    anything not spelled the way ``_CONVERSIONS`` lists it is refused rather than guessed.
    """
    return " ".join(units.strip().lower().split())


def _documented_units(func: Any) -> dict[str, str]:
    """Return ``{parameter: expected_unit}`` parsed from the function's numpydoc block.

    earthkit-meteo documents each physical parameter as ``Temperature (K)`` on the line
    following the ``name : type`` line. Parameters whose trailing parenthetical is not a
    known unit (``method``, ``eps``, …) are omitted — they are prose, not units.
    """
    doc = inspect.getdoc(func) or ""
    block = re.search(r"Parameters\n-+\n(.*?)(?:\n\n|\Z)", doc, re.S)
    if not block:
        return {}
    units: dict[str, str] = {}
    lines = block.group(1).split("\n")
    for index, line in enumerate(lines):
        match = _PARAM_LINE.match(line)
        if not match or index + 1 >= len(lines):
            continue
        trailing = _TRAILING_PAREN.search(lines[index + 1].strip())
        if trailing and trailing.group(1) in _KNOWN_UNITS:
            units[match.group(1)] = trailing.group(1)
    return units


def _cube_parameters(func: Any) -> set[str]:
    """Return the parameters whose documented *type* is a cube rather than a scalar.

    The unit scan reads the description line; this reads the type line. Comparing the two
    is what catches a physical input whose unit went unparsed — see ``_UNIT_OVERRIDES``.
    """
    doc = inspect.getdoc(func) or ""
    block = re.search(r"Parameters\n-+\n(.*?)(?:\n\n|\Z)", doc, re.S)
    if not block:
        return set()
    names: set[str] = set()
    for line in block.group(1).split("\n"):
        match = re.match(r"^(\w+)\s*:(.*)$", line)
        if match and _ARRAY_LIKE_TYPE.search(match.group(2)):
            names.add(match.group(1))
    return names


def _coerce_units(process_id: str, name: str, value: Any, expected: str) -> Any:
    """Return ``value`` in ``expected`` units, converting when possible.

    Non-xarray values pass through untouched (there is no ``units`` attribute to check).
    A missing ``units`` attribute is an error rather than an assumption: guessing is
    exactly how a degC cube becomes a plausible, wrong answer.
    """
    attrs = getattr(value, "attrs", None)
    if attrs is None:
        return value
    units = str(attrs.get("units", "")).strip()
    if not units:
        raise ValueError(
            f"{process_id}: parameter {name!r} needs CF units to be interpreted safely "
            f"(expected {expected}), but the cube has no 'units' attribute. earthkit-meteo "
            "assumes ECMWF native units and cannot detect a mismatch. Declare 'units' on "
            "the variable in its dataset template so it is stamped at ingest."
        )
    converters = _CONVERSIONS[expected]
    convert = converters.get(_normalise_unit(units))
    if convert is None:
        raise ValueError(
            f"{process_id}: parameter {name!r} is in {units!r}, which cannot be converted to "
            f"the expected {expected}. Supported: {sorted(converters)}."
        )
    converted = convert(value)
    # Conversion produces a new object; restore the metadata and record the true unit.
    if hasattr(converted, "attrs"):
        converted.attrs = {**attrs, "units": expected}
    return converted


def scan() -> list[Any]:
    """Return earthkit-meteo functions as process callables (cached after first call)."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = _collect()
        return _cache
    except Exception:
        logger.warning("Failed to load earthkit-meteo processes", exc_info=True)
        return []


def _collect() -> list[Any]:
    import importlib

    funcs: list[Any] = []
    for module_name in _MODULES:
        module = importlib.import_module(f"earthkit.meteo.{module_name}")
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            # The module namespace also re-exports typing helpers and earthkit-utils'
            # `dispatch`; keep only functions earthkit.meteo itself defines.
            if not callable(obj) or not str(getattr(obj, "__module__", "")).startswith("earthkit.meteo"):
                continue
            signature = inspect.signature(obj)
            # A tuple return cannot be described by openEO's single-datacube return schema
            # (`lcl` is the only one today). Skipping beats advertising a contract we break.
            if "tuple" in str(signature.return_annotation):
                logger.debug("Skipping earthkit.meteo.%s.%s: returns multiple cubes", module_name, name)
                continue
            units = _documented_units(obj)
            units.update(_UNIT_OVERRIDES.get(name, {}))
            # Fail closed: a cube-typed parameter with no enforceable unit would be advertised
            # as a plain number and passed through unconverted, so the answer could be wrong
            # while looking right. Skipping beats advertising a contract we break — the same
            # call made above for tuple-returning functions.
            unitless_cubes = sorted(_cube_parameters(obj) - set(units))
            if unitless_cubes:
                logger.warning(
                    "Skipping earthkit.meteo.%s.%s: cube parameter(s) %s have no unit this "
                    "version can enforce, so a mis-united input could not be detected. Add an "
                    "entry to _UNIT_OVERRIDES (or _CONVERSIONS) to register it.",
                    module_name,
                    name,
                    ", ".join(unitless_cubes),
                )
                continue
            funcs.append(_make_callable(obj, _metadata(obj, name, signature, units), units))
    return funcs


def _metadata(func: Any, name: str, signature: inspect.Signature, units: dict[str, str]) -> dict[str, Any]:
    """Build the openEO process description for one earthkit-meteo function."""
    doc = inspect.getdoc(func) or ""
    # Drop the sphinx "Implementations" tail: it documents which internal backend handles
    # which input type, which is noise in a public process catalogue.
    description = doc.split("\nImplementations")[0].rstrip()
    params: list[dict[str, Any]] = []
    for param_name, param in signature.parameters.items():
        entry: dict[str, Any] = {"name": param_name}
        if param_name in units:
            entry["schema"] = {"type": "object", "subtype": "datacube"}
            entry["description"] = (
                f"Expected unit: {units[param_name]} "
                "(converted automatically when the cube declares a compatible CF unit)."
            )
        else:
            # Non-physical arguments (`method`, `phase`, `eps`) — type from the default.
            entry["schema"] = {"type": "string" if isinstance(param.default, str) else "number"}
        if param.default is not inspect.Parameter.empty:
            entry["optional"] = True
            if param.default is not None:
                entry["default"] = param.default
        params.append(entry)
    return {
        "id": name,
        "summary": description.splitlines()[0] if description else name,
        "description": description,
        "parameters": params,
        "returns": {"schema": {"type": "object", "subtype": "datacube"}},
    }


def _make_callable(func: Any, meta: dict[str, Any], units: dict[str, str]) -> Any:
    """Wrap an earthkit-meteo function as a bare callable with process metadata attached."""
    process_id = meta["id"]

    def _call(**kwargs: Any) -> Any:
        coerced = {
            key: _coerce_units(process_id, key, value, units[key]) if key in units else value
            for key, value in kwargs.items()
        }
        result = func(**coerced)
        # earthkit stamps `units`/`standard_name` on the output but inherits the *input's*
        # name, which mislabels the result (a humidity cube called "t2m"). Name it for the
        # process that produced it — including when the input was unnamed and the inherited
        # name is therefore None, which otherwise left the result anonymous.
        # `data_vars` excludes a Dataset, whose `rename` takes a mapping rather than a name.
        if hasattr(result, "rename") and not hasattr(result, "data_vars"):
            result = result.rename(process_id)
        return result

    setattr(_call, _GEOSERVICE_PROCESS_ATTR, meta)
    return _call
