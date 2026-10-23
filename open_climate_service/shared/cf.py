"""CF Conventions metadata helpers (issue #280).

Stamp CF attributes (``units``, ``standard_name``, ``cell_methods``, ``long_name``,
``description``) onto stored variables from the dataset template, so CF-aware tools — xclim's
climate indices, cf-xarray, QGIS, earthkit — work against our GeoZarr stores without
per-process wrappers.

The single source of truth is the dataset template. Attributes are stamped only on the
write paths (streaming ingest and managed publish) so the store is CF-compliant on disk;
there is no read-time backfill — to make an existing store CF-compliant, re-ingest it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# Variable-level CF attributes we propagate from the dataset template.
CF_VARIABLE_FIELDS = ("units", "standard_name", "cell_methods")

# Template fields stamped as the variable's descriptive attributes, under these names.
LABEL_FIELDS = {"name": "long_name", "description": "description"}


def cf_attrs_from_template(template: dict[str, Any] | None) -> dict[str, Any]:
    """Return the CF variable attributes declared on a dataset template.

    ``units: ""`` is kept (an explicitly dimensionless quantity, e.g. a standardized index).
    The template's ``name`` and ``description`` become ``long_name`` and ``description``.
    ``standard_name: null`` maps to None, which removes one a source or process left.
    A categorical dataset's classes become ``flag_values`` and ``flag_meanings`` (CLIM-895).
    """
    if not template:
        return {}
    out: dict[str, Any] = {}
    for key in CF_VARIABLE_FIELDS:
        value = template.get(key)
        if isinstance(value, str):
            out[key] = value
    if "standard_name" in template and template["standard_name"] is None:
        out["standard_name"] = None
    for field, attr in LABEL_FIELDS.items():
        value = template.get(field)
        if isinstance(value, str) and value.strip():
            out[attr] = " ".join(value.split())
    from open_climate_service.shared.classes import cf_flag_attrs, classification_from_template

    classification = classification_from_template(template)
    if classification is not None:
        out.update(cf_flag_attrs(classification))
    return out


@overload
def apply_cf_metadata(
    obj: "xr.Dataset", attrs: dict[str, Any], *, variable: str | None = ..., overwrite: bool = ...
) -> "xr.Dataset": ...


@overload
def apply_cf_metadata(
    obj: "xr.DataArray", attrs: dict[str, Any], *, variable: str | None = ..., overwrite: bool = ...
) -> "xr.DataArray": ...


def apply_cf_metadata(
    obj: "xr.Dataset | xr.DataArray",
    attrs: dict[str, Any],
    *,
    variable: str | None = None,
    overwrite: bool = False,
) -> "xr.Dataset | xr.DataArray":
    """Set CF attributes on the target variable(s) in place and return ``obj``.

    With ``overwrite=False`` existing attributes are preserved (the data wins). Ingest and
    managed-publish stamp with ``overwrite=True`` because the dataset template is the
    authoritative source for the CF fields it declares: an explicit template value replaces
    any generic or placeholder value a source/transform left on the variable (e.g. GRIB's
    ``standard_name="unknown"`` or a unit conversion's dimensionally-generic ``"mm"`` where
    the template declares the rate ``"mm/d"``). Fields the template omits are untouched.
    For a ``Dataset``, applies to ``variable`` when given and present, else all data vars.
    """
    import xarray as xr

    if not attrs:
        return obj

    def _set(da: xr.DataArray) -> None:
        for key, value in attrs.items():
            # None declares the attribute absent (see `cf_attrs_from_template`).
            if value is None:
                if overwrite:
                    da.attrs.pop(key, None)
                continue
            # An empty-string units means "dimensionless" and is intentional, so treat it
            # as present; for the others, only fill when missing/blank.
            present = key in da.attrs and (da.attrs[key] != "" or key == "units")
            if overwrite or not present:
                da.attrs[key] = value

    if isinstance(obj, xr.DataArray):
        _set(obj)
        return obj

    targets: list[Any] = [variable] if variable and variable in obj.data_vars else list(obj.data_vars)
    for name in targets:
        _set(obj[name])
    return obj


def validate_units(units: str) -> str | None:
    """Return an error message if ``units`` is not a recognised CF/udunits unit, else None.

    Uses xclim's unit registry (what the xclim indices parse against), so "valid here"
    means "xclim can use it". ``""`` is allowed (dimensionless). Returns None — rather
    than raising — when the validator is unavailable (client-only install).
    """
    if units == "":
        return None
    from open_climate_service.shared.import_lock import IMPORT_LOCK

    try:
        # Not alongside the process warm-up's own first import of xclim (CLIM-1549).
        with IMPORT_LOCK:
            from xclim.core.units import units2pint
    except ImportError:
        # xclim is an optional (server-only) dependency; skip validation when absent.
        return None
    try:
        units2pint(units)
    except Exception as exc:  # noqa: BLE001 — any parse failure means invalid units
        return f"units '{units}' is not a recognised CF/udunits unit ({exc})"
    return None


# Units and standard names that mark an interval-scale temperature. Two callers need the same
# judgement: a *relative* (percent-of-normal) anomaly is meaningless for temperature, and a
# diverging colour scale runs the opposite way — warm is red, whereas for precipitation wet is
# blue.
_TEMPERATURE_UNITS = frozenset(
    {
        "degc",
        "°c",
        "c",
        "celsius",
        "degree_celsius",
        "degrees_celsius",
        "k",
        "kelvin",
        "degree_kelvin",
        "degrees_kelvin",
    }
)


def is_temperature_like(*objects: Any) -> bool:
    """Whether any object's ``units``/``standard_name`` attrs mark it as a temperature."""
    for obj in objects:
        attrs: dict[str, Any] = getattr(obj, "attrs", None) or {}
        units = str(attrs.get("units", "")).strip().lower()
        standard_name = str(attrs.get("standard_name", "")).strip().lower()
        if units in _TEMPERATURE_UNITS or "temperature" in standard_name:
            return True
    return False
