"""CF Conventions metadata helpers (issue #280).

Stamp CF attributes (``units``, ``standard_name``, ``cell_methods``) onto stored variables
from the dataset template, so CF-aware tools — xclim's climate indices, cf-xarray, QGIS,
earthkit — work against our GeoZarr stores without per-process wrappers.

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

# What a NetCDF attribute may hold. Derived by trying each type against `to_netcdf` rather than
# from the error message it raises, which lists `Number` and so implies `bool` and `complex` are
# fine — both are rejected in practice. Zarr is looser (its attributes are JSON, so a nested
# mapping writes happily), which is the trap: the store accepts the attribute and every later
# NetCDF export of that store fails instead.
_NETCDF_SCALARS: tuple[type, ...] = (str, bytes, int, float)


def drop_unserializable_attrs(obj: xr.Dataset, *, context: str = "") -> xr.Dataset:
    """Remove attributes NetCDF cannot hold, from the dataset and every variable in it.

    Sources hand us attributes shaped for their own catalogue: dynamical.org's GEFS store carries
    ``statistics_approximate`` as a nested mapping, and cfgrib attaches GRIB provenance. Zarr
    stores those without complaint, then `to_netcdf` raises ``TypeError: Invalid value for attr``
    — so an openEO export of the store fails with a 500 while the ingest that produced it looked
    clean.

    Applied at the boundaries rather than per plugin: any plugin can import foreign attributes,
    and a guard that lives in one plugin only protects that plugin. A blocklist of unwriteable
    *types* rather than an allowlist of known keys, because provenance a source chose to publish
    is worth keeping when it can be written — unlike the STAC payload, which deliberately keeps
    only declared CF fields (see ``stac.services._sanitize_variable_attrs``).
    """
    import numpy as np

    def _clean(attrs: dict[Any, Any], where: str) -> None:
        for key in [k for k, value in attrs.items() if not _is_netcdf_writeable(value, np)]:
            logger.debug(
                "Dropping attribute %r on %s%s: %s is not writeable to NetCDF",
                key,
                where,
                f" ({context})" if context else "",
                type(attrs[key]).__name__,
            )
            del attrs[key]

    _clean(obj.attrs, "the dataset")
    for name in list(obj.coords) + list(obj.data_vars):
        _clean(obj[name].attrs, str(name))
    return obj


def _is_netcdf_writeable(value: Any, np: Any) -> bool:
    """Whether `to_netcdf` accepts this attribute value.

    ``bool`` is excluded before the numeric check: NetCDF has no boolean attribute type, and a
    bool passes ``isinstance(..., int)``. A list or tuple must be flat — a nested one raises when
    numpy tries to make an array of it.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (*_NETCDF_SCALARS, np.number)):
        return True
    if isinstance(value, np.ndarray):
        return bool(value.dtype != object)
    if isinstance(value, list | tuple):
        return all(isinstance(item, (*_NETCDF_SCALARS, np.number)) and not isinstance(item, bool) for item in value)
    return False


def cf_attrs_from_template(template: dict[str, Any] | None) -> dict[str, str]:
    """Return the CF variable attributes declared on a dataset template.

    ``units: ""`` is kept (an explicitly dimensionless quantity, e.g. a standardized index).
    """
    if not template:
        return {}
    out: dict[str, str] = {}
    for key in CF_VARIABLE_FIELDS:
        value = template.get(key)
        if isinstance(value, str):
            out[key] = value
    return out


@overload
def apply_cf_metadata(
    obj: "xr.Dataset", attrs: dict[str, str], *, variable: str | None = ..., overwrite: bool = ...
) -> "xr.Dataset": ...


@overload
def apply_cf_metadata(
    obj: "xr.DataArray", attrs: dict[str, str], *, variable: str | None = ..., overwrite: bool = ...
) -> "xr.DataArray": ...


def apply_cf_metadata(
    obj: "xr.Dataset | xr.DataArray",
    attrs: dict[str, str],
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
    try:
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
