"""Unit conversion transforms: named functions for common unit conversions."""

import logging
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)


def _apply(ds: xr.Dataset, dataset: dict[str, Any], *, scale: float, offset: float, units: str) -> xr.Dataset:
    varname = dataset["variable"]
    da = ds[varname]
    converted = da * scale + offset if scale != 1.0 else da + offset
    return ds.assign({varname: converted.assign_attrs({**da.attrs, "units": units})})


def kelvin_to_celsius(ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset:
    """Convert the dataset variable from Kelvin to degrees Celsius."""
    logger.info("Converting '%s' from K to °C", dataset["variable"])
    return _apply(ds, dataset, scale=1.0, offset=-273.15, units="degC")


def metres_to_mm(ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset:
    """Convert the dataset variable from metres to millimetres."""
    logger.info("Converting '%s' from m to mm", dataset["variable"])
    return _apply(ds, dataset, scale=1000.0, offset=0.0, units="mm")


def is_accumulated(da: xr.DataArray) -> bool:
    """Whether a GRIB-derived variable holds amounts accumulated over each step (``GRIB_stepType: accum``).

    A variable without the attribute is treated as instantaneous, with a warning.
    """
    step_type = da.attrs.get("GRIB_stepType")
    if step_type is None:
        logger.warning("%s declares no GRIB_stepType; treating it as instantaneous", da.name)
    return step_type == "accum"


def amount_as_rate(ds: xr.Dataset, variable: str, per: str) -> xr.Dataset:
    """Label an amount over one interval as a rate per ``per`` (a udunits time unit, e.g. ``"d"``).

    ``mm`` accumulated over a day reads as the rate ``mm d-1``; the values are unchanged. Units that
    are missing, unparseable (``m of water equivalent``) or already a rate (``kg m-2 s-1``) are
    returned unchanged, since appending ``per`` to them would not describe the values.
    """
    from xclim.core.units import units2pint

    da = ds[variable]
    units = da.attrs.get("units")
    if not units:
        return ds
    try:
        quantity = 1 * units2pint(units)
    except Exception:  # noqa: BLE001 — not a unit we can reason about, so leave it as declared
        return ds
    # A rate has a per-time factor: a time unit (s, d, Hz) or a power (W = J/s). Its dimensionality
    # alone cannot tell: J m-2 is an amount, yet a joule carries time in its base dimensions.
    power = units2pint("W").dimensionality
    for name, _ in quantity.unit_items():
        dimensionality = units2pint(name).dimensionality
        if dimensionality == power or set(dimensionality) == {"[time]"}:
            return ds
    return ds.assign({variable: da.assign_attrs(units=f"{units} {per}-1")})
