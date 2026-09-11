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


def kelvin_difference_to_celsius(ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset:
    """Relabel a temperature *difference* from kelvin to degrees Celsius.

    A difference, not a temperature — an anomaly, a range, a bias. One kelvin of difference *is*
    one degree Celsius of difference, so the conversion is a relabel and the values are untouched.

    Deliberately not :func:`kelvin_to_celsius`: subtracting 273.15 from an anomaly of +1.4 K would
    report −271.75 °C. The two cases are a single character apart in a template, so they are
    separate named functions rather than a flag.
    """
    logger.info("Relabelling the '%s' difference from K to °C (values unchanged)", dataset["variable"])
    return _apply(ds, dataset, scale=1.0, offset=0.0, units="degC")


def metres_per_second_to_mm_per_day(ds: xr.Dataset, dataset: dict[str, Any]) -> xr.Dataset:
    """Convert the dataset variable from a rate in m s-1 to one in mm per day."""
    logger.info("Converting '%s' from m s-1 to mm/d", dataset["variable"])
    return _apply(ds, dataset, scale=1000.0 * 86400.0, offset=0.0, units="mm/d")
