"""Forecast cubes: two temporal axes instead of one.

A forecast is not a time series. The same date has as many values as there are runs that
predicted it, each better than the last, so a store that keeps only one of them cannot say
which — and a store keyed on the date being predicted overwrites yesterday's forecast every
time it refreshes. Forecast cubes therefore carry two axes:

    (reference_time, lead_time, y, x)

``reference_time`` is when the forecast was issued and ``lead_time`` how far ahead it reaches,
so the date a value describes is ``reference_time + lead_time`` — published as the
``forecast_valid_time`` auxiliary coordinate.

**Lead time rather than valid time** as the second axis, because keyed on valid time the array
is triangular: each run fills a different diagonal and a rolling archive is mostly fill value.
Keyed on lead time it is dense, every run having exactly the same number of steps.

**Neither axis is named ``t``.** ``get_time_dim`` resolves ``t``/``valid_time``/``time`` and a
dozen modules act on the result, all taking it to mean *the period this value describes*.
Neither forecast axis means that: keyed on the issue date, coverage would report when forecasts
were made rather than what they cover, and the map slider would scrub the day the forecast was
produced. So ``get_time_dim`` raises on a forecast cube and callers have to opt in — a loud
failure rather than a plausible wrong answer. Same reasoning as ``Cadence.IRREGULAR`` versus
returning ``None``.

For the same reason the valid-time coordinate is **not** called ``time`` or ``valid_time``:
``get_time_dim`` tests ``hasattr``, so either name would be returned as the time *dimension*
and callers would fail on ``ds.sizes[...]`` with a confusing error.

Consumers that only want "the current forecast" do not need to know any of this — see
:func:`latest_reference_view`, which collapses a cube to an ordinary ``(t, y, x)`` dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

REFERENCE_DIM = "reference_time"
"""Issue time: when the forecast was produced. CF ``forecast_reference_time``."""

LEAD_DIM = "lead_time"
"""How far ahead each step reaches, counted in :data:`LEAD_UNITS`. CF ``forecast_period``."""

VALID_COORD = "forecast_valid_time"
"""The date each value describes, as a ``(reference_time, lead_time)`` auxiliary coordinate."""

CF_STANDARD_NAMES = {
    REFERENCE_DIM: "forecast_reference_time",
    LEAD_DIM: "forecast_period",
}

LEAD_UNITS = {"day": "D", "month": "M"}
"""Units a lead may be counted in, mapped onto the matching numpy datetime64 unit.

A lead is not always in days. A medium-range run steps in days (GEFS: 35 of them); a seasonal
forecast steps in *months* (C3S: 6 of them), and a month is not a fixed duration, so it cannot be
a ``timedelta64`` at all — the arithmetic has to be calendar arithmetic. The unit is therefore a
property of the axis, read from its ``units`` attribute, rather than a constant in the code.
"""

DEFAULT_LEAD_UNIT = "day"
"""Assumed when the axis does not say. Days is what a store written before this was added holds."""


def is_forecast_cube(ds: Any) -> bool:
    """True when a dataset carries both forecast axes as dimensions.

    Both, deliberately: a cube with only ``lead_time`` is a single run that has lost its issue
    time, and one with only ``reference_time`` is a time series whose axis has been misnamed.
    Neither should be treated as a forecast archive.
    """
    sizes = getattr(ds, "sizes", None)
    if sizes is None:
        return False
    return REFERENCE_DIM in sizes and LEAD_DIM in sizes


def lead_unit(ds: xr.Dataset | xr.DataArray) -> str:
    """The unit ``lead_time`` counts in — a key of :data:`LEAD_UNITS`.

    Read from the axis's ``units`` attribute, which is where CF puts it, accepting the plural
    spelling a plugin naturally writes (``days``). An unrecognised unit raises rather than
    defaulting: silently treating months as days would put a six-month outlook inside one week.
    """
    declared = str(ds.coords[LEAD_DIM].attrs.get("units", DEFAULT_LEAD_UNIT)).strip().lower()
    singular = declared[:-1] if declared.endswith("s") else declared
    if singular not in LEAD_UNITS:
        raise ValueError(f"Unsupported {LEAD_DIM} unit {declared!r}; expected one of {sorted(LEAD_UNITS)}")
    return singular


def lead_values(ds: xr.Dataset | xr.DataArray) -> np.ndarray:
    """``lead_time`` as whole steps of :func:`lead_unit`, whichever way the store spelled it.

    A day-based lead is written as an integer with ``units: days``, the CF encoding for
    ``forecast_period`` — which xarray then decodes back to ``timedelta64`` on read. So the same
    axis is integers in a plugin and nanoseconds in a reader, and anything taking the raw values
    as a step count publishes ``lead_time: 86400000000000``. Both spellings are legitimate, so
    every consumer that wants a number of steps comes through here.
    """
    values = np.asarray(ds.coords[LEAD_DIM].values)
    if np.issubdtype(values.dtype, np.timedelta64):
        # Only a fixed-duration unit decodes to timedelta64 at all: xarray leaves ``months``
        # as integers because a month has no fixed length. So days is the only case here, and
        # a month axis that somehow arrived as a duration is corrupt rather than convertible.
        unit = lead_unit(ds)
        if unit != "day":
            raise ValueError(f"A {unit} lead cannot be a duration; {LEAD_DIM} should hold integers, not timedelta64")
        counts: np.ndarray = (values / np.timedelta64(1, "D")).astype("int64")
        return counts
    return values.astype("int64")


def valid_times_for(reference: np.ndarray, leads: np.ndarray, unit: str) -> np.ndarray:
    """``(reference, lead)`` grid of the dates being described, as ``datetime64[ns]``.

    Month leads are calendar arithmetic, not duration arithmetic: adding "one month" to 31 March
    is a different number of days than adding it to 30 April, so a fixed offset cannot express
    it. Numpy's ``datetime64[M]`` does the calendar step, and truncating the issue time to its
    month first is correct for a monthly forecast — the value describes the month, not the hour
    the run started.
    """
    issued = np.asarray(reference, dtype="datetime64[ns]")
    steps = np.asarray(leads, dtype="int64")
    if unit == "month":
        months = issued.astype("datetime64[M]")[:, None] + steps[None, :].astype("timedelta64[M]")
        grid: np.ndarray = months.astype("datetime64[ns]")
        return grid
    offsets = steps.astype(f"timedelta64[{LEAD_UNITS[unit]}]").astype("timedelta64[ns]")
    stamps: np.ndarray = issued[:, None] + offsets[None, :]
    return stamps


def valid_time(ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Return the valid-time coordinate, computing it if the store did not publish one.

    Recomputed rather than required, so a cube written by a plugin that omitted the auxiliary
    coordinate is still describable. Leads count in :func:`lead_unit`.
    """
    if VALID_COORD in getattr(ds, "coords", {}):
        return ds.coords[VALID_COORD]
    grid = valid_times_for(
        np.asarray(ds.coords[REFERENCE_DIM].values, dtype="datetime64[ns]"),
        lead_values(ds),
        lead_unit(ds),
    )
    return xr.DataArray(
        grid,
        dims=(REFERENCE_DIM, LEAD_DIM),
        coords={REFERENCE_DIM: ds.coords[REFERENCE_DIM], LEAD_DIM: ds.coords[LEAD_DIM]},
        name=VALID_COORD,
    )


def valid_time_bounds(ds: xr.Dataset | xr.DataArray) -> tuple[np.datetime64, np.datetime64]:
    """Earliest and latest date the cube says anything about.

    This — not the span of issue times — is a forecast store's temporal coverage. A run issued
    today covering ten days is not a store that ends today.
    """
    values = np.asarray(valid_time(ds).values, dtype="datetime64[ns]")
    return values.min(), values.max()


def latest_reference(ds: xr.Dataset | xr.DataArray) -> np.datetime64:
    """The most recent issue time in the cube — the current forecast."""
    latest = np.asarray(ds.coords[REFERENCE_DIM].values, dtype="datetime64[ns]").max()
    return np.datetime64(latest, "ns")


def latest_reference_view(ds: xr.Dataset, *, reference: Any = None, time_dim: str = "t") -> xr.Dataset:
    """Collapse a forecast cube to an ordinary ``(t, y, x)`` dataset for one issue time.

    The archive keeps every run; almost every consumer wants one. Selecting a single
    ``reference_time`` leaves valid time one-dimensional along ``lead_time``, so it can become
    the ``t`` axis — after which the DHIS2 and CHAP exports, the aggregation processes and the
    anomaly process work unchanged, with no forecast awareness at all.

    ``reference`` defaults to the latest run. The chosen issue time is kept as a scalar
    ``reference_time`` coordinate, so the result still records which forecast it came from.
    """
    if not is_forecast_cube(ds):
        return ds
    chosen = latest_reference(ds) if reference is None else np.datetime64(reference, "ns")
    one = ds.sel({REFERENCE_DIM: chosen})
    stamps = np.asarray(valid_time(ds).sel({REFERENCE_DIM: chosen}).values, dtype="datetime64[ns]")
    one = one.assign_coords({time_dim: (LEAD_DIM, stamps)}).swap_dims({LEAD_DIM: time_dim})
    # `lead_time` survives as a coordinate along `t`, which keeps the lead of each step
    # recoverable — a consumer can still tell a one-day forecast from a ten-day one.
    return one.drop_vars(VALID_COORD, errors="ignore")


def period_bounds_as_strings(ds: xr.Dataset, period_type: str) -> tuple[str, str]:
    """Valid-time bounds as dataset-native period id strings."""
    from open_climate_service.shared.time import datetime_to_period_string

    first, last = valid_time_bounds(ds)
    return (
        datetime_to_period_string(pd.Timestamp(first).to_pydatetime(), period_type),
        datetime_to_period_string(pd.Timestamp(last).to_pydatetime(), period_type),
    )
