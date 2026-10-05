"""Built-in openEO process: aggregate a dekadal cube to months or ISO weeks.

Dekads do not have equal lengths — day 1-10, 11-20, then 21 to the end of the month, so
the third runs 8, 9, 10 or 11 days — which makes a plain ``mean`` the wrong reduction for
an intensive quantity. Three dekads tile a calendar month exactly, but weighting them
equally over-weights February's 8-day dekad by 4.8 percentage points, a 14% relative error
on that dekad's contribution. The bias is systematic rather than noise: it always favours
short third dekads, so an unweighted monthly series carries a seasonal artefact that
tracks month length.

This process weights each contributing dekad by the number of days it shares with the
target period, which is exact under the only model the data supports — that a dekad's
value describes its whole span uniformly. Weights come from the real calendar via
``dekad_bounds``, so they are recomputed per year rather than assumed; that matters for
ISO weeks, which never align with dekads and have no fixed mapping at all.

Deliberately *not* interpolation. Splining through dekad midpoints does not conserve the
annual total, invents sub-dekad structure the sensor never observed (day-scale GPP
variability is driven by cloud and rainfall, absent from a 10-day composite), and can
undershoot below zero for a non-negative quantity. If a smooth daily curve is ever needed,
the defensible route is mean-preserving interpolation, not a plain spline.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from open_climate_service.process import process
from open_climate_service.shared.time import DEKAD_START_DAYS, dekad_bounds

logger = logging.getLogger(__name__)

_PERIODS = ("month", "week")
_METHODS = ("mean", "sum")

# Units that mark a per-day rate. Summing these would be meaningless — a month's worth of
# "gC/m²/day" values added together is not a monthly total in any unit.
_PER_DAY_MARKERS = ("/day", "/d", " d-1", "d-1")


def _target_start(day: date, period: str) -> date:
    """First day of the target period containing ``day``."""
    if period == "month":
        return day.replace(day=1)
    return day - timedelta(days=day.weekday())  # ISO week starts Monday


def _target_end(start: date, period: str) -> date:
    """Last day (inclusive) of the target period beginning at ``start``."""
    if period == "month":
        next_month = start.replace(day=28) + timedelta(days=4)
        return next_month.replace(day=1) - timedelta(days=1)
    return start + timedelta(days=6)


def _overlap_days(a: tuple[date, date], b: tuple[date, date]) -> int:
    """Number of days shared by two inclusive date ranges."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0, (hi - lo).days + 1)


def _dekad_dates(data: xr.DataArray, time_dim: str) -> list[date]:
    """Validate that every timestep is a dekad start and return them as dates.

    Guarding rather than trusting: run on a daily or monthly cube the weighting would
    produce plausible-looking nonsense, so refuse anything that is not dekadal.

    Two checks, because the first is not sufficient. Every dekad starts on the 1st, 11th or
    21st — but so does every *month*, so a monthly cube passes a per-timestamp day test and is
    then spread across the first dekad of each month: with ``period="week"`` that yields a
    fully populated weekly series covering only the first ten days of each month. A real
    dekadal series visits the 11th or the 21st, so requiring that separates the two.

    Gap-based detection cannot: January 1 to February 1 is 31 days, which is exactly three
    dekads, so a monthly series is indistinguishable from a dekadal one with two missing
    dekads. Missing dekads are legitimate here (a partially covered period is computed from
    what exists), so the day-of-month set is the only discriminator that does not also reject
    valid input.
    """
    stamps = pd.DatetimeIndex(np.asarray(data[time_dim].values, dtype="datetime64[ns]"))
    days = [d.date() for d in stamps]
    offenders = sorted({d.isoformat() for d in days if d.day not in DEKAD_START_DAYS})
    if offenders:
        raise ValueError(
            "aggregate_dekads expects a dekadal cube: every timestep must start a dekad "
            f"(day {', '.join(str(d) for d in DEKAD_START_DAYS)}). Offending timesteps: "
            f"{', '.join(offenders[:5])}{' …' if len(offenders) > 5 else ''}"
        )
    if len(days) > 1 and all(d.day == DEKAD_START_DAYS[0] for d in days):
        raise ValueError(
            "aggregate_dekads expects a dekadal cube, but every timestep falls on the 1st "
            f"({len(days)} of them, {days[0].isoformat()} to {days[-1].isoformat()}), which is "
            "monthly, quarterly or yearly spacing rather than dekadal. A dekadal series also "
            "visits the 11th and 21st. Aggregate this with aggregate_temporal_period instead."
        )
    return days


@process(
    summary="Aggregate a dekadal cube to months or ISO weeks, weighted by day overlap",
    parameters={
        "data": {"description": "Dekadal data cube (timesteps on the 1st, 11th and 21st)."},
        "period": {"description": "Target period: 'month' (default) or 'week'."},
        "method": {
            "description": (
                "'mean' (default) for a per-day rate — the day-weighted average daily value, "
                "in the same units. 'sum' for a per-dekad total — reallocated by day overlap "
                "into a target-period total."
            )
        },
    },
)
def aggregate_dekads(
    data: xr.DataArray,
    period: str = "month",
    method: str = "mean",
) -> xr.DataArray:
    """Aggregate dekads to ``period``, weighting each by the days it shares with the target.

    ``mean`` treats each value as a mean daily rate and returns the day-weighted average,
    so the units are unchanged. ``sum`` treats each value as a total accumulated over its
    dekad, converts it to a daily rate, and sums over the target's days — exact regardless
    of dekad length.

    Both are NaN-aware per pixel — a dekad missing over part of the grid does not poison the
    whole target period there — but they handle the gap differently, because the right answer
    differs:

    * ``mean`` **renormalises**: the missing dekad's weight leaves the denominator, so the
      result is the day-weighted mean of the dekads that do exist. A rate estimated from two
      dekads is still a rate.
    * ``sum`` **omits**: the missing dekad contributes nothing and the rest are not scaled up,
      so the result is a *partial* total. Renormalising would be extrapolation — inventing
      accumulation that was never observed — so the total reports only what is there.

    A pixel with no data at all in a target period stays NaN rather than becoming 0.

    The same asymmetry applies to a partially covered period at either end of the record,
    where only some of its dekads were loaded: the ``mean`` is well defined, while a ``sum``
    is a partial total. That case is detectable from the weights alone rather than per pixel,
    so it is logged — silently returning a month's total computed from one dekad is the kind
    of number that gets published.
    """
    if period not in _PERIODS:
        raise ValueError(f"Unknown period '{period}'; expected one of {list(_PERIODS)}")
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'; expected one of {list(_METHODS)}")

    time_dim = next((d for d in ("t", "time", "valid_time") if d in data.dims), None)
    if time_dim is None:
        raise ValueError("aggregate_dekads requires a temporal dimension named 't', 'time' or 'valid_time'")

    units = str(data.attrs.get("units", ""))
    if method == "sum" and any(marker in units for marker in _PER_DAY_MARKERS):
        logger.warning(
            "aggregate_dekads called with method='sum' on units %r, which look like a per-day rate. "
            "Adding daily rates does not produce a total; method='mean' gives the average daily value.",
            units,
        )

    source_days = _dekad_dates(data, time_dim)
    spans = [dekad_bounds(day) for day in source_days]

    # Group source steps by the target period they touch. A dekad can straddle two ISO
    # weeks, so this is a many-to-many mapping, not a partition.
    contributions: dict[date, list[tuple[int, int]]] = {}
    for index, span in enumerate(spans):
        day = span[0]
        while day <= span[1]:
            start = _target_start(day, period)
            overlap = _overlap_days(span, (start, _target_end(start, period)))
            if overlap:
                contributions.setdefault(start, []).append((index, overlap))
            day = _target_end(start, period) + timedelta(days=1)

    slices: list[xr.DataArray] = []
    incomplete: list[str] = []
    for start in sorted(contributions):
        indices = [i for i, _ in contributions[start]]
        weights = [w for _, w in contributions[start]]
        # Only `sum` is misread when a period is short of dekads: it yields a partial total
        # that looks like a whole one. A `mean` over fewer dekads is still a valid mean.
        period_days = (_target_end(start, period) - start).days + 1
        if method == "sum" and sum(weights) < period_days:
            incomplete.append(start.isoformat())
        subset = data.isel({time_dim: indices})
        lengths = [(spans[i][1] - spans[i][0]).days + 1 for i in indices]
        # Per-day rate: `mean` inputs already are one, `sum` inputs are a dekad total.
        daily = subset if method == "mean" else subset / xr.DataArray(lengths, dims=[time_dim])
        w = xr.DataArray(np.asarray(weights, dtype="float64"), dims=[time_dim])
        finite = daily.notnull()
        numerator = (daily.fillna(0) * w).sum(time_dim)
        effective = (w * finite).sum(time_dim)
        aggregated = numerator if method == "sum" else numerator / effective
        # All-NaN pixels would divide by zero above; keep them NaN rather than 0 or inf.
        aggregated = aggregated.where(effective > 0)
        slices.append(aggregated.assign_coords({time_dim: np.datetime64(start, "ns")}).expand_dims(time_dim))

    if incomplete:
        logger.warning(
            "aggregate_dekads: %d of %d target period(s) are not fully covered by the loaded "
            "dekads, so method='sum' reports a partial total for them (%s%s). Widen "
            "temporal_extent to cover whole periods, or use method='mean'.",
            len(incomplete),
            len(contributions),
            ", ".join(incomplete[:5]),
            " …" if len(incomplete) > 5 else "",
        )

    if not slices:
        raise ValueError("aggregate_dekads produced no target periods — the input cube has no timesteps")

    # Execution evidence, as aggregate_temporal_period leaves it, so an export can verify that
    # the period it emits came from this step with the reducer it declares and refuse or drop
    # the target periods a `sum` only partly covered (CLIM-1302).
    from open_climate_service.shared.provenance import (
        observe_temporal_aggregation,
        record_incomplete_periods,
        record_reduction,
    )
    from open_climate_service.shared.time import export_period_label, stamp_cadence

    cadence = {"month": "monthly", "week": "weekly"}.get(period)
    with observe_temporal_aggregation(cadence):
        record_reduction(method)
        if cadence is not None:
            record_incomplete_periods([export_period_label(np.datetime64(start), cadence) for start in incomplete])

    out = xr.concat(slices, dim=time_dim)
    out.attrs = dict(data.attrs)
    stamp_cadence(out, cadence)
    # Record how this was derived: a consumer must be able to tell a day-weighted
    # aggregate from an observation, and from an unweighted mean.
    interval = "10 day"
    comment = f"day-weighted {method} from dekads"
    out.attrs["cell_methods"] = f"time: {method} (interval: {interval} comment: {comment})"
    return out
