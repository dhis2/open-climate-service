"""Time helpers shared across Open Climate Service modules."""

import calendar
import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ISO_DURATION_RE = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")

_PERIOD_TYPE_ISO_STEP = {
    "hourly": "PT1H",
    "daily": "P1D",
    # A week is exactly 7 days; "P7D" (not "P1W") so the map viewer's duration parser,
    # which handles H/D/M/Y but not W, can build the slider.
    "weekly": "P7D",
    "monthly": "P1M",
    "quarterly": "P3M",
    "yearly": "P1Y",
}

# Cadences that are real calendar periods but have no single ISO 8601 duration, because
# their members differ in length. A dekad is the first: day 1-10, 11-20, then 21 to the
# end of the month, so the third runs 8, 9, 10 or 11 days. "P10D" would be wrong for
# every third dekad of the year, which is why these carry no step at all.
_IRREGULAR_PERIOD_TYPES = frozenset({"dekadal"})

# Period types whose ids are not calendar instants. "climatology" ids are day-of-year
# ordinals (1..366), so asking for their cadence or ISO step is a category error.
_NON_TEMPORAL_PERIOD_TYPES = frozenset({"climatology"})

# Period types a dataset template may declare. Deliberately *not* derived from
# _PERIOD_TYPE_ISO_STEP: that map exists to give STAC a step for whatever is already in a
# store, and includes "quarterly", which none of datetime_to_period_string,
# normalize_period_string, numpy_datetime_to_period_string, next_period_string or
# _default_target_end implement. Deriving the two from one set would advertise quarterly as
# registerable and then fail at ingest, so they are kept apart.
SUPPORTED_PERIOD_TYPES = (
    frozenset({"hourly", "daily", "weekly", "monthly", "yearly"}) | _IRREGULAR_PERIOD_TYPES | _NON_TEMPORAL_PERIOD_TYPES
)


class Cadence(StrEnum):
    """How a dataset's periods are spaced, as far as the rest of the system must care.

    The distinction that matters is ``IRREGULAR`` versus ``UNKNOWN``. Both lack an ISO
    8601 step, but they mean opposite things: an irregular cadence is correctly declared
    and its spacing is genuinely variable, whereas an unknown one is a template we cannot
    honour. Collapsing them — as returning ``None`` from the step lookup for both would —
    makes a valid dekadal dataset indistinguishable from a misconfigured one.
    """

    REGULAR = "regular"
    """Fixed-length periods; ``period_type_to_iso_step`` yields a duration."""

    IRREGULAR = "irregular"
    """Known calendar cadence with variable-length periods; no duration exists."""

    NON_TEMPORAL = "non_temporal"
    """Ids are not calendar instants (day-of-year ordinals); cadence does not apply."""

    UNKNOWN = "unknown"
    """Unsupported ``period_type``. Reject at registration rather than degrade."""


def period_cadence(period_type: Any) -> Cadence:
    """Classify a dataset ``period_type``."""
    if not isinstance(period_type, str):
        return Cadence.UNKNOWN
    if period_type in _PERIOD_TYPE_ISO_STEP:
        return Cadence.REGULAR
    if period_type in _IRREGULAR_PERIOD_TYPES:
        return Cadence.IRREGULAR
    if period_type in _NON_TEMPORAL_PERIOD_TYPES:
        return Cadence.NON_TEMPORAL
    return Cadence.UNKNOWN


def period_type_to_iso_step(period_type: Any) -> str | None:
    """Map a dataset ``period_type`` to its ISO 8601 step, or None if it has none.

    Used as a fallback for the temporal cube dimension's ``step`` when a template does
    not declare ``extents.temporal.resolution`` (e.g. openEO ``save_result`` outputs).
    Without a step the map viewer cannot build a time slider.

    None covers three different situations — irregular, non-temporal and unsupported —
    so callers that need to tell them apart must use :func:`period_cadence`.
    """
    if not isinstance(period_type, str):
        return None
    return _PERIOD_TYPE_ISO_STEP.get(period_type)


def resolve_iso_period_step(dataset: dict[str, Any]) -> str | None:
    """Return the ISO 8601 duration step from ``extents.temporal.resolution``.

    Returns None if the field is absent or not a valid ISO 8601 duration, logging
    a warning in the latter case.
    """
    extents = dataset.get("extents")
    if not isinstance(extents, dict):
        return None
    temporal = extents.get("temporal")
    if not isinstance(temporal, dict):
        return None
    resolution = temporal.get("resolution")
    if not resolution:
        return None
    resolution_str = str(resolution)
    try:
        _iso_step_to_approx_hours(resolution_str)
    except ValueError:
        logger.warning("Invalid ISO 8601 duration in extents.temporal.resolution: %r", resolution_str)
        return None
    return resolution_str


def _iso_step_to_approx_hours(step: str) -> float:
    """Return the approximate duration in hours for an ISO 8601 duration string.

    Months and years use calendar averages (30.4375 days/month, 365.25 days/year).
    Raises ValueError for unrecognised formats.
    """
    m = _ISO_DURATION_RE.fullmatch(step)
    if not m:
        raise ValueError(f"Cannot parse ISO 8601 duration: '{step}'")
    years, months, weeks, days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
    result = (
        years * 365.25 * 24 + months * 30.4375 * 24 + weeks * 7 * 24 + days * 24 + hours + minutes / 60 + seconds / 3600
    )
    if result <= 0:
        raise ValueError(f"ISO 8601 duration '{step}' resolves to zero — cannot derive chunk size")
    return result


def time_chunk_for_iso_step(step: str) -> int:
    """Return a suitable zarr time chunk size for a given ISO 8601 duration step.

    Targets roughly one week of data for sub-daily steps, one month for daily/sub-weekly
    steps, and one year for weekly and coarser steps.  This keeps individual chunk files
    at a manageable size while covering a natural analysis window in one read.
    """
    hours = _iso_step_to_approx_hours(step)
    if hours < 24:
        return max(1, round(24 * 7 / hours))  # ~1 week
    if hours < 24 * 7:
        return max(1, round(24 * 30 / hours))  # ~1 month
    return max(1, round(24 * 365.25 / hours))  # ~1 year


_WEEKLY_PERIOD_PATTERN = re.compile(r"^(?P<year>\d{4})-W(?P<week>\d{2})$")


def _normalize_datetime_for_period(value: datetime) -> datetime:
    """Convert aware datetimes to UTC before deriving dataset-native periods."""
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    return value


def _coerce_numpy_datetime(value: object) -> datetime:
    """Convert a numpy or Python datetime-like scalar to a datetime."""
    if isinstance(value, datetime):
        return value
    np_value = np.datetime64(cast(Any, value))
    return datetime.fromisoformat(np.datetime_as_string(np_value, unit="s"))


def datetime_to_period_string(value: datetime, period_type: str) -> str:
    """Convert a datetime to the dataset-native period string format."""
    value = _normalize_datetime_for_period(value)
    if period_type == "hourly":
        return value.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")
    if period_type == "daily":
        return value.date().isoformat()
    if period_type == "dekadal":
        return dekad_start(value.date()).isoformat()
    if period_type == "weekly":
        iso_year, iso_week, _ = value.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if period_type == "monthly":
        return f"{value.year:04d}-{value.month:02d}"
    if period_type == "yearly":
        return str(value.year)
    raise ValueError(f"Unsupported period_type '{period_type}'")


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def utc_today() -> date:
    """Return the current UTC calendar date."""
    return utc_now().date()


def daily_period_ids(start: str | date, end: str | date, *, cutoff: str | date | None = None) -> list[str]:
    """Return ISO ``YYYY-MM-DD`` strings for every day in ``[start, end]`` inclusive.

    Accepts ISO date strings or ``date``/``datetime`` objects; returns an empty
    list when ``start`` is after ``end``. ``cutoff`` (a source's latest available
    day) caps ``end`` so daily plugins express their availability clamp in one call
    rather than re-deriving it. Shared by the daily streaming plugins so they only
    own the cutoff, not the day enumeration.
    """
    current = _as_date(start)
    last = _as_date(end)
    if cutoff is not None:
        last = min(last, _as_date(cutoff))
    out: list[str] = []
    while current <= last:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


def monthly_period_ids(start: str | date, end: str | date, *, cutoff: str | date | None = None) -> list[str]:
    """Return ``YYYY-MM`` strings for every month in ``[start, end]`` inclusive.

    The monthly counterpart of :func:`daily_period_ids`, with the same contract: accepts ISO
    strings (``2024-03`` or ``2024-03-17``) or ``date``/``datetime`` objects, returns an
    empty list when ``start`` is after ``end``, and ``cutoff`` caps ``end`` so a plugin owns
    only its availability clamp rather than the month arithmetic.

    Any day within a month selects that month, so a caller need not normalise to the first.

    Raises:
        ValueError: for a malformed or out-of-range value, matching how ``daily_period_ids``
            fails fast via ``date.fromisoformat``. Strictness is not cosmetic here: a month
            outside 1..12 used to walk the increment past its 12 → 1 wrap and loop forever.
    """

    def _as_year_month(value: str | date) -> tuple[int, int]:
        if isinstance(value, datetime):
            return value.year, value.month
        if isinstance(value, date):
            return value.year, value.month
        # Reuse the canonical monthly parser rather than slicing: it accepts YYYY-MM or any
        # ISO date within the month and rejects everything else with a clear message.
        canonical = normalize_period_string(str(value), "monthly")
        return int(canonical[:4]), int(canonical[5:7])

    year, month = _as_year_month(start)
    last_year, last_month = _as_year_month(end)
    if cutoff is not None:
        cutoff_year, cutoff_month = _as_year_month(cutoff)
        if (cutoff_year, cutoff_month) < (last_year, last_month):
            last_year, last_month = cutoff_year, cutoff_month

    out: list[str] = []
    while (year, month) <= (last_year, last_month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


DEKAD_START_DAYS = (1, 11, 21)
"""Day-of-month on which each dekad begins, per the openEO ``dekad`` definition."""


def _as_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def dekad_start(value: str | date) -> date:
    """Return the first day of the dekad containing ``value``.

    Truncation for the dekadal cadence: 15 January lands in the second dekad, so it
    normalises to 11 January.
    """
    day = _as_date(value)
    start_day = max(d for d in DEKAD_START_DAYS if d <= day.day)
    return day.replace(day=start_day)


def dekad_bounds(value: str | date) -> tuple[date, date]:
    """Return the inclusive ``(first_day, last_day)`` of the dekad containing ``value``.

    The third dekad of a month ends on the month's last day, so its length varies from
    8 (February, common year) to 11 days. This is the only honest expression of a dekad's
    extent — there is no duration that describes all three — and it is what CF
    ``time_bnds`` should be written from.
    """
    start = dekad_start(value)
    if start.day == DEKAD_START_DAYS[-1]:
        last_day = calendar.monthrange(start.year, start.month)[1]
        return start, start.replace(day=last_day)
    return start, start.replace(day=start.day + 9)


def dekad_period_ids(start: str | date, end: str | date, *, cutoff: str | date | None = None) -> list[str]:
    """Return ``YYYY-MM-DD`` ids for every dekad overlapping ``[start, end]`` inclusive.

    Ids are the dekad's first day, so they sort chronologically and parse as plain dates.
    A partially covered dekad at either end is included — the dekad is the smallest unit
    the data has, so overlapping it means it is needed. The dekadal counterpart of
    :func:`daily_period_ids`, including the ``cutoff`` availability clamp.
    """
    requested_start = _as_date(start)
    last = _as_date(end)
    if cutoff is not None:
        last = min(last, _as_date(cutoff))
    # Tested against the *requested* start, not the snapped one: an inverted or
    # cutoff-exhausted range covers nothing, and snapping back into the containing dekad
    # must not turn it into a hit. `daily_period_ids` returns [] for the same input.
    if requested_start > last:
        return []
    first = dekad_start(requested_start)
    out: list[str] = []
    current = first
    while current <= last:
        out.append(current.isoformat())
        _, dekad_end = dekad_bounds(current)
        current = dekad_end + timedelta(days=1)
    return out


def next_period_string(period: str, period_type: str) -> str:
    """Return the dataset-native period immediately following ``period``."""
    if period_type == "hourly":
        timestamp = parse_hourly_period_string(period)
        return datetime_to_period_string(timestamp + timedelta(hours=1), period_type)
    if period_type == "daily":
        return (date.fromisoformat(period) + timedelta(days=1)).isoformat()
    if period_type == "dekadal":
        _, end = dekad_bounds(period)
        return (end + timedelta(days=1)).isoformat()
    if period_type == "weekly":
        current = parse_period_string_to_datetime(period).date()
        return datetime_to_period_string(datetime.combine(current + timedelta(days=7), time(0)), period_type)
    if period_type == "monthly":
        current = date.fromisoformat(f"{period}-01")
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        return f"{year:04d}-{month:02d}"
    if period_type == "yearly":
        return str(int(period) + 1)
    if period_type == "climatology":
        return str(int(period) + 1)
    raise ValueError(f"Unsupported period_type '{period_type}'")


def parse_hourly_period_string(value: str) -> datetime:
    """Parse a dataset-native hourly period string or full ISO datetime."""
    if len(value) == 13:
        return datetime.strptime(value, "%Y-%m-%dT%H")
    return datetime.fromisoformat(value)


def parse_weekly_period_string(value: str) -> datetime:
    """Parse a dataset-native weekly period string or full ISO datetime."""
    match = _WEEKLY_PERIOD_PATTERN.fullmatch(value)
    if match is not None:
        iso_year = int(match.group("year"))
        iso_week = int(match.group("week"))
        return datetime.combine(date.fromisocalendar(iso_year, iso_week, 1), datetime.min.time())
    return datetime.fromisoformat(value)


def normalize_period_string(value: str, period_type: str) -> str:
    """Normalize an input period string to the dataset-native period format."""
    if period_type == "hourly":
        try:
            return datetime_to_period_string(parse_hourly_period_string(value), period_type)
        except ValueError as exc:
            raise ValueError(f"Invalid hourly period '{value}'; expected YYYY-MM-DDTHH or ISO datetime") from exc
    if period_type == "daily":
        try:
            return datetime_to_period_string(datetime.fromisoformat(value), period_type)
        except ValueError as exc:
            raise ValueError(f"Invalid daily period '{value}'; expected YYYY-MM-DD or ISO datetime") from exc
    if period_type == "dekadal":
        try:
            return datetime_to_period_string(datetime.fromisoformat(value), period_type)
        except ValueError as exc:
            raise ValueError(
                f"Invalid dekadal period '{value}'; expected YYYY-MM-DD (any day within the "
                "dekad, normalised to the 1st, 11th or 21st) or ISO datetime"
            ) from exc
    if period_type == "weekly":
        try:
            return datetime_to_period_string(parse_weekly_period_string(value), period_type)
        except ValueError as exc:
            raise ValueError(f"Invalid weekly period '{value}'; expected YYYY-Www or ISO datetime") from exc
    if period_type == "monthly":
        try:
            if len(value) == 7:
                datetime.fromisoformat(f"{value}-01")
                return value
            return datetime_to_period_string(datetime.fromisoformat(value), period_type)
        except ValueError as exc:
            raise ValueError(f"Invalid monthly period '{value}'; expected YYYY-MM or ISO datetime") from exc
    if period_type == "yearly":
        try:
            if len(value) == 4:
                int(value)
                return value
            return datetime_to_period_string(datetime.fromisoformat(value), period_type)
        except ValueError as exc:
            raise ValueError(f"Invalid yearly period '{value}'; expected YYYY or ISO datetime") from exc
    if period_type == "climatology":
        # Non-temporal (day-of-year) dataset: period ids are ordinal dayofyear values,
        # not dates, and the plugin enumerates 1..366 independent of the request range —
        # so there is nothing to normalize beyond trimming incidental whitespace.
        return value.strip()
    raise ValueError(f"Unsupported period_type '{period_type}'")


def parse_period_string_to_datetime(value: str) -> datetime:
    """Parse a dataset-native period string to a UTC datetime."""
    normalized = value.strip()
    if _WEEKLY_PERIOD_PATTERN.fullmatch(normalized) is not None:
        return parse_weekly_period_string(normalized).replace(tzinfo=UTC)
    if "T" not in normalized:
        if len(normalized) == 4:
            normalized = f"{normalized}-01-01T00:00:00"
        elif len(normalized) == 7:
            normalized = f"{normalized}-01T00:00:00"
        else:
            normalized = f"{normalized}T00:00:00"

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def numpy_datetime_to_period_string(datetimes: np.ndarray[Any, Any], period_type: str) -> np.ndarray[Any, Any]:
    """Convert an array of numpy datetimes to truncated period strings."""
    if period_type == "weekly":
        dt_index = pd.DatetimeIndex(np.atleast_1d(np.asarray(datetimes, dtype="datetime64[ns]")))
        iso = dt_index.isocalendar()
        strings = iso["year"].astype(str).str.zfill(4) + "-W" + iso["week"].astype(str).str.zfill(2)
        return cast(np.ndarray[Any, Any], strings.to_numpy().astype("U8"))

    if period_type == "dekadal":
        # Snapping, not truncation: a stored timestamp anywhere inside a dekad must yield
        # that dekad's id. Truncating to 10 characters would only be correct for stores
        # whose timestamps already sit on the 1st/11th/21st.
        dt_index = pd.DatetimeIndex(np.atleast_1d(np.asarray(datetimes, dtype="datetime64[ns]")))
        day = dt_index.day.to_numpy()
        start_day = np.select([day >= 21, day >= 11], [21, 11], default=1)
        dekad_ids = [
            f"{year:04d}-{month:02d}-{start:02d}"
            for year, month, start in zip(dt_index.year, dt_index.month, start_day, strict=True)
        ]
        return cast(np.ndarray[Any, Any], np.asarray(dekad_ids, dtype="U10"))

    lengths = {"hourly": 13, "daily": 10, "monthly": 7, "yearly": 4}
    return np.datetime_as_string(datetimes, unit="s").astype(f"U{lengths[period_type]}")


# ---------------------------------------------------------------------------
# Period reachability for exports (CLIM-1302)
# ---------------------------------------------------------------------------

EXPORT_PERIOD_TYPES = ("daily", "weekly", "monthly", "quarterly", "yearly")
"""The DHIS2 periods an export can emit, in the dataset ``period_type`` vocabulary."""

_EXPORT_PERIOD_ALIASES = {
    "day": "daily",
    "daily": "daily",
    "week": "weekly",
    "weekly": "weekly",
    "month": "monthly",
    "monthly": "monthly",
    "quarter": "quarterly",
    "quarterly": "quarterly",
    "year": "yearly",
    "yearly": "yearly",
}

_OPENEO_PERIOD_TO_CADENCE = {
    "hour": "hourly",
    "day": "daily",
    "week": "weekly",
    "dekad": "dekadal",
    "month": "monthly",
    "year": "yearly",
}
_CADENCE_TO_OPENEO_PERIOD = {cadence: period for period, cadence in _OPENEO_PERIOD_TO_CADENCE.items()}

# Which destination periods are unions of whole source periods. Weekly tiles nothing: ISO
# weeks straddle months, quarters and years. Dekads tile months (three per month) and so
# everything built from months. Hours tile days, and days tile ISO weeks.
_TILES: dict[str, frozenset[str]] = {
    # Calendar quarters are intentionally absent as destinations. The standard openEO
    # aggregate_temporal_period vocabulary has no calendar-quarter period, so advertising
    # those transitions would accept a declaration that no instrumented process can satisfy.
    "hourly": frozenset({"daily", "weekly", "monthly", "yearly"}),
    "daily": frozenset({"weekly", "monthly", "yearly"}),
    "dekadal": frozenset({"monthly", "yearly"}),
    "weekly": frozenset(),
    "monthly": frozenset({"yearly"}),
    "quarterly": frozenset({"yearly"}),
    "yearly": frozenset(),
}
_CADENCE_RANK = {"hourly": 0, "daily": 1, "weekly": 2, "dekadal": 2, "monthly": 3, "quarterly": 4, "yearly": 5}


CADENCE_ATTR = "ocs_cadence"
"""Attribute a cube carries to say what cadence its time axis is at.

Stamped from the dataset template when a collection is loaded, rewritten by every temporal
aggregation, and carried through spatial aggregation. It describes the data at that node of
the graph, which neither the loaded dataset's declaration nor the spacing of a possibly sparse
axis can do on their own.
"""


def cadence_of(data: Any) -> str | None:
    """The consistent cadence a cube says it is at, or None when absent or ambiguous.

    Converting an xarray DataArray to a Dataset keeps the array attributes on its data
    variable. Recovering a cadence shared by all stamped variables makes that harmless while
    refusing to guess when variables genuinely carry different cadences.
    """
    attrs = getattr(data, "attrs", None)
    value = attrs.get(CADENCE_ATTR) if isinstance(attrs, dict) else None
    if isinstance(value, str) and value:
        return value
    variables = getattr(data, "data_vars", None)
    if variables is None:
        return None
    cadences = {cadence for variable in variables.values() if (cadence := cadence_of(variable)) is not None}
    return next(iter(cadences)) if len(cadences) == 1 else None


def stamp_cadence(data: Any, cadence: str | None) -> Any:
    """Set the cadence a cube is at, in place, and return it; a None cadence clears it."""
    attrs = getattr(data, "attrs", None)
    if isinstance(attrs, dict):
        if cadence is None:
            attrs.pop(CADENCE_ATTR, None)
        else:
            attrs[CADENCE_ATTR] = cadence
    return data


class Reachability(StrEnum):
    """How an export period can be produced from a dataset cadence."""

    PASS_THROUGH = "pass_through"
    """Same cadence: each stored timestamp is one destination period."""

    AGGREGATE = "aggregate"
    """Coarser, and every destination period is a union of whole source periods."""

    UNREACHABLE = "unreachable"
    """Finer than the source, or not tiled by it; no computation produces it honestly."""


def normalise_export_period(value: Any) -> str | None:
    """Map ``day``/``daily``, ``month``/``monthly`` and the rest to the dataset vocabulary."""
    if not isinstance(value, str):
        return None
    return _EXPORT_PERIOD_ALIASES.get(value.strip().lower())


def openeo_period_to_cadence(period: Any) -> str | None:
    """The dataset cadence an openEO ``aggregate_temporal_period`` period produces, if any."""
    return _OPENEO_PERIOD_TO_CADENCE.get(period) if isinstance(period, str) else None


def cadence_to_openeo_period(cadence: str) -> str | None:
    """The openEO period that aggregates to ``cadence``; ``quarterly`` has none."""
    return _CADENCE_TO_OPENEO_PERIOD.get(cadence)


def period_reachability(source: str | None, destination: str) -> tuple[Reachability, str]:
    """Decide whether ``destination`` can be produced from ``source``, and say why.

    ``source`` is a dataset cadence (``hourly`` ... ``yearly``, ``dekadal``, ``climatology``);
    ``destination`` an export period in the same vocabulary. The second value is a sentence
    for an error message or a manifest, never empty.
    """
    if source is None:
        return Reachability.UNREACHABLE, "the source cadence is unknown"
    if source == destination:
        return Reachability.PASS_THROUGH, f"{source} data is exported as {destination} periods as is"
    if destination in _TILES.get(source, frozenset()):
        return Reachability.AGGREGATE, f"every {destination} period is a union of whole {source} periods"
    if source == "climatology":
        return Reachability.UNREACHABLE, "a climatology is indexed by day of year, not by calendar date"
    if source == "weekly":
        return Reachability.UNREACHABLE, "ISO weeks straddle months, quarters and years"
    if source not in _CADENCE_RANK or destination not in _CADENCE_RANK:
        return Reachability.UNREACHABLE, f"{source!r} cannot be related to {destination!r}"
    if _CADENCE_RANK[destination] < _CADENCE_RANK[source]:
        return Reachability.UNREACHABLE, f"{destination} is finer than {source} and cannot be derived from it"
    return Reachability.UNREACHABLE, f"{destination} periods are not unions of whole {source} periods"


def is_dekadal_axis(t_values: Any) -> bool:
    """Whether every timestamp starts a dekad, at dekadal spacing.

    Two conditions, and both are needed: every timestamp falls on the 1st, 11th or 21st,
    because a regular 10-day series on any other day is not dekadal; and some adjacent pair
    is 8 to 11 days apart, because the day-of-month test alone accepts a monthly axis. Tested
    on the minimum rather than the median so an axis with missing dekads is still recognised.
    """
    import numpy as np
    import pandas as pd

    stamps = pd.DatetimeIndex(np.unique(np.asarray(t_values, dtype="datetime64[ns]")))
    if not set(stamps.day) <= set(DEKAD_START_DAYS):
        return False
    gaps = np.diff(stamps.values).astype("timedelta64[D]").astype(int)
    return bool(gaps.size and 8 <= gaps.min() <= 11)


def infer_cadence(t_values: Any) -> str | None:
    """Infer a dataset cadence from a sorted time axis, or None below two timestamps.

    The median step decides, except for dekads, which are recognised by their structure:
    a dekad starts on the 1st, 11th or 21st by definition, so that test is exact where a
    median is a guess. There is deliberately no quarterly branch: it is not an ingest cadence.
    """
    import numpy as np

    values = np.sort(np.asarray(t_values, dtype="datetime64[ns]"))
    if values.size < 2:
        return None
    deltas = np.diff(values).astype("timedelta64[s]").astype(float)
    median_seconds = float(np.median(deltas))
    if median_seconds <= 3600:
        return "hourly"
    if median_seconds <= 86400:
        return "daily"
    if is_dekadal_axis(values):
        return "dekadal"
    if median_seconds <= 8 * 86400:
        return "weekly"
    if median_seconds <= 32 * 86400:
        return "monthly"
    if 330 * 86400 <= median_seconds <= 370 * 86400:
        return "yearly"
    return None


def export_period_label(timestamp: Any, destination: str) -> str:
    """The DHIS2 period string a timestamp falls in: 20250131, 2025W05, 202501, 2025Q1, 2025."""
    import pandas as pd

    stamp = pd.Timestamp(timestamp)
    if destination == "daily":
        return stamp.strftime("%Y%m%d")
    if destination == "weekly":
        iso = stamp.isocalendar()
        return f"{iso.year}W{iso.week:02d}"
    if destination == "monthly":
        return stamp.strftime("%Y%m")
    if destination == "quarterly":
        return f"{stamp.year}Q{stamp.quarter}"
    if destination == "yearly":
        return stamp.strftime("%Y")
    raise ValueError(f"Unsupported export period {destination!r}")


def period_label_start(label: str, cadence: str) -> Any:
    """The first instant of a DHIS2 period string, so a label can be relabelled at a coarser cadence."""
    import datetime as dt

    import pandas as pd

    if cadence == "weekly":
        year, week = label.split("W")
        return pd.Timestamp(dt.date.fromisocalendar(int(year), int(week), 1))
    if cadence == "quarterly":
        year, quarter = label.split("Q")
        return pd.Timestamp(year=int(year), month=3 * (int(quarter) - 1) + 1, day=1)
    pattern = {"daily": "%Y%m%d", "monthly": "%Y%m", "yearly": "%Y"}[cadence]
    return pd.Timestamp(dt.datetime.strptime(label, pattern))


def _destination_bounds(timestamp: Any, destination: str) -> tuple[Any, Any]:
    """Inclusive start and exclusive end of the destination period holding ``timestamp``."""
    import pandas as pd

    stamp = pd.Timestamp(timestamp).normalize()
    if destination == "daily":
        start = stamp
        return start, start + pd.Timedelta(days=1)
    if destination == "weekly":
        start = stamp - pd.Timedelta(days=int(stamp.weekday()))
        return start, start + pd.Timedelta(days=7)
    if destination == "monthly":
        start = stamp.replace(day=1)
        return start, start + pd.offsets.MonthBegin(1)
    if destination == "quarterly":
        start = pd.Timestamp(year=stamp.year, month=3 * (stamp.quarter - 1) + 1, day=1)
        return start, start + pd.offsets.MonthBegin(3)
    if destination == "yearly":
        start = pd.Timestamp(year=stamp.year, month=1, day=1)
        return start, start + pd.offsets.YearBegin(1)
    raise ValueError(f"Unsupported export period {destination!r}")


def expected_source_steps(start: Any, end: Any, source: str) -> int:
    """How many whole ``source`` periods lie in ``[start, end)``."""
    import pandas as pd

    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    if source == "hourly":
        return int((finish - begin) / pd.Timedelta(hours=1))
    if source == "daily":
        return int((finish - begin) / pd.Timedelta(days=1))
    if source == "dekadal":
        return 3 * len(pd.period_range(begin, finish - pd.Timedelta(days=1), freq="M"))
    if source == "monthly":
        return len(pd.period_range(begin, finish - pd.Timedelta(days=1), freq="M"))
    if source == "quarterly":
        return len(pd.period_range(begin, finish - pd.Timedelta(days=1), freq="Q"))
    raise ValueError(f"{source!r} periods cannot be counted inside a coarser period")


def incomplete_destination_periods(t_values: Any, source: str, destination: str) -> list[str]:
    """Destination periods for which the axis holds fewer than every whole source period.

    Timestamps are taken as source period starts, which is how stores and
    ``aggregate_temporal_period`` label them. Returns DHIS2 period strings, in order, so an
    export can refuse or drop exactly those.
    """
    import numpy as np
    import pandas as pd

    stamps = pd.DatetimeIndex(np.unique(np.asarray(t_values, dtype="datetime64[ns]")))
    if stamps.empty:
        return []
    # One pass: label every stamp, count per label, compare with the label's expected count.
    labels = pd.Series([export_period_label(stamp, destination) for stamp in stamps], index=stamps)
    present = labels.value_counts()
    first_stamp = labels.reset_index().drop_duplicates(subset=0).set_index(0)["index"]
    incomplete: list[str] = []
    for label in labels.drop_duplicates():
        start, end = _destination_bounds(first_stamp[label], destination)
        if int(present[label]) < expected_source_steps(start, end, source):
            incomplete.append(str(label))
    return incomplete
