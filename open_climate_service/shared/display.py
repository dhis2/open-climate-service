"""The value range a dataset is coloured over (CLIM-1439).

Each end of a template's `display.range` is a number or the name of a measured statistic:
`[0, 20]`, `[p2, p98]`, or `[0, p98]` to keep zero while ignoring the record storm. Without a
range it is `[p2, p98]`, so a template no longer has to guess values that suit every country
and period, and one outlier does not wash out the rest. A statistic not measured yet, on a
record from before it existed, falls back to `[min, max]` until the startup backfill adds it.

`display.midpoint` centres a range with a measured end on a value, so a diverging colormap keeps
its neutral colour on the neutral value: zero for an anomaly, where blue has to mean below
normal. A range of two numbers is used as written: its ends are the author's choice, and one who
wants it centred declares it symmetric.

Decided here once, for the map viewer's STAC render hints, the dataset and data source pages,
and openEO PNG output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from open_climate_service.ingestions.schemas import ArtifactStatistics

STATISTICS = frozenset(ArtifactStatistics.model_fields)
DEFAULT_RANGE = ("p2", "p98")
_UNMEASURED_RANGE = ("min", "max")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _is_end(value: Any) -> bool:
    return _number(value) is not None or (isinstance(value, str) and value in STATISTICS)


def range_spec(declared: Any) -> tuple[Any, Any] | None:
    """*declared* as two ends, each a number or a statistic name, or None if it is not that."""
    if isinstance(declared, list | tuple) and len(declared) == 2 and all(_is_end(end) for end in declared):
        return declared[0], declared[1]
    return None


def check_display(display: Any) -> None:
    """Refuse a `display.range` or `display.midpoint` that could never be resolved, saying why."""
    if not isinstance(display, Mapping):
        return
    if "range" in display and range_spec(display["range"]) is None:
        raise ValueError(
            f"display.range must be two ends, each a number or one of {', '.join(sorted(STATISTICS))}; "
            f"got {display['range']!r}"
        )
    if "midpoint" in display and _number(display["midpoint"]) is None:
        raise ValueError(f"display.midpoint must be a number, got {display['midpoint']!r}")


def display_midpoint(display: Mapping[str, Any] | None) -> float | None:
    """The value a template's `display.midpoint` centres its colours on, or None."""
    return _number((display or {}).get("midpoint"))


def display_range(
    display: Mapping[str, Any] | None, statistics: ArtifactStatistics | None
) -> tuple[float, float] | None:
    """The `(low, high)` range to colour over, or None when it needs statistics that haven't
    been written yet.

    A constant field is widened by half a unit either side, so a colour scale never spans zero
    width; it renders as one flat colour, which is what a constant field looks like.
    """

    def resolve(end: Any) -> float | None:
        if isinstance(end, str):
            return _number(getattr(statistics, end)) if statistics is not None else None
        return _number(end)

    spec = range_spec((display or {}).get("range")) or DEFAULT_RANGE
    low, high = resolve(spec[0]), resolve(spec[1])
    if low is None or high is None:
        spec = _UNMEASURED_RANGE
        low, high = resolve(spec[0]), resolve(spec[1])
    if low is None or high is None:
        return None
    if high <= low:
        low, high = low - 0.5, low + 0.5
    midpoint = display_midpoint(display)
    if midpoint is not None and any(isinstance(end, str) for end in spec):
        half = max(abs(low - midpoint), abs(high - midpoint))
        low, high = midpoint - half, midpoint + half
    return low, high
