"""The value range a dataset is coloured over (CLIM-1439).

A template's `display.range` wins where it is declared, as written. Otherwise the range is the
dataset's measured minimum and maximum, so a template no longer has to guess values that suit
every country and every period. `display.midpoint` centres a measured range on a value, so a
diverging colormap keeps its neutral colour on the neutral value: zero for an anomaly, where
blue has to mean below normal. A declared range is not centred: its ends are the author's
choice, and one who wants it centred declares it symmetric.

Decided here once, for the map viewer's STAC render hints, the dataset and data source pages,
and openEO PNG output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from open_climate_service.ingestions.schemas import ArtifactStatistics


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def display_midpoint(display: Mapping[str, Any] | None) -> float | None:
    """The value a template's `display.midpoint` centres its colours on, or None."""
    return _number((display or {}).get("midpoint"))


def display_range(
    display: Mapping[str, Any] | None, statistics: ArtifactStatistics | None
) -> tuple[float, float] | None:
    """The `(low, high)` range to colour over, or None when nothing declares or measured one.

    A constant field is widened by half a unit either side, so a colour scale never spans zero
    width; it renders as one flat colour, which is what a constant field looks like.
    """
    declared = (display or {}).get("range")
    low, high = (
        (_number(declared[0]), _number(declared[1]))
        if isinstance(declared, list | tuple) and len(declared) == 2
        else (None, None)
    )
    measured = False
    if (low is None or high is None) and statistics is not None:
        low, high = _number(statistics.minimum), _number(statistics.maximum)
        measured = True
    if low is None or high is None:
        return None
    if high <= low:
        low, high = low - 0.5, low + 0.5
    midpoint = display_midpoint(display)
    if measured and midpoint is not None:
        half = max(abs(low - midpoint), abs(high - midpoint))
        low, high = midpoint - half, midpoint + half
    return low, high
