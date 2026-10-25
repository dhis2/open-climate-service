"""The range a dataset is coloured over: declared, else measured, then centred (CLIM-1439).

The wiring into STAC render hints, the pages and thumbnails is tested beside each of them;
this is the decision they share.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_climate_service.ingestions.schemas import ArtifactStatistics
from open_climate_service.shared.display import display_range

MEASURED = ArtifactStatistics(min=-4.0, max=6.5)


@pytest.mark.parametrize(
    ("display", "statistics", "expected"),
    [
        pytest.param({"range": [0, 20]}, MEASURED, (0.0, 20.0), id="declared-wins"),
        pytest.param({}, MEASURED, (-4.0, 6.5), id="measured-without-a-declaration"),
        pytest.param({"range": "nonsense"}, MEASURED, (-4.0, 6.5), id="unusable-declaration-falls-back"),
        pytest.param({"range": ["a", "b"]}, MEASURED, (-4.0, 6.5), id="unusable-pair-falls-back"),
        pytest.param({"midpoint": 0}, MEASURED, (-6.5, 6.5), id="midpoint-centres-the-measured"),
        pytest.param({"range": [-1, 3], "midpoint": 0}, MEASURED, (-1.0, 3.0), id="declared-is-kept-as-written"),
        pytest.param({"midpoint": True}, MEASURED, (-4.0, 6.5), id="a-boolean-is-no-midpoint"),
        pytest.param({}, ArtifactStatistics(min=2.0, max=2.0), (1.5, 2.5), id="constant-is-widened"),
        pytest.param({}, ArtifactStatistics(min=0.0), None, id="half-measured"),
        pytest.param(None, None, None, id="nothing-to-go-on"),
    ],
)
def test_the_range_is_declared_else_measured_then_centred(
    display: dict[str, Any] | None, statistics: ArtifactStatistics | None, expected: tuple[float, float] | None
) -> None:
    assert display_range(display, statistics) == expected
