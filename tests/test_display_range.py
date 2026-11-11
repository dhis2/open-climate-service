"""The range a dataset is coloured over: declared or named statistics, then centred (CLIM-1439).

The wiring into STAC render hints, the pages and thumbnails is tested beside each of them;
this is the decision they share.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_climate_service.data_registry.services.datasets import _validate_dataset_template
from open_climate_service.ingestions.schemas import ArtifactStatistics
from open_climate_service.shared.display import display_range

MEASURED = ArtifactStatistics(min=-4.0, max=6.5, mean=1.0, p2=-3.0, p50=1.0, p98=5.0)
BEFORE_PERCENTILES = ArtifactStatistics(min=-4.0, max=6.5)


@pytest.mark.parametrize(
    ("display", "statistics", "expected"),
    [
        pytest.param({"range": [0, 20]}, MEASURED, (0.0, 20.0), id="numbers-win"),
        pytest.param({}, MEASURED, (-3.0, 5.0), id="default-p2-p98"),
        pytest.param({"range": ["min", "max"]}, MEASURED, (-4.0, 6.5), id="named"),
        pytest.param({"range": [0, "p98"]}, MEASURED, (0.0, 5.0), id="mixed"),
        pytest.param({}, BEFORE_PERCENTILES, (-4.0, 6.5), id="unmeasured-falls-back-to-min-max"),
        pytest.param({"range": "nonsense"}, MEASURED, (-3.0, 5.0), id="unusable-falls-back-to-default"),
        pytest.param({"midpoint": 0}, MEASURED, (-5.0, 5.0), id="midpoint-centres-the-measured"),
        pytest.param({"range": [-1, 3], "midpoint": 0}, MEASURED, (-1.0, 3.0), id="numbers-kept-as-written"),
        pytest.param({"range": [-1, "p98"], "midpoint": 0}, MEASURED, (-5.0, 5.0), id="a-measured-end-is-centred"),
        pytest.param({"midpoint": True}, MEASURED, (-3.0, 5.0), id="a-boolean-is-no-midpoint"),
        pytest.param({}, ArtifactStatistics(min=2.0, max=2.0), (1.5, 2.5), id="constant-is-widened"),
        pytest.param({}, ArtifactStatistics(min=0.0), None, id="half-measured"),
        pytest.param(None, None, None, id="nothing-to-go-on"),
    ],
)
def test_the_range_is_declared_or_measured_then_centred(
    display: dict[str, Any] | None, statistics: ArtifactStatistics | None, expected: tuple[float, float] | None
) -> None:
    assert display_range(display, statistics) == expected


@pytest.mark.parametrize(
    ("display", "error"),
    [
        ({"range": ["p2", "p99"]}, "display.range must be two ends"),
        ({"range": [0, 20, 40]}, "display.range must be two ends"),
        ({"midpoint": "p50"}, "display.midpoint must be a number"),
    ],
)
def test_a_template_whose_range_could_never_resolve_is_refused(display: dict[str, Any], error: str) -> None:
    """A misspelt statistic would otherwise fall back to the default without a word."""
    template = {"id": "rain", "name": "Rain", "variable": "precip", "sync": {"kind": "static"}, "display": display}

    with pytest.raises(ValueError, match=f"Dataset template 'rain' in t.yaml: {error}"):
        _validate_dataset_template(template, source="t.yaml")
