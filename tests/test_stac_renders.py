"""Render-extension output for colormapped and true-colour collections (CLIM-947)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
)
from open_climate_service.stac.services import _build_renders


def _artifact(variable: str = "reflectance") -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="a1",
        dataset_id="planet_flood_rgb",
        dataset_name="Flood true colour",
        variable=variable,
        format=ArtifactFormat.ICECHUNK,
        request_scope=ArtifactRequestScope(),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=85.3, ymin=28.1, xmax=85.5, ymax=28.3),
            temporal=CoverageTemporal(),
        ),
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )


def _render(display: Any) -> dict[str, Any] | None:
    renders = _build_renders(_artifact(), {"display": display})
    return None if renders is None else renders["default"]


def test_three_bands_produce_a_composite_render_with_no_colour_ramp() -> None:
    """A composite is not a colour ramp: `colormap_name` must be absent, `bands` present."""
    render = _render({"bands": ["red", "green", "blue"], "range": [0, 3000]})

    assert render is not None
    assert render["bands"] == ["red", "green", "blue"]
    assert "colormap_name" not in render
    # One rescale pair per band, as the Render extension expects.
    assert render["rescale"] == [[0.0, 3000.0], [0.0, 3000.0], [0.0, 3000.0]]
    assert render["open_climate_service:band_dimension"] == "band"


def test_per_band_ranges_are_kept_separate() -> None:
    """Bands with different dynamic ranges need their own stretch, or the colour is wrong."""
    render = _render({"bands": ["red", "green", "blue"], "range": [[0, 2500], [0, 2600], [0, 3000]]})

    assert render is not None
    assert render["rescale"] == [[0.0, 2500.0], [0.0, 2600.0], [0.0, 3000.0]]


def test_the_band_dimension_can_be_named() -> None:
    render = _render({"bands": ["r", "g", "b"], "range": [0, 255], "band_dimension": "spectral"})

    assert render is not None
    assert render["open_climate_service:band_dimension"] == "spectral"


@pytest.mark.parametrize(
    "bands",
    [["red", "green"], ["red", "green", "blue", "nir"], ["red", "green", 3], ["red", "green", ""]],
)
def test_a_band_list_that_is_not_three_names_is_refused(bands: Any) -> None:
    """Refused rather than half-rendered: a two-band composite has no defined meaning here."""
    assert _render({"bands": bands, "range": [0, 3000]}) is None


@pytest.mark.parametrize("value_range", [None, [0], [0, 1, 2], [[0, 1], [0, 1]], "0-3000"])
def test_a_composite_without_a_usable_range_is_refused(value_range: Any) -> None:
    assert _render({"bands": ["red", "green", "blue"], "range": value_range}) is None


@pytest.mark.parametrize(
    "value_range",
    [
        [1, 1],
        [10, 0],
        [float("nan"), 10],
        [0, float("inf")],
        [[0, 255], [1, 1], [0, 255]],
        [[0, 255], [0, 255], [float("-inf"), 255]],
    ],
)
def test_a_stretch_the_shader_cannot_use_is_refused(value_range: Any) -> None:
    """Shape is not enough — these are all well-formed and all unusable.

    The shader divides by `max - min` and embeds both numbers as GLSL literals, so an equal
    or inverted pair divides by zero or negates, and a non-finite endpoint gives NaN colours
    or a shader that will not compile. `NaN` and `Infinity` are not valid JSON either, so the
    collection response itself would be malformed.
    """
    assert _render({"bands": ["red", "green", "blue"], "range": value_range}) is None


def test_nodata_is_carried_onto_a_composite() -> None:
    render = _render({"bands": ["red", "green", "blue"], "range": [0, 255], "nodata": 0})

    assert render is not None
    assert render["nodata"] == 0.0


def test_a_colormapped_collection_is_unchanged() -> None:
    """The single-variable path must be untouched by the composite branch."""
    render = _render({"colormap": "rdbu_r", "range": [-30, 30], "nodata": 0})

    assert render is not None
    assert render["colormap_name"] == "rdbu_r"
    assert render["rescale"] == [[-30.0, 30.0]]
    assert render["nodata"] == 0.0
    assert "bands" not in render


def test_a_display_block_with_neither_bands_nor_colormap_yields_no_render() -> None:
    assert _render({"range": [0, 1]}) is None
    assert _build_renders(_artifact(), {}) is None
