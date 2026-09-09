"""A dataset template's description must reach the published collection (CLIM-973).

Until this, a template could carry a `description` and nothing read it: the collection always
published a generated sentence. That mattered for datasets whose values mislead without a
caveat — Meta RWI ranks micro-regions within one country, MODIS LST is surface rather than 2 m
air temperature, CHIRPS3 monthly is a mean daily rate — because the published collection is the
only place an API consumer would look.
"""

from typing import Any

import pytest

from open_climate_service.ingestions.schemas import ArtifactRecord
from open_climate_service.stac.services import _collection_description, _sanitize_variable_attrs

_GENERATED_PREFIX = "Published GeoZarr dataset for"


class _Artifact:
    """Only the field `_collection_description` reads."""

    def __init__(self, dataset_name: str = "Relative Wealth Index") -> None:
        self.dataset_name = dataset_name


def _artifact(name: str = "Relative Wealth Index") -> ArtifactRecord:
    # Structurally typed on purpose: constructing a full ArtifactRecord here would pin the test
    # to fields the function under test never touches.
    return _Artifact(name)  # type: ignore[return-value]


def test_template_description_becomes_the_collection_description() -> None:
    caveat = (
        "Values are relative WITHIN a single country, not globally, so they cannot be compared "
        "between countries or between OCS instances."
    )
    assert _collection_description("rwi", _artifact(), {"description": caveat}) == caveat


def test_generated_description_is_used_when_the_template_has_none() -> None:
    """openEO save_result outputs have no template block at all, so the fallback must stay."""
    result = _collection_description("derived", _artifact("Temperature anomaly"), {})
    assert result == f"{_GENERATED_PREFIX} Temperature anomaly"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_a_blank_description_falls_back_rather_than_publishing_emptiness(blank: str) -> None:
    """A template with an empty description is a mistake, not a request for a blank field."""
    result = _collection_description("d", _artifact("Rainfall"), {"description": blank})
    assert result == f"{_GENERATED_PREFIX} Rainfall"


def test_a_non_string_description_falls_back() -> None:
    """YAML will happily produce a list or a number here; neither is a description."""
    for bad in (42, ["a", "b"], {"text": "x"}, None):
        result = _collection_description("d", _artifact("Rainfall"), {"description": bad})
        assert result == f"{_GENERATED_PREFIX} Rainfall"


def test_description_is_stripped() -> None:
    """`description: >-` folded YAML leaves a trailing newline that would reach clients."""
    assert _collection_description("d", _artifact(), {"description": "  Caveat.\n"}) == "Caveat."


# -- the per-variable counterpart ------------------------------------------------------


def _collection_with_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {"cube:variables": {"rwi": {"type": "data", "attrs": dict(attrs)}}}


def test_variable_comment_survives_sanitisation() -> None:
    """CF `comment` is where a per-variable caveat lives; the allowlist used to drop it."""
    comment = "Land surface temperature, not 2 m air temperature."
    collection = _collection_with_attrs({"long_name": "LST", "units": "degC", "comment": comment})
    _sanitize_variable_attrs(collection)
    assert collection["cube:variables"]["rwi"]["attrs"]["comment"] == comment


@pytest.mark.parametrize("attr", ["comment", "references", "institution", "source"])
def test_every_cf_2_6_2_variable_attribute_survives(attr: str) -> None:
    """CF s2.6.2 allows institution, source, references and comment "to be either global or
    assigned to individual variables", so all four are legal at variable scope and all four
    pass through. `references` matters most: it is the standard home for attribution, so a
    plugin should use it rather than inventing an `attribution` attribute.

    CF is formally a netCDF convention and GeoZarr does not mandate it; OCS adopts it anyway
    (`shared/cf.py`, and the STAC CF extension is already emitted), so 2.6.2 is what decides
    which of these belong on a variable rather than globally."""
    collection = _collection_with_attrs({"long_name": "X", attr: "value"})
    _sanitize_variable_attrs(collection)
    assert collection["cube:variables"]["rwi"]["attrs"][attr] == "value"


@pytest.mark.parametrize("attr", ["title", "history"])
def test_global_only_cf_attributes_are_not_passed_through(attr: str) -> None:
    """`title` and `history` are global-only in CF 2.6.2 — the "newly defined attributes"
    sentence names only the other four — so they have no place on a variable."""
    collection = _collection_with_attrs({"long_name": "X", attr: "value"})
    _sanitize_variable_attrs(collection)
    assert attr not in collection["cube:variables"]["rwi"]["attrs"]


def test_comment_is_not_emitted_as_a_cf_field() -> None:
    """The STAC CF extension v1.0.0 defines only `cf:standard_name` and `cf:cell_methods`.

    Emitting `cf:comment` would invent a field and then declare conformance to an extension
    that does not define it — worse than not publishing the comment at all, because a client
    validating against the schema would reject the collection.
    """
    collection = _collection_with_attrs({"comment": "A caveat.", "standard_name": "air_temperature"})
    _sanitize_variable_attrs(collection)
    variable = collection["cube:variables"]["rwi"]
    assert "cf:comment" not in variable
    # The genuinely-defined one is still prefixed, so this is not just "nothing is prefixed".
    assert variable["cf:standard_name"] == "air_temperature"


def test_unlisted_attrs_are_still_dropped() -> None:
    """Adding `comment` must not turn the allowlist into a passthrough of everything."""
    # The real noise: WorldPop rasters arrive with all of these on the variable.
    collection = _collection_with_attrs(
        {
            "long_name": "RWI",
            "comment": "keep",
            "_FillValue": "drop",
            "grid_mapping": "drop",
            "AREA_OR_POINT": "drop",
            "TIFFTAG_COPYRIGHT": "drop",
            "STATISTICS_MEAN": "drop",
            "source_license": "drop, CLIM-946 owns licensing",
        }
    )
    _sanitize_variable_attrs(collection)
    kept = collection["cube:variables"]["rwi"]["attrs"]
    assert set(kept) == {"long_name", "comment"}


def test_a_non_string_comment_is_dropped() -> None:
    collection = _collection_with_attrs({"long_name": "RWI", "comment": 3.14})
    _sanitize_variable_attrs(collection)
    assert "comment" not in collection["cube:variables"]["rwi"]["attrs"]


# -- /datasets applies the same normalisation as STAC -----------------------------------


def test_datasets_description_is_stripped_and_blank_becomes_null() -> None:
    """The two public catalogue surfaces must agree about the same template field.

    Without this, `/datasets` published "   " where the STAC collection fell back to the
    generated sentence — and `description: >-` folded YAML routinely leaves a trailing newline.
    """
    from open_climate_service.ingestions.services import _as_optional_text

    assert _as_optional_text("  Caveat.\n") == "Caveat."
    for blank in ("", "   ", "\n\t "):
        assert _as_optional_text(blank) is None
    for bad in (42, ["a"], {"t": "x"}, None):
        assert _as_optional_text(bad) is None
