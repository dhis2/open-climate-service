"""Feature identity: the one implementation of what identifies a feature, and what breaks it.

A feature's identifier becomes the geometry-dimension label that the DHIS2 and CHAP exports use
as their location column. It therefore has to identify exactly one feature, and this is the only
place that decides whether it does — the stored collection, a provider's output and both export
paths all come through here, so the rule cannot drift between them.

The failure this exists to prevent is not a loud one. A duplicate identifier does not drop a
feature: two features map onto one org unit, DHIS2 keeps whichever value arrives last, and the
push succeeds with the wrong number in it.
"""

from collections.abc import Sequence
from typing import Any

GEOJSON_FEATURE_TYPES = {"Feature", "FeatureCollection"}


def validate_feature_ids(geometries: Any, *, id_property: str | None = None) -> list[str]:
    """Return each feature's identifier in order, or raise ValueError naming what is wrong.

    Where the identifier is read from depends on whether a collection declares one, and the two
    modes do not overlap.

    With `id_property`, it is read from a feature's `properties` and from nowhere else. That is
    what the openEO specification guarantees survives aggregation ("feature properties are
    preserved for vector data cubes and all GeoJSON Features"), and what the usual
    GeoJSON-to-frame conversion keeps as a column. Falling back to a top-level `id` here would
    be worse than useless: the conversion drops that field, so the check would pass on a
    collection whose stored file has no identifier column at all, and the loss would surface
    only when an export pushed values against nothing.

    Without `id_property` the top-level `id` is read directly. That is the hand-made call — an
    inline FeatureCollection pasted into a process graph, which has no template to name a
    property — and it is the only case the top-level id serves.
    """
    features = _members(geometries)
    identifiers: list[str] = []
    first_seen: dict[str, int] = {}
    for index, feature in enumerate(features):
        identifier = _identifier_of(feature, index=index, id_property=id_property)
        if identifier in first_seen:
            # Named offender first, like every other error here, so a provider with thousands of
            # features gets the two indices to look at rather than only the repeated value.
            raise ValueError(
                f"Feature {index} repeats {_describe_source(id_property)} '{identifier}', "
                f"first seen at feature {first_seen[identifier]}; an identifier must name exactly "
                "one feature, or two features push values against the same org unit"
            )
        first_seen[identifier] = index
        identifiers.append(identifier)
    return identifiers


def _members(geometries: Any) -> list[Any]:
    """Return the features of a Feature or FeatureCollection, or raise ValueError."""
    if not isinstance(geometries, dict) or geometries.get("type") not in GEOJSON_FEATURE_TYPES:
        raise ValueError("expected GeoJSON Feature or FeatureCollection with explicit feature identifiers")
    if geometries["type"] == "Feature":
        return [geometries]
    features: Any = geometries.get("features")
    # A non-string Sequence, matching `ingestions.services._feature_collection_members`. A tuple
    # is what a provider that built its features with a comprehension hands over, and the two
    # checks sit on the same path: registration accepting one while the writer refused it made
    # a collection that could be recorded and not stored.
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise ValueError("FeatureCollection.features must be an array of features")
    return list(features)


def _identifier_of(feature: Any, *, index: int, id_property: str | None) -> str:
    """Return one feature's identifier, or raise ValueError saying which feature and why."""
    if not isinstance(feature, dict):
        raise ValueError(f"Feature {index} is not a GeoJSON Feature object")
    value: Any = None
    if id_property is not None:
        properties = feature.get("properties")
        value = properties.get(id_property) if isinstance(properties, dict) else None
    else:
        value = feature.get("id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        hint = ""
        if id_property is not None and isinstance(feature.get("id"), (str, int)):
            # The most likely mistake, and one a top-level fallback would have hidden: the
            # provider put the identifier where GeoJSON puts it rather than where the template
            # named it, and a frame conversion would drop it on the way to the store.
            hint = (
                f" (the feature carries a top-level id {feature['id']!r}; a stored collection "
                f"reads properties.{id_property}, so the provider must put it there)"
            )
        raise ValueError(
            f"Feature {index} has no usable {_describe_source(id_property)}; "
            f"got {value!r}, expected a non-empty string or an integer{hint}"
        )
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"Feature {index} has a blank {_describe_source(id_property)}")
    return identifier


def _describe_source(id_property: str | None) -> str:
    return f"properties.{id_property}" if id_property is not None else "feature.id"
