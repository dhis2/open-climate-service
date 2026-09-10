"""Feature identity checks shared by aggregation and feature providers."""

from typing import Any


def validate_feature_ids(geometries: Any) -> list[str]:
    """Require explicit, unique string IDs and identify the offending feature."""
    if not isinstance(geometries, dict) or geometries.get("type") not in {"Feature", "FeatureCollection"}:
        raise ValueError("Named DHIS2 export requires GeoJSON features with explicit feature.id values")
    features: Any = geometries.get("features") if geometries["type"] == "FeatureCollection" else [geometries]
    if not isinstance(features, list):
        raise ValueError("FeatureCollection.features must be a list")
    identifiers: list[str] = []
    seen: set[str] = set()
    for index, feature in enumerate(features):
        identifier = feature.get("id") if isinstance(feature, dict) else None
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Feature {index} is missing a non-empty string feature.id")
        if identifier in seen:
            raise ValueError(f"Feature {index} repeats feature.id '{identifier}'")
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers
