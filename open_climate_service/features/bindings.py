"""Bind a workflow argument to a registered feature collection by id instead of inline geometry.

A pipeline binds a workflow's geometry parameter as ``{"from_features": "<collection id>"}``.
At submission the marker becomes a ``load_features`` node pinned to the collection's current
record, so a queued job reads the collection it was submitted against, or fails if a refresh
has replaced it.
"""

from __future__ import annotations

from typing import Any

FROM_FEATURES = "from_features"
"""Argument key naming a registered feature collection instead of carrying its geometry."""


def _node_name(feature_id: str) -> str:
    """The graph node that loads one feature collection. Stable, so two references share one node."""
    return f"features_{feature_id}"


def is_inline_geojson(value: Any) -> bool:
    """Whether a mapping is an inline GeoJSON feature or collection."""
    return isinstance(value, dict) and value.get("type") in {"Feature", "FeatureCollection"}


def iter_feature_references(value: Any) -> Any:
    """Yield the feature collection ids ``value`` references; raise on a malformed marker."""
    if is_inline_geojson(value):
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_feature_references(item)
    elif isinstance(value, dict):
        if FROM_FEATURES in value:
            feature_id = value.get(FROM_FEATURES)
            if set(value) != {FROM_FEATURES} or not isinstance(feature_id, str) or not feature_id.strip():
                raise ValueError(
                    f"{FROM_FEATURES!r} must be the only key in its object and name a non-empty feature id"
                )
            if feature_id != feature_id.strip():
                raise ValueError(f"{FROM_FEATURES!r} feature id must not have leading or trailing whitespace")
            yield feature_id
            return
        for item in value.values():
            yield from iter_feature_references(item)


def resolve_feature_references(
    value: Any,
    nodes: dict[str, Any] | None = None,
    versions: dict[str, str] | None = None,
) -> Any:
    """Rewrite each feature marker into a reference to a ``load_features`` node added to ``nodes``.

    Geometry is resolved during execution and never copied into the persisted graph. With
    ``versions``, each node is pinned to the collection version it names.
    """
    if is_inline_geojson(value):
        return value
    if isinstance(value, list):
        return [resolve_feature_references(item, nodes, versions) for item in value]
    if not isinstance(value, dict):
        return value
    if FROM_FEATURES in value:
        feature_id = next(iter_feature_references(value))
        name = _node_name(feature_id)
        if nodes is not None:
            arguments = {"id": feature_id}
            if versions is not None:
                arguments["version"] = versions[feature_id]
            nodes[name] = {"process_id": "load_features", "arguments": arguments}
        return {"from_node": name}
    return {key: resolve_feature_references(item, nodes, versions) for key, item in value.items()}


def feature_versions(arguments: Any) -> dict[str, str]:
    """Pin every collection ``arguments`` references to its current registered record."""
    from open_climate_service.features import services as feature_services

    feature_ids = sorted(set(iter_feature_references(arguments)))
    if not feature_ids:
        return {}
    records = feature_services.registered_collections()
    versions: dict[str, str] = {}
    for feature_id in feature_ids:
        record = records.get(feature_id)
        if record is None:
            raise ValueError(
                f"Feature collection {feature_id!r} is declared but unregistered; refresh it before submitting "
                "a workflow that references it"
            )
        versions[feature_id] = record.created_at.isoformat()
    return versions
