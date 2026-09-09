"""Execution-scoped observations, independent of mutable xarray attributes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any


def json_digest(value: Any) -> str:
    """Hash a canonical JSON value without retaining its contents."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class ExecutionEvidence:
    """Observed inputs; this is not per-output lineage or an aggregation proof."""

    process_sha256: str | None
    require_feature_ids: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, str] = field(default_factory=dict, repr=False)

    def describe(self) -> dict[str, Any]:
        missing = ["per_output_lineage", "aggregation_semantics"]
        if not self.sources:
            missing.append("source_artifacts")
        if any(source["snapshot_id"] is None for source in self.sources):
            missing.append("immutable_source_snapshots")
        if not self.features:
            missing.append("feature_inputs")
        # Named feature collection versioning is not implemented in this checkout.
        missing.append("named_feature_collection_versions")
        return {
            "scope": "executed_processes",
            "process_sha256": self.process_sha256,
            "sources": list(self.sources),
            "features": list(self.features),
            "missing": missing,
        }


_current: ContextVar[ExecutionEvidence | None] = ContextVar("ocs_execution_evidence", default=None)


@contextmanager
def capture_execution(process: dict[str, Any]) -> Generator[ExecutionEvidence]:
    """Isolate observations between simultaneous graph executions."""
    try:
        digest = json_digest(process)
    except (ValueError, TypeError):
        digest = None
    evidence = ExecutionEvidence(digest, require_feature_ids=_has_named_dhis2_export(process))
    token = _current.set(evidence)
    try:
        yield evidence
    finally:
        _current.reset(token)


def record_snapshot(path: str, snapshot_id: str) -> None:
    """Record the snapshot belonging to the actual opened readonly session."""
    evidence = _current.get()
    if evidence is not None:
        evidence.snapshots[str(Path(path).resolve())] = snapshot_id


def record_source(collection_id: str, artifact: Any) -> None:
    """Record a successfully opened artifact without copying paths or credentials."""
    evidence = _current.get()
    if evidence is None:
        return
    observation: dict[str, Any] = {"collection_id": collection_id}
    for key in ("artifact_id", "source_dataset_id"):
        value = getattr(artifact, key, None)
        observation[key] = value if isinstance(value, str) else None
    raw_path = getattr(artifact, "path", None)
    paths = getattr(artifact, "asset_paths", [])
    if raw_path is None and isinstance(paths, list) and paths:
        raw_path = paths[0]
    path = str(Path(raw_path).resolve()) if isinstance(raw_path, (str, PathLike)) else None
    observation["snapshot_id"] = evidence.snapshots.pop(path, None) if path is not None else None
    evidence.sources.append(observation)


def record_features(geometries: Any) -> None:
    """Fingerprint actual inline feature inputs before positional labels are assigned."""
    evidence = _current.get()
    if evidence is None:
        return
    if evidence.require_feature_ids:
        from open_climate_service.shared.features import validate_feature_ids

        validate_feature_ids(geometries)
    if not isinstance(geometries, dict) or geometries.get("type") not in {"Feature", "FeatureCollection"}:
        evidence.features.append({"input_sha256": None, "ids_valid": False, "reason": "no_feature_ids"})
        return
    features = geometries.get("features", []) if geometries.get("type") == "FeatureCollection" else [geometries]
    if not isinstance(features, list) or not all(isinstance(feature, dict) for feature in features):
        return
    # Properties are irrelevant to spatial aggregation and may contain incidental
    # metadata. Fingerprint only the geometry and identity actually used here.
    relevant = [{"id": feature.get("id"), "geometry": feature.get("geometry")} for feature in features]
    identifiers = [feature["id"] for feature in relevant]
    valid = all(isinstance(identifier, str) and bool(identifier.strip()) for identifier in identifiers)
    valid = valid and len(set(identifiers)) == len(identifiers)
    try:
        digest = json_digest(relevant)
    except (TypeError, ValueError):
        digest = None
    evidence.features.append({"input_sha256": digest, "feature_count": len(features), "ids_valid": valid})


def _has_named_dhis2_export(value: Any) -> bool:
    if isinstance(value, dict):
        arguments = value.get("arguments", {})
        if value.get("process_id") == "save_result" and isinstance(arguments, dict):
            options = arguments.get("options", {})
            if (
                str(arguments.get("format", "")).upper() == "DHIS2JSON"
                and isinstance(options, dict)
                and "export" in options
            ):
                return True
        return any(_has_named_dhis2_export(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_named_dhis2_export(child) for child in value)
    return False
