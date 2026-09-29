"""Execution-scoped observations, independent of mutable xarray attributes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Generator, Mapping, Sequence
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
    # One entry per completed aggregate_spatial call: its named method, or None when its
    # reducer is not a single named reduction.
    spatial_aggregations: list[str | None] = field(default_factory=list)
    snapshots: dict[str, str] = field(default_factory=dict, repr=False)

    def describe(self) -> dict[str, Any]:
        missing = ["per_output_lineage", "aggregation_semantics"]
        if not self.sources:
            missing.append("source_artifacts")
        if any(source["snapshot_id"] is None for source in self.sources):
            missing.append("immutable_source_snapshots")
        if not self.features:
            missing.append("feature_inputs")
        if len(self.spatial_aggregations) != 1 or self.spatial_aggregations[0] is None:
            # Only one named spatial aggregation can be attributed to the saved result
            # without per-output lineage; none, several, or an unnamed reducer cannot.
            missing.append("spatial_aggregation_method")
        # Named feature collection versioning is not implemented in this checkout.
        missing.append("named_feature_collection_versions")
        return {
            "scope": "executed_processes",
            "process_sha256": self.process_sha256,
            "sources": list(self.sources),
            "features": list(self.features),
            "spatial_aggregations": list(self.spatial_aggregations),
            "missing": missing,
        }


_current: ContextVar[ExecutionEvidence | None] = ContextVar("ocs_execution_evidence", default=None)
_spatial_methods: ContextVar[set[str] | None] = ContextVar("ocs_spatial_methods", default=None)


@contextmanager
def capture_execution(
    process: dict[str, Any],
    workflows: Mapping[str, Any] | None = None,
) -> Generator[ExecutionEvidence]:
    """Isolate observations between simultaneous graph executions.

    ``workflows`` maps workflow (user-defined process) IDs to their process graphs,
    so a named DHIS2 export saved inside a called workflow is detected as well as
    one saved directly in the submitted graph.
    """
    try:
        digest = json_digest(process)
    except (ValueError, TypeError):
        digest = None
    evidence = ExecutionEvidence(
        digest,
        require_feature_ids=_has_named_dhis2_export(process, (workflows or {}).get, set()),
    )
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


@contextmanager
def observe_spatial_aggregation() -> Generator[None]:
    """Attribute named reductions to one aggregate_spatial call and record its method.

    Only reductions running inside this scope count, so a named reducer used for a
    temporal reduction or anywhere else never reads as a spatial aggregation. The
    call is recorded only when it completes.
    """
    methods: set[str] = set()
    token = _spatial_methods.set(methods)
    try:
        yield
    finally:
        _spatial_methods.reset(token)
    evidence = _current.get()
    if evidence is not None:
        evidence.spatial_aggregations.append(next(iter(methods)) if len(methods) == 1 else None)


def record_spatial_reduction(method: str) -> None:
    """Note a named reduction; ignored outside an aggregate_spatial call."""
    methods = _spatial_methods.get()
    if methods is not None:
        methods.add(method)


@contextmanager
def unattributed_spatial_reduction() -> Generator[None]:
    """Run a reducer whose result is not the named reductions it calls.

    Inside an aggregate_spatial call, a graph such as ``reduce_by_method(mean)`` times 2 records
    ``mean`` as it runs, though what it returns is not a mean. Its steps are ignored here, so the
    call records no method rather than a wrong one.
    """
    token = _spatial_methods.set(None)
    try:
        yield
    finally:
        _spatial_methods.reset(token)


def record_features(geometries: Any) -> None:
    """Fingerprint actual inline feature inputs before positional labels are assigned."""
    evidence = _current.get()
    if evidence is None:
        return
    from open_climate_service.shared.features import validate_dhis2_feature_ids, validate_feature_ids

    if evidence.require_feature_ids:
        validate_dhis2_feature_ids(geometries)
    if not isinstance(geometries, dict) or geometries.get("type") not in {"Feature", "FeatureCollection"}:
        evidence.features.append({"input_sha256": None, "ids_valid": False, "reason": "no_feature_ids"})
        return
    features = geometries.get("features", []) if geometries.get("type") == "FeatureCollection" else [geometries]
    # The same shapes `validate_feature_ids` accepts. A tuple is what a provider that built its
    # features with a comprehension hands over, and a recorder that silently skipped one would
    # leave an execution with no feature fingerprint at all while the aggregation succeeded.
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return
    # Annotated, because the union of the two branches above narrows to dict on one of them and
    # the members of a FeatureCollection are whatever the caller put there.
    members: list[Any] = list(features)
    if not all(isinstance(feature, dict) for feature in members):
        return
    # Properties are irrelevant to spatial aggregation and may contain incidental
    # metadata. Fingerprint only the geometry and identity actually used here.
    relevant = [{"id": feature.get("id"), "geometry": feature.get("geometry")} for feature in members]
    # Asked of the validator rather than re-derived here. Two implementations of "is this
    # identity usable" drift: this one read non-blank strings while the validator also accepts
    # an integer id, so a named DHIS2 export could pass validation and still be manifested as
    # `ids_valid: false`. One rule, one answer.
    try:
        validate_feature_ids(geometries)
        valid = True
    except ValueError:
        valid = False
    try:
        digest = json_digest(relevant)
    except (TypeError, ValueError):
        digest = None
    evidence.features.append({"input_sha256": digest, "feature_count": len(members), "ids_valid": valid})


def _has_named_dhis2_export(
    value: Any,
    resolve_workflow: Callable[[str], Any],
    expanded: set[str],
) -> bool:
    """Return True when the graph, or a workflow it calls, saves a named DHIS2 export.

    Each workflow is expanded at most once, which also stops recursive workflows.
    """
    if isinstance(value, dict):
        arguments = value.get("arguments", {})
        process_id = value.get("process_id")
        if process_id == "save_result" and isinstance(arguments, dict):
            options = arguments.get("options", {})
            if (
                str(arguments.get("format", "")).upper() == "DHIS2JSON"
                and isinstance(options, dict)
                and "export" in options
            ):
                return True
        if isinstance(process_id, str) and process_id not in expanded:
            workflow_graph = resolve_workflow(process_id)
            if workflow_graph is not None:
                expanded.add(process_id)
                if _has_named_dhis2_export(workflow_graph, resolve_workflow, expanded):
                    return True
        return any(_has_named_dhis2_export(child, resolve_workflow, expanded) for child in value.values())
    if isinstance(value, list):
        return any(_has_named_dhis2_export(child, resolve_workflow, expanded) for child in value)
    return False
