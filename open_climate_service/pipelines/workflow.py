"""Bind a pipeline's workflow reference to a registered workflow and invoke it for one interval.

The pipeline never looks inside the workflow's process graph beyond its declared interface:
its parameters, and whether its result is already saved. Workflows come from the one
workflow registry (`openeo.workflows`), and feature collections are bound with the same
``from_features`` marker workflow triggers use, resolved by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from open_climate_service.automation.service import (
    _feature_versions,
    _is_inline_geojson,
    _iter_feature_references,
    _resolve_feature_references,
)
from open_climate_service.openeo import workflows as workflow_registry
from open_climate_service.openeo.schemas import WorkflowRecord
from open_climate_service.pipelines.schemas import WorkflowReference
from open_climate_service.shared.provenance import json_digest

TEMPORAL_INTERVAL = "temporal-interval"
COLLECTION_ID = "collection-id"
GEOJSON = "geojson"

WORKFLOW_NODE = "workflow"
SAVE_NODE = "save"


class WorkflowBindingError(ValueError):
    """The reference does not fit the workflow's declared interface; the message lists every problem."""


@dataclass(frozen=True)
class BoundWorkflow:
    """A validated reference: the workflow, the parameter each run fills, and the inputs it binds.

    ``record`` is the workflow frozen when it was bound: its graph has every nested workflow
    call inlined, so the digest and every graph built from it describe the same computation.
    """

    record: WorkflowRecord
    interval_parameter: str
    dataset_ids: tuple[str, ...]
    feature_collections: tuple[str, ...]
    definition_digest: str


def _schema(parameter: dict[str, Any]) -> dict[str, Any]:
    schema = parameter.get("schema")
    return schema if isinstance(schema, dict) else {}


def _required(parameter: dict[str, Any]) -> bool:
    return "default" not in parameter and not parameter.get("optional", False)


def parameters(record: WorkflowRecord) -> list[dict[str, Any]]:
    """The declared parameters that have a name."""
    return [p for p in record.parameters if isinstance(p.get("name"), str)]


def saves_own_result(record: WorkflowRecord) -> bool:
    """True when the workflow's result node is a ``save_result``, which fixes its output format."""
    return any(
        isinstance(node, dict) and node.get("result") and node.get("process_id") == "save_result"
        for node in record.process_graph.values()
    )


def interval_parameters(record: WorkflowRecord) -> list[str]:
    """Names of the parameters declared as a temporal interval."""
    return [str(p["name"]) for p in parameters(record) if _schema(p).get("subtype") == TEMPORAL_INTERVAL]


def pipeline_ready(record: WorkflowRecord) -> bool:
    """Whether a pipeline can invoke this workflow at all: it takes an interval and returns data."""
    return bool(record.process_graph) and bool(interval_parameters(record)) and not saves_own_result(record)


def frozen(record: WorkflowRecord) -> WorkflowRecord:
    """The workflow with every workflow it calls inlined from the registry as of now."""
    return record.model_copy(update={"process_graph": workflow_registry.freeze_process_graph(record.process_graph)})


def definition_digest(record: WorkflowRecord) -> str:
    """Digest of what a frozen workflow computes, nested workflows included, so any change is noticed."""
    return json_digest({"parameters": record.parameters, "process_graph": record.process_graph})


def _type_problem(name: str, schema: dict[str, Any], value: Any) -> str | None:
    """A readable reason ``value`` does not fit a simple JSON Schema, or None."""
    subtype = schema.get("subtype")
    if subtype == GEOJSON:
        if _is_inline_geojson(value):
            return None
        if isinstance(value, dict) and "from_features" in value:
            try:
                list(_iter_feature_references(value))
            except ValueError as exc:
                return f"'{name}': {exc}"
            return None
        return f'\'{name}\' takes a feature collection: bind it as {{"from_features": "<collection id>"}}'
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"'{name}' must be one of {', '.join(map(str, enum))}, not {value!r}"
    expected = schema.get("type")
    checks: dict[str, Any] = {
        "string": lambda v: isinstance(v, str) and bool(v.strip()),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    if isinstance(expected, str) and expected in checks and not checks[expected](value):
        return f"'{name}' must be a non-empty {expected}" if expected == "string" else f"'{name}' must be {expected}"
    return None


def bind(reference: WorkflowReference) -> BoundWorkflow:
    """Check a reference against its registered workflow; raise WorkflowBindingError listing every problem."""
    record = workflow_registry.get_workflow(reference.id)
    if record is None:
        raise WorkflowBindingError(f"'{reference.id}' is not a registered workflow (see GET /process_graphs)")
    if not record.process_graph:
        raise WorkflowBindingError(f"workflow '{reference.id}' has no process graph to run")
    problems: list[str] = []
    if saves_own_result(record):
        problems.append(
            f"workflow '{reference.id}' saves its own result in a fixed format; a pipeline saves through its "
            "destination, so choose a workflow that returns data, such as aggregate_by_features"
        )
    declared = {str(p["name"]): p for p in parameters(record)}
    unknown = sorted(set(reference.arguments) - set(declared))
    if unknown:
        problems.append(f"unknown argument(s) {', '.join(unknown)}; the workflow declares {', '.join(declared)}")
    intervals = interval_parameters(record)
    unbound_intervals = [name for name in intervals if name not in reference.arguments]
    interval = ""
    if len(unbound_intervals) == 1:
        interval = unbound_intervals[0]
    elif not intervals:
        problems.append(
            f"workflow '{reference.id}' declares no temporal-interval parameter, so a run cannot give it its interval"
        )
    elif not unbound_intervals:
        problems.append(
            f"leave '{intervals[0]}' unbound: each run supplies it from its own interval"
            if len(intervals) == 1
            else f"leave one of {', '.join(intervals)} unbound for each run's interval"
        )
    else:
        problems.append(
            f"bind all but one of {', '.join(unbound_intervals)}; the one left receives each run's interval"
        )
    missing = [
        name for name, p in declared.items() if _required(p) and name != interval and name not in reference.arguments
    ]
    if missing:
        problems.append(f"missing required argument(s) {', '.join(missing)}")
    datasets: list[str] = []
    for name, value in reference.arguments.items():
        parameter = declared.get(name)
        if parameter is None:
            continue
        problem = _type_problem(name, _schema(parameter), value)
        if problem is not None:
            problems.append(problem)
        elif _schema(parameter).get("subtype") == COLLECTION_ID:
            datasets.append(str(value))
    if problems:
        raise WorkflowBindingError("; ".join(problems))
    try:
        record = frozen(record)
    except ValueError as exc:
        raise WorkflowBindingError(f"workflow '{reference.id}' cannot be frozen for a run: {exc}") from None
    if saves_own_result(record):
        raise WorkflowBindingError(
            f"workflow '{reference.id}' saves its own result in a fixed format; a pipeline saves through its "
            "destination, so choose a workflow that returns data, such as aggregate_by_features"
        )
    return BoundWorkflow(
        record=record,
        interval_parameter=interval,
        dataset_ids=tuple(datasets),
        feature_collections=tuple(sorted(set(_iter_feature_references(reference.arguments)))),
        definition_digest=definition_digest(record),
    )


def build_process(
    reference: WorkflowReference,
    bound: BoundWorkflow,
    *,
    start: str,
    end: str,
    save_format: str,
    export_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The process graph for one run, and the arguments it resolved.

    The workflow is called as one node, then frozen: the call, and any workflow it calls, is
    inlined from the registry as of now, with declared defaults applied. A queued or retried
    job therefore runs the definition that was approved, even if the workflow is edited
    meanwhile. The result is saved through the pipeline's named export in the destination
    plugin's format. Feature references are pinned to the collection version current now.
    """
    versions = _feature_versions(reference.arguments)
    feature_nodes: dict[str, Any] = {}
    arguments: dict[str, Any] = _resolve_feature_references(reference.arguments, feature_nodes, versions)
    arguments[bound.interval_parameter] = [start, end]
    graph = {
        **feature_nodes,
        WORKFLOW_NODE: {"process_id": bound.record.id, "arguments": arguments},
        SAVE_NODE: {
            "process_id": "save_result",
            "arguments": {
                "data": {"from_node": WORKFLOW_NODE},
                "format": save_format,
                "options": {"export": export_id},
            },
            "result": True,
        },
    }
    # Inline the bound (already frozen) definition only: nothing is read from the registry again.
    process = {
        "process_graph": workflow_registry.freeze_process_graph(
            graph, resolve=lambda process_id: bound.record if process_id == bound.record.id else None
        )
    }
    resolved = {**reference.arguments, bound.interval_parameter: [start, end]}
    if versions:
        resolved["$feature_versions"] = versions
    return process, resolved
