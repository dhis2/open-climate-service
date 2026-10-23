"""The automation graph: how schedules, workflow triggers and pipelines connect.

Assembled from what OCS already records, so it needs no configuration of its own:

- a sync schedule keeps a dataset current from its data source;
- a workflow trigger runs when a dataset updates, reads further datasets and feature
  collections, and publishes a dataset or delivers through a named export;
- a pipeline runs when its input dataset updates (or by hand) and delivers to a destination.

The graph is configuration, not history: what is set up to run, with the latest run of each
step as its status. ``GET /automation/graph`` returns it as JSON; the Pipelines pages draw each
pipeline's part of it as SVG (``pipeline_focus``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from markupsafe import Markup, escape
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

NodeKind = Literal["source", "dataset", "features", "workflow", "destination"]
EdgeKind = Literal["origin", "updates", "input", "delivers"]
FiresOn = Literal["schedule", "update", "manual"]
StatusState = Literal["ok", "idle", "waiting", "running", "failed", "paused", "dry_run"]


class NodeStatus(BaseModel):
    """The latest run of a step, reduced to a state a page can colour and a short label."""

    state: StatusState
    label: str


class GraphNode(BaseModel):
    """One component: a data source, dataset, feature collection, workflow or destination."""

    id: str = Field(description="Unique within the graph, prefixed by kind (dataset:era5land_temperature_daily).")
    kind: NodeKind
    label: str = Field(description="Display name.")
    ref: str | None = Field(default=None, description="The id the node stands for (dataset id, trigger id).")
    meta: list[str] = Field(default_factory=list, description="Short facts: coverage, schedule, workflow.")
    status: NodeStatus | None = None
    href: str | None = Field(default=None, description="Path of the node's own page, relative to the mount.")


class GraphEdge(BaseModel):
    """How one step reaches the next."""

    source: str
    target: str
    kind: EdgeKind = Field(
        description="origin: provenance; updates: drives the target; input: read by it; delivers: sent to it."
    )
    label: str | None = Field(
        default=None, description="When this edge fires: a sync schedule, 'on update' or 'by hand'."
    )
    fires: FiresOn | None = Field(
        default=None,
        description="What starts the target: its sync schedule, an update of the source, or a person.",
    )


class AutomationGraph(BaseModel):
    """Every configured path through the instance's automation, as nodes and edges."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# --- assembly -----------------------------------------------------------------------------


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def node(self, node: GraphNode) -> str:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
        elif node.status is not None and existing.status is None:
            # A dataset first met as an input can later turn out to be scheduled.
            self.nodes[node.id] = node
        return node.id

    def edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        label: str | None = None,
        fires: FiresOn | None = None,
    ) -> None:
        if not any(e.source == source and e.target == target for e in self.edges):
            self.edges.append(GraphEdge(source=source, target=target, kind=kind, label=label, fires=fires))

    def graph(self) -> AutomationGraph:
        return AutomationGraph(nodes=list(self.nodes.values()), edges=self.edges)


def _when(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        moment = value
    return moment.strftime("%d %b %H:%M UTC")


def _coverage(record: Any) -> str | None:
    temporal = getattr(getattr(record, "extent", None), "temporal", None)
    start, end = getattr(temporal, "start", None), getattr(temporal, "end", None)
    if start and end:
        return f"{start} to {end}"
    return None


def _dataset_node(dataset_id: str, datasets: Mapping[str, Any]) -> GraphNode:
    record = datasets.get(dataset_id)
    coverage = _coverage(record) if record is not None else None
    return GraphNode(
        id=f"dataset:{dataset_id}",
        kind="dataset",
        label=getattr(record, "dataset_name", None) or dataset_id,
        ref=dataset_id,
        meta=[coverage] if coverage else [],
        href=f"/datasets/{dataset_id}" if record is not None else None,
    )


def _schedule_status(schedule: Any) -> NodeStatus:
    outcome = str(getattr(schedule, "last_outcome", "") or "")
    last = getattr(schedule, "last_check", None)
    if not getattr(schedule, "enabled", True):
        return NodeStatus(state="paused", label="Schedule paused")
    if outcome == "error":
        return NodeStatus(state="failed", label=f"Check failed {_when(last)}".strip())
    if outcome == "submitted":
        return NodeStatus(state="ok", label=f"Checked {_when(last)}")
    if outcome:
        return NodeStatus(state="idle", label=f"{outcome.replace('_', ' ').capitalize()} {_when(last)}".strip())
    next_check = getattr(schedule, "next_check", None)
    return NodeStatus(state="waiting", label=f"Next check {_when(next_check)}" if next_check else "No checks yet")


_WEEKDAY_PLURALS = {
    "mon": "Mondays",
    "tue": "Tuesdays",
    "wed": "Wednesdays",
    "thu": "Thursdays",
    "fri": "Fridays",
    "sat": "Saturdays",
    "sun": "Sundays",
}


def _schedule_label(schedule: Any) -> str:
    """When a sync schedule checks, short enough to sit on an arrow: 'daily 06:00 UTC'."""
    from open_climate_service.scheduler.presets import form_values

    if not getattr(schedule, "enabled", True):
        return "sync paused"
    tz = schedule.timezone
    values = form_values(schedule.cron, None)
    clock = values.get("check_time")
    frequency = values.get("frequency")
    if frequency == "daily":
        return f"daily {clock} {tz}"
    if frequency == "weekly":
        day = _WEEKDAY_PLURALS.get(str(values.get("weekday")), str(values.get("weekday")))
        return f"{day} {clock} {tz}"
    if frequency == "monthly":
        return f"monthly, day {values.get('month_day')} {clock} {tz}"
    every = _interval_label(schedule.cron)
    return every if every else f"cron {schedule.cron} {tz}"


def _interval_label(cron: str) -> str | None:
    """'every 5 min' for '*/5 * * * *', 'every 6 h' for '0 */6 * * *'; None for anything else."""
    fields = cron.split()
    if len(fields) != 5 or fields[2:] != ["*", "*", "*"]:
        return None
    minute, hour = fields[0], fields[1]
    if minute.startswith("*/") and hour == "*" and minute[2:].isdigit():
        return f"every {minute[2:]} min"
    if minute.isdigit() and hour.startswith("*/") and hour[2:].isdigit():
        return f"every {hour[2:]} h"
    return None


def _schedule_text(schedule: Any) -> str:
    from open_climate_service.scheduler.presets import schedule_description

    described = schedule_description(schedule.cron, schedule.timezone)
    return f"Sync: {described}" if described else f"Sync: {schedule.cron} ({schedule.timezone})"


def _source_node(record: Any, dataset_id: str) -> GraphNode:
    label = getattr(record, "source", None) or getattr(record, "source_dataset_id", None) or dataset_id
    source_id = getattr(record, "source_dataset_id", None)
    # Keyed on the template, not its display name: six ERA5-Land templates share the name
    # "ERA5-Land Reanalysis", and one box for all of them would link to whichever came first.
    return GraphNode(
        id=f"source:{source_id or label}",
        kind="source",
        label=str(label),
        href=f"/data-sources/{source_id}" if source_id else None,
    )


def _job_status(job: Any | None) -> NodeStatus:
    if job is None:
        return NodeStatus(state="waiting", label="Waiting for an update")
    status = str(getattr(getattr(job, "status", None), "value", getattr(job, "status", "")))
    when = _when(getattr(job, "updated", None) or getattr(job, "created", None))
    if status == "finished":
        return NodeStatus(state="ok", label=f"Ran {when}")
    if status == "error":
        return NodeStatus(state="failed", label=f"Failed {when}")
    if status in ("queued", "running", "created"):
        return NodeStatus(state="running", label=status.capitalize())
    return NodeStatus(state="idle", label=f"{status.capitalize()} {when}".strip())


def _pipeline_status(record: Any) -> NodeStatus:
    if record.spec.delivery.mode == "paused":
        return NodeStatus(state="paused", label="Paused")
    if not record.runs:
        return NodeStatus(state="waiting", label="No runs yet")
    run = record.runs[0]
    if run.job_outcome is not None or run.submission == "failed" or run.delivery == "failed":
        return NodeStatus(state="failed", label=f"Failed {_when(run.submitted_at)}")
    if run.delivery_result == "delivered":
        return NodeStatus(state="ok", label=f"Delivered {_when(run.delivered_at or run.submitted_at)}")
    if run.delivery_result == "validated":
        return NodeStatus(state="dry_run", label=f"Dry run {_when(run.submitted_at)}")
    if run.delivery_result is not None:
        return NodeStatus(state="failed", label=f"Delivery {run.delivery_result}")
    return NodeStatus(state="running", label="Running")


def _literal_datasets(value: Any, known: Mapping[str, Any]) -> Iterable[str]:
    """Dataset ids a trigger argument names literally: a string that is a dataset id."""
    if isinstance(value, str):
        if value in known:
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _literal_datasets(item, known)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _literal_datasets(item, known)


def _feature_refs(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        if isinstance(value.get("from_features"), str):
            yield value["from_features"]
        else:
            for item in value.values():
                yield from _feature_refs(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _feature_refs(item)


def _features_node(collection_id: str, feature_names: Mapping[str, str]) -> GraphNode:
    return GraphNode(
        id=f"features:{collection_id}",
        kind="features",
        label=feature_names.get(collection_id, collection_id),
        ref=collection_id,
        href=f"/features/{collection_id}",
    )


def _pipeline_inputs(record: Any, datasets: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The datasets and feature collections a pipeline's workflow binds."""
    try:
        from open_climate_service.pipelines.workflow import bind

        bound = bind(record.spec.workflow)
        return list(bound.dataset_ids), list(bound.feature_collections)
    except Exception:
        # A pipeline whose workflow no longer binds still shows, with what its arguments name.
        arguments = record.spec.workflow.arguments
        return list(dict.fromkeys(_literal_datasets(arguments, datasets))), list(_feature_refs(arguments))


def assemble_graph(
    *,
    datasets: Mapping[str, Any],
    feature_names: Mapping[str, str],
    schedules: Sequence[Any],
    triggers: Sequence[Any],
    trigger_jobs: Mapping[str, Any],
    pipelines: Sequence[Any],
    destination_titles: Mapping[str, str] | None = None,
    workflow_titles: Mapping[str, str] | None = None,
    ingested: Collection[str] = (),
) -> AutomationGraph:
    """Build the graph from already-read inputs.

    ``trigger_jobs`` holds each trigger's latest job. ``ingested`` names the datasets fetched
    from a data source, as opposed to produced by a workflow: only those get a data source box.
    """
    b = _Builder()
    destination_titles = destination_titles or {}
    workflow_titles = workflow_titles or {}

    for schedule in schedules:
        record = datasets.get(schedule.dataset_id)
        node = _dataset_node(schedule.dataset_id, datasets)
        node.meta = [*node.meta, _schedule_text(schedule)]
        node.status = _schedule_status(schedule)
        dataset = b.node(node)
        if record is not None:
            source = b.node(_source_node(record, schedule.dataset_id))
            b.edge(source, dataset, "updates", _schedule_label(schedule), "schedule")

    for trigger in triggers:
        arguments = dict(trigger.arguments or {})
        output = arguments.get("output_dataset_id")
        workflow_title = workflow_titles.get(trigger.workflow_id) or str(trigger.workflow_id)
        workflow_node = b.node(
            GraphNode(
                id=f"workflow:trigger:{trigger.id}",
                kind="workflow",
                label=workflow_title,
                ref=trigger.workflow_id,
                meta=[trigger.workflow_id],
                status=_job_status(trigger_jobs.get(trigger.id)),
                href=f"/workflows/{trigger.workflow_id}",
            )
        )
        b.edge(b.node(_dataset_node(trigger.on_update_of, datasets)), workflow_node, "updates", "on update", "update")
        inputs = {k: v for k, v in arguments.items() if k != "output_dataset_id"}
        for dataset_id in dict.fromkeys(_literal_datasets(inputs, datasets)):
            if dataset_id != trigger.on_update_of:
                b.edge(b.node(_dataset_node(dataset_id, datasets)), workflow_node, "input")
        for collection_id in dict.fromkeys(_feature_refs(inputs)):
            b.edge(b.node(_features_node(collection_id, feature_names)), workflow_node, "input")
        if isinstance(output, str) and not output.startswith("$"):
            b.edge(workflow_node, b.node(_dataset_node(output, datasets)), "updates")
        deliver = getattr(trigger, "deliver", None)
        if deliver is not None:
            destination = b.node(
                GraphNode(
                    id=f"destination:export:{deliver.export}",
                    kind="destination",
                    label=deliver.export,
                    ref=deliver.export,
                    meta=["Named export", "Dry run" if deliver.dry_run else "Live"],
                )
            )
            b.edge(workflow_node, destination, "delivers")

    for record in pipelines:
        spec = record.spec
        workflow_title = workflow_titles.get(spec.workflow.id) or str(spec.workflow.id)
        workflow_node = b.node(
            GraphNode(
                id=f"workflow:pipeline:{spec.id}",
                kind="workflow",
                label=workflow_title,
                ref=spec.workflow.id,
                meta=[spec.workflow.id, "On update" if spec.delivery.policy == "on_update" else "By hand"],
                status=_pipeline_status(record),
                href=f"/workflows/{spec.workflow.id}",
            )
        )
        dataset_ids, collection_ids = _pipeline_inputs(record, datasets)
        drive: EdgeKind = "updates" if spec.delivery.policy == "on_update" else "input"
        for dataset_id in dataset_ids:
            b.edge(
                b.node(_dataset_node(dataset_id, datasets)),
                workflow_node,
                drive,
                "on update" if drive == "updates" else "by hand",
                "update" if drive == "updates" else "manual",
            )
        for collection_id in collection_ids:
            b.edge(b.node(_features_node(collection_id, feature_names)), workflow_node, "input")
        # Destination settings are plugin-owned and opaque to the graph. A destination node is
        # scoped to its pipeline until plugins expose a generic, credential-free display binding;
        # guessing a target from a setting such as ``connection`` would make the core integration-
        # specific and could incorrectly merge two unknown destinations.
        destination_id = f"destination:pipeline:{spec.id}"
        destination_title = destination_titles.get(spec.destination.plugin) or str(spec.destination.plugin)
        destination = b.node(
            GraphNode(
                id=destination_id,
                kind="destination",
                label=destination_title,
                ref=spec.destination.plugin,
            )
        )
        b.edge(workflow_node, destination, "delivers")

    _add_provenance(b, datasets, ingested)
    return b.graph()


def _add_provenance(b: _Builder, datasets: Mapping[str, Any], ingested: Collection[str]) -> None:
    """Give every ingested dataset in the graph its data source, wherever the graph reaches it.

    Decided once, after every step is in: a dataset only a trigger reads needs its origin as much
    as one a pipeline reads. A scheduled dataset already has its source, through the edge that
    carries the schedule; any other gets a plain ``origin`` edge, so the clock stays on scheduled
    datasets only. A dataset a workflow produces has no data source of its own.
    """
    produced = {edge.target for edge in b.edges if edge.source.startswith("workflow:")}
    sourced = {edge.target for edge in b.edges if edge.source.startswith("source:")}
    for node in list(b.nodes.values()):
        if node.kind != "dataset" or node.ref is None or node.id in produced or node.id in sourced:
            continue
        record = datasets.get(node.ref)
        if record is None or node.ref not in ingested:
            continue
        b.edge(b.node(_source_node(record, node.ref)), node.id, "origin")


def build_graph() -> AutomationGraph:
    """Read the instance's schedules, triggers, pipelines and datasets, and assemble the graph."""
    datasets = _read_datasets()
    return assemble_graph(
        datasets=datasets,
        feature_names=_read_feature_names(),
        schedules=_read_schedules(),
        triggers=_read_triggers(),
        trigger_jobs=_read_trigger_jobs(),
        pipelines=_read_pipelines(),
        destination_titles=_read_destination_titles(),
        workflow_titles=_read_workflow_titles(),
        ingested=_read_ingested(datasets),
    )


def _read_ingested(datasets: Mapping[str, Any]) -> set[str]:
    """The datasets whose data source template fetches them, rather than a workflow producing them.

    A workflow's output gets an auto-generated template that copies ``source`` from its input's,
    so the name alone cannot tell the two apart; the registry's ``is_ingestable`` rule can.
    """
    try:
        from open_climate_service.data_registry.services import datasets as registry

        templates = {template["id"]: template for template in registry.list_datasets()}
        ingested: set[str] = set()
        for dataset_id, record in datasets.items():
            template = templates.get(getattr(record, "source_dataset_id", None) or dataset_id)
            if template is not None and registry.is_ingestable(template):
                ingested.add(dataset_id)
        return ingested
    except Exception:
        _log.exception("Data source templates could not be read for the automation graph")
        return set()


def _read_destination_titles() -> dict[str, str]:
    """Plugin-owned destination titles, without interpreting any destination settings."""
    try:
        from open_climate_service.pipelines.service import destination_plugins

        titles: dict[str, str] = {}
        for plugin_id, plugin in destination_plugins().items():
            presentation = plugin.pipeline_settings_form()
            titles[plugin_id] = presentation.title if presentation is not None else plugin_id
        return titles
    except Exception:
        _log.exception("Destination plugin titles could not be read for the automation graph")
        return {}


def _read_workflow_titles() -> dict[str, str]:
    """Human-readable workflow titles from the workflow registry."""
    try:
        from open_climate_service.openeo.workflows import list_workflows

        return {record.id: record.summary or record.id for record in list_workflows().processes}
    except Exception:
        _log.exception("Workflow titles could not be read for the automation graph")
        return {}


def _read_datasets() -> dict[str, Any]:
    try:
        from open_climate_service.ingestions.services import list_datasets

        return {item.dataset_id: item for item in list_datasets().items}
    except Exception:
        _log.exception("Datasets could not be listed for the automation graph")
        return {}


def _read_feature_names() -> dict[str, str]:
    try:
        from open_climate_service.features.services import list_feature_collections

        return {item.id: item.name for item in list_feature_collections().items}
    except Exception:
        _log.exception("Feature collections could not be listed for the automation graph")
        return {}


def _read_schedules() -> list[Any]:
    try:
        from open_climate_service.scheduler.service import get_scheduler_service

        return list(get_scheduler_service().status().schedules)
    except Exception:
        _log.exception("Schedules could not be read for the automation graph")
        return []


def _read_triggers() -> list[Any]:
    try:
        from open_climate_service.automation.config import get_automation_config

        return list(get_automation_config().workflow_triggers)
    except Exception:
        _log.exception("Workflow triggers could not be read for the automation graph")
        return []


def _read_trigger_jobs() -> dict[str, Any]:
    try:
        from open_climate_service.openeo.jobs import read_job_records

        latest: dict[str, Any] = {}
        for job in read_job_records():
            if job.trigger_id and (job.trigger_id not in latest or job.created > latest[job.trigger_id].created):
                latest[job.trigger_id] = job
        return latest
    except Exception:
        _log.exception("Triggered jobs could not be read for the automation graph")
        return {}


def _read_pipelines() -> list[Any]:
    try:
        from open_climate_service.pipelines.store import list_records

        return list_records()
    except Exception:
        _log.exception("Pipelines could not be read for the automation graph")
        return []


def pipeline_focus(graph: AutomationGraph, pipeline_id: str) -> AutomationGraph:
    """The end-to-end lineage that feeds one pipeline's destination.

    The enclosing diagram is the pipeline. Each executable box inside it is a workflow,
    including an upstream workflow activated by a dataset update. Update and schedule semantics
    stay on the connecting arrows rather than becoming component names.
    """
    node_id = f"workflow:pipeline:{pipeline_id}"
    kinds = {node.id: node.kind for node in graph.nodes}
    if kinds.get(node_id) != "workflow":
        return AutomationGraph()

    up: dict[str, list[str]] = defaultdict(list)
    down: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        up[edge.target].append(edge.source)
        down[edge.source].append(edge.target)

    def reach(links: Mapping[str, list[str]]) -> set[str]:
        seen, queue = {node_id}, [node_id]
        while queue:
            for nxt in links.get(queue.pop(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    keep = reach(up) | reach(down)
    return AutomationGraph(
        nodes=[node for node in graph.nodes if node.id in keep],
        edges=[edge for edge in graph.edges if edge.source in keep and edge.target in keep],
    )


# --- layout --------------------------------------------------------------------------------

_BASE_LAYER: dict[str, int] = {
    "source": 0,
    "dataset": 1,
    "features": 1,
    "workflow": 2,
    "destination": 2,
}


def layout(graph: AutomationGraph) -> dict[str, tuple[int, int]]:
    """Place each node in a layer (column) and a row within it.

    A node's layer is one past its furthest predecessor, so a chain through a derived dataset
    reads left to right; destinations go to the last layer. Rows are ordered by the mean row of
    the steps that drive each node (its inputs when nothing drives it), which keeps most edges
    short and uncrossed.
    """
    preds: dict[str, list[str]] = defaultdict(list)
    drivers: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        preds[edge.target].append(edge.source)
        if edge.kind != "input":
            drivers[edge.target].append(edge.source)
    kinds = {node.id: node.kind for node in graph.nodes}
    layer: dict[str, int] = {node.id: _BASE_LAYER[node.kind] for node in graph.nodes}
    for _ in range(len(graph.nodes)):  # longest path, bounded in case of a cycle
        changed = False
        for node in graph.nodes:
            wanted = max([_BASE_LAYER[node.kind], *(layer[p] + 1 for p in preds[node.id] if p in layer)])
            if wanted != layer[node.id]:
                layer[node.id], changed = wanted, True
        if not changed:
            break
    # Destinations share one column, past every other step, so each delivery reads as an end.
    last = max((layer[n] for n, kind in kinds.items() if kind != "destination"), default=0) + 1
    for node_id, kind in kinds.items():
        if kind == "destination":
            layer[node_id] = last

    # A focused pipeline can begin with an unscheduled or derived dataset, so no source node
    # occupies layer zero. Remove empty leading (or intermediate) layers instead of rendering
    # unexplained whitespace where a Data source would otherwise appear.
    occupied = {old: new for new, old in enumerate(sorted(set(layer.values())))}
    layer = {node_id: occupied[index] for node_id, index in layer.items()}

    columns: dict[int, list[str]] = defaultdict(list)
    for node in graph.nodes:
        columns[layer[node.id]].append(node.id)
    row: dict[str, float] = {}
    for index in sorted(columns):
        ids = columns[index]
        first_seen = {node_id: position for position, node_id in enumerate(ids)}
        for position, node_id in enumerate(ids):
            # Order by what drives a step, not by what it reads: a trigger sits beside the
            # dataset whose update runs it, not halfway towards a normal it also reads.
            parents = [row[p] for p in (drivers[node_id] or preds[node_id]) if p in row]
            row[node_id] = sum(parents) / len(parents) if parents else float(position)
        ids.sort(key=lambda n: (row[n], first_seen[n]))
        # Keep each node level with what drives it where the column allows, leaving a gap
        # rather than packing the column to the top and bending every edge.
        next_free = 0
        for node_id in ids:
            row[node_id] = float(max(next_free, round(row[node_id])))
            next_free = int(row[node_id]) + 1
    return {node_id: (layer[node_id], int(row[node_id])) for node_id in layer}


# --- rendering -----------------------------------------------------------------------------

_NODE_W, _NODE_H = 224, 70
_COL_GAP, _ROW_GAP, _HEAD = 96, 16, 30
_KIND_LABELS = {
    "source": "Data source",
    "dataset": "Dataset",
    "features": "Feature collection",
    "workflow": "Workflow",
    "destination": "Destination",
}


def _column_title(kinds: list[str], index: int) -> str:
    """The heading over a column, singular when the column holds one box."""
    one = len(kinds) == 1
    if all(k == "source" for k in kinds):
        return "Data source" if one else "Data sources"
    if all(k == "destination" for k in kinds):
        return "Destination" if one else "Destinations"
    if all(k == "workflow" for k in kinds):
        return "Workflow" if one else "Workflows"
    if all(k in ("dataset", "features") for k in kinds):
        if index <= 1:
            if one:
                return "Dataset" if kinds[0] == "dataset" else "Input"
            return "Datasets and inputs"
        return "Derived dataset" if one else "Derived datasets"
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Symbols for what starts a step, drawn 10 units square with the origin at the top left.
_FIRES_ICONS = {
    "schedule": '<circle cx="5" cy="5" r="4.2"/><path d="M5 2.6 V5 L6.8 6.2"/>',
    "update": '<path class="ag-icon-solid" d="M6.2 0.4 L1.6 5.6 H4.6 L3.6 9.6 L8.4 4.2 H5.4 Z"/>',
    "manual": '<path d="M3.4 9.4 V3.2 a1 1 0 0 1 2 0 V6 M5.4 5.4 a1 1 0 0 1 2 0 V6.6 M7.4 6 a1 1 0 0 1 2 0 '
    'V7.6 C9.4 9 8.4 9.8 7 9.8 H5.2 L2 7.2 a0.9 0.9 0 0 1 1.4 -1"/>',
}
_ICON_SIZE = 14
_FIRES_TITLES = {
    "schedule": "Runs on its sync schedule",
    "update": "Runs when the dataset updates",
    "manual": "Runs when someone starts it",
}


def render_legend() -> Markup:
    """The diagram's key: the three kinds of arrow, and the symbols for what starts a step."""
    lines = (
        ("origin", "comes from"),
        ("updates", "updates"),
        ("input", "read as an input"),
        ("delivers", "delivers"),
    )
    icons = (("schedule", "sync schedule"), ("update", "runs on update"), ("manual", "runs by hand"))
    parts = [f'<span><i class="ag-key ag-key-{kind}"></i>{escape(text)}</span>' for kind, text in lines]
    parts += [
        f'<span><svg class="ag-key-icon" width="14" height="14" viewBox="-0.5 -0.5 11 11" aria-hidden="true">'
        f'<g class="ag-icon">{_FIRES_ICONS[kind]}</g></svg>{escape(text)}</span>'
        for kind, text in icons
    ]
    return Markup('<div class="automation-legend" aria-label="Legend">{}</div>').format(Markup("".join(parts)))


def _edge_label(edge: GraphEdge, mid_x: float, y: float) -> str:
    """An arrow's label, centred on the arrow, under the symbol for what fires it."""
    label = edge.label or ""
    icon = _FIRES_ICONS.get(edge.fires or "")
    data = f'data-source="{escape(edge.source)}" data-target="{escape(edge.target)}"'
    parts = [f'<g class="ag-edge-text" {data}>']
    if icon:
        # The 10-unit symbol drawn at 14px, its foot just clear of the label.
        left, top = mid_x - _ICON_SIZE / 2, y - 11 - _ICON_SIZE
        parts.append(
            f'<g class="ag-icon" transform="translate({left:.1f},{top:.1f}) scale({_ICON_SIZE / 10:g})">'
            f"<title>{escape(_FIRES_TITLES[edge.fires or ''])}</title>{icon}</g>"
        )
    parts.append(f'<text x="{mid_x:.1f}" y="{y:.1f}" text-anchor="middle">{escape(label)}</text></g>')
    return "".join(parts)


def render_svg(
    graph: AutomationGraph,
    mount: str = "",
    *,
    current: str | None = None,
    current_href: str | None = None,
    key: str = "",
) -> Markup:
    """Draw the graph as an inline SVG, laid out in columns, each node linking to its page.

    ``current`` marks the step whose page the diagram sits on. A link from it to that same page
    would go nowhere, so ``current_href`` gives it somewhere useful instead, such as the
    workflow a pipeline runs. ``key`` keeps the ids inside the SVG unique when a page draws more
    than one diagram.
    """
    suffix = f"-{key}" if key else ""
    if not graph.nodes:
        return Markup("")
    places = layout(graph)
    layers = max(layer for layer, _ in places.values()) + 1
    rows = max(row for _, row in places.values()) + 1
    width = layers * _NODE_W + (layers - 1) * _COL_GAP
    height = _HEAD + rows * _NODE_H + (rows - 1) * _ROW_GAP

    def box(node_id: str) -> tuple[float, float]:
        layer, row = places[node_id]
        return layer * (_NODE_W + _COL_GAP), _HEAD + row * (_NODE_H + _ROW_GAP)

    parts: list[str] = [
        f'<svg class="automation-graph" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="group" aria-labelledby="automation-graph-title{suffix}">',
        f'<title id="automation-graph-title{suffix}">How data moves through this pipeline</title>',
        "<defs>",
        *(
            f'<marker id="ag-arrow-{kind}{suffix}" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path class="ag-arrow ag-arrow-{kind}" d="M0,0 L10,5 L0,10 z"/></marker>'
            for kind in ("origin", "updates", "input", "delivers", "lit")
        ),
        "</defs>",
    ]
    by_layer: dict[int, list[str]] = defaultdict(list)
    kinds = {node.id: node.kind for node in graph.nodes}
    for node_id, (layer, _) in places.items():
        by_layer[layer].append(kinds[node_id])
    for layer in range(layers):
        title = _column_title(by_layer[layer], layer)
        if title:
            column_x = layer * (_NODE_W + _COL_GAP)
            parts.append(f'<text class="ag-column" x="{column_x}" y="14">{escape(title.upper())}</text>')

    for edge in graph.edges:
        if edge.source not in places or edge.target not in places:
            continue
        sx, sy = box(edge.source)
        tx, ty = box(edge.target)
        x1, y1 = sx + _NODE_W, sy + _NODE_H / 2
        x2, y2 = tx - 2, ty + _NODE_H / 2
        dx = max(24.0, (x2 - x1) / 2)
        parts.append(
            f'<path class="ag-edge ag-edge-{edge.kind}" data-source="{escape(edge.source)}" '
            f'data-target="{escape(edge.target)}" marker-end="url(#ag-arrow-{edge.kind}{suffix})" '
            f'd="M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
        )
        if edge.label:
            # At the curve's midpoint, so two arrows leaving one box keep their labels apart, with a
            # symbol for what starts the step: a clock, an update or a person.
            parts.append(_edge_label(edge, (x1 + x2) / 2, (y1 + y2) / 2 - 5))

    for node in graph.nodes:
        x, y = box(node.id)
        lines = [
            f'<text class="ag-label" x="{x + 10}" y="{y + 21}">{escape(_clip(node.label, 29))}</text>',
        ]
        detail = node.meta[0] if node.meta else (node.ref if node.ref and node.ref != node.label else "")
        if detail:
            lines.append(f'<text class="ag-meta" x="{x + 10}" y="{y + 39}">{escape(_clip(detail, 36))}</text>')
        if node.status is not None:
            label = _clip(node.status.label, 30)
            chip_w = 18 + 6.2 * len(label)
            lines.append(
                f'<g class="ag-status ag-{node.status.state}"><rect x="{x + 10}" y="{y + 47}" rx="9" '
                f'width="{chip_w:.0f}" height="17"/><text x="{x + 19}" y="{y + 59}">{escape(label)}</text></g>'
            )
        tooltip = " · ".join(
            [f"{_KIND_LABELS[node.kind]}: {node.label}", *node.meta, *([node.status.label] if node.status else [])]
        )
        shape = (
            f'<g class="ag-node ag-node-{node.kind}{" ag-current" if node.id == current else ""}" '
            f'data-id="{escape(node.id)}">'
            f"<title>{escape(tooltip)}</title>"
            f'<rect x="{x}" y="{y}" width="{_NODE_W}" height="{_NODE_H}" rx="4"/>' + "".join(lines) + "</g>"
        )
        href = current_href if node.id == current and current_href else node.href
        if href:
            shape = f'<a href="{escape(mount + href)}">{shape}</a>'
        parts.append(shape)
    parts.append("</svg>")
    return Markup("".join(parts))
