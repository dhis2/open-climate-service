"""The automation graph: how schedules, workflow triggers and pipelines connect.

Assembled from what OCS already records, so it needs no configuration of its own:

- a sync schedule keeps a dataset current from its data source;
- a workflow trigger runs when a dataset updates, reads further datasets and feature
  collections, and publishes a dataset or delivers through a named export;
- a pipeline runs when its input dataset updates (or by hand) and delivers to a destination.

The graph is configuration, not history: what is set up to run, with the latest run of each
step as its status. ``GET /automation/graph`` returns it as JSON, and the Automation page draws
the same graph as SVG.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from markupsafe import Markup, escape
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

NodeKind = Literal["source", "dataset", "features", "trigger", "pipeline", "destination"]
EdgeKind = Literal["updates", "input", "delivers"]
StatusState = Literal["ok", "idle", "waiting", "running", "failed", "paused", "dry_run"]


class NodeStatus(BaseModel):
    """The latest run of a step, reduced to a state a page can colour and a short label."""

    state: StatusState
    label: str


class GraphNode(BaseModel):
    """One step: a data source, dataset, feature collection, trigger, pipeline or destination."""

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
    kind: EdgeKind = Field(description="updates: drives the target; input: read by it; delivers: sent to it.")
    label: str | None = Field(
        default=None, description="When this edge fires: a sync schedule, 'on update' or 'by hand'."
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

    def edge(self, source: str, target: str, kind: EdgeKind, label: str | None = None) -> None:
        if not any(e.source == source and e.target == target for e in self.edges):
            self.edges.append(GraphEdge(source=source, target=target, kind=kind, label=label))

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
    return GraphNode(
        id=f"source:{label}",
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
) -> AutomationGraph:
    """Build the graph from already-read inputs. ``trigger_jobs`` holds each trigger's latest job."""
    b = _Builder()

    for schedule in schedules:
        record = datasets.get(schedule.dataset_id)
        node = _dataset_node(schedule.dataset_id, datasets)
        node.meta = [*node.meta, _schedule_text(schedule)]
        node.status = _schedule_status(schedule)
        dataset = b.node(node)
        if record is not None:
            source = b.node(_source_node(record, schedule.dataset_id))
            b.edge(source, dataset, "updates", _schedule_label(schedule))

    for trigger in triggers:
        arguments = dict(trigger.arguments or {})
        output = arguments.get("output_dataset_id")
        trigger_node = b.node(
            GraphNode(
                id=f"trigger:{trigger.id}",
                kind="trigger",
                label=trigger.id,
                ref=trigger.id,
                meta=[f"Runs {trigger.workflow_id}"],
                status=_job_status(trigger_jobs.get(trigger.id)),
                href=f"/workflows/{trigger.workflow_id}",
            )
        )
        b.edge(b.node(_dataset_node(trigger.on_update_of, datasets)), trigger_node, "updates", "on update")
        inputs = {k: v for k, v in arguments.items() if k != "output_dataset_id"}
        for dataset_id in dict.fromkeys(_literal_datasets(inputs, datasets)):
            if dataset_id != trigger.on_update_of:
                b.edge(b.node(_dataset_node(dataset_id, datasets)), trigger_node, "input")
        for collection_id in dict.fromkeys(_feature_refs(inputs)):
            b.edge(b.node(_features_node(collection_id, feature_names)), trigger_node, "input")
        if isinstance(output, str) and not output.startswith("$"):
            b.edge(trigger_node, b.node(_dataset_node(output, datasets)), "updates")
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
            b.edge(trigger_node, destination, "delivers")

    for record in pipelines:
        spec = record.spec
        pipeline_node = b.node(
            GraphNode(
                id=f"pipeline:{spec.id}",
                kind="pipeline",
                label=spec.name or spec.id,
                ref=spec.id,
                meta=[f"Runs {spec.workflow.id}", "On update" if spec.delivery.policy == "on_update" else "By hand"],
                status=_pipeline_status(record),
                href=f"/pipelines/{spec.id}",
            )
        )
        dataset_ids, collection_ids = _pipeline_inputs(record, datasets)
        drive: EdgeKind = "updates" if spec.delivery.policy == "on_update" else "input"
        for dataset_id in dataset_ids:
            b.edge(
                b.node(_dataset_node(dataset_id, datasets)),
                pipeline_node,
                drive,
                "on update" if drive == "updates" else "by hand",
            )
        for collection_id in collection_ids:
            b.edge(b.node(_features_node(collection_id, feature_names)), pipeline_node, "input")
        connection = spec.destination.settings.get("connection")
        key = f"{spec.destination.plugin}:{connection}" if connection else spec.destination.plugin
        destination = b.node(
            GraphNode(
                id=f"destination:{key}",
                kind="destination",
                label=spec.destination.plugin.upper()
                if spec.destination.plugin == "dhis2"
                else spec.destination.plugin,
                ref=key,
                meta=[f"Connection {connection}"] if connection else [],
            )
        )
        b.edge(pipeline_node, destination, "delivers")

    return b.graph()


def build_graph() -> AutomationGraph:
    """Read the instance's schedules, triggers, pipelines and datasets, and assemble the graph."""
    return assemble_graph(
        datasets=_read_datasets(),
        feature_names=_read_feature_names(),
        schedules=_read_schedules(),
        triggers=_read_triggers(),
        trigger_jobs=_read_trigger_jobs(),
        pipelines=_read_pipelines(),
    )


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


def focus(graph: AutomationGraph, node_id: str) -> AutomationGraph:
    """The part of *graph* one step belongs to: everything it comes from and everything it feeds.

    The same set the Overview lights up on hover, so a pipeline's page shows its own chain and
    nothing beside it. Empty when the step is not in the graph.
    """
    if not any(node.id == node_id for node in graph.nodes):
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

    before, after = reach(up), reach(down)
    keep = before | after
    return AutomationGraph(
        nodes=[node for node in graph.nodes if node.id in keep],
        edges=[
            edge
            for edge in graph.edges
            if (edge.source in before and edge.target in before) or (edge.source in after and edge.target in after)
        ],
    )


# --- layout --------------------------------------------------------------------------------

_BASE_LAYER: dict[str, int] = {"source": 0, "dataset": 1, "features": 1, "trigger": 2, "pipeline": 2, "destination": 2}


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

_NODE_W, _NODE_H = 224, 84
_COL_GAP, _ROW_GAP, _HEAD = 96, 16, 30
_KIND_LABELS = {
    "source": "Data source",
    "dataset": "Dataset",
    "features": "Feature collection",
    "trigger": "Trigger",
    "pipeline": "Pipeline",
    "destination": "Destination",
}


def _column_title(kinds: list[str], index: int) -> str:
    if all(k == "source" for k in kinds):
        return "Data sources"
    if all(k == "destination" for k in kinds):
        return "Destinations"
    if all(k in ("trigger", "pipeline") for k in kinds):
        return "Workflows"
    if all(k in ("dataset", "features") for k in kinds):
        return "Datasets" if index <= 1 else "Derived datasets"
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_svg(
    graph: AutomationGraph,
    mount: str = "",
    *,
    current: str | None = None,
    current_href: str | None = None,
) -> Markup:
    """Draw the graph as an inline SVG, laid out in columns, each node linking to its page.

    ``current`` marks the step whose page the diagram sits on. A link from it to that same page
    would go nowhere, so ``current_href`` gives it somewhere useful instead, such as the
    workflow a pipeline runs.
    """
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
        'role="img" aria-labelledby="automation-graph-title">',
        '<title id="automation-graph-title">How schedules, triggers and pipelines connect on this instance</title>',
        "<defs>",
        *(
            f'<marker id="ag-arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ag-arrow ag-arrow-{kind}" d="M0,0 L10,5 L0,10 z"/></marker>'
            for kind in ("updates", "input", "delivers", "lit")
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
            f'data-target="{escape(edge.target)}" marker-end="url(#ag-arrow-{edge.kind})" '
            f'd="M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
        )
        if edge.label:
            # At the curve's midpoint, so two arrows leaving one box keep their labels apart.
            parts.append(
                f'<text class="ag-edge-text" data-source="{escape(edge.source)}" '
                f'data-target="{escape(edge.target)}" x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 5:.1f}" '
                f'text-anchor="middle">{escape(edge.label)}</text>'
            )

    for node in graph.nodes:
        x, y = box(node.id)
        lines = [
            f'<text class="ag-kind" x="{x + 10}" y="{y + 16}">{escape(_KIND_LABELS[node.kind].upper())}</text>',
            f'<text class="ag-label" x="{x + 10}" y="{y + 34}">{escape(_clip(node.label, 29))}</text>',
        ]
        detail = node.meta[0] if node.meta else (node.ref if node.ref and node.ref != node.label else "")
        if detail:
            lines.append(f'<text class="ag-meta" x="{x + 10}" y="{y + 52}">{escape(_clip(detail, 36))}</text>')
        if node.status is not None:
            label = _clip(node.status.label, 30)
            chip_w = 18 + 6.2 * len(label)
            lines.append(
                f'<g class="ag-status ag-{node.status.state}"><rect x="{x + 10}" y="{y + 61}" rx="9" '
                f'width="{chip_w:.0f}" height="17"/><text x="{x + 19}" y="{y + 73}">{escape(label)}</text></g>'
            )
        tooltip = " · ".join([node.label, *node.meta, *([node.status.label] if node.status else [])])
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
