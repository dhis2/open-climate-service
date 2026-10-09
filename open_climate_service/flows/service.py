"""The flow graph: what each configured task reads and writes, with each step's latest status (CLIM-1377).

The graph is configuration, not history: it shows what is set up to run, and puts the latest run
of each task on its node. Nodes are data sources, datasets, feature collections, workflow tasks,
exports and destinations; edges say what feeds what. The Automation page draws it as chains, one
row per path from a start to an end, a dataset page draws the chains through that dataset, and
``GET /flows`` returns the graph as JSON.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from open_climate_service.scheduler.presets import schedule_description
from open_climate_service.tasks.models import Task

NodeType = Literal["source", "dataset", "collection", "workflow", "export", "destination"]

COLUMNS: tuple[NodeType, ...] = ("source", "dataset", "collection", "workflow", "export", "destination")
"""Left to right, the order data moves in."""


class FlowNode(BaseModel):
    """One thing data passes through, with how it is kept current and how its last run went."""

    id: str
    type: NodeType
    label: str
    detail: str | None = None
    href: str | None = None
    starts: str | None = None
    task_id: str | None = None
    status: str | None = None


class FlowEdge(BaseModel):
    """Data moving from one node to the next."""

    source: str
    target: str
    label: str | None = None


ORG_UNITS = "org units"
"""The label of an edge that feeds a workflow its org units: drawn on the workflow, not as a step."""


class FlowStep(BaseModel):
    """One box in a chain: the node, what the arrow into it says, and the inputs drawn on it."""

    node: FlowNode
    via: str | None = None
    inputs: list[FlowNode] = []


class FlowGraph(BaseModel):
    """Every configured path, as nodes and edges."""

    nodes: list[FlowNode]
    edges: list[FlowEdge]

    def chains(self, through: str | None = None) -> list[list[FlowStep]]:
        """Every path from a start to an end, as rows of steps, optionally only those through a node.

        A node that more than one path passes through appears in each of them, so every row
        reads on its own from left to right. Org units feeding a workflow are drawn on the
        workflow's box rather than as a row of their own, and a row starts at the dataset or
        collection a sync or refresh keeps current: its box says how, so a source box would
        add nothing.
        """
        nodes = {node.id: node for node in self.nodes}
        main = [edge for edge in self.edges if edge.label != ORG_UNITS and nodes[edge.source].type != "source"]
        inputs: dict[str, list[FlowNode]] = {}
        for edge in self.edges:
            if edge.label == ORG_UNITS:
                inputs.setdefault(edge.target, []).append(nodes[edge.source])
        outgoing: dict[str, list[FlowEdge]] = {}
        for edge in main:
            outgoing.setdefault(edge.source, []).append(edge)
        targets = {edge.target for edge in main}
        joined = {edge.source for edge in main} | targets
        # Org units with no refresh task of their own are only drawn on the workflows they feed.
        side_only = {node.id for row in inputs.values() for node in row if node.task_id is None} - joined
        starts = [
            node.id
            for node in self.nodes
            if node.type != "source" and node.id not in targets and node.id not in side_only
        ]
        rows: list[list[FlowStep]] = []

        def walk(node_id: str, via: str | None, row: list[FlowStep]) -> None:
            row = [*row, FlowStep(node=nodes[node_id], via=via, inputs=inputs.get(node_id, []))]
            seen = {step.node.id for step in row}
            onward = [edge for edge in outgoing.get(node_id, []) if edge.target not in seen]
            if not onward:
                rows.append(row)
            for edge in onward:
                walk(edge.target, edge.label, row)

        for start in starts:
            walk(start, None, [])
        if through is not None:
            rows = [row for row in rows if any(step.node.id == through for step in row)]
        return rows


def _starts(task: Task, timezone: str) -> str:
    """How a task is started, in a few words for its box."""
    if task.cron is not None:
        return schedule_description(task.cron, timezone) or f"On cron {task.cron}"
    if task.after is not None:
        return "After each run" if task.after.task is not None else "After each update"
    return "By hand"


def _latest_status(task_id: str) -> str | None:
    from open_climate_service.runs import service as runs

    latest = runs.list_runs(task_id=task_id, limit=1)
    return runs.view(latest[0]).status if latest else None


def _from_features(arguments: Any) -> list[str]:
    """Feature collections a workflow task's arguments name with ``{from_features: id}``."""
    found: list[str] = []
    if isinstance(arguments, dict):
        if isinstance(arguments.get("from_features"), str):
            return [arguments["from_features"]]
        for value in arguments.values():
            found.extend(_from_features(value))
    elif isinstance(arguments, list):
        for value in arguments:
            found.extend(_from_features(value))
    return found


def build_graph(
    tasks: list[Task], exports: dict[str, dict[str, Any]], names: dict[str, str], timezone: str = "UTC"
) -> FlowGraph:
    """The graph of ``tasks``, with ``exports`` by id and display ``names`` of datasets.

    A pure function of its inputs, so it is tested without a running instance.
    """
    from open_climate_service.system.templates import workflow_title

    nodes: dict[str, FlowNode] = {}
    edges: list[FlowEdge] = []

    def node(node_id: str, **values: Any) -> str:
        if node_id not in nodes:
            nodes[node_id] = FlowNode(id=node_id, **values)
        else:
            for key, value in values.items():
                if value is not None and getattr(nodes[node_id], key) in (None, ""):
                    setattr(nodes[node_id], key, value)
        return node_id

    def dataset(dataset_id: str) -> str:
        return node(
            f"dataset:{dataset_id}",
            type="dataset",
            label=names.get(dataset_id, dataset_id),
            detail=dataset_id if names.get(dataset_id, dataset_id) != dataset_id else None,
            href=f"/datasets/{dataset_id}",
        )

    def collection(collection_id: str) -> str:
        return node(
            f"collection:{collection_id}",
            type="collection",
            label=collection_id,
            href=f"/data-sources/{collection_id}",
        )

    workflows = {task.id: task for task in tasks if task.kind == "workflow"}
    for task in tasks:
        status = _latest_status(task.id) if task.enabled else "paused"
        starts = _starts(task, timezone)
        if task.kind == "sync":
            source = node(f"source:{task.target}", type="source", label="Data source")
            target = dataset(task.target)
            nodes[target].task_id, nodes[target].status = task.id, status
            nodes[target].starts = f"Synced {starts[0].lower()}{starts[1:]}"
            edges.append(FlowEdge(source=source, target=target, label="sync"))
        elif task.kind == "refresh":
            source = node(f"source:{task.target}", type="source", label="Feature provider")
            target = collection(task.target)
            nodes[target].task_id, nodes[target].status = task.id, status
            nodes[target].starts = f"Refreshed {starts[0].lower()}{starts[1:]}"
            edges.append(FlowEdge(source=source, target=target, label="refresh"))
        elif task.kind == "workflow":
            workflow = node(
                f"task:{task.id}",
                type="workflow",
                label=workflow_title(task.target),
                detail=task.id,
                starts=starts,
                href=f"/workflows/{task.target}",
                task_id=task.id,
                status=status,
            )
            if task.after is not None and task.after.dataset is not None:
                edges.append(FlowEdge(source=dataset(task.after.dataset), target=workflow))
            if task.after is not None and task.after.collection is not None:
                edges.append(FlowEdge(source=collection(task.after.collection), target=workflow))
            read = task.arguments.get("dataset_id")
            reads_after = task.after is not None and task.after.dataset is not None
            if isinstance(read, str) and not read.startswith("$event") and not reads_after:
                # A task on a schedule or by hand names its input in its arguments.
                edges.append(FlowEdge(source=dataset(read), target=workflow, label="reads"))
            for feature_id in _from_features(task.arguments):
                edges.append(FlowEdge(source=collection(feature_id), target=workflow, label=ORG_UNITS))
            output = task.arguments.get("output_dataset_id")
            if isinstance(output, str) and not output.startswith("$event"):
                edges.append(FlowEdge(source=workflow, target=dataset(output), label="publishes"))
        elif task.kind == "deliver" and task.after is not None and task.after.task in workflows:
            definition = exports.get(task.target, {})
            export = node(
                f"export:{task.target}",
                type="export",
                label=task.target,
                detail=f"{'dry run' if task.dry_run else 'live'}, {definition.get('plugin', 'unknown plugin')}",
                starts=starts,
                href=f"/exports/{task.target}",
                task_id=task.id,
                status=status,
            )
            edges.append(FlowEdge(source=f"task:{task.after.task}", target=export, label="deliver"))
            connection = definition.get("connection")
            if isinstance(connection, str):
                destination = node(f"destination:{connection}", type="destination", label=connection, detail="DHIS2")
                edges.append(FlowEdge(source=export, target=destination))
    ordered = sorted(nodes.values(), key=lambda item: (COLUMNS.index(item.type), item.label.lower()))
    return FlowGraph(nodes=ordered, edges=edges)


def current_graph() -> FlowGraph:
    """The graph of the instance's tasks as stored now."""
    from open_climate_service import config as api_config
    from open_climate_service.exports import store as export_store
    from open_climate_service.tasks import store as task_store

    names: dict[str, str] = {}
    try:
        from open_climate_service.ingestions.services import list_datasets

        names = {item.dataset_id: item.dataset_name for item in list_datasets().items}
    except Exception:  # names are a nicety; the graph stands without them
        names = {}
    exports = {str(item.get("id")): item for item in export_store.list_definitions()}
    timezone = str((api_config.get_config().get("scheduler") or {}).get("timezone") or "UTC")
    return build_graph(task_store.list_tasks(), exports, names, timezone)
