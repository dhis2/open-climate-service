"""The flow graph: what each configured task reads and writes, with each step's latest status (CLIM-1377).

The graph is configuration, not history: it shows what is set up to run, and puts the latest run
of each task on its node. Nodes are data sources, datasets, feature collections, workflow tasks,
exports and destinations; edges say what feeds what. The Flows page draws it, a dataset page
shows the part around one dataset, and ``GET /flows`` returns it as JSON.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

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
    task_id: str | None = None
    status: str | None = None


class FlowEdge(BaseModel):
    """Data moving from one node to the next."""

    source: str
    target: str
    label: str | None = None


class FlowGraph(BaseModel):
    """Every configured path, as nodes and edges."""

    nodes: list[FlowNode]
    edges: list[FlowEdge]

    def around(self, node_id: str) -> FlowGraph:
        """The nodes one edge either side of ``node_id``, for a dataset page's Flow panel."""
        edges = [edge for edge in self.edges if node_id in (edge.source, edge.target)]
        keep = {node_id} | {edge.source for edge in edges} | {edge.target for edge in edges}
        return FlowGraph(nodes=[node for node in self.nodes if node.id in keep], edges=edges)


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


def build_graph(tasks: list[Task], exports: dict[str, dict[str, Any]], names: dict[str, str]) -> FlowGraph:
    """The graph of ``tasks``, with ``exports`` by id and display ``names`` of datasets.

    A pure function of its inputs, so it is tested without a running instance.
    """
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
            detail=dataset_id,
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
        if task.kind == "sync":
            source = node(f"source:{task.target}", type="source", label="Data source", detail=task.target)
            target = dataset(task.target)
            nodes[target].task_id, nodes[target].status = task.id, status
            nodes[target].detail = f"synced {task.when}"
            edges.append(FlowEdge(source=source, target=target, label="sync"))
        elif task.kind == "refresh":
            source = node(f"source:{task.target}", type="source", label="Feature provider", detail=task.target)
            target = collection(task.target)
            nodes[target].task_id, nodes[target].status = task.id, status
            nodes[target].detail = f"refreshed {task.when}"
            edges.append(FlowEdge(source=source, target=target, label="refresh"))
        elif task.kind == "workflow":
            workflow = node(
                f"task:{task.id}",
                type="workflow",
                label=task.target,
                detail=task.when,
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
                edges.append(FlowEdge(source=collection(feature_id), target=workflow, label="org units"))
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
                href=f"/exports/{task.target}",
                task_id=task.id,
                status=status,
            )
            edges.append(FlowEdge(source=f"task:{task.after.task}", target=export, label="deliver"))
            connection = definition.get("connection")
            if isinstance(connection, str):
                destination = node(
                    f"destination:{connection}", type="destination", label=connection, detail="DHIS2 connection"
                )
                edges.append(FlowEdge(source=export, target=destination))
    ordered = sorted(nodes.values(), key=lambda item: (COLUMNS.index(item.type), item.label.lower()))
    return FlowGraph(nodes=ordered, edges=edges)


def current_graph() -> FlowGraph:
    """The graph of the instance's tasks as stored now."""
    from open_climate_service.exports import store as export_store
    from open_climate_service.tasks import store as task_store

    names: dict[str, str] = {}
    try:
        from open_climate_service.ingestions.services import list_datasets

        names = {item.dataset_id: item.dataset_name for item in list_datasets().items}
    except Exception:  # names are a nicety; the graph stands without them
        names = {}
    exports = {str(item.get("id")): item for item in export_store.list_definitions()}
    return build_graph(task_store.list_tasks(), exports, names)
