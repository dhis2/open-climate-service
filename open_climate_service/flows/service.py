"""The flow graph: what each configured task reads and writes, with each step's latest run (CLIM-1377).

The graph is configuration, not history: it shows what is set up to run, and puts the latest run
of each task on its node. Nodes are datasets, feature collections, workflow tasks, exports and
destinations; edges say what feeds what. `FlowGraph.layout` places every node once, left to right
by dependency, so the Tasks page's Flow view, a dataset's page and a task's page draw the same
diagram; ``GET /flows`` returns the graph as JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from open_climate_service.scheduler.presets import schedule_description
from open_climate_service.tasks.models import Task

NodeType = Literal["dataset", "collection", "workflow", "export", "result", "destination"]

ORG_UNITS = "org units"
"""The label of an edge that feeds a workflow its org units."""

PUBLISHES = "publishes"
"""The label of an edge from a workflow to the dataset it writes."""

BOX_WIDTH = 200
BOX_HEIGHT = 120
COLUMN_GAP = 64
ROW_GAP = 16


class WorkflowInfo(BaseModel):
    """What the graph needs to know about a workflow: its title, and what it produces."""

    title: str
    results: list[tuple[str, str]] = Field(
        default_factory=list, description="(kind, label) pairs, as the workflow page shows them."
    )


class FlowNode(BaseModel):
    """One thing data passes through, with how it is kept current and how its last run went."""

    id: str
    type: NodeType
    label: str
    detail: str | None = None
    starts: str | None = None
    href: str | None = None
    task_id: str | None = None
    status: str | None = None
    last_run_at: datetime | None = None


class FlowEdge(BaseModel):
    """Data moving from one node to the next."""

    source: str
    target: str
    label: str | None = None


class PlacedNode(FlowNode):
    """A node with its top-left corner in the diagram, in pixels."""

    x: int
    y: int


class PlacedEdge(FlowEdge):
    """An edge as an SVG path from the right edge of its source to the left edge of its target."""

    path: str
    label_x: int
    label_y: int


class FlowLayout(BaseModel):
    """The graph placed for drawing: every node once, columns by dependency, no two boxes overlapping."""

    nodes: list[PlacedNode]
    edges: list[PlacedEdge]
    width: int
    height: int


class FlowGraph(BaseModel):
    """Every configured path, as nodes and edges."""

    nodes: list[FlowNode]
    edges: list[FlowEdge]

    def around(self, node_id: str) -> FlowGraph:
        """Everything upstream and downstream of ``node_id``, with the org units its workflows use."""
        if node_id not in {node.id for node in self.nodes}:
            return FlowGraph(nodes=[], edges=[])
        keep = {node_id}
        for forward in (True, False):
            frontier = [node_id]
            while frontier:
                current = frontier.pop()
                for edge in self.edges:
                    near, far = (edge.source, edge.target) if forward else (edge.target, edge.source)
                    if near == current and far not in keep:
                        keep.add(far)
                        frontier.append(far)
        keep |= {edge.source for edge in self.edges if edge.label == ORG_UNITS and edge.target in keep}
        return FlowGraph(
            nodes=[node for node in self.nodes if node.id in keep],
            edges=[edge for edge in self.edges if edge.source in keep and edge.target in keep],
        )

    def layout(self) -> FlowLayout:
        """Place the nodes: a column per step of dependency, rows so that a chain reads straight across.

        Columns come from the longest path to each node; a node nothing feeds is pulled next to
        what it feeds, so org units sit beside the workflow that uses them rather than far left.
        Rows follow the barycentre of a node's neighbours, swept forward, back and forward again,
        and a node takes the first free row at or below where its neighbours put it, so a fan-out
        stacks beneath its first branch and nothing overlaps.
        """
        nodes = {node.id: node for node in self.nodes}
        preds: dict[str, list[str]] = {node.id: [] for node in self.nodes}
        succs: dict[str, list[str]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            if edge.source in nodes and edge.target in nodes:
                succs[edge.source].append(edge.target)
                preds[edge.target].append(edge.source)

        column: dict[str, int] = {}
        for node_id in self._topological(preds, succs):
            column[node_id] = max((column[pred] + 1 for pred in preds[node_id]), default=0)
        for node_id in nodes:
            if not preds[node_id] and succs[node_id]:
                column[node_id] = min(column[succ] for succ in succs[node_id]) - 1

        columns: dict[int, list[str]] = {}
        for node_id in nodes:
            columns.setdefault(column[node_id], []).append(node_id)
        ordered = [columns[index] for index in sorted(columns)]
        row: dict[str, int] = {node_id: index for members in ordered for index, node_id in enumerate(members)}
        for sweep in ("forward", "backward", "forward"):
            sequence = ordered if sweep == "forward" else list(reversed(ordered))
            neighbours = preds if sweep == "forward" else succs
            for members in sequence:
                self._assign_rows(members, neighbours, row, nodes)

        placed = [
            PlacedNode(
                **nodes[node_id].model_dump(),
                x=column[node_id] * (BOX_WIDTH + COLUMN_GAP),
                y=row[node_id] * (BOX_HEIGHT + ROW_GAP),
            )
            for node_id in nodes
        ]
        by_id = {node.id: node for node in placed}
        edges = []
        for edge in self.edges:
            if edge.source not in by_id or edge.target not in by_id:
                continue
            source, target = by_id[edge.source], by_id[edge.target]
            x1, y1 = source.x + BOX_WIDTH, source.y + BOX_HEIGHT // 2
            x2, y2 = target.x, target.y + BOX_HEIGHT // 2
            bend = max((x2 - x1) // 2, COLUMN_GAP // 2)
            path = f"M{x1},{y1} C{x1 + bend},{y1} {x2 - bend},{y2} {x2},{y2}"
            # The label sits mid-way on a short edge; a long edge passes behind other boxes, so
            # its label goes by the arrowhead, where it says what is entering the box.
            along = 0.5 if x2 - x1 <= COLUMN_GAP else 1 - COLUMN_GAP / 2 / (x2 - x1)
            label_x, label_y = _bezier_point(along, (x1, y1), (x1 + bend, y1), (x2 - bend, y2), (x2, y2))
            edges.append(PlacedEdge(**edge.model_dump(), path=path, label_x=label_x, label_y=label_y))
        width = max((node.x + BOX_WIDTH for node in placed), default=0)
        height = max((node.y + BOX_HEIGHT for node in placed), default=0)
        return FlowLayout(nodes=placed, edges=edges, width=width, height=height)

    @staticmethod
    def _topological(preds: dict[str, list[str]], succs: dict[str, list[str]]) -> list[str]:
        """Kahn's order.

        The graph is acyclic by construction (a task that would write what it waits for is
        refused); a node a cycle somehow left unreached is appended rather than lost.
        """
        pending = {node_id: len(members) for node_id, members in preds.items()}
        ready = [node_id for node_id, count in pending.items() if count == 0]
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for succ in succs[node_id]:
                pending[succ] -= 1
                if pending[succ] == 0:
                    ready.append(succ)
        return order + [node_id for node_id in preds if node_id not in order]

    @staticmethod
    def _assign_rows(
        members: list[str], neighbours: dict[str, list[str]], row: dict[str, int], nodes: dict[str, FlowNode]
    ) -> None:
        def wanted(node_id: str) -> float | None:
            rows = [row[other] for other in neighbours[node_id] if other in row]
            return sum(rows) / len(rows) if rows else None

        placed = sorted(
            members,
            key=lambda node_id: (
                wanted(node_id) is None,
                wanted(node_id) or 0.0,
                row[node_id],
                nodes[node_id].label.lower(),
            ),
        )
        next_free = 0
        for node_id in placed:
            target = wanted(node_id)
            desired = next_free if target is None else int(target + 0.49)
            row[node_id] = max(desired, next_free)
            next_free = row[node_id] + 1


def _bezier_point(
    t: float, p0: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]
) -> tuple[int, int]:
    """The point a fraction ``t`` along a cubic Bezier curve."""
    u = 1 - t
    weights = (u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t)
    x = sum(weight * point[0] for weight, point in zip(weights, (p0, p1, p2, p3)))
    y = sum(weight * point[1] for weight, point in zip(weights, (p0, p1, p2, p3)))
    return round(x), round(y)


def _starts(task: Task, timezone: str) -> str:
    """How a task is started, in a few words for its box."""
    if task.cron is not None:
        return schedule_description(task.cron, timezone) or f"On cron {task.cron}"
    if task.after is not None:
        return "After each run" if task.after.task is not None else "After each update"
    return "By hand"


def _latest_run(task_id: str) -> tuple[str | None, datetime | None]:
    from open_climate_service.runs import service as runs

    latest = runs.list_runs(task_id=task_id, limit=1)
    if not latest:
        return None, None
    return runs.view(latest[0]).status, latest[0].started_at


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
    tasks: list[Task],
    exports: dict[str, dict[str, Any]],
    names: dict[str, str],
    workflows: dict[str, WorkflowInfo] | None = None,
    timezone: str = "UTC",
) -> FlowGraph:
    """The graph of ``tasks``, with ``exports`` by id, display ``names`` of datasets and ``workflows`` by id.

    A pure function of its inputs, so it is tested without a running instance.
    """
    workflows = workflows or {}
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
        name = names.get(dataset_id, dataset_id)
        return node(
            f"dataset:{dataset_id}",
            type="dataset",
            label=name,
            detail=dataset_id if name != dataset_id else None,
            href=f"/datasets/{dataset_id}",
        )

    def collection(collection_id: str) -> str:
        return node(
            f"collection:{collection_id}", type="collection", label=collection_id, href=f"/data-sources/{collection_id}"
        )

    def ran(node_id: str, task: Task, starts: str) -> None:
        status, at = _latest_run(task.id) if task.enabled else ("paused", None)
        nodes[node_id].task_id, nodes[node_id].status, nodes[node_id].last_run_at = task.id, status, at
        nodes[node_id].starts = starts

    workflow_tasks = {task.id: task for task in tasks if task.kind == "workflow"}
    for task in tasks:
        starts = _starts(task, timezone)
        if task.kind == "sync":
            ran(dataset(task.target), task, f"Synced {starts[0].lower()}{starts[1:]}")
        elif task.kind == "refresh":
            ran(collection(task.target), task, f"Refreshed {starts[0].lower()}{starts[1:]}")
        elif task.kind == "workflow":
            info = workflows.get(task.target)
            workflow = node(
                f"task:{task.id}",
                type="workflow",
                label=info.title if info is not None else task.target,
                detail=task.id,
                href=f"/tasks/{task.id}",
            )
            ran(workflow, task, starts)
            if task.after is not None and task.after.dataset is not None:
                edges.append(FlowEdge(source=dataset(task.after.dataset), target=workflow))
            if task.after is not None and task.after.collection is not None:
                edges.append(FlowEdge(source=collection(task.after.collection), target=workflow))
            read = task.arguments.get("dataset_id")
            reads_after = task.after is not None and task.after.dataset is not None
            if isinstance(read, str) and not read.startswith("$event") and not reads_after:
                # A task on a schedule or by hand names its input in its arguments.
                edges.append(FlowEdge(source=dataset(read), target=workflow))
            for feature_id in _from_features(task.arguments):
                edges.append(FlowEdge(source=collection(feature_id), target=workflow, label=ORG_UNITS))
            output = task.arguments.get("output_dataset_id")
            export_id = task.arguments.get("export")
            if isinstance(output, str) and not output.startswith("$event"):
                edges.append(FlowEdge(source=workflow, target=dataset(output), label=PUBLISHES))
            elif not isinstance(export_id, str) and info is not None:
                # A workflow that writes a file each run, with no named export to carry it on.
                for kind, label in info.results:
                    if kind == "export":
                        result = node(
                            f"result:{task.id}",
                            type="result",
                            label=label.removeprefix("Exports "),
                            detail="A file from each run",
                        )
                        edges.append(FlowEdge(source=workflow, target=result))
        elif task.kind == "deliver" and task.after is not None and task.after.task in workflow_tasks:
            definition = exports.get(task.target, {})
            export = node(
                f"export:{task.target}",
                type="export",
                label=task.target,
                detail=f"{'Dry run' if task.dry_run else 'Live'} · {definition.get('plugin', 'unknown plugin')}",
                href=f"/tasks/{task.id}",
            )
            ran(export, task, starts)
            edges.append(FlowEdge(source=f"task:{task.after.task}", target=export))
            connection = definition.get("connection")
            if isinstance(connection, str):
                destination = node(f"destination:{connection}", type="destination", label=connection, detail="DHIS2")
                edges.append(FlowEdge(source=export, target=destination))
    return FlowGraph(nodes=list(nodes.values()), edges=edges)


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
    workflows: dict[str, WorkflowInfo] = {}
    try:
        from open_climate_service.openeo.workflows import list_workflows
        from open_climate_service.system.templates import workflow_results, workflow_title

        workflows = {
            record.id: WorkflowInfo(title=workflow_title(record.id), results=workflow_results(record))
            for record in list_workflows().processes
        }
    except Exception:
        workflows = {}
    exports = {str(item.get("id")): item for item in export_store.list_definitions()}
    timezone = str((api_config.get_config().get("scheduler") or {}).get("timezone") or "UTC")
    return build_graph(task_store.list_tasks(), exports, names, workflows, timezone)
