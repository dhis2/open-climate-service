"""File-based storage for openEO workflows (user-defined processes)."""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import portalocker

from open_geo_service import config as api_config
from open_geo_service.auth.tokens import Principal, Tier
from open_geo_service.openeo.schemas import WorkflowListResponse, WorkflowRecord

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def _resolve_workflow_dir() -> Path:
    return api_config.get_data_root() / "process_graphs"


WORKFLOW_DIR = _resolve_workflow_dir()
WORKFLOW_INDEX_PATH = WORKFLOW_DIR / "process_graphs.json"


def _ensure_store() -> None:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    if not WORKFLOW_INDEX_PATH.exists():
        WORKFLOW_INDEX_PATH.write_text("[]\n", encoding="utf-8")


class WorkflowAccessError(Exception):
    """A principal tried to store a workflow under an id it may not take."""


def _stored_owner(raw: dict[str, object]) -> str | None:
    owner = raw.get("owner")
    return owner if isinstance(owner, str) else None


def _visible_records(visible_to: Principal | None) -> list[dict[str, object]]:
    """Stored workflows `visible_to` may see; all of them when None, as execution needs."""
    return [raw for raw in _load_records() if visible_to is None or visible_to.may_access(_stored_owner(raw))]


def list_workflows(visible_to: Principal | None = None) -> WorkflowListResponse:
    """Return all workflows: built-ins, plugin, and runtime-registered.

    Plugin workflows override built-ins with the same id; runtime-registered
    workflows override both. Given `visible_to`, a stored workflow is listed only to its owner
    and to operators; built-in and plugin workflows are listed to everyone.
    """
    merged: dict[str, dict[str, object]] = {}
    for raw in (
        *_load_builtin_workflows(),
        *_load_entry_point_workflows(),
        *_load_plugin_workflows(),
        *_visible_records(visible_to),
    ):
        wid = raw.get("id")
        if not isinstance(wid, str) or not wid:
            logger.warning("Skipping workflow with missing or non-string id")
            continue
        merged[wid] = raw
    records = []
    for raw in merged.values():
        try:
            records.append(WorkflowRecord.model_validate(raw))
        except Exception:
            logger.warning("Skipping invalid workflow record: %s", raw.get("id", "<unknown>"))
    return WorkflowListResponse(
        processes=records,
        links=[{"rel": "self", "href": "/process_graphs", "type": "application/json"}],
    )


def get_workflow(process_graph_id: str, visible_to: Principal | None = None) -> WorkflowRecord | None:
    """Return one workflow by id, or None if not found.

    Checks runtime-registered first, then plugin, then built-in. Given `visible_to`, a stored
    workflow it may not see is passed over, as `list_workflows` passes it over.
    """
    for raw in _visible_records(visible_to):
        if raw.get("id") == process_graph_id:
            return WorkflowRecord.model_validate(raw)
    for raw in _load_plugin_workflows():
        if raw.get("id") == process_graph_id:
            return WorkflowRecord.model_validate(raw)
    for raw in _load_entry_point_workflows():
        if raw.get("id") == process_graph_id:
            return WorkflowRecord.model_validate(raw)
    for raw in _load_builtin_workflows():
        if raw.get("id") == process_graph_id:
            return WorkflowRecord.model_validate(raw)
    return None


def put_workflow(process_graph_id: str, body: dict[str, object], by: Principal) -> WorkflowRecord:
    """Store (create or replace) a workflow owned by `by`; a replaced workflow keeps its owner.

    Raises WorkflowAccessError when `by` would replace another principal's workflow, or, unless
    it is an operator, take the id of a built-in or plugin workflow, which every caller and
    pipeline would then run instead.
    """
    record = WorkflowRecord.model_validate({**body, "id": process_graph_id})
    if by.tier is not Tier.OPERATOR and any(
        raw.get("id") == process_graph_id
        for raw in (*_load_builtin_workflows(), *_load_entry_point_workflows(), *_load_plugin_workflows())
    ):
        raise WorkflowAccessError(f"Process graph id '{process_graph_id}' is a built-in or plugin workflow")

    def _mutation(records: list[dict[str, object]]) -> WorkflowRecord:
        payload = record.model_dump(mode="json")
        for index, existing in enumerate(records):
            if existing.get("id") == process_graph_id:
                if not by.may_access(_stored_owner(existing)):
                    raise WorkflowAccessError(f"Process graph id '{process_graph_id}' is taken")
                records[index] = {**payload, "owner": _stored_owner(existing)}
                return record
        records.append({**payload, "owner": by.owner_id})
        return record

    return _mutate_records(_mutation)


def delete_workflow(process_graph_id: str, by: Principal) -> bool:
    """Delete a workflow `by` may see from the runtime index; returns True if it existed.

    Note: plugin-backed workflows (loaded from plugins_dir/workflows/) cannot be
    deleted via the API — they are re-loaded from disk on each request.  To remove
    a plugin-backed workflow, delete or rename the corresponding JSON file on disk.
    """

    def _mutation(records: list[dict[str, object]]) -> bool:
        for index, existing in enumerate(records):
            if existing.get("id") == process_graph_id and by.may_access(_stored_owner(existing)):
                records.pop(index)
                return True
        return False

    return _mutate_records(_mutation)


def _load_records() -> list[dict[str, object]]:
    _ensure_store()
    return _read_records_from_disk()


def _read_records_from_disk() -> list[dict[str, object]]:
    with open(WORKFLOW_INDEX_PATH, encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_SH)
        try:
            payload = json.load(handle)
        finally:
            portalocker.unlock(handle)
    if not isinstance(payload, list):
        raise ValueError("process_graphs.json must contain a list")
    return payload


def _load_builtin_workflows() -> list[dict[str, object]]:
    """Load built-in workflows from open_geo_service/plugins/workflows/."""
    pkg = importlib.resources.files("open_geo_service") / "plugins" / "workflows"
    workflows: list[dict[str, object]] = []
    try:
        for resource in pkg.iterdir():
            if not resource.name.endswith(".json"):
                continue
            try:
                data = json.loads(resource.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    logger.warning("Skipping built-in workflow %s: expected a JSON object", resource.name)
                    continue
                workflows.append(data)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping built-in workflow %s: %s", resource.name, exc)
    except (FileNotFoundError, NotADirectoryError):
        pass
    return workflows


def _load_plugin_workflows() -> list[dict[str, object]]:
    """Load workflows from plugins_dir/workflows/ if configured."""
    config = api_config.get_config()
    plugins_dir = config.get("plugins_dir") if config else None
    if not plugins_dir:
        return []
    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    workflows_dir = (base / plugins_dir).resolve() / "workflows"
    if not workflows_dir.is_dir():
        return []
    workflows: list[dict[str, object]] = []
    for path in sorted(workflows_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Skipping workflow file %s: expected a JSON object", path)
                continue
            workflows.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping workflow file %s: %s", path, exc)
    return workflows


def _load_entry_point_workflows() -> list[dict[str, object]]:
    """Load workflows/*.json shipped by installed plugin packages (#118)."""
    from open_geo_service.plugin_discovery import iter_plugin_subdirs

    workflows: list[dict[str, object]] = []
    for _name, _package, workflows_res in iter_plugin_subdirs("workflows"):
        for resource in workflows_res.iterdir():
            if not resource.name.endswith(".json"):
                continue
            try:
                data = json.loads(resource.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping plugin workflow file %s: %s", resource.name, exc)
                continue
            if not isinstance(data, dict):
                logger.warning("Skipping plugin workflow file %s: expected a JSON object", resource.name)
                continue
            workflows.append(data)
    return workflows


def _mutate_records(mutation: Callable[[list[dict[str, object]]], _T]) -> _T:
    _ensure_store()
    with open(WORKFLOW_INDEX_PATH, "r+", encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_EX)
        try:
            payload = json.load(handle)
            records: list[dict[str, object]] = payload if isinstance(payload, list) else []
            result = mutation(records)
            handle.seek(0)
            json.dump(records, handle, indent=2)
            handle.write("\n")
            handle.truncate()
            return result
        finally:
            portalocker.unlock(handle)


# --- freezing a graph against the registry ---------------------------------------------------

_MAX_INLINE_DEPTH = 16


def _rewrite_node_refs(value: Any, rename: Callable[[str], str]) -> Any:
    """Rewrite ``from_node`` references of one graph level, leaving callback graphs alone.

    A callback (a mapping holding ``process_graph``) has its own node namespace, so its
    references are not this graph's nodes.
    """
    if isinstance(value, dict):
        if set(value) == {"from_node"} and isinstance(value["from_node"], str):
            return {"from_node": rename(value["from_node"])}
        if "process_graph" in value:
            return value
        return {key: _rewrite_node_refs(child, rename) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_node_refs(child, rename) for child in value]
    return value


def _freeze_callbacks(value: Any, resolve: Callable[[str], WorkflowRecord | None], stack: tuple[str, ...]) -> Any:
    """Freeze the callback graphs nested in a node's arguments."""
    if isinstance(value, dict):
        frozen = {
            key: _freeze_callbacks(child, resolve, stack) for key, child in value.items() if key != "process_graph"
        }
        graph = value.get("process_graph")
        if isinstance(graph, dict):
            frozen["process_graph"] = freeze_process_graph(graph, resolve=resolve, _stack=stack)
        elif "process_graph" in value:
            frozen["process_graph"] = graph
        return frozen
    if isinstance(value, list):
        return [_freeze_callbacks(child, resolve, stack) for child in value]
    return value


def freeze_process_graph(
    process_graph: dict[str, Any],
    *,
    resolve: Callable[[str], WorkflowRecord | None] | None = None,
    _stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Inline every call to a registered workflow, recursively, as of now.

    The result runs the definitions current at this moment even if a workflow is edited
    later, which a graph that names workflows by id does not: those are resolved again at
    execution. Arguments and declared defaults are substituted exactly as the executor does
    (`execution._resolve_workflow_parameters`); each inlined node is named
    ``<calling node>__<workflow node>``, and references to the calling node are redirected to
    the workflow's result node.
    """
    from open_geo_service.openeo.execution import _resolve_workflow_parameters

    lookup = resolve or get_workflow
    if len(_stack) > _MAX_INLINE_DEPTH:
        raise ValueError(f"workflow calls nest deeper than {_MAX_INLINE_DEPTH}: {' > '.join(_stack)}")
    nodes: dict[str, Any] = {}
    redirect: dict[str, str] = {}

    def _insert(node_id: str, node: dict[str, Any]) -> None:
        if node_id in nodes:
            raise ValueError(f"workflow inlining would overwrite node '{node_id}'")
        nodes[node_id] = node

    for node_id, raw in process_graph.items():
        node = {**raw, "arguments": _freeze_callbacks(raw.get("arguments", {}), lookup, _stack)}
        process_id = node.get("process_id")
        record = lookup(process_id) if isinstance(process_id, str) else None
        if record is None or not record.process_graph:
            _insert(node_id, node)
            continue
        if record.id in _stack:
            raise ValueError(f"workflow '{record.id}' calls itself: {' > '.join((*_stack, record.id))}")
        prefix = f"{node_id}__"

        def _prefixed(name: str, prefix: str = prefix) -> str:
            return f"{prefix}{name}"

        body = {
            _prefixed(inner_id): _rewrite_node_refs(inner, _prefixed)
            for inner_id, inner in record.process_graph.items()
        }
        defaults = {p["name"]: p["default"] for p in record.parameters if "name" in p and "default" in p}
        body = _resolve_workflow_parameters(body, {**defaults, **node["arguments"]})
        body = freeze_process_graph(body, resolve=lookup, _stack=(*_stack, record.id))
        results = [inner_id for inner_id, inner in body.items() if inner.get("result")]
        if len(results) != 1:
            raise ValueError(f"workflow '{record.id}' must have exactly one result node, found {len(results)}")
        for inner_id, inner in body.items():
            inner = {key: value for key, value in inner.items() if key != "result"}
            if inner_id == results[0] and node.get("result"):
                inner["result"] = True
            _insert(inner_id, inner)
        redirect[node_id] = results[0]
    return {node_id: _rewrite_node_refs(node, lambda name: redirect.get(name, name)) for node_id, node in nodes.items()}
