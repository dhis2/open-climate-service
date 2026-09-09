"""File-based storage for openEO workflows (user-defined processes)."""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import portalocker

from open_climate_service import config as api_config
from open_climate_service.openeo.schemas import WorkflowListResponse, WorkflowRecord

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


def list_workflows() -> WorkflowListResponse:
    """Return all workflows: built-ins, plugin, and runtime-registered.

    Plugin workflows override built-ins with the same id; runtime-registered
    workflows override both.
    """
    merged: dict[str, dict[str, object]] = {}
    for raw in (
        *_load_builtin_workflows(),
        *_load_entry_point_workflows(),
        *_load_plugin_workflows(),
        *_load_records(),
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


def get_workflow(process_graph_id: str) -> WorkflowRecord | None:
    """Return one workflow by id, or None if not found.

    Checks runtime-registered first, then plugin, then built-in.
    """
    for raw in _load_records():
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


def put_workflow(process_graph_id: str, body: dict[str, object]) -> WorkflowRecord:
    """Store (create or replace) a workflow."""
    record = WorkflowRecord.model_validate({**body, "id": process_graph_id})

    def _mutation(records: list[dict[str, object]]) -> WorkflowRecord:
        payload = record.model_dump(mode="json")
        for index, existing in enumerate(records):
            if existing.get("id") == process_graph_id:
                records[index] = payload
                return record
        records.append(payload)
        return record

    return _mutate_records(_mutation)


def delete_workflow(process_graph_id: str) -> bool:
    """Delete a workflow from the runtime index; returns True if it existed.

    Note: plugin-backed workflows (loaded from plugins_dir/workflows/) cannot be
    deleted via the API — they are re-loaded from disk on each request.  To remove
    a plugin-backed workflow, delete or rename the corresponding JSON file on disk.
    """

    def _mutation(records: list[dict[str, object]]) -> bool:
        for index, existing in enumerate(records):
            if existing.get("id") == process_graph_id:
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
    """Load built-in workflows from open_climate_service/plugins/workflows/."""
    pkg = importlib.resources.files("open_climate_service") / "plugins" / "workflows"
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
    from open_climate_service.plugin_discovery import iter_plugin_subdirs

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
