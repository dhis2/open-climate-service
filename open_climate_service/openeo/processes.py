"""openEO /processes endpoint — wraps the openeo-processes-dask spec registry."""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_PROCESSES: list[dict[str, Any]] = [
    {
        "id": "load_collection",
        "summary": "Load a collection",
        "description": (
            "Loads a collection from the current back-end by its id and returns it as a processable data cube. "
            "A raster collection loads as a raster data cube. A vector collection (one whose `cube:dimensions` "
            "has a `geometry` dimension, such as administrative boundaries) loads as a vector cube of its "
            "features in WGS 84, the same result as `load_features`; it has no temporal dimension, so "
            "`temporal_extent` does not apply, and no bands, so `bands` must be omitted."
        ),
        "parameters": [
            {"name": "id", "description": "Collection ID, raster or vector", "schema": {"type": "string"}},
            {
                "name": "spatial_extent",
                "description": "Bounding box filter",
                "optional": True,
                "default": None,
                "schema": [{"type": "object"}, {"type": "null"}],
            },
            {
                "name": "temporal_extent",
                "description": "Date range filter as [start, end]",
                "optional": True,
                "default": None,
                "schema": [{"type": "array"}, {"type": "null"}],
            },
            {
                "name": "bands",
                "description": "Variable names to load",
                "optional": True,
                "default": None,
                "schema": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            },
        ],
        "returns": {
            "description": "A data cube for further processing: a raster data cube, or a vector cube for a vector "
            "collection.",
            "schema": {"type": "object"},
        },
        "links": [{"rel": "about", "href": "https://processes.openeo.org/#load_collection"}],
    },
    {
        "id": "save_result",
        "summary": "Save processed data to storage",
        "description": "Saves processed data to the local user workspace.",
        "parameters": [
            {"name": "data", "description": "The data to save", "schema": {"type": "object"}},
            {
                "name": "format",
                "description": "Output format (e.g. Zarr, JSON, GeoParquet)",
                "schema": {"type": "string"},
            },
            {
                "name": "options",
                "description": "Format-specific output options",
                "optional": True,
                "default": {},
                "schema": {"type": "object"},
            },
        ],
        "returns": {"description": "false if saving was not successfully finished.", "schema": {"type": "boolean"}},
        "links": [{"rel": "about", "href": "https://processes.openeo.org/#save_result"}],
    },
]


def _load_standard_processes() -> list[dict[str, Any]]:
    """Return process descriptions for all openeo-processes-dask implementations.

    Derives the list from the same execution registry used for process graph
    execution, guaranteeing that listed processes are actually callable.
    """
    # Use the execution registry as the authoritative source of available processes.
    # This avoids duplicating the openeo-processes-dask import logic and ensures
    # the listed processes exactly match what run_process_graph can execute.
    from open_climate_service.openeo.execution import _build_process_registry

    try:
        registry = _build_process_registry()
        predefined: dict[str, Any] = registry[("predefined", None)] or {}
    except Exception:
        logger.warning("Could not load process registry for listing", exc_info=True)
        return []

    # Backend-specific processes are listed separately via _BACKEND_PROCESSES;
    # skip them here to avoid duplicates.
    _backend_ids = {p["id"] for p in _BACKEND_PROCESSES}

    result = []
    for process_id, proc in sorted(predefined.items()):
        if process_id in _backend_ids:
            continue
        # Prefer the stored spec (loaded from openeo-processes-dask JSON files when
        # available) as it contains parameter names, types, and required flags.
        if proc.spec and isinstance(proc.spec, dict) and proc.spec.get("id"):
            result.append(proc.spec)
        else:
            impl = proc.implementation
            doc = inspect.getdoc(impl) or ""
            result.append(
                {
                    "id": process_id,
                    "summary": doc.splitlines()[0] if doc else process_id,
                    "description": doc,
                    "parameters": [],
                    "returns": {"description": "Result.", "schema": {}},
                    "links": [{"rel": "about", "href": f"https://processes.openeo.org/#{process_id}"}],
                }
            )
    return result


def get_openeo_process(process_id: str) -> dict[str, Any] | None:
    """Return one openEO process description by id, or None if not found.

    Avoids rebuilding the full list just to look up a single entry.
    """
    from open_climate_service.openeo.plugin_processes import load_plugin_processes, to_openeo_descriptor

    # Plugin processes take highest precedence
    for pid, func in load_plugin_processes():
        if pid == process_id:
            return to_openeo_descriptor(func)

    # Backend processes
    for p in _BACKEND_PROCESSES:
        if p["id"] == process_id:
            return p

    # Standard openeo-processes-dask spec
    try:
        from open_climate_service.openeo.execution import _build_process_registry

        registry = _build_process_registry()
        predefined: dict[str, Any] = registry[("predefined", None)] or {}
        proc = predefined.get(process_id)
        if proc is not None:
            if proc.spec and isinstance(proc.spec, dict) and proc.spec.get("id"):
                return dict(proc.spec)
            doc = inspect.getdoc(proc.implementation) or ""
            return {
                "id": process_id,
                "summary": doc.splitlines()[0] if doc else process_id,
                "description": doc,
                "parameters": [],
                "returns": {"description": "Result.", "schema": {}},
                "links": [{"rel": "about", "href": f"https://processes.openeo.org/#{process_id}"}],
            }
    except Exception:
        pass

    return None


def list_openeo_processes() -> list[dict[str, Any]]:
    """Return all openEO-compatible process descriptions.

    Merges openeo-processes-dask standard processes, backend-specific processes
    (load_collection, save_result), and @process-decorated plugin functions.
    Plugin processes override standard processes with the same id.
    """
    from open_climate_service.openeo.plugin_processes import load_plugin_processes, to_openeo_descriptor

    standard = _load_standard_processes()

    # Backend processes take precedence over any stub in the standard list
    result: list[dict[str, Any]] = list(_BACKEND_PROCESSES)

    merged: dict[str, dict[str, Any]] = {p["id"]: p for p in result if p.get("id")}

    for p in standard:
        pid = p.get("id")
        if pid and pid not in merged:
            merged[pid] = p

    # Plugin processes override everything with the same id
    for _pid, func in load_plugin_processes():
        descriptor = to_openeo_descriptor(func)
        merged[descriptor["id"]] = descriptor

    return list(merged.values())
