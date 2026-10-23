"""Discovery and loading of @process-decorated plugin functions."""

from __future__ import annotations

import importlib
import importlib.resources
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from open_climate_service import config as api_config
from open_climate_service.openeo import earthkit_processes, xclim_processes
from open_climate_service.process import get_process_metadata
from open_climate_service.shared.import_lock import IMPORT_LOCK

logger = logging.getLogger(__name__)


def load_plugin_processes() -> list[tuple[str, Any]]:
    """Return (process_id, callable) for all @process-decorated functions.

    Resolution order (last wins):
    1. xclim indicators — auto-registered from all indicator modules
    2. earthkit-meteo functions — auto-registered from the curated module list
    3. Built-in file plugins — ``open_climate_service/plugins/processes/``
    4. Installed plugin packages — ``<package>/processes/`` (entry-point plugins, #118)
    5. Instance plugins — ``plugins_dir/processes/`` (override everything)
    """
    # Plugin modules import xclim at module level; see shared/import_lock (CLIM-1549).
    with IMPORT_LOCK:
        found: dict[str, Any] = {}
        for func in xclim_processes.scan():
            meta = get_process_metadata(func)
            if meta:
                found[meta["id"]] = func
        for func in earthkit_processes.scan():
            meta = get_process_metadata(func)
            if meta:
                found[meta["id"]] = func
        for func in _scan_builtin_processes():
            meta = get_process_metadata(func)
            if meta:
                found[meta["id"]] = func
        for func in _scan_plugin_package_processes():
            meta = get_process_metadata(func)
            if meta:
                found[meta["id"]] = func
        for func in _scan_instance_processes():
            meta = get_process_metadata(func)
            if meta:
                found[meta["id"]] = func
        return list(found.items())


def to_openeo_descriptor(func: Any) -> dict[str, Any]:
    """Convert a @process-decorated function to an openEO process descriptor."""
    meta = get_process_metadata(func)
    if meta is None:
        raise ValueError(f"{func!r} is not decorated with @process")
    return dict(meta)


def _scan_builtin_processes() -> list[Any]:
    pkg = importlib.resources.files("open_climate_service") / "plugins" / "processes"
    funcs: list[Any] = []
    try:
        for resource in pkg.iterdir():
            if not resource.name.endswith(".py") or resource.name.startswith("_"):
                continue
            module_name = f"open_climate_service.plugins.processes.{resource.name[:-3]}"
            funcs.extend(_load_from_module(module_name))
    except (FileNotFoundError, NotADirectoryError):
        pass
    return funcs


def _scan_plugin_package_processes() -> list[Any]:
    """Scan ``processes/`` in each installed plugin package (#118).

    The package is on ``PYTHONPATH`` (it is installed), so its process modules are
    imported by their full dotted path — no ``sys.path`` handling like the
    ``plugins_dir`` scan needs.
    """
    from open_climate_service.plugin_discovery import iter_plugin_subdirs

    funcs: list[Any] = []
    for _name, package, processes_res in iter_plugin_subdirs("processes"):
        for resource in processes_res.iterdir():
            if not resource.name.endswith(".py") or resource.name.startswith("_"):
                continue
            funcs.extend(_load_from_module(f"{package}.processes.{resource.name[:-3]}"))
    return funcs


def _scan_instance_processes() -> list[Any]:
    config = api_config.get_config()
    plugins_dir_raw = config.get("plugins_dir") if config else None
    if not plugins_dir_raw:
        return []
    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    processes_dir = (base / plugins_dir_raw).resolve() / "processes"
    if not processes_dir.is_dir():
        return []
    parent = str(processes_dir.parent)
    if parent not in sys.path:
        sys.path.append(parent)
    funcs: list[Any] = []
    for path in sorted(processes_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        funcs.extend(_load_from_module(f"processes.{path.stem}"))
    return funcs


def _load_from_module(module_name: str) -> list[Any]:
    try:
        module: ModuleType = importlib.import_module(module_name)
        return [obj for obj in vars(module).values() if callable(obj) and get_process_metadata(obj) is not None]
    except Exception:
        logger.warning("Failed to load plugin processes from %s", module_name, exc_info=True)
        return []
