"""Discover built-in, installed, and instance export plugins."""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType

from open_climate_service import config, plugin_discovery
from open_climate_service.exports.base import BaseExportPlugin


def load_export_plugins() -> dict[str, BaseExportPlugin]:
    """Load renderers in precedence order, failing explicitly on broken plugins."""
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

    builtin = Dhis2ExportPlugin()
    found: dict[str, BaseExportPlugin] = {builtin.id: builtin}
    for _, package, resource in sorted(plugin_discovery.iter_plugin_subdirs("exports"), key=lambda item: item[:2]):
        for file in sorted(resource.iterdir(), key=lambda item: item.name):
            if file.name.endswith(".py") and not file.name.startswith("_"):
                _register(found, f"{package}.exports.{file.name[:-3]}")

    raw = config.get_config().get("plugins_dir")
    if raw:
        if not isinstance(raw, str):
            raise ValueError("plugins_dir must be a path string")
        config_path = config.get_config_path()
        directory = ((config_path.parent if config_path else Path.cwd()) / raw / "exports").resolve()
        if directory.is_dir():
            # Unique namespace per instance prevents collisions with installed
            # 'exports' packages and supports relative helper imports.
            namespace = "_ocs_exports_" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
            if namespace not in sys.modules:
                module = ModuleType(namespace)
                module.__path__ = [str(directory)]
                sys.modules[namespace] = module
            for path in sorted(directory.glob("*.py")):
                if not path.name.startswith("_"):
                    _register(found, f"{namespace}.{path.stem}")
    return found


def _register(found: dict[str, BaseExportPlugin], name: str) -> None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise ValueError(f"Could not load export plugin module '{name}'") from exc
    plugin = getattr(module, "plugin", None)
    if not isinstance(plugin, BaseExportPlugin):
        raise ValueError(f"Export module '{name}' must expose a BaseExportPlugin instance as 'plugin'")
    for field, pattern in (
        ("id", r"[A-Za-z0-9][A-Za-z0-9_-]*"),
        ("format", r"[A-Z][A-Z0-9_]*"),
        ("extension", r"\.[a-z0-9]+"),
        ("media_type", r"[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+"),
    ):
        value = getattr(plugin, field, None)
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"Export plugin '{name}' has an invalid {field}")
    if plugin.extension == ".zarr":
        raise ValueError("Export renderers produce files, not Zarr directories")
    found[plugin.id] = plugin
