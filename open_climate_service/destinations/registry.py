"""Discover built-in, installed, and instance destination plugins."""

from __future__ import annotations

import hashlib
import importlib
import logging
import re
import sys
from pathlib import Path
from types import ModuleType

from open_climate_service import config, plugin_discovery
from open_climate_service.destinations.base import BaseDestinationPlugin

logger = logging.getLogger(__name__)


def load_destination_plugins() -> dict[str, BaseDestinationPlugin]:
    """Load renderers in precedence order, skipping broken plugins."""
    from open_climate_service.destinations.dhis2_renderer import Dhis2DestinationPlugin
    from open_climate_service.destinations.table import TableDestinationPlugin

    found: dict[str, BaseDestinationPlugin] = {
        builtin.id: builtin for builtin in (Dhis2DestinationPlugin(), TableDestinationPlugin())
    }
    for _, package, resource in sorted(plugin_discovery.iter_plugin_subdirs("destinations"), key=lambda item: item[:2]):
        for file in sorted(resource.iterdir(), key=lambda item: item.name):
            if file.name.endswith(".py") and not file.name.startswith("_"):
                _register(found, f"{package}.destinations.{file.name[:-3]}")

    raw = config.get_config().get("plugins_dir")
    if raw:
        if not isinstance(raw, str):
            raise ValueError("plugins_dir must be a path string")
        config_path = config.get_config_path()
        directory = ((config_path.parent if config_path else Path.cwd()) / raw / "destinations").resolve()
        if directory.is_dir():
            # Unique namespace per instance prevents collisions with installed
            # 'destinations' packages and supports relative helper imports.
            namespace = "_ocs_destinations_" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
            if namespace not in sys.modules:
                module = ModuleType(namespace)
                module.__path__ = [str(directory)]
                sys.modules[namespace] = module
            for path in sorted(directory.glob("*.py")):
                if not path.name.startswith("_"):
                    _register(found, f"{namespace}.{path.stem}")
    return found


def _register(found: dict[str, BaseDestinationPlugin], name: str) -> None:
    """Register one plugin module, skipping broken or invalid plugins.

    A malformed third-party destination plugin must not take down discovery endpoints
    such as ``GET /file_formats`` that every openEO client hits on connect. Log
    the problem and skip, like the process plugin loader does.
    """
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        logger.warning("Skipping destination plugin module '%s': %s", name, exc)
        return
    plugin = getattr(module, "plugin", None)
    if not isinstance(plugin, BaseDestinationPlugin):
        logger.warning("Skipping destination module '%s': missing 'plugin' BaseDestinationPlugin instance", name)
        return
    for field, pattern in (
        ("id", r"[A-Za-z0-9][A-Za-z0-9_-]*"),
        ("format", r"[A-Z][A-Z0-9_]*"),
        ("extension", r"\.[a-z0-9]+"),
        ("media_type", r"[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+"),
    ):
        value = getattr(plugin, field, None)
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            logger.warning("Skipping destination plugin '%s': invalid %s", name, field)
            return
    if plugin.extension == ".zarr":
        logger.warning("Skipping destination plugin '%s': renderers produce files, not Zarr directories", name)
        return
    found[plugin.id] = plugin
