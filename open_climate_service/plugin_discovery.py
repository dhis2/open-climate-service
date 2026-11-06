"""Discovery of installable OCS plugins via entry points (#118).

A plugin package declares an ``open_climate_service.plugins`` entry point whose value
is its top-level import package. The framework then loads the package's ``rasters/``,
``vectors/``, ``processes/``, ``workflows/`` and ``destinations/`` folders the same way it reads an instance's
``plugins_dir`` — so an installed package contributes the same extension points, with
no config wiring beyond installing it.
"""

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Iterator
from importlib.metadata import entry_points
from importlib.resources.abc import Traversable

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "open_climate_service.plugins"

_installed: tuple[tuple[str, str], ...] | None = None
"""``(name, package)`` of every installed plugin, found by the first lookup in this process.

Reading entry points scans the metadata of every installed distribution, and templates,
workflows, processes and destinations are looked up on many requests. A package installed later
is found after a restart. Tests that install one call `rescan`.
"""


def _installed_plugins() -> tuple[tuple[str, str], ...]:
    global _installed
    if _installed is None:
        _installed = tuple(
            (entry_point.name, entry_point.module) for entry_point in entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        )
    return _installed


def rescan() -> None:
    """Forget the installed plugins found so far; the next lookup reads the entry points again."""
    global _installed
    _installed = None


def iter_plugin_subdirs(subdir: str) -> Iterator[tuple[str, str, Traversable]]:
    """Yield ``(plugin_name, package, <package>/<subdir>)`` for installed plugins.

    Only plugins that actually ship the requested ``<subdir>`` (``rasters`` / ``vectors`` /
    ``processes`` / ``workflows`` / ``destinations``) are yielded. ``package`` is the plugin's import
    package, so a caller that needs to *import* modules (processes, vector providers) can build
    the dotted path, while a caller that reads files (dataset templates, workflows) can iterate
    the returned traversable.
    """
    for name, package in _installed_plugins():
        try:
            resource = importlib.resources.files(package) / subdir
        except (ImportError, ModuleNotFoundError, TypeError):
            logger.exception("Could not resolve resources for plugin '%s'", name)
            continue
        if resource.is_dir():
            yield name, package, resource
