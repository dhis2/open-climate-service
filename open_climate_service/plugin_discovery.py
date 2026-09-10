"""Discovery of installable OCS plugins via entry points (#118).

A plugin package declares an ``open_climate_service.plugins`` entry point whose value
is its top-level import package. The framework then loads the package's ``datasets/``,
``processes/``, ``workflows/`` and ``exports/`` folders the same way it reads an instance's
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


def iter_plugin_subdirs(subdir: str) -> Iterator[tuple[str, str, Traversable]]:
    """Yield ``(plugin_name, package, <package>/<subdir>)`` for installed plugins.

    Only plugins that actually ship the requested ``<subdir>`` (``datasets`` /
    ``processes`` / ``workflows`` / ``exports``) are yielded. ``package`` is the plugin's import
    package, so a caller that needs to *import* modules (processes) can build the
    dotted path, while a caller that reads files (datasets, workflows) can iterate
    the returned traversable.
    """
    for entry_point in entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        package = entry_point.module
        try:
            resource = importlib.resources.files(package) / subdir
        except (ImportError, ModuleNotFoundError, TypeError):
            logger.exception("Could not resolve resources for plugin '%s'", entry_point.name)
            continue
        if resource.is_dir():
            yield entry_point.name, package, resource
