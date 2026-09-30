"""Startup diagnostics for instance plugins (datasets, processes, workflows).

A misconfigured or unset ``plugins_dir`` silently drops instance plugin workflows and
processes — the loaders just return an empty list — which later surfaces as a cryptic
"process not found in namespace predefined". Logging a one-line summary at startup (and a
warning for a missing ``plugins_dir`` or sub-directory) makes that immediately obvious.
"""

from __future__ import annotations

import logging
from pathlib import Path

from open_climate_service import config as api_config

logger = logging.getLogger("open_climate_service")

# (subdirectory, glob) for each plugin kind under plugins_dir.
_PLUGIN_SUBDIRS: tuple[tuple[str, str], ...] = (
    (api_config.RASTERS_SUBDIR, "*.y*ml"),
    (api_config.VECTORS_SUBDIR, "*.y*ml"),
    ("processes", "*.py"),
    ("workflows", "*.json"),
)


def log_plugin_loading() -> None:
    """Log a summary of instance plugins discovered under ``plugins_dir``.

    Best-effort and never raises — purely diagnostic.
    """
    try:
        config = api_config.get_config()
    except Exception:
        logger.debug("Plugin loading summary skipped: configuration unavailable", exc_info=True)
        return

    plugins_dir = config.get("plugins_dir") if config else None
    if not plugins_dir:
        logger.info(
            "Instance plugins: no plugins_dir configured — using built-in datasets, processes and workflows only."
        )
        return

    if not isinstance(plugins_dir, (str, Path)):
        logger.warning(
            "Instance plugins: plugins_dir has unexpected type %s — no instance plugins will load.",
            type(plugins_dir).__name__,
        )
        return

    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    root = (base / str(plugins_dir)).resolve()
    if not root.is_dir():
        logger.warning(
            "Instance plugins: plugins_dir '%s' does not exist or is not a directory — no instance plugins will load.",
            root,
        )
        return

    counts: list[str] = []
    for sub, pattern in _PLUGIN_SUBDIRS:
        directory = root / sub
        if not directory.is_dir():
            # A missing subdir is normal — most instances ship only one or two of
            # datasets/processes/workflows. The summary line below reports it as
            # sub=0, so this is only a debug detail, not a warning.
            logger.debug("Instance plugins: no '%s/' directory under %s.", sub, root)
            counts.append(f"{sub}=0")
            continue
        # Only skip _-prefixed files for Python process plugins; dataset YAML and
        # workflow JSON loaders load all matching files without that convention.
        skip_private = pattern.endswith(".py")
        n = len([p for p in directory.glob(pattern) if not (skip_private and p.name.startswith("_"))])
        counts.append(f"{sub}={n}")
    logger.info("Instance plugins loaded from %s — %s.", root, ", ".join(counts))
