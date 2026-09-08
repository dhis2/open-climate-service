"""Resolve named mappings and persist pure export results."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from open_climate_service import config
from open_climate_service.exports.base import BaseExportPlugin, RenderedExport
from open_climate_service.exports.registry import load_export_plugins


def render_named_export(data: Any, fmt: str, options: dict[str, Any]) -> tuple[BaseExportPlugin, RenderedExport]:
    """Render a declared export; per-request mapping overrides are not accepted."""
    export_id = options.get("export")
    if not isinstance(export_id, str) or not export_id.strip() or set(options) != {"export"}:
        raise ValueError("Named exports require options containing only a non-empty 'export' ID")
    definitions = config.get_config().get("exports", [])
    if not isinstance(definitions, list):
        raise ValueError("exports must be a list of named mappings")
    by_id: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("Each export definition must be a mapping")
        identifier = definition.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError("Each export requires an ID containing letters, digits, underscores, or hyphens")
        if identifier in by_id:
            raise ValueError(f"Duplicate export ID '{identifier}'")
        by_id[identifier] = definition
    if export_id not in by_id:
        raise ValueError(f"Unknown export '{export_id}'")
    definition = deepcopy(by_id[export_id])
    definition.pop("id")
    plugin_id = definition.pop("plugin", None)
    if not isinstance(plugin_id, str):
        raise ValueError("Export definition requires a plugin ID")
    plugin = load_export_plugins().get(plugin_id)
    if plugin is None:
        raise ValueError(f"Unknown export plugin '{plugin_id}'")
    if plugin.format != fmt:
        raise ValueError(f"Export '{export_id}' requires format '{plugin.format}', received '{fmt}'")
    # Source references are declarations only in this slice. Execution provenance
    # and binding to a delivery target are part of the subsequent manifest phase.
    for field in ("dataset", "org_units", "connection"):
        value = definition.pop(field, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Export {field} must be a non-empty reference string")
    mapping = plugin.validate_mapping(definition)
    rendered = plugin.render(data, mapping)
    # External Python plugins are not necessarily type-checked.
    if not isinstance(rendered, RenderedExport):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("Export plugin render() must return RenderedExport")
    return plugin, rendered


def write_named_export(data: Any, directory: Path, fmt: str, options: dict[str, Any]) -> str:
    """Write the payload and file metadata after rendering succeeds.

    This metadata describes a downloadable file, not a delivery manifest. It does
    not establish source provenance, freeze a target, or authorize later delivery.
    """
    plugin, rendered = render_named_export(data, fmt, options)
    path = directory / f"export{plugin.extension}"
    path.write_bytes(rendered.content)
    metadata = {
        "filename": path.name,
        "media_type": plugin.media_type,
        "format": plugin.format,
        "record_count": rendered.record_count,
        "skipped_count": rendered.skipped_count,
    }
    (directory / ".export.json").write_text(json.dumps(metadata), encoding="utf-8")
    return str(path)


def read_export_metadata(path: Path) -> dict[str, Any] | None:
    """Return saved file metadata only for the matching result asset."""
    metadata_path = path.parent / ".export.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("filename") != path.name:
        return None
    return metadata
