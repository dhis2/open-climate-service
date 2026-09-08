"""Resolve named mappings and persist pure export results."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from open_climate_service import config
from open_climate_service.exports.base import BaseExportPlugin, RenderedExport
from open_climate_service.exports.registry import load_export_plugins


@dataclass(frozen=True)
class ResolvedExport:
    """Invocation-local mapping and source/target references."""

    export_id: str
    plugin: BaseExportPlugin
    mapping: dict[str, Any]
    references: dict[str, str]


def resolve_named_export(fmt: str, options: dict[str, Any]) -> ResolvedExport:
    """Resolve and validate a mapping without rendering or resolving a target."""
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
    references: dict[str, str] = {}
    for field in ("dataset", "org_units", "connection"):
        value = definition.pop(field, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Export {field} must be a non-empty reference string")
        if value is not None:
            references[field] = value.strip()
    from open_climate_service.exports.manifest import validate_public_mapping

    validate_public_mapping(definition)
    mapping = plugin.validate_mapping(definition)
    validate_public_mapping(mapping)
    return ResolvedExport(export_id, plugin, deepcopy(mapping), references)


def render_named_export(data: Any, fmt: str, options: dict[str, Any]) -> tuple[BaseExportPlugin, RenderedExport]:
    """Render a declared export without resolving a connection or credential."""
    resolved = resolve_named_export(fmt, options)
    return resolved.plugin, _render(data, resolved)


def _render(data: Any, resolved: ResolvedExport) -> RenderedExport:
    rendered = resolved.plugin.render(data, deepcopy(resolved.mapping))
    # External Python plugins are not necessarily type-checked.
    if not isinstance(rendered, RenderedExport):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("Export plugin render() must return RenderedExport")
    return rendered


def write_named_export(
    data: Any,
    directory: Path,
    fmt: str,
    options: dict[str, Any],
    *,
    job_id: str,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Freeze a resolved invocation and publish its payload and versioned manifest."""
    import hashlib

    from open_climate_service.exports.manifest import (
        ExportManifest,
        plugin_identity,
        publish_manifest,
        target_binding,
        validate_public_mapping,
    )
    from open_climate_service.shared.provenance import json_digest
    from open_climate_service.shared.time import utc_now

    resolved = resolve_named_export(fmt, options)
    identity = plugin_identity(resolved.plugin)
    target = target_binding(resolved.plugin.id, resolved.references)
    evidence = (
        deepcopy(provenance)
        if provenance is not None
        else {
            "scope": "unavailable",
            "sources": [],
            "features": [],
            "missing": ["execution_provenance"],
        }
    )
    validate_public_mapping(evidence)
    declared = resolved.references.get("dataset")
    sources_value = evidence.get("sources", [])
    if not isinstance(sources_value, list) or not all(isinstance(source, dict) for source in sources_value):
        raise ValueError("Export provenance sources must be a list of mappings")
    sources = cast(list[dict[str, Any]], sources_value)
    if (
        declared is not None
        and sources
        and not any(declared in (source.get("collection_id"), source.get("source_dataset_id")) for source in sources)
    ):
        raise ValueError("Declared export dataset was not observed during execution")
    rendered = _render(data, resolved)
    manifest = ExportManifest(
        source_job_id=job_id,
        export_id=resolved.export_id,
        created_at=utc_now().isoformat(),
        filename=f"export-{uuid4().hex}{identity.extension}",
        payload_sha256=hashlib.sha256(rendered.content).hexdigest(),
        payload_size=len(rendered.content),
        record_count=rendered.record_count,
        skipped_count=rendered.skipped_count,
        periods=sorted(set(rendered.periods)) if rendered.periods is not None else None,
        plugin=identity,
        mapping=resolved.mapping,
        mapping_sha256=json_digest(resolved.mapping),
        references=resolved.references,
        target=target,
        provenance=evidence,
    )
    return publish_manifest(directory, manifest, rendered.content)


def read_export_metadata(path: Path) -> dict[str, Any] | None:
    """Return saved file metadata only for the matching result asset."""
    metadata_path = path.parent / ".export.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("filename") != path.name:
        return None
    return metadata
