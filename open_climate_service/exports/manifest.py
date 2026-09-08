"""Versioned export manifests and atomic publication of file generations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from open_climate_service.exports.base import BaseExportPlugin
from open_climate_service.shared.provenance import json_digest


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PluginIdentity(_Model):
    """Renderer identity whose version is owned by the plugin author."""

    id: str
    version: str | None
    implementation: str
    format: str
    media_type: str
    extension: str


class TargetBinding(_Model):
    """Target identity without its URL, token, or environment-variable value."""

    connection_id: str
    url_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExportManifest(_Model):
    """A saved payload bound to its source job, mapping, and optional target."""

    schema_version: Literal[1] = 1
    source_job_id: str
    export_id: str
    created_at: str
    filename: str = Field(pattern=r"^export-[0-9a-f]{32}\.[a-z0-9]+$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_size: int = Field(ge=0)
    record_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    periods: list[str] | None
    plugin: PluginIdentity
    mapping: dict[str, Any]
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    references: dict[str, str]
    target: TargetBinding | None
    provenance: dict[str, Any]


def validate_public_mapping(value: Any) -> None:
    """Require JSON-compatible, non-secret literal configuration for export metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Export mapping keys must be strings")
            if key.lower().replace("-", "_") in {
                "token",
                "password",
                "secret",
                "authorization",
                "credentials",
                "api_key",
                "access_token",
            }:
                raise ValueError("Export mappings must not contain credentials; use connection references")
            validate_public_mapping(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_public_mapping(child)
    elif isinstance(value, str) and "${" in value:
        raise ValueError("Export mappings must be literal; environment interpolation is not supported")
    # Canonical serialization also rejects unsupported types and NaN/Infinity.
    try:
        json_digest(value)
    except (ValueError, TypeError):
        raise ValueError("Export mappings must contain finite JSON values") from None


def plugin_identity(plugin: BaseExportPlugin) -> PluginIdentity:
    """Capture the renderer descriptor before calling its implementation."""
    cls = type(plugin)
    return PluginIdentity(
        id=plugin.id,
        version=plugin.version,
        implementation=f"{cls.__module__}.{cls.__qualname__}",
        format=plugin.format,
        media_type=plugin.media_type,
        extension=plugin.extension,
    )


def target_binding(plugin_id: str, references: dict[str, str]) -> TargetBinding | None:
    """Bind a named DHIS2 target for built-in or installed export implementations."""
    connection_id = references.get("connection")
    if connection_id is None:
        return None
    from open_climate_service.exports.dhis2 import get_connection_config

    parts = urlsplit(get_connection_config(connection_id).url)
    hostname = parts.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parts.port
    if port is not None and (parts.scheme, port) not in {("https", 443), ("http", 80)}:
        hostname += f":{port}"
    canonical = urlunsplit((parts.scheme, hostname, parts.path.rstrip("/"), "", ""))
    return TargetBinding(connection_id=connection_id, url_sha256=hashlib.sha256(canonical.encode()).hexdigest())


def atomic_write(path: Path, content: bytes) -> None:
    """Publish one file by replacement after flushing its complete contents."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".export-", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def publish_manifest(directory: Path, manifest: ExportManifest, payload: bytes) -> str:
    """Commit a fresh generation by replacing the metadata pointer last.

    Failed generations may leave unreferenced files, but never overwrite a
    previously committed payload or advertise a partially written generation.
    """
    path = directory / manifest.filename
    manifest_name = f"{path.stem}.manifest.json"
    manifest_bytes = manifest.model_dump_json().encode()
    atomic_write(path, payload)
    atomic_write(directory / manifest_name, manifest_bytes)
    metadata = {
        "filename": path.name,
        "format": manifest.plugin.format,
        "media_type": manifest.plugin.media_type,
        "record_count": manifest.record_count,
        "skipped_count": manifest.skipped_count,
        "manifest": manifest_name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    # Ensure generation entries precede the pointer in the filesystem journal.
    if os.name == "nt":
        # Windows does not expose POSIX directory fsync; replacement remains atomic.
        atomic_write(directory / ".export.json", json.dumps(metadata).encode())
        return str(path)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
        atomic_write(directory / ".export.json", json.dumps(metadata).encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return str(path)
