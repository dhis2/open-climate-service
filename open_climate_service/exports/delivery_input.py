"""Validate and lease immutable input for a future delivery worker; never send it."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

from open_climate_service.exports.manifest import ExportManifest, plugin_identity, target_binding
from open_climate_service.exports.retention import result_lease
from open_climate_service.exports.service import read_export_metadata, resolve_named_export
from open_climate_service.shared.provenance import json_digest


@dataclass(frozen=True)
class VerifiedExport:
    """Exact saved bytes and their manifest, valid while the source lease is held."""

    content: bytes
    manifest: ExportManifest


@contextmanager
def lease_export_input(export_id: str, job_id: str) -> Iterator[VerifiedExport]:
    """Check job status, integrity, mapping, renderer, and target under a lease.

    This is not authorization or delivery. The future operator endpoint must apply
    access/read-only policy before invoking it and hold the lease through sending.
    """
    with result_lease(job_id):
        yield _verify(export_id, job_id)


def _verify(export_id: str, job_id: str) -> VerifiedExport:
    from open_climate_service.openeo.jobs import _JOBS_DIR, store_get_job
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    record = store_get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Source job not found")
    if record.status != OpenEOJobStatus.FINISHED:
        raise HTTPException(status_code=409, detail="Export requires a successfully completed source job")
    directory = (_JOBS_DIR / job_id / "results").resolve()
    try:
        output = (record.usage or {}).get("output_path")
        if not isinstance(output, str):
            raise ValueError("Source job has no saved output")
        path = Path(output)
        if path.is_symlink() or path.resolve().parent != directory or not path.is_file():
            raise ValueError("Source payload is missing or outside its job results")
        metadata = read_export_metadata(path)
        if metadata is None or "manifest" not in metadata:
            raise ValueError("This result predates delivery manifests; create a new named export")
        manifest_path = directory / metadata["manifest"]
        if manifest_path.is_symlink() or manifest_path.resolve().parent != directory:
            raise ValueError("Invalid manifest path")
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata.get("manifest_sha256"):
            raise ValueError("Saved export manifest has changed")
        manifest = ExportManifest.model_validate_json(raw)
        if manifest.source_job_id != job_id or manifest.export_id != export_id or manifest.filename != path.name:
            raise ValueError("Saved payload does not belong to this job and export")
        if (
            metadata.get("filename") != manifest.filename
            or metadata.get("format") != manifest.plugin.format
            or metadata.get("media_type") != manifest.plugin.media_type
            or metadata.get("record_count") != manifest.record_count
            or metadata.get("skipped_count") != manifest.skipped_count
        ):
            raise ValueError("Saved export metadata does not match its manifest")
        content = path.read_bytes()
        if len(content) != manifest.payload_size or hashlib.sha256(content).hexdigest() != manifest.payload_sha256:
            raise ValueError("Saved export payload has changed")
        if json_digest(manifest.mapping) != manifest.mapping_sha256:
            raise ValueError("Saved mapping digest does not match its contents")
        resolved = resolve_named_export(manifest.plugin.format, {"export": export_id})
        if json_digest(resolved.mapping) != manifest.mapping_sha256 or resolved.references != manifest.references:
            raise ValueError("Export mapping or references changed; create a new named export")
        if not manifest.plugin.version or plugin_identity(resolved.plugin) != manifest.plugin:
            raise ValueError("Export renderer changed or is unversioned; create a new named export")
        if manifest.target is None:
            raise ValueError("Export was rendered without a delivery target; create a new bound export")
        if target_binding(resolved.plugin.id, resolved.references) != manifest.target:
            raise ValueError("DHIS2 target changed; create a new named export")
        graph_digest = manifest.provenance.get("process_sha256")
        if graph_digest is not None and json_digest(record.process) != graph_digest:
            raise ValueError("Source job process changed after rendering")
    except (OSError, ValueError, TypeError, KeyError, ValidationError) as exc:
        # ValidationError includes arbitrary input; never expose its rendered dump.
        detail = "Invalid export manifest" if isinstance(exc, ValidationError) else str(exc)
        raise HTTPException(status_code=409, detail=detail) from None
    return VerifiedExport(content, manifest)
