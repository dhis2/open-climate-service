"""Pipelines persist as one JSON file each under the data directory."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

from open_climate_service import config as api_config
from open_climate_service.pipelines.schemas import PipelineRecord, PipelineSpec, ValidationResult
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def pipelines_dir() -> Path:
    """Where pipeline records live: `<data_dir>/pipelines`."""
    return api_config.get_data_root() / "pipelines"


def _path(pipeline_id: str) -> Path:
    return pipelines_dir() / f"{pipeline_id}.json"


def list_records() -> list[PipelineRecord]:
    """Every stored pipeline, by id; an unreadable file is skipped rather than failing the list."""
    directory = pipelines_dir()
    if not directory.is_dir():
        return []
    records: list[PipelineRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(PipelineRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.exception("Skipping unreadable pipeline record %s", path.name)
            continue
    return records


def get_record(pipeline_id: str) -> PipelineRecord | None:
    """One stored pipeline, or None."""
    path = _path(pipeline_id)
    try:
        return PipelineRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def save_record(record: PipelineRecord) -> PipelineRecord:
    """Write a record atomically and return it."""
    directory = pipelines_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(record.spec.id)
    with _LOCK:
        _write_record(record, target, directory)
    return record


def _write_record(record: PipelineRecord, target: Path, directory: Path) -> None:
    """Write while the caller holds ``_LOCK``."""
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".pipeline-", encoding="utf-8", delete=False) as handle:
        handle.write(record.model_dump_json(indent=2))
        handle.write("\n")
    os.replace(handle.name, target)


def create_record(spec: PipelineSpec, validation: ValidationResult | None = None) -> PipelineRecord:
    """Store a new pipeline with the validation it was saved on; an existing id is refused."""
    directory = pipelines_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(spec.id)
    record = PipelineRecord(spec=spec, created_at=utc_now().isoformat(), validation=validation)
    with _LOCK:
        if target.exists():
            raise ValueError(f"A pipeline with id '{spec.id}' already exists")
        _write_record(record, target, directory)
    return record


def delete_record(pipeline_id: str) -> bool:
    """Remove a stored pipeline; True when something was removed."""
    with _LOCK:
        try:
            _path(pipeline_id).unlink()
        except FileNotFoundError:
            return False
    return True
