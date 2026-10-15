"""Pipelines persist as one JSON file each under the data directory.

Every change is a read-modify-write under a per-pipeline file lock shared by all API
workers, so a run hand-off on a job thread and an edit in a request never write over
each other. Callers pass a mutation, not a record they read earlier.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from open_climate_service import config as api_config
from open_climate_service.pipelines.schemas import EventCheckpoint, PipelineRecord, PipelineSpec, ValidationResult
from open_climate_service.shared.persistence import index_lock
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class StaleRevision(ValueError):
    """An edit was based on a revision that has since changed."""


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
    try:
        return PipelineRecord.model_validate_json(_path(pipeline_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_record(record: PipelineRecord, target: Path) -> None:
    """Write atomically while the caller holds the record's lock."""
    with tempfile.NamedTemporaryFile(
        "w", dir=target.parent, prefix=".pipeline-", encoding="utf-8", delete=False
    ) as handle:
        handle.write(record.model_dump_json(indent=2))
        handle.write("\n")
    os.replace(handle.name, target)


def update_record(pipeline_id: str, mutation: Callable[[PipelineRecord], _T]) -> tuple[PipelineRecord, _T]:
    """Apply ``mutation`` to the current record under its lock and store the result.

    The mutation edits the record in place and may raise to abort without writing. Raises
    LookupError when the pipeline does not exist.
    """
    target = _path(pipeline_id)
    with index_lock(target):
        record = get_record(pipeline_id)
        if record is None:
            raise LookupError(f"Unknown pipeline '{pipeline_id}'")
        result = mutation(record)
        record.updated_at = utc_now().isoformat()
        _write_record(record, target)
    return record, result


def create_record(spec: PipelineSpec, validation: ValidationResult | None = None) -> PipelineRecord:
    """Store a new pipeline with the validation it was saved on; an existing id is refused."""
    target = _path(spec.id)
    now = utc_now().isoformat()
    if validation is not None:
        validation = validation.model_copy(update={"revision": 1})
    record = PipelineRecord(spec=spec, created_at=now, updated_at=now, validation=validation)
    if automates(spec):
        _start_automation(record, now)
    with index_lock(target):
        if target.exists():
            raise ValueError(f"A pipeline with id '{spec.id}' already exists")
        _write_record(record, target)
    return record


def save_record(record: PipelineRecord) -> PipelineRecord:
    """Overwrite a record as a whole. Only for tests and migrations; requests use `update_record`."""
    target = _path(record.spec.id)
    with index_lock(target):
        _write_record(record, target)
    return record


def delete_record(pipeline_id: str) -> bool:
    """Remove a stored pipeline; True when something was removed."""
    target = _path(pipeline_id)
    with index_lock(target):
        try:
            target.unlink()
        except FileNotFoundError:
            return False
    return True


def install_validation(record: PipelineRecord, validation: ValidationResult, *, judged_revision: int) -> None:
    """Store a validation on the record, refusing one computed for a revision that has since changed.

    Validation runs outside the lock (it may contact the destination), so the record can move
    on meanwhile; installing the old verdict would publish an export for a spec it never saw.
    """
    if record.revision != judged_revision:
        raise StaleRevision(
            f"pipeline '{record.spec.id}' changed while it was being checked (revision {judged_revision} became "
            f"{record.revision}); check it again"
        )
    record.validation = validation.model_copy(update={"revision": record.revision})


def automates(spec: PipelineSpec) -> bool:
    """Whether dataset updates start runs of this pipeline by themselves."""
    return spec.delivery.policy == "on_update" and spec.delivery.mode != "paused"


def replace_spec(record: PipelineRecord, spec: PipelineSpec, *, expected_revision: int | None) -> None:
    """Swap the spec in place, refusing a stale edit and stamping when automation starts or stops."""
    if expected_revision is not None and expected_revision != record.revision:
        raise StaleRevision(
            f"pipeline '{record.spec.id}' changed since revision {expected_revision} was read (now "
            f"{record.revision}); reload it and apply the change again"
        )
    was, now = automates(record.spec), automates(spec)
    mode_changed = spec.delivery.mode != record.spec.delivery.mode
    bindings_changed = spec.workflow != record.spec.workflow or spec.destination != record.spec.destination
    if spec != record.spec:
        record.revision += 1
    record.spec = spec
    # A new boundary whenever automation starts, changes mode or changes what it computes:
    # updates from before going live are never delivered live, and updates that arrived for
    # other bindings are never run with the new ones.
    if now and (not was or mode_changed or bindings_changed):
        _start_automation(record, utc_now().isoformat())
    elif not now:
        record.automation_since = None
        record.events = EventCheckpoint(
            skipped=record.events.skipped,
            skipped_discarded=record.events.skipped_discarded,
            skipped_discarded_through=record.events.skipped_discarded_through,
        )


def _start_automation(record: PipelineRecord, since: str) -> None:
    """Start reacting to dataset updates from ``since``: nothing earlier is ever run by itself."""
    record.automation_since = since
    record.events = EventCheckpoint(
        through=since,
        skipped=record.events.skipped,
        skipped_discarded=record.events.skipped_discarded,
        skipped_discarded_through=record.events.skipped_discarded_through,
    )
