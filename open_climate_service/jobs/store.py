"""JSON-backed persistence for native job records."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from open_climate_service import config as api_config
from open_climate_service.jobs.models import JobRecord
from open_climate_service.shared.persistence import atomic_json, index_lock


def _resolve_jobs_dir() -> Path:
    return api_config.get_data_root() / "jobs"


JOBS_DIR = _resolve_jobs_dir()
JOBS_INDEX_PATH = JOBS_DIR / "jobs.json"


def ensure_store() -> None:
    """Create the jobs metadata store if it does not exist."""
    with index_lock(JOBS_INDEX_PATH):
        if not JOBS_INDEX_PATH.exists():
            atomic_json(JOBS_INDEX_PATH, [])


def list_job_records() -> list[JobRecord]:
    """Return all persisted job records."""
    return [JobRecord.model_validate(raw) for raw in _load_records()]


def get_job_record(job_id: str) -> JobRecord | None:
    """Return one job record if present."""
    for raw in _load_records():
        if raw.get("job_id") == job_id:
            return JobRecord.model_validate(raw)
    return None


def create_job_record(record: JobRecord) -> JobRecord:
    """Persist a newly created job record."""

    def _mutation(records: list[dict[str, object]]) -> JobRecord:
        if any(existing.get("job_id") == record.job_id for existing in records):
            raise ValueError(f"Job '{record.job_id}' already exists")
        records.append(record.model_dump(mode="json"))
        return record

    return _mutate_records(_mutation)


def upsert_job_record(record: JobRecord) -> JobRecord:
    """Persist the full replacement state for one job."""

    def _mutation(records: list[dict[str, object]]) -> JobRecord:
        payload = record.model_dump(mode="json")
        for index, existing in enumerate(records):
            if existing.get("job_id") == record.job_id:
                records[index] = payload
                return record
        records.append(payload)
        return record

    return _mutate_records(_mutation)


def mutate_job_record(job_id: str, mutation: Callable[[JobRecord], JobRecord]) -> JobRecord:
    """Load, mutate, and persist one existing job record."""

    def _apply(records: list[dict[str, object]]) -> JobRecord:
        for index, existing in enumerate(records):
            if existing.get("job_id") != job_id:
                continue
            current = JobRecord.model_validate(existing)
            updated = mutation(current)
            records[index] = updated.model_dump(mode="json")
            return updated
        raise KeyError(job_id)

    return _mutate_records(_apply)


def _load_records() -> list[dict[str, object]]:
    ensure_store()
    return _read_records_from_disk()


def _read_records_from_disk() -> list[dict[str, object]]:
    with index_lock(JOBS_INDEX_PATH):
        with open(JOBS_INDEX_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("jobs.json must contain a list")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("jobs.json must only contain objects")
    return payload


def _mutate_records(mutation: Callable[[list[dict[str, object]]], JobRecord]) -> JobRecord:
    ensure_store()
    with index_lock(JOBS_INDEX_PATH):
        with open(JOBS_INDEX_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("jobs.json must contain a list")
        records = payload
        result = mutation(records)
        _atomic_write_records(records)
        return result


def _atomic_write_records(records: list[dict[str, object]]) -> None:
    """Replace jobs.json by first flushing a complete temporary copy.

    The in-place rewrite used before could leave a truncated index if the process
    crashed mid-write. Writing a sibling file and replacing the index atomically
    keeps delivery checkpoints and idempotency state durable across a crash.
    """
    atomic_json(JOBS_INDEX_PATH, records)
