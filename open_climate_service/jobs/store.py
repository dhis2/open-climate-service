"""JSON-backed persistence for native job records."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import portalocker

from open_climate_service import config as api_config
from open_climate_service.jobs.models import JobRecord


def _resolve_jobs_dir() -> Path:
    return api_config.get_data_root() / "jobs"


JOBS_DIR = _resolve_jobs_dir()
JOBS_INDEX_PATH = JOBS_DIR / "jobs.json"


def ensure_store() -> None:
    """Create the jobs metadata store if it does not exist."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_INDEX_PATH.exists():
        JOBS_INDEX_PATH.write_text("[]\n", encoding="utf-8")


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
    with open(JOBS_INDEX_PATH, encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_SH)
        try:
            payload = json.load(handle)
        finally:
            portalocker.unlock(handle)
    if not isinstance(payload, list):
        raise ValueError("jobs.json must contain a list")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("jobs.json must only contain objects")
    return payload


def _mutate_records(mutation: Callable[[list[dict[str, object]]], JobRecord]) -> JobRecord:
    ensure_store()
    with open(JOBS_INDEX_PATH, "r+", encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_EX)
        try:
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("jobs.json must contain a list")
            records = payload
            result = mutation(records)
            handle.seek(0)
            json.dump(records, handle, indent=2)
            handle.write("\n")
            handle.truncate()
            return result
        finally:
            portalocker.unlock(handle)
