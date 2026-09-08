"""Durable idempotency reservations, written before a delivery is enqueued."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from open_climate_service import config as api_config
from open_climate_service.shared.persistence import atomic_json, index_lock


def _reservations_path() -> Path:
    return api_config.get_data_root() / "exports" / "delivery_reservations.json"


def _load() -> dict[str, dict[str, object]]:
    path = _reservations_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(value, dict) for value in payload.values()):
        raise ValueError("Invalid delivery reservation index")
    return payload


@contextmanager
def submission_lock() -> Iterator[None]:
    """Serialize reservation, job creation and enqueue across all API workers."""
    with index_lock(_reservations_path()):
        yield


def find_delivery(idempotency_key: str) -> dict[str, object] | None:
    """Return the reservation for one idempotency key, or None."""
    return _load().get(idempotency_key)


def reserve_delivery(
    idempotency_key: str,
    *,
    delivery_job_id: str,
    export_id: str,
    source_job_id: str,
    dry_run: bool,
    fingerprint: str,
) -> None:
    """Persist one delivery reservation, refusing a duplicate key."""
    # Caller holds submission_lock through enqueue; a retry can recreate a job
    # missing after a crash using this already reserved ID.
    records = _load()
    if idempotency_key in records:
        raise ValueError("Delivery reservation already exists")
    records[idempotency_key] = {
        "delivery_job_id": delivery_job_id,
        "export_id": export_id,
        "source_job_id": source_job_id,
        "dry_run": dry_run,
        "fingerprint": fingerprint,
    }
    atomic_json(_reservations_path(), records)
