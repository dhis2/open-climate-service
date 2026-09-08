"""Idempotency reservations for delivery submissions.

Slice-4 persistence: a JSON index guarded by an exclusive lock. Crash-safe
atomic reservation across a send operation is deferred to CLIM-927 (see the
CLIM-840 approach doc); this store prevents duplicate submission within a
running process and across processes via ``portalocker``, but a crash between
enqueue and reservation can still leave a delivery without a reservation entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import portalocker

from open_climate_service import config as api_config


def _reservations_path() -> Path:
    return api_config.get_data_root() / "exports" / "delivery_reservations.json"


def _ensure_store() -> None:
    path = _reservations_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")


def _load() -> dict[str, dict[str, object]]:
    _ensure_store()
    with open(_reservations_path(), encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_SH)
        try:
            payload = json.load(handle)
        finally:
            portalocker.unlock(handle)
    return payload if isinstance(payload, dict) else {}


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
    _ensure_store()
    with open(_reservations_path(), "r+", encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_EX)
        try:
            payload = json.load(handle)
            records = payload if isinstance(payload, dict) else {}
            if idempotency_key in records:
                raise ValueError(f"Delivery reservation '{idempotency_key}' already exists")
            records[idempotency_key] = {
                "delivery_job_id": delivery_job_id,
                "export_id": export_id,
                "source_job_id": source_job_id,
                "dry_run": dry_run,
                "fingerprint": fingerprint,
            }
            handle.seek(0)
            json.dump(records, handle, indent=2)
            handle.write("\n")
            handle.truncate()
        finally:
            portalocker.unlock(handle)
