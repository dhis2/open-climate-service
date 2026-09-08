"""Background delivery of a saved export payload and its submission."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from open_climate_service.exports.base import DeliveryContext
from open_climate_service.exports.delivery_input import VerifiedExport, lease_export_input
from open_climate_service.exports.report import ExportReport
from open_climate_service.shared.provenance import json_digest

logger = logging.getLogger(__name__)


class JobDeliveryContext:
    """Adapt the native job's execution hooks to the plugin delivery protocol.

    Checkpoints are namespaced under ``delivery_checkpoints`` inside the job
    cursor so a plugin's chunk state never collides with other cursor users.
    """

    def __init__(
        self,
        *,
        on_progress: Any = None,
        is_cancel_requested: Any = None,
        save_cursor: Any = None,
        load_cursor: Any = None,
    ) -> None:
        self._on_progress = on_progress
        self._is_cancel_requested = is_cancel_requested
        self._save_cursor = save_cursor
        self._load_cursor = load_cursor

    def report_progress(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total, message)

    def is_cancel_requested(self) -> bool:
        return bool(self._is_cancel_requested is not None and self._is_cancel_requested())

    def save_checkpoint(self, key: str, state: dict[str, Any]) -> None:
        if self._save_cursor is None:
            return
        cursor = self._load_cursor() if self._load_cursor is not None else None
        cursor = cursor if isinstance(cursor, dict) else {}
        checkpoints = cursor.get("delivery_checkpoints")
        checkpoints = dict(checkpoints) if isinstance(checkpoints, dict) else {}
        checkpoints[key] = state
        self._save_cursor({**cursor, "delivery_checkpoints": checkpoints})

    def load_checkpoint(self, key: str) -> dict[str, Any] | None:
        if self._load_cursor is None:
            return None
        cursor = self._load_cursor()
        if not isinstance(cursor, dict):
            return None
        checkpoints = cursor.get("delivery_checkpoints")
        if not isinstance(checkpoints, dict):
            return None
        return checkpoints.get(key)


def deliver_named_export(
    export_id: str,
    job_id: str,
    dry_run: bool = False,
    on_progress: Any = None,
    is_cancel_requested: Any = None,
    save_cursor: Any = None,
    load_cursor: Any = None,
) -> dict[str, Any]:
    """Send a completed source job's saved payload through its export plugin.

    Module-level so the native job store can re-import it on restart. Runs on the
    native job thread; the source result lease is held for the entire send.
    """
    if on_progress is not None:
        on_progress(0, 1, "Delivering export")
    context = JobDeliveryContext(
        on_progress=on_progress,
        is_cancel_requested=is_cancel_requested,
        save_cursor=save_cursor,
        load_cursor=load_cursor,
    )
    with lease_export_input(export_id, job_id) as verified:
        report = _deliver(verified, export_id, dry_run, context)
    if on_progress is not None:
        on_progress(1, 1, "Delivery complete")
    return report.model_dump(mode="json")


def _deliver(verified: VerifiedExport, export_id: str, dry_run: bool, context: DeliveryContext) -> ExportReport:
    from open_climate_service.exports.service import resolve_named_export

    resolved = resolve_named_export(verified.manifest.plugin.format, {"export": export_id})
    if not resolved.plugin.supports_delivery:
        raise ValueError(f"Export plugin '{resolved.plugin.id}' does not support delivery")
    target = verified.manifest.target.connection_id if verified.manifest.target else None
    if target is None:
        raise ValueError("Export was rendered without a delivery target")
    report = resolved.plugin.send(verified.content, target, dry_run=dry_run, context=context)
    # External Python plugins are not necessarily type-checked.
    if not isinstance(report, ExportReport):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("Export plugin send() must return ExportReport")
    return report


def submit_delivery(
    export_id: str,
    source_job_id: str,
    dry_run: bool,
    idempotency_key: str,
) -> tuple[str, bool]:
    """Verify, dedupe by idempotency key, enqueue, and reserve one delivery.

    Returns ``(delivery_job_id, reused)``. Raises 409 when an idempotency key is
    reused with different content, and 404/409 for invalid source jobs or
    mappings via :func:`lease_export_input`.
    """
    from open_climate_service.exports.reservations import find_delivery, reserve_delivery
    from open_climate_service.jobs.service import get_job_service

    # Verify once under a brief lease to capture the content fingerprint before
    # enqueuing; the worker re-acquires the lease for the actual send.
    with lease_export_input(export_id, source_job_id) as verified:
        connection_id = verified.manifest.target.connection_id if verified.manifest.target else None
        fingerprint = _fingerprint(export_id, source_job_id, dry_run, verified.manifest.payload_sha256, connection_id)

    existing = find_delivery(idempotency_key)
    if existing is not None:
        if existing.get("fingerprint") == fingerprint:
            return str(existing["delivery_job_id"]), True
        raise HTTPException(status_code=409, detail="Idempotency key reused with different delivery content")

    record = get_job_service().submit_callable_job(
        func=deliver_named_export,
        label=f"export:{export_id}",
        request={"export_id": export_id, "job_id": source_job_id, "dry_run": dry_run},
        job_href_base=f"/exports/{export_id}/jobs",
    )
    reserve_delivery(
        idempotency_key,
        delivery_job_id=record.job_id,
        export_id=export_id,
        source_job_id=source_job_id,
        dry_run=dry_run,
        fingerprint=fingerprint,
    )
    link_source_job(source_job_id, export_id, record.job_id)
    return record.job_id, False


def link_source_job(source_job_id: str, export_id: str, delivery_job_id: str) -> None:
    """Advertise a delivery job from its source openEO job's result metadata."""
    from open_climate_service.openeo.jobs import store_update_job
    from open_climate_service.shared.time import utc_now

    entry = {
        "delivery_job_id": delivery_job_id,
        "export_id": export_id,
        "status_url": f"/exports/{export_id}/jobs/{delivery_job_id}",
        "created_at": utc_now().isoformat(),
    }

    def _mutation(record: Any) -> Any:
        usage = dict(record.usage or {})
        deliveries = usage.get("deliveries")
        deliveries = list(deliveries) if isinstance(deliveries, list) else []
        if not any(isinstance(item, dict) and item.get("delivery_job_id") == delivery_job_id for item in deliveries):
            deliveries.append(entry)
        usage["deliveries"] = deliveries
        return record.model_copy(update={"usage": usage})

    try:
        store_update_job(source_job_id, _mutation)
    except KeyError:
        # The source job vanished between verification and linking; the delivery
        # job is already enqueued, so do not fail the accepted submission.
        logger.warning("Could not link delivery '%s' to missing source job '%s'", delivery_job_id, source_job_id)


def _fingerprint(
    export_id: str,
    source_job_id: str,
    dry_run: bool,
    payload_sha256: str,
    connection_id: str | None,
) -> str:
    return json_digest(
        {
            "export_id": export_id,
            "source_job_id": source_job_id,
            "dry_run": dry_run,
            "payload_sha256": payload_sha256,
            "connection_id": connection_id,
        }
    )
