"""Operator delivery routes for named exports.

These routes are deliberately closed in read-only mode (see ``read_only.py``):
delivery exposes server-held credentials to an operation with external effects.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from open_climate_service.exports.delivery import submit_delivery
from open_climate_service.exports.report import ExportReport
from open_climate_service.jobs.service import get_job_service

router = APIRouter(tags=["Exports"])


class DeliveryRequest(BaseModel):
    """Request body for POST /exports/{export_id}."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    dry_run: bool = False


class DeliveryAccepted(BaseModel):
    """Accepted delivery job plus links to its status and report."""

    delivery_job_id: str
    export_id: str
    dry_run: bool
    reused: bool
    status_url: str
    report_url: str


@router.post("/{export_id}", status_code=202, response_model=DeliveryAccepted)
def deliver_export(
    export_id: str,
    body: DeliveryRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DeliveryAccepted:
    """Queue delivery of a completed source job's saved export payload.

    Reusing an idempotency key with identical content returns the existing
    delivery; different content is a conflict.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="An Idempotency-Key header is required")
    delivery_job_id, reused = submit_delivery(export_id, body.job_id, body.dry_run, idempotency_key.strip())
    status_url = f"/exports/{export_id}/jobs/{delivery_job_id}"
    return DeliveryAccepted(
        delivery_job_id=delivery_job_id,
        export_id=export_id,
        dry_run=body.dry_run,
        reused=reused,
        status_url=status_url,
        report_url=status_url,
    )


@router.get("/{export_id}/jobs/{delivery_job_id}")
def get_delivery_job(export_id: str, delivery_job_id: str) -> dict[str, Any]:
    """Return one delivery job's status and, when finished, its export report."""
    record = get_job_service().get_job_or_404(delivery_job_id)
    report: ExportReport | None = None
    if isinstance(record.result, dict):
        try:
            report = ExportReport.model_validate(record.result)
        except Exception:
            report = None
    return {
        "delivery_job_id": record.job_id,
        "export_id": export_id,
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
        "error": record.error.model_dump(mode="json") if record.error else None,
        "report": report.model_dump(mode="json") if report else None,
    }
