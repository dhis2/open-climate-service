"""Operator delivery routes for named destinations.

These routes are deliberately closed in read-only mode (see ``read_only.py``):
delivery exposes server-held credentials to an operation with external effects.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from open_climate_service.destinations.delivery import submit_delivery
from open_climate_service.destinations.report import DeliveryReport
from open_climate_service.jobs.service import get_job_service

router = APIRouter(tags=["Destinations"])


class DeliveryRequest(BaseModel):
    """Request body for POST /destinations/{destination_id}."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    dry_run: bool = False


class DeliveryAccepted(BaseModel):
    """Accepted delivery job plus links to its status and report."""

    delivery_job_id: str
    destination_id: str
    dry_run: bool
    reused: bool
    status_url: str
    report_url: str


@router.post("/{destination_id}", status_code=202, response_model=DeliveryAccepted)
def deliver_export(
    destination_id: str,
    body: DeliveryRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> DeliveryAccepted:
    """Queue delivery of a completed source job's saved payload.

    Reusing an idempotency key with identical content returns the existing
    delivery; different content is a conflict.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="An Idempotency-Key header is required")
    delivery_job_id, reused = submit_delivery(destination_id, body.job_id, body.dry_run, idempotency_key.strip())
    status_url = f"/destinations/{destination_id}/jobs/{delivery_job_id}"
    return DeliveryAccepted(
        delivery_job_id=delivery_job_id,
        destination_id=destination_id,
        dry_run=body.dry_run,
        reused=reused,
        status_url=status_url,
        report_url=status_url,
    )


@router.get("/{destination_id}/jobs/{delivery_job_id}")
def get_delivery_job(destination_id: str, delivery_job_id: str) -> dict[str, Any]:
    """Return one delivery job's status and, when finished, its delivery report."""
    record = get_job_service().get_job_or_404(delivery_job_id)
    if record.process_id != f"destination:{destination_id}" or record.request.get("destination_id") != destination_id:
        raise HTTPException(status_code=404, detail="Delivery job not found for this destination")
    report: DeliveryReport | None = None
    if isinstance(record.result, dict):
        try:
            report = DeliveryReport.model_validate(record.result)
        except Exception:
            report = None
    return {
        "delivery_job_id": record.job_id,
        "destination_id": destination_id,
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
        "error": record.error.model_dump(mode="json") if record.error else None,
        "report": report.model_dump(mode="json") if report else None,
    }
