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


class ExportList(BaseModel):
    """Every named export, by id."""

    exports: list[dict[str, Any]]


def _require_writable() -> None:
    from open_climate_service import config as api_config

    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; exports cannot be changed")


def _check(definition: dict[str, Any]) -> None:
    """Refuse a definition its plugin would refuse when a workflow or a delivery uses it."""
    from open_climate_service.exports.service import resolve_definition

    resolved = resolve_definition(definition)
    if resolved.references.get("connection") is not None:
        resolved.plugin.check_delivery_target(resolved.references)


def _apply() -> None:
    """Automation re-validates the deliveries that name exports; the clock's reload tells it."""
    from open_climate_service.scheduler.service import get_scheduler_service

    get_scheduler_service().reload()


@router.get("", response_model=ExportList)
def list_exports() -> ExportList:
    """Every named export definition (CLIM-1089)."""
    from open_climate_service.exports import store

    return ExportList(exports=sorted(store.list_definitions(), key=lambda item: str(item.get("id"))))


@router.get("/{export_id}")
def get_export(export_id: str) -> dict[str, Any]:
    """One named export definition."""
    from open_climate_service.exports import store

    definition = store.get_definition(export_id)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"No export '{export_id}'")
    return definition


@router.put("/{export_id}")
def put_export(export_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Create or replace a named export, validated by its plugin before it is saved. No restart."""
    from open_climate_service.exports import store

    _require_writable()
    definition = {**body, "id": export_id}
    try:
        store.save_definition(definition, check=_check)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _apply()
    return definition


@router.delete("/{export_id}", status_code=204)
def delete_export(export_id: str) -> None:
    """Remove a named export. A deliver task that still names it is refused at its next save."""
    from open_climate_service.exports import store

    _require_writable()
    if not store.delete_definition(export_id):
        raise HTTPException(status_code=404, detail=f"No export '{export_id}'")
    _apply()


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
    if record.process_id != f"export:{export_id}" or record.request.get("export_id") != export_id:
        raise HTTPException(status_code=404, detail="Delivery job not found for this export")
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
