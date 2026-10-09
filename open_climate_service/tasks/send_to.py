"""Send a dataset to DHIS2 from its own page: export, workflow task and deliver task in one form (CLIM-1290).

What the pipelines draft (PR #438) did as a new kind of object is here three ordinary things,
created together: a named export (the mapping and the declaration its gate checks), a workflow
task that aggregates the dataset to the org units after each update, and a deliver task after it,
a dry run until someone switches it to live. Each stays visible and editable on its own: on the
Automation page, in ``/exports``, and in the Flows page as one path.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from open_climate_service.tasks.models import Task

router = APIRouter()

STATISTICS = ("mean", "sum", "min", "max", "median")
WORKFLOW = "aggregate_to_dhis2_json"


class SendTo(BaseModel):
    """What the Send to form asks for."""

    connection: str = Field(min_length=1)
    collection: str = Field(min_length=1, description="The org units: a registered feature collection.")
    data_element: str = Field(min_length=11, max_length=11)
    statistic: str = "mean"
    period_type: str | None = Field(default=None, description="Defaults to the dataset's own cadence.")
    dry_run: bool = True


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:60] or "x"


def plan(dataset_id: str, period_type: str | None, request: SendTo) -> tuple[dict[str, Any], Task, Task]:
    """The export and the two tasks a Send to creates, before anything is saved."""
    if request.statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {', '.join(STATISTICS)}")
    export_id = _slug(f"{dataset_id}-{request.data_element}")
    export = {
        "id": export_id,
        "plugin": "dhis2",
        "dataset": dataset_id,
        "connection": request.connection,
        "aggregation": request.statistic,
        "period_type": request.period_type or period_type,
        "series": [{"select": {}, "data_element": request.data_element}],
    }
    workflow = Task(
        id=f"{export_id}-aggregate",
        kind="workflow",
        target=WORKFLOW,
        after={"dataset": dataset_id},  # type: ignore[arg-type]
        arguments={
            "dataset_id": "$event.dataset_id",
            "temporal_extent": ["$event.previous_end", "$event.current_end"],
            "geometries": {"from_features": request.collection},
            "export": export_id,
            "method": request.statistic,
        },
    )
    deliver = Task(
        id=f"{export_id}-deliver",
        kind="deliver",
        target=export_id,
        after={"task": workflow.id},  # type: ignore[arg-type]
        dry_run=request.dry_run,
    )
    return export, workflow, deliver


def create(dataset_id: str, period_type: str | None, request: SendTo) -> tuple[str, str, str]:
    """Save the export and both tasks, or none of them. Raises ValueError naming what was refused."""
    from open_climate_service.exports import store as export_store
    from open_climate_service.exports.routes import _check as check_export
    from open_climate_service.scheduler.service import get_scheduler_service
    from open_climate_service.tasks import store as task_store
    from open_climate_service.tasks.routes import _validator

    export, workflow, deliver = plan(dataset_id, period_type, request)
    if export_store.get_definition(export["id"]) is not None:
        raise ValueError(f"{dataset_id} is already sent to data element {request.data_element}")
    export_store.save_definition(export, check=check_export)
    saved: list[str] = []
    try:
        for task in (workflow, deliver):
            task_store.save_task(task, create=True, check=_validator(task))
            saved.append(task.id)
    except Exception:
        for task_id in reversed(saved):
            task_store.delete_task(task_id)
        export_store.delete_definition(export["id"])
        raise
    get_scheduler_service().reload()
    return export["id"], workflow.id, deliver.id


def options() -> dict[str, list[str]]:
    """What the form can choose from: DHIS2 connections and registered org unit collections."""
    from open_climate_service import config as api_config
    from open_climate_service.exports.dhis2_config import parse_connections

    try:
        connections = sorted(parse_connections(api_config.get_config().get("dhis2_connections", [])))
    except ValueError:
        connections = []
    try:
        from open_climate_service.features.services import registered_collections

        collections = sorted(registered_collections())
    except Exception:
        collections = []
    return {"connections": connections, "collections": collections, "statistics": list(STATISTICS)}


@router.post("/{dataset_id}/send-to", include_in_schema=False)
async def send_to(request: Request, dataset_id: str) -> Any:
    """The dataset page's Send to form, or the same as JSON."""
    from open_climate_service import config as api_config
    from open_climate_service.ingestions.services import get_dataset_or_404
    from open_climate_service.shared.urls import mount_prefix
    from open_climate_service.system.templates import render_dataset_page

    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; nothing can be sent from it")
    record = get_dataset_or_404(dataset_id)
    is_form = "form" in request.headers.get("content-type", "")
    if is_form:
        form = {key: str(value).strip() for key, value in (await request.form()).items() if isinstance(value, str)}
        body: dict[str, Any] = {**form, "dry_run": form.get("dry_run") == "true"}
        if not body.get("period_type"):
            body.pop("period_type", None)
    else:
        body = await request.json()
    try:
        created = create(dataset_id, record.period_type, SendTo.model_validate(body))
    except ValueError as exc:  # pydantic's ValidationError is a ValueError too
        if is_form:
            page = render_dataset_page(record, mount_prefix(request), send_error=str(exc), send_draft=body)
            return HTMLResponse(page, status_code=400)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if is_form:
        return RedirectResponse(f"{mount_prefix(request)}/datasets/{dataset_id}#flow", status_code=303)
    export_id, workflow_id, deliver_id = created
    return {"export": export_id, "tasks": [workflow_id, deliver_id]}
