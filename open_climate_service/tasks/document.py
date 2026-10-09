"""The operational configuration as one document: every task and export, out and back in (CLIM-1378).

An instance whose configuration is tracked in git exports this document, commits it, and imports
it on another instance or after a rebuild. There is one live source, the operational database;
the document is a copy of it, not a second source with a precedence rule.

``climate-service.yaml`` keeps the infrastructure: the clock's settings, connections and their
secrets, plugin folders, read-only. What runs, and where its results go, is in this document.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError

from open_climate_service.tasks.models import Task

router = APIRouter()

DOCUMENT_VERSION = 1


class OperationalConfiguration(BaseModel):
    """Every task and named export on the instance."""

    version: int = DOCUMENT_VERSION
    tasks: list[Task] = Field(default_factory=list)
    exports: list[dict[str, Any]] = Field(default_factory=list)


def current() -> OperationalConfiguration:
    """The configuration as stored now."""
    from open_climate_service.exports import store as export_store
    from open_climate_service.tasks import store as task_store

    return OperationalConfiguration(
        tasks=task_store.list_tasks(),
        exports=sorted(export_store.list_definitions(), key=lambda item: str(item.get("id"))),
    )


def validate(document: OperationalConfiguration) -> None:
    """Refuse a document any part of which would be refused on its own. Raises ValueError."""
    from open_climate_service.automation.config import compile_tasks
    from open_climate_service.automation.service import validate_automation
    from open_climate_service.exports.service import resolve_definition
    from open_climate_service.tasks.routes import _validate_structure, check_target

    ids = [str(item.get("id")) for item in document.exports]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Export ids must be unique: {duplicates}")
    for definition in document.exports:
        resolved = resolve_definition(definition)
        if resolved.references.get("connection") is not None:
            resolved.plugin.check_delivery_target(resolved.references)
    task_ids = [task.id for task in document.tasks]
    duplicates = sorted({item for item in task_ids if task_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Task ids must be unique: {duplicates}")
    _validate_structure(document.tasks)
    for task in document.tasks:
        check_target(task)
    validate_automation(compile_tasks(document.tasks), exports={str(item["id"]): item for item in document.exports})


def apply(document: OperationalConfiguration) -> None:
    """Replace every task and export with the document's, after validating all of it."""
    from open_climate_service.exports import store as export_store
    from open_climate_service.scheduler.service import get_scheduler_service
    from open_climate_service.tasks import store as task_store

    validate(document)
    export_store.replace_definitions(document.exports)
    task_store.replace_tasks(document.tasks)
    get_scheduler_service().reload()


def _wants_yaml(request: Request) -> bool:
    return request.query_params.get("format") == "yaml" or "yaml" in request.headers.get("accept", "")


@router.get("")
def get_configuration(request: Request) -> Response:
    """Every task and export, as JSON or (``?format=yaml``) YAML, ready to commit."""
    document = current().model_dump(mode="json", exclude={"tasks": {"__all__": {"created_at", "updated_at"}}})
    if _wants_yaml(request):
        return Response(yaml.safe_dump(document, sort_keys=False), media_type="application/yaml")
    return JSONResponse(document)


@router.put("")
async def put_configuration(request: Request) -> Response:
    """Replace every task and export with a document from ``GET /configuration``. JSON or YAML."""
    from open_climate_service import config as api_config

    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; its configuration cannot change")
    raw = await request.body()
    try:
        body = yaml.safe_load(raw) if "yaml" in request.headers.get("content-type", "") else json.loads(raw)
        document = OperationalConfiguration.model_validate(body)
    except (ValueError, ValidationError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        apply(document)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"tasks": len(document.tasks), "exports": len(document.exports)})
