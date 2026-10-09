"""Pipelines: pages and JSON for creating, editing, validating, dry-running, running and deleting one."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.pipelines import service, store
from open_climate_service.pipelines.schemas import PipelineRecord, PipelineSpec, ValidationResult
from open_climate_service.shared.time import utc_now
from open_climate_service.shared.urls import mount_prefix
from open_climate_service.system.templates import wants_json

router = APIRouter()


def _require_writable() -> None:
    if api_config.is_read_only():
        raise HTTPException(status_code=403, detail="This instance is read-only; pipelines cannot be changed or run")


def _record_or_404(pipeline_id: str) -> PipelineRecord:
    record = store.get_record(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline '{pipeline_id}'")
    return record


def _json_request(request: Request) -> bool:
    return "application/json" in request.headers.get("content-type", "")


def _public(record: PipelineRecord) -> dict[str, Any]:
    period = record.validation.period_type if record.validation else None
    return {**record.model_dump(mode="json"), "compiled": service.compile_pipeline(record.spec, period)}


def _failures(validation: ValidationResult) -> str:
    return "; ".join(f"{check.id}: {check.message}" for check in validation.checks if check.status == "fail")


async def _body(request: Request) -> dict[str, Any]:
    """A JSON body as is, or a form turned into the pipeline shape."""
    if _json_request(request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Pipeline body must be an object")
        return body
    form = await request.form()
    field = {key: str(value).strip() for key, value in form.items() if isinstance(value, str)}
    series: dict[str, Any] = {"data_element": field.get("data_element", "")}
    if field.get("variable"):
        series["variable"] = field["variable"]
    return {
        "id": field.get("id", ""),
        "name": field.get("name") or None,
        "source": {"dataset": field.get("dataset", "")},
        "destination": {
            "connection": field.get("connection", ""),
            "data_set": field.get("data_set") or None,
            "organisation_units": {"feature_collection": field.get("feature_collection", "")},
            "series": [series],
        },
        "aggregation": {"spatial": {"reducer": field.get("reducer", "mean")}},
        "delivery": {
            "mode": field.get("mode", "dry_run"),
            "policy": field.get("policy", "on_update"),
            "release_cron": field.get("release_cron") or None,
            "range": field.get("range", "updated"),
        },
    }


def _render_page(
    request: Request,
    tab: Literal["list", "create"],
    *,
    error: str | None = None,
    checks: ValidationResult | None = None,
    draft: dict[str, Any] | None = None,
    editing: str | None = None,
) -> HTMLResponse:
    from open_climate_service.system.templates import render_pipelines_page

    return HTMLResponse(
        render_pipelines_page(
            store.list_records(),
            service.choices() if tab == "create" else {},
            mount_prefix(request),
            tab=tab,
            error=error,
            checks=checks,
            draft=draft,
            editing=editing,
        ),
        status_code=400 if error else 200,
    )


def _spec_from(body: dict[str, Any], request: Request, *, editing: str | None) -> PipelineSpec | HTMLResponse:
    """Parse a spec, or the page that shows why it did not parse."""
    try:
        return PipelineSpec.model_validate(body)
    except ValidationError as exc:
        message = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        if _json_request(request):
            raise HTTPException(status_code=422, detail=message) from exc
        return _render_page(request, "create", error=message, draft=body, editing=editing)


@router.get("", response_class=Response)
def pipelines(request: Request) -> Response:
    """The pipelines page, or the list as JSON."""
    if wants_json(request):
        return JSONResponse({"kind": "PipelineList", "items": [_public(record) for record in store.list_records()]})
    return _render_page(request, "list")


@router.get("/new", response_class=HTMLResponse, include_in_schema=False)
def new_pipeline(request: Request) -> HTMLResponse:
    """The pipeline creation tab."""
    return _render_page(request, "create")


@router.post("", response_class=Response)
async def create_pipeline(request: Request) -> Response:
    """Save a new pipeline. It is validated first and refused when any check fails."""
    _require_writable()
    body = await _body(request)
    parsed = _spec_from(body, request, editing=None)
    if isinstance(parsed, HTMLResponse):
        return parsed
    if store.get_record(parsed.id) is not None:
        if _json_request(request):
            raise HTTPException(status_code=409, detail=f"A pipeline with id '{parsed.id}' already exists")
        return _render_page(request, "create", error=f"A pipeline with id '{parsed.id}' already exists", draft=body)
    candidate = PipelineRecord(spec=parsed, created_at=utc_now().isoformat())
    validation = service.validate_pipeline(candidate)
    if not validation.valid:
        if _json_request(request):
            raise HTTPException(
                status_code=422,
                detail={"message": "Pipeline validation failed", "checks": validation.model_dump(mode="json")},
            )
        return _render_page(request, "create", error="The pipeline was not saved", checks=validation, draft=body)
    try:
        record = store.create_record(parsed, validation)
    except ValueError as exc:
        if _json_request(request):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _render_page(request, "create", error=str(exc), draft=body)
    if _json_request(request):
        return JSONResponse(status_code=201, content=_public(record))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{record.spec.id}", status_code=303)


@router.get("/{pipeline_id}", response_class=Response)
def pipeline(request: Request, pipeline_id: str) -> Response:
    """Read one stored pipeline without revalidating it or changing its record."""
    record = _record_or_404(pipeline_id)
    if wants_json(request):
        return JSONResponse({**_public(record), "runs": service.run_views(record)})
    from open_climate_service.system.templates import render_pipeline_page

    period = record.validation.period_type if record.validation else None
    return HTMLResponse(
        render_pipeline_page(
            record, service.compiled_yaml(record.spec, period), service.run_views(record), mount_prefix(request)
        )
    )


@router.get("/{pipeline_id}/edit", response_class=HTMLResponse, include_in_schema=False)
def edit_pipeline(request: Request, pipeline_id: str) -> HTMLResponse:
    """The create form, prefilled with a stored pipeline."""
    record = _record_or_404(pipeline_id)
    return _render_page(request, "create", draft=record.spec.model_dump(mode="json"), editing=pipeline_id)


@router.post("/{pipeline_id}", response_class=Response)
async def save_pipeline(request: Request, pipeline_id: str) -> Response:
    """Save an edited pipeline. Validated first; a change that fails any check is refused."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    body = await _body(request)
    body["id"] = pipeline_id
    parsed = _spec_from(body, request, editing=pipeline_id)
    if isinstance(parsed, HTMLResponse):
        return parsed
    candidate = record.model_copy(deep=True)
    changed = parsed != record.spec
    candidate.spec = parsed
    if changed:
        # The bindings a dry run was judged on no longer hold; runs stay as history.
        candidate.dry_run = None
    candidate.validation = service.validate_pipeline(candidate)
    if not candidate.validation.valid:
        if _json_request(request):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Pipeline validation failed",
                    "checks": candidate.validation.model_dump(mode="json"),
                },
            )
        return _render_page(
            request,
            "create",
            error="The changes were not saved",
            checks=candidate.validation,
            draft=body,
            editing=pipeline_id,
        )
    store.save_record(candidate)
    if _json_request(request):
        return JSONResponse(_public(candidate))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}", status_code=303)


@router.delete("/{pipeline_id}", status_code=204)
def delete_pipeline(pipeline_id: str) -> Response:
    """Remove a stored pipeline. Configuration already merged into the instance file is untouched."""
    _require_writable()
    _record_or_404(pipeline_id)
    store.delete_record(pipeline_id)
    return Response(status_code=204)


@router.post("/{pipeline_id}/delete", response_class=Response, include_in_schema=False)
async def delete_pipeline_form(request: Request, pipeline_id: str) -> Response:
    """The page's delete action; needs the confirmation box ticked."""
    _require_writable()
    _record_or_404(pipeline_id)
    form = await request.form()
    if str(form.get("confirm", "")).strip().lower() not in {"yes", "on", "true"}:
        raise HTTPException(status_code=400, detail="Tick the confirmation to delete the pipeline")
    store.delete_record(pipeline_id)
    return RedirectResponse(f"{mount_prefix(request)}/pipelines", status_code=303)


@router.post("/{pipeline_id}/validate", response_class=Response)
async def validate_pipeline(request: Request, pipeline_id: str) -> Response:
    """Re-check every binding the pipeline relies on, in OCS and in DHIS2, and keep the result."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    record.validation = service.validate_pipeline(record)
    store.save_record(record)
    if wants_json(request) or _json_request(request):
        return JSONResponse(record.validation.model_dump(mode="json"))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}#validation", status_code=303)


async def _range(request: Request) -> tuple[str, str, dict[str, Any]]:
    if _json_request(request):
        body = await request.json()
        body = body if isinstance(body, dict) else {}
    else:
        form = await request.form()
        body = {key: str(value).strip() for key, value in form.items() if isinstance(value, str)}
    start, end = str(body.get("start", "")), str(body.get("end", ""))
    if not start or not end:
        raise HTTPException(status_code=400, detail="A run needs a start and an end date")
    return start, end, body


@router.post("/{pipeline_id}/dry-run", response_class=Response)
async def dry_run_pipeline(request: Request, pipeline_id: str) -> Response:
    """Aggregate a bounded range and submit it to DHIS2 as a dry run; nothing is stored there."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    start, end, _ = await _range(request)
    record.dry_run = service.dry_run_pipeline(record, start, end)
    store.save_record(record)
    if wants_json(request) or _json_request(request):
        return JSONResponse(json.loads(record.dry_run.model_dump_json()))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}#dry-run", status_code=303)


@router.post("/{pipeline_id}/runs", response_class=Response)
async def run_pipeline(request: Request, pipeline_id: str) -> Response:
    """Run once: a batch job over the range through the pipeline's export, delivered when it finishes."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    start, end, body = await _range(request)
    mode = str(body.get("mode", "dry_run"))
    try:
        run = service.start_run(record, start, end, mode)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    store.save_record(record)
    if wants_json(request) or _json_request(request):
        return JSONResponse(status_code=202, content=run.model_dump(mode="json"))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}#runs", status_code=303)


@router.post("/{pipeline_id}/runs/{job_id}/deliver", response_class=Response)
def deliver_pipeline_run(request: Request, pipeline_id: str, job_id: str) -> Response:
    """Submit the delivery a finished run still owes, when the automatic hand-off did not."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    try:
        run = service.deliver_run(record, job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.save_record(record)
    if wants_json(request) or _json_request(request):
        return JSONResponse(run.model_dump(mode="json"))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}#runs", status_code=303)


@router.post("/{pipeline_id}/mode/{mode}", response_class=Response)
def set_delivery_mode(request: Request, pipeline_id: str, mode: Literal["dry_run", "live", "paused"]) -> Response:
    """Switch delivery between dry run, live and paused; live needs a passed dry run."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    if mode == "live" and (record.dry_run is None or not record.dry_run.passed):
        raise HTTPException(status_code=409, detail="Live delivery requires a successful dry run")
    candidate = record.model_copy(deep=True)
    candidate.spec = candidate.spec.model_copy(
        update={"delivery": candidate.spec.delivery.model_copy(update={"mode": mode})}
    )
    candidate.validation = service.validate_pipeline(candidate)
    if not candidate.validation.valid:
        raise HTTPException(status_code=409, detail=f"Pipeline validation failed: {_failures(candidate.validation)}")
    store.save_record(candidate)
    if wants_json(request) or _json_request(request):
        return JSONResponse(_public(candidate))
    return RedirectResponse(f"{mount_prefix(request)}/pipelines/{pipeline_id}#config", status_code=303)
