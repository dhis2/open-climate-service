"""Pipelines: pages and JSON for creating, editing, validating, dry-running, running and deleting one."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from open_climate_service import config as api_config
from open_climate_service.pipelines import forms, runner, service, store
from open_climate_service.pipelines.schemas import PipelineRecord, PipelineSpec, ValidationResult
from open_climate_service.shared.time import utc_now
from open_climate_service.shared.urls import mount_prefix
from open_climate_service.system.templates import wants_json

router = APIRouter()

_DEFAULT_WORKFLOW = "aggregate_by_features"


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
    public = record.model_dump(mode="json")
    for run in public.get("runs", []):
        run.pop("process", None)
    return {**public, "approved": service.approved(record)}


def _failures(validation: ValidationResult) -> str:
    return "; ".join(f"{check.id}: {check.message}" for check in validation.checks if check.status == "fail")


def _form_selection(choices: dict[str, Any], workflow_id: str | None, plugin_id: str | None) -> dict[str, Any]:
    """The workflow and destination the form is for, with the fields each one asks for."""
    workflows = {item["id"]: item for item in choices.get("workflows", [])}
    destinations = {item["id"]: item for item in choices.get("destinations", [])}
    workflow = (
        workflows.get(workflow_id or "") or workflows.get(_DEFAULT_WORKFLOW) or next(iter(workflows.values()), None)
    )
    destination = destinations.get(plugin_id or "") or next(iter(destinations.values()), None)
    settings_form = (
        forms.settings_form(destination["schema"], destination.get("settings_form"), plugin_id=destination["id"])
        if destination
        else None
    )
    return {
        "workflow": workflow,
        "destination": destination,
        "parameter_fields": forms.parameter_fields(workflow["parameters"]) if workflow else [],
        "settings_fields": [
            field for section in (settings_form or {}).get("sections", []) for field in section["fields"]
        ],
        "settings_form": settings_form,
    }


async def _body(request: Request) -> tuple[dict[str, Any], int | None]:
    """A JSON body as is, or a form turned into the pipeline shape; with the revision it edits."""
    if _json_request(request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Pipeline body must be an object")
        revision = body.pop("revision", None)
        return body, revision if isinstance(revision, int) and not isinstance(revision, bool) else None
    form = await request.form()
    field = {key: str(value) for key, value in form.items() if isinstance(value, str)}
    choices = service.choices()
    selection = _form_selection(choices, field.get("workflow"), field.get("destination"))
    workflow_id = field.get("workflow") or (selection["workflow"] or {}).get("id", "")
    try:
        arguments = forms.read_arguments(field, selection["parameter_fields"])
        settings = forms.read_settings(field, selection["settings_fields"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    revision = field.get("revision", "").strip()
    return (
        {
            "id": field.get("id", "").strip(),
            "name": field.get("name", "").strip() or None,
            "workflow": {"id": workflow_id, "arguments": arguments},
            "destination": {"plugin": field.get("destination", ""), "settings": settings},
            "delivery": {"mode": field.get("mode", "dry_run"), "policy": field.get("policy", "on_update")},
        },
        int(revision) if revision.isdigit() else None,
    )


def _render_page(
    request: Request,
    tab: Literal["list", "create"],
    *,
    error: str | None = None,
    checks: ValidationResult | None = None,
    draft: dict[str, Any] | None = None,
    editing: str | None = None,
    revision: int | None = None,
    status_code: int | None = None,
) -> HTMLResponse:
    from open_climate_service.system.templates import render_pipelines_page

    choices = service.choices() if tab == "create" else {}
    draft = draft or {}
    params = request.query_params
    workflow_id = params.get("workflow") or (draft.get("workflow") or {}).get("id")
    plugin_id = params.get("destination") or (draft.get("destination") or {}).get("plugin")
    if params.get("workflow") and workflow_id != (draft.get("workflow") or {}).get("id"):
        draft = {**draft, "workflow": {"id": workflow_id, "arguments": {}}}
    if params.get("destination") and plugin_id != (draft.get("destination") or {}).get("plugin"):
        draft = {**draft, "destination": {"plugin": plugin_id, "settings": {}}}
    if params.get("dataset") and not draft.get("workflow"):
        draft = {**draft, "prefill_dataset": params["dataset"]}
    return HTMLResponse(
        render_pipelines_page(
            store.list_records(),
            choices,
            mount_prefix(request),
            tab=tab,
            error=error,
            checks=checks,
            draft=draft,
            editing=editing,
            revision=revision,
            selection=_form_selection(choices, workflow_id, plugin_id) if tab == "create" else {},
            read_only=api_config.is_read_only(),
        ),
        status_code=status_code or (400 if error else 200),
    )


def _spec_from(
    body: dict[str, Any], request: Request, *, editing: str | None, revision: int | None
) -> PipelineSpec | HTMLResponse:
    """Parse a spec, or the page that shows why it did not parse."""
    try:
        return PipelineSpec.model_validate(body)
    except ValidationError as exc:
        message = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        if _json_request(request):
            raise HTTPException(status_code=422, detail=message) from exc
        return _render_page(request, "create", error=message, draft=body, editing=editing, revision=revision)


def _done(request: Request, location: str, content: Any, status_code: int = 200, *, by_body: bool = False) -> Response:
    """JSON for an API caller, a redirect for the page. ``by_body`` decides by the request body alone."""
    if _json_request(request) or (not by_body and wants_json(request)):
        return JSONResponse(status_code=status_code, content=content)
    return RedirectResponse(f"{mount_prefix(request)}{location}", status_code=303)


@router.get("", response_class=Response)
def pipelines(request: Request) -> Response:
    """The pipelines page, or the list as JSON."""
    if wants_json(request):
        return JSONResponse({"kind": "PipelineList", "items": [_public(record) for record in store.list_records()]})
    return _render_page(request, "list")


@router.get("/new", response_class=HTMLResponse, include_in_schema=False)
def new_pipeline(request: Request) -> HTMLResponse:
    """The creation tab; ``?workflow=``, ``?destination=`` and ``?dataset=`` preselect its inputs."""
    return _render_page(request, "create")


@router.post("", response_class=Response)
async def create_pipeline(request: Request) -> Response:
    """Save a new pipeline. It is validated first and refused when any check fails."""
    _require_writable()
    body, _ = await _body(request)
    parsed = _spec_from(body, request, editing=None, revision=None)
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
    return _done(request, f"/pipelines/{record.spec.id}", _public(record), status_code=201, by_body=True)


@router.get("/{pipeline_id}", response_class=Response)
def pipeline(request: Request, pipeline_id: str) -> Response:
    """Read one stored pipeline without revalidating it or changing its record."""
    record = _record_or_404(pipeline_id)
    if wants_json(request):
        return JSONResponse({**_public(record), "runs": runner.run_views(record)})
    from open_climate_service.system.templates import render_pipeline_page

    return HTMLResponse(render_pipeline_page(record, runner.run_views(record), mount_prefix(request)))


@router.get("/{pipeline_id}/edit", response_class=HTMLResponse, include_in_schema=False)
def edit_pipeline(request: Request, pipeline_id: str) -> HTMLResponse:
    """The create form, prefilled with a stored pipeline."""
    record = _record_or_404(pipeline_id)
    return _render_page(
        request, "create", draft=record.spec.model_dump(mode="json"), editing=pipeline_id, revision=record.revision
    )


@router.post("/{pipeline_id}", response_class=Response)
async def save_pipeline(request: Request, pipeline_id: str) -> Response:
    """Save an edited pipeline. Validated first; a change that fails any check, or a stale edit, is refused."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    body, revision = await _body(request)
    body["id"] = pipeline_id
    parsed = _spec_from(body, request, editing=pipeline_id, revision=revision)
    if isinstance(parsed, HTMLResponse):
        return parsed
    candidate = record.model_copy(deep=True)
    candidate.spec = parsed
    validation = service.validate_pipeline(candidate)
    if not validation.valid:
        if _json_request(request):
            raise HTTPException(
                status_code=422,
                detail={"message": "Pipeline validation failed", "checks": validation.model_dump(mode="json")},
            )
        return _render_page(
            request,
            "create",
            error="The changes were not saved",
            checks=validation,
            draft=body,
            editing=pipeline_id,
            revision=revision,
        )

    def _apply(current: PipelineRecord) -> None:
        judged = current.revision if revision is None else revision
        # The validation judged `parsed` itself; it becomes the verdict on the revision `parsed` creates.
        store.replace_spec(current, parsed, expected_revision=judged)
        current.validation = validation.model_copy(update={"revision": current.revision})
        # The approval names the bindings it judged; it lapses by itself when they change,
        # and is dropped so the page does not show a dry run for something else.
        if current.dry_run is not None and current.dry_run.bindings != validation.bindings:
            current.dry_run = None

    try:
        saved, _ = store.update_record(pipeline_id, _apply)
    except store.StaleRevision as exc:
        if _json_request(request):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _render_page(request, "create", error=str(exc), draft=body, editing=pipeline_id, status_code=409)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _done(request, f"/pipelines/{pipeline_id}", _public(saved), by_body=True)


@router.delete("/{pipeline_id}", status_code=204)
def delete_pipeline(pipeline_id: str) -> Response:
    """Remove a stored pipeline and its run history. Jobs and deliveries already made are kept."""
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
    """Re-check every binding the pipeline relies on, in OCS and at its destination, and keep the result."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    validation = service.validate_pipeline(record)

    def _apply(current: PipelineRecord) -> None:
        store.install_validation(current, validation, judged_revision=record.revision)

    try:
        store.update_record(pipeline_id, _apply)
    except store.StaleRevision as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _done(request, f"/pipelines/{pipeline_id}#validation", validation.model_dump(mode="json"))


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
    """Run the workflow over a bounded range and dry-run the result at the destination; nothing is stored there."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    start, end, _ = await _range(request)
    result = await run_in_threadpool(service.dry_run_pipeline, record, start, end)

    def _apply(current: PipelineRecord) -> None:
        if current.revision != record.revision:
            raise store.StaleRevision(f"pipeline '{pipeline_id}' changed during the dry run; run it again")
        current.dry_run = result

    try:
        store.update_record(pipeline_id, _apply)
    except store.StaleRevision as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _done(request, f"/pipelines/{pipeline_id}#dry-run", json.loads(result.model_dump_json()))


def _start(pipeline_id: str, context: runner.RunContext) -> Any:
    try:
        return runner.run_pipeline(pipeline_id, context)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{pipeline_id}/runs", response_class=Response)
async def run_pipeline(request: Request, pipeline_id: str) -> Response:
    """Run once over an explicit range: a batch job of the workflow, delivered when it finishes."""
    _require_writable()
    _record_or_404(pipeline_id)
    start, end, body = await _range(request)
    mode = str(body.get("mode", "dry_run"))
    if mode not in ("dry_run", "live"):
        raise HTTPException(status_code=400, detail="mode must be dry_run or live")
    run = _start(pipeline_id, runner.RunContext(start=start, end=end, mode="live" if mode == "live" else "dry_run"))
    return _done(request, f"/pipelines/{pipeline_id}#runs", run.model_dump(mode="json", exclude={"process"}), 202)


@router.post("/{pipeline_id}/runs/{run_id}/retry", response_class=Response)
def retry_pipeline_run(request: Request, pipeline_id: str, run_id: str) -> Response:
    """Run a failed or unwanted run's interval again, in the same mode, as a new run."""
    _require_writable()
    record = _record_or_404(pipeline_id)
    previous = next((item for item in record.runs if run_id in (item.run_id, item.job_id)), None)
    if previous is None:
        raise HTTPException(status_code=404, detail=f"pipeline '{pipeline_id}' has no run '{run_id}'")
    context = runner.RunContext(
        start=previous.start, end=previous.end, trigger="retry", mode=previous.mode, retry_of=previous.run_id
    )
    run = _start(pipeline_id, context)
    return _done(request, f"/pipelines/{pipeline_id}#runs", run.model_dump(mode="json", exclude={"process"}), 202)


@router.post("/{pipeline_id}/runs/{run_id}/deliver", response_class=Response)
def deliver_pipeline_run(request: Request, pipeline_id: str, run_id: str) -> Response:
    """Submit the delivery a finished run still owes: a missed hand-off, a held one or a failed one."""
    _require_writable()
    _record_or_404(pipeline_id)
    try:
        run = runner.deliver_run(pipeline_id, run_id, operator=True)
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        # A transient failure while handing off; the run records it and nothing was sent twice.
        raise HTTPException(status_code=503, detail=f"Delivery could not be submitted now; try again: {exc}") from exc
    return _done(request, f"/pipelines/{pipeline_id}#runs", run.model_dump(mode="json", exclude={"process"}))


@router.post("/{pipeline_id}/runs/{run_id}/redeliver", response_class=Response)
def redeliver_pipeline_run(request: Request, pipeline_id: str, run_id: str) -> Response:
    """Send a run again whose delivery job failed at the destination, as a new delivery."""
    _require_writable()
    _record_or_404(pipeline_id)
    try:
        run = runner.deliver_run(pipeline_id, run_id, operator=True, redeliver=True)
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        # A transient failure while handing off; the run records it and nothing was sent twice.
        raise HTTPException(status_code=503, detail=f"Delivery could not be submitted now; try again: {exc}") from exc
    return _done(request, f"/pipelines/{pipeline_id}#runs", run.model_dump(mode="json", exclude={"process"}))


@router.post("/{pipeline_id}/runs/{run_id}/acknowledge", response_class=Response)
def acknowledge_pipeline_delivery(request: Request, pipeline_id: str, run_id: str) -> Response:
    """Record that the destination was checked and an unconfirmed delivery is settled; nothing is sent."""
    _require_writable()
    _record_or_404(pipeline_id)
    try:
        run = runner.acknowledge_delivery(pipeline_id, run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _done(request, f"/pipelines/{pipeline_id}#runs", run.model_dump(mode="json", exclude={"process"}))


@router.post("/{pipeline_id}/mode/{mode}", response_class=Response)
def set_delivery_mode(request: Request, pipeline_id: str, mode: Literal["dry_run", "live", "paused"]) -> Response:
    """Switch delivery between dry run, live and paused.

    Going live re-runs every check, the destination's included, and needs a dry run that
    passed for the bindings being validated.
    """
    _require_writable()
    record = _record_or_404(pipeline_id)
    candidate = record.model_copy(deep=True)
    candidate.spec = candidate.spec.model_copy(
        update={"delivery": candidate.spec.delivery.model_copy(update={"mode": mode})}
    )
    validation = service.validate_pipeline(candidate)
    if not validation.valid:
        raise HTTPException(status_code=409, detail=f"Pipeline validation failed: {_failures(validation)}")
    if mode == "live" and not (candidate.dry_run and candidate.dry_run.bindings == validation.bindings):
        raise HTTPException(
            status_code=409, detail="Live delivery requires a successful dry run of the current bindings"
        )

    def _apply(current: PipelineRecord) -> None:
        store.replace_spec(current, candidate.spec, expected_revision=record.revision)
        current.validation = validation.model_copy(update={"revision": current.revision})

    try:
        saved, _ = store.update_record(pipeline_id, _apply)
    except store.StaleRevision as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if mode != "paused":
        # Resuming, or going live, releases the held deliveries that are eligible now.
        runner.release_held(pipeline_id)
        saved = store.get_record(pipeline_id) or saved
    return _done(request, f"/pipelines/{pipeline_id}#delivery", _public(saved))
