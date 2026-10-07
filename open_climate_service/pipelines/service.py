"""Validate, compile and dry-run a pipeline against the instance and its DHIS2 destination.

Nothing here activates a pipeline. `compile_pipeline` returns the schedule, trigger and
export an operator would put in the instance configuration; `validate_pipeline` checks
every binding that would make that configuration fail or deliver the wrong thing;
`dry_run_pipeline` aggregates a bounded range and submits it to DHIS2 as a dry run.
"""

from __future__ import annotations

import json
import logging
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from open_climate_service import config as api_config
from open_climate_service.pipelines.schemas import (
    Check,
    DryRunResult,
    PipelineRecord,
    PipelineRun,
    PipelineSpec,
    ValidationResult,
)
from open_climate_service.shared.time import normalise_export_period, utc_now

logger = logging.getLogger(__name__)

_DHIS2_PERIOD_TYPE = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
}
_POLYGON_TYPES = {"Polygon", "MultiPolygon"}


@dataclass(frozen=True)
class FeatureBinding:
    """Stored geometry plus the DHIS2 template parameters that produced it."""

    ids: list[str]
    geometry_types: list[str]
    connection: str | None
    level: int | None


def dataset_record(dataset_id: str) -> Any:
    """The managed dataset list entry, or None."""
    from open_climate_service.ingestions.services import list_datasets

    for item in list_datasets().items:
        if item.dataset_id == dataset_id:
            return item
    return None


def export_period_for(dataset: Any) -> str | None:
    """The DHIS2 period a dataset exports as, by pass-through: its own cadence, or None."""
    return normalise_export_period(getattr(dataset, "period_type", None))


def connections() -> dict[str, Any]:
    """The named DHIS2 connections the instance configures, by id."""
    from open_climate_service.exports.dhis2_config import parse_connections

    return parse_connections(api_config.get_config().get("dhis2_connections", []))


def choices() -> dict[str, Any]:
    """What the create form offers: exportable published datasets, collections, connections."""
    from open_climate_service.features.services import list_feature_collections
    from open_climate_service.ingestions.services import list_datasets

    datasets = [
        {
            "id": item.dataset_id,
            "name": item.dataset_name,
            "variable": item.variable,
            "period_type": export_period_for(item),
            "cadence": item.period_type,
            "coverage": item.extent.temporal,
        }
        for item in list_datasets().items
        if item.publication.status == "published" and item.item_type == "coverage" and export_period_for(item)
    ]
    from open_climate_service.features import templates as feature_templates

    collections = []
    for record in list_feature_collections().items:
        if not record.geometry_types or not set(record.geometry_types).issubset(_POLYGON_TYPES):
            continue
        template = feature_templates.get_feature_template(record.id) or {}
        raw_params = template.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        collections.append(
            {
                "id": record.id,
                "name": record.name,
                "feature_count": record.feature_count,
                "geometry_types": record.geometry_types,
                "connection": params.get("connection") if template.get("provider") == "dhis2" else None,
                "level": params.get("level") if template.get("provider") == "dhis2" else None,
            }
        )
    return {"datasets": datasets, "collections": collections, "connections": sorted(connections())}


# --- validation ------------------------------------------------------------------------------


def validate_pipeline(record: PipelineRecord) -> ValidationResult:
    """Check every binding a pipeline relies on, in the order a run would hit them."""
    spec = record.spec
    checks: list[Check] = []
    period_type: str | None = None
    feature_ids: list[str] = []

    dataset = dataset_record(spec.source.dataset)
    if dataset is None:
        checks.append(Check(id="dataset", status="fail", message=f"'{spec.source.dataset}' is not a managed dataset"))
    elif dataset.publication.status != "published":
        checks.append(Check(id="dataset", status="fail", message=f"'{spec.source.dataset}' is not published"))
    else:
        coverage = dataset.extent.temporal
        checks.append(
            Check(
                id="dataset",
                status="pass",
                message=f"published, {coverage.start} to {coverage.end}, variable '{dataset.variable}'",
            )
        )
        period_type = export_period_for(dataset)
        if period_type is None:
            checks.append(
                Check(
                    id="cadence",
                    status="fail",
                    message=(
                        f"the dataset is {dataset.period_type}, which is not a DHIS2 period; this pipeline does not "
                        "resample in time, so choose a daily, weekly, monthly or yearly dataset"
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    id="cadence",
                    status="pass",
                    message=f"{dataset.period_type} data exports as {period_type} periods as is",
                )
            )
        for series in spec.destination.series:
            if series.variable is not None and series.variable != dataset.variable:
                checks.append(
                    Check(
                        id="series",
                        status="fail",
                        message=f"variable '{series.variable}' is not the dataset's '{dataset.variable}'",
                    )
                )
        if not any(check.id == "series" for check in checks):
            elements = ", ".join(series.data_element for series in spec.destination.series)
            checks.append(
                Check(id="series", status="pass", message=f"data element {elements} receives '{dataset.variable}'")
            )

    try:
        binding = _feature_binding(spec.destination.organisation_units.feature_collection)
        feature_ids = binding.ids
        if not binding.geometry_types or not set(binding.geometry_types).issubset(_POLYGON_TYPES):
            raise ValueError(
                f"zonal aggregation requires Polygon or MultiPolygon geometry; found {binding.geometry_types or 'none'}"
            )
        if binding.connection is None:
            raise ValueError(
                "the collection was not fetched from DHIS2, so its feature ids cannot be verified as "
                "organisation unit UIDs"
            )
        if binding.connection != spec.destination.connection:
            raise ValueError(
                f"the collection was declared for DHIS2 connection {binding.connection!r}, but the destination uses "
                f"{spec.destination.connection!r}"
            )
        level = f" at level {binding.level}" if binding.level is not None else ""
        checks.append(
            Check(
                id="organisation_units",
                status="pass",
                message=f"{len(feature_ids)} polygon features{level} from connection {binding.connection!r}",
            )
        )
    except LookupError as exc:
        checks.append(Check(id="organisation_units", status="fail", message=str(exc)))
    except ValueError as exc:
        checks.append(
            Check(
                id="organisation_units",
                status="fail",
                message=f"the feature collection cannot be used for this DHIS2 pipeline: {exc}",
            )
        )

    if spec.destination.connection in connections():
        checks.append(Check(id="connection", status="pass", message=f"'{spec.destination.connection}' is configured"))
    else:
        checks.append(
            Check(
                id="connection",
                status="fail",
                message=f"'{spec.destination.connection}' is not a configured DHIS2 connection",
            )
        )

    checks.append(
        Check(
            id="spatial_reducer",
            status="pass",
            message=f"{spec.aggregation.spatial.reducer} within each organisation unit",
        )
    )

    checks.append(_check_dhis2_metadata(spec, period_type, feature_ids))
    checks.append(_check_configuration_bindings(spec, period_type))

    checks.append(_check_schedule(spec))
    checks.append(_check_delivery(record))

    return ValidationResult(
        valid=not any(check.status == "fail" for check in checks),
        checked_at=utc_now().isoformat(),
        period_type=period_type,
        checks=checks,
    )


def _feature_binding(collection_id: str) -> FeatureBinding:
    """Resolve stored polygon identity and its declared DHIS2 source."""
    from open_climate_service.features import templates as feature_templates
    from open_climate_service.features.services import get_feature_collection_or_404, registered_collections
    from open_climate_service.plugins.processes.load_features import load_features
    from open_climate_service.shared.features import validate_dhis2_feature_ids

    if collection_id not in registered_collections():
        raise LookupError(f"'{collection_id}' is not a registered feature collection; fetch it first")
    record = get_feature_collection_or_404(collection_id)
    template = feature_templates.get_feature_template(collection_id) or {}
    if template.get("provider") != "dhis2":
        raise ValueError(
            f"the collection was not fetched from DHIS2 (provider {template.get('provider')!r}), so its ids are not "
            "organisation unit UIDs; choose a collection fetched through the DHIS2 provider"
        )
    raw_params = template.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    connection = params.get("connection")
    level = params.get("level")
    return FeatureBinding(
        ids=validate_dhis2_feature_ids(load_features(collection_id)),
        geometry_types=record.geometry_types,
        connection=connection if isinstance(connection, str) else None,
        level=level if isinstance(level, int) and not isinstance(level, bool) else None,
    )


def _check_dhis2_metadata(spec: PipelineSpec, period_type: str | None, feature_ids: list[str]) -> Check:
    """Ask DHIS2 whether the data set, elements and org units line up; skip when it cannot be asked."""
    data_set = spec.destination.data_set
    if data_set is None:
        return Check(
            id="dhis2_metadata",
            status="skip",
            message="no data set declared, so period type and assignment are not checked",
        )
    if spec.destination.connection not in connections():
        return Check(id="dhis2_metadata", status="skip", message="connection not configured")
    try:
        from open_climate_service.exports.dhis2 import get_connection

        with closing(get_connection(spec.destination.connection)) as client:
            metadata = client.get(
                f"/api/dataSets/{data_set}",
                params={"fields": "id,displayName,periodType,dataSetElements[dataElement[id]],organisationUnits[id]"},
            )
    except Exception as exc:  # the connection, its token, or DHIS2 itself
        return Check(id="dhis2_metadata", status="fail", message=f"could not read the data set from DHIS2: {exc}")
    problems: list[str] = []
    expected = _DHIS2_PERIOD_TYPE.get(period_type or "")
    if expected and metadata.get("periodType") != expected:
        problems.append(f"data set period type is {metadata.get('periodType')!r}, the export is {expected}")
    elements = {entry.get("dataElement", {}).get("id") for entry in metadata.get("dataSetElements", [])}
    missing = [series.data_element for series in spec.destination.series if series.data_element not in elements]
    if missing:
        problems.append(f"data element(s) not in the data set: {', '.join(missing)}")
    assigned = {entry.get("id") for entry in metadata.get("organisationUnits", [])}
    unassigned = [identifier for identifier in feature_ids if identifier not in assigned]
    if feature_ids and unassigned:
        sample = ", ".join(unassigned[:5]) + (" ..." if len(unassigned) > 5 else "")
        problems.append(
            f"{len(unassigned)} of {len(feature_ids)} organisation units are not assigned to the data set: {sample}"
        )
    if problems:
        return Check(id="dhis2_metadata", status="fail", message="; ".join(problems))
    return Check(
        id="dhis2_metadata",
        status="pass",
        message=(
            f"'{metadata.get('displayName', data_set)}' is {expected}, holds the element(s), "
            "every organisation unit assigned"
        ),
    )


def _check_configuration_bindings(spec: PipelineSpec, period_type: str | None) -> Check:
    """Refuse IDs already owned by a different configured export or trigger."""
    wanted_export = compile_export(spec, period_type)
    wanted_trigger = compile_trigger(spec)
    raw_exports = api_config.get_config().get("exports", [])
    raw_automation = api_config.get_config().get("automation", {})
    raw_triggers = raw_automation.get("workflow_triggers", []) if isinstance(raw_automation, dict) else []
    raw_exports = raw_exports if isinstance(raw_exports, list) else []
    raw_triggers = raw_triggers if isinstance(raw_triggers, list) else []
    existing_exports = [item for item in raw_exports if isinstance(item, dict) and item.get("id") == spec.id]
    existing_triggers = [item for item in raw_triggers if isinstance(item, dict) and item.get("id") == spec.id]
    if existing_exports and existing_exports != [wanted_export]:
        return Check(
            id="configuration", status="fail", message=f"export id {spec.id!r} is already configured differently"
        )
    if existing_triggers and not _same_pipeline_trigger(existing_triggers[0], wanted_trigger):
        return Check(
            id="configuration", status="fail", message=f"trigger id {spec.id!r} is already configured differently"
        )
    state = (
        "already configured for this pipeline"
        if existing_exports or existing_triggers
        else "export and trigger ids are available"
    )
    return Check(id="configuration", status="pass", message=state)


def _same_pipeline_trigger(existing: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Treat a dry-run/live switch as an update to the same pipeline binding."""
    existing = json.loads(json.dumps(existing))
    wanted = json.loads(json.dumps(wanted))
    for item in (existing, wanted):
        delivery = item.get("deliver")
        if isinstance(delivery, dict):
            delivery.pop("dry_run", None)
    return existing == wanted


def _check_schedule(spec: PipelineSpec) -> Check:
    """Report how the source dataset is kept current. The pipeline owns no sync clock of its own."""
    raw = api_config.get_config().get("scheduler", {})
    entries = raw.get("dataset_sync", []) if isinstance(raw, dict) else []
    existing = [item for item in entries if isinstance(item, dict) and item.get("dataset_id") == spec.source.dataset]
    if not existing:
        return Check(
            id="schedule",
            status="skip",
            message="no sync schedule for the source dataset; the pipeline runs after a manual sync or when run once",
        )
    cron = existing[0].get("cron", "?")
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        return Check(
            id="schedule",
            status="skip",
            message=f"the source dataset has a sync schedule ({cron!r}), but the instance scheduler is disabled",
        )
    return Check(id="schedule", status="pass", message=f"the source dataset is synced on schedule {cron!r}")


def _check_delivery(record: PipelineRecord) -> Check:
    """Whether and when values go out, and whether the pipeline has earned a live mode."""
    delivery = record.spec.delivery
    when = {
        "on_update": "after each dataset update",
        "manual": "only when run or delivered by hand",
        "scheduled": f"at {delivery.release_cron!r}, not applied yet: behaves as manual until then",
    }[delivery.policy]
    if delivery.mode == "paused":
        return Check(id="delivery", status="pass", message=f"paused; nothing is sent ({when})")
    if delivery.mode == "dry_run":
        return Check(id="delivery", status="pass", message=f"dry run {when}; DHIS2 validates and stores nothing")
    if record.dry_run is not None and record.dry_run.passed:
        return Check(
            id="delivery", status="pass", message=f"live {when}, after a dry run that passed on {record.dry_run.ran_at}"
        )
    return Check(id="delivery", status="fail", message="live delivery needs a dry run that passed first")


# --- compilation -----------------------------------------------------------------------------


def compile_pipeline(spec: PipelineSpec, period_type: str | None) -> dict[str, Any]:
    """The instance configuration this pipeline stands for.

    A named export and, when the pipeline delivers after each update, the trigger that runs it.
    Paused, manual and scheduled pipelines compile to the export alone.
    """
    compiled: dict[str, Any] = {"exports": [compile_export(spec, period_type)]}
    if spec.delivery.mode == "paused" or spec.delivery.policy != "on_update":
        return compiled
    compiled["automation"] = {"workflow_triggers": [compile_trigger(spec)]}
    return compiled


def compile_trigger(spec: PipelineSpec) -> dict[str, Any]:
    """The update trigger this pipeline would run, whatever its current mode and policy."""
    temporal_extent = (
        ["$event.previous_end", "$event.current_end"]
        if spec.delivery.range == "updated"
        else ["$event.current_start", "$event.current_end"]
    )
    trigger: dict[str, Any] = {
        "id": spec.id,
        "on_update_of": spec.source.dataset,
        "workflow_id": "aggregate_to_dhis2_json",
        "arguments": {
            "dataset_id": "$event.dataset_id",
            "temporal_extent": temporal_extent,
            "geometries": {"from_features": spec.destination.organisation_units.feature_collection},
            "export": spec.id,
            "method": spec.aggregation.spatial.reducer,
        },
    }
    trigger["deliver"] = {"export": spec.id, "dry_run": spec.delivery.mode != "live"}
    return trigger


def compile_export(spec: PipelineSpec, period_type: str | None) -> dict[str, Any]:
    """Compile the exact named export definition used by both dry runs and activation."""
    return {
        "id": spec.id,
        "plugin": "dhis2",
        "dataset": spec.source.dataset,
        "connection": spec.destination.connection,
        "aggregation": spec.aggregation.spatial.reducer,
        "period_type": period_type or "<dataset cadence>",
        "series": [
            {"select": {"variable": series.variable} if series.variable else {}, "data_element": series.data_element}
            for series in spec.destination.series
        ],
    }


def compiled_yaml(spec: PipelineSpec, period_type: str | None) -> str:
    """The compiled configuration as YAML, for the page and for pasting into the instance config."""
    return str(yaml.safe_dump(compile_pipeline(spec, period_type), sort_keys=False, allow_unicode=True))


# --- dry run ---------------------------------------------------------------------------------


def dry_run_pipeline(record: PipelineRecord, start: str, end: str) -> DryRunResult:
    """Render the pipeline's named export and send it through its plugin as a DHIS2 dry run."""
    spec = record.spec
    ran_at = utc_now().isoformat()
    try:
        resolved, rendered = _render_export(spec, start, end)
        target = resolved.references.get("connection")
        if target is None:
            raise ValueError("the named export has no DHIS2 connection")
        report_model = resolved.plugin.send(rendered.content, target, dry_run=True)
        payload = json.loads(rendered.content)
        report = report_model.model_dump(mode="json")
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        message = str(detail) if detail else str(exc) or type(exc).__name__
        logger.info("Pipeline '%s' dry run failed: %s", spec.id, message)
        return DryRunResult(ran_at=ran_at, start=start, end=end, error=message)
    values = payload["dataValues"]
    return DryRunResult(
        ran_at=ran_at,
        start=start,
        end=end,
        values=len(values),
        org_units=len({value["orgUnit"] for value in values}),
        periods=sorted({value["period"] for value in values}),
        sample=values[:5],
        report=report,
    )


def _render_export(spec: PipelineSpec, start: str, end: str) -> tuple[Any, Any]:
    """Aggregate and render the exact named-export definition generated for this draft."""
    from open_climate_service.exports.service import (
        render_resolved_export,
        resolve_export_definition,
    )
    from open_climate_service.openeo import execution
    from open_climate_service.openeo.execution import SaveResultEnvelope

    dataset = dataset_record(spec.source.dataset)
    if dataset is None:
        raise ValueError(f"'{spec.source.dataset}' is not a managed dataset")
    period_type = export_period_for(dataset)
    if period_type is None:
        raise ValueError(f"the dataset is {dataset.period_type}, which is not a DHIS2 period")
    graph = {
        "process_graph": {
            "features": {
                "process_id": "load_features",
                "arguments": {"id": spec.destination.organisation_units.feature_collection},
            },
            "load": {
                "process_id": "load_collection",
                "arguments": {"id": spec.source.dataset, "temporal_extent": [start, end]},
            },
            "zonal": {
                "process_id": "aggregate_spatial",
                "arguments": {
                    "data": {"from_node": "load"},
                    "geometries": {"from_node": "features"},
                    "reducer": {
                        "process_graph": {
                            "reduce": {
                                "process_id": "reduce_by_method",
                                "arguments": {
                                    "data": {"from_parameter": "data"},
                                    "method": spec.aggregation.spatial.reducer,
                                },
                                "result": True,
                            }
                        }
                    },
                },
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {"from_node": "zonal"},
                    # save_result captures execution provenance without resolving a process-wide
                    # export that is not active yet. The exact draft mapping is rendered below.
                    "format": "NetCDF",
                },
                "result": True,
            },
        }
    }
    envelope = execution.run_process_graph(graph)
    if not isinstance(envelope, SaveResultEnvelope):
        raise TypeError("pipeline graph did not return a save_result envelope")
    if envelope.provenance is None:
        raise ValueError("pipeline graph returned no execution provenance")
    resolved = resolve_export_definition(compile_export(spec, period_type))
    rendered = render_resolved_export(envelope.data, resolved, provenance=envelope.provenance)
    if rendered.record_count == 0:
        raise ValueError("the aggregation produced no values to submit")
    return resolved, rendered


# --- runs: a batch job through the export, then an audited delivery -------------------------

_RUNS_KEPT = 20


def _run_process(spec: PipelineSpec, start: str, end: str) -> dict[str, Any]:
    """The built-in org-unit workflow over the pipeline's bindings, saving through its export."""
    return {
        "process_graph": {
            "features": {
                "process_id": "load_features",
                "arguments": {"id": spec.destination.organisation_units.feature_collection},
            },
            "agg": {
                "process_id": "aggregate_to_dhis2_json",
                "arguments": {
                    "dataset_id": spec.source.dataset,
                    "temporal_extent": [start, end],
                    "geometries": {"from_node": "features"},
                    "export": spec.id,
                    "method": spec.aggregation.spatial.reducer,
                },
                "result": True,
            },
        }
    }


def start_run(record: PipelineRecord, start: str, end: str, mode: str) -> PipelineRun:
    """Submit a batch job for ``start`` to ``end`` through the pipeline's export, to be delivered in ``mode``.

    Live needs a dry run that passed, and any run needs a validation that passed, because
    the export is resolvable only for a validated pipeline. The delivery itself is submitted
    when the job finishes, by `on_job_finished`, or by hand through `deliver_run`.
    """
    from open_climate_service.openeo.jobs import get_openeo_job_service
    from open_climate_service.openeo.schemas import OpenEOJobCreate

    spec = record.spec
    if mode not in ("dry_run", "live"):
        raise ValueError("mode must be dry_run or live")
    delivery_mode: Literal["dry_run", "live"] = "live" if mode == "live" else "dry_run"
    if record.validation is None or not record.validation.valid:
        raise ValueError("the pipeline must validate before it can run; save it again to re-check")
    if mode == "live" and (record.dry_run is None or not record.dry_run.passed):
        raise ValueError("a live run needs a dry run that passed first")
    service = get_openeo_job_service()
    job = service.create_job(
        OpenEOJobCreate(
            process=_run_process(spec, start, end),
            title=f"{spec.name or spec.id}: {start} to {end} ({mode.replace('_', ' ')})",
            description=f"Pipeline '{spec.id}' run once, {start} to {end}, delivered as {mode.replace('_', ' ')}",
        )
    )
    service.start_job(job.id)
    run = PipelineRun(
        job_id=job.id,
        mode=delivery_mode,
        start=start,
        end=end,
        submitted_at=utc_now().isoformat(),
        idempotency_key=f"pipeline:{spec.id}:{job.id}:{delivery_mode}",
    )
    record.runs.insert(0, run)
    del record.runs[_RUNS_KEPT:]
    return run


def deliver_run(record: PipelineRecord, job_id: str) -> PipelineRun:
    """Submit the delivery a finished run owes, through the same path as POST /exports/{id}."""
    from open_climate_service.exports.delivery import submit_delivery
    from open_climate_service.openeo.jobs import store_get_job
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    run = next((item for item in record.runs if item.job_id == job_id), None)
    if run is None:
        raise LookupError(f"pipeline '{record.spec.id}' has no run for job '{job_id}'")
    if run.delivery_job_id is not None:
        return run
    job = store_get_job(job_id)
    if job is None:
        raise LookupError(f"job '{job_id}' no longer exists")
    if job.status != OpenEOJobStatus.FINISHED:
        raise ValueError(f"job '{job_id}' is {job.status.value}, not finished")
    try:
        delivery_job_id, _ = submit_delivery(record.spec.id, job_id, run.mode == "dry_run", run.idempotency_key)
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        run.error = str(detail) if detail else str(exc) or type(exc).__name__
        raise
    run.delivery_job_id = delivery_job_id
    run.delivery_status_url = f"/exports/{record.spec.id}/jobs/{delivery_job_id}"
    run.delivered_at = utc_now().isoformat()
    run.error = None
    return run


def on_job_finished(job: Any) -> None:
    """Deliver the run a pipeline owes for a job that just finished. Never raises."""
    from open_climate_service.openeo.schemas import OpenEOJobStatus
    from open_climate_service.pipelines import store

    if getattr(job, "status", None) != OpenEOJobStatus.FINISHED:
        return
    try:
        for record in store.list_records():
            if any(run.job_id == job.id and run.delivery_job_id is None for run in record.runs):
                try:
                    deliver_run(record, job.id)
                except Exception:
                    logger.exception("Pipeline '%s' could not deliver run %s", record.spec.id, job.id)
                store.save_record(record)
    except Exception:
        logger.exception("Pipeline run delivery after job %s failed", getattr(job, "id", "?"))


def run_views(record: PipelineRecord) -> list[dict[str, Any]]:
    """Runs with their job and delivery states, for the page."""
    from open_climate_service.jobs.service import get_job_service
    from open_climate_service.openeo.jobs import store_get_job

    views: list[dict[str, Any]] = []
    for run in record.runs:
        job = store_get_job(run.job_id)
        view: dict[str, Any] = {**run.model_dump(mode="json"), "job_status": job.status.value if job else "gone"}
        if run.delivery_job_id:
            try:
                delivery = get_job_service().get_job_or_404(run.delivery_job_id)
                result = delivery.result if isinstance(delivery.result, dict) else {}
                view["delivery_status"] = str(getattr(delivery.status, "value", delivery.status))
                view["delivery_outcome"] = result.get("outcome")
                view["delivery_submitted"] = result.get("submitted")
                view["delivery_conflicts"] = len(result.get("conflicts") or [])
            except Exception:
                view["delivery_status"] = "unknown"
        views.append(view)
    return views
