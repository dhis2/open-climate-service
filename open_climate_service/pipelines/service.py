"""Validate and dry-run a pipeline: its workflow binding, its inputs and its destination.

Nothing here knows a destination. The workflow reference is checked against the workflow
registry; the destination's settings, export definition and destination-side checks come
from its export plugin's pipeline hooks. Running and delivering is `pipelines.runner`.
"""

from __future__ import annotations

import logging
from typing import Any

import jsonschema

from open_climate_service import config as api_config
from open_climate_service.exports.base import BaseExportPlugin, DestinationContext, PipelineDataset
from open_climate_service.exports.manifest import plugin_identity
from open_climate_service.pipelines import workflow as workflow_binding
from open_climate_service.pipelines.schemas import (
    Check,
    DryRunResult,
    PipelineRecord,
    PipelineSpec,
    ValidationResult,
)
from open_climate_service.pipelines.workflow import BoundWorkflow, WorkflowBindingError
from open_climate_service.shared.provenance import json_digest
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)


def dataset_record(dataset_id: str) -> Any:
    """The managed dataset list entry, or None."""
    from open_climate_service.ingestions.services import list_datasets

    for item in list_datasets().items:
        if item.dataset_id == dataset_id:
            return item
    return None


def destination_plugins() -> dict[str, BaseExportPlugin]:
    """The export plugins that declare themselves pipeline destinations, by id."""
    from open_climate_service.exports.registry import load_export_plugins

    return {
        plugin_id: plugin
        for plugin_id, plugin in load_export_plugins().items()
        if plugin.destination_settings_schema() is not None
    }


def destination_plugin(spec: PipelineSpec) -> BaseExportPlugin:
    """The spec's destination plugin, or a ValueError naming the ones available."""
    plugins = destination_plugins()
    plugin = plugins.get(spec.destination.plugin)
    if plugin is None:
        available = ", ".join(sorted(plugins)) or "none"
        raise ValueError(f"'{spec.destination.plugin}' is not a pipeline destination plugin; available: {available}")
    return plugin


def destination_context(spec: PipelineSpec, bound: BoundWorkflow) -> DestinationContext:
    """What the workflow binds, in the form a destination plugin reads."""
    datasets = []
    for dataset_id in bound.dataset_ids:
        dataset = dataset_record(dataset_id)
        datasets.append(
            PipelineDataset(
                id=dataset_id,
                variable=getattr(dataset, "variable", None),
                cadence=getattr(dataset, "period_type", None),
            )
        )
    return DestinationContext(
        export_id=spec.id, datasets=tuple(datasets), feature_collections=bound.feature_collections
    )


def compile_export(spec: PipelineSpec, bound: BoundWorkflow) -> dict[str, Any]:
    """The named-export definition the destination plugin compiles from its settings."""
    plugin = destination_plugin(spec)
    schema = plugin.destination_settings_schema()
    if schema is None:
        raise ValueError(f"export plugin '{plugin.id}' no longer declares the pipeline destination capability")
    try:
        jsonschema.validate(instance=spec.destination.settings, schema=schema)
    except jsonschema.SchemaError as exc:
        message = f"export plugin '{plugin.id}' has an invalid destination settings schema: {exc.message}"
        raise ValueError(message) from None
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        raise ValueError(f"destination setting {location + ': ' if location else ''}{exc.message}") from None
    definition = plugin.export_definition(spec.destination.settings, destination_context(spec, bound))
    return {**definition, "id": spec.id, "plugin": plugin.id}


def bindings_digest(spec: PipelineSpec, bound: BoundWorkflow | None) -> str:
    """Digest of everything a dry run's verdict depends on.

    The workflow reference and its arguments, the workflow's own definition, and the
    destination with its settings. The name, mode and policy are left out: switching a
    pipeline live is what an approval is for, and must not invalidate it.
    """
    from open_climate_service.exports.service import resolve_export_definition

    plugin: BaseExportPlugin | None = None
    resolved_export: dict[str, Any] | None = None
    try:
        plugin = destination_plugin(spec)
        if bound is not None:
            resolved = resolve_export_definition(compile_export(spec, bound))
            resolved_export = {
                "id": resolved.export_id,
                "mapping": resolved.mapping,
                "references": resolved.references,
            }
    except ValueError:
        # Invalid bindings still need a stable digest for their validation result.
        pass
    return json_digest(
        {
            "workflow": spec.workflow.model_dump(mode="json"),
            "definition": bound.definition_digest if bound is not None else None,
            "destination": spec.destination.model_dump(mode="json"),
            "destination_plugin": plugin_identity(plugin).model_dump(mode="json") if plugin is not None else None,
            "export": resolved_export,
        }
    )


def approved(record: PipelineRecord) -> bool:
    """Whether a passed dry run approves the bindings the pipeline was last validated on."""
    dry_run, validation = record.dry_run, record.validation
    return (
        dry_run is not None
        and dry_run.passed
        and validation is not None
        and dry_run.bindings is not None
        and dry_run.bindings == validation.bindings
    )


def choices() -> dict[str, Any]:
    """What the create form offers: pipeline-ready workflows, inputs and destinations."""
    from open_climate_service.features.services import list_feature_collections
    from open_climate_service.ingestions.services import list_datasets
    from open_climate_service.openeo.workflows import list_workflows

    workflows = [
        {
            "id": record.id,
            "summary": record.summary or record.id,
            "parameters": workflow_binding.parameters(record),
            "interval": workflow_binding.interval_parameters(record),
        }
        for record in list_workflows().processes
        if workflow_binding.pipeline_ready(record)
    ]
    datasets = [
        {
            "id": item.dataset_id,
            "name": item.dataset_name,
            "variable": item.variable,
            "cadence": item.period_type,
            "coverage": item.extent.temporal,
        }
        for item in list_datasets().items
        if item.publication.status == "published" and item.item_type == "coverage"
    ]
    collections = [
        {
            "id": record.id,
            "name": record.name,
            "feature_count": record.feature_count,
            "geometry_types": record.geometry_types,
        }
        for record in list_feature_collections().items
    ]
    destinations = [
        {
            "id": plugin_id,
            "format": plugin.format,
            "schema": plugin.destination_settings_schema(),
            "settings_form": plugin.pipeline_settings_form(),
        }
        for plugin_id, plugin in sorted(destination_plugins().items())
    ]
    return {"workflows": workflows, "datasets": datasets, "collections": collections, "destinations": destinations}


# --- validation ------------------------------------------------------------------------------


def validate_pipeline(record: PipelineRecord) -> ValidationResult:
    """Check every binding a pipeline relies on, in the order a run would hit them."""
    from open_climate_service.exports.service import resolve_export_definition

    spec = record.spec
    checks: list[Check] = []
    bound: BoundWorkflow | None = None
    export: dict[str, Any] | None = None

    try:
        bound = workflow_binding.bind(spec.workflow)
        checks.append(
            Check(
                id="workflow",
                status="pass",
                message=(
                    f"'{bound.record.id}' with {len(spec.workflow.arguments)} bound argument(s); "
                    f"each run supplies '{bound.interval_parameter}'"
                ),
            )
        )
    except WorkflowBindingError as exc:
        checks.append(Check(id="workflow", status="fail", message=str(exc)))

    if bound is not None:
        checks.append(_check_inputs(bound, on_update=spec.delivery.policy == "on_update"))
        plugin: BaseExportPlugin | None = None
        try:
            plugin = destination_plugin(spec)
            export = compile_export(spec, bound)
            resolve_export_definition(export)
            checks.append(
                Check(
                    id="destination",
                    status="pass",
                    message=f"saves as {plugin.format} through export '{spec.id}' of plugin '{plugin.id}'",
                )
            )
        except ValueError as exc:
            export = None
            checks.append(Check(id="destination", status="fail", message=str(exc)))
        if plugin is not None:
            # Run even when the definition did not compile, so every destination problem shows at once.
            context = destination_context(spec, bound)
            for check in plugin.check_destination(spec.destination.settings, context):
                checks.append(Check(id=check.id, status=check.status, message=check.message))
        checks.append(_check_schedule(bound))

    checks.append(_check_configuration(spec))
    digest = bindings_digest(spec, bound)
    checks.append(_check_delivery(record, digest))

    return ValidationResult(
        valid=not any(check.status == "fail" for check in checks),
        checked_at=utc_now().isoformat(),
        bindings=digest,
        export=export,
        checks=checks,
    )


def _check_inputs(bound: BoundWorkflow, *, on_update: bool) -> Check:
    """Every dataset the workflow reads must be published; every collection registered.

    A pipeline that runs on updates must bind a source dataset, a ``collection-id``
    parameter, or no update could ever start it.
    """
    from open_climate_service.features.services import registered_collections

    problems: list[str] = []
    if on_update and not bound.dataset_ids:
        problems.append(
            "the pipeline runs after source dataset updates but binds no dataset: the workflow declares no "
            "collection-id parameter; choose manual runs, or a workflow that declares one"
        )
    found: list[str] = []
    for dataset_id in bound.dataset_ids:
        dataset = dataset_record(dataset_id)
        if dataset is None:
            problems.append(f"'{dataset_id}' is not a managed dataset")
        elif dataset.publication.status != "published":
            problems.append(f"'{dataset_id}' is not published")
        else:
            coverage = dataset.extent.temporal
            found.append(f"'{dataset_id}' published, {coverage.start} to {coverage.end}")
    registered = registered_collections() if bound.feature_collections else {}
    for collection_id in bound.feature_collections:
        if collection_id not in registered:
            problems.append(f"'{collection_id}' is not a registered feature collection; fetch it first")
        else:
            found.append(f"features '{collection_id}'")
    if problems:
        return Check(id="inputs", status="fail", message="; ".join(problems))
    return Check(id="inputs", status="pass", message="; ".join(found) or "the workflow binds no stored input")


def _check_configuration(spec: PipelineSpec) -> Check:
    """Refuse an id a configured named export already owns: the two would be ambiguous."""
    raw = api_config.get_config().get("exports", [])
    configured = {item.get("id") for item in raw if isinstance(item, dict)} if isinstance(raw, list) else set()
    if spec.id in configured:
        return Check(
            id="configuration",
            status="fail",
            message=(
                f"export id '{spec.id}' is already configured in climate-service.yaml; a pipeline owns the export "
                "named after it, so rename the pipeline or remove the configured export"
            ),
        )
    return Check(id="configuration", status="pass", message=f"export id '{spec.id}' belongs to this pipeline")


def _check_schedule(bound: BoundWorkflow) -> Check:
    """Report how the source datasets are kept current. The pipeline owns no sync clock of its own."""
    raw = api_config.get_config().get("scheduler", {})
    entries = raw.get("dataset_sync", []) if isinstance(raw, dict) else []
    enabled = isinstance(raw, dict) and raw.get("enabled") is True
    if not bound.dataset_ids:
        return Check(id="schedule", status="skip", message="the workflow reads no managed dataset")
    parts: list[str] = []
    status: str = "pass"
    for dataset_id in bound.dataset_ids:
        existing = [item for item in entries if isinstance(item, dict) and item.get("dataset_id") == dataset_id]
        if not existing:
            status = "skip"
            parts.append(f"'{dataset_id}' has no sync schedule; it updates only after a manual sync")
        elif not enabled:
            status = "skip"
            parts.append(
                f"'{dataset_id}' has a sync schedule ({existing[0].get('cron', '?')!r}), but the instance scheduler "
                "is disabled"
            )
        else:
            parts.append(f"'{dataset_id}' is synced on schedule {existing[0].get('cron', '?')!r}")
    return Check(id="schedule", status="pass" if status == "pass" else "skip", message="; ".join(parts))


def _check_delivery(record: PipelineRecord, digest: str) -> Check:
    """Whether and when results go out, and whether the pipeline has earned a live mode."""
    delivery = record.spec.delivery
    when = {"on_update": "after each source dataset update", "manual": "only when run by hand"}[delivery.policy]
    if delivery.mode == "paused":
        return Check(
            id="delivery", status="pass", message=f"paused; no run starts by itself and nothing is sent ({when})"
        )
    if delivery.mode == "dry_run":
        return Check(id="delivery", status="pass", message=f"dry run {when}; the destination stores nothing")
    try:
        plugin = destination_plugin(record.spec)
    except ValueError:
        return Check(id="delivery", status="fail", message="live delivery needs a destination plugin")
    if not plugin.supports_delivery:
        return Check(id="delivery", status="fail", message=f"plugin '{plugin.id}' renders only; it cannot deliver live")
    dry_run = record.dry_run
    if dry_run is not None and dry_run.passed and dry_run.bindings == digest:
        return Check(
            id="delivery", status="pass", message=f"live {when}, after a dry run that passed on {dry_run.ran_at}"
        )
    if dry_run is not None and dry_run.passed:
        return Check(
            id="delivery",
            status="fail",
            message="the last dry run passed for different bindings; dry-run the current ones before going live",
        )
    return Check(id="delivery", status="fail", message="live delivery needs a dry run that passed first")


# --- dry run ---------------------------------------------------------------------------------


def dry_run_pipeline(record: PipelineRecord, start: str, end: str) -> DryRunResult:
    """Run the workflow for a bounded range, render it through the destination, and dry-run it there.

    The graph is the one a run submits, executed synchronously. Where the destination can
    validate without storing (``supports_dry_run``) the payload is sent with ``dry_run=True``;
    otherwise it is only rendered, and the result says so.
    """
    spec = record.spec
    ran_at = utc_now().isoformat()
    digest: str | None = None
    try:
        bound = workflow_binding.bind(spec.workflow)
        digest = bindings_digest(spec, bound)
        plugin = destination_plugin(spec)
        resolved, rendered = _render(spec, bound, plugin, start, end)
        report: dict[str, Any] | None = None
        if plugin.supports_delivery and plugin.supports_dry_run:
            # The same target binding a saved export records in its manifest and delivers to.
            binding = plugin.target_binding(resolved.references)
            if binding is None:
                raise ValueError(f"the destination names no target plugin '{plugin.id}' can deliver to")
            report_model = plugin.send(rendered.content, plugin.delivery_target(binding), dry_run=True)
            report = report_model.model_dump(mode="json")
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        message = str(detail) if detail else str(exc) or type(exc).__name__
        logger.info("Pipeline '%s' dry run failed: %s", spec.id, message)
        return DryRunResult(ran_at=ran_at, start=start, end=end, bindings=digest, error=message)
    return DryRunResult(
        ran_at=ran_at,
        start=start,
        end=end,
        bindings=digest,
        values=rendered.record_count,
        periods=sorted(set(rendered.periods or ())),
        sample=plugin.preview(rendered.content),
        report=report,
        rendered_only=report is None,
    )


def _render(
    spec: PipelineSpec, bound: BoundWorkflow, plugin: BaseExportPlugin, start: str, end: str
) -> tuple[Any, Any]:
    """Execute the run graph and render it through the export definition compiled for this spec.

    The definition is resolved directly rather than by name, so a draft dry-runs exactly what
    it would save, before its export is resolvable by other jobs.
    """
    from open_climate_service.exports.service import render_resolved_export, resolve_export_definition
    from open_climate_service.openeo import execution
    from open_climate_service.openeo.execution import SaveResultEnvelope

    resolved = resolve_export_definition(compile_export(spec, bound))
    process, _ = workflow_binding.build_process(
        spec.workflow, bound, start=start, end=end, save_format=plugin.format, export_id=spec.id
    )
    envelope = execution.run_process_graph(process)
    if not isinstance(envelope, SaveResultEnvelope):
        raise TypeError("pipeline graph did not return a save_result envelope")
    if envelope.provenance is None:
        raise ValueError("pipeline graph returned no execution provenance")
    rendered = render_resolved_export(envelope.data, resolved, provenance=envelope.provenance)
    if rendered.record_count == 0:
        raise ValueError("the workflow produced no values to submit")
    return resolved, rendered
