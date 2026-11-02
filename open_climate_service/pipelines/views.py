"""A stored pipeline as clients and pages read it: its definition resolved into what it reads, runs and targets.

The destination is described from its plugin's own settings form and the labels resolved at
the last validation, so nothing here knows a destination, and reading never contacts one.
"""

from __future__ import annotations

from typing import Any

from open_climate_service.exports.base import BaseExportPlugin
from open_climate_service.openeo import workflows as workflow_registry
from open_climate_service.pipelines import runner, service
from open_climate_service.pipelines import workflow as workflow_binding
from open_climate_service.pipelines.schemas import (
    DestinationSetting,
    PipelineDestinationView,
    PipelineInput,
    PipelineRecord,
    PipelineTrigger,
    PipelineView,
    PipelineWorkflow,
)


class Names:
    """Titles of workflows, datasets and feature collections, read once for a request."""

    def __init__(self) -> None:
        from open_climate_service.features.services import list_feature_collections
        from open_climate_service.ingestions.services import list_datasets
        from open_climate_service.openeo.workflows import list_workflows

        self.workflows = {record.id: record.summary or record.id for record in list_workflows().processes}
        self.datasets = {item.dataset_id: item.dataset_name for item in list_datasets().items}
        self.collections = {record.id: record.name for record in list_feature_collections().items}


def destination_settings(
    plugin: BaseExportPlugin, settings: dict[str, Any], labels: dict[str, dict[str, str]]
) -> list[DestinationSetting]:
    """The settings as the plugin's form names them, each looked-up value with the label it resolved to."""

    def item(field: dict[str, Any], value: Any, path: str) -> DestinationSetting:
        label = labels.get(path, {}).get(value) if isinstance(value, str) else None
        return DestinationSetting(name=field["name"], title=field["title"], value=value, label=label)

    items: list[DestinationSetting] = []
    for field in service.settings_fields(plugin):
        stored = settings.get(field["name"])
        if field["kind"] == "rows":
            rows = [row for row in stored if isinstance(row, dict)] if isinstance(stored, list) else []
            columns = field["columns"]
            items.append(
                DestinationSetting(
                    name=field["name"],
                    title=field["title"],
                    rows=[
                        [
                            item(column, row[column["name"]], f"{field['name']}.{column['name']}")
                            for column in columns
                            if row.get(column["name"]) is not None
                        ]
                        for row in rows
                    ],
                )
            )
        elif stored is not None:
            items.append(item(field, stored, field["name"]))
    return items


def _destination(record: PipelineRecord) -> PipelineDestinationView:
    spec = record.spec
    plugin = service.destination_plugins().get(spec.destination.plugin)
    if plugin is None:
        return PipelineDestinationView(plugin=spec.destination.plugin, title=spec.destination.plugin, settings=[])
    presentation = plugin.pipeline_settings_form()
    labels = record.validation.labels if record.validation else {}
    return PipelineDestinationView(
        plugin=spec.destination.plugin,
        title=presentation.title if presentation is not None else spec.destination.plugin,
        settings=destination_settings(plugin, spec.destination.settings, labels),
    )


def pipeline_view(record: PipelineRecord, mount: str, names: Names) -> PipelineView:
    """The pipeline with its inputs, workflow, destination, trigger and latest run; links carry ``mount``."""
    spec = record.spec
    inputs = [
        PipelineInput(
            kind=kind,
            id=input_id,
            name=(names.datasets if kind == "dataset" else names.collections).get(input_id, input_id),
            parameter=parameter,
            href=f"{mount}/{'datasets' if kind == 'dataset' else 'features'}/{input_id}",
        )
        for kind, input_id, parameter in workflow_binding.inputs(spec.workflow)
    ]
    workflow = workflow_registry.get_workflow(spec.workflow.id)
    unbound = [
        name
        for name in (workflow_binding.interval_parameters(workflow) if workflow is not None else [])
        if name not in spec.workflow.arguments
    ]
    here = f"{mount}/pipelines/{spec.id}"
    return PipelineView(
        **record.model_dump(include={"spec", "revision", "created_at", "updated_at", "automation_since"}),
        approved=service.approved(record),
        validation=record.validation,
        dry_run=record.dry_run,
        events=record.events,
        trigger=PipelineTrigger(
            policy=spec.delivery.policy,
            datasets=[item.id for item in inputs if item.kind == "dataset"]
            if spec.delivery.policy == "on_update"
            else [],
        ),
        inputs=inputs,
        workflow=PipelineWorkflow(
            id=spec.workflow.id,
            title=names.workflows.get(spec.workflow.id, spec.workflow.id),
            arguments=spec.workflow.arguments,
            interval_parameter=unbound[0] if len(unbound) == 1 else None,
            href=f"{mount}/process_graphs/{spec.workflow.id}",
        ),
        destination=_destination(record),
        latest_run=runner.run_view(spec.id, record.runs[0], mount) if record.runs else None,
        links={"self": here, "runs": f"{here}/runs"},
    )
