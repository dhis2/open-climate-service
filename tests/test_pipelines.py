"""Pipelines: one workflow, its bindings and a destination, validated, dry-run and run.

Without a server or DHIS2: datasets, feature collections and DHIS2 are stubbed, and every
test writes under its own temporary data directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.exports import dhis2_destination
from open_climate_service.exports.base import (
    BaseExportPlugin,
    DestinationCheck,
    DestinationContext,
    PipelineSettingsForm,
    RenderedExport,
    SettingsFormField,
    SettingsFormSection,
)
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.pipelines import forms, runner, service, store
from open_climate_service.pipelines import workflow as workflow_binding
from open_climate_service.pipelines.schemas import (
    Check,
    DryRunResult,
    PipelineRecord,
    PipelineSpec,
    ValidationResult,
    WorkflowReference,
)

_OU = ["ImspTQPwCqd", "O6uvpzGd5pu"]
_ELEMENT = "LeePHYkZwBk"
_SETTINGS: dict[str, Any] = {"connection": "local", "data_set": "N63vVmt2uY0", "series": [{"data_element": _ELEMENT}]}
_SPEC: dict[str, Any] = {
    "id": "rain-to-dhis2",
    "name": "Rain to DHIS2",
    "workflow": {
        "id": "aggregate_by_features",
        "arguments": {"dataset_id": "rain_monthly", "geometries": {"from_features": "districts"}, "method": "mean"},
    },
    "destination": {"plugin": "dhis2", "settings": _SETTINGS},
}
_FORM = {
    "id": "rain-to-dhis2",
    "name": "Rain to DHIS2",
    "workflow": "aggregate_by_features",
    "destination": "dhis2",
    "arg.dataset_id": "rain_monthly",
    "arg.geometries": "districts",
    "arg.method": "mean",
    "dest.connection": "local",
    "dest.data_set": "N63vVmt2uY0",
    "dest.series.0.data_element": _ELEMENT,
}


def _spec(**changes: Any) -> PipelineSpec:
    return PipelineSpec.model_validate({**_SPEC, **changes})


def _dataset(period_type: str = "monthly", published: bool = True, dataset_id: str = "rain_monthly") -> Any:
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_name="Rain, monthly",
        variable="tp",
        period_type=period_type,
        item_type="coverage",
        extent=SimpleNamespace(temporal=SimpleNamespace(start="2025-01", end="2025-02")),
        publication=SimpleNamespace(status="published" if published else "unpublished"),
    )


def _binding(geometry: str = "Polygon", connection: str | None = "local") -> dhis2_destination.FeatureBinding:
    return dhis2_destination.FeatureBinding(ids=list(_OU), geometry_types=[geometry], connection=connection, level=2)


@pytest.fixture
def datasets() -> list[Any]:
    return [_dataset()]


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets: list[Any]) -> None:
    from open_climate_service.features import services as feature_services
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.system import templates as system_templates

    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    # Event replay reads both job stores: keep them inside this test's directory.
    from open_climate_service.jobs import store as native_store
    from open_climate_service.openeo import jobs as openeo_jobs

    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(
        api_config, "_cache", {"dhis2_connections": [{"id": "local", "url": "http://dhis2.test", "token_env": "T"}]}
    )
    # Built-in workflows only: never read or create the developer's runtime workflow store.
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [])
    listing = SimpleNamespace(items=datasets)
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: listing)
    monkeypatch.setattr(system_templates, "list_datasets", lambda: listing)
    monkeypatch.setattr(
        feature_services,
        "list_feature_collections",
        lambda: SimpleNamespace(
            items=[SimpleNamespace(id="districts", name="Districts", feature_count=2, geometry_types=["Polygon"])]
        ),
    )
    monkeypatch.setattr(feature_services, "registered_collections", lambda: {"districts": SimpleNamespace()})
    monkeypatch.setattr(
        workflow_binding,
        "_feature_versions",
        lambda arguments: {"districts": "2026-01-01T00:00:00"} if "districts" in json.dumps(arguments) else {},
    )
    monkeypatch.setattr(
        dhis2_destination,
        "feature_binding",
        lambda collection_id: (
            _binding()
            if collection_id == "districts"
            else (_ for _ in ()).throw(LookupError(f"'{collection_id}' is not a registered feature collection"))
        ),
    )


class _FakeClient:
    def __init__(self, data_set: dict[str, Any], summary: dict[str, Any]) -> None:
        self.data_set, self.summary, self.posts = data_set, summary, []

    def get(self, path: str, *, params: Any = None) -> dict[str, Any]:
        return self.data_set

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        self.posts.append(kwargs)
        return self.summary

    def close(self) -> None:
        pass


def _dhis2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    period_type: str = "Monthly",
    assigned: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> _FakeClient:
    client = _FakeClient(
        {
            "id": "N63vVmt2uY0",
            "displayName": "OCS test",
            "periodType": period_type,
            "dataSetElements": [{"dataElement": {"id": _ELEMENT}}],
            "organisationUnits": [{"id": ou} for ou in (assigned if assigned is not None else _OU)],
        },
        summary
        or {
            "status": "SUCCESS",
            "importCount": {"imported": 0, "updated": 4, "ignored": 0, "deleted": 0},
            "conflicts": [],
        },
    )
    import open_climate_service.exports.dhis2 as dhis2_module

    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: client)
    return client


def _by_id(result: ValidationResult) -> dict[str, Check]:
    return {check.id: check for check in result.checks}


# --- schema: one workflow, opaque destination settings ---------------------------------------


@pytest.mark.parametrize("pipeline_id", ["bad id", "a.b", "-x"])
def test_spec_rejects_ids_that_cannot_name_an_export(pipeline_id: str) -> None:
    with pytest.raises(ValueError, match="id must start"):
        _spec(id=pipeline_id)


def test_spec_references_exactly_one_workflow_and_no_steps() -> None:
    with pytest.raises(ValueError):
        PipelineSpec.model_validate({**_SPEC, "steps": [{"workflow": "a"}, {"workflow": "b"}]})
    with pytest.raises(ValueError):
        PipelineSpec.model_validate({**_SPEC, "workflow": [_SPEC["workflow"], _SPEC["workflow"]]})
    with pytest.raises(ValueError):
        PipelineSpec.model_validate({key: value for key, value in _SPEC.items() if key != "workflow"})


def test_destination_settings_are_opaque_to_the_core() -> None:
    # A DHIS2-invalid UID is accepted by the core schema; the DHIS2 plugin refuses it.
    spec = _spec(destination={"plugin": "dhis2", "settings": {**_SETTINGS, "series": [{"data_element": "nope"}]}})
    assert spec.destination.settings["series"][0]["data_element"] == "nope"
    assert set(PipelineSpec.model_fields) == {"id", "name", "workflow", "destination", "delivery"}


def test_a_scheduled_release_is_refused_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        _spec(delivery={"policy": "scheduled", "release_cron": "0 2 5 * *"})


def test_binding_reports_every_problem_with_the_reference(instance: None) -> None:
    with pytest.raises(workflow_binding.WorkflowBindingError, match="not a registered workflow"):
        workflow_binding.bind(WorkflowReference(id="nope"))
    with pytest.raises(workflow_binding.WorkflowBindingError) as caught:
        workflow_binding.bind(
            WorkflowReference(
                id="aggregate_by_features",
                arguments={
                    "temporal_extent": ["2025-01-01", "2025-01-31"],
                    "method": "mode",
                    "geometries": "districts",
                    "colour": "red",
                },
            )
        )
    message = str(caught.value)
    assert "unknown argument(s) colour" in message
    assert "leave 'temporal_extent' unbound" in message
    assert "missing required argument(s) dataset_id" in message
    assert "'method' must be one of" in message
    assert "from_features" in message


def test_a_workflow_that_saves_its_own_result_is_refused(instance: None) -> None:
    with pytest.raises(workflow_binding.WorkflowBindingError, match="saves its own result"):
        workflow_binding.bind(
            WorkflowReference(
                id="aggregate_to_dhis2_json",
                arguments={"dataset_id": "rain_monthly", "geometries": {"from_features": "districts"}, "export": "x"},
            )
        )


def test_a_workflow_that_calls_a_saving_workflow_is_refused(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    saving = {
        "id": "saving_child",
        "parameters": [{"name": "period", "schema": {"type": "array", "subtype": "temporal-interval"}}],
        "process_graph": {
            "save": {
                "process_id": "save_result",
                "arguments": {"data": 1, "format": "JSON"},
                "result": True,
            }
        },
    }
    outer = {
        "id": "outer",
        "parameters": [{"name": "period", "schema": {"type": "array", "subtype": "temporal-interval"}}],
        "process_graph": {
            "child": {
                "process_id": "saving_child",
                "arguments": {"period": {"from_parameter": "period"}},
                "result": True,
            }
        },
    }
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [outer, saving])
    with pytest.raises(workflow_binding.WorkflowBindingError, match="saves its own result"):
        workflow_binding.bind(WorkflowReference(id="outer"))


def test_run_graph_freezes_the_registered_workflow(instance: None) -> None:
    spec = _spec(
        workflow={
            **_SPEC["workflow"],
            "arguments": {k: v for k, v in _SPEC["workflow"]["arguments"].items() if k != "method"},
        }
    )
    bound = workflow_binding.bind(spec.workflow)
    assert bound.interval_parameter == "temporal_extent"
    assert bound.dataset_ids == ("rain_monthly",) and bound.feature_collections == ("districts",)
    process, resolved = workflow_binding.build_process(
        spec.workflow, bound, start="2025-01-01", end="2025-01-31", save_format="DHIS2JSON", export_id=spec.id
    )
    graph = process["process_graph"]
    registered = workflow_store.get_workflow("aggregate_by_features")
    assert registered is not None
    # The workflow's own nodes, inlined from the registry: nothing here is authored by the pipeline.
    assert set(graph) == {"features_districts", "save", *(f"workflow__{node}" for node in registered.process_graph)}
    assert graph["features_districts"]["arguments"] == {"id": "districts", "version": "2026-01-01T00:00:00"}
    assert graph["workflow__load"] == {
        "process_id": "load_collection",
        "arguments": {"id": "rain_monthly", "temporal_extent": ["2025-01-01", "2025-01-31"]},
    }
    assert graph["workflow__zonal"]["arguments"] == {
        "data": {"from_node": "workflow__load"},
        "geometries": {"from_node": "features_districts"},
        "reducer": "mean",  # the declared default, frozen with the run
    }
    assert graph["save"]["arguments"] == {
        "data": {"from_node": "workflow__zonal"},
        "format": "DHIS2JSON",
        "options": {"export": "rain-to-dhis2"},
    }
    assert [node_id for node_id, node in graph.items() if node.get("result")] == ["save"]
    assert resolved["temporal_extent"] == ["2025-01-01", "2025-01-31"]
    assert resolved["$feature_versions"] == {"districts": "2026-01-01T00:00:00"}


_COMPOSED: dict[str, Any] = {
    "id": "anomaly_by_features",
    "summary": "Scale a dataset, then aggregate it by features",
    "parameters": [
        {"name": "observed", "schema": {"type": "string", "subtype": "collection-id"}},
        {"name": "period", "schema": {"type": "array", "subtype": "temporal-interval"}},
        {"name": "zones", "schema": {"type": "object", "subtype": "geojson"}},
        {"name": "factor", "schema": {"type": "number"}, "default": 1},
    ],
    "process_graph": {
        "load": {
            "process_id": "load_collection",
            "arguments": {"id": {"from_parameter": "observed"}, "temporal_extent": {"from_parameter": "period"}},
        },
        "scale": {
            "process_id": "apply",
            "arguments": {"data": {"from_node": "load"}, "process": {"process_graph": {}}},
        },
        "zonal": {
            "process_id": "aggregate_spatial_weighted",
            "arguments": {"data": {"from_node": "scale"}, "geometries": {"from_parameter": "zones"}, "reducer": "mean"},
            "result": True,
        },
    },
}


def test_a_multi_process_workflow_runs_without_the_pipeline_knowing_its_steps(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [_COMPOSED])
    reference = WorkflowReference(
        id="anomaly_by_features",
        arguments={"observed": "rain_monthly", "zones": {"from_features": "districts"}, "factor": 2.5},
    )
    bound = workflow_binding.bind(reference)
    assert bound.interval_parameter == "period" and bound.dataset_ids == ("rain_monthly",)
    process, _ = workflow_binding.build_process(
        reference, bound, start="2025-01-01", end="2025-03-31", save_format="CSV", export_id="x"
    )
    graph = process["process_graph"]
    assert {node for node in graph if node.startswith("workflow__")} == {
        f"workflow__{node}" for node in _COMPOSED["process_graph"]
    }
    assert graph["workflow__load"]["arguments"]["temporal_extent"] == ["2025-01-01", "2025-03-31"]
    assert graph["workflow__scale"]["arguments"]["process"] == {"process_graph": {}}
    assert graph["save"]["arguments"]["data"] == {"from_node": "workflow__zonal"}
    with pytest.raises(workflow_binding.WorkflowBindingError, match="'factor' must be number"):
        workflow_binding.bind(reference.model_copy(update={"arguments": {**reference.arguments, "factor": "x"}}))


_OUTER: dict[str, Any] = {
    "id": "rain_by_districts",
    "summary": "A workflow that composes another",
    "parameters": [
        {"name": "dataset_id", "schema": {"type": "string", "subtype": "collection-id"}},
        {"name": "temporal_extent", "schema": {"type": "array", "subtype": "temporal-interval"}},
        {"name": "geometries", "schema": {"type": "object", "subtype": "geojson"}},
    ],
    "process_graph": {
        "agg": {
            "process_id": "aggregate_by_features",
            "arguments": {
                "dataset_id": {"from_parameter": "dataset_id"},
                "temporal_extent": {"from_parameter": "temporal_extent"},
                "geometries": {"from_parameter": "geometries"},
                "method": "max",
            },
            "result": True,
        }
    },
}


def test_a_queued_run_keeps_the_workflow_it_was_approved_with(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from openeo_pg_parser_networkx import OpenEOProcessGraph

    builtin = workflow_store.get_workflow("aggregate_by_features")
    assert builtin is not None
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [_OUTER])
    reference = WorkflowReference(
        id="rain_by_districts", arguments={"dataset_id": "rain_monthly", "geometries": {"from_features": "districts"}}
    )
    bound = workflow_binding.bind(reference)
    process, _ = workflow_binding.build_process(
        reference, bound, start="2025-01-01", end="2025-01-31", save_format="DHIS2JSON", export_id="x"
    )
    graph = process["process_graph"]
    # The nested call is inlined too, so no node names a workflow the registry could change.
    assert set(graph) == {"features_districts", "workflow__agg__load", "workflow__agg__zonal", "save"}
    assert graph["workflow__agg__zonal"]["arguments"]["reducer"] == "max"
    registry = {record.id for record in workflow_store.list_workflows().processes}
    assert not any(node["process_id"] in registry for node in graph.values())

    # Editing the nested workflow afterwards changes neither the stored graph nor what it runs,
    # but it does change the digest an approval is bound to.
    edited = builtin.model_dump()
    edited["process_graph"] = {**builtin.process_graph, "zonal": {**builtin.process_graph["zonal"], "result": False}}
    edited["process_graph"]["extra"] = {
        "process_id": "absolute",
        "arguments": {"x": {"from_node": "zonal"}},
        "result": True,
    }
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [_OUTER, edited])
    assert workflow_binding.bind(reference).definition_digest != bound.definition_digest
    assert "workflow__agg__extra" not in graph
    # The frozen graph is a valid openEO graph on its own.
    OpenEOProcessGraph(process)


def test_freezing_refuses_a_recursive_workflow(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    looping = {
        **_OUTER,
        "process_graph": {"agg": {**_OUTER["process_graph"]["agg"], "process_id": "rain_by_districts"}},
    }
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [looping])
    with pytest.raises(workflow_binding.WorkflowBindingError, match="calls itself"):
        workflow_binding.bind(
            WorkflowReference(
                id="rain_by_districts",
                arguments={"dataset_id": "rain_monthly", "geometries": {"from_features": "districts"}},
            )
        )


def test_freezing_refuses_inlined_node_id_collisions() -> None:
    child = workflow_store.WorkflowRecord.model_validate(
        {
            "id": "child",
            "process_graph": {"load": {"process_id": "constant", "arguments": {"x": 1}, "result": True}},
        }
    )
    graph = {
        "call": {"process_id": "child", "arguments": {}},
        "call__load": {"process_id": "constant", "arguments": {"x": 2}, "result": True},
    }
    with pytest.raises(ValueError, match="would overwrite node 'call__load'"):
        workflow_store.freeze_process_graph(graph, resolve=lambda process_id: child if process_id == "child" else None)


def test_one_workflow_serves_several_pipelines(
    instance: None, monkeypatch: pytest.MonkeyPatch, datasets: list[Any]
) -> None:
    _dhis2(monkeypatch)
    datasets.append(_dataset(dataset_id="rain_weekly", period_type="weekly"))
    first = store.create_record(_spec())
    second = store.create_record(
        _spec(
            id="weekly-rain-sum",
            workflow={
                **_SPEC["workflow"],
                "arguments": {**_SPEC["workflow"]["arguments"], "dataset_id": "rain_weekly", "method": "sum"},
            },
            destination={
                "plugin": "dhis2",
                "settings": {"connection": "local", "series": [{"data_element": "K2mt1TTI2Xd"}]},
            },
        )
    )
    one, two = service.validate_pipeline(first), service.validate_pipeline(second)
    assert one.valid and two.valid, (one.checks, two.checks)
    assert first.spec.workflow.id == second.spec.workflow.id == "aggregate_by_features"
    assert one.export is not None and two.export is not None
    assert (one.export["id"], one.export["period_type"]) == ("rain-to-dhis2", "monthly")
    assert (two.export["id"], two.export["period_type"]) == ("weekly-rain-sum", "weekly")


# --- validation ------------------------------------------------------------------------------


def test_validation_passes_when_everything_lines_up(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    result = service.validate_pipeline(store.create_record(_spec()))
    assert result.valid, result.checks
    assert {check.id: check.status for check in result.checks} == {
        "workflow": "pass",
        "inputs": "pass",
        "destination": "pass",
        "cadence": "pass",
        "series": "pass",
        "organisation_units": "pass",
        "connection": "pass",
        "dhis2_metadata": "pass",
        "schedule": "skip",
        "configuration": "pass",
        "delivery": "pass",
    }
    assert result.export == {
        "id": "rain-to-dhis2",
        "plugin": "dhis2",
        "dataset": "rain_monthly",
        "connection": "local",
        "period_type": "monthly",
        "series": [{"select": {}, "data_element": _ELEMENT}],
    }
    assert result.bindings == service.bindings_digest(_spec(), workflow_binding.bind(_spec().workflow))


def test_invalid_bindings_are_named_and_stop_the_save(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    spec = _spec(workflow={"id": "aggregate_by_features", "arguments": {"dataset_id": "rain_monthly", "extra": 1}})
    by_id = _by_id(service.validate_pipeline(store.create_record(spec)))
    assert by_id["workflow"].status == "fail" and "unknown argument(s) extra" in by_id["workflow"].message
    assert "missing required argument(s) geometries" in by_id["workflow"].message
    unknown = _spec(id="other", workflow={"id": "nope", "arguments": {}})
    assert _by_id(service.validate_pipeline(store.create_record(unknown)))["workflow"].status == "fail"


def test_plugin_validation_errors_surface_as_checks(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch, period_type="Daily", assigned=[_OU[0]])
    spec = _spec(
        workflow={
            **_SPEC["workflow"],
            "arguments": {**_SPEC["workflow"]["arguments"], "geometries": {"from_features": "nowhere"}},
        },
        destination={"plugin": "dhis2", "settings": {**_SETTINGS, "connection": "elsewhere"}},
    )
    result = service.validate_pipeline(store.create_record(spec))
    by_id = _by_id(result)
    assert not result.valid
    assert by_id["inputs"].status == "fail" and "'nowhere' is not a registered" in by_id["inputs"].message
    assert by_id["organisation_units"].status == "fail" and "not a registered" in by_id["organisation_units"].message
    assert by_id["connection"].status == "fail"
    assert by_id["dhis2_metadata"].status == "skip"  # no connection to ask

    bad_uid = _spec(id="bad-uid", destination={"plugin": "dhis2", "settings": {**_SETTINGS, "data_set": "nope"}})
    by_id = _by_id(service.validate_pipeline(store.create_record(bad_uid)))
    assert by_id["destination"].status == "fail" and "data_set must be a DHIS2 UID" in by_id["destination"].message


def test_validation_reports_dhis2_period_and_assignment(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch, period_type="Daily", assigned=[_OU[0]])
    message = _by_id(service.validate_pipeline(store.create_record(_spec())))["dhis2_metadata"].message
    assert "period type is 'Daily', the export is Monthly" in message
    assert "1 of 2 organisation units are not assigned" in message


def test_validation_rejects_wrong_connection_or_non_polygon_collection(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(dhis2_destination, "feature_binding", lambda collection_id: _binding(geometry="Point"))
    check = _by_id(service.validate_pipeline(store.create_record(_spec())))["organisation_units"]
    assert check.status == "fail" and "Polygon" in check.message

    monkeypatch.setattr(dhis2_destination, "feature_binding", lambda collection_id: _binding(connection="another"))
    check = _by_id(service.validate_pipeline(store.create_record(_spec(id="other"))))["organisation_units"]
    assert check.status == "fail" and "destination uses 'local'" in check.message

    monkeypatch.setattr(dhis2_destination, "feature_binding", lambda collection_id: _binding(connection=None))
    check = _by_id(service.validate_pipeline(store.create_record(_spec(id="overture"))))["organisation_units"]
    assert check.status == "fail" and "was not fetched from DHIS2" in check.message


def test_validation_does_not_claim_to_reuse_a_disabled_schedule(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    api_config._cache["scheduler"] = {  # type: ignore[index]
        "enabled": False,
        "dataset_sync": [{"dataset_id": "rain_monthly", "cron": "0 6 * * *"}],
    }
    check = _by_id(service.validate_pipeline(store.create_record(_spec())))["schedule"]
    assert check.status == "skip" and "disabled" in check.message and "0 6 * * *" in check.message


def test_validation_names_the_source_schedule_and_the_delivery_policy(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    api_config._cache["scheduler"] = {  # type: ignore[index]
        "enabled": True,
        "dataset_sync": [{"dataset_id": "rain_monthly", "cron": "0 6 * * *"}],
    }
    result = service.validate_pipeline(store.create_record(_spec(delivery={"mode": "paused", "policy": "manual"})))
    by_id = _by_id(result)
    assert result.valid
    assert by_id["schedule"].status == "pass" and "0 6 * * *" in by_id["schedule"].message
    assert by_id["delivery"].message.startswith("paused") and "by hand" in by_id["delivery"].message


def test_validation_fails_when_declared_dhis2_dataset_cannot_be_read(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.dhis2 as dhis2_module

    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: (_ for _ in ()).throw(OSError("down")))
    check = _by_id(service.validate_pipeline(store.create_record(_spec())))["dhis2_metadata"]
    assert check.status == "fail" and "could not read" in check.message


def test_validation_refuses_a_cadence_dhis2_cannot_take(
    instance: None, monkeypatch: pytest.MonkeyPatch, datasets: list[Any]
) -> None:
    datasets[0] = _dataset(period_type="dekadal")
    _dhis2(monkeypatch)
    result = service.validate_pipeline(store.create_record(_spec()))
    assert _by_id(result)["cadence"].status == "fail" and _by_id(result)["destination"].status == "fail"
    assert result.export is None and not result.valid


def test_live_delivery_needs_a_dry_run_of_the_current_bindings(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    record = store.create_record(_spec(delivery={"mode": "live"}))
    assert _by_id(service.validate_pipeline(record))["delivery"].status == "fail"
    record.dry_run = DryRunResult(ran_at="x", start="a", end="b", bindings="other", report={"outcome": "dry_run"})
    check = _by_id(service.validate_pipeline(record))["delivery"]
    assert check.status == "fail" and "different bindings" in check.message
    record.dry_run.bindings = service.bindings_digest(record.spec, workflow_binding.bind(record.spec.workflow))
    assert _by_id(service.validate_pipeline(record))["delivery"].status == "pass"


class _DropPlugin(BaseExportPlugin):
    """A destination that is not DHIS2 and cannot dry-run: a file drop with one setting."""

    id = "drop"
    format = "CSV"
    extension = ".csv"
    media_type = "text/csv"

    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        return mapping

    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        return RenderedExport(content=b"a,b\n1,2\n", record_count=1)

    def destination_settings_schema(self) -> dict[str, Any] | None:
        return {"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]}

    def pipeline_settings_form(self) -> PipelineSettingsForm | None:
        return PipelineSettingsForm(
            title="Secure file drop",
            description="A deliberately different destination form used to exercise the public plugin contract.",
            sections=(SettingsFormSection(title="Storage", fields=("folder",)),),
            fields=(SettingsFormField(name="folder", label="Destination folder"),),
        )

    def export_definition(self, settings: dict[str, Any], context: DestinationContext) -> dict[str, Any]:
        if not settings.get("folder"):
            raise ValueError("folder is required")
        return {"dataset": context.datasets[0].id}

    def check_destination(self, settings: dict[str, Any], context: DestinationContext) -> list[DestinationCheck]:
        return [DestinationCheck(id="folder", status="pass", message=f"writes to {settings['folder']}")]


def _only_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    import open_climate_service.exports.registry as registry
    import open_climate_service.exports.service as export_service

    plugin = _DropPlugin()
    monkeypatch.setattr(registry, "load_export_plugins", lambda: {"drop": plugin})
    monkeypatch.setattr(export_service, "load_export_plugins", lambda: {"drop": plugin})


def test_approval_digest_includes_plugin_identity_and_resolved_export(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.registry as registry
    import open_climate_service.exports.service as export_service

    plugin = _DropPlugin()
    plugin.version = "1"
    monkeypatch.setattr(service, "destination_plugins", lambda: {"drop": plugin})
    monkeypatch.setattr(registry, "load_export_plugins", lambda: {"drop": plugin})
    monkeypatch.setattr(export_service, "load_export_plugins", lambda: {"drop": plugin})
    spec = _spec(destination={"plugin": "drop", "settings": {"folder": "/srv/out"}})
    bound = workflow_binding.bind(spec.workflow)
    original = service.bindings_digest(spec, bound)
    plugin.version = "2"
    assert service.bindings_digest(spec, bound) != original
    plugin.version = "1"
    monkeypatch.setattr(plugin, "export_definition", lambda settings, context: {"dataset": "different"})
    assert service.bindings_digest(spec, bound) != original


def test_form_parsing_preserves_enum_and_row_scalar_types() -> None:
    arguments = forms.parameter_fields([{"name": "level", "schema": {"type": "integer", "enum": [1, 2]}}])
    assert forms.read_arguments({"arg.level": "2"}, arguments) == {"level": 2}
    settings = forms.settings_fields(
        {
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"factor": {"type": "number"}, "enabled": {"type": "boolean"}},
                    },
                }
            },
        }
    )
    assert forms.read_settings({"dest.series.0.factor": "1.5", "dest.series.0.enabled": "true"}, settings) == {
        "series": [{"factor": 1.5, "enabled": True}]
    }


def test_pipeline_settings_metadata_orders_sections_and_safely_falls_back() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["append", "replace"]},
            "notes": {"type": "string"},
            "advanced": {"type": "object"},
        },
        "required": ["mode"],
    }
    presentation = PipelineSettingsForm(
        title="Warehouse",
        sections=(SettingsFormSection(title="Description", fields=("notes", "missing")),),
        fields=(
            SettingsFormField(name="notes", label="Notes", widget="textarea"),
            SettingsFormField(name="mode", widget="unsupported-widget"),
        ),
    )

    view = forms.settings_form(schema, presentation, plugin_id="warehouse")

    assert view["title"] == "Warehouse"
    assert [section["title"] for section in view["sections"]] == ["Description", "Additional settings"]
    assert [field["name"] for section in view["sections"] for field in section["fields"]] == [
        "notes",
        "mode",
        "advanced",
    ]
    fields = {field["name"]: field for section in view["sections"] for field in section["fields"]}
    assert fields["notes"]["title"] == "Notes" and fields["notes"]["widget"] == "textarea"
    assert fields["mode"]["widget"] == "select", "unknown hints fall back to the schema-derived control"
    assert fields["advanced"]["widget"] == "json"


def test_schema_only_destination_gets_sensible_form_defaults() -> None:
    view = forms.settings_form(
        {"type": "object", "properties": {"folder": {"type": "string", "title": "Folder"}}},
        None,
        plugin_id="drop",
    )

    assert view["title"] == "drop" and view["description"] == ""
    assert view["sections"][0]["title"] == ""
    assert view["sections"][0]["fields"][0]["title"] == "Folder"


def test_an_incompatible_text_hint_keeps_a_boolean_checkbox() -> None:
    fields = forms.settings_fields(
        {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
        PipelineSettingsForm(
            title="Destination",
            fields=(SettingsFormField(name="enabled", widget="text"),),
        ),
    )

    assert fields[0]["widget"] == "checkbox"


def test_dhis2_destination_rejects_the_removed_variable_selector() -> None:
    legacy = {**_SETTINGS, "series": [{"data_element": _ELEMENT, "variable": "tp"}]}

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        dhis2_destination.parse_settings(legacy)


def test_deprecated_plugin_settings_are_not_rendered() -> None:
    fields = forms.settings_fields(
        {
            "type": "object",
            "properties": {
                "current": {"type": "string"},
                "legacy": {"type": "string", "deprecated": True},
            },
        }
    )

    assert [field["name"] for field in fields] == ["current"]


def test_pipeline_settings_form_round_trips_scalar_enum_json_and_repeatable_rows() -> None:
    fields = forms.settings_fields(
        {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "mode": {"type": "string", "enum": ["append", "replace"]},
                "options": {"type": "object"},
                "series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "factor": {"type": "number"}},
                    },
                },
            },
        }
    )
    submitted = forms.read_settings(
        {
            "dest.enabled.__present": "true",
            "dest.mode": "replace",
            "dest.options": '{"precision": 2}',
            "dest.series.0.name": "rain",
            "dest.series.0.factor": "1.5",
            "dest.series.1.name": "",
            "dest.series.1.factor": "",
        },
        fields,
    )

    assert submitted == {
        "enabled": False,
        "mode": "replace",
        "options": {"precision": 2},
        "series": [{"name": "rain", "factor": 1.5}],
    }
    by_name = {field["name"]: field for field in fields}
    assert forms.setting_value(submitted, by_name["enabled"]) == "false"
    assert forms.setting_value(submitted, by_name["options"]) == '{"precision": 2}'
    assert forms.setting_rows(submitted, by_name["series"])[0] == {
        "values": {"name": "rain", "factor": 1.5},
        "existing": True,
    }


def test_pipeline_settings_form_preserves_an_existing_false_only_row() -> None:
    fields = forms.settings_fields(
        {
            "type": "object",
            "properties": {
                "switches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                    },
                }
            },
        }
    )
    rendered = forms.setting_rows({"switches": [{"enabled": False}]}, fields[0])

    submitted = forms.read_settings(
        {
            "dest.switches.0.__present": "true",
            "dest.switches.0.enabled.__present": "true",
            "dest.switches.1.enabled.__present": "true",
        },
        fields,
    )

    assert rendered == [
        {"values": {"enabled": False}, "existing": True},
        {"values": {}, "existing": False},
    ]
    assert submitted == {"switches": [{"enabled": False}]}


def test_boolean_enums_use_the_same_text_as_their_select_options() -> None:
    setting = forms.settings_fields(
        {"type": "object", "properties": {"enabled": {"type": "boolean", "enum": [True, False]}}}
    )[0]
    argument = forms.parameter_fields([{"name": "enabled", "schema": {"type": "boolean", "enum": [True, False]}}])[0]

    assert setting["widget"] == "select"
    assert forms.setting_value({"enabled": False}, setting) == "False"
    assert forms.argument_value({"enabled": False}, argument) == "False"


def test_an_empty_required_enum_does_not_populate_a_spare_row() -> None:
    fields = forms.settings_fields(
        {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"mode": {"type": "string", "enum": ["append", "replace"]}},
                        "required": ["mode"],
                    },
                }
            },
        }
    )

    assert forms.read_settings({"dest.rules.0.mode": ""}, fields) == {}


def test_the_core_needs_no_dhis2_identifiers(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _only_drop(monkeypatch)
    spec = _spec(destination={"plugin": "drop", "settings": {"folder": "/srv/out"}})
    result = service.validate_pipeline(store.create_record(spec))
    assert result.valid, result.checks
    assert [check.id for check in result.checks] == [
        "workflow",
        "inputs",
        "destination",
        "folder",
        "schedule",
        "configuration",
        "delivery",
    ]
    assert result.export == {"dataset": "rain_monthly", "id": "rain-to-dhis2", "plugin": "drop"}
    live = store.create_record(_spec(id="drop-live", delivery={"mode": "live"}, destination=spec.destination))
    assert "renders only" in _by_id(service.validate_pipeline(live))["delivery"].message

    source = Path(service.__file__).parent
    core = ("schemas.py", "service.py", "runner.py", "workflow.py", "store.py", "routes.py", "forms.py", "exports.py")
    for module in core:
        text = (source / module).read_text(encoding="utf-8")
        assert "dhis2" not in text.lower(), module
    for template in ("pipeline_page.html", "pipelines_page.html"):
        assert "dhis2" not in (source.parent / "templates" / template).read_text(encoding="utf-8").lower(), template


def test_destination_settings_are_validated_by_the_plugin_schema(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _only_drop(monkeypatch)
    invalid = _spec(destination={"plugin": "drop", "settings": {"folder": 42}})

    check = _by_id(service.validate_pipeline(store.create_record(invalid)))["destination"]

    assert check.status == "fail"
    assert "destination setting folder" in check.message and "string" in check.message


# --- dry run ---------------------------------------------------------------------------------


def _envelope(frame: pd.DataFrame) -> Any:
    from open_climate_service.openeo.execution import SaveResultEnvelope

    envelope = SaveResultEnvelope(frame, "DHIS2JSON", {"export": "rain-to-dhis2"})
    envelope.provenance = {"sources": [{"collection_id": "rain_monthly"}], "spatial_aggregations": ["mean"]}
    return envelope


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geometry": _OU * 2,
            "t": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"]),
            "tp": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Synchronous graph execution returning a monthly frame; records the graphs it ran."""
    from open_climate_service.openeo import execution

    graphs: list[dict[str, Any]] = []

    def run(process: dict[str, Any]) -> Any:
        graphs.append(process)
        return _envelope(_frame())

    monkeypatch.setattr(execution, "run_process_graph", run)
    return graphs


def test_dry_run_executes_the_run_graph_and_folds_the_report(
    instance: None, monkeypatch: pytest.MonkeyPatch, executed: list[dict[str, Any]]
) -> None:
    client = _dhis2(monkeypatch)
    record = store.create_record(_spec(), service.validate_pipeline(store.create_record(_spec(id="probe"))))
    result = service.dry_run_pipeline(record, "2025-01-01", "2025-02-28")
    assert result.error is None, result.error
    assert result.values == 4 and result.periods == ["202501", "202502"]
    assert result.sample[0]["orgUnit"] == _OU[0] and result.sample[0]["dataElement"] == _ELEMENT
    assert client.posts[0]["params"] == {"importStrategy": "CREATE_AND_UPDATE", "dryRun": "true"}
    assert result.report is not None and result.report["outcome"] == "dry_run" and result.passed
    graph = executed[0]["process_graph"]
    assert graph["workflow__load"]["arguments"]["temporal_extent"] == ["2025-01-01", "2025-02-28"]
    assert graph["save"]["arguments"]["data"] == {"from_node": "workflow__zonal"}
    assert result.bindings == service.bindings_digest(record.spec, workflow_binding.bind(record.spec.workflow))


def test_dry_run_applies_the_export_data_cadence_gate(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import execution

    daily = pd.DataFrame(
        {"geometry": [_OU[0], _OU[0]], "t": pd.to_datetime(["2025-01-01", "2025-01-02"]), "tp": [1.5, 2.0]}
    )
    monkeypatch.setattr(execution, "run_process_graph", lambda process: _envelope(daily))
    result = service.dry_run_pipeline(store.create_record(_spec()), "2025-01-01", "2025-01-31")
    assert result.error is not None and "spaced daily" in result.error and not result.passed


def test_dry_run_of_a_destination_without_dry_run_support_only_renders(
    instance: None, monkeypatch: pytest.MonkeyPatch, executed: list[dict[str, Any]]
) -> None:
    _only_drop(monkeypatch)
    record = store.create_record(_spec(destination={"plugin": "drop", "settings": {"folder": "/srv/out"}}))
    result = service.dry_run_pipeline(record, "2025-01-01", "2025-02-28")
    assert result.rendered_only and result.report is None and result.passed and result.values == 1
    assert executed[0]["process_graph"]["save"]["arguments"]["format"] == "CSV"


# --- pages and API ---------------------------------------------------------------------------


@pytest.fixture
def client(instance: None) -> TestClient:
    from open_climate_service.main import app

    return TestClient(app)


def test_repeatable_plugin_fields_render_stored_rows_and_an_empty_spare_row(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RowsPlugin(_DropPlugin):
        id = "rows"

        def destination_settings_schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "enum": [True, False]},
                    "switches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"enabled": {"type": "boolean"}},
                        },
                    },
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"mode": {"type": "string", "enum": ["append", "replace"]}},
                            "required": ["mode"],
                        },
                    },
                },
            }

        def pipeline_settings_form(self) -> None:
            return None

    plugin = _RowsPlugin()
    monkeypatch.setattr(service, "destination_plugins", lambda: {"rows": plugin})
    record = store.create_record(
        _spec(
            id="row-form",
            destination={
                "plugin": "rows",
                "settings": {"enabled": False, "switches": [{"enabled": False}]},
            },
        )
    )

    page = client.get(f"/pipelines/{record.spec.id}/edit", headers={"Accept": "text/html"})

    assert page.status_code == 200
    assert '<option value="False" selected>False</option>' in page.text
    assert 'name="dest.switches.0.__present" value="true"' in page.text
    assert 'name="dest.switches.1.__present"' not in page.text
    assert 'name="dest.rules.0.mode">\n                        <option value="">(select)</option>' in page.text


def test_page_creates_validates_and_shows_a_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    page = client.get("/pipelines", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Configured pipelines" in page.text and "New pipeline" not in page.text
    create_page = client.get("/pipelines/new?dataset=rain_monthly", headers={"Accept": "text/html"})
    assert create_page.status_code == 200 and "New pipeline" in create_page.text
    assert "Configure DHIS2" in create_page.text
    assert "Target" in create_page.text and "Series mapping" in create_page.text
    assert '<select id="dest-connection"' in create_page.text
    assert 'name="dest.series.0.variable"' not in create_page.text
    for name in ("arg.dataset_id", "arg.geometries", "arg.method", "dest.connection", "dest.series.0.data_element"):
        assert f'name="{name}"' in create_page.text, name
    assert 'name="arg.temporal_extent"' in create_page.text and "Supplied by each run" in create_page.text
    assert '<option value="rain_monthly" selected>' in create_page.text
    created = client.post("/pipelines", data=_FORM, follow_redirects=False)
    assert created.status_code == 303, created.text[:800]
    assert created.headers["location"].endswith("/pipelines/rain-to-dhis2")
    stored = store.get_record("rain-to-dhis2")
    assert stored is not None and stored.spec == _spec()
    validated = client.post(
        "/pipelines/rain-to-dhis2/validate", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert validated.status_code == 303
    shown = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert shown.status_code == 200
    assert "dhis2_metadata" in shown.text and "every organisation unit assigned" in shown.text
    assert "aggregate_by_features" in shown.text and "Export definition" in shown.text
    assert '<span class="flow-label">Destination</span><strong>DHIS2</strong>' in shown.text
    assert "<dt>Destination</dt><dd>DHIS2</dd>" in shown.text
    listed = client.get("/pipelines", headers={"Accept": "application/json"}).json()
    assert [item["spec"]["id"] for item in listed["items"]] == ["rain-to-dhis2"]
    assert listed["items"][0]["validation"]["valid"] is True


def test_a_second_plugin_renders_its_own_pipeline_settings(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _DropPlugin()
    monkeypatch.setattr(service, "destination_plugins", lambda: {plugin.id: plugin})

    page = client.get("/pipelines/new", headers={"Accept": "text/html"})

    assert page.status_code == 200
    assert "Configure Secure file drop" in page.text
    assert "Storage" in page.text and "Destination folder" in page.text
    assert 'type="text"' in page.text and 'name="dest.folder"' in page.text
    assert "Configure DHIS2" not in page.text


def test_pipeline_list_never_exposes_a_pending_run_process(client: TestClient) -> None:
    record = store.create_record(_spec())
    record.runs.append(
        runner.PipelineRun(
            run_id="r1",
            job_id="j1",
            mode="dry_run",
            start="a",
            end="b",
            submitted_at="x",
            idempotency_key="k",
            process={"process_graph": {"secretly-large": {"process_id": "constant", "arguments": {}}}},
        )
    )
    store.update_record(record.spec.id, lambda current: setattr(current, "runs", record.runs))
    listed = client.get("/pipelines", headers={"Accept": "application/json"}).json()
    assert "process" not in listed["items"][0]["runs"][0]


def test_read_only_pipeline_pages_hide_create_and_edit_forms(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_record(_spec())
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)
    listing = client.get("/pipelines", headers={"Accept": "text/html"})
    assert 'href="/pipelines/new"' not in listing.text
    create = client.get("/pipelines/new", headers={"Accept": "text/html"})
    edit = client.get("/pipelines/rain-to-dhis2/edit", headers={"Accept": "text/html"})
    assert "read-only" in create.text and "<form" not in create.text
    assert "read-only" in edit.text and "<form" not in edit.text


def test_the_dataset_page_offers_to_create_a_pipeline() -> None:
    template = Path(service.__file__).parent.parent / "templates" / "dataset_page.html"
    assert "/pipelines/new?dataset=" in template.read_text(encoding="utf-8")


def test_detail_get_is_read_only_and_renders_stored_invalid_validation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = store.create_record(
        _spec(),
        ValidationResult(
            valid=False,
            checked_at="2026-10-06T10:00:00Z",
            checks=[Check(id="dhis2_metadata", status="fail", message="DHIS2 was unavailable")],
        ),
    )

    def refuse_validation(record: Any) -> Any:
        raise AssertionError("GET must not validate")

    monkeypatch.setattr(service, "validate_pipeline", refuse_validation)
    shown = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert shown.status_code == 200 and "DHIS2 was unavailable" in shown.text
    as_json = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "application/json"})
    assert as_json.status_code == 200 and as_json.json()["validation"]["valid"] is False
    assert store.get_record("rain-to-dhis2") == record


def test_page_keeps_the_draft_on_a_bad_form(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    response = client.post("/pipelines", data={**_FORM, "dest.series.0.data_element": "bad"})
    assert response.status_code == 400
    assert "data_element must be a DHIS2 UID" in response.text and 'value="rain-to-dhis2"' in response.text
    assert store.list_records() == []


def test_api_creates_dry_runs_and_switches_modes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, executed: list[dict[str, Any]]
) -> None:
    _dhis2(monkeypatch)
    created = client.post("/pipelines", json=_SPEC)
    assert created.status_code == 201, created.text
    assert created.json()["validation"]["export"]["id"] == "rain-to-dhis2" and created.json()["approved"] is False
    assert client.post("/pipelines", json=_SPEC).status_code == 409
    early = client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"})
    assert early.status_code == 409 and "dry run" in early.text
    ran = client.post("/pipelines/rain-to-dhis2/dry-run", json={"start": "2025-01-01", "end": "2025-02-28"})
    assert ran.status_code == 200 and ran.json()["values"] == 4 and ran.json()["report"]["outcome"] == "dry_run"
    live = client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"})
    assert live.status_code == 200 and live.json()["spec"]["delivery"]["mode"] == "live" and live.json()["approved"]
    safe = client.post("/pipelines/rain-to-dhis2/mode/dry_run", headers={"Accept": "application/json"})
    assert safe.status_code == 200 and safe.json()["spec"]["delivery"]["mode"] == "dry_run"
    paused = client.post("/pipelines/rain-to-dhis2/mode/paused", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["automation_since"] is None
    page = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Resume as dry run" in page.text
    assert client.get("/pipelines/missing", headers={"Accept": "application/json"}).status_code == 404


def test_failed_live_validation_does_not_persist_live_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, executed: list[dict[str, Any]]
) -> None:
    _dhis2(monkeypatch)
    assert client.post("/pipelines", json=_SPEC).status_code == 201
    assert client.post("/pipelines/rain-to-dhis2/dry-run", json={"start": "2025-01-01", "end": "2025-02-28"}).json()[
        "report"
    ]
    _dhis2(monkeypatch, period_type="Weekly")
    response = client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"})
    assert response.status_code == 409 and "period type is 'Weekly'" in response.text
    assert store.get_record("rain-to-dhis2").spec.delivery.mode == "dry_run"  # type: ignore[union-attr]


# --- save, edit, delete ----------------------------------------------------------------------


def _validated(spec: PipelineSpec | None = None) -> PipelineRecord:
    """Store a pipeline with the validation the real checks give it."""
    spec = spec or _spec()
    record = PipelineRecord(spec=spec, created_at="x")
    return store.create_record(spec, service.validate_pipeline(record))


def _approve(pipeline_id: str = "rain-to-dhis2", *, live: bool = False) -> PipelineRecord:
    def _apply(record: PipelineRecord) -> None:
        assert record.validation is not None
        record.dry_run = DryRunResult(
            ran_at="x", start="a", end="b", bindings=record.validation.bindings, report={"outcome": "dry_run"}
        )
        if live:
            store.replace_spec(
                record,
                record.spec.model_copy(update={"delivery": record.spec.delivery.model_copy(update={"mode": "live"})}),
                expected_revision=None,
            )
            store.install_validation(record, service.validate_pipeline(record), judged_revision=record.revision)

    record, _ = store.update_record(pipeline_id, _apply)
    return record


def test_a_validated_pipeline_owns_its_named_export_and_a_clash_is_refused(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.exports.service import resolve_named_export

    _dhis2(monkeypatch)
    with pytest.raises(ValueError, match="Unknown export"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    _validated()
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    assert resolved.references["dataset"] == "rain_monthly" and resolved.mapping["period_type"] == "monthly"

    def _invalidate(record: PipelineRecord) -> None:
        assert record.validation is not None
        record.validation.valid = False

    store.update_record("rain-to-dhis2", _invalidate)
    with pytest.raises(ValueError, match="Unknown export"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})

    store.delete_record("rain-to-dhis2")
    _validated()
    api_config._cache["exports"] = [  # type: ignore[index]
        {"id": "rain-to-dhis2", "plugin": "dhis2", "period_type": "monthly", "series": [{"data_element": _ELEMENT}]}
    ]
    with pytest.raises(ValueError, match="declared in the configuration and owned by the pipeline"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    check = _by_id(service.validate_pipeline(store.get_record("rain-to-dhis2")))["configuration"]  # type: ignore[arg-type]
    assert check.status == "fail" and "already configured" in check.message


def test_save_refuses_an_invalid_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import open_climate_service.exports.dhis2 as dhis2_module

    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: (_ for _ in ()).throw(OSError("down")))
    refused = client.post("/pipelines", json=_SPEC)
    assert refused.status_code == 422
    assert any(check["id"] == "dhis2_metadata" for check in refused.json()["detail"]["checks"]["checks"])
    page = client.post("/pipelines", data=_FORM)
    assert page.status_code == 400 and "not saved" in page.text and "could not read" in page.text
    assert 'value="rain-to-dhis2"' in page.text
    assert store.list_records() == []


def test_edit_keeps_the_approval_only_for_the_same_bindings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    _approve()
    form = client.get("/pipelines/rain-to-dhis2/edit", headers={"Accept": "text/html"})
    assert form.status_code == 200 and "Edit pipeline" in form.text and "readonly" in form.text
    assert 'action="/pipelines/rain-to-dhis2"' in form.text and 'name="revision" value="1"' in form.text

    renamed = client.post("/pipelines/rain-to-dhis2", json={**_SPEC, "name": "Renamed"})
    assert renamed.status_code == 200 and renamed.json()["approved"] is True

    import open_climate_service.exports.dhis2 as dhis2_module

    good = dhis2_module.get_connection
    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: (_ for _ in ()).throw(OSError("down")))
    summed = {
        **_SPEC,
        "workflow": {**_SPEC["workflow"], "arguments": {**_SPEC["workflow"]["arguments"], "method": "sum"}},
    }
    bad = client.post("/pipelines/rain-to-dhis2", json=summed)
    assert bad.status_code == 422 and "dhis2_metadata" in bad.text
    kept = store.get_record("rain-to-dhis2")
    assert kept is not None and kept.spec.workflow.arguments["method"] == "mean" and service.approved(kept)

    monkeypatch.setattr(dhis2_module, "get_connection", good)
    changed = client.post("/pipelines/rain-to-dhis2", json=summed)
    assert changed.status_code == 200 and changed.json()["dry_run"] is None and changed.json()["approved"] is False
    saved = store.get_record("rain-to-dhis2")
    assert saved is not None and saved.spec.workflow.arguments["method"] == "sum" and saved.revision == 3
    assert client.post("/pipelines/missing", json=_SPEC).status_code == 404


def test_a_stale_edit_is_refused_and_concurrent_run_state_is_kept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()

    def _run_lands(record: PipelineRecord) -> None:
        record.runs.insert(
            0,
            runner.PipelineRun(
                run_id="r1", job_id="j1", mode="dry_run", start="a", end="b", submitted_at="x", idempotency_key="k"
            ),
        )

    # A run hand-off lands between the edit form being read and saved: the edit keeps it.
    store.update_record("rain-to-dhis2", _run_lands)
    saved = client.post(
        "/pipelines/rain-to-dhis2", data={**_FORM, "name": "Renamed", "revision": "1"}, follow_redirects=False
    )
    assert saved.status_code == 303, saved.text[:600]
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.spec.name == "Renamed" and [run.run_id for run in record.runs] == ["r1"]

    stale = client.post("/pipelines/rain-to-dhis2", json={**_SPEC, "name": "Again", "revision": 1})
    assert stale.status_code == 409 and "changed since revision 1" in stale.text
    assert store.get_record("rain-to-dhis2").spec.name == "Renamed"  # type: ignore[union-attr]


def test_delete_from_api_and_from_the_page(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    _validated()
    assert client.delete("/pipelines/rain-to-dhis2").status_code == 204
    assert client.delete("/pipelines/rain-to-dhis2").status_code == 404
    assert store.get_record("rain-to-dhis2") is None

    _validated()
    unconfirmed = client.post("/pipelines/rain-to-dhis2/delete", data={}, follow_redirects=False)
    assert unconfirmed.status_code == 400 and store.get_record("rain-to-dhis2") is not None
    confirmed = client.post("/pipelines/rain-to-dhis2/delete", data={"confirm": "yes"}, follow_redirects=False)
    assert confirmed.status_code == 303 and confirmed.headers["location"].endswith("/pipelines")
    assert store.get_record("rain-to-dhis2") is None


# --- running ---------------------------------------------------------------------------------


class _FakeOpenEO:
    """A job service that records what it was asked to run and finishes nothing by itself."""

    def __init__(self) -> None:
        from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus

        self.jobs: dict[str, OpenEOJobRecord] = {}
        self.started: list[str] = []
        self.intents: list[Any] = []
        self.fail_create: BaseException | None = None
        self._record, self._status = OpenEOJobRecord, OpenEOJobStatus

    def create_triggered_job(
        self, body: Any, *, source_event_id: str, trigger_id: str, max_attempts: int = 1
    ) -> tuple[Any, bool]:
        from open_climate_service.openeo.jobs import triggered_job_id

        job_id = triggered_job_id(source_event_id, trigger_id)
        pipeline_id = trigger_id.removeprefix(runner.TRIGGER_PREFIX)
        # The run must already be stored, pending, under this very job id.
        self.intents.append(next(r for r in store.get_record(pipeline_id).runs if r.job_id == job_id))  # type: ignore[union-attr]
        if self.fail_create is not None:
            raise self.fail_create
        if job_id in self.jobs:
            return self.jobs[job_id], False
        job = self._record(
            id=job_id,
            title=body.title,
            process=body.process,
            status=self._status.CREATED,
            created=datetime.now(UTC),
            trigger_id=trigger_id,
            source_event_id=source_event_id,
            max_attempts=max_attempts,
        )
        self.jobs[job_id] = job
        return job, True

    def start_triggered_job(self, job_id: str) -> bool:
        if self.jobs[job_id].status != self._status.CREATED:
            return False
        self.started.append(job_id)
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": self._status.RUNNING})
        return True

    def drop(self, job_id: str) -> None:
        del self.jobs[job_id]

    def finish(self, job_id: str, status: str = "finished") -> Any:
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": self._status(status)})
        return self.jobs[job_id]


@pytest.fixture
def openeo(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenEO:
    import open_climate_service.openeo.jobs as jobs_module

    fake = _FakeOpenEO()
    monkeypatch.setattr(jobs_module, "get_openeo_job_service", lambda: fake)
    monkeypatch.setattr(jobs_module, "store_get_job", lambda job_id: fake.jobs.get(job_id))
    return fake


class _Deliveries(list[tuple[str, str, bool, str]]):
    """Submitted deliveries, and the outcome each delivery job reports (successful by default)."""

    def __init__(self) -> None:
        super().__init__()
        self.outcomes: dict[str, tuple[str, dict[str, Any] | None]] = {}

    def job(self, delivery_id: str) -> Any:
        index = int(delivery_id.removeprefix("delivery-")) - 1
        dry_run = self[index][2]
        default = ("successful", {"outcome": "dry_run" if dry_run else "success", "conflicts": []})
        status, result = self.outcomes.get(delivery_id, default)
        if status == "missing":
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Job '{delivery_id}' not found")
        from open_climate_service.jobs.models import JobStatus

        return SimpleNamespace(status=JobStatus(status), result=result)


@pytest.fixture
def deliveries(monkeypatch: pytest.MonkeyPatch) -> _Deliveries:
    import open_climate_service.exports.delivery as delivery_module
    from open_climate_service.jobs import service as job_service_module

    calls = _Deliveries()

    def fake_submit(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        for index, call in enumerate(calls):
            if call[3] == idempotency_key:
                return f"delivery-{index + 1}", True
        calls.append((export_id, source_job_id, dry_run, idempotency_key))
        return f"delivery-{len(calls)}", False

    monkeypatch.setattr(delivery_module, "submit_delivery", fake_submit)
    monkeypatch.setattr(job_service_module, "get_job_service", lambda: SimpleNamespace(get_job_or_404=calls.job))
    return calls


def test_run_once_stores_its_intent_first_and_delivers_once(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    live_too_early = client.post(
        "/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31", "mode": "live"}
    )
    assert live_too_early.status_code == 409 and "live" in live_too_early.text
    assert client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01"}).status_code == 400

    submitted = client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert submitted.status_code == 202, submitted.text
    run = submitted.json()
    assert run["mode"] == "dry_run" and run["submission"] == "submitted" and run["trigger"] == "manual"
    assert openeo.intents[0].submission == "pending" and openeo.intents[0].job_id == run["job_id"]
    assert openeo.started == [run["job_id"]]
    graph = openeo.jobs[run["job_id"]].process["process_graph"]
    assert graph["workflow__load"]["arguments"]["temporal_extent"] == ["2025-01-01", "2025-01-31"]
    assert graph["workflow__zonal"]["arguments"]["reducer"] == "mean"
    assert graph["save"]["arguments"]["options"] == {"export": "rain-to-dhis2"}
    assert run["arguments"]["method"] == "mean" and run["workflow_id"] == "aggregate_by_features"
    assert (run["start"], run["end"]) == ("2025-01-01", "2025-01-31")

    too_early = client.post(
        f"/pipelines/rain-to-dhis2/runs/{run['run_id']}/deliver", headers={"Accept": "application/json"}
    )
    assert too_early.status_code == 409 and "not finished" in too_early.text
    assert client.post("/pipelines/rain-to-dhis2/runs/nope/deliver").status_code == 404

    runner.on_job_finished(openeo.jobs[run["job_id"]])
    assert deliveries == []
    finished = openeo.finish(run["job_id"])
    runner.on_job_finished(finished)
    runner.on_job_finished(finished)  # a repeated completion event
    key = f"pipeline:rain-to-dhis2:pipeline-run:{run['run_id']}:dry_run"
    assert deliveries == [("rain-to-dhis2", run["job_id"], True, key)]
    stored = store.get_record("rain-to-dhis2")
    assert stored is not None and stored.runs[0].delivery == "submitted"
    assert stored.runs[0].delivery_status_url == "/exports/rain-to-dhis2/jobs/delivery-1"

    again = client.post(
        f"/pipelines/rain-to-dhis2/runs/{run['run_id']}/deliver", headers={"Accept": "application/json"}
    )
    assert again.status_code == 200 and again.json()["delivery_job_id"] == "delivery-1" and len(deliveries) == 1

    page = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Run once" in page.text and run["job_id"][:8] in page.text
    as_json = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "application/json"}).json()
    assert as_json["runs"][0]["job_status"] == "finished" and "process" not in as_json["runs"][0]


def test_a_failed_hand_off_is_recorded_and_recoverable(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import open_climate_service.exports.delivery as delivery_module

    def refuse(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        raise HTTPException(status_code=409, detail="No result for this job")

    monkeypatch.setattr(delivery_module, "submit_delivery", refuse)
    _dhis2(monkeypatch)
    _validated()
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    stored = store.get_record("rain-to-dhis2")
    assert (
        stored is not None and stored.runs[0].delivery == "failed" and stored.runs[0].error == "No result for this job"
    )

    monkeypatch.setattr(delivery_module, "submit_delivery", lambda *args: ("delivery-9", False))
    delivered = client.post(
        f"/pipelines/rain-to-dhis2/runs/{run.run_id}/deliver", headers={"Accept": "application/json"}
    )
    assert delivered.status_code == 200 and delivered.json()["delivery_job_id"] == "delivery-9"
    assert delivered.json()["error"] is None


def test_submission_retries_a_transient_failure_a_bounded_number_of_times(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    openeo.fail_create = OSError("job store unavailable")
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    assert run.submission == "pending" and run.submission_attempts == 1 and "unavailable" in (run.error or "")
    assert run.process is not None  # kept until a job holds it

    openeo.fail_create = None
    runner.recover_runs()
    recovered = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
    assert recovered.submission == "submitted" and recovered.error is None and recovered.process is None
    assert openeo.started == [run.job_id]

    other = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    openeo.fail_create = OSError("still unavailable")
    for _ in range(runner.MAX_SUBMISSION_ATTEMPTS + 2):
        runner.recover_runs()
    del other
    record = store.get_record("rain-to-dhis2")
    assert record is not None
    gave_up = record.runs[0]
    # Note the second run submitted normally before the outage; only the first was retried.
    assert gave_up.submission == "submitted"
    openeo.fail_create = OSError("down")
    flaky = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-03-01", end="2025-03-31"))
    for _ in range(runner.MAX_SUBMISSION_ATTEMPTS + 2):
        runner.recover_runs()
    stored = next(r for r in store.get_record("rain-to-dhis2").runs if r.run_id == flaky.run_id)  # type: ignore[union-attr]
    assert stored.submission == "failed" and stored.submission_attempts == runner.MAX_SUBMISSION_ATTEMPTS


def test_a_permanent_submission_failure_is_not_retried_and_can_run_again(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    openeo.fail_create = ValueError("process.process_graph must be an object")
    failed = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    assert failed.submission == "failed" and failed.submission_attempts == 1
    openeo.fail_create = None
    runner.recover_runs()
    assert store.get_record("rain-to-dhis2").runs[0].submission == "failed"  # type: ignore[union-attr]
    assert openeo.started == []
    retried = client.post(
        f"/pipelines/rain-to-dhis2/runs/{failed.run_id}/retry", headers={"Accept": "application/json"}
    )
    assert retried.status_code == 202
    body = retried.json()
    assert body["trigger"] == "retry" and body["retry_of"] == failed.run_id and body["submission"] == "submitted"
    assert (body["start"], body["end"]) == ("2025-01-01", "2025-01-31") and body["job_id"] != failed.job_id


def test_delivery_hand_off_recovers_transient_failures_and_stops_on_permanent_ones(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import open_climate_service.exports.delivery as delivery_module

    _dhis2(monkeypatch)
    _validated()
    attempts: list[str] = []
    outcome: dict[str, Any] = {"raise": OSError("lock timeout")}

    def submit(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        attempts.append(idempotency_key)
        if outcome["raise"] is not None:
            raise outcome["raise"]
        return "delivery-1", len(attempts) > 1

    monkeypatch.setattr(delivery_module, "submit_delivery", submit)
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    stored = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
    assert stored.delivery == "awaiting" and stored.delivery_attempts == 1 and "lock timeout" in (stored.error or "")

    outcome["raise"] = None
    runner.recover_runs()
    runner.recover_runs()
    stored = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
    assert stored.delivery == "submitted" and stored.delivery_job_id == "delivery-1"
    # Every attempt used the run's one key: the framework's reservation prevents a second send.
    assert set(attempts) == {run.idempotency_key} and len(attempts) == 2

    second = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    outcome["raise"] = HTTPException(status_code=409, detail="Export mapping or references changed")
    runner.on_job_finished(openeo.finish(second.job_id))
    for _ in range(3):
        runner.recover_runs()
    stored = next(r for r in store.get_record("rain-to-dhis2").runs if r.run_id == second.run_id)  # type: ignore[union-attr]
    assert stored.delivery == "failed" and stored.delivery_attempts == 1  # permanent: not retried

    third = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-03-01", end="2025-03-31"))
    outcome["raise"] = OSError("still locked")
    runner.on_job_finished(openeo.finish(third.job_id))
    for _ in range(runner.MAX_DELIVERY_ATTEMPTS + 2):
        runner.recover_runs()
    stored = next(r for r in store.get_record("rain-to-dhis2").runs if r.run_id == third.run_id)  # type: ignore[union-attr]
    assert stored.delivery == "failed" and stored.delivery_attempts == runner.MAX_DELIVERY_ATTEMPTS


def test_jobs_that_end_without_a_result_are_recorded_not_delivered(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    errored = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    gone = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    openeo.finish(errored.job_id, "error")
    openeo.drop(gone.job_id)
    runner.recover_runs()
    by_id = {run.run_id: run for run in store.get_record("rain-to-dhis2").runs}  # type: ignore[union-attr]
    assert by_id[errored.run_id].job_outcome == "error" and by_id[gone.run_id].job_outcome == "missing"
    assert deliveries == []


def test_a_missed_completion_is_delivered_by_recovery(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    openeo.finish(run.job_id)  # the process stopped before the finish listener ran
    runner.recover_runs()
    runner.recover_runs()
    assert [call[1] for call in deliveries] == [run.job_id]


def test_resuming_releases_eligible_held_deliveries(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    _approve(live=True)
    live = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31", mode="live"))
    html = {"Accept": "text/html"}
    assert client.post("/pipelines/rain-to-dhis2/mode/paused", headers=html, follow_redirects=False).status_code == 303
    dry = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    runner.on_job_finished(openeo.finish(live.job_id))
    runner.on_job_finished(openeo.finish(dry.job_id))
    runner.recover_runs()  # paused: recovery releases nothing
    assert deliveries == [] and all(run.delivery == "held" for run in store.get_record("rain-to-dhis2").runs)  # type: ignore[union-attr]

    # Resuming as a dry run releases the dry-run delivery; the live one waits for live.
    assert client.post("/pipelines/rain-to-dhis2/mode/dry_run", headers=html, follow_redirects=False).status_code == 303
    assert [(call[1], call[2]) for call in deliveries] == [(dry.job_id, True)]
    assert client.post("/pipelines/rain-to-dhis2/mode/live", headers=html, follow_redirects=False).status_code == 303
    assert [(call[1], call[2]) for call in deliveries] == [(dry.job_id, True), (live.job_id, False)]


def test_an_unconfirmed_delivery_can_be_sent_again_under_a_new_key(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    url = f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver"
    json_only = {"Accept": "application/json"}

    deliveries.outcomes["delivery-1"] = ("running", None)
    early = client.post(url, headers=json_only)
    assert early.status_code == 409 and "still running" in early.text and len(deliveries) == 1

    deliveries.outcomes["delivery-1"] = ("successful", {"outcome": "dry_run", "conflicts": []})
    confirmed = client.post(url, headers=json_only)
    assert confirmed.status_code == 409 and "settled" in confirmed.text and len(deliveries) == 1

    # An answer that never came: the destination may hold part of it. DHIS2 upserts, so a repeat is safe.
    other = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    runner.on_job_finished(openeo.finish(other.job_id))
    deliveries.outcomes["delivery-2"] = ("successful", {"outcome": "unknown"})
    again = client.post(f"/pipelines/rain-to-dhis2/runs/{other.run_id}/redeliver", headers=json_only)
    assert again.status_code == 200 and again.json()["delivery_job_id"] == "delivery-3"
    assert deliveries[2][3] == f"{other.idempotency_key}:round-1" and again.json()["delivery_round"] == 1


def test_redelivery_is_refused_where_a_repeat_could_duplicate(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

    _dhis2(monkeypatch)
    _validated()
    monkeypatch.setattr(Dhis2ExportPlugin, "redelivery_safe", False)
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    deliveries.outcomes["delivery-1"] = ("failed", None)
    with pytest.raises(runner.RunRefused, match="does not declare that a repeated delivery is harmless"):
        runner.deliver_run("rain-to-dhis2", run.run_id, operator=True, redeliver=True)
    assert len(deliveries) == 1


def test_handed_off_is_not_delivered_until_the_delivery_job_confirms(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    outcomes = {
        "delivery-1": (("successful", {"outcome": "success", "conflicts": []}), "delivered"),
        "delivery-2": (("successful", {"outcome": "success", "conflicts": [{"object": "x"}]}), "partial"),
        "delivery-3": (("successful", {"outcome": "rejected"}), "rejected"),
        "delivery-4": (("successful", {"outcome": "unknown"}), "unknown"),
        "delivery-5": (("failed", None), "failed"),
        "delivery-6": (("running", None), None),
    }
    runs = []
    for index, (delivery_id, (job, _)) in enumerate(outcomes.items()):
        run = runner.run_pipeline(
            "rain-to-dhis2", runner.RunContext(start=f"2025-0{index + 1}-01", end=f"2025-0{index + 1}-28")
        )
        runner.on_job_finished(openeo.finish(run.job_id))
        deliveries.outcomes[delivery_id] = job
        runs.append(run.run_id)
    runner.recover_runs()
    by_id = {run.run_id: run for run in store.get_record("rain-to-dhis2").runs}  # type: ignore[union-attr]
    assert all(by_id[run_id].delivery == "submitted" for run_id in runs)
    assert [by_id[run_id].delivery_result for run_id in runs] == [expected for _, expected in outcomes.values()]
    # Only the confirmed one may be pruned.
    superseded: set[str] = set()
    assert [runner._settled(by_id[run_id], superseded) for run_id in runs] == [True, False, False, False, False, False]


def test_a_missing_delivery_record_is_unknown_and_resolved_by_an_operator(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

    _dhis2(monkeypatch)
    _validated()
    json_only = {"Accept": "application/json"}
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    deliveries.outcomes["delivery-1"] = ("missing", None)
    runner.recover_runs()
    runner.recover_runs()  # resolved once, not polled or re-sent on every pass
    stored = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
    assert stored.delivery_result == "unknown" and "no longer exists" in (stored.error or "")
    assert len(deliveries) == 1 and not runner._settled(stored, set())

    # Where a repeat is safe, the operator may send it again: a new round under a new key.
    again = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert again.status_code == 200 and deliveries[1][3] == f"{run.idempotency_key}:round-1"

    # Where it is not, the operator checks the destination and acknowledges; nothing is sent.
    other = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    runner.on_job_finished(openeo.finish(other.job_id))
    deliveries.outcomes["delivery-3"] = ("missing", None)
    monkeypatch.setattr(Dhis2ExportPlugin, "redelivery_safe", False)
    refused = client.post(f"/pipelines/rain-to-dhis2/runs/{other.run_id}/redeliver", headers=json_only)
    assert refused.status_code == 409 and "acknowledge" in refused.text and len(deliveries) == 3
    acknowledged = client.post(f"/pipelines/rain-to-dhis2/runs/{other.run_id}/acknowledge", headers=json_only)
    assert acknowledged.status_code == 200 and acknowledged.json()["delivery_result"] == "acknowledged"
    stored = next(r for r in store.get_record("rain-to-dhis2").runs if r.run_id == other.run_id)  # type: ignore[union-attr]
    assert runner._settled(stored, set()) and len(deliveries) == 3


def test_a_refused_redelivery_leaves_the_earlier_round_unchanged(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_climate_service.exports.delivery as delivery_module
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

    _dhis2(monkeypatch)
    _validated()
    _approve(live=True)
    json_only, html = {"Accept": "application/json"}, {"Accept": "text/html"}
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31", mode="live"))
    runner.on_job_finished(openeo.finish(run.job_id))
    deliveries.outcomes["delivery-1"] = ("failed", None)
    runner.recover_runs()

    def delivery_state() -> tuple[Any, ...]:
        stored = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
        return (
            stored.delivery_round,
            stored.delivery,
            stored.delivery_result,
            stored.delivery_job_id,
            stored.delivery_status_url,
            stored.delivery_attempts,
        )

    before = delivery_state()
    assert before == (0, "submitted", "failed", "delivery-1", "/exports/rain-to-dhis2/jobs/delivery-1", 1)

    # Paused: a live delivery may not go out, so sending again is refused before anything changes.
    assert client.post("/pipelines/rain-to-dhis2/mode/paused", headers=html, follow_redirects=False).status_code == 303
    paused = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert paused.status_code == 409 and "make it live" in paused.text
    assert delivery_state() == before and len(deliveries) == 1

    # A destination that does not declare a repeat harmless: refused, unchanged.
    assert client.post("/pipelines/rain-to-dhis2/mode/live", headers=html, follow_redirects=False).status_code == 303
    monkeypatch.setattr(Dhis2ExportPlugin, "redelivery_safe", False)
    unsafe = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert unsafe.status_code == 409 and delivery_state() == before
    monkeypatch.setattr(Dhis2ExportPlugin, "redelivery_safe", True)

    # The source job is gone: refused, unchanged.
    finished = openeo.jobs[run.job_id]
    openeo.drop(run.job_id)
    gone = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert gone.status_code == 404 and delivery_state() == before
    openeo.jobs[run.job_id] = finished

    # Sending fails after the checks: the earlier round stays, and the retry reuses the same new key.
    real_submit = delivery_module.submit_delivery
    keys: list[str] = []

    def flaky(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        keys.append(idempotency_key)
        raise OSError("lock timeout")

    monkeypatch.setattr(delivery_module, "submit_delivery", flaky)
    failed = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert failed.status_code == 503 and "lock timeout" in failed.text
    assert delivery_state() == before
    monkeypatch.setattr(delivery_module, "submit_delivery", real_submit)
    ok = client.post(f"/pipelines/rain-to-dhis2/runs/{run.run_id}/redeliver", headers=json_only)
    assert ok.status_code == 200 and ok.json()["delivery_round"] == 1
    assert keys == [f"{run.idempotency_key}:round-1"] and deliveries[-1][3] == keys[0]


def test_the_skipped_list_is_bounded_and_says_what_it_dropped(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    monkeypatch.setattr(runner, "_SKIPPED_KEPT", 2)
    html = {"Accept": "text/html"}
    assert client.post("/pipelines/rain-to-dhis2/mode/paused", headers=html, follow_redirects=False).status_code == 303
    start = datetime.now(UTC)
    for n in range(4):
        runner.consume([_update_event(f"evt-{n}", when=start + timedelta(seconds=n))])
    events = store.get_record("rain-to-dhis2").events  # type: ignore[union-attr]
    assert [item.event_id for item in events.skipped] == ["evt-3", "evt-2"]
    assert events.skipped_discarded == 2
    assert events.skipped_discarded_through == (start + timedelta(seconds=1)).isoformat()
    assert client.post("/pipelines/rain-to-dhis2/mode/dry_run", headers=html, follow_redirects=False).status_code == 303
    resumed = store.get_record("rain-to-dhis2").events  # type: ignore[union-attr]
    assert resumed.skipped_discarded == 2
    assert resumed.skipped_discarded_through == (start + timedelta(seconds=1)).isoformat()
    page = client.get("/pipelines/rain-to-dhis2", headers=html)
    assert "2 older entries were discarded" in page.text and "not an audit trail" in page.text


@pytest.mark.parametrize(
    ("status", "permanent"),
    [
        (400, True),
        (404, True),
        (409, True),
        (422, True),
        (408, False),
        (425, False),
        (429, False),
        (500, False),
        (503, False),
    ],
)
def test_only_refusals_are_permanent(status: int, permanent: bool) -> None:
    from fastapi import HTTPException

    assert runner._is_permanent(HTTPException(status_code=status, detail="x")) is permanent


def test_a_busy_result_or_export_is_retried_not_given_up(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.delivery as delivery_module
    from open_climate_service.exports.retention import ResourceBusy

    assert not runner._is_permanent(ResourceBusy("Source job results are currently in use"))
    _dhis2(monkeypatch)
    _validated()
    busy = {"on": True}

    def submit(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        if busy["on"]:
            raise ResourceBusy("Source job results are currently in use")
        return "delivery-1", False

    monkeypatch.setattr(delivery_module, "submit_delivery", submit)
    run = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    runner.on_job_finished(openeo.finish(run.job_id))
    assert store.get_record("rain-to-dhis2").runs[0].delivery == "awaiting"  # type: ignore[union-attr]
    busy["on"] = False
    runner.recover_runs()
    assert store.get_record("rain-to-dhis2").runs[0].delivery == "submitted"  # type: ignore[union-attr]


def test_run_needs_a_valid_pipeline_with_unchanged_bindings(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_record(
        _spec(), ValidationResult(valid=False, checked_at="x", checks=[Check(id="x", status="fail", message="no")])
    )
    refused = client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert refused.status_code == 409 and "validate" in refused.text and openeo.started == []

    store.delete_record("rain-to-dhis2")
    _dhis2(monkeypatch)
    _validated()
    builtin = workflow_store.get_workflow("aggregate_by_features")
    assert builtin is not None
    redefined = {**builtin.model_dump(), "description": "same interface, different graph"}
    redefined["process_graph"] = {**builtin.process_graph, "extra": {"process_id": "absolute", "arguments": {"x": 1}}}
    # The workflow definition the pipeline validated changes underneath it.
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [redefined])
    stale = client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert stale.status_code == 409 and "changed since the pipeline was validated" in stale.text


def test_live_runs_and_deliveries_follow_the_mode(
    client: TestClient,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    _approve(live=True)
    live = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31", mode="live"))
    assert live.mode == "live"
    # Paused while the live job ran: the job finishes, its delivery is held.
    assert (
        client.post(
            "/pipelines/rain-to-dhis2/mode/paused", headers={"Accept": "text/html"}, follow_redirects=False
        ).status_code
        == 303
    )
    runner.on_job_finished(openeo.finish(live.job_id))
    held = store.get_record("rain-to-dhis2")
    assert held is not None and held.runs[0].delivery == "held" and deliveries == []
    refused = client.post(
        f"/pipelines/rain-to-dhis2/runs/{live.run_id}/deliver", headers={"Accept": "application/json"}
    )
    assert refused.status_code == 409 and "make it live" in refused.text

    # A paused pipeline still runs a dry run by hand, and an operator may deliver it.
    dry = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    runner.on_job_finished(openeo.finish(dry.job_id))
    assert store.get_record("rain-to-dhis2").runs[0].delivery == "held"  # type: ignore[union-attr]
    assert (
        client.post(
            f"/pipelines/rain-to-dhis2/runs/{dry.run_id}/deliver",
            headers={"Accept": "text/html"},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert deliveries[-1][2] is True

    with pytest.raises(runner.RunRefused, match="needs the pipeline to be live"):
        runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31", mode="live"))
    assert (
        client.post(
            "/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "text/html"}, follow_redirects=False
        ).status_code
        == 303
    )
    released = client.post(
        f"/pipelines/rain-to-dhis2/runs/{live.run_id}/deliver", headers={"Accept": "application/json"}
    )
    assert released.status_code == 200 and deliveries[-1][1:3] == (live.job_id, False)


def _update_event(event_id: str = "evt-1", *, dataset_id: str = "rain_monthly", when: datetime | None = None) -> Any:
    from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent

    return JobEvent(
        event_id=event_id,
        type=DATASET_UPDATED_EVENT_TYPE,
        source="ingestion",
        time=when or datetime.now(UTC) + timedelta(seconds=1),
        data={
            "dataset_id": dataset_id,
            "artifact_id": "artifact-7",
            "action": "append",
            "previous_end": "2025-01-31",
            "current_start": "2024-01-01",
            "current_end": "2025-02-28",
        },
    )


def test_a_dataset_update_runs_the_same_path_once(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    runner.consume([_update_event(when=datetime.now(UTC) - timedelta(days=1))])  # before the pipeline existed
    runner.consume([_update_event("evt-other", dataset_id="elsewhere")])
    assert openeo.started == []

    runner.consume([_update_event()])
    runner.consume([_update_event()])  # delivered twice
    record = store.get_record("rain-to-dhis2")
    assert record is not None and len(record.runs) == 1
    run = record.runs[0]
    assert run.trigger == "dataset_update" and run.event_id == "evt-1" and run.mode == "dry_run"
    assert (run.start, run.end) == ("2025-01-31", "2025-02-28")
    assert run.source["artifact_id"] == "artifact-7" and run.source["datasets"] == ["rain_monthly"]
    assert len(openeo.started) == 1

    # Switching mode moves the automation boundary: the old update is never replayed live.
    _approve(live=True)
    runner.consume([_update_event()])
    assert len(store.get_record("rain-to-dhis2").runs) == 1  # type: ignore[union-attr]


def test_paused_or_manual_pipelines_ignore_updates(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated(_spec(delivery={"policy": "manual"}))
    _validated(_spec(id="paused-one", delivery={"mode": "paused"}))
    runner.consume([_update_event()])
    assert openeo.started == [] and all(not record.runs for record in store.list_records())
    with pytest.raises(runner.RunRefused, match="does not run on updates"):
        runner.run_pipeline(
            "paused-one", runner.RunContext(start="a", end="b", trigger="dataset_update", event_id="evt-x")
        )


def _job_with(*events: Any, status: str = "successful", created: datetime | None = None) -> Any:
    """A native job record as the update log holds it."""
    from open_climate_service.jobs.models import JobStatus

    return SimpleNamespace(events=list(events), status=JobStatus(status), created_at=created or datetime.now(UTC))


def test_reconcile_finishes_what_a_restart_interrupted(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.jobs import store as job_store

    _dhis2(monkeypatch)
    _validated()
    # A run stored as pending whose job was never created: the process died in between.
    openeo.fail_create = OSError("killed")
    pending = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))
    openeo.fail_create = None
    # A finished job whose hand-off never happened, and an update nobody dispatched.
    done = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-02-01", end="2025-02-28"))
    openeo.finish(done.job_id)
    monkeypatch.setattr(job_store, "list_job_records", lambda: [_job_with(_update_event("evt-9"))])

    runner.reconcile()
    runner.reconcile()
    record = store.get_record("rain-to-dhis2")
    assert record is not None
    by_id = {run.run_id: run for run in record.runs}
    assert by_id[pending.run_id].submission == "submitted" and pending.job_id in openeo.started
    assert by_id[done.run_id].delivery == "submitted" and [call[1] for call in deliveries] == [done.job_id]
    assert [run.event_id for run in record.runs if run.trigger == "dataset_update"] == ["evt-9"]


def test_history_pruning_keeps_every_unresolved_run(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dhis2(monkeypatch)
    _validated()
    monkeypatch.setattr(runner, "_RUNS_KEPT", 3)
    still_running = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2024-01-01", end="2024-01-31"))
    errored = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2024-02-01", end="2024-02-29"))
    openeo.finish(errored.job_id, "error")
    gone = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2024-03-01", end="2024-03-31"))
    openeo.drop(gone.job_id)
    openeo.fail_create = ValueError("refused")
    refused = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2024-04-01", end="2024-04-30"))
    openeo.fail_create = None
    runner.recover_runs()
    for month in range(5, 11):
        run = runner.run_pipeline(
            "rain-to-dhis2", runner.RunContext(start=f"2024-{month:02d}-01", end=f"2024-{month:02d}-28")
        )
        runner.on_job_finished(openeo.finish(run.job_id))
        runner.recover_runs()  # the delivery job confirmed it: settled
    record = store.get_record("rain-to-dhis2")
    assert record is not None
    ids = {run.run_id for run in record.runs}
    # Three newest, plus every unresolved older run: running, errored, job deleted, refused.
    assert len(ids) == 7 and {still_running.run_id, errored.run_id, gone.run_id, refused.run_id} <= ids
    assert all(run.process is None for run in record.runs if run.submission == "submitted")

    # Running it again supersedes the errored run, which may then go.
    client.post(f"/pipelines/rain-to-dhis2/runs/{errored.run_id}/retry", headers={"Accept": "application/json"})
    runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2024-11-01", end="2024-11-30"))
    assert errored.run_id not in {run.run_id for run in store.get_record("rain-to-dhis2").runs}  # type: ignore[union-attr]
    # The kept running run is still found when its job finishes, and delivered.
    runner.on_job_finished(openeo.finish(still_running.job_id))
    assert deliveries[-1][1] == still_running.job_id


# --- dataset updates: semantics and the event checkpoint -------------------------------------


def test_an_update_names_the_interval_it_affected() -> None:
    appended = {"previous_end": "2025-01-31", "current_start": "2020-01-01", "current_end": "2025-02-28"}
    assert runner.update_interval(appended) == ("2025-01-31", "2025-02-28", "appended")
    rewritten = {"previous_end": None, "current_start": "2020-01-01", "current_end": "2025-02-28"}
    assert runner.update_interval(rewritten) == ("2020-01-01", "2025-02-28", "rewritten")
    assert runner.update_interval({"previous_end": None, "current_end": "2025-02-28"}) is None


def test_a_rewrite_runs_the_whole_coverage(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    event = _update_event()
    event.data.update({"previous_end": None, "action": "rematerialize"})
    runner.consume([event])
    run = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
    assert (run.start, run.end) == ("2024-01-01", "2025-02-28") and run.source["update"] == "rewritten"


def test_out_of_order_and_delayed_updates_are_each_run_once(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.jobs import store as job_store

    _dhis2(monkeypatch)
    _validated()
    now = datetime.now(UTC)
    earlier, later = (
        _update_event("evt-a", when=now + timedelta(seconds=5)),
        _update_event("evt-b", when=now + timedelta(seconds=9)),
    )
    runner.consume([later])  # the later update arrives first
    runner.consume([earlier])
    assert sorted(run.event_id or "" for run in store.get_record("rain-to-dhis2").runs) == ["evt-a", "evt-b"]  # type: ignore[union-attr]

    # A restart while an ingestion is still running: the checkpoint stays behind that job,
    # so its update, persisted after the restart, is still run.
    running_since = now + timedelta(seconds=20)
    monkeypatch.setattr(
        job_store,
        "list_job_records",
        lambda: [_job_with(earlier, later), _job_with(status="running", created=running_since)],
    )
    runner.reconcile_events()
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.events.through is not None
    # It never moves past an unfinished job, nor back.
    assert datetime.fromisoformat(record.events.through) <= running_since
    delayed = _update_event("evt-c", when=running_since + timedelta(minutes=30))
    runner.consume([delayed])
    runner.consume([delayed])
    assert sorted(run.event_id or "" for run in store.get_record("rain-to-dhis2").runs) == ["evt-a", "evt-b", "evt-c"]  # type: ignore[union-attr]
    assert len(openeo.started) == 3


def test_pruned_history_and_restarts_never_replay_an_update(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.jobs import store as job_store

    _dhis2(monkeypatch)
    _validated()
    monkeypatch.setattr(runner, "_RUNS_KEPT", 2)
    start = datetime.now(UTC)
    updates = [_update_event(f"evt-{n}", when=start + timedelta(seconds=n + 1)) for n in range(6)]
    for update in updates:
        runner.consume([update])
        run = store.get_record("rain-to-dhis2").runs[0]  # type: ignore[union-attr]
        runner.on_job_finished(openeo.finish(run.job_id))
        runner.recover_runs()
    record = store.get_record("rain-to-dhis2")
    assert record is not None and len(record.runs) == 2  # four delivered runs pruned
    assert set(record.events.handled) == {f"evt-{n}" for n in range(6)}

    monkeypatch.setattr(job_store, "list_job_records", lambda: [_job_with(*updates)])
    for _ in range(3):  # restarts
        runner.reconcile()
    assert len(openeo.started) == 6 and len(deliveries) == 6

    # Once the checkpoint passes them, the handled ids are forgotten: the checkpoint covers them.
    monkeypatch.setattr(runner, "_EVENT_MARGIN", timedelta(0))
    monkeypatch.setattr(runner, "utc_now", lambda: start + timedelta(minutes=1))
    runner.reconcile_events()
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.events.handled == {}
    runner.reconcile()
    assert len(openeo.started) == 6


def test_an_update_that_failed_to_dispatch_is_retried_and_holds_the_checkpoint(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.jobs import store as job_store

    _dhis2(monkeypatch)
    _validated()
    when = datetime.now(UTC) + timedelta(seconds=1)
    update = _update_event("evt-flaky", when=when)
    real = workflow_binding.build_process
    monkeypatch.setattr(workflow_binding, "build_process", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    runner.consume([update])
    assert store.get_record("rain-to-dhis2").runs == []  # type: ignore[union-attr]

    monkeypatch.setattr(job_store, "list_job_records", lambda: [_job_with(update)])
    monkeypatch.setattr(runner, "utc_now", lambda: when + timedelta(hours=2))
    runner.reconcile_events()
    record = store.get_record("rain-to-dhis2")
    assert record is not None and "evt-flaky" not in record.events.handled
    assert datetime.fromisoformat(record.events.through or "") < when  # held back by the undecided update

    monkeypatch.setattr(workflow_binding, "build_process", real)
    runner.reconcile_events()
    assert [run.event_id for run in store.get_record("rain-to-dhis2").runs] == ["evt-flaky"]  # type: ignore[union-attr]


def test_a_refused_update_is_decided_once_and_not_retried(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()

    def _invalidate(record: PipelineRecord) -> None:
        assert record.validation is not None
        record.validation.valid = False

    store.update_record("rain-to-dhis2", _invalidate)
    update = _update_event()
    runner.consume([update])
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.runs == [] and "evt-1" in record.events.handled
    _approve()  # valid again later: the refused update is not run after the fact

    def _revalidate(record: PipelineRecord) -> None:
        assert record.validation is not None
        record.validation.valid = True

    store.update_record("rain-to-dhis2", _revalidate)
    runner.consume([update])
    assert store.get_record("rain-to-dhis2").runs == []  # type: ignore[union-attr]


def test_changing_the_bindings_restarts_the_update_boundary(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch, datasets: list[Any]
) -> None:
    _dhis2(monkeypatch)
    datasets.append(_dataset(dataset_id="rain_weekly", period_type="weekly"))
    _validated()
    weekly_update = _update_event("evt-weekly", dataset_id="rain_weekly", when=datetime.now(UTC))
    runner.consume([weekly_update])  # not this pipeline's source yet
    arguments = {**_SPEC["workflow"]["arguments"], "dataset_id": "rain_weekly"}
    edited = client.post(
        "/pipelines/rain-to-dhis2",
        json={
            **_SPEC,
            "workflow": {**_SPEC["workflow"], "arguments": arguments},
            "destination": {
                "plugin": "dhis2",
                "settings": {"connection": "local", "series": [{"data_element": _ELEMENT}]},
            },
        },
    )
    assert edited.status_code == 200, edited.text
    runner.consume([weekly_update])  # arrived before the new bindings: not run with them
    assert store.get_record("rain-to-dhis2").runs == []  # type: ignore[union-attr]


def test_manual_and_update_runs_share_the_bindings_and_the_export(
    instance: None,
    openeo: _FakeOpenEO,
    deliveries: _Deliveries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.exports.service import resolve_named_export

    _dhis2(monkeypatch)
    record = _validated()
    manual = runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-31", end="2025-02-28"))
    runner.consume([_update_event()])
    automatic = next(r for r in store.get_record("rain-to-dhis2").runs if r.trigger == "dataset_update")  # type: ignore[union-attr]
    assert (manual.start, manual.end) == (automatic.start, automatic.end)
    assert openeo.jobs[manual.job_id].process == openeo.jobs[automatic.job_id].process
    assert manual.bindings == automatic.bindings == record.validation.bindings  # type: ignore[union-attr]
    assert manual.arguments == automatic.arguments
    for run in (manual, automatic):
        runner.on_job_finished(openeo.finish(run.job_id))
    assert {call[0] for call in deliveries} == {"rain-to-dhis2"}
    assert record.validation is not None and record.validation.export is not None
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    assert resolved.mapping["series"] == record.validation.export["series"]


def test_updates_while_paused_are_listed_not_run_on_resume(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    html = {"Accept": "text/html"}
    assert client.post("/pipelines/rain-to-dhis2/mode/paused", headers=html, follow_redirects=False).status_code == 303
    runner.consume([_update_event("evt-paused", when=datetime.now(UTC))])
    assert client.post("/pipelines/rain-to-dhis2/mode/dry_run", headers=html, follow_redirects=False).status_code == 303
    runner.consume([_update_event("evt-paused", when=datetime.now(UTC) - timedelta(seconds=1))])  # replayed late
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.runs == [] and openeo.started == []
    skipped = record.events.skipped
    assert [(item.event_id, item.reason) for item in skipped] == [("evt-paused", "the pipeline was paused")]
    assert (skipped[0].start, skipped[0].end) == ("2025-01-31", "2025-02-28")  # the interval to run by hand


def test_a_refused_update_is_listed_with_its_interval(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()

    def _edit_without_check(record: PipelineRecord) -> None:
        store.replace_spec(record, record.spec.model_copy(update={"name": "Renamed"}), expected_revision=None)

    store.update_record("rain-to-dhis2", _edit_without_check)  # validation is now stale
    runner.consume([_update_event()])
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.runs == [] and "evt-1" in record.events.handled
    assert record.events.skipped[0].reason == "the pipeline changed since it was last validated; re-check it"
    assert (record.events.skipped[0].start, record.events.skipped[0].end) == ("2025-01-31", "2025-02-28")


def test_an_unreachable_destination_does_not_stop_an_update_run(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.dhis2 as dhis2_module

    _dhis2(monkeypatch)
    _validated()
    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: (_ for _ in ()).throw(OSError("down")))
    runner.consume([_update_event()])
    # Starting a run never contacts the destination; its delivery retries transient failures.
    assert len(openeo.started) == 1 and store.get_record("rain-to-dhis2").events.skipped == []  # type: ignore[union-attr]


def test_dispatch_errors_are_bounded_or_skipped_at_once(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.jobs import store as job_store

    _dhis2(monkeypatch)
    _validated()
    transient = _update_event("evt-transient")
    monkeypatch.setattr(workflow_binding, "build_process", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    runner.consume([transient])
    monkeypatch.setattr(job_store, "list_job_records", lambda: [_job_with(transient)])
    for _ in range(runner.MAX_EVENT_ATTEMPTS + 2):
        runner.reconcile_events()
    record = store.get_record("rain-to-dhis2")
    assert record is not None and "evt-transient" in record.events.handled
    assert record.events.skipped[0].reason.startswith(
        f"could not be started after {runner.MAX_EVENT_ATTEMPTS} attempts"
    )

    permanent = _update_event("evt-permanent")
    monkeypatch.setattr(
        workflow_binding,
        "build_process",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("collection 'districts' is unregistered")),
    )
    runner.consume([permanent])
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.events.retries.get("evt-permanent") is None
    assert record.events.skipped[0].event_id == "evt-permanent" and "unregistered" in record.events.skipped[0].reason


def test_an_update_older_than_the_checkpoint_is_not_run_but_listed(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented limit: an update persisted after the checkpoint passed its time is not run."""

    _dhis2(monkeypatch)
    _validated()
    stamped = datetime.now(UTC) + timedelta(seconds=1)

    def _checkpoint_past(record: PipelineRecord) -> None:
        record.events.through = (stamped + timedelta(minutes=1)).isoformat()

    store.update_record("rain-to-dhis2", _checkpoint_past)
    runner.consume([_update_event("evt-late", when=stamped)])
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.runs == []
    assert record.events.skipped[0].event_id == "evt-late"
    assert record.events.skipped[0].reason == "it arrived after the checkpoint had passed its time"


def _openeo_owned_update(event_id: str, *, dataset_id: str = "rain_monthly") -> Any:
    """A full-rewrite update persisted on the openEO job that published the dataset."""
    from open_climate_service.ingestions.processes import FULL_REWRITE_EVENT_ACTION, dataset_update_event
    from open_climate_service.jobs.models import JobEvent
    from open_climate_service.openeo import jobs as openeo_jobs
    from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus

    draft = dataset_update_event(
        dataset_id=dataset_id,
        artifact_id="derived-artifact",
        action=FULL_REWRITE_EVENT_ACTION,
        previous_end=None,
        current_start="2024-01",
        current_end="2025-02",
        producer_type="openeo",
        producer_job_id="derived-job",
        output_snapshot_id="SNAPSHOT00000000000A",
    )
    event = JobEvent(event_id=event_id, time=datetime.now(UTC) + timedelta(seconds=1), **draft.model_dump())
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id="derived-job", status=OpenEOJobStatus.FINISHED, created=datetime.now(UTC), events=[event])
    )
    return event


def test_an_update_owned_by_an_openeo_job_runs_a_pipeline_like_any_other(
    instance: None, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    _validated()
    event = _openeo_owned_update("derived-job:publication:derived-artifact")

    # Startup reconciliation finds it through the shared event log, owned by an openEO job.
    runner.reconcile_events()
    runner.reconcile_events()
    runner.consume([event])  # and the live consumer, given it again, adds nothing
    record = store.get_record("rain-to-dhis2")
    assert record is not None
    [run] = record.runs
    assert run.trigger == "dataset_update" and run.event_id == event.event_id
    # A full rewrite covers the whole committed coverage.
    assert (run.start, run.end) == ("2024-01", "2025-02") and run.source["update"] == "rewritten"
    assert run.source["action"] == "full_rewrite"
    assert len(openeo.started) == 1

    # Nothing in the runner knows where an update came from.
    source = Path(runner.__file__).read_text(encoding="utf-8").lower()
    for origin in ("producer", "derived", "full_rewrite", "output_snapshot", "publication"):
        assert origin not in source, origin


def test_only_workflows_that_meet_the_contract_are_offered() -> None:
    ready = {
        record.id for record in workflow_store.list_workflows().processes if workflow_binding.pipeline_ready(record)
    }
    assert ready == {"aggregate_by_features"}


def test_an_update_driven_pipeline_must_bind_a_source_dataset(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    plain = {
        **_COMPOSED,
        "id": "plain",
        "parameters": [
            {**p, "schema": {"type": "string"}} if p["name"] == "observed" else p for p in _COMPOSED["parameters"]
        ],
    }
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [plain])
    spec = _spec(
        workflow={"id": "plain", "arguments": {"observed": "rain_monthly", "zones": {"from_features": "districts"}}}
    )
    check = _by_id(service.validate_pipeline(store.create_record(spec)))["inputs"]
    assert check.status == "fail" and "binds no dataset" in check.message
    manual = _spec(id="plain-manual", workflow=spec.workflow.model_dump(), delivery={"policy": "manual"})
    assert "binds no dataset" not in _by_id(service.validate_pipeline(store.create_record(manual)))["inputs"].message


class _WarehousePlugin(_DropPlugin):
    """A delivering destination whose target is not a DHIS2 connection."""

    id = "warehouse"
    format = "CSV"
    supports_delivery = True
    supports_dry_run = True

    def __init__(self) -> None:
        self.sent: list[tuple[Any, bool]] = []

    def destination_settings_schema(self) -> dict[str, Any] | None:
        return {"type": "object", "properties": {"database": {"type": "string"}}, "required": ["database"]}

    def pipeline_settings_form(self) -> PipelineSettingsForm | None:
        return None

    def export_definition(self, settings: dict[str, Any], context: DestinationContext) -> dict[str, Any]:
        return {"dataset": context.datasets[0].id, "connection": settings["database"]}

    def target_binding(self, references: dict[str, str]) -> dict[str, str] | None:
        return {"database": references["connection"]} if "connection" in references else None

    def delivery_target(self, binding: dict[str, str]) -> Any:
        return f"warehouse://{binding['database']}"

    def send(self, payload: bytes, target: Any, *, dry_run: bool = False, context: Any = None) -> Any:
        from open_climate_service.exports.report import ExportOutcome, ExportReport
        from open_climate_service.shared.time import utc_now

        self.sent.append((target, dry_run))
        return ExportReport(
            plugin_id=self.id,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )


def test_the_delivery_target_belongs_to_the_plugin(
    instance: None, monkeypatch: pytest.MonkeyPatch, executed: list[dict[str, Any]]
) -> None:
    import open_climate_service.exports.registry as registry
    import open_climate_service.exports.service as export_service
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
    from open_climate_service.exports.manifest import target_binding

    plugin = _WarehousePlugin()
    monkeypatch.setattr(registry, "load_export_plugins", lambda: {"warehouse": plugin})
    monkeypatch.setattr(export_service, "load_export_plugins", lambda: {"warehouse": plugin})
    record = store.create_record(_spec(destination={"plugin": "warehouse", "settings": {"database": "climate"}}))
    result = service.dry_run_pipeline(record, "2025-01-01", "2025-02-28")
    assert result.error is None and result.passed and not result.rendered_only
    assert plugin.sent == [("warehouse://climate", True)]
    # The manifest records the plugin's own binding, the same one the dry run sent to.
    assert target_binding(plugin, {"connection": "climate"}) == {"database": "climate"}

    # DHIS2 keeps the binding its saved manifests already carry.
    dhis2 = Dhis2ExportPlugin()
    binding = target_binding(dhis2, {"connection": "local"})
    assert binding == {"connection_id": "local", "url_sha256": hashlib.sha256(b"http://dhis2.test").hexdigest()}
    assert binding is not None and dhis2.delivery_target(binding) == "local"

    class _Broken(_DropPlugin):
        def target_binding(self, references: dict[str, str]) -> dict[str, str] | None:
            return {"port": 5432}  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="mapping of strings"):
        target_binding(_Broken(), {})


def test_a_check_for_an_older_revision_is_never_installed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports.service import resolve_named_export

    _dhis2(monkeypatch)
    _validated()
    real = service.validate_pipeline

    def validate_while_edited(record: PipelineRecord) -> ValidationResult:
        verdict = real(record)

        def _edit(current: PipelineRecord) -> None:
            store.replace_spec(
                current,
                current.spec.model_copy(
                    update={
                        "destination": current.spec.destination.model_copy(
                            update={"settings": {**_SETTINGS, "data_set": None}}
                        )
                    }
                ),
                expected_revision=None,
            )

        store.update_record("rain-to-dhis2", _edit)  # lands while the check runs
        return verdict

    monkeypatch.setattr(service, "validate_pipeline", validate_while_edited)
    response = client.post("/pipelines/rain-to-dhis2/validate", headers={"Accept": "application/json"})
    assert response.status_code == 409 and "changed while it was being checked" in response.text
    record = store.get_record("rain-to-dhis2")
    assert record is not None and record.revision == 2 and record.validation is not None
    assert record.validation.revision == 1
    # The revision-1 export is not published for the revision-2 spec, and runs are refused.
    with pytest.raises(ValueError, match="Unknown export"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    with pytest.raises(runner.RunRefused, match="changed since it was last validated"):
        runner.run_pipeline("rain-to-dhis2", runner.RunContext(start="2025-01-01", end="2025-01-31"))


# --- end to end ------------------------------------------------------------------------------


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}


def test_end_to_end_a_registered_workflow_dry_runs_and_delivers_live(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real job service and workflow engine, the built-in workflow, and the DHIS2 delivery path with a fake send."""
    from openeo_pg_parser_networkx.process_registry import Process

    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
    from open_climate_service.exports.report import ExportOutcome, ExportReport
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobStatus
    from open_climate_service.openeo import execution
    from open_climate_service.openeo import jobs as openeo_jobs
    from open_climate_service.openeo.schemas import OpenEOJobStatus
    from open_climate_service.shared.provenance import record_source
    from open_climate_service.shared.time import utc_now

    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")

    def load_collection(id: str | None = None, temporal_extent: Any = None, **_: Any) -> xr.DataArray:
        record_source(str(id), SimpleNamespace(artifact_id="artifact-1", source_dataset_id=id, path=None))
        ds = xr.Dataset(
            {"tp": (("t", "y", "x"), np.arange(2 * 4 * 5, dtype="float32").reshape(2, 4, 5))},
            coords={
                "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
                "y": [1.0, 2.0, 3.0, 4.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        )
        return ds["tp"]

    def load_features(id: str, version: str | None = None, **_: Any) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": _OU[0], "geometry": _box(0.5, 0.5, 2.5, 2.5)},
                {"type": "Feature", "id": _OU[1], "geometry": _box(3.5, 2.5, 5.5, 4.5)},
            ],
        }

    base = execution._build_process_registry()
    mocked = execution._RegistryOverlay(
        base,
        {
            "load_collection": Process(spec={}, implementation=load_collection),
            "load_features": Process(spec={}, implementation=load_features),
        },
    )
    monkeypatch.setattr(execution, "_build_process_registry", lambda: mocked)
    sent: list[dict[str, Any]] = []

    def send(
        self: Dhis2ExportPlugin, payload: bytes, target: Any, *, dry_run: bool = False, context: Any = None
    ) -> ExportReport:
        sent.append({"payload": json.loads(payload), "target": target, "dry_run": dry_run})
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=4,
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )

    monkeypatch.setattr(Dhis2ExportPlugin, "send", send)
    _dhis2(monkeypatch)
    job_service = openeo_jobs.OpenEOJobService()
    monkeypatch.setattr(openeo_jobs, "get_openeo_job_service", lambda: job_service)
    job_service.set_finished_listener(runner.on_job_finished)
    try:
        assert client.post("/pipelines", json=_SPEC).status_code == 201
        dry = client.post("/pipelines/rain-to-dhis2/dry-run", json={"start": "2025-01-01", "end": "2025-02-28"}).json()
        assert dry["error"] is None and dry["values"] == 4 and dry["periods"] == ["202501", "202502"], dry
        assert sent[0]["dry_run"] is True and sent[0]["target"] == "local"
        assert (
            client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"}).status_code == 200
        )

        run = client.post(
            "/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-02-28", "mode": "live"}
        )
        assert run.status_code == 202, run.text
        job_id = run.json()["job_id"]
        deadline = time.monotonic() + 30
        record = store.get_record("rain-to-dhis2")
        while time.monotonic() < deadline:
            record = store.get_record("rain-to-dhis2")
            job = openeo_jobs.store_get_job(job_id)
            assert job is None or job.status != OpenEOJobStatus.ERROR, job.error_message if job else None
            if record is not None and record.runs[0].delivery_job_id:
                break
            time.sleep(0.05)
        assert record is not None and record.runs[0].delivery == "submitted", record.runs[0] if record else None
        delivery_id = record.runs[0].delivery_job_id
        assert delivery_id is not None
        native = job_service_module.get_job_service()
        delivery = native.get_job_or_404(delivery_id)
        while time.monotonic() < deadline:
            delivery = native.get_job_or_404(delivery_id)
            if delivery.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
                break
            time.sleep(0.05)
        assert delivery.status == JobStatus.SUCCESSFUL, delivery.error
        live = sent[-1]
        assert live["dry_run"] is False and live["target"] == "local"
        assert {value["orgUnit"] for value in live["payload"]["dataValues"]} == set(_OU)
        assert {value["dataElement"] for value in live["payload"]["dataValues"]} == {_ELEMENT}
        assert len(live["payload"]["dataValues"]) == 4

        # A dataset update goes through the same runner, job service and delivery path.
        from open_climate_service.exports.delivery_input import lease_export_input

        runner.consume([_update_event("evt-e2e")])
        automatic = None
        while time.monotonic() < deadline:
            record = store.get_record("rain-to-dhis2")
            automatic = next((run for run in record.runs if run.event_id == "evt-e2e"), None) if record else None
            if automatic is not None and automatic.delivery_job_id:
                break
            time.sleep(0.05)
        assert automatic is not None and automatic.delivery == "submitted", automatic
        assert automatic.trigger == "dataset_update" and automatic.mode == "live"
        assert automatic.delivery_job_id is not None
        second = native.get_job_or_404(automatic.delivery_job_id)
        while time.monotonic() < deadline:
            second = native.get_job_or_404(automatic.delivery_job_id)
            if second.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
                break
            time.sleep(0.05)
        assert second.status == JobStatus.SUCCESSFUL, second.error
        assert sent[-1]["dry_run"] is False and sent[-1]["target"] == "local"
        with lease_export_input("rain-to-dhis2", job_id) as manual_saved:
            manual_manifest = manual_saved.manifest
        with lease_export_input("rain-to-dhis2", automatic.job_id) as automatic_saved:
            automatic_manifest = automatic_saved.manifest
        assert manual_manifest.mapping == automatic_manifest.mapping
        assert manual_manifest.references == automatic_manifest.references
        assert manual_manifest.target == automatic_manifest.target
    finally:
        job_service.shutdown()
        job_service_module.reset_job_service()


def test_end_to_end_an_anomaly_dataset_publishes_and_drives_a_pipeline_to_dhis2(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, datasets: list[Any]
) -> None:
    """A real anomaly workflow publishes a managed dataset; its update runs an aggregation pipeline to DHIS2.

    The real openEO job service, workflow engine, managed Icechunk writer, event consumer and
    delivery path. Only data loading, feature loading and DHIS2 itself are stubbed; no network.
    """
    from openeo_pg_parser_networkx.process_registry import Process

    from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset
    from open_climate_service.data_registry.services import datasets as registry
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
    from open_climate_service.exports.report import ExportOutcome, ExportReport
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs.models import JobStatus
    from open_climate_service.openeo import execution
    from open_climate_service.openeo import jobs as openeo_jobs
    from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
    from open_climate_service.shared.provenance import record_source
    from open_climate_service.shared.time import utc_now

    # A temporary template registry and store directory, as `managed_instance` provides.
    configs = tmp_path / "managed" / "plugins" / "rasters"
    configs.mkdir(parents=True)
    (configs / "source.yaml").write_text(
        "- id: chirps3_precipitation_daily\n"
        "  name: Total precipitation (CHIRPS3)\n"
        "  variable: precip\n"
        "  period_type: daily\n"
        "  units: mm/d\n"
        "  license: CC-BY-4.0\n"
        "  sync:\n"
        "    kind: temporal\n"
        "  ingestion:\n"
        "    plugin: open_climate_service.plugins.rasters.chirps3.CHIRPS3DailyPlugin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", configs)
    registry.reset_template_caches()
    (tmp_path / "managed" / "rasters").mkdir()
    monkeypatch.setattr(
        "open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path / "managed" / "rasters"
    )
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")

    derived = "rain_anomaly_monthly"
    datasets.append(_dataset(dataset_id=derived))
    grid = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]}

    def observed() -> xr.DataArray:
        values = np.arange(2 * 4 * 5, dtype="float32").reshape(2, 4, 5) + 10
        cube = xr.DataArray(
            values,
            dims=("t", "y", "x"),
            coords={"t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"), **grid},
            name="precip",
            attrs={"units": "mm/d"},
        )
        return cube.rio.write_crs("EPSG:4326")

    def normal() -> xr.DataArray:
        values = np.broadcast_to(np.arange(1, 13, dtype="float32")[:, None, None], (12, 4, 5)).copy()
        cube = xr.DataArray(
            values, dims=("month", "y", "x"), coords={"month": list(range(1, 13)), **grid}, attrs={"units": "mm/d"}
        )
        return cube.rio.write_crs("EPSG:4326")

    def load_collection(id: str | None = None, temporal_extent: Any = None, **_: Any) -> xr.DataArray:
        record_source(str(id), SimpleNamespace(artifact_id=f"{id}-artifact", source_dataset_id=id, path=None))
        if id == "chirps3_precipitation_daily":
            return observed()
        if id == "rain_normal_monthly":
            return normal()
        if id == derived:  # what the anomaly job published, read back from its store
            store = tmp_path / "managed" / "rasters" / f"{derived}.icechunk"
            with open_icechunk_dataset(str(store)) as ds:
                return ds["precip"].load()
        raise AssertionError(f"unexpected collection {id}")

    def load_features(id: str, version: str | None = None, **_: Any) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": _OU[0], "geometry": _box(0.5, 0.5, 2.5, 2.5)},
                {"type": "Feature", "id": _OU[1], "geometry": _box(3.5, 2.5, 5.5, 4.5)},
            ],
        }

    base = execution._build_process_registry()
    mocked = execution._RegistryOverlay(
        base,
        {
            "load_collection": Process(spec={}, implementation=load_collection),
            "load_features": Process(spec={}, implementation=load_features),
        },
    )
    monkeypatch.setattr(execution, "_build_process_registry", lambda: mocked)
    sent: list[dict[str, Any]] = []

    def send(
        self: Dhis2ExportPlugin, payload: bytes, target: Any, *, dry_run: bool = False, context: Any = None
    ) -> ExportReport:
        sent.append({"payload": json.loads(payload), "target": target, "dry_run": dry_run})
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=4,
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )

    monkeypatch.setattr(Dhis2ExportPlugin, "send", send)
    _dhis2(monkeypatch)

    # The aggregation pipeline over the derived dataset, created before the dataset updates.
    arguments = {**_SPEC["workflow"]["arguments"], "dataset_id": derived}
    _validated(_spec(id="anomaly-to-dhis2", workflow={**_SPEC["workflow"], "arguments": arguments}))

    job_service = openeo_jobs.OpenEOJobService()
    monkeypatch.setattr(openeo_jobs, "get_openeo_job_service", lambda: job_service)
    job_service.set_finished_listener(runner.on_job_finished)
    job_service.set_event_consumer(runner.consume)  # as main wires it, beside native jobs
    try:
        openeo_jobs.store_create_job(
            OpenEOJobRecord(
                id="anomaly-job",
                status=OpenEOJobStatus.QUEUED,
                created=utc_now(),
                process={
                    "process_graph": {
                        "anomaly": {
                            "process_id": "climate_anomaly",
                            "arguments": {
                                "observed_dataset_id": "chirps3_precipitation_daily",
                                "normal_dataset_id": "rain_normal_monthly",
                                "output_dataset_id": derived,
                                "variable": "precip",
                                "temporal_extent": ["2025-01-01", "2025-02-28"],
                            },
                            "result": True,
                        }
                    }
                },
            )
        )
        job_service._execute("anomaly-job")
        anomaly = openeo_jobs.store_get_job("anomaly-job")
        assert anomaly is not None and anomaly.status == OpenEOJobStatus.FINISHED, (
            anomaly.error_message if anomaly else None
        )
        [update] = anomaly.events
        assert update.data["dataset_id"] == derived and update.data["action"] == "full_rewrite"
        assert update.data["previous_end"] is None and update.data["producer_job_id"] == "anomaly-job"
        [artifact] = [record for record in ingestion_services._load_records() if record.dataset_id == derived]
        assert (
            artifact.provenance is not None
            and artifact.provenance.output_snapshot_id == update.data["output_snapshot_id"]
        )

        # The update ran the pipeline through the live consumer; wait for its delivery.
        deadline = time.monotonic() + 30
        record = store.get_record("anomaly-to-dhis2")
        while time.monotonic() < deadline:
            record = store.get_record("anomaly-to-dhis2")
            if record is not None and record.runs and record.runs[0].delivery_job_id:
                break
            time.sleep(0.05)
        assert record is not None and len(record.runs) == 1, record.runs if record else None
        [run] = record.runs
        assert run.trigger == "dataset_update" and run.event_id == update.event_id
        assert run.delivery == "submitted", (run.error, openeo_jobs.store_get_job(run.job_id))
        native = job_service_module.get_job_service()
        delivery = native.get_job_or_404(run.delivery_job_id or "")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and delivery.status not in {
            JobStatus.SUCCESSFUL,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            time.sleep(0.05)
            delivery = native.get_job_or_404(run.delivery_job_id or "")
        assert delivery.status == JobStatus.SUCCESSFUL, delivery.error
        [delivered] = sent
        assert delivered["dry_run"] is True and delivered["target"] == "local"
        values = delivered["payload"]["dataValues"]
        assert {value["orgUnit"] for value in values} == set(_OU)
        assert {value["period"] for value in values} == {"202501", "202502"}
        assert {value["dataElement"] for value in values} == {_ELEMENT}

        # Replaying the update at a restart runs nothing again.
        runner.reconcile_events()
        assert len(store.get_record("anomaly-to-dhis2").runs) == 1  # type: ignore[union-attr]

        # A deliberate re-run commits a new snapshot: a new update, and the pipeline runs again.
        openeo_jobs.store_update_job(
            "anomaly-job", lambda r: r.model_copy(update={"status": OpenEOJobStatus.QUEUED, "attempt": 0})
        )
        job_service._execute("anomaly-job")
        rerun = openeo_jobs.store_get_job("anomaly-job")
        assert rerun is not None and rerun.status == OpenEOJobStatus.FINISHED and len(rerun.events) == 2
        second_update = rerun.events[1]
        assert second_update.event_id != update.event_id
        assert second_update.data["output_snapshot_id"] != update.data["output_snapshot_id"]
        while time.monotonic() < deadline:
            record = store.get_record("anomaly-to-dhis2")
            again = next((r for r in record.runs if r.event_id == second_update.event_id), None) if record else None
            if again is not None and again.delivery_job_id:
                break
            time.sleep(0.05)
        record = store.get_record("anomaly-to-dhis2")
        assert record is not None and {r.event_id for r in record.runs} == {update.event_id, second_update.event_id}
        again = next(r for r in record.runs if r.event_id == second_update.event_id)
        assert again.delivery == "submitted"
        delivery = native.get_job_or_404(again.delivery_job_id or "")
        while time.monotonic() < deadline and delivery.status not in {
            JobStatus.SUCCESSFUL,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            time.sleep(0.05)
            delivery = native.get_job_or_404(again.delivery_job_id or "")
        assert delivery.status == JobStatus.SUCCESSFUL, delivery.error
        assert len(sent) == 2 and sent[1]["dry_run"] is True
    finally:
        job_service.shutdown()
        job_service_module.reset_job_service()
