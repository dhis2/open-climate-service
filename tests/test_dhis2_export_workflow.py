"""CLIM-1216: the built-in DHIS2 aggregation workflow produces a deliverable named export."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from openeo_pg_parser_networkx.process_registry import Process

from open_climate_service import config
from open_climate_service.exports import ExportReport
from open_climate_service.exports.delivery_input import lease_export_input
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
from open_climate_service.exports.report import ExportOutcome
from open_climate_service.exports.service import check_execution_declarations, resolve_named_export
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.jobs.models import JobRecord, JobStatus
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.plugins.processes.aggregate_spatial import reduce_by_method
from open_climate_service.shared.provenance import capture_execution, observe_spatial_aggregation, record_source
from open_climate_service.shared.time import utc_now

_DATA_ELEMENT = "BXgDHhPdFVU"
_OU_A = "ImspTQPwCqd"
_OU_B = "O6uvpzGd5pu"


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }


def _geometries(first_id: str | None = _OU_A) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": first_id, "geometry": _box(0.5, 0.5, 2.5, 2.5)},
            {"type": "Feature", "id": _OU_B, "geometry": _box(3.5, 2.5, 5.5, 4.5)},
        ],
    }


def _load_collection(id: str | None = None, temporal_extent: Any = None, **_: Any) -> xr.DataArray:
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


def _process(
    geometries: dict[str, Any],
    export: str = "rain-monthly",
    method: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "dataset_id": "rain_monthly",
        "temporal_extent": ["2025-01-01", "2025-02-28"],
        "geometries": geometries,
        "export": export,
    }
    if method is not None:
        arguments["method"] = method
    return {
        "process_graph": {
            "agg": {
                "process_id": "aggregate_to_dhis2_json",
                "arguments": arguments,
                "result": True,
            }
        }
    }


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[openeo_jobs.OpenEOJobService]:
    """An instance with one bound named export and a mocked dataset source."""
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [])
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "exports": [
                {
                    "id": "rain-monthly",
                    "plugin": "dhis2",
                    "dataset": "rain_monthly",
                    "connection": "hmis",
                    "period_type": "monthly",
                    "aggregation": "mean",
                    "series": [{"data_element": _DATA_ELEMENT}],
                }
            ],
            "dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "TEST_TOKEN"}],
        },
    )
    base = execution._build_process_registry()
    mocked = execution._RegistryOverlay(base, {"load_collection": Process(spec={}, implementation=_load_collection)})
    monkeypatch.setattr(execution, "_build_process_registry", lambda: mocked)
    service = openeo_jobs.OpenEOJobService()
    try:
        yield service
    finally:
        service.shutdown()
        job_service_module.reset_job_service()


def _run(service: openeo_jobs.OpenEOJobService, job_id: str, process: dict[str, Any]) -> OpenEOJobRecord:
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id=job_id, status=OpenEOJobStatus.QUEUED, created=utc_now(), process=process)
    )
    service._execute(job_id)
    record = openeo_jobs.store_get_job(job_id)
    assert record is not None
    return record


def _await_terminal(job_id: str, timeout: float = 5.0) -> JobRecord:
    service = job_service_module.get_job_service()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.get_job_or_404(job_id)
        if record.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    pytest.fail(f"Delivery job '{job_id}' did not finish within {timeout}s")


@pytest.fixture
def fake_send(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def send(
        self: Dhis2ExportPlugin,
        payload: bytes,
        target: Any,
        *,
        dry_run: bool = False,
        context: Any = None,
    ) -> ExportReport:
        calls.append({"payload": payload, "target": target, "dry_run": dry_run})
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            message="validated",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=4,
            imported=0,
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )

    monkeypatch.setattr(Dhis2ExportPlugin, "send", send)
    return calls


def test_workflow_is_builtin_and_takes_only_an_export_id() -> None:
    workflow = workflow_store.get_workflow("aggregate_to_dhis2_json")
    assert workflow is not None
    names = {parameter["name"] for parameter in workflow.parameters or []}
    # Instance-specific mapping lives in configuration, not in workflow parameters.
    assert names == {"dataset_id", "temporal_extent", "geometries", "export", "method"}
    save = workflow.process_graph["save"]["arguments"]
    assert save["format"] == "DHIS2JSON"
    assert save["options"] == {"export": {"from_parameter": "export"}}


def test_workflow_job_produces_verified_delivery_input(instance: openeo_jobs.OpenEOJobService) -> None:
    record = _run(instance, "agg-job", _process(_geometries()))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message

    with lease_export_input("rain-monthly", "agg-job") as verified:
        manifest = verified.manifest
        payload = json.loads(verified.content)

    assert manifest.export_id == "rain-monthly"
    assert manifest.source_job_id == "agg-job"
    assert manifest.target is not None
    assert manifest.references == {"dataset": "rain_monthly", "connection": "hmis"}
    assert manifest.periods == ["202501", "202502"]
    assert manifest.record_count == 4
    assert manifest.provenance["sources"][0]["collection_id"] == "rain_monthly"
    assert manifest.provenance["features"][0]["ids_valid"] is True
    # The declared `aggregation: mean` is checked against the default method that ran.
    assert manifest.provenance["spatial_aggregations"] == ["mean"]
    values = payload["dataValues"]
    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}
    assert {value["period"] for value in values} == {"202501", "202502"}
    assert {value["dataElement"] for value in values} == {_DATA_ELEMENT}


def test_workflow_export_submits_dry_run_delivery(
    instance: openeo_jobs.OpenEOJobService, fake_send: list[dict[str, Any]], client: TestClient
) -> None:
    record = _run(instance, "agg-job", _process(_geometries()))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message

    response = client.post(
        "/exports/rain-monthly",
        json={"job_id": "agg-job", "dry_run": True},
        headers={"Idempotency-Key": "agg-job-dry-run"},
    )
    assert response.status_code == 202, response.text
    delivery = _await_terminal(response.json()["delivery_job_id"])
    assert delivery.status == JobStatus.SUCCESSFUL, delivery.error
    assert isinstance(delivery.result, dict)
    assert delivery.result["outcome"] == ExportOutcome.DRY_RUN
    assert len(fake_send) == 1
    assert fake_send[0]["dry_run"] is True
    assert fake_send[0]["target"] == "hmis"
    sent = json.loads(fake_send[0]["payload"])
    assert len(sent["dataValues"]) == 4


def test_workflow_rejects_features_without_original_ids(instance: openeo_jobs.OpenEOJobService) -> None:
    # Detected inside the called workflow, so the original IDs are checked before
    # aggregation assigns positional labels, as for a direct named DHIS2 graph.
    record = _run(instance, "agg-job", _process(_geometries(first_id=None)))
    assert record.status == OpenEOJobStatus.ERROR
    assert record.error_message is not None and "Feature 0 has no usable feature.id" in record.error_message


def test_workflow_rejects_method_contradicting_the_export(instance: openeo_jobs.OpenEOJobService) -> None:
    record = _run(instance, "agg-job", _process(_geometries(), method="sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert record.error_message is not None
    assert "Declared export aggregation 'mean' does not match the executed spatial aggregation 'sum'" in (
        record.error_message
    )


def test_workflow_accepts_median_declared_by_the_export(instance: openeo_jobs.OpenEOJobService) -> None:
    config.get_config()["exports"][0]["aggregation"] = "median"
    record = _run(instance, "agg-job", _process(_geometries(), method="median"))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    with lease_export_input("rain-monthly", "agg-job") as verified:
        assert verified.manifest.provenance["spatial_aggregations"] == ["median"]


def test_undeclared_aggregation_accepts_any_method(instance: openeo_jobs.OpenEOJobService) -> None:
    del config.get_config()["exports"][0]["aggregation"]
    record = _run(instance, "agg-job", _process(_geometries(), method="max"))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message


def test_workflow_rejects_unknown_export(instance: openeo_jobs.OpenEOJobService) -> None:
    record = _run(instance, "agg-job", _process(_geometries(), export="missing"))
    assert record.status == OpenEOJobStatus.ERROR
    assert record.error_message is not None and "Unknown export 'missing'" in record.error_message


def test_named_export_detection_expands_called_workflows() -> None:
    graphs = {
        "outer": {"inner_call": {"process_id": "inner", "arguments": {}, "result": True}},
        "inner": {
            "save": {
                "process_id": "save_result",
                "arguments": {"format": "DHIS2JSON", "options": {"export": {"from_parameter": "export"}}},
                "result": True,
            }
        },
        "cycle": {"again": {"process_id": "cycle", "arguments": {}, "result": True}},
    }

    def graph(process_id: str) -> dict[str, Any]:
        return {"process_graph": {"call": {"process_id": process_id, "arguments": {}, "result": True}}}

    with capture_execution(graph("outer"), graphs) as evidence:
        assert evidence.require_feature_ids is True
    with capture_execution(graph("cycle"), graphs) as evidence:
        assert evidence.require_feature_ids is False
    with capture_execution(graph("unrelated"), graphs) as evidence:
        assert evidence.require_feature_ids is False


def test_ad_hoc_dhis2_json_graph_remains_supported(instance: openeo_jobs.OpenEOJobService) -> None:
    """The documented escape hatch: legacy save_result options, no named export."""
    graph = {
        "process_graph": {
            "load": {
                "process_id": "load_collection",
                "arguments": {"id": "rain_monthly", "temporal_extent": ["2025-01-01", "2025-02-28"]},
            },
            "zonal": {
                "process_id": "aggregate_spatial",
                "arguments": {
                    "data": {"from_node": "load"},
                    "geometries": _geometries(),
                    "reducer": {
                        "process_graph": {
                            "mean": {
                                "process_id": "mean",
                                "arguments": {"data": {"from_parameter": "data"}},
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
                    "format": "DHIS2JSON",
                    "options": {"data_element_id": _DATA_ELEMENT, "org_unit_field": "geometry", "period_type": "month"},
                },
                "result": True,
            },
        }
    }
    record = _run(instance, "adhoc-job", graph)
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    output = Path(str((record.usage or {})["output_path"]))
    values = json.loads(output.read_text())["dataValues"]
    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}
    assert {value["period"] for value in values} == {"202501", "202502"}
    assert {value["dataElement"] for value in values} == {_DATA_ELEMENT}


@pytest.mark.parametrize(("method", "status"), [("mean", 200), ("sum", 400)])
def test_synchronous_run_checks_aggregation_like_a_batch_job(
    instance: openeo_jobs.OpenEOJobService, client: TestClient, method: str, status: int
) -> None:
    response = client.post("/result", json={"process": _process(_geometries(), method=method)})
    assert response.status_code == status, response.text
    if status == 200:
        assert len(response.json()["dataValues"]) == 4
    else:
        assert "does not match the executed spatial aggregation 'sum'" in response.text


def test_named_reductions_outside_aggregate_spatial_are_not_spatial_aggregations() -> None:
    with capture_execution({}) as evidence:
        reduce_by_method(np.array([1.0, 2.0]), "sum")  # e.g. a temporal reduction
        with observe_spatial_aggregation():
            reduce_by_method(np.array([1.0, 2.0]), "mean")
        with observe_spatial_aggregation():
            pass  # an aggregate_spatial whose reducer is not a named reduction
    assert evidence.spatial_aggregations == ["mean", None]
    assert "spatial_aggregation_method" in evidence.describe()["missing"]


@pytest.mark.parametrize(
    ("observed", "rejected"),
    [
        (["sum"], True),
        (["mean"], False),
        (["mean", "sum"], False),  # cannot be attributed to the saved result
        ([None], False),  # unnamed reducer
        ([], False),
    ],
)
def test_aggregation_is_checked_only_when_attributable(
    instance: openeo_jobs.OpenEOJobService, observed: list[str | None], rejected: bool
) -> None:
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain-monthly"})
    provenance = {"sources": [], "spatial_aggregations": observed}
    if rejected:
        with pytest.raises(ValueError, match="does not match the executed spatial aggregation"):
            check_execution_declarations(resolved, provenance)
    else:
        check_execution_declarations(resolved, provenance)
