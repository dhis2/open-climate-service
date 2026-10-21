"""The Automation section: the graph of schedules, triggers and pipelines, and its pages."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service.automation import config as automation_config
from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger
from open_climate_service.automation.graph import AutomationGraph, layout
from open_climate_service.features import services as feature_services
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.main import app
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.pipelines import store as pipeline_store
from open_climate_service.pipelines.schemas import PipelineRecord, PipelineRun, PipelineSpec
from open_climate_service.scheduler import service as scheduler_service
from open_climate_service.scheduler.dispatcher import CheckOutcome
from open_climate_service.scheduler.schemas import ScheduleListResponse, ScheduleStatus
from open_climate_service.system.templates import page_nav

CHECKED = datetime(2026, 10, 21, 6, 0, tzinfo=UTC)
NEXT = datetime(2026, 10, 22, 6, 0, tzinfo=UTC)


def _dataset(dataset_id: str, name: str, source: str, start: str, end: str) -> Any:
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_name=name,
        source=source,
        source_dataset_id=dataset_id,
        extent=SimpleNamespace(temporal=SimpleNamespace(start=start, end=end)),
    )


DATASETS = [
    _dataset("era5_daily", "ERA5-Land 2m temperature, daily", "ERA5-Land Reanalysis", "2026-01-01", "2026-10-15"),
    _dataset("chirps_monthly", "CHIRPS3 precipitation, monthly", "CHIRPS v3 (monthly)", "1990-01", "2026-09"),
    _dataset("chirps_normal", "CHIRPS3 precipitation normal", "Derived", "", ""),
    _dataset("chirps_anomaly", "CHIRPS3 precipitation anomaly", "Derived", "1990-01-01", "2026-09-01"),
]

SCHEDULES = [
    ScheduleStatus(
        schedule_id="dataset-sync:era5_daily",
        dataset_id="era5_daily",
        cron="0 6 * * *",
        timezone="UTC",
        publish=True,
        max_attempts=3,
        registered=True,
        next_check=NEXT,
        last_check=CHECKED,
        last_outcome=CheckOutcome.SUBMITTED,
    ),
    ScheduleStatus(
        schedule_id="dataset-sync:chirps_monthly",
        dataset_id="chirps_monthly",
        cron="0 6 * * *",
        timezone="UTC",
        publish=True,
        max_attempts=3,
        registered=True,
        next_check=NEXT,
    ),
]

TRIGGERS = [
    WorkflowTrigger(
        id="chirps-anomaly",
        on_update_of="chirps_monthly",
        workflow_id="climate_anomaly",
        arguments={
            "observed_dataset_id": "$event.dataset_id",
            "normal_dataset_id": "chirps_normal",
            "output_dataset_id": "chirps_anomaly",
            "temporal_extent": ["$event.current_start", "$event.current_end"],
        },
    ),
    WorkflowTrigger(
        id="anomaly-to-districts",
        on_update_of="chirps_anomaly",
        workflow_id="aggregate_to_dhis2_json",
        arguments={"dataset_id": "$event.dataset_id", "geometries": {"from_features": "districts"}},
    ),
]

PIPELINE = PipelineRecord(
    spec=PipelineSpec.model_validate(
        {
            "id": "daily-temperature-to-dhis2",
            "name": "Daily temperature to DHIS2",
            "workflow": {
                "id": "aggregate_by_features",
                "arguments": {"dataset_id": "era5_daily", "geometries": {"from_features": "districts"}},
            },
            "destination": {"plugin": "dhis2", "settings": {"connection": "national-hmis"}},
            "delivery": {"mode": "dry_run", "policy": "on_update"},
        }
    ),
    created_at="2026-10-20T10:00:00Z",
    runs=[
        PipelineRun(
            run_id="r1",
            job_id="j1",
            mode="dry_run",
            start="2026-10-15",
            end="2026-10-15",
            submitted_at="2026-10-21T06:03:00Z",
            idempotency_key="k1",
            delivery="submitted",
            delivery_result="validated",
        )
    ],
)


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Nepal-like instance: two schedules, a trigger chain through a derived dataset, a pipeline."""
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: SimpleNamespace(items=DATASETS))
    monkeypatch.setattr(
        feature_services,
        "list_feature_collections",
        lambda: SimpleNamespace(items=[SimpleNamespace(id="districts", name="Districts (Overture Maps)")]),
    )
    status = ScheduleListResponse(enabled=True, running=True, timezone="UTC", schedules=SCHEDULES)
    monkeypatch.setattr(scheduler_service, "get_scheduler_service", lambda: SimpleNamespace(status=lambda: status))
    monkeypatch.setattr(
        automation_config, "get_automation_config", lambda: AutomationConfig(workflow_triggers=TRIGGERS)
    )
    finished = SimpleNamespace(
        trigger_id="chirps-anomaly",
        status="finished",
        created=CHECKED,
        updated=datetime(2026, 10, 21, 6, 2, tzinfo=UTC),
    )
    monkeypatch.setattr(openeo_jobs, "read_job_records", lambda: [finished])
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [PIPELINE])


@pytest.fixture
def client(instance: None) -> TestClient:
    return TestClient(app)


def _graph(client: TestClient) -> AutomationGraph:
    response = client.get("/automation/graph")
    assert response.status_code == 200
    return AutomationGraph.model_validate(response.json())


def _edges(graph: AutomationGraph) -> set[tuple[str, str, str]]:
    return {(e.source, e.target, e.kind) for e in graph.edges}


def test_the_graph_connects_schedules_triggers_and_pipelines(client: TestClient) -> None:
    graph = _graph(client)

    assert _edges(graph) == {
        ("source:ERA5-Land Reanalysis", "dataset:era5_daily", "updates"),
        ("source:CHIRPS v3 (monthly)", "dataset:chirps_monthly", "updates"),
        ("dataset:chirps_monthly", "trigger:chirps-anomaly", "updates"),
        ("dataset:chirps_normal", "trigger:chirps-anomaly", "input"),
        ("trigger:chirps-anomaly", "dataset:chirps_anomaly", "updates"),
        ("dataset:chirps_anomaly", "trigger:anomaly-to-districts", "updates"),
        ("features:districts", "trigger:anomaly-to-districts", "input"),
        ("dataset:era5_daily", "pipeline:daily-temperature-to-dhis2", "updates"),
        ("features:districts", "pipeline:daily-temperature-to-dhis2", "input"),
        ("pipeline:daily-temperature-to-dhis2", "destination:dhis2:national-hmis", "delivers"),
    }


def test_each_step_carries_its_latest_run(client: TestClient) -> None:
    nodes = {node.id: node for node in _graph(client).nodes}

    assert nodes["dataset:era5_daily"].status.state == "ok"  # type: ignore[union-attr]
    assert nodes["dataset:era5_daily"].status.label == "Checked 21 Oct 06:00 UTC"  # type: ignore[union-attr]
    assert "2026-01-01 to 2026-10-15" in nodes["dataset:era5_daily"].meta
    assert nodes["dataset:chirps_monthly"].status.state == "waiting"  # type: ignore[union-attr]
    assert nodes["trigger:chirps-anomaly"].status.state == "ok"  # type: ignore[union-attr]
    assert nodes["trigger:anomaly-to-districts"].status.label == "Waiting for an update"  # type: ignore[union-attr]
    assert nodes["pipeline:daily-temperature-to-dhis2"].status.state == "dry_run"  # type: ignore[union-attr]
    # An input-only dataset has no schedule, so no status.
    assert nodes["dataset:chirps_normal"].status is None


def test_a_chain_through_a_derived_dataset_reads_left_to_right(client: TestClient) -> None:
    places = layout(_graph(client))
    column = {node_id: layer for node_id, (layer, _) in places.items()}

    assert column["source:CHIRPS v3 (monthly)"] < column["dataset:chirps_monthly"] < column["trigger:chirps-anomaly"]
    assert column["trigger:chirps-anomaly"] < column["dataset:chirps_anomaly"] < column["trigger:anomaly-to-districts"]
    # Destinations get a column of their own, past the last trigger.
    assert column["destination:dhis2:national-hmis"] == column["trigger:anomaly-to-districts"] + 1


def test_the_overview_page_draws_the_graph_with_links(client: TestClient) -> None:
    page = client.get("/automation", headers={"Accept": "text/html"}).text

    assert '<svg class="automation-graph"' in page
    assert 'href="/datasets/era5_daily"' in page
    assert 'href="/pipelines/daily-temperature-to-dhis2"' in page
    assert "Checked 21 Oct 06:00 UTC" in page
    assert 'class="page-tabs automation-tabs"' in page


def test_the_overview_answers_json_to_an_api_client(client: TestClient) -> None:
    response = client.get("/automation", headers={"Accept": "application/json"})

    assert response.headers["content-type"].startswith("application/json")
    assert {node["id"] for node in response.json()["nodes"]} >= {"dataset:era5_daily", "trigger:chirps-anomaly"}


def test_the_triggers_page_lists_triggers_read_only(client: TestClient) -> None:
    page = client.get("/automation/triggers").text

    assert "chirps-anomaly" in page and "anomaly-to-districts" in page
    assert 'href="/datasets/chirps_anomaly"' in page
    assert "<form" not in page


def test_an_instance_with_nothing_automated_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: SimpleNamespace(items=[]))
    empty = ScheduleListResponse(enabled=False, running=False, timezone="UTC", schedules=[])
    monkeypatch.setattr(scheduler_service, "get_scheduler_service", lambda: SimpleNamespace(status=lambda: empty))
    monkeypatch.setattr(automation_config, "get_automation_config", lambda: AutomationConfig())
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [])

    page = TestClient(app).get("/automation", headers={"Accept": "text/html"}).text

    assert "Nothing is automated yet" in page
    assert "<svg" not in page.split("<main", 1)[1]


def test_schedules_and_pipelines_live_under_automation(client: TestClient) -> None:
    nav = str(page_nav("", "automation"))
    assert 'href="/automation" aria-current="page"' in nav
    assert ">Schedules<" not in nav and ">Pipelines<" not in nav

    schedules = client.get("/schedules", headers={"Accept": "text/html"}).text
    assert '<a href="/schedules" aria-current="page">Schedules</a>' in schedules
    pipelines = client.get("/pipelines", headers={"Accept": "text/html"}).text
    assert '<a href="/pipelines" aria-current="page">Pipelines</a>' in pipelines
