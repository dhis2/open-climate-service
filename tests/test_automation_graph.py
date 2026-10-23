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
from open_climate_service.data_registry.services import datasets as registry_datasets
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


# Fetched from a data source; the normal and the anomaly are workflow outputs, whose generated
# templates have no ingestion block.
INGESTED = {"era5_daily", "era5_precip_monthly", "chirps_monthly"}


def _template(dataset_id: str) -> dict[str, Any]:
    template: dict[str, Any] = {"id": dataset_id, "source": "copied from the input for a derived one"}
    if dataset_id in INGESTED:
        template["ingestion"] = {"plugin": "example.Plugin"}
    return template


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Nepal-like instance: two schedules, a trigger chain through a derived dataset, a pipeline."""
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: SimpleNamespace(items=DATASETS))
    templates = [_template(d.dataset_id) for d in DATASETS] + [_template("era5_precip_monthly")]
    monkeypatch.setattr(registry_datasets, "list_datasets", lambda: templates)
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
        ("source:era5_daily", "dataset:era5_daily", "updates"),
        ("source:chirps_monthly", "dataset:chirps_monthly", "updates"),
        ("dataset:chirps_monthly", "workflow:trigger:chirps-anomaly", "updates"),
        ("dataset:chirps_normal", "workflow:trigger:chirps-anomaly", "input"),
        ("workflow:trigger:chirps-anomaly", "dataset:chirps_anomaly", "updates"),
        ("dataset:chirps_anomaly", "workflow:trigger:anomaly-to-districts", "updates"),
        ("features:districts", "workflow:trigger:anomaly-to-districts", "input"),
        ("dataset:era5_daily", "workflow:pipeline:daily-temperature-to-dhis2", "updates"),
        ("features:districts", "workflow:pipeline:daily-temperature-to-dhis2", "input"),
        ("workflow:pipeline:daily-temperature-to-dhis2", "destination:pipeline:daily-temperature-to-dhis2", "delivers"),
    }


def test_arrows_say_when_they_fire(client: TestClient) -> None:
    labels = {(e.source, e.target): e.label for e in _graph(client).edges}

    assert labels[("source:era5_daily", "dataset:era5_daily")] == "daily 06:00 UTC"
    assert labels[("dataset:chirps_monthly", "workflow:trigger:chirps-anomaly")] == "on update"
    assert labels[("dataset:era5_daily", "workflow:pipeline:daily-temperature-to-dhis2")] == "on update"
    # Inputs and deliveries carry no label: they read or send, they do not start anything.
    assert labels[("dataset:chirps_normal", "workflow:trigger:chirps-anomaly")] is None
    assert (
        labels[("workflow:pipeline:daily-temperature-to-dhis2", "destination:pipeline:daily-temperature-to-dhis2")]
        is None
    )

    fires = {(e.source, e.target): e.fires for e in _graph(client).edges}
    assert fires[("source:era5_daily", "dataset:era5_daily")] == "schedule"
    assert fires[("dataset:chirps_monthly", "workflow:trigger:chirps-anomaly")] == "update"
    assert fires[("dataset:chirps_normal", "workflow:trigger:chirps-anomaly")] is None


def test_each_step_carries_its_latest_run(client: TestClient) -> None:
    nodes = {node.id: node for node in _graph(client).nodes}

    assert nodes["dataset:era5_daily"].status.state == "ok"  # type: ignore[union-attr]
    assert nodes["dataset:era5_daily"].status.label == "Checked 21 Oct 06:00 UTC"  # type: ignore[union-attr]
    assert "2026-01-01 to 2026-10-15" in nodes["dataset:era5_daily"].meta
    assert nodes["dataset:chirps_monthly"].status.state == "waiting"  # type: ignore[union-attr]
    assert nodes["workflow:trigger:chirps-anomaly"].status.state == "ok"  # type: ignore[union-attr]
    assert nodes["workflow:trigger:anomaly-to-districts"].status.label == "Waiting for an update"  # type: ignore[union-attr]
    assert nodes["workflow:pipeline:daily-temperature-to-dhis2"].status.state == "dry_run"  # type: ignore[union-attr]
    # An input-only dataset has no schedule, so no status.
    assert nodes["dataset:chirps_normal"].status is None


def test_a_chain_through_a_derived_dataset_reads_left_to_right(client: TestClient) -> None:
    places = layout(_graph(client))
    column = {node_id: layer for node_id, (layer, _) in places.items()}

    assert (
        column["source:chirps_monthly"] < column["dataset:chirps_monthly"] < column["workflow:trigger:chirps-anomaly"]
    )
    assert (
        column["workflow:trigger:chirps-anomaly"]
        < column["dataset:chirps_anomaly"]
        < column["workflow:trigger:anomaly-to-districts"]
    )
    # Destinations get a column of their own, past the last trigger.
    assert (
        column["destination:pipeline:daily-temperature-to-dhis2"] == column["workflow:trigger:anomaly-to-districts"] + 1
    )


def test_automation_opens_on_pipelines_then_schedules(client: TestClient) -> None:
    nav = str(page_nav("", "automation"))
    assert 'href="/pipelines" aria-current="page">Pipelines<' in nav
    assert ">Schedules<" not in nav and ">Automation<" not in nav
    assert nav.index(">Data sources<") < nav.index(">Pipelines<") < nav.index(">Workflows<")

    schedules = client.get("/schedules", headers={"Accept": "text/html"}).text
    assert '<a href="/schedules" aria-current="page">Schedules</a>' in schedules
    pipelines = client.get("/pipelines", headers={"Accept": "text/html"}).text
    assert '<a href="/pipelines" aria-current="page">Pipelines</a>' in pipelines
    assert pipelines.index(">Pipelines</a>") < pipelines.index(">Schedules</a>")
    assert "Overview</a>" not in pipelines.split("<main", 1)[1] and ">Triggers</a>" not in pipelines

    response = client.get("/automation", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/pipelines"
    assert client.get("/automation/triggers").status_code == 404


def test_a_pipeline_page_draws_only_that_pipelines_chain(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: PIPELINE)

    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]

    # Its chain: the data source, the dataset that runs it, the districts it reads, its destination.
    chain = (
        "source:era5_daily",
        "dataset:era5_daily",
        "features:districts",
        "destination:pipeline:daily-temperature-to-dhis2",
    )
    for node in chain:
        assert f'data-id="{node}"' in diagram
    assert ">DHIS2<" in diagram
    assert "national-hmis" not in diagram
    assert 'class="ag-node ag-node-workflow ag-current"' in diagram
    # The pipeline is the whole diagram; its workflow box opens the reusable workflow.
    assert 'href="/workflows/aggregate_by_features"><g class="ag-node ag-node-workflow ag-current"' in diagram
    assert 'href="/pipelines/daily-temperature-to-dhis2"' not in diagram
    # Its arrows: the schedule that keeps its dataset current, the update that runs it.
    assert ">daily 06:00 UTC</text>" in diagram and ">on update</text>" in diagram
    assert diagram.count("<title>Runs on its sync schedule</title>") == 1
    assert diagram.count("<title>Runs when the dataset updates</title>") == 1
    assert 'data-source="dataset:era5_daily" data-target="workflow:pipeline:daily-temperature-to-dhis2"' in diagram
    assert "<title>Workflow: Aggregate a GeoZarr" in diagram
    assert 'id="ag-arrow-lit"' in diagram
    assert ">runs on update</span>" in page and ">sync schedule</span>" in page
    # Nothing beside it: the CHIRPS3 chain is not part of this pipeline.
    assert "trigger:chirps-anomaly" not in diagram
    assert "dataset:chirps_monthly" not in diagram
    # One box to a column here, so the headings are singular.
    assert ">DATA SOURCE<" in diagram
    assert ">DATASETS AND INPUTS<" in diagram
    assert ">WORKFLOW<" in diagram
    assert ">DESTINATION<" in diagram


def test_an_unscheduled_ingested_dataset_still_shows_its_data_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: PIPELINE)
    status = ScheduleListResponse(enabled=True, running=True, timezone="UTC", schedules=[])
    monkeypatch.setattr(scheduler_service, "get_scheduler_service", lambda: SimpleNamespace(status=lambda: status))

    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]

    assert 'data-id="source:era5_daily"' in diagram
    assert 'class="ag-edge ag-edge-origin" data-source="source:era5_daily" data-target="dataset:era5_daily"' in diagram
    assert ">DATA SOURCE<" in diagram
    assert "daily 06:00 UTC" not in diagram
    # The legend names the provenance arrow; its dotted style lives in the stylesheet.
    assert '<i class="ag-key ag-key-origin"></i>comes from' in page


def test_a_pipeline_view_includes_the_upstream_workflow_that_published_its_input(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = PIPELINE.spec.workflow.model_copy(
        update={"arguments": {"dataset_id": "chirps_anomaly", "geometries": {"from_features": "districts"}}}
    )
    derived_input_pipeline = PIPELINE.model_copy(
        update={"spec": PIPELINE.spec.model_copy(update={"workflow": workflow})}
    )
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: derived_input_pipeline)
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [derived_input_pipeline])

    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]

    assert 'data-id="dataset:chirps_anomaly"' in diagram
    assert 'data-id="workflow:trigger:chirps-anomaly"' in diagram
    assert 'data-id="dataset:chirps_monthly"' in diagram
    assert ">DATA SOURCE<" in diagram
    assert ">DERIVED DATASET<" in diagram


def test_the_pipelines_page_draws_each_pipeline_or_lists_them(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [PIPELINE])

    page = client.get("/pipelines", headers={"Accept": "text/html"}).text

    # Diagrams first, then List: the switch remembers the choice under its own key.
    assert 'data-views="pipelines"' in page
    assert page.index('data-view-button="diagrams"') < page.index('data-view-button="list"')
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]
    assert 'data-id="dataset:era5_daily"' in diagram and "trigger:chirps-anomaly" not in diagram
    # Its ids carry the pipeline's id, so several diagrams can share a page.
    assert 'id="ag-arrow-lit-daily-temperature-to-dhis2"' in diagram
    # The card title opens the pipeline; the workflow box opens the reusable workflow.
    assert '<a href="/pipelines/daily-temperature-to-dhis2">Daily temperature to DHIS2</a>' in page
    assert 'href="/workflows/aggregate_by_features"><g class="ag-node ag-node-workflow ag-current"' in diagram
    assert '<div class="panel table-wrap pipeline-table">' in page


def test_two_templates_with_one_provider_name_keep_their_own_data_source_box(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ERA5-Land templates share the name "ERA5-Land Reanalysis"; each box links to its own template."""
    precipitation = _dataset(
        "era5_precip_monthly", "ERA5-Land precipitation, monthly", "ERA5-Land Reanalysis", "2025-01", "2026-09"
    )
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: SimpleNamespace(items=[*DATASETS, precipitation]))
    workflow = PIPELINE.spec.workflow.model_copy(
        update={"arguments": {"dataset_id": "era5_precip_monthly", "geometries": {"from_features": "districts"}}}
    )
    reads_precipitation = PIPELINE.model_copy(update={"spec": PIPELINE.spec.model_copy(update={"workflow": workflow})})
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: reads_precipitation)
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [reads_precipitation])

    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]

    # The scheduled temperature template is read first; the precipitation box must not borrow its link.
    assert 'href="/data-sources/era5_precip_monthly"' in diagram
    assert 'href="/data-sources/era5_daily"' not in diagram
    assert ">ERA5-Land Reanalysis<" in diagram


def test_the_diagram_keeps_its_box_links_reachable_to_assistive_technology(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: PIPELINE)
    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    svg = page.split('<svg class="automation-graph"', 1)[1].split(">", 1)[0]

    # role="img" would make the <a> elements inside presentational.
    assert 'role="group"' in svg and 'role="img"' not in svg
    assert 'aria-labelledby="automation-graph-title"' in svg


def test_only_a_pipeline_that_reads_nothing_says_nothing_feeds_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = client.get("/pipelines", headers={"Accept": "text/html"}).text
    assert "Nothing feeds this pipeline yet" not in page

    workflow = PIPELINE.spec.workflow.model_copy(update={"arguments": {}})
    reads_nothing = PIPELINE.model_copy(update={"spec": PIPELINE.spec.model_copy(update={"workflow": workflow})})
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [reads_nothing])

    page = client.get("/pipelines", headers={"Accept": "text/html"}).text
    assert "Nothing feeds this pipeline yet" in page
    # Its workflow and destination are still drawn.
    assert 'data-id="workflow:pipeline:daily-temperature-to-dhis2"' in page


def test_an_unscheduled_dataset_upstream_of_a_trigger_keeps_its_data_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline reads a derived dataset whose input nobody schedules: the chain still starts at its source."""
    only_era5 = ScheduleListResponse(enabled=True, running=True, timezone="UTC", schedules=SCHEDULES[:1])
    monkeypatch.setattr(scheduler_service, "get_scheduler_service", lambda: SimpleNamespace(status=lambda: only_era5))
    workflow = PIPELINE.spec.workflow.model_copy(
        update={"arguments": {"dataset_id": "chirps_anomaly", "geometries": {"from_features": "districts"}}}
    )
    reads_anomaly = PIPELINE.model_copy(update={"spec": PIPELINE.spec.model_copy(update={"workflow": workflow})})
    monkeypatch.setattr(pipeline_store, "get_record", lambda pipeline_id: reads_anomaly)
    monkeypatch.setattr(pipeline_store, "list_records", lambda: [reads_anomaly])

    page = client.get("/pipelines/daily-temperature-to-dhis2", headers={"Accept": "text/html"}).text
    diagram = page.split('<svg class="automation-graph"', 1)[1].split("</svg>", 1)[0]

    # chirps_monthly is read only by the trigger, and has no schedule: its source comes by origin, no clock.
    assert (
        'class="ag-edge ag-edge-origin" data-source="source:chirps_monthly" data-target="dataset:chirps_monthly"'
        in diagram
    )
    # Nothing in this chain is scheduled, so no arrow carries a clock.
    assert "Runs on its sync schedule" not in diagram
    # Workflow outputs have no data source of their own, though their templates copy a source name.
    assert 'data-id="source:chirps_anomaly"' not in diagram
    assert 'data-id="source:chirps_normal"' not in diagram
