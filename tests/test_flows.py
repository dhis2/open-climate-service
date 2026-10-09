"""Flows and the Automation page: the configured graph, and the pages that manage and draw it (CLIM-1377, CLIM-1378)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from open_climate_service.flows.service import build_graph
from tests.test_schedules import client, instance  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tests.test_tasks import _WORKFLOW, _task

BROWSER = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def test_the_graph_follows_a_source_through_a_derived_dataset_to_dhis2(instance: None) -> None:  # noqa: F811
    tasks = [
        _task(id="sync-temp", kind="sync", target="temp_daily", cron="0 6 * * *"),
        _task(id="kommuner", kind="refresh", target="kommuner", cron="0 5 * * 1"),
        _task(
            id="anomaly",
            kind="workflow",
            target="climate_anomaly",
            after={"dataset": "temp_daily"},
            arguments={"output_dataset_id": "temp_anomaly"},
        ),
        _task(
            id="to-kommuner",
            kind="workflow",
            target="aggregate_to_dhis2_json",
            after={"dataset": "temp_anomaly"},
            arguments={"geometries": {"from_features": "kommuner"}, "export": "anomaly-kommuner"},
        ),
        _task(id="send", kind="deliver", target="anomaly-kommuner", after={"task": "to-kommuner"}, dry_run=False),
    ]
    exports = {"anomaly-kommuner": {"id": "anomaly-kommuner", "plugin": "dhis2", "connection": "hmis"}}

    graph = build_graph(tasks, exports, {"temp_daily": "Temperature, daily"})
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("source:temp_daily", "dataset:temp_daily") in edges
    assert ("dataset:temp_daily", "task:anomaly") in edges
    assert ("task:anomaly", "dataset:temp_anomaly") in edges  # the derived dataset
    assert ("dataset:temp_anomaly", "task:to-kommuner") in edges
    assert ("collection:kommuner", "task:to-kommuner") in edges  # org units as an input
    assert ("task:to-kommuner", "export:anomaly-kommuner") in edges
    assert ("export:anomaly-kommuner", "destination:hmis") in edges
    labels = {node.id: node.label for node in graph.nodes}
    assert labels["dataset:temp_daily"] == "Temperature, daily"
    around = graph.around("dataset:temp_anomaly")
    assert {node.id for node in around.nodes} == {"dataset:temp_anomaly", "task:anomaly", "task:to-kommuner"}


def test_flows_are_json_for_clients_and_a_page_for_browsers(client: TestClient) -> None:  # noqa: F811
    client.post("/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"})
    client.post("/tasks", json={"id": "agg", "kind": "workflow", "target": _WORKFLOW, "after": {"dataset": "chirps"}})

    graph = client.get("/flows").json()
    assert {"dataset:chirps", "task:agg"} <= {node["id"] for node in graph["nodes"]}

    page = client.get("/flows", headers=BROWSER)
    assert page.status_code == 200 and "Flows" in page.text and _WORKFLOW in page.text


def test_the_automation_page_adds_pauses_and_removes_tasks(client: TestClient) -> None:  # noqa: F811
    added = client.post(
        "/tasks/form",
        data={"id": "nightly", "kind": "workflow", "target": _WORKFLOW, "cron": "0 2 * * *", "arguments": ""},
        follow_redirects=False,
    )
    assert added.status_code == 303

    page = client.get("/tasks", headers=BROWSER)
    assert page.status_code == 200 and "nightly" in page.text and "Automation" in page.text

    assert client.post("/tasks/nightly/pause?next=tasks", follow_redirects=False).status_code == 303
    assert client.get("/tasks/nightly").json()["enabled"] is False

    refused = client.post("/tasks/form", data={"id": "bad", "kind": "workflow", "target": "no_such_workflow"})
    assert refused.status_code == 400 and "unknown workflow" in refused.text

    assert client.post("/tasks/nightly/delete", follow_redirects=False).status_code == 303
    assert client.get("/tasks").json()["tasks"] == []


# --- Send to DHIS2 from the dataset page (CLIM-1290) --------------------------------------------------------


def test_send_to_creates_an_export_and_two_tasks_as_one_path(
    client: TestClient,  # noqa: F811
    monkeypatch: Any,
) -> None:
    from types import SimpleNamespace

    from open_climate_service import config as api_config
    from open_climate_service.features import templates as feature_templates
    from open_climate_service.ingestions import services as ingestion_services

    monkeypatch.setattr(
        api_config,
        "_cache",
        {
            "scheduler": {"enabled": False},
            "dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "T"}],
        },
    )
    monkeypatch.setattr(feature_templates, "list_feature_templates", lambda: [{"id": "districts"}])
    monkeypatch.setattr(
        ingestion_services, "get_dataset_or_404", lambda dataset_id: SimpleNamespace(period_type="monthly")
    )
    body = {"connection": "hmis", "collection": "districts", "data_element": "BXgDHhPdFVU", "statistic": "mean"}

    created = client.post("/datasets/chirps/send-to", json=body)
    assert created.status_code == 200, created.text
    assert created.json() == {
        "export": "chirps-BXgDHhPdFVU",
        "tasks": ["chirps-BXgDHhPdFVU-aggregate", "chirps-BXgDHhPdFVU-deliver"],
    }
    assert client.get("/tasks/chirps-BXgDHhPdFVU-deliver").json()["dry_run"] is True

    edges = {(edge["source"], edge["target"]) for edge in client.get("/flows").json()["edges"]}
    assert ("dataset:chirps", "task:chirps-BXgDHhPdFVU-aggregate") in edges
    assert ("task:chirps-BXgDHhPdFVU-aggregate", "export:chirps-BXgDHhPdFVU") in edges
    assert ("export:chirps-BXgDHhPdFVU", "destination:hmis") in edges

    again = client.post("/datasets/chirps/send-to", json=body)
    assert again.status_code == 409 and "already sent" in again.json()["detail"]

    refused = client.post("/datasets/era5/send-to", json={**body, "statistic": "mode"})
    assert refused.status_code == 409
    assert [item["id"] for item in client.get("/exports").json()["exports"]] == ["chirps-BXgDHhPdFVU"]
