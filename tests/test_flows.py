"""Flows and the Tasks pages: the configured graph, and the pages that manage and draw it (CLIM-1377, CLIM-1378)."""

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
    nightly = _task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *", arguments={"dataset_id": "era5"})
    assert ("dataset:era5", "task:nightly") in {(e.source, e.target) for e in build_graph([nightly], {}, {}).edges}


def test_chains_read_one_path_per_row_with_org_units_on_the_workflow(instance: None) -> None:  # noqa: F811
    tasks = [
        _task(id="sync-temp", kind="sync", target="temp_daily", cron="0 6 * * *"),
        _task(id="kommuner", kind="refresh", target="kommuner", cron="0 5 * * 1"),
        _task(
            id="to-kommuner",
            kind="workflow",
            target="aggregate_to_dhis2_json",
            after={"dataset": "temp_daily"},
            arguments={"geometries": {"from_features": "kommuner"}, "export": "temp-kommuner"},
        ),
        _task(id="send", kind="deliver", target="temp-kommuner", after={"task": "to-kommuner"}),
        _task(id="anomaly", kind="workflow", target="climate_anomaly", after={"dataset": "temp_daily"}),
    ]
    exports = {"temp-kommuner": {"id": "temp-kommuner", "plugin": "dhis2", "connection": "hmis"}}
    graph = build_graph(tasks, exports, {"temp_daily": "Temperature, daily"})

    rows = [[step.node.id for step in row] for row in graph.chains()]
    assert rows == [
        ["dataset:temp_daily", "task:to-kommuner", "export:temp-kommuner", "destination:hmis"],
        ["dataset:temp_daily", "task:anomaly"],
        ["collection:kommuner"],
    ]
    workflow = graph.chains()[0][1]
    assert workflow.via is None and [node.id for node in workflow.inputs] == ["collection:kommuner"]
    assert workflow.node.label == "Aggregate to DHIS2 JSON" and workflow.node.detail == "to-kommuner"
    assert graph.chains()[0][0].node.starts == "Synced every day at 06:00 (UTC)"
    assert len(graph.chains(through="task:anomaly")) == 1


def test_the_tasks_page_draws_the_flows_and_the_graph_stays_json(client: TestClient) -> None:  # noqa: F811
    client.post("/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"})
    client.post("/tasks", json={"id": "agg", "kind": "workflow", "target": _WORKFLOW, "after": {"dataset": "chirps"}})

    graph = client.get("/flows", headers=BROWSER).json()
    assert {"dataset:chirps", "task:agg"} <= {node["id"] for node in graph["nodes"]}

    page = client.get("/tasks", headers=BROWSER).text
    assert 'class="flow-chain"' in page and _WORKFLOW in page and "agg" in page


def test_a_task_is_added_on_the_list_and_paused_and_removed_on_its_own_page(client: TestClient) -> None:  # noqa: F811
    added = client.post(
        "/tasks/form",
        data={"id": "nightly", "kind": "workflow", "target": _WORKFLOW, "cron": "0 2 * * *", "arguments": ""},
        follow_redirects=False,
    )
    assert added.status_code == 303

    listing = client.get("/tasks", headers=BROWSER)
    assert listing.status_code == 200 and 'href="/tasks/nightly"' in listing.text

    page = client.get("/tasks/nightly", headers=BROWSER)
    assert page.status_code == 200 and "Runs the workflow" in page.text and "Not run yet" in page.text

    paused = client.post("/tasks/nightly/pause?next=task", follow_redirects=False)
    assert paused.status_code == 303 and paused.headers["location"] == "/tasks/nightly"
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
    page = client.get("/tasks/chirps-BXgDHhPdFVU-deliver", headers=BROWSER).text
    assert "Go live" in page and 'class="flow-step flow-export is-current"' in page
    live = client.post(
        "/tasks/chirps-BXgDHhPdFVU-deliver/dry-run?next=task", data={"dry_run": "false"}, follow_redirects=False
    )
    assert live.status_code == 303
    assert client.get("/tasks/chirps-BXgDHhPdFVU-deliver").json()["dry_run"] is False

    edges = {(edge["source"], edge["target"]) for edge in client.get("/flows").json()["edges"]}
    assert ("dataset:chirps", "task:chirps-BXgDHhPdFVU-aggregate") in edges
    assert ("task:chirps-BXgDHhPdFVU-aggregate", "export:chirps-BXgDHhPdFVU") in edges
    assert ("export:chirps-BXgDHhPdFVU", "destination:hmis") in edges

    again = client.post("/datasets/chirps/send-to", json=body)
    assert again.status_code == 409 and "already sent" in again.json()["detail"]

    refused = client.post("/datasets/era5/send-to", json={**body, "statistic": "mode"})
    assert refused.status_code == 409
    assert [item["id"] for item in client.get("/exports").json()["exports"]] == ["chirps-BXgDHhPdFVU"]
