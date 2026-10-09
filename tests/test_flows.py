"""Flows and the Tasks pages: the configured graph, and the pages that manage and draw it (CLIM-1377, CLIM-1378)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from open_climate_service.flows.service import BOX_HEIGHT, BOX_WIDTH, COLUMN_GAP, ROW_GAP, WorkflowInfo, build_graph
from tests.test_schedules import client, instance  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tests.test_tasks import _WORKFLOW, _task

BROWSER = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def _norway() -> tuple[list[Any], dict[str, Any], dict[str, str], dict[str, WorkflowInfo]]:
    """Daily temperature, an anomaly derived from it, and the anomaly aggregated to kommuner and delivered."""
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
    names = {"temp_daily": "Temperature, daily", "temp_anomaly": "Temperature anomaly"}
    workflows = {
        "climate_anomaly": WorkflowInfo(title="Climate anomaly", results=[("publish", "Publishes a dataset")]),
        "aggregate_to_dhis2_json": WorkflowInfo(
            title="Aggregate to DHIS2 JSON", results=[("export", "Exports DHIS2 JSON")]
        ),
    }
    return tasks, exports, names, workflows


def test_the_graph_follows_a_dataset_through_a_derived_dataset_to_dhis2(instance: None) -> None:  # noqa: F811
    tasks, exports, names, workflows = _norway()

    graph = build_graph(tasks, exports, names, workflows)
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert edges == {
        ("dataset:temp_daily", "task:anomaly"),
        ("task:anomaly", "dataset:temp_anomaly"),  # the derived dataset
        ("dataset:temp_anomaly", "task:to-kommuner"),
        ("collection:kommuner", "task:to-kommuner"),  # org units as an input
        ("task:to-kommuner", "export:anomaly-kommuner"),
        ("export:anomaly-kommuner", "destination:hmis"),
    }
    by_id = {node.id: node for node in graph.nodes}
    assert by_id["dataset:temp_daily"].label == "Temperature, daily"
    assert by_id["dataset:temp_daily"].starts == "Synced every day at 06:00 (UTC)"
    assert by_id["task:to-kommuner"].label == "Aggregate to DHIS2 JSON"
    assert by_id["task:to-kommuner"].href == "/tasks/to-kommuner"
    assert by_id["export:anomaly-kommuner"].detail == "Live · dhis2"
    assert by_id["export:anomaly-kommuner"].task_id == "send"

    # A workflow with no named export writes a file each run; the graph says so.
    nightly = _task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *", arguments={"dataset_id": "era5"})
    info = {_WORKFLOW: WorkflowInfo(title="Aggregate to CHAP CSV", results=[("export", "Exports CHAP CSV")])}
    chap = build_graph([nightly], {}, {}, info)
    assert {(e.source, e.target) for e in chap.edges} == {
        ("dataset:era5", "task:nightly"),
        ("task:nightly", "result:nightly"),
    }
    assert {node.id: node.label for node in chap.nodes}["result:nightly"] == "CHAP CSV"

    # Around a node: everything upstream and downstream of it, and the org units its workflows use.
    assert {node.id for node in graph.around("collection:kommuner").nodes} == {
        "collection:kommuner",
        "task:to-kommuner",
        "export:anomaly-kommuner",
        "destination:hmis",
    }
    assert {node.id for node in graph.around("task:anomaly").nodes} == {
        "dataset:temp_daily",
        "task:anomaly",
        "dataset:temp_anomaly",
        "task:to-kommuner",
        "collection:kommuner",
        "export:anomaly-kommuner",
        "destination:hmis",
    }
    assert graph.around("dataset:unknown").nodes == []


def test_the_layout_places_each_node_once_with_a_chain_on_one_row(instance: None) -> None:  # noqa: F811
    tasks, exports, names, workflows = _norway()
    graph = build_graph(tasks, exports, names, workflows)

    layout = graph.layout()
    at = {node.id: (node.x, node.y) for node in layout.nodes}

    assert len(layout.nodes) == len(graph.nodes) and len(set(at.values())) == len(at)  # once each, no overlap
    chain = [
        "dataset:temp_daily",
        "task:anomaly",
        "dataset:temp_anomaly",
        "task:to-kommuner",
        "export:anomaly-kommuner",
        "destination:hmis",
    ]
    assert [at[node_id][0] for node_id in chain] == [index * (BOX_WIDTH + COLUMN_GAP) for index in range(6)]
    assert {at[node_id][1] for node_id in chain} == {0}
    # Org units sit in the column before the workflow that uses them, under the dataset it reads.
    assert at["collection:kommuner"] == (at["dataset:temp_anomaly"][0], BOX_HEIGHT + ROW_GAP)
    assert (layout.width, layout.height) == (6 * BOX_WIDTH + 5 * COLUMN_GAP, 2 * BOX_HEIGHT + ROW_GAP)
    edge = next(e for e in layout.edges if e.source == "collection:kommuner")
    assert edge.label == "org units" and edge.path.startswith(f"M{BOX_WIDTH * 3 + COLUMN_GAP * 2},")

    # A fan-out from one dataset stacks its branches, the first on the dataset's row.
    weekly = _task(id="weekly", kind="workflow", target="temporal_change", after={"dataset": "temp_daily"})
    fan = build_graph([tasks[0], tasks[2], weekly], {}, names).layout()
    rows = {node.id: node.y for node in fan.nodes}
    assert rows["dataset:temp_daily"] == 0 and rows["task:anomaly"] == 0 and rows["task:weekly"] == BOX_HEIGHT + ROW_GAP


def test_the_tasks_page_lists_tiles_and_draws_the_flow(client: TestClient) -> None:  # noqa: F811
    client.post("/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"})
    client.post("/tasks", json={"id": "agg", "kind": "workflow", "target": _WORKFLOW, "after": {"dataset": "chirps"}})

    graph = client.get("/flows", headers=BROWSER).json()
    assert {"dataset:chirps", "task:agg"} <= {node["id"] for node in graph["nodes"]}

    page = client.get("/tasks", headers=BROWSER).text
    assert 'data-kind="sync"' in page and 'data-kind="workflow"' in page and 'href="/tasks/agg"' in page
    assert 'data-view-button="flow"' in page and page.count('class="flow-node') >= 2


def test_a_task_is_added_from_the_form_and_managed_on_its_own_page(client: TestClient) -> None:  # noqa: F811
    assert client.get("/tasks/new", headers=BROWSER).status_code == 200

    added = client.post(
        "/tasks/form",
        data={
            "kind": "workflow",
            "target_workflow": _WORKFLOW,
            "starts": "schedule",
            "frequency": "daily",
            "check_time": "02:00",
        },
        follow_redirects=False,
    )
    assert added.status_code == 303
    task_id = added.headers["location"].rsplit("/", 1)[1]
    assert task_id.startswith("workflow-") and client.get(f"/tasks/{task_id}").json()["cron"] == "0 2 * * *"

    # The fields the API documents work too: an id, a cron, a plain target.
    legacy = client.post(
        "/tasks/form",
        data={"id": "nightly", "kind": "workflow", "target": _WORKFLOW, "cron": "0 3 * * *", "arguments": ""},
        follow_redirects=False,
    )
    assert legacy.status_code == 303

    page = client.get("/tasks/nightly", headers=BROWSER)
    assert page.status_code == 200 and "Runs the workflow" in page.text and "Not run yet" in page.text

    paused = client.post("/tasks/nightly/pause?next=task", follow_redirects=False)
    assert paused.status_code == 303 and paused.headers["location"] == "/tasks/nightly"
    assert client.get("/tasks/nightly").json()["enabled"] is False

    refused = client.post("/tasks/form", data={"id": "bad", "kind": "workflow", "target": "no_such_workflow"})
    assert refused.status_code == 400 and "unknown workflow" in refused.text

    assert client.post("/tasks/nightly/delete", follow_redirects=False).status_code == 303
    assert [task["id"] for task in client.get("/tasks").json()["tasks"]] == [task_id]


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
    assert "Go live" in page and 'class="flow-node flow-export is-current"' in page
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
