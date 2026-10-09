"""Run records: one per task run, linked to its job and to the run that set it off (CLIM-1378)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from open_climate_service.automation.config import compile_tasks
from open_climate_service.automation.service import WorkflowAutomationService
from open_climate_service.runs import service as runs
from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult
from open_climate_service.scheduler.service import SchedulerService
from open_climate_service.tasks import store
from tests.test_schedules import client, instance  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tests.test_tasks import _WORKFLOW, _event, _openeo, _task


def _sync_submitting(job_id: str) -> Any:
    def dispatch(schedule: DatasetSyncSchedule) -> CheckResult:
        return CheckResult(
            schedule_id=schedule.schedule_id,
            dataset_id=schedule.dataset_id,
            outcome=CheckOutcome.SUBMITTED,
            message="Scheduled sync submitted",
            job_id=job_id,
        )

    return dispatch


def test_a_sync_is_followed_to_the_workflow_it_set_off(client: TestClient) -> None:  # noqa: F811
    """The event names the sync's job, so the workflow run is linked to the sync run."""
    SchedulerService(dispatcher=_sync_submitting("native-1")).check_now(
        DatasetSyncSchedule(dataset_id="chirps", cron="0 6 * * *")
    )
    config = compile_tasks([_task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"})])
    openeo = _openeo()
    job, _ = openeo.create_triggered_job.return_value
    openeo.create_triggered_job.side_effect = [(job, True), (job, False)]  # the second finds the first
    automation = WorkflowAutomationService(config_loader=lambda: config, openeo_service=openeo)
    automation.consume([_event().model_copy(update={"event_id": "native-1:0"})])
    automation.consume([_event().model_copy(update={"event_id": "native-1:0"})])  # replayed: no second run

    (sync_run,) = runs.list_runs(task_id="sync-chirps")
    (workflow_run,) = runs.list_runs(task_id="agg")
    assert sync_run.cause == "cron" and sync_run.job_id == "native-1"
    assert workflow_run.cause == "event" and workflow_run.parent_run_id == sync_run.id
    assert workflow_run.job_kind == "openeo" and workflow_run.job_id == "job-1"

    chain = client.get(f"/runs/{sync_run.id}").json()
    assert [item["run"]["task_id"] for item in chain["caused"]] == ["agg"]


def test_a_refused_start_is_a_failed_run_and_failures_are_counted(instance: None) -> None:  # noqa: F811
    config = compile_tasks([_task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"})])
    openeo = _openeo()
    openeo.create_triggered_job.side_effect = RuntimeError("store locked")
    automation = WorkflowAutomationService(config_loader=lambda: config, openeo_service=openeo)

    automation.consume([_event().model_copy(update={"event_id": f"native-{index}:0"}) for index in range(3)])

    assert runs.consecutive_failures("agg") == 3
    assert {runs.view(run).status for run in runs.list_runs(task_id="agg")} == {"refused"}


def test_the_last_run_survives_a_restart(client: TestClient) -> None:  # noqa: F811
    store.save_task(_task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *"), create=True)

    def runner(task: Any, cause: str) -> CheckResult:
        return CheckResult(
            schedule_id=task.id, dataset_id=task.target, outcome=CheckOutcome.SUBMITTED, message="ok", job_id="job-9"
        )

    SchedulerService(task_runner=runner).run_task_now(store.get_task("nightly"), cause="manual:1")  # type: ignore[arg-type]

    # A new process: nothing in memory, the run read back from the database.
    from open_climate_service.scheduler import service as scheduler_service

    scheduler_service._service = None
    listed = client.get("/tasks/nightly").json()
    assert listed["last_run"]["cause"] == "manual" and listed["last_run"]["job_id"] == "job-9"
    assert listed["last_run"]["job_href"] == "/jobs/job-9"


def test_native_jobs_can_be_listed(client: TestClient) -> None:  # noqa: F811
    response = client.get("/ingestions/jobs")
    assert response.status_code == 200 and "jobs" in response.json()


# --- exports in the store (CLIM-1089, CLIM-1289) -------------------------------------------------------


def test_exports_are_saved_live_and_checked_by_their_plugin(
    client: TestClient,  # noqa: F811
    monkeypatch: Any,
) -> None:
    from open_climate_service import config as api_config

    monkeypatch.setattr(
        api_config,
        "_cache",
        {"dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "T"}]},
    )
    definition = {
        "plugin": "dhis2",
        "connection": "hmis",
        "period_type": "monthly",
        "series": [{"select": {}, "data_element": "BXgDHhPdFVU"}],
    }
    assert client.put("/exports/rain-monthly", json=definition).status_code == 200
    assert [item["id"] for item in client.get("/exports").json()["exports"]] == ["rain-monthly"]

    unknown = client.put("/exports/other", json={**definition, "connection": "nowhere"})
    assert unknown.status_code == 409 and "not configured under dhis2_connections" in unknown.json()["detail"]

    bad_mapping = client.put("/exports/bad", json={**definition, "series": []})
    assert bad_mapping.status_code == 409

    assert client.delete("/exports/rain-monthly").status_code == 204
    assert client.get("/exports").json()["exports"] == []


# --- the operational configuration as one document -----------------------------------------------------


def test_the_configuration_round_trips_as_one_document(client: TestClient) -> None:  # noqa: F811
    assert (
        client.post(
            "/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"}
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/tasks", json={"id": "agg", "kind": "workflow", "target": _WORKFLOW, "after": {"dataset": "chirps"}}
        ).status_code
        == 201
    )

    document = client.get("/configuration?format=yaml")
    assert document.headers["content-type"].startswith("application/yaml")

    for task_id in ("agg", "sync-chirps"):
        client.delete(f"/tasks/{task_id}")
    assert client.get("/tasks").json()["tasks"] == []

    restored = client.put("/configuration", content=document.content, headers={"Content-Type": "application/yaml"})
    assert restored.status_code == 200, restored.text
    assert [task["id"] for task in client.get("/tasks").json()["tasks"]] == ["agg", "sync-chirps"]


def test_a_document_with_one_bad_task_changes_nothing(client: TestClient) -> None:  # noqa: F811
    client.post("/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"})
    bad = {
        "tasks": [
            {"id": "sync-era5", "kind": "sync", "target": "era5", "cron": "0 6 * * *"},
            {"id": "x", "kind": "workflow", "target": "no_such_workflow"},
        ],
        "exports": [],
    }
    refused = client.put("/configuration", json=bad)
    assert refused.status_code == 409 and "unknown workflow" in refused.json()["detail"]
    assert [task["id"] for task in client.get("/tasks").json()["tasks"]] == ["sync-chirps"]
