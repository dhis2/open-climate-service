"""Tasks: every kind of automated work in one store, on a cron, after a change or by hand (CLIM-1378)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.automation import service as automation_service
from open_climate_service.automation.config import compile_tasks, get_automation_config
from open_climate_service.automation.service import MAX_CHAIN_DEPTH, WorkflowAutomationService
from open_climate_service.jobs.models import COLLECTION_UPDATED_EVENT_TYPE, DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult
from open_climate_service.scheduler.service import SchedulerService
from open_climate_service.tasks import store
from open_climate_service.tasks.models import Task
from tests.test_schedules import client, instance  # noqa: F401  # pyright: ignore[reportUnusedImport]

_WORKFLOW = "aggregate_to_chap_csv"
_ARGUMENTS = {
    "dataset_id": "$event.dataset_id",
    "temporal_extent": ["$event.previous_end", "$event.current_end"],
    "method": "mean",
}


def _task(**values: Any) -> Task:
    return Task.model_validate(values)


def _event(event_type: str = DATASET_UPDATED_EVENT_TYPE, **data: Any) -> JobEvent:
    return JobEvent(
        event_id="native-job:0",
        time=datetime(2026, 10, 9, tzinfo=UTC),
        type=event_type,
        source="/test",
        data={"dataset_id": "chirps", "previous_end": "2026-09-30", "current_end": "2026-10-08", **data},
    )


def _openeo() -> MagicMock:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (
        OpenEOJobRecord(
            id="job-1",
            process={"process_graph": {}},
            status=OpenEOJobStatus.CREATED,
            created=datetime(2026, 10, 9, tzinfo=UTC),
        ),
        True,
    )
    return openeo


# --- the model -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"kind": "sync", "after": {"dataset": "x"}}, "cannot wait"),
        ({"kind": "refresh", "after": {"dataset": "x"}}, "cannot wait"),
        ({"kind": "workflow", "after": {"task": "x"}}, "not after another task"),
        ({"kind": "deliver"}, "after the workflow task"),
        ({"kind": "deliver", "after": {"task": "x"}, "cron": "0 6 * * *"}, "a task runs on a cron or after"),
        ({"kind": "sync", "arguments": {"a": 1}}, "only a workflow task takes arguments"),
        ({"kind": "workflow", "after": {"dataset": "x", "collection": "y"}}, "exactly one"),
        ({"kind": "sync", "cron": "every day"}, "invalid five-field cron"),
    ],
)
def test_each_kind_says_when_in_the_ways_it_can(values: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _task(id="s", target="t", **values)


def test_a_task_says_how_it_starts() -> None:
    assert _task(id="a", kind="sync", target="chirps", cron="0 6 * * *").when == "cron 0 6 * * *"
    assert _task(id="b", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"}).when == "after dataset chirps"
    assert _task(id="c", kind="refresh", target="districts").when == "by hand"


# --- workflow and deliver tasks are the automation configuration ----------------------------------


def test_workflow_and_deliver_tasks_compile_to_triggers() -> None:
    config = compile_tasks(
        [
            _task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"}, arguments=_ARGUMENTS),
            _task(id="send", kind="deliver", target="chirps-daily", after={"task": "agg"}, dry_run=False),
            _task(id="by-org-units", kind="workflow", target=_WORKFLOW, after={"collection": "districts"}),
            _task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *"),
        ]
    )
    triggers = {trigger.id: trigger for trigger in config.workflow_triggers}

    assert triggers["agg"].on_update_of == "chirps"
    assert triggers["agg"].deliver is not None and triggers["agg"].deliver.export == "chirps-daily"
    assert triggers["agg"].deliver.dry_run is False
    assert triggers["by-org-units"].event == COLLECTION_UPDATED_EVENT_TYPE
    assert triggers["nightly"].on_update_of is None


def test_a_paused_task_drops_out_of_automation() -> None:
    config = compile_tasks(
        [
            _task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"}),
            _task(id="send", kind="deliver", target="x", after={"task": "agg"}, enabled=False),
            _task(id="off", kind="workflow", target=_WORKFLOW, after={"dataset": "era5"}, enabled=False),
        ]
    )
    assert [(trigger.id, trigger.deliver) for trigger in config.workflow_triggers] == [("agg", None)]


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ([_task(id="send", kind="deliver", target="x", after={"task": "nope"})], "not a workflow task"),
        (
            [
                _task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"}),
                _task(id="one", kind="deliver", target="x", after={"task": "agg"}),
                _task(id="two", kind="deliver", target="y", after={"task": "agg"}),
            ],
            "delivered once",
        ),
    ],
)
def test_deliver_tasks_follow_exactly_one_workflow_task(tasks: list[Task], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compile_tasks(tasks)


def test_automation_reads_the_store_not_the_config_file(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    store.save_task(_task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"}), create=True)
    assert [trigger.id for trigger in get_automation_config().workflow_triggers] == ["agg"]

    monkeypatch.setattr(api_config, "_cache", {"automation": {"workflow_triggers": []}})
    with pytest.raises(ValueError, match="no longer supported"):
        get_automation_config()


# --- events from every task, and chains --------------------------------------------------------------


def test_a_collection_refresh_starts_the_task_after_it() -> None:
    config = compile_tasks(
        [_task(id="by-org-units", kind="workflow", target=_WORKFLOW, after={"collection": "districts"})]
    )
    openeo = _openeo()
    service = WorkflowAutomationService(config_loader=lambda: config, openeo_service=openeo)

    service.consume([_event(COLLECTION_UPDATED_EVENT_TYPE, collection_id="districts")])
    service.consume([_event(COLLECTION_UPDATED_EVENT_TYPE, collection_id="regions")])
    service.consume([_event(DATASET_UPDATED_EVENT_TYPE, dataset_id="districts")])

    assert openeo.create_triggered_job.call_count == 1


def test_a_chain_carries_its_depth_and_stops_at_the_limit() -> None:
    """A derived dataset's update starts the next task one level deeper; a cycle stops."""
    config = compile_tasks([_task(id="agg", kind="workflow", target=_WORKFLOW, after={"dataset": "chirps"})])
    openeo = _openeo()
    service = WorkflowAutomationService(config_loader=lambda: config, openeo_service=openeo)

    service.consume([_event(chain_depth=2)])
    assert openeo.create_triggered_job.call_args.kwargs["chain_depth"] == 3

    openeo.reset_mock()
    service.consume([_event(chain_depth=MAX_CHAIN_DEPTH)])
    openeo.create_triggered_job.assert_not_called()


def test_a_task_that_would_start_itself_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = MagicMock(parameters=[{"name": "output_dataset_id"}])
    monkeypatch.setattr(automation_service.workflows, "get_workflow", lambda _id: workflow)
    config = compile_tasks(
        [
            _task(
                id="loop",
                kind="workflow",
                target="aggregate_dekads_to_period",
                after={"dataset": "rain_monthly"},
                arguments={"output_dataset_id": "rain_monthly"},
            )
        ]
    )
    with pytest.raises(ValueError, match="would start itself"):
        automation_service.validate_automation(config)


def test_a_workflow_publish_emits_an_update_with_its_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIM-1243: what a sync announces, a workflow that publishes a dataset announces too."""
    from open_climate_service.ingestions import processes
    from open_climate_service.openeo import jobs

    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(processes, "record_inline_update", lambda **kwargs: recorded.append(kwargs))
    artifact = MagicMock(dataset_id="rain_monthly", artifact_id="a-1")
    artifact.coverage.temporal.start = "2025-01-01"
    artifact.coverage.temporal.end = "2025-03-01"

    jobs._record_publish_event(artifact, job_id="job-7", chain_depth=2)

    (event,) = recorded[0]["events"]
    assert event.type == DATASET_UPDATED_EVENT_TYPE
    assert event.data["dataset_id"] == "rain_monthly"
    assert event.data["chain_depth"] == 2
    assert event.data["producing_job_id"] == "job-7"
    assert event.data["action"] == "rematerialize" and event.data["previous_end"] is None


# --- cron and by hand -------------------------------------------------------------------------------


def test_a_workflow_task_runs_by_hand_with_a_deterministic_job() -> None:
    config = compile_tasks([_task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *")])
    openeo = _openeo()
    service = WorkflowAutomationService(config_loader=lambda: config, openeo_service=openeo)

    assert service.run_now("nightly", "cron:2026-10-09T02:00:00+00:00") == "job-1"
    assert openeo.create_triggered_job.call_args.kwargs["source_event_id"] == "cron:2026-10-09T02:00:00+00:00:nightly"
    with pytest.raises(ValueError, match="No enabled workflow task"):
        service.run_now("missing", "manual:1")


def test_the_clock_runs_refresh_and_workflow_tasks_beside_syncs(instance: None) -> None:  # noqa: F811
    tasks = [
        _task(id="org-units", kind="refresh", target="districts", cron="0 1 * * 1"),
        _task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *"),
    ]
    ran: list[tuple[str, str]] = []

    def runner(task: Task, cause: str) -> CheckResult:
        ran.append((task.id, cause))
        return CheckResult(schedule_id=task.id, dataset_id=task.target, outcome=CheckOutcome.SUBMITTED, message="ok")

    scheduler = MagicMock()
    scheduler.get_jobs.return_value = []
    service = SchedulerService(tasks_loader=lambda: tasks, task_runner=runner)
    service._scheduler = scheduler
    service._leader = True
    service.reload()

    added = {call.kwargs["id"] for call in scheduler.add_job.call_args_list}
    assert {"task:org-units", "task:nightly"} <= added

    service.run_task_now(tasks[1], cause="cron:x")
    assert ran == [("nightly", "cron:x")]
    assert service.task_status("nightly")[1] is not None


# --- the API ----------------------------------------------------------------------------------------


def test_a_sync_task_is_the_dataset_schedule(client: TestClient) -> None:  # noqa: F811
    created = client.post("/tasks", json={"id": "sync-chirps", "kind": "sync", "target": "chirps", "cron": "0 6 * * *"})
    assert created.status_code == 201, created.text
    assert created.json()["starts"] == "cron 0 6 * * *"

    schedules = client.get("/schedules", headers={"Accept": "application/json"}).json()["schedules"]
    assert [item["dataset_id"] for item in schedules] == ["chirps"]

    duplicate = client.post("/tasks", json={"id": "another", "kind": "sync", "target": "chirps", "cron": "0 7 * * *"})
    assert duplicate.status_code == 409 and "one sync schedule" in duplicate.json()["detail"]


def test_workflow_tasks_are_created_paused_and_removed_live(client: TestClient) -> None:  # noqa: F811
    body = {
        "id": "agg",
        "kind": "workflow",
        "target": _WORKFLOW,
        "after": {"dataset": "chirps"},
        "arguments": _ARGUMENTS,
    }
    assert client.post("/tasks", json=body).status_code == 201
    assert [trigger.id for trigger in get_automation_config().workflow_triggers] == ["agg"]

    assert client.post("/tasks/agg/pause").json()["enabled"] is False
    assert get_automation_config().workflow_triggers == []

    assert client.delete("/tasks/agg").status_code == 204
    assert client.get("/tasks").json()["tasks"] == []


def test_the_api_refuses_what_could_only_fail_later(client: TestClient) -> None:  # noqa: F811
    unknown = client.post("/tasks", json={"id": "x", "kind": "workflow", "target": "no_such_workflow"})
    assert unknown.status_code == 409 and "unknown workflow" in unknown.json()["detail"]

    orphan = client.post("/tasks", json={"id": "send", "kind": "deliver", "target": "e", "after": {"task": "nope"}})
    assert orphan.status_code == 409 and "not a workflow task" in orphan.json()["detail"]

    static = client.post("/tasks", json={"id": "w", "kind": "sync", "target": "worldpop", "cron": "0 6 * * *"})
    assert static.status_code == 409 and "not syncable" in static.json()["detail"]


def test_a_deliver_task_cannot_run_alone(client: TestClient) -> None:  # noqa: F811
    client.post("/tasks", json={"id": "agg", "kind": "workflow", "target": _WORKFLOW, "after": {"dataset": "chirps"}})
    store.save_task(_task(id="send", kind="deliver", target="e", after={"task": "agg"}), create=True)

    assert client.post("/tasks/send/run").status_code == 409
    removing_upstream = client.delete("/tasks/agg")
    assert removing_upstream.status_code == 409 and "not a workflow task" in removing_upstream.json()["detail"]


# --- one clock across processes (CLIM-997) ----------------------------------------------------------


def test_the_clock_lease_has_one_holder_until_it_expires() -> None:
    from open_climate_service.state import db

    assert db.acquire_lease("scheduler", "a", 90, now=1000)
    assert not db.acquire_lease("scheduler", "b", 90, now=1050)
    assert db.acquire_lease("scheduler", "a", 90, now=1050)  # renewed
    assert db.lease_holder("scheduler", now=1100) == "a"
    assert db.acquire_lease("scheduler", "b", 90, now=1141)  # a stopped renewing
    db.release_lease("scheduler", "b")
    assert db.lease_holder("scheduler", now=1142) is None


def test_two_processes_run_one_clock_and_the_standby_takes_over(
    instance: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_climate_service.state import db

    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"enabled": True}})
    store.save_task(_task(id="nightly", kind="workflow", target=_WORKFLOW, cron="0 2 * * *"), create=True)
    clock = {"now": 1000.0}

    def lease(holder: str, ttl: float) -> bool:
        return db.acquire_lease("scheduler", holder, ttl, now=clock["now"])

    clocks: list[MagicMock] = []

    def fake_scheduler(**_: Any) -> MagicMock:
        fake = MagicMock()
        fake.get_jobs.return_value = []
        clocks.append(fake)
        return fake

    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", fake_scheduler)
    first = SchedulerService(lease=lease)
    second = SchedulerService(lease=lease)
    first.start()
    second.start()

    def task_jobs(fake: MagicMock) -> set[str]:
        return {call.kwargs["id"] for call in fake.add_job.call_args_list if call.kwargs["id"].startswith("task:")}

    assert task_jobs(clocks[0]) == {"task:nightly"} and task_jobs(clocks[1]) == set()
    assert first.status().running and not second.status().running

    clock["now"] += 200  # the first process hangs past the lease's expiry
    second.watch()
    assert task_jobs(clocks[1]) == {"task:nightly"}
    assert second.status().clock_holder is not None
