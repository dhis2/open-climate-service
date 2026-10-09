"""CLIM-1213: deliver a triggered workflow's named export automatically when the job finishes."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from open_climate_service import config
from open_climate_service.automation import service as automation_module
from open_climate_service.automation.config import AutomationConfig, TriggerDelivery, WorkflowTrigger
from open_climate_service.automation.service import WorkflowAutomationService, delivery_idempotency_key
from open_climate_service.exports import ExportReport
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
from open_climate_service.exports.report import ExportOutcome
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent, JobRecord, JobStatus
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now

_TRIGGER = "rain-to-districts"
_EXPORT = "rain-monthly"
_OTHER_EXPORT = "rain-monthly-national"


def _event(index: int = 0) -> JobEvent:
    return JobEvent(
        event_id=f"native-job-{index}:0",
        time=utc_now(),
        type=DATASET_UPDATED_EVENT_TYPE,
        source="/datasets/rain_monthly",
        data={
            "dataset_id": "rain_monthly",
            "artifact_id": "artifact-1",
            "action": "append",
            "previous_end": "2026-07-31",
            "current_end": "2026-08-31",
        },
    )


def _automation(deliver: TriggerDelivery | None = None, max_attempts: int = 3) -> AutomationConfig:
    return AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id=_TRIGGER,
                on_update_of="rain_monthly",
                workflow_id="aggregate_to_dhis2_json",
                arguments={
                    "dataset_id": "$event.dataset_id",
                    "export": deliver.export if deliver is not None else _EXPORT,
                },
                deliver=deliver,
                max_attempts=max_attempts,
            )
        ]
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"geometry": ["ImspTQPwCqd"], "t": ["202608"], "rain": [12.5]})


@pytest.fixture(name="instance")
def instance_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A writable instance whose workflow jobs save a bound named DHIS2 export."""
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    settings: dict[str, Any] = {
        "exports": [
            {
                "id": export_id,
                "plugin": "dhis2",
                "connection": "hmis",
                "period_type": "monthly",
                "series": [{"data_element": "BXgDHhPdFVU"}],
            }
            for export_id in (_EXPORT, _OTHER_EXPORT)
        ],
        "dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "TEST_TOKEN"}],
    }
    monkeypatch.setattr(config, "_cache", settings)
    # The workflow graph itself is CLIM-1216's concern; here a job only has to save a
    # deliverable named export, so execution returns one directly. Tests switch which
    # export the workflow saves through `saved_export`.
    state: dict[str, Any] = {"saved_export": _EXPORT}
    monkeypatch.setattr(
        execution,
        "run_process_graph",
        lambda *args, **kwargs: SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": state["saved_export"]}),
    )
    # Deliveries run on the native job service singleton; never inherit one bound to
    # another test's store.
    job_service_module.reset_job_service()
    openeo = openeo_jobs.OpenEOJobService()
    try:
        yield {"settings": settings, "openeo": openeo, "state": state}
    finally:
        openeo.shutdown()
        job_service_module.reset_job_service()


@pytest.fixture(name="sent")
def sent_fixture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Fake DHIS2 client: record each send instead of contacting a server."""
    calls: list[dict[str, Any]] = []

    def send(
        self: Dhis2ExportPlugin, payload: bytes, target: Any, *, dry_run: bool = False, context: Any = None
    ) -> ExportReport:
        calls.append({"target": target, "dry_run": dry_run})
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            message="accepted",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=1,
            imported=0 if dry_run else 1,
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )

    monkeypatch.setattr(Dhis2ExportPlugin, "send", send)
    return calls


def _service(
    instance: dict[str, Any], deliver: TriggerDelivery | None, *, listen: bool = True, max_attempts: int = 3
) -> Any:
    service = WorkflowAutomationService(
        config_loader=lambda: _automation(deliver, max_attempts), openeo_service=instance["openeo"]
    )
    service.start()
    # `listen=False` models a process killed after the FINISHED write (which records the
    # delivery owed) and before the listener could submit it.
    instance["openeo"].set_delivery_due_provider(service.delivery_due_for)
    instance["openeo"].set_finished_listener(service.on_job_finished if listen else None)
    return service


def _await_openeo(job_id: str, timeout: float = 5.0) -> OpenEOJobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = openeo_jobs.store_get_job(job_id)
        if record is not None and record.status in {
            OpenEOJobStatus.FINISHED,
            OpenEOJobStatus.ERROR,
            OpenEOJobStatus.CANCELED,
        }:
            # The listener runs right after the FINISHED write; give it a moment to link.
            time.sleep(0.2)
            return openeo_jobs.store_get_job(job_id) or record
        time.sleep(0.02)
    pytest.fail(f"openEO job {job_id} did not finish within {timeout}s")


def _await_deliveries(count: int, timeout: float = 5.0) -> list[JobRecord]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = [record for record in native_store.list_job_records() if record.process_id.startswith("export:")]
        if len(records) >= count and all(
            record.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED} for record in records
        ):
            return records
        time.sleep(0.05)
    pytest.fail(f"expected {count} finished delivery jobs within {timeout}s")


def _deliveries() -> list[JobRecord]:
    return [record for record in native_store.list_job_records() if record.process_id.startswith("export:")]


def _triggered_job_id(service: Any, index: int = 0) -> str:
    service.consume([_event(index)])
    jobs = [record for record in openeo_jobs.store_list_jobs() if record.source_event_id == _event(index).event_id]
    assert len(jobs) == 1
    return jobs[0].id


def test_finished_triggered_job_delivers_once_and_links_the_delivery(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)
    source = _await_openeo(job_id)
    assert source.status == OpenEOJobStatus.FINISHED
    assert source.trigger_id == _TRIGGER

    [delivery] = _await_deliveries(1)
    assert delivery.status == JobStatus.SUCCESSFUL
    assert delivery.request["dry_run"] is True  # the default
    assert isinstance(delivery.result, dict) and delivery.result["outcome"] == ExportOutcome.DRY_RUN
    assert sent == [{"target": "hmis", "dry_run": True}]
    links = (openeo_jobs.store_get_job(job_id).usage or {})["deliveries"]  # type: ignore[union-attr]
    assert [link["delivery_job_id"] for link in links] == [delivery.job_id]


def test_failed_submission_is_visible_and_cleared_after_restart_retry(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.exports import delivery as delivery_module

    original = delivery_module.submit_delivery

    def fail_submission(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("delivery queue unavailable")

    monkeypatch.setattr(delivery_module, "submit_delivery", fail_submission)
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)
    _await_openeo(job_id)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stored = openeo_jobs.store_get_job(job_id)
        error = (stored.usage or {}).get("delivery_error") if stored is not None else None
        if error is not None:
            break
        time.sleep(0.01)
    else:
        pytest.fail("delivery submission error was not recorded on the source job")

    assert error["export_id"] == _EXPORT
    assert error["message"] == "RuntimeError: delivery queue unavailable"
    assert _deliveries() == []

    # Submission retries are deliberately restart/reconciliation driven for CLIM-1213.
    monkeypatch.setattr(delivery_module, "submit_delivery", original)
    service.reconcile_deliveries()
    _await_deliveries(1)
    stored = openeo_jobs.store_get_job(job_id)
    assert stored is not None
    assert "delivery_error" not in (stored.usage or {})


def test_replayed_event_and_repeated_reconciliation_create_no_second_delivery(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)
    _await_openeo(job_id)
    _await_deliveries(1)

    service.consume([_event()])  # the same durable event again
    service.reconcile_deliveries()
    service.reconcile_deliveries()
    _service(instance, TriggerDelivery(export=_EXPORT)).reconcile_deliveries()  # a restart

    assert len(_deliveries()) == 1
    assert len(sent) == 1


def test_switching_to_live_delivers_new_jobs_and_leaves_earlier_ones(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    first = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT, dry_run=True)))
    _await_openeo(first)
    _await_deliveries(1)

    live = _service(instance, TriggerDelivery(export=_EXPORT, dry_run=False))
    live.reconcile_deliveries()
    assert len(_deliveries()) == 1  # the earlier job is not re-delivered live
    second = _triggered_job_id(live, index=1)
    _await_openeo(second)
    deliveries = _await_deliveries(2)

    assert all(record.status == JobStatus.SUCCESSFUL for record in deliveries)
    modes = {record.request["job_id"]: record.request["dry_run"] for record in deliveries}
    assert modes == {first: True, second: False}
    assert sent[-1] == {"target": "hmis", "dry_run": False}


def test_undelivered_dry_run_job_is_not_imported_after_switching_to_live(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    # A job finishes under dry_run, but its delivery is missed (no listener, as after a crash).
    dry = _service(instance, TriggerDelivery(export=_EXPORT, dry_run=True), listen=False)
    missed = _triggered_job_id(dry)
    _await_openeo(missed)
    assert _deliveries() == []

    live = _service(instance, TriggerDelivery(export=_EXPORT, dry_run=False))
    live.reconcile_deliveries()
    assert _deliveries() == []  # never imported live
    assert sent == []

    later = _triggered_job_id(live, index=1)
    _await_openeo(later)
    [delivery] = _await_deliveries(1)
    assert (delivery.request["job_id"], delivery.request["dry_run"]) == (later, False)
    assert sent == [{"target": "hmis", "dry_run": False}]


def test_boundary_without_a_recorded_mode_is_restamped(instance: dict[str, Any]) -> None:
    # A boundary stored before the mode was recorded cannot prove which mode it covered.
    automation_module._save_delivery_activations(
        {_TRIGGER: {"export": _EXPORT, "activated_at": "2020-01-01T00:00:00+00:00"}}  # type: ignore[dict-item]
    )
    _service(instance, TriggerDelivery(export=_EXPORT, dry_run=False))
    stored = automation_module._load_delivery_activations()[_TRIGGER]
    assert stored["mode"] == "live"
    assert stored["activated_at"] > "2020-01-01T00:00:00+00:00"


def test_adding_deliver_does_not_push_jobs_finished_before_activation(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    without = _service(instance, None)
    job_id = _triggered_job_id(without)
    assert _await_openeo(job_id).status == OpenEOJobStatus.FINISHED

    later = _service(instance, TriggerDelivery(export=_EXPORT))
    later.reconcile_deliveries()
    later.on_job_finished(openeo_jobs.store_get_job(job_id))  # type: ignore[arg-type]

    assert _deliveries() == []
    assert sent == []


def test_removing_and_readding_deliver_resets_the_boundary(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    _service(instance, TriggerDelivery(export=_EXPORT))
    without = _service(instance, None)  # delivery removed while jobs keep finishing
    job_id = _triggered_job_id(without)
    _await_openeo(job_id)

    _service(instance, TriggerDelivery(export=_EXPORT)).reconcile_deliveries()
    assert _deliveries() == []


@pytest.mark.parametrize("status", [OpenEOJobStatus.ERROR, OpenEOJobStatus.CANCELED])
def test_failed_or_canceled_job_produces_no_delivery(
    instance: dict[str, Any], sent: list[dict[str, Any]], status: OpenEOJobStatus
) -> None:
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    record = OpenEOJobRecord(
        id="triggered",
        status=status,
        created=utc_now(),
        trigger_id=_TRIGGER,
        finished_at=utc_now() + timedelta(seconds=1),
    )
    openeo_jobs.store_create_job(record)
    service.on_job_finished(record)
    service.reconcile_deliveries()
    assert _deliveries() == []


def test_failed_workflow_job_produces_no_delivery(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(execution, "run_process_graph", fail)
    # A single attempt: this test is about delivery, not retries (tests/test_workflow_retries.py).
    service = _service(instance, TriggerDelivery(export=_EXPORT), max_attempts=1)
    job_id = _triggered_job_id(service)
    assert _await_openeo(job_id).status == OpenEOJobStatus.ERROR
    service.reconcile_deliveries()
    assert _deliveries() == []


def test_job_finished_while_no_listener_ran_delivers_on_next_start(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    # Simulates a process killed after FINISHED and before the delivery was submitted.
    stopped = _service(instance, TriggerDelivery(export=_EXPORT), listen=False)
    job_id = _triggered_job_id(stopped)
    _await_openeo(job_id)
    assert _deliveries() == []

    _service(instance, TriggerDelivery(export=_EXPORT)).reconcile_deliveries()
    [delivery] = _await_deliveries(1)
    assert delivery.request["job_id"] == job_id
    assert delivery.status == JobStatus.SUCCESSFUL


def test_crash_between_reservation_and_link_is_repaired(instance: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)
    _await_openeo(job_id)
    [delivery] = _await_deliveries(1)
    # Simulate a crash after the delivery was reserved and enqueued but before it was linked.
    openeo_jobs.store_update_job(
        job_id, lambda r: r.model_copy(update={"usage": {**(r.usage or {}), "deliveries": []}})
    )

    service.reconcile_deliveries()

    links = (openeo_jobs.store_get_job(job_id).usage or {})["deliveries"]  # type: ignore[union-attr]
    assert [link["delivery_job_id"] for link in links] == [delivery.job_id]
    assert len(_deliveries()) == 1


def test_delivery_keys_are_deterministic_per_job_export_and_mode() -> None:
    assert delivery_idempotency_key("job", "rain", True) == "auto:job:rain:dry-run"
    assert delivery_idempotency_key("job", "rain", False) == "auto:job:rain:live"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda settings: settings.update(exports=[]), "which is invalid"),
        (lambda settings: settings["exports"][0].update(plugin="summary"), "which is invalid"),
        (lambda settings: settings["exports"][0].pop("connection"), "which has no DHIS2 connection"),
        (lambda settings: settings.update(dhis2_connections=[]), "is not configured under dhis2_connections"),
    ],
)
def test_invalid_delivery_is_rejected_at_startup_naming_the_trigger(
    instance: dict[str, Any], change: Any, message: str
) -> None:
    change(instance["settings"])
    service = WorkflowAutomationService(
        config_loader=lambda: _automation(TriggerDelivery(export=_EXPORT)), openeo_service=instance["openeo"]
    )
    with pytest.raises(ValueError, match=message) as error:
        service.start()
    assert f"Workflow task '{_TRIGGER}' delivers export '{_EXPORT}'" in str(error.value)


def test_workflow_and_delivery_exports_must_match(instance: dict[str, Any]) -> None:
    automation = _automation(TriggerDelivery(export=_EXPORT))
    automation.workflow_triggers[0].arguments["export"] = _OTHER_EXPORT
    service = WorkflowAutomationService(config_loader=lambda: automation, openeo_service=instance["openeo"])

    with pytest.raises(ValueError, match="workflow and delivery must use the same named export"):
        service.start()


def test_read_only_instance_validates_delivery_but_keeps_it_inactive(instance: dict[str, Any]) -> None:
    instance["settings"]["read_only"] = True
    service = WorkflowAutomationService(
        config_loader=lambda: _automation(TriggerDelivery(export=_EXPORT)), openeo_service=instance["openeo"]
    )

    service.start()

    assert service._delivery_steps == {}
    assert automation_module._load_delivery_activations() == {}
    finished = utc_now()
    record = OpenEOJobRecord(
        id="read-only",
        status=OpenEOJobStatus.FINISHED,
        created=finished,
        finished_at=finished,
        trigger_id=_TRIGGER,
    )
    assert service.delivery_due_for(record) is None


def test_job_without_finish_time_counts_as_before_activation(instance: dict[str, Any]) -> None:
    record = OpenEOJobRecord(
        id="legacy", status=OpenEOJobStatus.FINISHED, created=datetime(2026, 1, 1, tzinfo=UTC), trigger_id=_TRIGGER
    )
    assert automation_module._finished_after(record, utc_now()) is False


def test_switching_export_starts_a_new_boundary(instance: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    before = _service(instance, TriggerDelivery(export=_EXPORT))
    delivered = _triggered_job_id(before)
    _await_openeo(delivered)
    _await_deliveries(1)
    # The workflow switched exports before the trigger did: this job saved the new export,
    # which the trigger's old export refuses, so it finishes undelivered.
    instance["state"]["saved_export"] = _OTHER_EXPORT
    undelivered = _triggered_job_id(before, index=1)
    _await_openeo(undelivered)
    assert len(_deliveries()) == 1

    switched = _service(instance, TriggerDelivery(export=_OTHER_EXPORT))
    switched.reconcile_deliveries()
    # Both jobs finished before the switch, so neither is delivered to the new export,
    # even the one whose saved result would verify against it.
    assert len(_deliveries()) == 1

    after = _triggered_job_id(switched, index=2)
    _await_openeo(after)
    deliveries = _await_deliveries(2)

    assert all(record.status == JobStatus.SUCCESSFUL for record in deliveries)
    targets = {record.request["job_id"]: record.request["export_id"] for record in deliveries}
    assert targets == {delivered: _EXPORT, after: _OTHER_EXPORT}


def test_reconciliation_reads_boundaries_once_per_start(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(3):
        openeo_jobs.store_create_job(
            OpenEOJobRecord(
                id=f"history-{index}",
                status=OpenEOJobStatus.FINISHED,
                created=utc_now(),
                trigger_id=_TRIGGER,
                finished_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    reads: list[None] = []
    original = automation_module._load_delivery_activations
    monkeypatch.setattr(automation_module, "_load_delivery_activations", lambda: reads.append(None) or original())

    service.reconcile_deliveries()
    service.on_job_finished(openeo_jobs.store_get_job("history-0"))  # type: ignore[arg-type]

    assert reads == []
    assert _deliveries() == []


def test_trigger_fields_survive_persistence(instance: dict[str, Any]) -> None:
    finished = utc_now()
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="persisted",
            status=OpenEOJobStatus.FINISHED,
            created=finished,
            trigger_id=_TRIGGER,
            source_event_id="event-1",
            finished_at=finished,
        )
    )
    stored = openeo_jobs.store_get_job("persisted")
    assert stored is not None
    assert (stored.trigger_id, stored.source_event_id, stored.finished_at) == (_TRIGGER, "event-1", finished)
    # Internal fields stay out of openEO API responses.
    assert "trigger_id" not in stored.model_dump()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda settings: settings["exports"].append(dict(settings["exports"][0])), "which is invalid"),
        (lambda settings: settings["exports"][0].update(series=[]), "which is invalid"),
    ],
)
def test_export_that_delivery_would_reject_fails_startup(instance: dict[str, Any], change: Any, message: str) -> None:
    change(instance["settings"])
    service = WorkflowAutomationService(
        config_loader=lambda: _automation(TriggerDelivery(export=_EXPORT)), openeo_service=instance["openeo"]
    )
    with pytest.raises(ValueError, match=message):
        service.start()


def test_cancel_arriving_while_the_result_is_saved_wins(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    openeo = instance["openeo"]
    original = openeo._persist_result

    def cancel_during_save(job_id: str, result: Any) -> Any:
        openeo_jobs.store_update_job(job_id, lambda r: r.model_copy(update={"cancel_requested": True}))
        return original(job_id, result)

    monkeypatch.setattr(openeo, "_persist_result", cancel_during_save)
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)

    record = _await_openeo(job_id)
    assert record.status == OpenEOJobStatus.CANCELED
    service.reconcile_deliveries()
    assert _deliveries() == []


@pytest.mark.parametrize(
    "content",
    [
        "not a mapping",
        {"export": "rain-monthly", "mode": "dry-run", "activated_at": "yesterday"},
    ],
)
def test_corrupt_boundary_is_restamped(instance: dict[str, Any], content: Any) -> None:
    from open_climate_service.state import db

    db.save_activations("delivery", {_TRIGGER: content})

    _service(instance, TriggerDelivery(export=_EXPORT))

    stored = automation_module._load_delivery_activations()[_TRIGGER]
    assert automation_module._parse_time(stored["activated_at"]) is not None


def test_job_finishing_while_delivery_is_off_is_never_delivered(
    instance: dict[str, Any], sent: list[dict[str, Any]]
) -> None:
    _service(instance, TriggerDelivery(export=_EXPORT))  # boundary stamped
    # A read-only start without `deliver` cannot clear the boundary, but a job finishing
    # then records that it owes nothing.
    instance["settings"]["read_only"] = True
    offline = _service(instance, None)
    instance["settings"]["read_only"] = False  # consume() itself refuses read-only instances
    job_id = _triggered_job_id(offline)
    record = _await_openeo(job_id)
    assert record.status == OpenEOJobStatus.FINISHED
    assert record.delivery_due is None

    _service(instance, TriggerDelivery(export=_EXPORT)).reconcile_deliveries()
    assert _deliveries() == []


def test_rerun_after_switching_to_live_is_not_imported(instance: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT, dry_run=True)))
    _await_openeo(job_id)
    [dry] = _await_deliveries(1)

    _service(instance, TriggerDelivery(export=_EXPORT, dry_run=False))
    instance["openeo"].start_job(job_id)  # a manual re-run
    rerun = _await_openeo(job_id)

    assert rerun.status == OpenEOJobStatus.FINISHED
    assert [link["delivery_job_id"] for link in (rerun.usage or {})["deliveries"]] == [dry.job_id]
    assert len(_deliveries()) == 1
    assert sent == [{"target": "hmis", "dry_run": True}]


def test_cancel_losing_the_race_to_completion_is_refused(
    instance: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    openeo = instance["openeo"]
    finished = utc_now()
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="raced",
            status=OpenEOJobStatus.FINISHED,
            created=finished,
            trigger_id=_TRIGGER,
            finished_at=finished,
            delivery_due={"export": _EXPORT, "mode": "dry-run"},
        )
    )
    # The cancel request read the job while it was still running; the worker finished it
    # before the cancel request's own store write.
    stale = openeo_jobs.store_get_job("raced").model_copy(update={"status": OpenEOJobStatus.RUNNING})  # type: ignore[union-attr]
    monkeypatch.setattr(openeo, "get_job_or_404", lambda job_id: stale)

    with pytest.raises(HTTPException) as error:
        openeo.cancel_job("raced")

    assert error.value.status_code == 400
    stored = openeo_jobs.store_get_job("raced")
    assert stored is not None
    assert stored.status == OpenEOJobStatus.FINISHED
    assert stored.cancel_requested is False


def test_cancel_winning_the_race_is_recorded_for_the_worker(instance: dict[str, Any]) -> None:
    openeo = instance["openeo"]
    openeo_jobs.store_create_job(OpenEOJobRecord(id="running", status=OpenEOJobStatus.RUNNING, created=utc_now()))
    openeo.cancel_job("running")
    stored = openeo_jobs.store_get_job("running")
    assert stored is not None and stored.cancel_requested is True
    # The worker's finishing mutation then records CANCELED, not FINISHED.
    assert openeo._finish(stored, None).status == OpenEOJobStatus.CANCELED
