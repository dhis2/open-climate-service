"""Tests for durable dataset-update workflow automation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger
from open_climate_service.automation.service import WorkflowAutomationService, _resolve_event_values
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo.jobs import OpenEOJobService
from open_climate_service.openeo.schemas import OpenEOJobCreate, OpenEOJobRecord, OpenEOJobStatus


def _event(dataset_id: str = "chirps") -> JobEvent:
    return JobEvent(
        event_id="native-job:0",
        time=datetime(2026, 8, 19, tzinfo=UTC),
        type=DATASET_UPDATED_EVENT_TYPE,
        source=f"/datasets/{dataset_id}",
        data={
            "dataset_id": dataset_id,
            "artifact_id": "artifact-2",
            "action": "append",
            "previous_end": "2026-08-17",
            "current_end": "2026-08-18",
        },
    )


def _config() -> AutomationConfig:
    return AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id="chap-after-chirps",
                on_update_of="chirps",
                workflow_id="aggregate_to_chap_csv",
                arguments={
                    "dataset_id": "$event.dataset_id",
                    "temporal_extent": ["$event.previous_end", "$event.current_end"],
                    "method": "mean",
                },
            )
        ]
    )


def _openeo_record(status: OpenEOJobStatus = OpenEOJobStatus.CREATED) -> OpenEOJobRecord:
    return OpenEOJobRecord(
        id="triggered-job",
        process={"process_graph": {}},
        status=status,
        created=datetime(2026, 8, 19, tzinfo=UTC),
    )


def test_trigger_ids_must_be_unique() -> None:
    trigger = _config().workflow_triggers[0]
    with pytest.raises(ValidationError, match="must be unique"):
        AutomationConfig(workflow_triggers=[trigger, trigger])


def test_event_references_are_resolved_recursively() -> None:
    assert _resolve_event_values(
        {"source": "$event.dataset_id", "range": ["$event.previous_end", "$event.current_end"]},
        _event(),
    ) == {"source": "chirps", "range": ["2026-08-17", "2026-08-18"]}


def test_matching_update_submits_and_starts_workflow_once() -> None:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(), True)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)

    service.consume([_event()])

    body = openeo.create_triggered_job.call_args.args[0]
    node = body.process["process_graph"]["workflow"]
    assert node == {
        "process_id": "aggregate_to_chap_csv",
        "arguments": {
            "dataset_id": "chirps",
            "temporal_extent": ["2026-08-17", "2026-08-18"],
            "method": "mean",
        },
        "result": True,
    }
    assert openeo.create_triggered_job.call_args.kwargs == {
        "source_event_id": "native-job:0",
        "trigger_id": "chap-after-chirps",
    }
    openeo.start_triggered_job.assert_called_once_with("triggered-job")


def test_nonmatching_and_non_update_events_are_ignored() -> None:
    openeo = MagicMock()
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)
    other = _event("era5")
    unrelated = other.model_copy(update={"type": "dataset.created"})

    service.consume([other, unrelated])

    openeo.create_triggered_job.assert_not_called()


def test_replay_starts_an_existing_created_job_after_interrupted_submission() -> None:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(), False)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)

    service.consume([_event()])

    openeo.start_triggered_job.assert_called_once_with("triggered-job")


def test_replay_delegates_atomic_claim_for_an_existing_queued_job() -> None:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(OpenEOJobStatus.QUEUED), False)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)

    service.consume([_event()])

    openeo.start_triggered_job.assert_called_once_with("triggered-job")


def test_triggered_job_creation_is_deterministic_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict[str, object]] = []

    def mutate(mutation: Callable[[list[dict[str, object]]], Any]) -> Any:
        return mutation(records)

    monkeypatch.setattr("open_climate_service.openeo.jobs._mutate_store", mutate)
    service = OpenEOJobService()
    body = OpenEOJobCreate(process={"process_graph": {"result": {"process_id": "constant"}}})
    try:
        first, first_created = service.create_triggered_job(
            body,
            source_event_id="native-job:0",
            trigger_id="chap-after-chirps",
        )
        second, second_created = service.create_triggered_job(
            body,
            source_event_id="native-job:0",
            trigger_id="chap-after-chirps",
        )
    finally:
        service.shutdown()

    assert first.id == second.id
    assert first_created is True
    assert second_created is False
    assert len(records) == 1


def test_triggered_job_start_is_an_atomic_one_time_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict[str, object]] = []

    def mutate(mutation: Callable[[list[dict[str, object]]], Any]) -> Any:
        return mutation(records)

    monkeypatch.setattr("open_climate_service.openeo.jobs._mutate_store", mutate)
    service = OpenEOJobService()
    enqueue = MagicMock()
    monkeypatch.setattr(service, "_enqueue", enqueue)
    body = OpenEOJobCreate(process={"process_graph": {"result": {"process_id": "constant"}}})
    try:
        job, _ = service.create_triggered_job(
            body,
            source_event_id="native-job:0",
            trigger_id="chap-after-chirps",
        )

        assert service.start_triggered_job(job.id) is True
        assert service.start_triggered_job(job.id) is False
        enqueue.assert_called_once_with(job.id)
    finally:
        service.shutdown()


def _record_with(event: JobEvent) -> MagicMock:
    record = MagicMock()
    record.events = [event]
    return record


def test_new_trigger_does_not_backfill_events_before_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(), True)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)

    activation = datetime(2026, 8, 19, tzinfo=UTC)
    old_event = _event().model_copy(update={"time": datetime(2026, 8, 18, tzinfo=UTC)})
    monkeypatch.setattr(
        "open_climate_service.automation.service.job_store.list_job_records", lambda: [_record_with(old_event)]
    )
    monkeypatch.setattr(
        "open_climate_service.automation.service._load_activations",
        lambda: {"chap-after-chirps": activation.isoformat()},
    )
    monkeypatch.setattr("open_climate_service.automation.service._save_activations", lambda _: None)

    service.replay()

    openeo.create_triggered_job.assert_not_called()


def test_replay_existing_replays_events_before_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(), True)
    trigger = WorkflowTrigger(
        id="chap-after-chirps",
        on_update_of="chirps",
        workflow_id="aggregate_to_chap_csv",
        arguments={"dataset_id": "$event.dataset_id"},
        replay_existing=True,
    )
    service = WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=[trigger]), openeo_service=openeo
    )

    activation = datetime(2026, 8, 19, tzinfo=UTC)
    old_event = _event().model_copy(update={"time": datetime(2026, 8, 18, tzinfo=UTC)})
    monkeypatch.setattr(
        "open_climate_service.automation.service.job_store.list_job_records", lambda: [_record_with(old_event)]
    )
    monkeypatch.setattr(
        "open_climate_service.automation.service._load_activations",
        lambda: {"chap-after-chirps": activation.isoformat()},
    )
    monkeypatch.setattr("open_climate_service.automation.service._save_activations", lambda _: None)

    service.replay()

    openeo.create_triggered_job.assert_called_once()


def test_start_records_activation_for_new_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        "open_climate_service.automation.service.workflows.get_workflow",
        lambda _: {"id": "aggregate_to_chap_csv"},
    )
    monkeypatch.setattr("open_climate_service.automation.service._load_activations", lambda: {})
    monkeypatch.setattr("open_climate_service.automation.service._save_activations", saved.update)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=MagicMock())

    service.start()

    assert "chap-after-chirps" in saved


def _workflow(*parameter_names: str) -> MagicMock:
    workflow = MagicMock()
    workflow.parameters = [{"name": name} for name in parameter_names]
    return workflow


def _service_for_output_check(
    monkeypatch: pytest.MonkeyPatch, workflow: MagicMock, *triggers: WorkflowTrigger
) -> WorkflowAutomationService:
    monkeypatch.setattr("open_climate_service.automation.service.workflows.get_workflow", lambda _: workflow)
    monkeypatch.setattr("open_climate_service.automation.service._load_activations", lambda: {})
    monkeypatch.setattr("open_climate_service.automation.service._save_activations", lambda _: None)
    return WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=list(triggers)), openeo_service=MagicMock()
    )


def test_duplicate_managed_output_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = _workflow("dataset_id", "output_dataset_id")
    service = _service_for_output_check(
        monkeypatch,
        workflow,
        WorkflowTrigger(id="a", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "out"}),
        WorkflowTrigger(id="b", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "out"}),
    )

    with pytest.raises(ValueError, match="same managed output"):
        service.start()


def test_distinct_managed_outputs_are_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = _workflow("dataset_id", "output_dataset_id")
    service = _service_for_output_check(
        monkeypatch,
        workflow,
        WorkflowTrigger(id="a", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "out-a"}),
        WorkflowTrigger(id="b", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "out-b"}),
    )

    service.start()  # must not raise


def test_file_producing_workflow_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = _workflow("dataset_id")  # no output_dataset_id -> file-producing
    service = _service_for_output_check(
        monkeypatch,
        workflow,
        WorkflowTrigger(id="a", on_update_of="chirps", workflow_id="wf", arguments={"dataset_id": "$event.dataset_id"}),
        WorkflowTrigger(id="b", on_update_of="chirps", workflow_id="wf", arguments={"dataset_id": "$event.dataset_id"}),
    )

    service.start()  # must not raise


def test_unresolvable_output_reference_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = _workflow("dataset_id", "output_dataset_id")
    service = _service_for_output_check(
        monkeypatch,
        workflow,
        WorkflowTrigger(
            id="a", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "$event.dataset_id"}
        ),
        WorkflowTrigger(
            id="b", on_update_of="chirps", workflow_id="wf", arguments={"output_dataset_id": "$event.dataset_id"}
        ),
    )

    service.start()  # must not raise; a runtime $event reference is left to the store lock


def test_missing_activation_skips_replay_for_default_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing activation file must not turn into a full history replay."""
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (_openeo_record(), True)
    service = WorkflowAutomationService(config_loader=_config, openeo_service=openeo)
    monkeypatch.setattr(
        "open_climate_service.automation.service.job_store.list_job_records", lambda: [_record_with(_event())]
    )
    monkeypatch.setattr("open_climate_service.automation.service._load_activations", lambda: {})
    monkeypatch.setattr("open_climate_service.automation.service._save_activations", lambda _: None)

    service.replay()

    openeo.create_triggered_job.assert_not_called()


def test_one_failing_trigger_does_not_skip_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in one trigger's submission must not skip the remaining triggers."""
    openeo = MagicMock()
    openeo.create_triggered_job.side_effect = [RuntimeError("boom"), (_openeo_record(), True)]
    triggers = [
        WorkflowTrigger(id="a", on_update_of="chirps", workflow_id="wf", arguments={"x": 1}),
        WorkflowTrigger(id="b", on_update_of="chirps", workflow_id="wf", arguments={"x": 1}),
    ]
    service = WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=triggers), openeo_service=openeo
    )

    service.consume([_event()])

    assert openeo.create_triggered_job.call_count == 2
    openeo.start_triggered_job.assert_called_once_with("triggered-job")


def test_unknown_event_reference_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo like $event.datasetid must fail at startup, not submit as a literal."""
    workflow = _workflow("dataset_id")
    service = _service_for_output_check(
        monkeypatch,
        workflow,
        WorkflowTrigger(id="t", on_update_of="chirps", workflow_id="wf", arguments={"dataset_id": "$event.datasetid"}),
    )

    with pytest.raises(ValueError, match="looks like an event reference"):
        service.start()


def test_start_validates_triggers_on_a_read_only_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-only must skip running, not skip validating configuration."""
    monkeypatch.setattr("open_climate_service.automation.service.api_config.is_read_only", lambda: True)
    monkeypatch.setattr("open_climate_service.automation.service.workflows.get_workflow", lambda _: None)
    trigger = WorkflowTrigger(id="t", on_update_of="chirps", workflow_id="missing", arguments={})
    service = WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=[trigger]), openeo_service=MagicMock()
    )

    with pytest.raises(ValueError, match="references unknown workflow"):
        service.start()
