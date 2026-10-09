"""Scheduled dataset synchronization tests (CLIM-849)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service.jobs.models import JobRecord, JobStatus
from open_climate_service.scheduler.config import DatasetSyncSchedule, SchedulerConfig
from open_climate_service.scheduler.dispatcher import CheckOutcome, CheckResult, _dataset_is_materialized, enqueue_sync
from open_climate_service.scheduler.service import SchedulerService
from open_climate_service.scheduler.store import StoredSchedule


def _schedule(**updates: object) -> DatasetSyncSchedule:
    values: dict[str, object] = {
        "dataset_id": "chirps3_precipitation_daily",
        "cron": "0 6 * * *",
    }
    values.update(updates)
    return DatasetSyncSchedule.model_validate(values)


def _stored_schedule() -> StoredSchedule:
    return StoredSchedule.model_validate(_schedule().model_dump())


def _template(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "sync": {"kind": "temporal", "execution": "append"},
    }
    values.update(updates)
    return values


def _sync_jobs(scheduler: MagicMock) -> list[Any]:
    """The clock jobs registered for schedules, leaving out the store watch the clock also runs."""
    return [call for call in scheduler.add_job.call_args_list if call.kwargs["id"].startswith("dataset-sync:")]


def _job(*, process_id: str = "scheduled-sync", status: JobStatus = JobStatus.ACCEPTED) -> JobRecord:
    return JobRecord(
        job_id="job-123",
        process_id=process_id,
        status=status,
        created_at=datetime.now(timezone.utc),
        request={"dataset_id": "chirps3_precipitation_daily"},
    )


def test_scheduler_configuration_defaults_to_utc_and_three_attempts() -> None:
    config = SchedulerConfig.model_validate({"enabled": True})

    assert config.timezone_info.key == "UTC"
    assert _schedule().max_attempts == 3
    assert _schedule().schedule_id == "dataset-sync:chirps3_precipitation_daily"


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"timezone": "Not/A_Zone"}, "timezone"),
        ({"dataset_sync": [{"dataset_id": "x", "cron": "0 6 * * *"}]}, "Extra inputs"),
    ],
)
def test_scheduler_configuration_rejects_invalid_values(payload: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SchedulerConfig.model_validate(payload)


def test_schedule_definition_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="cron"):
        DatasetSyncSchedule(dataset_id="x", cron="daily")
    with pytest.raises(ValueError, match="greater than"):
        DatasetSyncSchedule(dataset_id="x", cron="0 6 * * *", max_attempts=0)


def test_due_schedule_enqueues_retryable_native_job(monkeypatch: pytest.MonkeyPatch) -> None:
    submit = MagicMock(return_value=_job())
    job_service = MagicMock()
    job_service.list_jobs.return_value.jobs = []
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.api_config.is_read_only", lambda: False)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.get_job_service", lambda: job_service)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher._dataset_is_materialized", lambda _: True)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.submit_sync_job", submit)

    result = enqueue_sync(_schedule(max_attempts=4))

    assert result.outcome == CheckOutcome.SUBMITTED
    assert result.job_id == "job-123"
    submit.assert_called_once_with(
        dataset_id="chirps3_precipitation_daily",
        end=None,
        publish=True,
        label="scheduled-sync",
        max_attempts=4,
    )


def test_unmaterialized_dataset_skips_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered-but-never-ingested dataset must not burn worker retries (CLIM-849)."""
    submit = MagicMock()
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.api_config.is_read_only", lambda: False)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher._has_active_writer_job", lambda _: False)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher._dataset_is_materialized", lambda _: False)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.submit_sync_job", submit)

    result = enqueue_sync(_schedule())

    assert result.outcome == CheckOutcome.NOT_MATERIALIZED
    assert result.job_id is None
    submit.assert_not_called()


def test_materialization_check_translates_missing_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_artifact(_: str) -> None:
        raise HTTPException(status_code=404, detail="No artifact found")

    monkeypatch.setattr(
        "open_climate_service.ingestions.services.get_latest_artifact_for_dataset_or_404",
        missing_artifact,
    )

    assert _dataset_is_materialized("registered-but-never-ingested") is False


def test_materialization_check_propagates_unexpected_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable_artifact_index(_: str) -> None:
        raise HTTPException(status_code=503, detail="Artifact index unavailable")

    monkeypatch.setattr(
        "open_climate_service.ingestions.services.get_latest_artifact_for_dataset_or_404",
        unavailable_artifact_index,
    )

    with pytest.raises(HTTPException) as exc_info:
        _dataset_is_materialized("chirps3_precipitation_daily")

    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("process_id", ["ingestion", "sync", "scheduled-sync"])
def test_active_writer_job_prevents_duplicate_submission(process_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    submit = MagicMock()
    job_service = MagicMock()
    job_service.list_jobs.return_value.jobs = [_job(process_id=process_id, status=JobStatus.RUNNING)]
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.api_config.is_read_only", lambda: False)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.get_job_service", lambda: job_service)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.submit_sync_job", submit)

    result = enqueue_sync(_schedule())

    assert result.outcome == CheckOutcome.ALREADY_RUNNING
    submit.assert_not_called()


def test_read_only_check_does_not_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    submit = MagicMock()
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.api_config.is_read_only", lambda: True)
    monkeypatch.setattr("open_climate_service.scheduler.dispatcher.submit_sync_job", submit)

    result = enqueue_sync(_schedule())

    assert result.outcome == CheckOutcome.READ_ONLY
    submit.assert_not_called()


def test_service_registers_coalesced_non_overlapping_cron_job(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = MagicMock()
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: _template(),
    )

    service.start()

    (registered,) = _sync_jobs(scheduler)
    assert registered.kwargs["coalesce"] is True
    assert registered.kwargs["max_instances"] == 1
    scheduler.start.assert_called_once_with()
    service.shutdown()
    scheduler.shutdown.assert_called_once_with(wait=False)


def test_start_loads_clock_config_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = MagicMock()
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    load_config = MagicMock(return_value=SchedulerConfig(enabled=True))
    service = SchedulerService(
        config_loader=load_config,
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: _template(),
    )
    service.start()
    assert load_config.call_count == 1
    service.shutdown()


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_real_scheduler_computes_timezone_aware_next_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(
            enabled=True,
            timezone="Europe/Oslo",
        ),
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: _template(),
    )

    service.start()
    try:
        status = service.status().schedules[0]
        assert status.next_check is not None
        assert status.next_check.tzinfo is not None
        assert getattr(status.next_check.tzinfo, "key", None) == "Europe/Oslo"
    finally:
        service.shutdown()


def test_service_skips_future_facing_schedule_without_failing_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = MagicMock()
    scheduler.get_jobs.return_value = []
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: _template(temporal_direction="future"),
    )

    service.start()

    scheduler.start.assert_called_once_with()
    assert _sync_jobs(scheduler) == []
    assert service.status().schedules[0].last_outcome == CheckOutcome.ERROR
    assert "future-facing" in (service.status().schedules[0].last_message or "")


def test_service_skips_static_schedule_without_failing_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = MagicMock()
    scheduler.get_jobs.return_value = []
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: _template(sync={"kind": "static"}),
    )

    service.start()

    scheduler.start.assert_called_once_with()
    assert _sync_jobs(scheduler) == []
    assert service.status().schedules[0].last_outcome == CheckOutcome.ERROR
    assert "not syncable" in (service.status().schedules[0].last_message or "")


def test_service_skips_missing_template_without_failing_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = MagicMock()
    warning = MagicMock()
    scheduler.get_jobs.return_value = []
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    monkeypatch.setattr("open_climate_service.scheduler.service.logger.warning", warning)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        template_loader=lambda _: None,
    )

    service.start()

    scheduler.start.assert_called_once_with()
    assert _sync_jobs(scheduler) == []
    assert service.status().schedules[0].last_outcome == CheckOutcome.ERROR
    assert "no registered data source" in (service.status().schedules[0].last_message or "")
    assert warning.call_args.args[-1] == 0


def test_service_does_not_start_on_read_only_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler_factory = MagicMock()
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", scheduler_factory)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: True)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True), store_loader=lambda: [_stored_schedule()]
    )

    service.start()

    scheduler_factory.assert_not_called()


def test_status_exposes_next_and_last_check(monkeypatch: pytest.MonkeyPatch) -> None:
    next_check = datetime(2026, 8, 19, 6, tzinfo=timezone.utc)
    scheduler = MagicMock()
    scheduler.get_jobs.return_value = [MagicMock(id=_schedule().schedule_id, next_run_time=next_check)]
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    result = CheckResult(
        schedule_id=_schedule().schedule_id,
        dataset_id=_schedule().dataset_id,
        outcome=CheckOutcome.SUBMITTED,
        message="submitted",
        job_id="job-123",
    )
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        dispatcher=lambda _: result,
        template_loader=lambda _: _template(),
    )
    service.start()
    service.check_now(_schedule())

    status = service.status()

    assert status.running is True
    assert status.schedules[0].next_check == next_check
    assert status.schedules[0].last_outcome == CheckOutcome.SUBMITTED
    assert status.schedules[0].last_job_id == "job-123"


def test_check_error_is_retained_without_escaping() -> None:
    def fail(_: DatasetSyncSchedule) -> CheckResult:
        raise RuntimeError("source unavailable")

    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored_schedule()],
        dispatcher=fail,
        template_loader=lambda _: _template(),
    )

    result = service.check_now(_schedule())

    assert result.outcome == CheckOutcome.ERROR
    assert "source unavailable" in result.message


def test_schedules_endpoint_is_read_only_status(client: TestClient) -> None:
    response = client.get("/schedules")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "running": False,
        "timezone": "UTC",
        "reload_error": None,
        "schedules": [],
    }
