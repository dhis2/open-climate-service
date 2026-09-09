"""Tests for async ingestion and sync via Prefer: respond-async."""

from collections.abc import Callable
from typing import NoReturn
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service.ingestions import processes
from open_climate_service.ingestions.schemas import SyncAction, SyncDetail, SyncKind, SyncResponse
from open_climate_service.jobs.models import (
    JobEventDraft,
    JobExecutionResult,
    JobProgress,
    JobRecord,
    JobStatus,
)
from open_climate_service.jobs.service import JobService


def _make_job(job_id: str = "job-123") -> JobRecord:
    from open_climate_service.shared.time import utc_now

    return JobRecord(
        job_id=job_id,
        process_id="ingestion",
        status=JobStatus.ACCEPTED,
        created_at=utc_now(),
        max_attempts=1,
        executor_kind="thread",
        request={},
        links=[],
    )


def _fake_job_callable() -> dict[str, object]:
    return {"ok": True}


def _fake_event_job_callable() -> JobExecutionResult:
    return JobExecutionResult(
        result={"ok": True},
        events=[JobEventDraft(type="dataset.updated", source="/datasets/example", data={"action": "append"})],
    )


def _sync_response(action: SyncAction, *, status: str = "completed") -> SyncResponse:
    return SyncResponse(
        sync_id="artifact-2" if status == "completed" else None,
        status=status,
        message="test",
        dataset=None,
        sync_detail=SyncDetail(
            source_dataset_id="source",
            sync_kind=SyncKind.TEMPORAL,
            action=action,
            reason="test",
            message="test",
            current_start="2026-01-01",
            current_end="2026-01-02",
            target_end="2026-01-03",
            target_end_source="request",
        ),
    )


def test_post_ingestion_respond_async_returns_202_with_location(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job("abc-123")
    mock_service = MagicMock()
    mock_service.submit_callable_job.return_value = job
    monkeypatch.setattr("open_climate_service.jobs.service.JobService.submit_callable_job", lambda self, **kw: job)
    monkeypatch.setattr(
        "open_climate_service.ingestions.routes._get_dataset_or_404",
        lambda _: {"id": "chirps3_precipitation_daily"},
    )
    monkeypatch.setattr("open_climate_service.ingestions.routes.get_extent_or_404", lambda: {"bbox": [0, 0, 1, 1]})

    response = client.post(
        "/ingestions",
        headers={"Prefer": "respond-async"},
        json={"dataset_id": "chirps3_precipitation_daily", "start": "2024-01-01"},
    )

    assert response.status_code == 202
    assert response.headers["Location"] == "/ingestions/jobs/abc-123"
    assert response.json()["ingestion_id"] == "abc-123"


def test_post_sync_respond_async_returns_202_with_location(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _make_job("sync-456")
    monkeypatch.setattr("open_climate_service.jobs.service.JobService.submit_callable_job", lambda self, **kw: job)
    monkeypatch.setattr(
        "open_climate_service.ingestions.routes.services.plan_sync_dataset",
        lambda dataset_id, end: MagicMock(),
    )

    response = client.post(
        "/sync/chirps3_precipitation_daily",
        headers={"Prefer": "respond-async"},
        json={},
    )

    assert response.status_code == 202
    assert response.headers["Location"] == "/ingestions/jobs/sync-456"


def test_get_ingestion_job_returns_job_with_correct_self_link(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job("job-789")
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: job)

    response = client.get("/ingestions/jobs/job-789")

    assert response.status_code == 200
    payload = response.json()
    assert payload["jobID"] == "job-789"
    links = {lnk["rel"]: lnk["href"] for lnk in payload["links"]}
    assert links["self"] == "/ingestions/jobs/job-789"


def test_cancel_ingestion_job_requests_cancellation_and_returns_native_self_link(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job("job-cancel")
    cancelled = job.model_copy(
        update={
            "cancel_requested": True,
            "progress": JobProgress(message="Cancellation requested"),
        }
    )
    monkeypatch.setattr(
        "open_climate_service.jobs.service.JobService.request_cancellation",
        lambda self, job_id: cancelled,
    )

    response = client.delete("/ingestions/jobs/job-cancel")

    assert response.status_code == 202
    payload = response.json()
    assert payload["jobID"] == "job-cancel"
    assert payload["status"] == "accepted"
    assert payload["cancelRequested"] is True
    links = {lnk["rel"]: lnk["href"] for lnk in payload["links"]}
    assert links["self"] == "/ingestions/jobs/job-cancel"


def test_get_ingestion_job_returns_404_for_unknown(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: None)

    response = client.get("/ingestions/jobs/does-not-exist")

    assert response.status_code == 404


def test_post_ingestion_without_prefer_does_not_return_202(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Prefer: respond-async the response is never 202 Accepted."""
    monkeypatch.setattr(
        "open_climate_service.ingestions.routes._get_dataset_or_404",
        lambda _: {"id": "chirps3_precipitation_daily"},
    )

    def raise_no_extent() -> NoReturn:
        raise HTTPException(status_code=422, detail="no extent")

    monkeypatch.setattr("open_climate_service.ingestions.routes.get_extent_or_404", raise_no_extent)

    response = client.post(
        "/ingestions",
        json={"dataset_id": "chirps3_precipitation_daily", "start": "2024-01-01"},
    )

    assert response.status_code == 422
    assert "Location" not in response.headers


def test_post_ingestion_respond_async_validates_dataset_and_extent_before_queueing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_submit_callable_job(
        *,
        func: object,
        label: str,
        request: dict[str, object],
        max_attempts: int = 1,
        **_ignored: object,
    ) -> JobRecord:
        del func, label, request, max_attempts, _ignored
        calls.append("queued")
        return _make_job("should-not-queue")

    monkeypatch.setattr(
        "open_climate_service.jobs.service.JobService.submit_callable_job",
        lambda self, **kw: fake_submit_callable_job(**kw),
    )

    def raise_missing_dataset(_: str) -> NoReturn:
        raise HTTPException(status_code=404, detail="Dataset not found")

    monkeypatch.setattr("open_climate_service.ingestions.routes._get_dataset_or_404", raise_missing_dataset)

    response = client.post(
        "/ingestions",
        headers={"Prefer": "respond-async"},
        json={"dataset_id": "missing-dataset", "start": "2024-01-01"},
    )

    assert response.status_code == 404
    assert calls == []
    assert "Location" not in response.headers


def test_post_sync_respond_async_validates_plan_before_queueing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_submit_callable_job(
        *,
        func: object,
        label: str,
        request: dict[str, object],
        max_attempts: int = 1,
        **_ignored: object,
    ) -> JobRecord:
        del func, label, request, max_attempts, _ignored
        calls.append("queued")
        return _make_job("should-not-queue")

    monkeypatch.setattr(
        "open_climate_service.jobs.service.JobService.submit_callable_job",
        lambda self, **kw: fake_submit_callable_job(**kw),
    )

    def raise_invalid_sync_plan(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise HTTPException(status_code=400, detail="invalid sync plan")

    monkeypatch.setattr("open_climate_service.ingestions.routes.services.plan_sync_dataset", raise_invalid_sync_plan)

    response = client.post(
        "/sync/chirps3_precipitation_daily",
        headers={"Prefer": "respond-async"},
        json={},
    )

    assert response.status_code == 400
    assert calls == []
    assert "Location" not in response.headers


def test_submit_callable_job_uses_default_jobs_base(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: dict[str, JobRecord] = {}

    monkeypatch.setattr("open_climate_service.jobs.service.JobService._enqueue_job", lambda self, job_id: None)
    monkeypatch.setattr(
        "open_climate_service.jobs.store.create_job_record",
        lambda record: persisted.setdefault(record.job_id, record),
    )
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: persisted.get(job_id))

    record = JobService().submit_callable_job(
        func=_fake_job_callable,
        label="ingestion",
        request={},
    )

    assert [link.href for link in record.links] == [f"/jobs/{record.job_id}"]


def test_submit_callable_job_normalizes_custom_jobs_base(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: dict[str, JobRecord] = {}

    monkeypatch.setattr("open_climate_service.jobs.service.JobService._enqueue_job", lambda self, job_id: None)
    monkeypatch.setattr(
        "open_climate_service.jobs.store.create_job_record",
        lambda record: persisted.setdefault(record.job_id, record),
    )
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: persisted.get(job_id))

    record = JobService().submit_callable_job(
        func=_fake_job_callable,
        label="sync",
        request={},
        job_href_base="/ingestions/jobs/",
    )

    assert [link.href for link in record.links] == [f"/ingestions/jobs/{record.job_id}"]


def test_successful_job_persists_result_and_events_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: dict[str, JobRecord] = {}

    monkeypatch.setattr("open_climate_service.jobs.service.JobService._enqueue_job", lambda self, job_id: None)
    monkeypatch.setattr(
        "open_climate_service.jobs.store.create_job_record",
        lambda record: persisted.setdefault(record.job_id, record),
    )
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: persisted.get(job_id))

    def mutate(job_id: str, mutation: Callable[[JobRecord], JobRecord]) -> JobRecord:
        updated = mutation(persisted[job_id])
        persisted[job_id] = updated
        return updated

    monkeypatch.setattr("open_climate_service.jobs.store.mutate_job_record", mutate)
    service = JobService()
    consume = MagicMock()
    service.set_event_consumer(consume)
    submitted = service.submit_callable_job(func=_fake_event_job_callable, label="sync", request={})

    service._execute_job(submitted.job_id)

    completed = persisted[submitted.job_id]
    assert completed.status == JobStatus.SUCCESSFUL
    assert completed.result == {"ok": True}
    assert len(completed.events) == 1
    assert completed.events[0].event_id == f"{submitted.job_id}:0"
    assert completed.events[0].type == "dataset.updated"
    assert completed.events[0].time == completed.finished_at
    consume.assert_called_once_with(completed.events)


@pytest.mark.parametrize("action", [SyncAction.APPEND, SyncAction.REMATERIALIZE])
def test_execute_sync_returns_dataset_updated_after_completed_change(
    action: SyncAction, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(processes.services, "sync_dataset", lambda **_: _sync_response(action))

    outcome = processes.execute_sync(dataset_id="managed-dataset")

    assert isinstance(outcome.result, SyncResponse)
    assert len(outcome.events) == 1
    assert outcome.events[0].type == "dataset.updated"
    assert outcome.events[0].source == "/datasets/managed-dataset"
    assert outcome.events[0].data == {
        "dataset_id": "managed-dataset",
        "artifact_id": "artifact-2",
        "action": action.value,
        "previous_end": "2026-01-02",
        "current_end": "2026-01-03",
    }


def test_execute_sync_does_not_emit_dataset_updated_for_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        processes.services,
        "sync_dataset",
        lambda **_: _sync_response(SyncAction.NO_OP, status="up_to_date"),
    )

    outcome = processes.execute_sync(dataset_id="managed-dataset")

    assert outcome.events == []
