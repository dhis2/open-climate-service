"""CLIM-1217: a successful initial ingestion emits `dataset.updated` for workflow automation."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service import config
from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger
from open_climate_service.automation.service import WorkflowAutomationService
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents import services as extent_services
from open_climate_service.ingestions import processes, services
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobRecord, JobStatus
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.shared.time import utc_now
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_DATASET_ID = "rolling_precip"
_TRIGGER = "chap-after-ingest"


class _Plugin(BaseDatasetPlugin):
    def __init__(self) -> None:
        self.available = ["2026-01-01", "2026-01-02"]
        self.fail = False
        self.on_fetch: Any = None

    async def periods(self, start: str, end: str) -> list[str]:
        return [period for period in self.available if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        if self.on_fetch is not None:
            self.on_fetch()
        if self.fail:
            raise RuntimeError("source unavailable")
        data = xr.DataArray(
            np.full((2, 2), int(period_id[-2:]), dtype="float32"),
            dims=("y", "x"),
            coords={"y": [3.5, 2.5], "x": [1.5, 2.5]},
        )
        return normalize_period(data, variable="precip", period=period_id)


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Real ingestion into a local store, a real native job service, and automation."""
    plugin = _Plugin()
    dataset: dict[str, object] = {
        "id": _DATASET_ID,
        "name": "Rolling precipitation",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {"plugin": "example.RollingPlugin"},
    }
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: tmp_path / "rolling.icechunk")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    monkeypatch.setattr(
        registry_datasets, "get_dataset", lambda dataset_id: dataset if dataset_id == _DATASET_ID else None
    )
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")

    job_service_module.reset_job_service()
    openeo = openeo_jobs.OpenEOJobService()
    # Workflow submission is what is under test; executing the workflow is not.
    monkeypatch.setattr(openeo, "_enqueue", lambda job_id: None)
    automation = WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(
            workflow_triggers=[
                WorkflowTrigger(
                    id=_TRIGGER,
                    on_update_of=_DATASET_ID,
                    workflow_id="aggregate_to_chap_csv",
                    arguments={
                        "dataset_id": "$event.dataset_id",
                        "temporal_extent": ["$event.previous_end", "$event.current_end"],
                        "period_type": "day",
                    },
                )
            ]
        ),
        openeo_service=openeo,
    )
    automation.start()
    jobs = job_service_module.get_job_service()
    jobs.set_event_consumer(automation.consume)
    try:
        yield {"plugin": plugin, "dataset": dataset, "jobs": jobs, "automation": automation}
    finally:
        jobs.set_event_consumer(None)
        openeo.shutdown()
        job_service_module.reset_job_service()


def _ingest(instance: dict[str, Any], **request: Any) -> JobRecord:
    job = instance["jobs"].submit_callable_job(
        func=processes.execute_ingestion,
        label="ingestion",
        request={"dataset_id": _DATASET_ID, "start": "2026-01-01", "end": "2026-01-02", "publish": False, **request},
    )
    return _await_terminal(instance, job.job_id)


def _await_terminal(instance: dict[str, Any], job_id: str, timeout: float = 20.0) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = instance["jobs"].get_job_or_404(job_id)
        if record.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    pytest.fail(f"Ingestion job {job_id} did not finish within {timeout}s")


def _triggered_jobs(expected: int = 1, timeout: float = 10.0) -> list[openeo_jobs.OpenEOJobRecord]:
    """The workflow jobs the trigger submitted, once there are `expected` of them.

    The job service persists a job as successful first and hands its events to automation after,
    so a test that saw the ingestion finish can look before the workflow job exists. Waiting for
    the expected count, and returning whatever is there at the deadline, keeps the assertion on
    the count rather than on timing.
    """
    # A triggered job's description names its source event and its automation rule.
    rule = f"automation rule {_TRIGGER}"
    deadline = time.monotonic() + timeout
    while True:
        jobs = [record for record in openeo_jobs.store_list_jobs() if rule in (record.description or "")]
        if len(jobs) >= expected or time.monotonic() > deadline:
            return jobs
        time.sleep(0.05)


def _submitted_extent(record: openeo_jobs.OpenEOJobRecord) -> Any:
    return record.process["process_graph"]["workflow"]["arguments"]["temporal_extent"]


def test_initial_ingestion_emits_one_event_and_triggers_the_workflow_once(instance: dict[str, Any]) -> None:
    record = _ingest(instance)

    assert record.status == JobStatus.SUCCESSFUL, record.error
    [event] = record.events
    assert event.type == DATASET_UPDATED_EVENT_TYPE
    assert event.source == f"/datasets/{_DATASET_ID}"
    assert isinstance(record.result, dict)
    assert event.data == {
        "dataset_id": _DATASET_ID,
        "artifact_id": record.result["ingestion_id"],
        "action": "ingest",
        "previous_end": None,  # nothing was stored before: every period is new
        "current_start": "2026-01-01",
        "current_end": "2026-01-02",
    }
    [triggered] = _triggered_jobs()
    assert f"Triggered by {event.event_id} " in (triggered.description or "")
    assert _submitted_extent(triggered) == [None, "2026-01-02"]


def test_reingesting_a_current_dataset_emits_nothing(instance: dict[str, Any]) -> None:
    _ingest(instance)
    again = _ingest(instance)

    assert again.status == JobStatus.SUCCESSFUL, again.error
    assert again.events == []
    assert len(_triggered_jobs()) == 1


def test_extending_ingestion_reports_what_was_stored_before(instance: dict[str, Any]) -> None:
    _ingest(instance)
    instance["plugin"].available.append("2026-01-03")
    extended = _ingest(instance, end="2026-01-03")

    [event] = extended.events
    assert (event.data["previous_end"], event.data["current_end"]) == ("2026-01-02", "2026-01-03")
    assert len(_triggered_jobs(expected=2)) == 2


def test_failed_ingestion_emits_nothing(instance: dict[str, Any]) -> None:
    instance["plugin"].fail = True
    record = _ingest(instance)

    assert record.status == JobStatus.FAILED
    assert record.events == []
    assert _triggered_jobs(expected=0) == []


def test_cancelled_ingestion_emits_nothing(instance: dict[str, Any]) -> None:
    submitted = threading.Event()
    holder: dict[str, str] = {}

    def cancel_on_first_fetch() -> None:
        submitted.wait(5)
        instance["jobs"].request_cancellation(holder["job_id"])
        instance["plugin"].on_fetch = None

    instance["plugin"].on_fetch = cancel_on_first_fetch
    job = instance["jobs"].submit_callable_job(
        func=processes.execute_ingestion,
        label="ingestion",
        request={"dataset_id": _DATASET_ID, "start": "2026-01-01", "end": "2026-01-02", "publish": False},
    )
    holder["job_id"] = job.job_id
    submitted.set()
    record = _await_terminal(instance, job.job_id)

    assert record.status == JobStatus.CANCELLED
    assert record.events == []
    assert _triggered_jobs(expected=0) == []


@pytest.mark.parametrize("marker", [True, False])
def test_recovered_ingestion_emits_the_event_its_committed_attempt_owed(instance: dict[str, Any], marker: bool) -> None:
    # An attempt that committed its data and died before completing: the data is stored,
    # and the job is left RUNNING with the marker it wrote before fetching.
    services.create_artifact(
        dataset=instance["dataset"],
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    native_store.create_job_record(
        JobRecord(
            job_id="interrupted",
            process_id="ingestion",
            status=JobStatus.RUNNING,
            created_at=utc_now(),
            attempt=1,
            request={
                "__fn_path__": "open_climate_service.ingestions.processes.execute_ingestion",
                "dataset_id": _DATASET_ID,
                "start": "2026-01-01",
                "end": "2026-01-02",
                "publish": False,
            },
            cursor={"last_committed": "2026-01-02", "dataset_update_planned": {"previous_end": None}}
            if marker
            else {"last_committed": "2026-01-02"},
        )
    )

    instance["jobs"].recover_pending_jobs()
    record = _await_terminal(instance, "interrupted")

    assert record.status == JobStatus.SUCCESSFUL, record.error
    if not marker:
        # Without the marker the re-run finds current data and owes nothing: the case
        # the marker exists for.
        assert record.events == []
        return
    [event] = record.events
    assert event.event_id == "interrupted:0"
    assert event.data["previous_end"] is None
    instance["automation"].replay()
    instance["automation"].replay()
    assert len(_triggered_jobs()) == 1


def test_streaming_checkpoints_keep_the_marker(instance: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, Any]] = []
    original = instance["jobs"].save_cursor

    def record_cursor(job_id: str, cursor: dict[str, Any]) -> Any:
        saved.append(dict(cursor))
        return original(job_id, cursor)

    monkeypatch.setattr(instance["jobs"], "save_cursor", record_cursor)
    _ingest(instance)

    assert saved, "the ingestion saved no checkpoints"
    assert all(cursor.get("dataset_update_planned") == {"previous_end": None} for cursor in saved)
    assert any("last_committed" in cursor for cursor in saved)


def test_inline_ingestion_is_recorded_as_a_completed_job_with_its_event(
    instance: dict[str, Any], client: TestClient
) -> None:
    response = client.post(
        "/ingestions", json={"dataset_id": _DATASET_ID, "start": "2026-01-01", "end": "2026-01-02", "publish": False}
    )

    assert response.status_code == 200, response.text
    [record] = [job for job in native_store.list_job_records() if job.process_id == "ingestion"]
    assert (record.status, record.executor_kind) == (JobStatus.SUCCESSFUL, "inline")
    [event] = record.events
    assert event.data["artifact_id"] == response.json()["ingestion_id"]
    assert len(_triggered_jobs()) == 1
    # A recorded job is terminal: recovery never re-runs it.
    instance["jobs"].recover_pending_jobs()
    assert native_store.get_job_record(record.job_id).status == JobStatus.SUCCESSFUL  # type: ignore[union-attr]


def test_inline_sync_is_recorded_as_a_completed_job_with_its_event(
    instance: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Before this change only queued and scheduled syncs reached automation.
    instance["dataset"]["sync"] = {"kind": "temporal"}
    monkeypatch.setattr(
        "open_climate_service.shared.plugin_loader.instantiate_plugin", lambda *args, **kwargs: instance["plugin"]
    )
    _ingest(instance)
    instance["plugin"].available.append("2026-01-03")

    response = client.post(f"/sync/{_DATASET_ID}", json={"end": "2026-01-03", "publish": False})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    [record] = [job for job in native_store.list_job_records() if job.process_id == "sync"]
    assert record.executor_kind == "inline"
    [event] = record.events
    # The action is the sync planner's own choice; the event reports it as planned.
    assert event.data["action"] in {"append", "rematerialize"}
    expected_previous_end = "2026-01-02" if event.data["action"] == "append" else None
    assert (event.data["previous_end"], event.data["current_end"]) == (expected_previous_end, "2026-01-03")
    assert len(_triggered_jobs(expected=2)) == 2
