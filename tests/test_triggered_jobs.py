"""openEO jobs created for an event: one job per event and owner, persisted owner fields, cancel races.

A pipeline run and a derived dataset refresh each create their batch job with
`create_triggered_job`, keyed on the event they answer. The fixtures here run such a job
against a stubbed process graph that saves through a named export, and are shared with the
retry tests (test_workflow_retries.py).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from open_climate_service import config
from open_climate_service.exports import service as export_service
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.schemas import OpenEOJobCreate, OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now

_OWNER = "derive:rain_anomaly"
_EXPORT = "rain-monthly"


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"geometry": ["ImspTQPwCqd"], "t": ["202608"], "rain": [12.5]})


@pytest.fixture(name="instance")
def instance_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A writable instance whose workflow jobs save a named DHIS2 export."""
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(
        config,
        "_cache",
        {"dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "TEST_TOKEN"}]},
    )
    # A pipeline owns a named export; the stored pipeline is not what these tests are about.
    definition = {
        "id": _EXPORT,
        "plugin": "dhis2",
        "connection": "hmis",
        "period_type": "monthly",
        "series": [{"data_element": "BXgDHhPdFVU"}],
    }
    monkeypatch.setattr(export_service, "_export_definitions", lambda: {_EXPORT: definition})
    # The workflow graph itself is not under test: a job only has to save a named export.
    monkeypatch.setattr(
        execution,
        "run_process_graph",
        lambda *args, **kwargs: SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": _EXPORT}),
    )
    job_service_module.reset_job_service()
    openeo = openeo_jobs.OpenEOJobService()
    try:
        yield {"openeo": openeo}
    finally:
        openeo.shutdown()
        job_service_module.reset_job_service()


def _triggered_job_id(instance: dict[str, Any], index: int = 0, *, max_attempts: int = 3) -> str:
    """Create and start the job answering event ``index``, as a pipeline run or a refresh does."""
    openeo = instance["openeo"]
    body = OpenEOJobCreate(process={"process_graph": {"result": {"process_id": "constant", "result": True}}})
    job, _ = openeo.create_triggered_job(
        body, source_event_id=f"native-job-{index}:0", trigger_id=_OWNER, max_attempts=max_attempts
    )
    openeo.start_triggered_job(job.id)
    return str(job.id)


def _await_openeo(job_id: str, timeout: float = 5.0) -> OpenEOJobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = openeo_jobs.store_get_job(job_id)
        if record is not None and record.status in {
            OpenEOJobStatus.FINISHED,
            OpenEOJobStatus.ERROR,
            OpenEOJobStatus.CANCELED,
        }:
            return record
        time.sleep(0.02)
    pytest.fail(f"openEO job {job_id} did not finish within {timeout}s")


def test_one_event_and_owner_create_one_job_that_starts_once(
    instance: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    openeo = instance["openeo"]
    enqueued: list[str] = []
    monkeypatch.setattr(openeo, "_enqueue", enqueued.append)
    body = OpenEOJobCreate(process={"process_graph": {"result": {"process_id": "constant", "result": True}}})

    first, first_created = openeo.create_triggered_job(body, source_event_id="native-job:0", trigger_id=_OWNER)
    second, second_created = openeo.create_triggered_job(body, source_event_id="native-job:0", trigger_id=_OWNER)

    assert (first.id, first_created, second_created) == (second.id, True, False)
    assert len(openeo_jobs.store_list_jobs()) == 1
    assert [openeo.start_triggered_job(first.id), openeo.start_triggered_job(first.id)] == [True, False]
    assert enqueued == [first.id]


def test_owner_fields_survive_persistence(instance: dict[str, Any]) -> None:
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="persisted",
            status=OpenEOJobStatus.FINISHED,
            created=utc_now(),
            trigger_id=_OWNER,
            source_event_id="event-1",
        )
    )
    stored = openeo_jobs.store_get_job("persisted")
    assert stored is not None
    assert (stored.trigger_id, stored.source_event_id) == (_OWNER, "event-1")
    # Internal fields stay out of openEO API responses.
    assert "trigger_id" not in stored.model_dump()


def test_cancel_arriving_while_the_result_is_saved_wins(
    instance: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    openeo = instance["openeo"]
    original = openeo._persist_result

    def cancel_during_save(job_id: str, result: Any) -> Any:
        openeo_jobs.store_update_job(job_id, lambda r: r.model_copy(update={"cancel_requested": True}))
        return original(job_id, result)

    monkeypatch.setattr(openeo, "_persist_result", cancel_during_save)

    record = _await_openeo(_triggered_job_id(instance))

    assert record.status == OpenEOJobStatus.CANCELED


def test_cancel_losing_the_race_to_completion_is_refused(
    instance: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    openeo = instance["openeo"]
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id="raced", status=OpenEOJobStatus.FINISHED, created=utc_now(), trigger_id=_OWNER)
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
