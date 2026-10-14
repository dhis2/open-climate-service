"""An openEO job that publishes a managed dataset owns that publication and its update event.

Real Icechunk stores and the real job service; the graph result and external services are
stubbed. A process crash is simulated with a BaseException, which the job service does not
handle, so the job record is left RUNNING exactly as a killed process leaves it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi import HTTPException

from open_climate_service.data_manager.services import downloader
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import PublicationStatus
from open_climate_service.jobs import store as native_store
from open_climate_service.jobs.events import persisted_events, read_event_log
from open_climate_service.jobs.models import JobEvent
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now
from tests.test_managed_publish_units import managed_instance  # noqa: F401  # pyright: ignore[reportUnusedImport]

_DATASET = "derived_precip"
_EVIDENCE: dict[str, Any] = {
    "scope": "executed_processes",
    "process_sha256": "f" * 64,
    "sources": [
        {
            "collection_id": "chirps3_precipitation_daily_sle",
            "artifact_id": "input-artifact",
            "source_dataset_id": "chirps3_precipitation_daily",
            "snapshot_id": "INPUTSNAPSHOT0000000",
        }
    ],
    "features": [],
    "spatial_aggregations": [],
    "temporal_aggregations": [],
    "missing": ["per_output_lineage"],
}


class _Crash(BaseException):
    """A process dying at this point: not handled, so the job stays RUNNING."""


def _cube(value: float, *, start: str = "2026-01-01", periods: int = 3, temporal: bool = True) -> xr.Dataset:
    if not temporal:
        flat = xr.DataArray(
            np.full((2, 2), value, dtype="float32"),
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
            name="precip",
            attrs={"units": "mm/d"},
        )
        return flat.to_dataset(name="precip").rio.write_crs("EPSG:4326")
    times = pd.date_range(start, periods=periods, freq="D")
    data = xr.DataArray(
        np.full((len(times), 2, 2), value, dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="precip",
        attrs={"units": "mm/d"},
    )
    return data.to_dataset(name="precip").rio.write_crs("EPSG:4326")


@pytest.fixture
def service(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    request.getfixturevalue("managed_instance")  # a temporary template registry and store directory
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "jobs" / "jobs.json")
    openeo = openeo_jobs.OpenEOJobService()
    try:
        yield openeo
    finally:
        openeo.shutdown()


@pytest.fixture
def received(service: Any) -> list[JobEvent]:
    """Events handed to the live consumer."""
    events: list[JobEvent] = []
    service.set_event_consumer(events.extend)
    return events


def _job(
    job_id: str,
    value: float,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publish: bool = True,
    graph_runs: list[str] | None = None,
    max_attempts: int = 1,
    start: str = "2026-01-01",
    periods: int = 3,
    temporal: bool = True,
) -> None:
    """A queued, user-submitted job whose graph publishes `value` as a managed dataset."""

    def run(*args: Any, **kwargs: Any) -> SaveResultEnvelope:
        if graph_runs is not None:
            graph_runs.append(job_id)
        envelope = SaveResultEnvelope(
            _cube(value, start=start, periods=periods, temporal=temporal),
            "ZARR",
            {"dataset_id": _DATASET, "variable": "precip", "publish": publish},
        )
        envelope.provenance = dict(_EVIDENCE)
        return envelope

    monkeypatch.setattr(execution, "run_process_graph", run)
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id=job_id,
            status=OpenEOJobStatus.QUEUED,
            created=utc_now(),
            process={"process_graph": {}},
            max_attempts=max_attempts,
        )
    )


def _store() -> Path:
    return downloader.DOWNLOAD_DIR / f"{_DATASET}.icechunk"


def _head() -> str:
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(_store())))
    return str(repo.lookup_branch("main"))


def _commits(job_id: str) -> list[str]:
    """Snapshots this job committed to the store, oldest first."""
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(_store())))
    mine = [s.id for s in repo.ancestry(branch="main") if s.message.startswith(f"Published from openEO job {job_id} ")]
    return list(reversed(mine))


def _artifacts() -> list[Any]:
    return [record for record in ingestion_services._load_records() if record.dataset_id == _DATASET]


def _record(job_id: str) -> OpenEOJobRecord:
    record = openeo_jobs.store_get_job(job_id)
    assert record is not None
    return record


def _wrap_writer(
    monkeypatch: pytest.MonkeyPatch, *, before_gate: bool = False, after_gate: bool = False, after_commit: bool = False
) -> None:
    """Crash once at one point of the managed store write; the restarted process does not."""
    original = downloader.write_to_icechunk_store
    armed = [True]

    def write(*args: Any, **kwargs: Any) -> Any:
        crash, armed[0] = armed[0], False
        gate: Callable[[], None] = kwargs["before_commit"]

        def crash_at_gate() -> None:
            if crash and before_gate:
                raise _Crash("before the gate")
            gate()
            if crash and after_gate:
                raise _Crash("after the gate, before the commit")

        snapshot = original(*args, **{**kwargs, "before_commit": crash_at_gate})
        if crash and after_commit:
            raise _Crash("immediately after the commit")
        return snapshot

    monkeypatch.setattr(downloader, "write_to_icechunk_store", write)


def _crash_once_after(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str, where: str) -> None:
    """Crash once right after ``owner.name`` returns; the restarted process does not."""
    original = getattr(owner, name)
    armed = [True]

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if armed[0]:
            armed[0] = False
            raise _Crash(where)
        return result

    monkeypatch.setattr(owner, name, wrapped)


def _assert_one_published_event(job_id: str) -> JobEvent:
    record = _record(job_id)
    [artifact] = _artifacts()
    assert record.status == OpenEOJobStatus.FINISHED
    assert record.publication is not None and record.publication.state == "completed"
    snapshot = record.publication.snapshot_id
    assert [event.event_id for event in record.events] == [f"{job_id}:publication:{artifact.artifact_id}:{snapshot}"]
    return record.events[0]


# --- the successful publication --------------------------------------------------------------


def test_a_published_managed_dataset_emits_one_full_rewrite_update(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")

    record = _record("job-1")
    [artifact] = _artifacts()
    assert artifact.publication.status == PublicationStatus.PUBLISHED
    assert record.status == OpenEOJobStatus.FINISHED and record.publishing is False
    publication = record.publication
    assert publication is not None
    assert (publication.state, publication.published, publication.artifact_id) == (
        "completed",
        True,
        artifact.artifact_id,
    )
    assert publication.snapshot_id == _head()  # the snapshot the job committed
    [event] = record.events
    # One committed publication: this job, this artifact, this snapshot.
    assert (
        event.event_id == publication.event_id == f"job-1:publication:{artifact.artifact_id}:{publication.snapshot_id}"
    )
    assert event.type == "dataset.updated" and event.source == f"/datasets/{_DATASET}"
    assert event.data == {
        "dataset_id": _DATASET,
        "artifact_id": artifact.artifact_id,
        "action": "full_rewrite",
        "previous_end": None,  # a full rewrite: no earlier period is known unchanged
        "current_start": artifact.coverage.temporal.start,
        "current_end": artifact.coverage.temporal.end,
        "producer_type": "openeo",
        "producer_job_id": "job-1",
        "output_snapshot_id": publication.snapshot_id,
    }
    assert (event.data["current_start"], event.data["current_end"]) == ("2026-01-01", "2026-01-03")
    # The live consumer got exactly that event, and the shared log holds it.
    assert received == [event]
    assert persisted_events() == [event]


def test_the_public_job_api_does_not_expose_publication_internals(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")
    public = _record("job-1").model_dump(mode="json")
    assert "publication" not in public and "events" not in public and "publishing" not in public


def test_an_unpublished_output_is_registered_without_an_update(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch, publish=False)
    service._execute("job-1")
    record = _record("job-1")
    [artifact] = _artifacts()
    assert artifact.publication.status == PublicationStatus.UNPUBLISHED
    assert record.status == OpenEOJobStatus.FINISHED
    assert record.publication is not None and record.publication.state == "completed"
    assert record.publication.published is False and record.publication.event_id is None
    assert record.events == [] and received == [] and persisted_events() == []


def test_an_artifact_already_published_stays_published_and_is_announced(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("first", 5.0, monkeypatch, publish=True)
    service._execute("first")
    # Re-published by a job that does not ask to publish: the upsert keeps the artifact
    # published, so the update is announced from the stored state, not the request.
    _job("second", 7.0, monkeypatch, publish=False)
    service._execute("second")
    [artifact] = _artifacts()
    assert artifact.publication.status == PublicationStatus.PUBLISHED
    second = _record("second")
    assert second.publication is not None and second.publication.published is True
    first = _record("first").publication
    assert first is not None and second.publication is not None
    assert [event.event_id for event in second.events] == [
        f"second:publication:{artifact.artifact_id}:{second.publication.snapshot_id}"
    ]
    assert [event.event_id for event in received] == [
        f"first:publication:{artifact.artifact_id}:{first.snapshot_id}",
        f"second:publication:{artifact.artifact_id}:{second.publication.snapshot_id}",
    ]


def test_a_full_store_rewrite_with_changed_coverage_keeps_the_published_artifact(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("first", 5.0, monkeypatch, publish=True, start="2026-01-01")
    service._execute("first")
    [before] = _artifacts()

    # A wider range: the rewrite may grow the published coverage, never shrink it.
    _job("second", 7.0, monkeypatch, publish=False, start="2025-12-31", periods=5)
    service._execute("second")

    [after] = _artifacts()
    assert after.artifact_id == before.artifact_id
    assert after.publication.status == PublicationStatus.PUBLISHED
    assert (after.coverage.temporal.start, after.coverage.temporal.end) == ("2025-12-31", "2026-01-04")
    [event] = _record("second").events
    assert event.data["artifact_id"] == before.artifact_id
    assert (event.data["current_start"], event.data["current_end"]) == ("2025-12-31", "2026-01-04")
    assert [item.event_id for item in received] == [_record("first").events[0].event_id, event.event_id]


@pytest.mark.parametrize(
    ("start", "periods", "side"),
    [("2025-12-30", 3, "ends at 2026-01-01"), ("2026-01-02", 5, "starts at 2026-01-02")],
    ids=["ends-earlier", "starts-later"],
)
def test_a_rewrite_that_would_shrink_a_published_dataset_is_refused(
    service: Any,
    received: list[JobEvent],
    monkeypatch: pytest.MonkeyPatch,
    start: str,
    periods: int,
    side: str,
) -> None:
    _job("first", 5.0, monkeypatch)
    service._execute("first")
    [before] = _artifacts()
    first_event = _assert_one_published_event("first")

    # Only the updated periods, as a trigger over [previous_end, current_end] computes them.
    _job("second", 7.0, monkeypatch, start=start, periods=periods, max_attempts=2)
    assert service._execute("second") is None  # permanent: not retried
    second = _record("second")
    assert second.status == OpenEOJobStatus.ERROR
    assert "Refusing to rewrite published dataset" in (second.error_message or "") and side in (
        second.error_message or ""
    )
    # Nothing planned, committed, registered or announced; the history is intact.
    assert second.publication is None and second.events == [] and _commits("second") == []
    [after] = _artifacts()
    assert after == before
    assert (after.coverage.temporal.start, after.coverage.temporal.end) == ("2026-01-01", "2026-01-03")
    assert received == [first_event]


def test_a_result_without_temporal_coverage_cannot_replace_a_published_temporal_dataset(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("first", 5.0, monkeypatch)
    service._execute("first")
    [before] = _artifacts()
    first_event = _assert_one_published_event("first")

    _job("second", 7.0, monkeypatch, temporal=False)
    assert service._execute("second") is None  # permanent: not retried
    second = _record("second")
    assert second.status == OpenEOJobStatus.ERROR
    assert "no temporal coverage to compare" in (second.error_message or "")
    assert second.publication is None and second.events == [] and _commits("second") == []
    [after] = _artifacts()
    assert after == before and received == [first_event]


def test_a_non_temporal_published_dataset_may_be_rewritten(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("first", 5.0, monkeypatch, temporal=False)
    service._execute("first")
    [before] = _artifacts()
    assert before.publication.status == PublicationStatus.PUBLISHED
    assert (before.coverage.temporal.start, before.coverage.temporal.end) == (None, None)
    _job("second", 7.0, monkeypatch, temporal=False)
    service._execute("second")
    assert _record("second").status == OpenEOJobStatus.FINISHED
    [after] = _artifacts()
    assert after.artifact_id == before.artifact_id and len(_record("second").events) == 1


def test_an_unpublished_dataset_may_be_rewritten_to_any_range(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("first", 5.0, monkeypatch, publish=False)
    service._execute("first")
    _job("second", 7.0, monkeypatch, publish=False, start="2026-01-02", periods=1)
    service._execute("second")
    assert _record("second").status == OpenEOJobStatus.FINISHED
    [artifact] = _artifacts()
    assert (artifact.coverage.temporal.start, artifact.coverage.temporal.end) == ("2026-01-02", "2026-01-02")


# --- duplicates ------------------------------------------------------------------------------


def test_completing_or_dispatching_again_adds_no_logical_event(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")
    event = _assert_one_published_event("job-1")
    snapshot = _record("job-1").publication.snapshot_id  # type: ignore[union-attr]
    assert snapshot is not None

    openeo_jobs.complete_managed_publication("job-1", snapshot)
    service._after_finished(_record("job-1"))  # a repeated completion notification
    assert [e.event_id for e in _record("job-1").events] == [event.event_id]
    assert len(_artifacts()) == 1
    # The consumer may see the event again; it is the same logical event, by id.
    assert {e.event_id for e in received} == {event.event_id} and len(received) == 2
    assert [e.event_id for e in persisted_events()] == [event.event_id]


def test_a_deliberate_rerun_that_commits_again_is_a_new_update(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")
    first = _assert_one_published_event("job-1")
    # A deliberate re-run under the same job id, as `start_job` records it, run here.
    openeo_jobs.store_update_job(
        "job-1", lambda r: r.model_copy(update={"status": OpenEOJobStatus.QUEUED, "attempt": 0})
    )
    service._execute("job-1")
    record = _record("job-1")
    assert record.status == OpenEOJobStatus.FINISHED
    # It committed a new snapshot: a second logical update, which consumers act on again.
    assert len(record.events) == 2 and record.events[0] == first
    second = record.events[1]
    assert second.event_id != first.event_id
    assert second.data["output_snapshot_id"] != first.data["output_snapshot_id"] == _commits("job-1")[0]
    assert second.data["output_snapshot_id"] == _head()
    assert [e.event_id for e in received] == [first.event_id, second.event_id]


# --- crash recovery --------------------------------------------------------------------------


def test_a_crash_before_the_commit_gate_publishes_nothing_and_runs_again(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _wrap_writer(monkeypatch, before_gate=True)
    with pytest.raises(_Crash):
        service._execute("job-1")
    left = _record("job-1")
    assert left.status == OpenEOJobStatus.RUNNING and left.publication is not None
    assert left.publication.state == "planned"

    assert service._recover("job-1") is True  # a user-submitted job, requeued all the same
    requeued = _record("job-1")
    assert requeued.status == OpenEOJobStatus.QUEUED and requeued.publication is None and requeued.events == []
    assert _artifacts() == [] and received == []

    service._execute("job-1")
    _assert_one_published_event("job-1")


def test_a_crash_between_the_gate_and_the_commit_publishes_nothing(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _wrap_writer(monkeypatch, after_gate=True)
    with pytest.raises(_Crash):
        service._execute("job-1")
    left = _record("job-1")
    assert left.publication is not None and left.publication.state == "committing" and left.publishing is True

    assert service._recover("job-1") is True
    assert _record("job-1").publication is None and _artifacts() == [] and received == []


def test_a_crash_right_after_the_commit_completes_the_publication_on_restart(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _wrap_writer(monkeypatch, after_commit=True)
    with pytest.raises(_Crash):
        service._execute("job-1")
    assert _record("job-1").publication.state == "committing"  # type: ignore[union-attr]
    assert _artifacts() == []  # committed, not yet registered
    committed = _head()

    assert service._recover("job-1") is False  # finished, not run again
    event = _assert_one_published_event("job-1")
    [artifact] = _artifacts()
    assert artifact.provenance is not None and artifact.provenance.output_snapshot_id == committed
    assert event.data["output_snapshot_id"] == committed
    assert received == [event]


def test_recovery_waits_when_another_writer_holds_the_store(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _wrap_writer(monkeypatch, after_commit=True)
    with pytest.raises(_Crash):
        service._execute("job-1")
    lock = ingestion_services._acquire_store_lock(_store())
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="recovery deferred"):
            service._recover("job-1")
        pending = _record("job-1")
        assert pending.status == OpenEOJobStatus.RUNNING
        assert pending.publication is not None and pending.publication.state == "committing"
        assert pending.events == [] and _artifacts() == []
    finally:
        lock.release()
    assert service._recover("job-1") is False
    assert received == [_assert_one_published_event("job-1")]


def test_startup_recovery_retries_after_the_store_lock_is_released(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _wrap_writer(monkeypatch, after_commit=True)
    with pytest.raises(_Crash):
        service._execute("job-1")
    lock = ingestion_services._acquire_store_lock(_store())
    assert lock.acquire(blocking=False)
    service.lease_poll_seconds = 0.01
    try:
        service.recover_pending_jobs()
        pending = _record("job-1")
        assert pending.status == OpenEOJobStatus.RUNNING
        assert pending.publication is not None and pending.publication.state == "committing"
    finally:
        lock.release()

    deadline = time.monotonic() + 5
    while _record("job-1").status != OpenEOJobStatus.FINISHED and time.monotonic() < deadline:
        time.sleep(0.01)

    assert received == [_assert_one_published_event("job-1")]


def test_a_crash_after_registration_registers_once_and_emits_once(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _crash_once_after(monkeypatch, ingestion_services, "register_artifact_record", "after registration")
    with pytest.raises(_Crash):
        service._execute("job-1")
    [registered] = _artifacts()
    assert _record("job-1").events == []

    assert service._recover("job-1") is False
    event = _assert_one_published_event("job-1")
    [artifact] = _artifacts()
    assert artifact.artifact_id == registered.artifact_id  # the upsert kept its identity
    assert received == [event]


def test_a_crash_after_the_event_is_durable_finishes_without_a_second_event(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    _crash_once_after(monkeypatch, openeo_jobs, "complete_managed_publication", "after the event, before FINISHED")
    with pytest.raises(_Crash):
        service._execute("job-1")
    left = _record("job-1")
    assert left.status == OpenEOJobStatus.RUNNING and len(left.events) == 1
    assert received == []

    assert service._recover("job-1") is False
    event = _assert_one_published_event("job-1")
    assert event == left.events[0]  # the same durable event, not a new one
    assert received == [event]
    assert service._recover("job-1") is False  # recovering again changes nothing
    assert _record("job-1").events == [event] and received == [event]


def test_a_crash_after_finished_but_before_dispatch_is_delivered_by_startup_replay(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger
    from open_climate_service.automation.service import WorkflowAutomationService

    _job("job-1", 5.0, monkeypatch)

    def die_in_listener(record: OpenEOJobRecord) -> None:
        raise _Crash("after FINISHED, before the events were dispatched")

    service.set_finished_listener(die_in_listener)
    with pytest.raises(_Crash):
        service._execute("job-1")
    event = _assert_one_published_event("job-1")
    assert received == []  # the live dispatch never ran
    # Normal recovery no longer looks at a FINISHED job.
    assert service._recover("job-1") is False and received == []

    # Startup replay reads the event from the openEO job that owns it.
    submitted: list[str] = []
    fake = SimpleNamespace(
        create_triggered_job=lambda body, *, source_event_id, trigger_id, max_attempts: (
            submitted.append(source_event_id) or SimpleNamespace(id=f"job-for-{source_event_id}"),
            True,
        ),
        start_triggered_job=lambda job_id: True,
    )
    config = AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id="after-derived",
                on_update_of=_DATASET,
                workflow_id="aggregate_by_features",
                arguments={"dataset_id": "$event.dataset_id"},
                replay_existing=True,
            )
        ]
    )
    automation = WorkflowAutomationService(config_loader=lambda: config, openeo_service=fake)  # type: ignore[arg-type]
    automation.replay()
    assert submitted == [event.event_id]


# --- ordinary failures after the commit -----------------------------------------------------


class _Failing:
    """Make one ingestion function raise an ordinary error while ``on``, after or instead of running."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str, *, after: bool = False, once: bool = False) -> None:
        self.on, self.calls = True, 0
        original = getattr(ingestion_services, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.on and not after:
                self.on = not once
                raise RuntimeError(f"{name} failed")
            result = original(*args, **kwargs)
            if self.on and after:
                raise RuntimeError(f"{name} failed after running")
            return result

        monkeypatch.setattr(ingestion_services, name, wrapped)


def test_a_registration_failure_after_the_commit_is_completed_in_the_same_attempt(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    # The attempt's registration fails; the reconciliation's own, right after, succeeds.
    failing = _Failing(monkeypatch, "register_artifact_record", once=True)
    service._execute("job-1")
    assert failing.calls == 2
    event = _assert_one_published_event("job-1")
    assert len(_commits("job-1")) == 1 and received == [event]


def test_registration_failing_before_writing_anything_is_completed_at_restart(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    failing = _Failing(monkeypatch, "register_artifact_record")
    service._execute("job-1")
    left = _record("job-1")
    # Committed, not registered, not announced: and not forgotten.
    assert left.status == OpenEOJobStatus.ERROR and left.publication is not None
    assert left.publication.state == "committing" and left.publishing is True
    assert left.events == [] and _artifacts() == [] and received == [] and len(_commits("job-1")) == 1
    with pytest.raises(HTTPException):
        service.cancel_job("job-1")  # nothing to cancel: the job ended

    failing.on = False
    service.reconcile_publications()  # the next start
    event = _assert_one_published_event("job-1")
    assert _record("job-1").error_message is None
    assert event.data["output_snapshot_id"] == _commits("job-1")[0]
    assert received == [event]
    service.reconcile_publications()  # again: nothing more
    assert _record("job-1").events == [event] and received == [event]


def test_failed_publication_completion_keeps_retrying_after_startup(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    failing = _Failing(monkeypatch, "register_artifact_record")
    service._execute("job-1")
    assert _record("job-1").status == OpenEOJobStatus.ERROR

    service.lease_poll_seconds = 0.01
    service.reconcile_publications()
    assert _record("job-1").publication is not None
    failing.on = False

    deadline = time.monotonic() + 5
    while _record("job-1").status != OpenEOJobStatus.FINISHED and time.monotonic() < deadline:
        time.sleep(0.01)

    assert received == [_assert_one_published_event("job-1")]


def test_a_job_with_an_unresolved_commit_cannot_be_deleted(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    failing = _Failing(monkeypatch, "register_artifact_record")
    service._execute("job-1")
    assert _record("job-1").status == OpenEOJobStatus.ERROR
    with pytest.raises(HTTPException, match="unresolved publication commit"):
        service.delete_job("job-1")
    assert _record("job-1").publication is not None
    failing.on = False
    service.reconcile_publications()
    assert received == [_assert_one_published_event("job-1")]


def test_publication_failing_after_registration_is_completed_at_restart(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    failing = _Failing(monkeypatch, "publish_artifact_record")
    service._execute("job-1")
    left = _record("job-1")
    [registered] = _artifacts()
    assert registered.publication.status == PublicationStatus.UNPUBLISHED
    assert left.status == OpenEOJobStatus.ERROR and left.publication.state == "committing"  # type: ignore[union-attr]
    assert left.events == [] and received == []

    failing.on = False
    service.reconcile_publications()
    event = _assert_one_published_event("job-1")
    [artifact] = _artifacts()
    assert artifact.artifact_id == registered.artifact_id
    assert artifact.publication.status == PublicationStatus.PUBLISHED
    assert received == [event]


def test_a_retry_completes_the_earlier_commit_instead_of_computing_again(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_runs: list[str] = []
    _job("job-1", 5.0, monkeypatch, graph_runs=graph_runs, max_attempts=2)
    failing = _Failing(monkeypatch, "register_artifact_record")
    delay = service._execute("job-1")
    queued = _record("job-1")
    assert delay is not None and queued.status == OpenEOJobStatus.QUEUED
    assert queued.publication is not None and queued.publication.state == "committing"
    # Still non-cancellable: the commit may have landed.
    with pytest.raises(HTTPException) as refused:
        service.cancel_job("job-1")
    assert refused.value.status_code == 409

    failing.on = False
    service._execute("job-1")
    event = _assert_one_published_event("job-1")
    assert graph_runs == ["job-1"] and len(_commits("job-1")) == 1  # one computation, one commit
    assert received == [event]


def test_the_final_attempt_failing_after_the_commit_is_completed_at_restart(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_runs: list[str] = []
    _job("job-1", 5.0, monkeypatch, graph_runs=graph_runs, max_attempts=2)
    failing = _Failing(monkeypatch, "register_artifact_record")
    assert service._execute("job-1") is not None  # first attempt: retried
    assert service._execute("job-1") is None  # final attempt: completing fails again
    left = _record("job-1")
    assert left.status == OpenEOJobStatus.ERROR and left.attempt == 2
    assert left.publication is not None and left.publication.state == "committing"
    assert graph_runs == ["job-1"] and len(_commits("job-1")) == 1 and received == []

    failing.on = False
    service.reconcile_publications()
    event = _assert_one_published_event("job-1")
    assert received == [event] and len(_commits("job-1")) == 1


# --- cancellation and concurrent writers ----------------------------------------------------


def test_a_cancel_before_the_gate_emits_nothing_and_clears_the_plan(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    original = downloader.write_to_icechunk_store

    def cancel_then_write(*args: Any, **kwargs: Any) -> Any:
        gate = kwargs["before_commit"]

        def cancel_then_gate() -> None:
            service.cancel_job("job-1")
            gate()

        return original(*args, **{**kwargs, "before_commit": cancel_then_gate})

    monkeypatch.setattr(downloader, "write_to_icechunk_store", cancel_then_write)
    service._execute("job-1")
    record = _record("job-1")
    assert record.status == OpenEOJobStatus.CANCELED and record.publication is None
    assert record.events == [] and received == [] and _artifacts() == []
    assert service._recover("job-1") is False  # a cancelled job is never requeued
    assert _record("job-1").status == OpenEOJobStatus.CANCELED


def test_a_cancel_after_the_gate_is_refused_and_the_publication_completes(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    refused: list[int] = []

    def cancel_after_commit(*args: Any, **kwargs: Any) -> None:
        try:
            service.cancel_job("job-1")
        except HTTPException as exc:
            refused.append(exc.status_code)

    monkeypatch.setattr(openeo_jobs, "write_dataset_thumbnail", cancel_after_commit)
    service._execute("job-1")
    assert refused == [409]
    event = _assert_one_published_event("job-1")
    assert received == [event]


def test_a_store_held_by_another_writer_emits_no_success(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    lock = ingestion_services._acquire_store_lock(_store())
    assert lock.acquire(blocking=False)
    try:
        service._execute("job-1")
    finally:
        lock.release()
    record = _record("job-1")
    assert record.status == OpenEOJobStatus.ERROR and "being written" in (record.error_message or "")
    assert record.events == [] and record.publication is None and received == [] and _artifacts() == []


def test_a_commit_conflict_emits_no_success(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    original = downloader.write_to_icechunk_store

    def conflict(*args: Any, **kwargs: Any) -> Any:
        gate = kwargs["before_commit"]

        def gate_then_conflict() -> None:
            gate()
            raise RuntimeError("Icechunk commit conflict: the branch moved")

        return original(*args, **{**kwargs, "before_commit": gate_then_conflict})

    monkeypatch.setattr(downloader, "write_to_icechunk_store", conflict)
    service._execute("job-1")
    record = _record("job-1")
    assert record.status == OpenEOJobStatus.ERROR and "conflict" in (record.error_message or "")
    assert record.events == [] and received == [] and _artifacts() == []


# --- provenance and identities ---------------------------------------------------------------


def test_provenance_keeps_producer_inputs_and_the_output_snapshot_apart_from_the_version(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")
    [artifact] = _artifacts()
    provenance = artifact.provenance
    assert provenance is not None
    assert provenance.producer_kind == "openeo_job" and provenance.producer_job_id == "job-1"
    assert provenance.execution == _EVIDENCE  # the execution evidence, verbatim
    assert provenance.execution["process_sha256"] == "f" * 64
    [source] = provenance.execution["sources"]
    assert (source["artifact_id"], source["source_dataset_id"], source["snapshot_id"]) == (
        "input-artifact",
        "chirps3_precipitation_daily",
        "INPUTSNAPSHOT0000000",
    )
    assert provenance.output_snapshot_id == _head()
    # Three identities, kept apart: the job, the logical artifact, the physical snapshot.
    assert artifact.version is None
    assert provenance.output_snapshot_id not in {artifact.artifact_id, "job-1"}


def test_the_writer_returns_the_committed_snapshot(tmp_path: Path) -> None:
    store = tmp_path / "direct.icechunk"
    snapshot = downloader.write_to_icechunk_store(_cube(1.0), store, commit_message="direct")
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
    assert snapshot == repo.lookup_branch("main")


# --- finding the commit after a restart ------------------------------------------------------


def _commit(store: Path, message: str) -> str:
    return downloader.write_to_icechunk_store(_cube(1.0), store, commit_message=message)


def test_the_commit_is_found_by_its_own_message_not_by_a_moved_head(tmp_path: Path) -> None:
    store = tmp_path / "history.icechunk"
    mine = "Published from openEO job job-1 attempt 1: ds"
    # A new store has no base snapshot.
    assert openeo_jobs.store_head(store) is None
    assert openeo_jobs.committed_snapshot(store, mine, None) is None

    first = _commit(store, mine)
    assert openeo_jobs.committed_snapshot(store, mine, None) == first  # the job's commit is the head

    later = _commit(store, "Synced by another writer")
    assert later != first
    assert openeo_jobs.committed_snapshot(store, mine, None) == first  # found beneath a later commit

    # A moved head is not proof: no commit of this attempt above its base.
    base = openeo_jobs.store_head(store)
    _commit(store, "Yet another writer")
    assert openeo_jobs.committed_snapshot(store, mine, base) is None
    # The search stops at the base: an earlier commit with the same message is not this attempt's.
    assert openeo_jobs.committed_snapshot(store, mine, later) is None


# --- the shared event log --------------------------------------------------------------------


def test_the_event_log_combines_both_owners_in_a_stable_order(
    service: Any, received: list[JobEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from open_climate_service.jobs.models import JobRecord, JobStatus

    _job("job-1", 5.0, monkeypatch)
    service._execute("job-1")
    openeo_event = _record("job-1").events[0]
    same_time = openeo_event.time
    native_events = [
        JobEvent(event_id="b-native:0", type="dataset.updated", source="/datasets/x", time=same_time, data={}),
        JobEvent(event_id="a-native:0", type="dataset.updated", source="/datasets/x", time=same_time, data={}),
    ]
    native_store.create_job_record(
        JobRecord(
            job_id="native-1",
            process_id="sync",
            status=JobStatus.SUCCESSFUL,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            events=native_events,
        )
    )
    log = read_event_log()
    assert [event.event_id for event in log.events] == sorted(["b-native:0", "a-native:0", openeo_event.event_id])
    # Neither a finished native job nor any openEO job holds the replay bound back.
    _job("job-2", 6.0, monkeypatch)  # queued, never run
    assert read_event_log().open_since is None
