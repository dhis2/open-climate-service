"""CLIM-1220: a failed ingestion stops completely and releases its store lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException

from open_climate_service.ingestions import services
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_PERIODS = [f"2026-01-{day:02d}" for day in range(1, 7)]


class _Plugin(BaseDatasetPlugin):
    def __init__(self) -> None:
        self.available = list(_PERIODS)
        self.fetched: list[str] = []
        self.max_concurrency = 1

    async def periods(self, start: str, end: str) -> list[str]:
        return [period for period in self.available if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        self.fetched.append(period_id)
        data = xr.DataArray(
            np.full((2, 2), int(period_id[-2:]), dtype="float32"),
            dims=("y", "x"),
            coords={"y": [3.5, 2.5], "x": [1.5, 2.5]},
        )
        return normalize_period(data, variable="precip", period=period_id)


@pytest.fixture
def ingestion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    plugin = _Plugin()
    dataset: dict[str, object] = {
        "id": "conflicted_precip",
        "name": "Conflicted precipitation",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {"plugin": "example.Plugin"},
    }
    store_path = tmp_path / "conflicted.icechunk"
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    return {"plugin": plugin, "dataset": dataset, "store_path": store_path}


def _create(ingestion: dict[str, Any], end: str = "2026-01-06") -> Any:
    return services.create_artifact(
        dataset=ingestion["dataset"],
        start="2026-01-01",
        end=end,
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )


def _committed_messages(store_path: Path) -> list[str]:
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    return [snapshot.message for snapshot in repo.ancestry(branch="main")]


def test_commit_conflict_stops_the_loop_and_releases_the_lock(
    ingestion: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    commits: list[str] = []
    original = icechunk.Session.commit

    def conflicting_commit(self: icechunk.Session, message: str, *args: Any, **kwargs: Any) -> Any:
        commits.append(message)
        if message == "ingest: 2026-01-03":
            raise icechunk.ConflictError("simulated concurrent writer")
        return original(self, message, *args, **kwargs)

    monkeypatch.setattr(icechunk.Session, "commit", conflicting_commit)

    with pytest.raises(Exception, match="simulated concurrent writer"):
        _create(ingestion)

    # Nothing is fetched or committed after the failing period.
    assert commits == ["ingest: 2026-01-01", "ingest: 2026-01-02", "ingest: 2026-01-03"]
    assert "2026-01-05" not in ingestion["plugin"].fetched
    # The lock is free: another ingestion can start at once and resume from the committed prefix.
    lock = services._acquire_store_lock(ingestion["store_path"])
    assert lock.acquire(blocking=False)
    lock.release()
    monkeypatch.setattr(icechunk.Session, "commit", original)
    artifact = _create(ingestion)
    assert (artifact.coverage.temporal.start, artifact.coverage.temporal.end) == ("2026-01-01", "2026-01-06")


# --- a second writer, as another process would be ---------------------------------------


class _OtherProcessLock:
    """Hold a file lock through a separate handle, as a second OCS process would."""

    def __init__(self, path: Path) -> None:
        import portalocker

        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115
        portalocker.lock(self._handle, portalocker.LOCK_EX | portalocker.LOCK_NB)

    def release(self) -> None:
        import portalocker

        portalocker.unlock(self._handle)
        self._handle.close()


def test_store_held_by_another_process_is_refused_before_any_work(ingestion: dict[str, Any]) -> None:
    store_path: Path = ingestion["store_path"]
    other = _OtherProcessLock(store_path.with_name(f"{store_path.name}.lock"))
    try:
        with pytest.raises(HTTPException) as error:
            _create(ingestion)
        assert error.value.status_code == 409
        assert ingestion["plugin"].fetched == []
        assert not store_path.exists()
    finally:
        other.release()

    artifact = _create(ingestion)
    assert artifact.coverage.temporal.end == "2026-01-06"


def test_store_contention_is_refused_before_plugin_is_created(
    ingestion: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path: Path = ingestion["store_path"]
    other = _OtherProcessLock(store_path.with_name(f"{store_path.name}.lock"))
    created: list[bool] = []
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: created.append(True))
    try:
        with pytest.raises(HTTPException, match="already running"):
            _create(ingestion)
    finally:
        other.release()
    assert created == []


def test_publishing_a_managed_dataset_waits_for_the_store_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from open_climate_service.data_manager.services import downloader
    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.openeo import jobs as openeo_jobs

    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        registry_datasets,
        "get_dataset",
        lambda dataset_id: {"id": dataset_id, "name": "Derived precipitation", "variable": "precip"},
    )
    written: list[Path] = []
    monkeypatch.setattr(downloader, "write_to_icechunk_store", lambda ds, path, *args, **kwargs: written.append(path))
    cube = xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((1, 2, 2), dtype="float32"))},
        coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]"), "y": [3.5, 2.5], "x": [1.5, 2.5]},
    )
    lock = services._acquire_store_lock(tmp_path / "derived_precip.icechunk")
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="is being written by an ingestion, sync, or another job"):
            openeo_jobs._write_managed_zarr(cube, {"dataset_id": "derived_precip"})
    finally:
        lock.release()
    assert written == []


# --- the job record ------------------------------------------------------------------------


@pytest.fixture
def jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs import store as native_store

    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "jobs" / "jobs.json")
    job_service_module.reset_job_service()
    service = job_service_module.get_job_service()
    yield service
    job_service_module.reset_job_service()


def _await(jobs: Any, job_id: str, timeout: float = 20.0) -> Any:
    import time

    from open_climate_service.jobs.models import JobStatus

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.get_job_or_404(job_id)
        if record.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish")


def _ingestion_job(**request: Any) -> Any:
    from open_climate_service.ingestions import processes

    return {
        "func": processes.execute_ingestion,
        "label": "ingestion",
        "request": {"dataset_id": "conflicted_precip", "start": "2026-01-01", "end": "2026-01-06", "publish": False}
        | request,
    }


def test_failed_job_exposes_failure_only_after_its_lock_is_free(
    ingestion: dict[str, Any], jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.extents import services as extent_services
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobStatus

    monkeypatch.setattr(registry_datasets, "get_dataset", lambda _: ingestion["dataset"])
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    original_commit = icechunk.Session.commit

    def conflicting_commit(self: icechunk.Session, message: str, *args: Any, **kwargs: Any) -> Any:
        if message == "ingest: 2026-01-03":
            raise icechunk.ConflictError("simulated concurrent writer")
        return original_commit(self, message, *args, **kwargs)

    monkeypatch.setattr(icechunk.Session, "commit", conflicting_commit)
    lock_free_when_failed: list[bool] = []
    original_mutate = native_store.mutate_job_record

    def observe(job_id: str, mutation: Any) -> Any:
        updated = original_mutate(job_id, mutation)
        if updated.status == JobStatus.FAILED:
            probe = services._acquire_store_lock(ingestion["store_path"])
            acquired = probe.acquire(blocking=False)
            lock_free_when_failed.append(acquired)
            if acquired:
                probe.release()
        return updated

    monkeypatch.setattr(native_store, "mutate_job_record", observe)

    failed = _await(jobs, jobs.submit_callable_job(**_ingestion_job()).job_id)

    assert failed.status == JobStatus.FAILED
    assert "simulated concurrent writer" in (failed.error.message if failed.error else "")
    assert lock_free_when_failed == [True]
    assert "2026-01-05" not in ingestion["plugin"].fetched
    # A new ingestion starts at once and completes the dataset.
    monkeypatch.setattr(icechunk.Session, "commit", original_commit)
    retried = _await(jobs, jobs.submit_callable_job(**_ingestion_job()).job_id)
    assert retried.status == JobStatus.SUCCESSFUL, retried.error


def test_job_running_in_another_process_is_not_recovered_or_rewritten(jobs: Any) -> None:
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobRecord, JobStatus
    from open_climate_service.shared.time import utc_now

    ran: list[str] = []
    native_store.create_job_record(
        JobRecord(
            job_id="elsewhere",
            process_id="ingestion",
            status=JobStatus.RUNNING,
            created_at=utc_now(),
            attempt=1,
            request={"__fn_path__": "builtins.print"},
        )
    )
    lease = job_service_module._lease_path("elsewhere")
    other = _OtherProcessLock(lease.with_suffix(lease.suffix + ".lock"))
    try:
        jobs.recover_pending_jobs()
        # A direct enqueue (as a second recovery would do) must not run it either.
        jobs._run_job("elsewhere")
        record = native_store.get_job_record("elsewhere")
        assert record is not None
        assert (record.status, record.attempt, record.error) == (JobStatus.RUNNING, 1, None)
        assert ran == []
    finally:
        other.release()


def test_recovery_still_requeues_a_job_nobody_is_running(
    jobs: Any, ingestion: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.extents import services as extent_services
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobRecord, JobStatus
    from open_climate_service.shared.time import utc_now

    monkeypatch.setattr(registry_datasets, "get_dataset", lambda _: ingestion["dataset"])
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    job = _ingestion_job()
    native_store.create_job_record(
        JobRecord(
            job_id="orphaned",
            process_id="ingestion",
            status=JobStatus.RUNNING,
            created_at=utc_now(),
            attempt=1,
            request={"__fn_path__": "open_climate_service.ingestions.processes.execute_ingestion", **job["request"]},
        )
    )
    jobs.recover_pending_jobs()
    assert _await(jobs, "orphaned").status == JobStatus.SUCCESSFUL


def _leased_running_job(job_id: str, request: dict[str, Any]) -> _OtherProcessLock:
    """A RUNNING job whose lease another process holds, as during an overlapping restart."""
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobRecord, JobStatus
    from open_climate_service.shared.time import utc_now

    native_store.create_job_record(
        JobRecord(
            job_id=job_id,
            process_id="ingestion",
            status=JobStatus.RUNNING,
            created_at=utc_now(),
            attempt=1,
            request={"__fn_path__": "open_climate_service.ingestions.processes.execute_ingestion", **request},
        )
    )
    lease = job_service_module._lease_path(job_id)
    return _OtherProcessLock(lease.with_suffix(lease.suffix + ".lock"))


def test_job_abandoned_by_the_other_process_is_taken_over(
    jobs: Any, ingestion: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.extents import services as extent_services
    from open_climate_service.jobs.models import JobStatus

    monkeypatch.setattr(registry_datasets, "get_dataset", lambda _: ingestion["dataset"])
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    jobs.lease_poll_seconds = 0.05
    other = _leased_running_job("abandoned", _ingestion_job()["request"])
    jobs.recover_pending_jobs()
    assert jobs.get_job_or_404("abandoned").status == JobStatus.RUNNING

    # The old process dies without finishing: the operating system frees its lease.
    other.release()

    record = _await(jobs, "abandoned")
    assert record.status == JobStatus.SUCCESSFUL, record.error
    assert record.attempt == 1  # the interrupted attempt is not counted twice


def test_job_finished_by_the_other_process_is_left_alone(jobs: Any) -> None:
    import time

    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobStatus
    from open_climate_service.shared.time import utc_now

    jobs.lease_poll_seconds = 0.05
    other = _leased_running_job("finished-elsewhere", {"dataset_id": "unused"})
    jobs.recover_pending_jobs()
    finished_at = utc_now()
    native_store.mutate_job_record(
        "finished-elsewhere",
        lambda current: current.model_copy(update={"status": JobStatus.SUCCESSFUL, "finished_at": finished_at}),
    )
    other.release()
    time.sleep(0.5)  # several takeover polls

    record = native_store.get_job_record("finished-elsewhere")
    assert record is not None
    assert (record.status, record.attempt, record.finished_at) == (JobStatus.SUCCESSFUL, 1, finished_at)
    # A worker that wins the lease for a terminal job never runs it either.
    jobs._run_job("finished-elsewhere")
    assert native_store.get_job_record("finished-elsewhere") == record


def test_takeover_stops_at_shutdown(jobs: Any) -> None:
    import time

    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobStatus

    jobs.lease_poll_seconds = 0.05
    other = _leased_running_job("left-at-shutdown", {"dataset_id": "unused"})
    jobs.recover_pending_jobs()
    jobs.shutdown()
    other.release()
    time.sleep(0.3)

    record = native_store.get_job_record("left-at-shutdown")
    assert record is not None and record.status == JobStatus.RUNNING


def test_worker_losing_the_lease_handoff_still_gets_the_job_taken_over(
    jobs: Any, ingestion: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery here released the lease; another process won it before this worker started."""
    import time

    from open_climate_service.data_registry.services import datasets as registry_datasets
    from open_climate_service.extents import services as extent_services
    from open_climate_service.jobs import service as job_service_module
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobRecord, JobStatus
    from open_climate_service.shared.time import utc_now

    monkeypatch.setattr(registry_datasets, "get_dataset", lambda _: ingestion["dataset"])
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    jobs.lease_poll_seconds = 0.05
    native_store.create_job_record(
        JobRecord(
            job_id="handed-off",
            process_id="ingestion",
            status=JobStatus.ACCEPTED,
            created_at=utc_now(),
            request={
                "__fn_path__": "open_climate_service.ingestions.processes.execute_ingestion",
                **_ingestion_job()["request"],
            },
        )
    )
    lease = job_service_module._lease_path("handed-off")
    other = _OtherProcessLock(lease.with_suffix(lease.suffix + ".lock"))

    # Run through the executor so the losing worker remains in `_futures` until
    # `_run_job` returns. Starting the watcher before removing that future used to
    # let takeover recover the job while its replacement enqueue was still refused.
    jobs._enqueue_job("handed-off")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and jobs._watched != {"handed-off"}:
        time.sleep(0.01)
    assert jobs._watched == {"handed-off"}
    assert "handed-off" not in jobs._futures
    assert jobs.get_job_or_404("handed-off").status == JobStatus.ACCEPTED

    # The other process exits without running the job.
    other.release()

    record = _await(jobs, "handed-off")
    assert record.status == JobStatus.SUCCESSFUL, record.error
    assert record.attempt == 1


def test_no_watcher_starts_once_the_service_is_stopping(jobs: Any) -> None:
    from open_climate_service.jobs import service as job_service_module

    lease = job_service_module._lease_path("late")
    other = _OtherProcessLock(lease.with_suffix(lease.suffix + ".lock"))
    try:
        jobs.shutdown()
        jobs._run_job("late")
        assert jobs._watched == set()
    finally:
        other.release()
