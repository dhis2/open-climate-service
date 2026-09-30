"""CLIM-1229: one dask pool per process, and job slots shared by native and openEO jobs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from typing import Any

import dask.array as da
import dask.config
import dask.threaded
import pytest

from open_climate_service.jobs.models import JobRecord, JobStatus
from open_climate_service.jobs.service import JobService
from open_climate_service.openeo.jobs import OpenEOJobService
from open_climate_service.shared import compute
from open_climate_service.shared.compute import JobSlots, SharedDaskPool

_ran = threading.Event()
_release = threading.Event()


def _job_callable() -> dict[str, object]:
    _ran.set()
    return {"ok": True}


def _blocking_job_callable() -> dict[str, object]:
    _ran.set()
    _release.wait(timeout=10)
    return {"ok": True}


_attempts: list[int] = []


def _fails_once_callable() -> dict[str, object]:
    _attempts.append(1)
    if len(_attempts) == 1:
        raise RuntimeError("first attempt fails")
    return {"ok": True}


@pytest.fixture(autouse=True)
def _fresh_state() -> Generator[None]:
    _ran.clear()
    _release.clear()
    _attempts.clear()
    compute.reset_job_slots()
    yield
    _release.set()
    compute.reset_job_slots()


@pytest.fixture
def one_slot(monkeypatch: pytest.MonkeyPatch) -> JobSlots:
    monkeypatch.setenv(compute.MAX_CONCURRENT_JOBS_ENV, "1")
    return compute.get_job_slots()


@pytest.fixture
def persisted(monkeypatch: pytest.MonkeyPatch) -> dict[str, JobRecord]:
    records: dict[str, JobRecord] = {}
    monkeypatch.setattr("open_climate_service.jobs.service.JobService._enqueue_job", lambda self, job_id: None)
    monkeypatch.setattr(
        "open_climate_service.jobs.store.create_job_record", lambda record: records.setdefault(record.job_id, record)
    )
    monkeypatch.setattr("open_climate_service.jobs.store.get_job_record", lambda job_id: records.get(job_id))

    def mutate(job_id: str, mutation: Callable[[JobRecord], JobRecord]) -> JobRecord:
        records[job_id] = mutation(records[job_id])
        return records[job_id]

    monkeypatch.setattr("open_climate_service.jobs.store.mutate_job_record", mutate)
    monkeypatch.setattr(compute, "_SLOT_POLL_SECONDS", 0.05)
    return records


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


# --- the dask pool -------------------------------------------------------------------------


def _compute_from_threads(count: int) -> set[threading.Thread]:
    """Compute from `count` threads at once; the threads that got a dask pool of their own."""
    with_own_pool: set[threading.Thread] = set()
    barrier = threading.Barrier(count)

    def work() -> None:
        da.ones((64, 64), chunks=(8, 8)).sum().compute()
        if threading.current_thread() in dask.threaded.pools:
            with_own_pool.add(threading.current_thread())
        barrier.wait(timeout=10)  # all alive together, as concurrent jobs are

    threads = [threading.Thread(target=work) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return with_own_pool


def test_without_the_shared_pool_every_thread_starts_a_pool_of_its_own() -> None:
    """The behaviour this fixes, pinned so a dask upgrade that changes it is noticed."""
    with dask.config.set(pool=None, num_workers=None):
        assert len(_compute_from_threads(3)) == 3


def test_computations_from_many_threads_share_one_bounded_pool() -> None:
    pool = SharedDaskPool(2)
    try:
        with dask.config.set(pool=pool):
            assert _compute_from_threads(6) == set()
        workers = [thread for thread in threading.enumerate() if thread.name.startswith("dask-shared")]
        assert 1 <= len(workers) <= 2
    finally:
        pool.shutdown()


def test_a_computation_inside_a_task_runs_inline_instead_of_deadlocking() -> None:
    pool = SharedDaskPool(1)

    def nested(block: Any) -> Any:
        # The pool's only thread is running this task: waiting for another would hang.
        return block + da.ones(4, chunks=2).sum().compute()

    try:
        with dask.config.set(pool=pool):
            done: list[Any] = []
            runner = threading.Thread(
                target=lambda: done.append(da.zeros(4, chunks=2).map_blocks(nested).compute()), daemon=True
            )
            runner.start()
            runner.join(timeout=10)
            assert done, "nested computation deadlocked"
            assert list(done[0]) == [4.0, 4.0, 4.0, 4.0]
    finally:
        pool.shutdown()


def test_an_exception_in_an_inline_task_reaches_the_caller() -> None:
    pool = SharedDaskPool(1)

    def boom() -> None:
        raise ValueError("inside")

    def run_inside() -> None:
        future = pool.submit(boom)
        with pytest.raises(ValueError, match="inside"):
            future.result()

    try:
        pool.submit(run_inside).result(timeout=10)
    finally:
        pool.shutdown()


def test_install_is_idempotent_and_sets_the_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "_pool", None)
    with dask.config.set(pool=None, num_workers=3):
        first = compute.install_shared_dask_pool()
        try:
            assert compute.install_shared_dask_pool() is first
            assert dask.config.get("pool") is first
            assert first._max_workers == 3
        finally:
            first.shutdown()


def test_thread_budget_defaults_to_half_the_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    for cores, expected in [(12, 6), (8, 4), (3, 1), (1, 1)]:
        monkeypatch.setattr(compute.os, "cpu_count", lambda cores=cores: cores)
        with dask.config.set(num_workers=None):
            assert compute.dask_thread_budget() == expected


def test_thread_budget_follows_dask_num_workers() -> None:
    with dask.config.set(num_workers=6):
        assert compute.dask_thread_budget() == 6


# --- the job limit -------------------------------------------------------------------------


def test_job_limit_defaults_and_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(compute.MAX_CONCURRENT_JOBS_ENV, raising=False)
    assert compute.max_concurrent_jobs() == compute.DEFAULT_MAX_CONCURRENT_JOBS
    monkeypatch.setenv(compute.MAX_CONCURRENT_JOBS_ENV, "5")
    assert compute.max_concurrent_jobs() == 5


@pytest.mark.parametrize("raw", ["0", "-1", "two"])
def test_job_limit_refuses_a_value_that_is_not_a_positive_integer(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(compute.MAX_CONCURRENT_JOBS_ENV, raw)
    with pytest.raises(ValueError, match=compute.MAX_CONCURRENT_JOBS_ENV):
        compute.max_concurrent_jobs()


def _run_in_background(target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_a_native_job_waits_accepted_for_a_free_slot(persisted: dict[str, JobRecord], one_slot: JobSlots) -> None:
    service = JobService()
    job = service.submit_callable_job(func=_job_callable, label="ingestion", request={})
    assert one_slot.acquire(should_stop=lambda: False)

    runner = _run_in_background(lambda: service._execute_job(job.job_id))
    _wait_until(lambda: persisted[job.job_id].progress.message == "Waiting for a free job slot")
    assert persisted[job.job_id].status == JobStatus.ACCEPTED
    assert not _ran.is_set()

    one_slot.release()
    runner.join(timeout=5)
    assert persisted[job.job_id].status == JobStatus.SUCCESSFUL
    assert one_slot.acquire(should_stop=lambda: True), "the job gave its slot back"


def test_a_native_job_cancelled_while_waiting_never_runs(persisted: dict[str, JobRecord], one_slot: JobSlots) -> None:
    service = JobService()
    job = service.submit_callable_job(func=_job_callable, label="ingestion", request={})
    assert one_slot.acquire(should_stop=lambda: False)

    runner = _run_in_background(lambda: service._execute_job(job.job_id))
    _wait_until(lambda: persisted[job.job_id].progress.message == "Waiting for a free job slot")
    persisted[job.job_id] = persisted[job.job_id].model_copy(update={"cancel_requested": True})
    runner.join(timeout=5)

    assert persisted[job.job_id].status == JobStatus.CANCELLED
    assert not _ran.is_set()


def test_a_native_job_still_waiting_at_shutdown_stays_accepted(
    persisted: dict[str, JobRecord], one_slot: JobSlots
) -> None:
    service = JobService()
    job = service.submit_callable_job(func=_job_callable, label="ingestion", request={})
    assert one_slot.acquire(should_stop=lambda: False)

    runner = _run_in_background(lambda: service._execute_job(job.job_id))
    _wait_until(lambda: persisted[job.job_id].progress.message == "Waiting for a free job slot")
    service.shutdown()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert persisted[job.job_id].status == JobStatus.ACCEPTED
    assert not _ran.is_set()


def _failed_once_and_backing_off(
    persisted: dict[str, JobRecord], monkeypatch: pytest.MonkeyPatch, delay: int
) -> tuple[JobService, JobRecord, list[str]]:
    """A job whose first attempt failed, left waiting out a `delay`-second backoff."""
    enqueued: list[str] = []
    monkeypatch.setattr(JobService, "_enqueue_job", lambda self, job_id: enqueued.append(job_id))
    monkeypatch.setattr("open_climate_service.jobs.service._retry_delay_seconds", lambda attempt: delay)
    service = JobService()
    job = service.submit_callable_job(func=_fails_once_callable, label="sync", request={}, max_attempts=2)
    enqueued.clear()
    service._execute_job(job.job_id)
    return service, job, enqueued


def test_a_retry_backoff_holds_neither_a_slot_nor_a_worker(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, job, enqueued = _failed_once_and_backing_off(persisted, monkeypatch, delay=60)

    # `_execute_job` returned, so the executor thread is free, and the slot with it.
    assert persisted[job.job_id].status == JobStatus.RETRYING
    assert one_slot.acquire(should_stop=lambda: True)
    one_slot.release()
    assert enqueued == []
    service.shutdown()


def test_a_job_is_requeued_once_its_backoff_has_passed(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, job, enqueued = _failed_once_and_backing_off(persisted, monkeypatch, delay=0)
    _wait_until(lambda: enqueued == [job.job_id])

    service._execute_job(job.job_id)

    assert persisted[job.job_id].status == JobStatus.SUCCESSFUL
    assert len(_attempts) == 2


def test_cancelling_during_a_backoff_is_recorded_at_once(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, job, enqueued = _failed_once_and_backing_off(persisted, monkeypatch, delay=60)

    record = service.request_cancellation(job.job_id)

    assert record.status == JobStatus.CANCELLED
    assert persisted[job.job_id].progress.message == "Cancelled before retry execution resumed"
    assert enqueued == []
    service.shutdown()


def test_a_cancellation_that_beats_the_timer_still_ends_the_job(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled after the attempt failed but before its backoff timer existed."""
    original = JobService._schedule_retry

    def cancel_first(self: JobService, job_id: str, seconds: int) -> None:
        persisted[job_id] = persisted[job_id].model_copy(update={"cancel_requested": True})
        original(self, job_id, seconds)

    monkeypatch.setattr(JobService, "_schedule_retry", cancel_first)
    service, job, enqueued = _failed_once_and_backing_off(persisted, monkeypatch, delay=60)

    assert enqueued == [job.job_id], "requeued at once instead of after the backoff"
    service._execute_job(job.job_id)
    assert persisted[job.job_id].status == JobStatus.CANCELLED
    assert len(_attempts) == 1


def test_a_job_backing_off_at_shutdown_stays_retrying(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, job, _ = _failed_once_and_backing_off(persisted, monkeypatch, delay=0)
    service.shutdown()
    time.sleep(0.1)

    # Whether the timer fired first or not, the next start's recovery requeues it.
    assert persisted[job.job_id].status == JobStatus.RETRYING
    assert len(_attempts) == 1


def test_openeo_and_native_jobs_share_the_slots(
    persisted: dict[str, JobRecord], one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = JobService()
    job = native.submit_callable_job(func=_blocking_job_callable, label="ingestion", request={})
    native_runner = _run_in_background(lambda: native._execute_job(job.job_id))
    _wait_until(_ran.is_set)

    openeo = OpenEOJobService(max_workers=1)
    executed: list[str] = []
    monkeypatch.setattr(openeo, "_execute", executed.append)
    monkeypatch.setattr("open_climate_service.openeo.jobs._cancel_requested", lambda job_id: False)
    openeo_runner = _run_in_background(lambda: openeo._run_job("openeo-1"))
    time.sleep(0.2)
    assert executed == [], "the openEO job ran while the ingest held the only slot"

    _release.set()
    native_runner.join(timeout=5)
    openeo_runner.join(timeout=5)
    assert executed == ["openeo-1"]
    openeo.shutdown()


def test_an_openeo_job_cancelled_while_waiting_is_recorded_without_a_slot(
    one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compute, "_SLOT_POLL_SECONDS", 0.05)
    assert one_slot.acquire(should_stop=lambda: False)
    openeo = OpenEOJobService(max_workers=1)
    executed: list[str] = []
    monkeypatch.setattr(openeo, "_execute", executed.append)
    monkeypatch.setattr("open_climate_service.openeo.jobs._cancel_requested", lambda job_id: True)

    openeo._run_job("openeo-1")

    # `_execute` sees the cancellation first and records it; nothing computes.
    assert executed == ["openeo-1"]
    one_slot.release()
    openeo.shutdown()


def test_an_openeo_job_still_waiting_at_shutdown_is_left_queued(
    one_slot: JobSlots, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compute, "_SLOT_POLL_SECONDS", 0.05)
    assert one_slot.acquire(should_stop=lambda: False)
    openeo = OpenEOJobService(max_workers=1)
    executed: list[str] = []
    monkeypatch.setattr(openeo, "_execute", executed.append)
    monkeypatch.setattr("open_climate_service.openeo.jobs._cancel_requested", lambda job_id: False)

    runner = _run_in_background(lambda: openeo._run_job("openeo-1"))
    time.sleep(0.1)
    openeo.shutdown()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert executed == []
    one_slot.release()
