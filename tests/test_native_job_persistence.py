"""Concurrent writers and interrupted replacement must retain committed job state."""

import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest

from open_climate_service.jobs import store
from open_climate_service.jobs.models import JobRecord, JobStatus
from open_climate_service.shared.time import utc_now


def _increment(directory: str, ready: Any, start: Any):
    store.JOBS_DIR = Path(directory)
    store.JOBS_INDEX_PATH = store.JOBS_DIR / "jobs.json"
    ready.put(True)
    start.wait(10)
    for _ in range(10):

        def mutate(record: JobRecord):
            time.sleep(0.002)
            return record.model_copy(update={"request": {"count": record.request["count"] + 1}})

        store.mutate_job_record("counter", mutate)


def test_processes_do_not_lose_updates_across_atomic_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(store, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(store, "JOBS_INDEX_PATH", tmp_path / "jobs.json")
    store.create_job_record(
        JobRecord(
            job_id="counter",
            process_id="test",
            status=JobStatus.ACCEPTED,
            created_at=utc_now(),
            request={"count": 0},
        )
    )
    ctx = multiprocessing.get_context("spawn")
    ready, start = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_increment, args=(str(tmp_path), ready, start)) for _ in range(4)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=20)
        start.set()
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        ready.close()
    record = store.get_job_record("counter")
    assert record is not None and record.request["count"] == 40


def test_failed_replace_preserves_reservations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from open_climate_service import config
    from open_climate_service.exports.reservations import find_delivery, reserve_delivery, submission_lock
    from open_climate_service.shared import persistence

    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path)
    arguments: dict[str, Any] = dict(
        delivery_job_id="job", export_id="rain", source_job_id="source", dry_run=False, fingerprint="a"
    )
    with submission_lock():
        reserve_delivery("first", **arguments)

    def fail(*args: Any):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(persistence.os, "replace", fail)
    with submission_lock(), pytest.raises(OSError):
        reserve_delivery("second", **arguments)
    assert find_delivery("first") is not None
    assert find_delivery("second") is None
