"""CLIM-1221: cancelling a running openEO job stops it before it can publish."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service.data_manager.services import downloader
from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared import cancellation
from open_climate_service.shared.cancellation import ExecutionCancelled, cancellation_scope, raise_if_cancelled
from open_climate_service.shared.time import utc_now
from tests.test_managed_publish_units import managed_instance  # noqa: F401  # pyright: ignore[reportUnusedImport]

_DATASET = "derived_precip"


def _cube(value: float) -> xr.Dataset:
    times = pd.date_range("2026-01-01", periods=3, freq="D")
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
    openeo = openeo_jobs.OpenEOJobService()
    try:
        yield openeo
    finally:
        openeo.shutdown()


def _publishing_job(openeo: Any, job_id: str, value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued job whose graph ends in publishing `value` as a managed dataset."""
    monkeypatch.setattr(
        execution,
        "run_process_graph",
        lambda *args, **kwargs: SaveResultEnvelope(
            _cube(value), "ZARR", {"dataset_id": _DATASET, "variable": "precip"}
        ),
    )
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id=job_id, status=OpenEOJobStatus.QUEUED, created=utc_now(), process={"process_graph": {}})
    )


def _cancel_during_the_store_write(openeo: Any, job_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Request cancellation after the data is written into the store session, before its commit."""
    original = downloader.write_to_icechunk_store

    def write(*args: Any, **kwargs: Any) -> Any:
        gate = kwargs["before_commit"]

        def cancel_then_gate() -> None:
            openeo.cancel_job(job_id)
            gate()

        return original(*args, **{**kwargs, "before_commit": cancel_then_gate})

    monkeypatch.setattr(downloader, "write_to_icechunk_store", write)


def _commit_messages(store: Path) -> list[str]:
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store)))
    return [snapshot.message for snapshot in repo.ancestry(branch="main")]


def _stored_value(store: Path) -> float:
    from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

    with open_icechunk_dataset(str(store)) as ds:
        return float(ds["precip"].isel(t=0, y=0, x=0))


def test_cancelling_before_the_commit_publishes_nothing(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _publishing_job(service, "cancelled", 5.0, monkeypatch)
    _cancel_during_the_store_write(service, "cancelled", monkeypatch)

    service._execute("cancelled")

    record = openeo_jobs.store_get_job("cancelled")
    assert record is not None and record.status == OpenEOJobStatus.CANCELED
    assert record.publishing is False
    # Nothing was committed, registered or left behind for the dataset.
    store = downloader.DOWNLOAD_DIR / f"{_DATASET}.icechunk"
    if store.exists():
        assert _commit_messages(store) == ["Repository initialized"]
    assert [r for r in ingestion_services._load_records() if r.dataset_id == _DATASET] == []
    assert registry.get_dataset(_DATASET) is None


def test_a_cancelled_republication_keeps_the_previous_data(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _publishing_job(service, "first", 5.0, monkeypatch)
    service._execute("first")
    assert openeo_jobs.store_get_job("first").status == OpenEOJobStatus.FINISHED  # type: ignore[union-attr]
    store = downloader.DOWNLOAD_DIR / f"{_DATASET}.icechunk"
    before = [r.artifact_id for r in ingestion_services._load_records() if r.dataset_id == _DATASET]

    _publishing_job(service, "second", 9.0, monkeypatch)
    _cancel_during_the_store_write(service, "second", monkeypatch)
    service._execute("second")

    assert openeo_jobs.store_get_job("second").status == OpenEOJobStatus.CANCELED  # type: ignore[union-attr]
    assert _stored_value(store) == 5.0  # the uncommitted 9.0 never became visible
    assert [r.artifact_id for r in ingestion_services._load_records() if r.dataset_id == _DATASET] == before
    assert registry.get_dataset(_DATASET) is not None  # the first run's template is kept


def test_a_cancel_arriving_after_the_point_of_no_return_is_refused(
    service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publishing_job(service, "late", 5.0, monkeypatch)
    refused: list[HTTPException] = []

    def cancel_after_commit(*args: Any, **kwargs: Any) -> None:
        try:
            service.cancel_job("late")
        except HTTPException as exc:
            refused.append(exc)

    monkeypatch.setattr(openeo_jobs, "write_dataset_thumbnail", cancel_after_commit)

    service._execute("late")

    assert [(exc.status_code, exc.detail) for exc in refused] == [
        (409, "Job is publishing its result and can no longer be cancelled")
    ]
    record = openeo_jobs.store_get_job("late")
    assert record is not None
    assert (record.status, record.cancel_requested, record.publishing) == (OpenEOJobStatus.FINISHED, False, False)
    assert [r for r in ingestion_services._load_records() if r.dataset_id == _DATASET]


def test_a_long_computation_stops_within_a_bounded_time(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    cancellation.install_dask_cancellation()
    done: list[int] = []

    def slow(block: np.ndarray) -> np.ndarray:
        time.sleep(0.02)
        done.append(1)
        return block

    def compute(*args: Any, **kwargs: Any) -> Any:
        # 400 tasks of 20 ms: about 8 s if it ran to completion on one thread.
        total = da.ones((400, 10), chunks=(1, 10)).map_blocks(slow).sum().compute(scheduler="single-threaded")
        return SaveResultEnvelope(float(total), "JSON", {})

    monkeypatch.setattr(execution, "run_process_graph", compute)
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id="long", status=OpenEOJobStatus.QUEUED, created=utc_now(), process={"process_graph": {}})
    )
    runner = threading.Thread(target=service._execute, args=("long",))
    runner.start()
    time.sleep(0.3)
    started = time.monotonic()
    service.cancel_job("long")
    runner.join(timeout=10)

    assert not runner.is_alive()
    assert time.monotonic() - started < cancellation.CHECK_INTERVAL_SECONDS + 1.5
    assert openeo_jobs.store_get_job("long").status == OpenEOJobStatus.CANCELED  # type: ignore[union-attr]
    assert len(done) < 400


def test_every_process_in_a_graph_checks_for_cancellation() -> None:
    graph = {
        "process": {"process_graph": {"one": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}}
    }
    assert execution.run_process_graph(graph["process"]) == 3
    with cancellation_scope(lambda: True), pytest.raises(ExecutionCancelled):
        execution.run_process_graph(graph["process"])  # not wrapped as a 400 or 500


_ROUTES = [
    "/jobs/{job}/results/result.nc",  # the generic file route
    "/jobs/{job}/results/result.geojson",  # its dedicated GeoJSON route
    "/jobs/{job}/results/result.zarr/zarr.json",  # and the Zarr store route
]


def _job_with_result_files(job_id: str, **state: Any) -> Path:
    results = openeo_jobs._JOBS_DIR / job_id / "results"
    (results / "result.zarr").mkdir(parents=True)
    (results / "result.nc").write_bytes(b"partial")
    (results / "result.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (results / "result.zarr" / "zarr.json").write_text("{}")
    openeo_jobs.store_create_job(OpenEOJobRecord(id=job_id, created=utc_now(), **state))
    return results


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "state",
    [
        {"status": OpenEOJobStatus.CANCELED},
        # Cancelled a moment ago: the worker has not noticed yet, and its files are partial.
        {"status": OpenEOJobStatus.RUNNING, "cancel_requested": True},
        {"status": OpenEOJobStatus.RUNNING},
        {"status": OpenEOJobStatus.ERROR},
    ],
    ids=["canceled", "cancel-requested", "running", "error"],
)
def test_result_files_of_an_unfinished_job_are_not_served(
    service: Any, client: TestClient, route: str, state: dict[str, Any]
) -> None:
    results = _job_with_result_files("partial", **state)

    response = client.get(route.format(job="partial"))

    assert response.status_code == 404
    assert "only served for a finished job" in response.json()["detail"]
    assert (results / "result.nc").exists()  # retained for inspection, not served


@pytest.mark.parametrize("route", _ROUTES)
def test_result_files_are_served_once_the_job_finished(service: Any, client: TestClient, route: str) -> None:
    _job_with_result_files("done", status=OpenEOJobStatus.FINISHED)
    assert client.get(route.format(job="done")).status_code == 200


def test_a_new_attempt_removes_files_left_by_an_earlier_attempt(
    service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _job_with_result_files(
        "rerun",
        status=OpenEOJobStatus.QUEUED,
        process={"process_graph": {}},
    )
    monkeypatch.setattr(
        execution,
        "run_process_graph",
        lambda *args, **kwargs: SaveResultEnvelope(_cube(7.0), "GTIFF", {}),
    )

    service._execute("rerun")

    record = openeo_jobs.store_get_job("rerun")
    assert record is not None and record.status == OpenEOJobStatus.FINISHED
    assert (results / "result.tif").exists()
    assert not (results / "result.nc").exists()
    assert not (results / "result.geojson").exists()
    assert not (results / "result.zarr").exists()


@pytest.mark.parametrize("route", _ROUTES)
def test_result_files_of_an_unknown_job_are_not_served(service: Any, client: TestClient, route: str) -> None:
    results = openeo_jobs._JOBS_DIR / "orphan" / "results"
    (results / "result.zarr").mkdir(parents=True)
    (results / "result.nc").write_bytes(b"stray")
    (results / "result.geojson").write_text("{}")
    (results / "result.zarr" / "zarr.json").write_text("{}")
    assert client.get(route.format(job="orphan")).status_code == 404


def test_checks_are_throttled_and_inert_outside_a_job() -> None:
    raise_if_cancelled()  # no scope: no-op
    reads: list[int] = []

    def is_cancelled() -> bool:
        reads.append(1)
        return False

    with cancellation_scope(is_cancelled, interval=60):
        for _ in range(100):
            raise_if_cancelled()
        assert reads == [1]
        raise_if_cancelled(force=True)
        assert reads == [1, 1]


def test_the_dask_check_is_reinstated_if_something_resets_dask_callbacks() -> None:
    from dask.callbacks import Callback

    cancellation.install_dask_cancellation()
    Callback.active.discard(cancellation._dask_callback._callback)  # e.g. a library reset
    done: list[int] = []

    def count(block: np.ndarray) -> np.ndarray:
        done.append(1)
        return block

    # Entering a scope reinstates the check, so the computation is still cancellable.
    with cancellation_scope(lambda: True, interval=0), pytest.raises(ExecutionCancelled):
        da.ones((50, 10), chunks=(1, 10)).map_blocks(count).sum().compute(scheduler="single-threaded")
    assert len(done) < 50
