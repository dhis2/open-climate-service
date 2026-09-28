"""Stored sizes: recorded when a store is written, read (never measured) by the overview page.

The overview used to walk every file of every store on a page load. On an instance with
hundreds of thousands of chunk files, during a heavy job, that took the page from seconds to
many minutes. These tests pin the replacement: each place that writes a store records its size,
driven through its real entry point, and the page reads records and never touches the stores.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_manager.services import downloader
from open_climate_service.features import services as feature_services
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.shared.storage_size import stored_bytes
from open_climate_service.system import templates as landing


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep records, features and thumbnails out of the developer's data, and reset page state."""
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "features")
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(landing, "_measured_sizes", {})
    monkeypatch.setattr(landing, "_size_backfill_running", False)


def _refresh_collection(collection_id: str = "districts") -> ingestion_services.ArtifactRecord:
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    return feature_services.refresh_feature_collection(
        template={"id": collection_id, "name": "Districts", "id_property": "code"},
        features={
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"code": "A"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
            ],
        },
    )


def _forget_sizes() -> None:
    """Make every stored record look like one written before sizes were recorded."""

    def clear(records: list[Any]) -> None:
        for index, record in enumerate(records):
            records[index] = record.model_copy(update={"size_bytes": None})

    ingestion_services._mutate_records(clear)


def _no_walking(*_args: object, **_kwargs: object) -> int:
    raise AssertionError("a store was measured while rendering a page")


# --- measuring ---------------------------------------------------------------------------


def test_a_store_directory_is_measured_by_its_files(tmp_path: Path) -> None:
    store = tmp_path / "store.icechunk"
    (store / "chunks" / "a").mkdir(parents=True)
    (store / "chunks" / "a" / "0").write_bytes(b"x" * 1000)
    (store / "refs").write_bytes(b"y" * 24)

    assert stored_bytes(store) == 1024


def test_a_file_is_measured_by_its_size_and_a_missing_path_counts_as_nothing(tmp_path: Path) -> None:
    parquet = tmp_path / "districts.parquet"
    parquet.write_bytes(b"z" * 300)

    assert stored_bytes(parquet) == 300
    assert stored_bytes(tmp_path / "gone.icechunk") == 0


# --- recorded where stores are written -----------------------------------------------------


def test_a_feature_refresh_records_the_size_of_the_file_it_wrote() -> None:
    record = _refresh_collection()

    assert record.path is not None
    assert record.size_bytes == stored_bytes(record.path) > 0


def test_a_streaming_ingest_records_the_size_of_the_published_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driven through `create_artifact`, as the thumbnail tests are, so the wiring is covered."""
    store_path = tmp_path / "sized.icechunk"
    dataset: dict[str, object] = {
        "id": "sized",
        "name": "Sized dataset",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {"plugin": "example.Plugin", "params": {}},
    }
    cube = xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((3, 2, 2), dtype="float32"))},
        coords={"t": pd.date_range("2026-01-01", periods=3, freq="D"), "y": [1.5, 0.5], "x": [10.5, 11.5]},
    )

    def fake_sync(**_kwargs: object) -> object:
        downloader.write_to_icechunk_store(cube, store_path, commit_message="test")
        return SimpleNamespace(periods_written=3)

    class _Plugin:
        time_dim = "t"

        async def periods(self, start: str, end: str) -> list[str]:
            return ["2026-01-01", "2026-01-02", "2026-01-03"]

    recorded: list[Any] = []
    monkeypatch.setattr(ingestion_services, "_load_streaming_plugin", lambda path, *, params: _Plugin())
    monkeypatch.setattr(ingestion_services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(ingestion_services, "run_streaming_ingest_sync", fake_sync)

    def keep(record: Any, **_: object) -> Any:
        recorded.append(record)
        return record

    monkeypatch.setattr(ingestion_services, "_upsert_artifact_record", keep)
    monkeypatch.setattr(
        ingestion_services,
        "get_data_coverage_for_paths",
        lambda dataset_arg, **_: {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-03"},
                "spatial": {"xmin": 10.0, "ymin": 0.0, "xmax": 12.0, "ymax": 2.0},
            }
        },
    )

    ingestion_services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[10.0, 0.0, 12.0, 2.0],
        country_code=None,
        overwrite=True,
        publish=False,
    )

    assert len(recorded) == 1
    assert recorded[0].size_bytes == stored_bytes(store_path) > 0


def test_an_openeo_publish_records_the_size_of_the_store_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_registry.services import datasets as registry
    from open_climate_service.openeo import jobs

    configs = tmp_path / "datasets"
    configs.mkdir()
    monkeypatch.setattr(registry, "CONFIGS_DIR", configs)
    registry.reset_template_caches()
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", tmp_path / "downloads")
    (tmp_path / "downloads").mkdir()
    cube = xr.Dataset(
        {"tp": (("t", "y", "x"), np.full((3, 2, 2), 2.0, dtype="float32"), {"units": "mm/d"})},
        coords={"t": pd.date_range("2026-01-01", periods=3, freq="D"), "y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).rio.write_crs("EPSG:4326")

    jobs._write_managed_zarr(cube, {"dataset_id": "published_sized", "variable": "tp"})

    records = [r for r in ingestion_services._load_records() if r.dataset_id == "published_sized"]
    assert len(records) == 1 and records[0].path is not None
    assert records[0].size_bytes == stored_bytes(records[0].path) > 0


# --- read by the overview page ---------------------------------------------------------------


def test_the_overview_reads_recorded_sizes_and_never_walks_a_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _refresh_collection()
    monkeypatch.setattr(landing, "stored_bytes", _no_walking)

    page = client.get("/", headers={"Accept": "text/html"})

    assert page.status_code == 200
    assert landing._format_bytes(record.size_bytes or 0) in page.text
    assert landing._stored_bytes() == (record.size_bytes, True)


def test_each_store_is_counted_once_from_its_newest_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successive syncs append to one store, each leaving a record; only the newest size is current."""

    def record(path: str, day: str, size: int) -> SimpleNamespace:
        return SimpleNamespace(path=path, created_at=pd.Timestamp(day, tz="UTC"), size_bytes=size)

    older = record("/data/chirps.icechunk", "2026-01-01", 1000)
    newer = record("/data/chirps.icechunk", "2026-02-01", 1500)
    other = record("/data/era5.icechunk", "2026-01-15", 200)
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[newer, other, older]))
    monkeypatch.setattr(landing, "stored_bytes", _no_walking)

    assert landing._stored_bytes() == (1700, True)


def test_records_without_a_size_are_measured_once_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _refresh_collection()
    _forget_sizes()
    started: list[list[str]] = []
    monkeypatch.setattr(landing, "_start_size_backfill", lambda paths: started.append(paths))

    assert landing._stored_bytes() == (0, False)
    assert started == [[record.path]]

    landing._backfill_sizes(started[0])

    stored = [r for r in ingestion_services._load_records() if r.dataset_id == "districts"]
    assert stored[0].size_bytes == record.size_bytes
    assert landing._stored_bytes() == (record.size_bytes, True)
    assert landing._format_stored_size(1500, False) == "1.5 KB+"


def test_a_read_only_instance_measures_in_memory_without_rewriting_records(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _refresh_collection()
    _forget_sizes()
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)
    assert record.path is not None

    landing._backfill_sizes([record.path])

    stored = [r for r in ingestion_services._load_records() if r.dataset_id == "districts"]
    assert stored[0].size_bytes is None
    monkeypatch.setattr(landing, "stored_bytes", _no_walking)
    assert landing._stored_bytes() == (record.size_bytes, True)
