"""Rolling source ingestion preserves committed history without refetching it."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset
from open_climate_service.ingestions import services
from open_climate_service.ingestions.schemas import CoverageTemporal
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period


class _RollingPlugin(BaseDatasetPlugin):
    def __init__(self) -> None:
        self.available = ["2026-01-01", "2026-01-02"]
        self.queries: list[tuple[str, str]] = []
        self.fetched: list[str] = []

    async def periods(self, start: str, end: str) -> list[str]:
        self.queries.append((start, end))
        return [period for period in self.available if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        assert period_id in self.available, "Retired source periods must not be fetched"
        self.fetched.append(period_id)
        data = xr.DataArray(
            np.full((2, 2), int(period_id[-2:]), dtype="float32"),
            dims=("y", "x"),
            coords={"y": [3.5, 2.5], "x": [1.5, 2.5]},
        )
        return normalize_period(data, variable="precip", period=period_id)


@pytest.fixture
def rolling_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[_RollingPlugin, dict[str, object], Path]:
    plugin = _RollingPlugin()
    dataset: dict[str, object] = {
        "id": "rolling_precip",
        "name": "Rolling precipitation",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {"plugin": "example.RollingPlugin"},
    }
    store_path = tmp_path / "rolling.icechunk"
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")

    return plugin, dataset, store_path


@pytest.mark.parametrize("request_start", ["2026-01-01", "2026-01-04"])
def test_forward_ingestion_preserves_retired_source_history(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], request_start: str
) -> None:
    plugin, dataset, store_path = rolling_store

    def ingest(start: str, end: str) -> services.ArtifactRecord:
        return services.create_artifact(
            dataset=dataset,
            start=start,
            end=end,
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    ingest("2026-01-01", "2026-01-02")
    plugin.available = ["2026-01-03", "2026-01-04"]
    artifact = ingest(request_start, "2026-01-04")

    assert plugin.queries == [("2026-01-01", "2026-01-02"), ("2026-01-03", "2026-01-04")]
    assert plugin.fetched == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert artifact.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-04")
    assert artifact.request_scope.start == request_start
    assert services._load_records()[-1].coverage == artifact.coverage
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0, 3.0, 4.0]

    # A gap in the missing delta must still fail before any data is written.
    repo = services.open_or_create_repo(store_path)
    snapshot = repo.lookup_branch("main")
    plugin.available = ["2026-01-06"]
    with pytest.raises(services.HTTPException, match="contiguous sequence beginning at 2026-01-05"):
        ingest("2026-01-06", "2026-01-06")

    # Extending backwards requires history that this rolling source has retired.
    plugin.available = ["2026-01-03", "2026-01-04"]
    with pytest.raises(services.HTTPException, match="Source cannot materialize"):
        ingest("2025-12-31", "2026-01-04")
    assert repo.lookup_branch("main") == snapshot
    assert services._load_records()[-1].artifact_id == artifact.artifact_id
    assert len(plugin.fetched) == 4


@pytest.mark.parametrize("publish", [False, True])
def test_complete_store_without_record_is_registered_without_source_access(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
    publish: bool,
) -> None:
    plugin, dataset, store_path = rolling_store
    services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    services.ARTIFACTS_INDEX_PATH.unlink()
    plugin.available = []
    plugin.queries.clear()
    plugin.fetched.clear()
    published: list[str] = []

    def publish_record(artifact_id: str) -> services.ArtifactRecord:
        published.append(artifact_id)
        return services._load_records()[-1]

    def unexpected_ingest(**kwargs: object) -> None:
        raise AssertionError("Complete stores must not run the fetch orchestrator")

    monkeypatch.setattr(services, "publish_artifact_record", publish_record)
    monkeypatch.setattr(services, "run_streaming_ingest_sync", unexpected_ingest)
    record = services.create_artifact(
        dataset=dataset,
        start="2026-01-02",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=publish,
    )
    assert record.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-02")
    assert record.request_scope.start == "2026-01-02"
    assert services._load_records()[-1].artifact_id == record.artifact_id
    assert plugin.queries == []
    assert plugin.fetched == []
    assert published == ([record.artifact_id] if publish else [])
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0]


def test_forward_append_progress_counts_only_new_periods(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, dataset, _ = rolling_store
    progress: list[tuple[int | None, int | None]] = []

    def report(done: int | None, total: int | None, message: str | None) -> None:
        progress.append((done, total))

    def ingest(end: str, on_progress: Callable[[int | None, int | None, str | None], None] | None = None) -> None:
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end=end,
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
            on_progress=on_progress,
        )

    ingest("2026-01-02")
    plugin.available = ["2026-01-03"]
    original_fetch = plugin.fetch_period
    at_fetch: list[tuple[int | None, int | None]] = []

    def fetch(period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        at_fetch.append(progress[-1])
        return original_fetch(period_id, bbox, **params)

    monkeypatch.setattr(plugin, "fetch_period", fetch)
    ingest("2026-01-03", report)
    assert at_fetch == [(0, 1)]
    assert progress[-1] == (1, 1)


def test_overwrite_rematerializes_existing_store(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
) -> None:
    plugin, dataset, store_path = rolling_store
    services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    plugin.available.append("2026-01-03")
    plugin.fetched.clear()

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=True,
        publish=False,
    )

    assert plugin.fetched == plugin.available
    assert artifact.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-03")
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0, 3.0]


def test_append_record_failure_restores_existing_store(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, dataset, store_path = rolling_store
    initial = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    plugin.available.append("2026-01-03")

    def fail_record(*args: object, **kwargs: object) -> None:
        raise OSError("record write failed")

    monkeypatch.setattr(services, "_upsert_artifact_record", fail_record)
    with pytest.raises(OSError, match="record write failed"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert services._load_records()[-1].artifact_id == initial.artifact_id
    repo = services.open_or_create_repo(store_path)
    assert not [branch for branch in repo.list_branches() if branch.startswith("ocs-ingest-rollback-")]
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0]
