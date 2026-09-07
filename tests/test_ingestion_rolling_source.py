"""Rolling source ingestion preserves committed history without refetching it."""

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


@pytest.mark.parametrize("request_start", ["2026-01-01", "2026-01-04"])
def test_forward_ingestion_preserves_retired_source_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request_start: str
) -> None:
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
