"""A streaming ingest publishes its time axis in one chunk.

Streaming appends one period per commit, which leaves the `t` coordinate at one value per
chunk. A map client reads the whole axis to build its time control, so that layout costs one
HTTP request per period: 441 for 1990-2026 monthly data, enough to trip a proxy's rate limit.
"""

from __future__ import annotations

from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from open_climate_service import config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents import services as extent_services
from open_climate_service.ingestions import processes, services
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_DATASET_ID = "monthly_temperature"
_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


class _MonthlyPlugin(BaseDatasetPlugin):
    async def periods(self, start: str, end: str) -> list[str]:
        return [period for period in _MONTHS if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        data = xr.DataArray(
            np.full((2, 2), int(period_id[-2:]), dtype="float32"),
            dims=("y", "x"),
            coords={"y": [3.5, 2.5], "x": [1.5, 2.5]},
        )
        return normalize_period(data, variable="t2m", period=period_id)


def test_streaming_ingest_writes_the_time_axis_as_one_chunk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_path = tmp_path / "monthly.icechunk"
    dataset: dict[str, object] = {
        "id": _DATASET_ID,
        "name": "Monthly temperature",
        "variable": "t2m",
        "period_type": "monthly",
        "ingestion": {"plugin": "example.MonthlyPlugin"},
    }
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: _MonthlyPlugin())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    monkeypatch.setattr(
        registry_datasets, "get_dataset", lambda dataset_id: dataset if dataset_id == _DATASET_ID else None
    )
    monkeypatch.setattr(extent_services, "get_extent_or_404", lambda: {"bbox": [1.0, 2.0, 3.0, 4.0]})
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")

    processes.execute_ingestion(dataset_id=_DATASET_ID, start=_MONTHS[0], end=_MONTHS[-1], publish=False)

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    group = zarr.open_group(repo.readonly_session("main").store, mode="r")
    t = group["t"]
    assert isinstance(t, zarr.Array)
    assert t.chunks == (len(_MONTHS),)
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    assert list(ds.t.values) == list(pd.to_datetime([f"{m}-01" for m in _MONTHS]))
