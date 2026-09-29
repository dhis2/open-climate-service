"""A sync of a pyramid store extends its coarser levels instead of rebuilding them (CLIM-1237).

Each level is reduced from the one above in space only, so a new period's overviews depend on
that period alone. The tests sync through ``create_artifact`` and compare the store with one
built from scratch over the same periods, level by level and array by array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
icechunk = pytest.importorskip("icechunk")
pytest.importorskip("topozarr")
zarr = pytest.importorskip("zarr")

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset  # noqa: E402
from open_climate_service.data_manager.services import downloader  # noqa: E402
from open_climate_service.ingestions import services  # noqa: E402
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period  # noqa: E402

# Odd sizes, so every level trims a trailing row or column.
NY, NX = 101, 75


class _GridPlugin(BaseDatasetPlugin):
    def __init__(self, *, categorical: bool = False) -> None:
        self.available = [f"2026-01-0{day}" for day in (1, 2, 3)]
        self.categorical = categorical

    async def periods(self, start: str, end: str) -> list[str]:
        return [period for period in self.available if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> Any:
        rng = np.random.default_rng(int(period_id[-2:]))
        if self.categorical:
            values = rng.integers(1, 6, size=(NY, NX)).astype("float32")
        else:
            values = rng.normal(5.0, 3.0, size=(NY, NX)).astype("float32")
            values[rng.random((NY, NX)) < 0.1] = np.nan
        data = xr.DataArray(
            values,
            dims=("y", "x"),
            coords={"y": np.linspace(4.0, 2.0, NY), "x": np.linspace(1.0, 3.0, NX)},
        )
        return normalize_period(data, variable="tg", period=period_id)


@pytest.fixture
def pyramid_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    # Small enough that a 101x75 grid gets a three-level pyramid.
    monkeypatch.setattr(downloader, "_PYRAMID_PIXEL_THRESHOLD", 64 * 64)
    monkeypatch.setattr(downloader, "_PYRAMID_TARGET_TILE_SIZE", 24)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    state: dict[str, Any] = {"plugin": _GridPlugin(), "store": tmp_path / "synced.icechunk", "rebuilds": 0}
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: state["plugin"])
    monkeypatch.setattr(downloader, "get_icechunk_path", lambda _: state["store"])

    write = downloader.write_to_icechunk_store

    def counting_write(*args: Any, **kwargs: Any) -> None:
        state["rebuilds"] += 1
        write(*args, **kwargs)

    monkeypatch.setattr(downloader, "write_to_icechunk_store", counting_write)
    state["dataset"] = {
        "id": "tg_daily",
        "name": "Daily temperature",
        "variable": "tg",
        "period_type": "daily",
        "ingestion": {"plugin": "example.GridPlugin"},
    }
    return state


def _ingest(state: dict[str, Any], end: str) -> services.ArtifactRecord:
    return services.create_artifact(
        dataset=state["dataset"],
        start="2026-01-01",
        end=end,
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )


def _root(store_path: Path) -> Any:
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    return zarr.open_group(repo.readonly_session("main").store, mode="r")


def _assert_same_store(synced: Path, rebuilt: Path) -> None:
    a, b = _root(synced), _root(rebuilt)
    assert dict(a.attrs) == dict(b.attrs)
    assert sorted(a.group_keys()) == sorted(b.group_keys())
    assert len(list(a.group_keys())) >= 3, "precondition: a pyramid with several levels"
    for path in ["", *sorted(a.group_keys())]:
        ga, gb = (a[path], b[path]) if path else (a, b)
        assert dict(ga.attrs) == dict(gb.attrs), path
        assert sorted(ga.array_keys()) == sorted(gb.array_keys()), path
        for name in ga.array_keys():
            xa, xb = ga[name], gb[name]
            where = f"{path}/{name}"
            assert xa.shape == xb.shape, where
            assert xa.dtype == xb.dtype, where
            assert np.array_equal(xa[...], xb[...], equal_nan=True), where
            # Chunks, not shards: topozarr shards a small array over its whole time axis, and
            # an append keeps the shard the store was built with.
            assert xa.chunks == xb.chunks, where


def _rebuild(state: dict[str, Any], tmp_path: Path, end: str) -> Path:
    state["store"] = tmp_path / "rebuilt.icechunk"
    _ingest(state, end)
    return state["store"]


@pytest.mark.parametrize("one_period_per_batch", [False, True])
def test_sync_appends_to_every_level_and_matches_a_rebuild(
    pyramid_sync: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_period_per_batch: bool
) -> None:
    state = pyramid_sync
    if one_period_per_batch:
        monkeypatch.setattr(downloader, "_PYRAMID_MAX_REGION_BYTES", 1)
    _ingest(state, "2026-01-03")
    assert state["rebuilds"] == 1, "a first ingest builds the pyramid"
    synced = state["store"]

    state["plugin"].available += ["2026-01-04", "2026-01-05"]
    artifact = _ingest(state, "2026-01-05")

    assert state["rebuilds"] == 1, "the sync must not rebuild the pyramid"
    assert artifact.coverage.temporal.end == "2026-01-05"
    assert not synced.with_name(f"{synced.name}.rebuild").exists()
    with open_icechunk_dataset(synced) as stored:
        assert stored.sizes["t"] == 5
    _assert_same_store(synced, _rebuild(state, tmp_path, "2026-01-05"))


def test_mode_levels_are_resampled_from_native_on_append(pyramid_sync: dict[str, Any], tmp_path: Path) -> None:
    state = pyramid_sync
    state["plugin"] = _GridPlugin(categorical=True)
    state["dataset"]["ingestion"]["resampling"] = "mode"
    _ingest(state, "2026-01-03")
    synced = state["store"]

    state["plugin"].available += ["2026-01-04"]
    _ingest(state, "2026-01-04")

    assert state["rebuilds"] == 1
    _assert_same_store(synced, _rebuild(state, tmp_path, "2026-01-04"))


def test_a_changed_resampling_method_rebuilds(pyramid_sync: dict[str, Any], tmp_path: Path) -> None:
    """Levels reduced with one method must not be extended with another."""
    state = pyramid_sync
    _ingest(state, "2026-01-03")
    synced = state["store"]

    state["dataset"]["ingestion"]["resampling"] = "max"
    state["plugin"].available += ["2026-01-04"]
    _ingest(state, "2026-01-04")

    assert state["rebuilds"] == 2
    _assert_same_store(synced, _rebuild(state, tmp_path, "2026-01-04"))


def test_a_sync_with_nothing_new_leaves_the_store_alone(pyramid_sync: dict[str, Any]) -> None:
    state = pyramid_sync
    _ingest(state, "2026-01-03")
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(state["store"])))
    before = repo.lookup_branch("main")

    assert downloader.append_pyramid_levels(state["store"]) == 0
    assert repo.lookup_branch("main") == before


def test_a_failed_record_write_rolls_back_the_appended_levels(
    pyramid_sync: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = pyramid_sync
    _ingest(state, "2026-01-03")
    state["plugin"].available += ["2026-01-04"]

    def fail_record(*args: object, **kwargs: object) -> None:
        raise OSError("record write failed")

    monkeypatch.setattr(services, "_upsert_artifact_record", fail_record)
    with pytest.raises(OSError, match="record write failed"):
        _ingest(state, "2026-01-04")

    root = _root(state["store"])
    lengths = {path or "root": (root[path] if path else root)["t"].shape[0] for path in ["", *root.group_keys()]}
    assert set(lengths.values()) == {3}, lengths
