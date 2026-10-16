from __future__ import annotations

from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pytest
import xarray as xr

import open_climate_service.streaming.orchestrator as streaming_orchestrator
from open_climate_service.streaming.orchestrator import _grid_spec_from_dataset, run_streaming_ingest_sync


def test_grid_spec_from_dataset_raises_clear_error_for_missing_spatial_dims() -> None:
    ds = xr.Dataset({"v": (("t", "lat", "lon"), np.zeros((1, 1, 1), dtype=np.float32))})
    with pytest.raises(RuntimeError, match="missing the expected spatial dimensions 'y' and 'x'"):
        _grid_spec_from_dataset(ds, time_dim="t", x_dim="x", y_dim="y", crs="EPSG:4326")


class _FakePlugin:
    max_concurrency = 2
    commit_batch_size = 2

    async def periods(self, start: str, end: str) -> list[str]:
        _ = start, end
        return ["2026-01-01", "2026-01-02", "2026-01-03"]

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        _ = bbox, params
        value = float(period_id[-2:])
        return xr.Dataset(
            {"precip": (("t", "y", "x"), np.array([[[value]]], dtype=np.float32))},
            coords={"t": [np.datetime64(period_id, "D")], "y": [0.0], "x": [1.0]},
        )


class _ShapeMismatchPlugin(_FakePlugin):
    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        _ = bbox, params
        if period_id == "2026-01-02":
            return xr.Dataset(
                {"precip": (("t", "y", "x"), np.array([[[1.0], [2.0]]], dtype=np.float32))},
                coords={"t": [np.datetime64(period_id, "D")], "y": [0.0, 1.0], "x": [1.0]},
            )
        return await super().fetch_period(period_id, bbox, **params)


class _CustomTimeDimPlugin(_FakePlugin):
    # Declare a non-default time dimension via the class attribute to exercise the
    # custom-dim code path; production plugins standardise on "t" but this test must
    # verify the orchestrator honours an overridden `time_dim`.
    time_dim = "valid_time"

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        _ = bbox, params
        value = float(period_id[-2:])
        return xr.Dataset(
            {"precip": (("valid_time", "y", "x"), np.array([[[value]]], dtype=np.float32))},
            coords={"valid_time": [np.datetime64(period_id, "D")], "y": [0.0], "x": [1.0]},
        )


class _NoProbePlugin:
    """Minimal plugin with no probe(), no concurrency attributes, and a *sync*
    fetch_period.

    Exercises the orchestrator defaults (max_concurrency/commit_batch_size = 1,
    dims t/x/y), GridSpec inference from the first fetched period, and the
    sync-fetch bridge (the orchestrator runs a blocking fetch_period in a thread).
    """

    async def periods(self, start: str, end: str) -> list[str]:
        _ = start, end
        return ["2026-01-01", "2026-01-02"]

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        _ = bbox, params
        value = float(period_id[-2:])
        return xr.Dataset(
            {"precip": (("t", "y", "x"), np.array([[[value]]], dtype=np.float32))},
            coords={"t": [np.datetime64(period_id, "D")], "y": [0.0], "x": [1.0]},
        )


class _FakeSession:
    def __init__(self, store: str) -> None:
        self.store = store
        self.messages: list[str] = []

    def commit(self, message: str) -> None:
        self.messages.append(message)


class _FakeRepo:
    def __init__(self, store: str) -> None:
        self.store = store
        self.sessions: list[_FakeSession] = []

    def writable_session(self, branch: str) -> _FakeSession:
        assert branch == "main"
        session = _FakeSession(self.store)
        self.sessions.append(session)
        return session


def _read_committed_periods_from_zarr(store_path: Path, period_type: str, *, time_dim: str = "t") -> set[str]:
    _ = period_type
    if not store_path.exists():
        return set()
    ds = xr.open_zarr(store_path, consolidated=None)
    try:
        if time_dim not in ds.coords:
            return set()
        return {str(item)[:10] for item in ds[time_dim].values}
    finally:
        ds.close()


def test_orchestrator_uses_store_state_as_resume_truth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))
    progress_updates: list[tuple[int | None, int | None, str | None]] = []
    cursor_saves: list[dict[str, Any]] = []

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    result = run_streaming_ingest_sync(
        plugin=_FakePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
        on_progress=lambda done, total, message: progress_updates.append((done, total, message)),
        save_cursor=lambda cursor: cursor_saves.append(cursor),
    )

    assert result.periods_written == 3
    assert store_path.exists()
    first = xr.open_zarr(store_path, consolidated=None)
    try:
        assert first["precip"].values[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    finally:
        first.close()
    import zarr

    root = zarr.open_group(store_path, mode="r")
    assert root.attrs["proj:code"] == "EPSG:4326"
    # Array order, (y, x) — see tests/test_geozarr_grid_metadata.py for why.
    assert root.attrs["spatial:dimensions"] == ["y", "x"]
    assert root.attrs["spatial:shape"] == [1, 1]
    # A 1x1 grid has no derivable cell size, so no affine is claimed for it.
    assert "spatial:transform" not in root.attrs

    run_streaming_ingest_sync(
        plugin=_FakePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
        on_progress=lambda done, total, message: progress_updates.append((done, total, message)),
        save_cursor=lambda cursor: cursor_saves.append(cursor),
    )

    second = xr.open_zarr(store_path, consolidated=None)
    try:
        assert second["precip"].values[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    finally:
        second.close()

    assert any(update[2] == "Wrote 2026-01-03" for update in progress_updates)
    assert cursor_saves[-1] == {"last_committed": "2026-01-03"}


class _TaggedPlugin(_FakePlugin):
    """Tags each fetched dataset so the test can count when the orchestrator closes it."""

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = await super().fetch_period(period_id, bbox, **params)
        ds.attrs["_test_tag"] = "plugin"
        return ds


def test_orchestrator_closes_each_fetched_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each period's dataset must be closed after it is written, so long backfills
    don't leak the open_rasterio / open_dataset handles backing it."""
    store_path = tmp_path / "close-store.zarr"
    repo = _FakeRepo(str(store_path))
    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    real_close = xr.Dataset.close
    plugin_closes = {"n": 0}

    def counting_close(self: xr.Dataset) -> None:
        if self.attrs.get("_test_tag") == "plugin":
            plugin_closes["n"] += 1
        real_close(self)

    monkeypatch.setattr(xr.Dataset, "close", counting_close)

    result = run_streaming_ingest_sync(
        plugin=_TaggedPlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 3
    assert plugin_closes["n"] == 3  # exactly one close per written period


def test_orchestrator_infers_grid_when_plugin_has_no_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A plugin with no probe() and no concurrency attrs still ingests: the grid,
    CRS, and dtype are inferred from the first fetched period and the defaults apply."""
    store_path = tmp_path / "noprobe-store.zarr"
    repo = _FakeRepo(str(store_path))
    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    result = run_streaming_ingest_sync(
        plugin=_NoProbePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-02",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 2
    ds = xr.open_zarr(store_path, consolidated=None)
    try:
        assert ds["precip"].values[:, 0, 0].tolist() == [1.0, 2.0]
    finally:
        ds.close()
    import zarr

    root = zarr.open_group(store_path, mode="r")
    assert root.attrs["proj:code"] == "EPSG:4326"


class _ProjectedNoProbePlugin(_NoProbePlugin):
    """No probe(), but declares a projected CRS via the `crs` class attribute.

    Emits UTM33 metres rather than the base fixture's toy (1, 0). A declared projected CRS is
    checked against the coordinates it claims to describe, so a store one metre from its own
    origin would — correctly — be treated as degrees mislabelled as UTM.
    """

    crs = 32633

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        _ = bbox, params
        value = float(period_id[-2:])
        return xr.Dataset(
            {"precip": (("t", "y", "x"), np.array([[[value]]], dtype=np.float32))},
            coords={"t": [np.datetime64(period_id, "D")], "y": [6_650_000.0], "x": [500_000.0]},
        )


def test_orchestrator_uses_plugin_crs_attr_as_inference_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plugin can declare its CRS via the `crs` attribute instead of writing a
    probe() purely to set the projection: inference uses it as the fallback when the
    fetched data carries no CRS of its own."""
    store_path = tmp_path / "projected-store.zarr"
    repo = _FakeRepo(str(store_path))
    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    run_streaming_ingest_sync(
        plugin=_ProjectedNoProbePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-02",
        store_path=store_path,
        period_type="daily",
    )

    import zarr

    root = zarr.open_group(store_path, mode="r")
    assert root.attrs["proj:code"] == "EPSG:32633"


class _PlaceholderAttrsPlugin(_FakePlugin):
    """Emits the kind of generic/placeholder CF attrs GRIB & unit-conversions leave behind."""

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = await super().fetch_period(period_id, bbox, **params)
        ds["precip"].attrs.update({"units": "mm", "standard_name": "unknown"})
        return ds


def test_orchestrator_stamps_cf_attrs_from_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The template is authoritative: its CF fields overwrite placeholder source attrs (#280).

    The source variable arrives with a dimensionally-generic ``units='mm'`` and a placeholder
    ``standard_name='unknown'``; the template declares the rate ``mm/d`` and the real
    standard name, which must win on disk so xclim-style unit checks pass.
    """
    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    run_streaming_ingest_sync(
        plugin=_PlaceholderAttrsPlugin(),
        params={},
        dataset={
            "units": "mm/d",
            "standard_name": "lwe_precipitation_rate",
            "cell_methods": "time: mean",
        },
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    written = xr.open_zarr(store_path, consolidated=None)
    try:
        attrs = written["precip"].attrs
        assert attrs.get("units") == "mm/d"  # template rate overrides source 'mm'
        assert attrs.get("standard_name") == "lwe_precipitation_rate"  # overrides 'unknown'
        assert attrs.get("cell_methods") == "time: mean"
    finally:
        written.close()


def test_orchestrator_refuses_destructive_first_write_when_existing_store_is_not_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "streaming-store.zarr"
    store_path.mkdir()
    repo = _FakeRepo(str(store_path))

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator, "read_committed_period_ids", lambda path, period_type, time_dim="t": set()
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: False)

    with pytest.raises(RuntimeError, match="committed periods could not be read"):
        run_streaming_ingest_sync(
            plugin=_FakePlugin(),
            params={},
            bbox=[0.0, 0.0, 1.0, 1.0],
            start="2026-01-01",
            end="2026-01-03",
            store_path=store_path,
            period_type="daily",
        )


def test_orchestrator_normalizes_invalid_plugin_batching_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _InvalidPlugin(_FakePlugin):
        max_concurrency = 0
        commit_batch_size = 0

    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))
    cursor_saves: list[dict[str, Any]] = []

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    result = run_streaming_ingest_sync(
        plugin=_InvalidPlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
        save_cursor=lambda cursor: cursor_saves.append(cursor),
    )

    assert result.periods_written == 3
    assert cursor_saves[-1] == {"last_committed": "2026-01-03"}


def test_orchestrator_does_not_skip_uncommitted_gaps_before_cursor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))

    existing = xr.Dataset(
        {"precip": (("t", "y", "x"), np.array([[[2.0]]], dtype=np.float32))},
        coords={"t": [np.datetime64("2026-01-02", "D")], "y": [0.0], "x": [1.0]},
    )
    existing.to_zarr(store_path, mode="w", zarr_format=3)

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": {"2026-01-02"},
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: False)

    result = run_streaming_ingest_sync(
        plugin=_FakePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 2
    ds = xr.open_zarr(store_path, consolidated=None)
    try:
        assert {str(item)[:10] for item in ds["t"].values} == {"2026-01-01", "2026-01-02", "2026-01-03"}
    finally:
        ds.close()


def test_orchestrator_rejects_spatial_shape_changes_across_appends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator, "read_committed_period_ids", lambda path, period_type, time_dim="t": set()
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    with pytest.raises(RuntimeError, match="has spatial shape"):
        run_streaming_ingest_sync(
            plugin=_ShapeMismatchPlugin(),
            params={},
            bbox=[0.0, 0.0, 1.0, 1.0],
            start="2026-01-01",
            end="2026-01-03",
            store_path=store_path,
            period_type="daily",
        )


def test_orchestrator_writes_and_reads_through_real_icechunk_repo(tmp_path: Path) -> None:
    store_path = tmp_path / "streaming-store.icechunk"

    result = run_streaming_ingest_sync(
        plugin=_FakePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 3
    assert store_path.exists()

    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store)
    try:
        assert ds["precip"].values[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
        assert {str(item)[:10] for item in ds["t"].values} == {"2026-01-01", "2026-01-02", "2026-01-03"}
    finally:
        ds.close()

    rerun = run_streaming_ingest_sync(
        plugin=_FakePlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    assert rerun.periods_written == 0


def test_orchestrator_resume_supports_custom_time_dimension(tmp_path: Path) -> None:
    store_path = tmp_path / "streaming-store-custom-time.icechunk"

    first = run_streaming_ingest_sync(
        plugin=_CustomTimeDimPlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )
    assert first.periods_written == 3

    rerun = run_streaming_ingest_sync(
        plugin=_CustomTimeDimPlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )
    assert rerun.periods_written == 0


class _ForeignAttrsPlugin(_FakePlugin):
    """A plugin that sanitizes nothing, like every plugin that has not met this problem yet."""

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = await super().fetch_period(period_id, bbox, **params)
        # Exactly the shape dynamical.org's GEFS store publishes on its coordinates.
        ds["x"].attrs["statistics_approximate"] = {"min": -180.0, "max": 179.75}
        ds["precip"].attrs["GRIB_extra"] = {"nested": "mapping"}
        ds.attrs["provenance"] = {"source": "somewhere"}
        ds["precip"].attrs["long_name"] = "kept"
        return ds


def test_orchestrator_drops_attrs_no_netcdf_export_could_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The guard belongs at the boundary, not in the plugins that happened to notice.

    Zarr stores a nested mapping happily, so the ingest looks clean and every later NetCDF export
    of the store fails with a 500 instead. This plugin does nothing about it, and must still
    produce an exportable store.
    """
    store_path = tmp_path / "streaming-store.zarr"
    repo = _FakeRepo(str(store_path))

    monkeypatch.setattr(streaming_orchestrator, "open_or_create_repo", lambda path: repo)
    monkeypatch.setattr(
        streaming_orchestrator,
        "read_committed_period_ids",
        lambda path, period_type, time_dim="t": _read_committed_periods_from_zarr(path, period_type, time_dim=time_dim),
    )
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    run_streaming_ingest_sync(
        plugin=_ForeignAttrsPlugin(),
        params={},
        dataset={"id": "foreign-attrs"},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    written = xr.open_zarr(store_path, consolidated=None)
    try:
        assert "statistics_approximate" not in written["x"].attrs
        assert "GRIB_extra" not in written["precip"].attrs
        assert "provenance" not in written.attrs
        assert written["precip"].attrs.get("long_name") == "kept"
    finally:
        written.close()
