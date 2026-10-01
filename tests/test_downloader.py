import tempfile
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from open_climate_service.data_accessor.services.accessor import (
    _coverage_from_dataset,
    open_icechunk_dataset,
    open_zarr_dataset,
)
from open_climate_service.data_manager.services import downloader
from open_climate_service.ingestions import services as ingestion_services


def test_resolve_download_dir_uses_data_dir_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nextent:\n  id: test\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert downloader._resolve_download_dir() == tmp_path / "data" / "rasters"


def test_resolve_download_dir_uses_xdg_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as xdg:
        monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
        assert downloader._resolve_download_dir() == Path(xdg) / "climate-service" / "rasters"


def test_resolve_artifacts_dir_uses_data_dir_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nextent:\n  id: test\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert ingestion_services._resolve_artifacts_dir() == tmp_path / "data" / "artifacts"


def test_resolve_artifacts_dir_uses_xdg_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as xdg:
        monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
        assert ingestion_services._resolve_artifacts_dir() == Path(xdg) / "climate-service" / "artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"pop_total": (["t", "lat", "lon"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "t": pd.date_range("2020-01-01", periods=2, freq="YS"),
            "lat": [10.0, 9.0, 8.0],
            "lon": [30.0, 31.0, 32.0],
        },
    )


# ---------------------------------------------------------------------------
# open_zarr_dataset
# ---------------------------------------------------------------------------


def test_open_zarr_dataset_flat(tmp_path: Path) -> None:
    """Flat zarr store is opened directly and exposes its data variables."""
    ds = _make_dataset()
    zarr_path = tmp_path / "flat.zarr"
    ds.to_zarr(str(zarr_path), mode="w")

    result = open_zarr_dataset(str(zarr_path))
    try:
        assert "pop_total" in result.data_vars
    finally:
        result.close()


def test_open_zarr_dataset_pyramid_falls_back_to_level_0(tmp_path: Path) -> None:
    """Pyramid zarr store (no root data vars) opens level 0 automatically."""
    ds = _make_dataset()
    zarr_path = tmp_path / "pyramid.zarr"
    ds.to_zarr(str(zarr_path) + "/0", mode="w")
    zarr.open_group(str(zarr_path), mode="a", zarr_format=3)

    result = open_zarr_dataset(str(zarr_path))
    try:
        assert "pop_total" in result.data_vars
        assert result.sizes["t"] == 2
    finally:
        result.close()


def test_open_zarr_dataset_pyramid_with_root_time_still_opens_level_0(tmp_path: Path) -> None:
    """Root time coord in a pyramid store does not prevent level 0 fallback."""
    ds = _make_dataset()
    zarr_path = tmp_path / "pyramid-with-time.zarr"
    ds.to_zarr(str(zarr_path) + "/0", mode="w")
    ds[["t"]].to_zarr(str(zarr_path), mode="a")

    result = open_zarr_dataset(str(zarr_path))
    try:
        assert "pop_total" in result.data_vars
        assert result.sizes["t"] == 2
    finally:
        result.close()


# ---------------------------------------------------------------------------
# open_icechunk_dataset
# ---------------------------------------------------------------------------


def test_open_icechunk_dataset_falls_back_to_level_0_when_root_has_no_data_vars(tmp_path: Path) -> None:
    """Icechunk pyramid store: level 0 is opened when root has no data variables."""
    store_path = tmp_path / "pyramid.icechunk"
    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = _make_dataset()
    ds.to_zarr(session.store, group="0", mode="w", zarr_format=3)
    session.commit("seed level 0")

    result = open_icechunk_dataset(store_path)
    try:
        assert "pop_total" in result.data_vars
        assert result.sizes["t"] == 2
    finally:
        result.close()


def test_open_icechunk_dataset_with_root_time_still_opens_level_0(tmp_path: Path) -> None:
    """Root-level time coord does not confuse the Icechunk fallback."""
    store_path = tmp_path / "pyramid.icechunk"
    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = _make_dataset()
    ds.to_zarr(session.store, group="0", mode="w", zarr_format=3)
    ds[["t"]].to_zarr(session.store, mode="a", zarr_format=3)
    session.commit("seed pyramid root time and level 0")

    result = open_icechunk_dataset(store_path)
    try:
        assert "pop_total" in result.data_vars
        assert result.sizes["t"] == 2
    finally:
        result.close()


def _daily_store(path: Path, *, days: int = 20, encoding: dict | None = None, group: str | None = None) -> xr.Dataset:
    """A daily float32 cube on a 16x16 grid, stored one day per chunk unless told otherwise."""
    ds = xr.Dataset(
        {"tg": (["t", "y", "x"], np.arange(days * 256, dtype="float32").reshape(days, 16, 16))},
        coords={"t": pd.date_range("2020-01-01", periods=days, freq="D"), "y": np.arange(16.0), "x": np.arange(16.0)},
    )
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    chunking = {"tg": encoding or {"chunks": (1, 8, 8)}}
    ds.to_zarr(session.store, group=group, mode="w", zarr_format=3, encoding=chunking)
    session.commit("seed")
    return ds


def _dask_chunks(ds: xr.Dataset) -> tuple[int, int, int]:
    chunks = ds["tg"].chunks
    assert chunks is not None
    return (chunks[0][0], chunks[1][0], chunks[2][0])


def test_open_icechunk_dataset_merges_stored_chunks_along_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One stored day per chunk becomes several per dask chunk; space keeps the stored chunks."""
    from open_climate_service.data_accessor.services import accessor

    # 5 stored chunks of 8x8 float32 (256 bytes each) per dask chunk
    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 5 * 256)
    written = _daily_store(tmp_path / "daily.icechunk")
    result = open_icechunk_dataset(tmp_path / "daily.icechunk")
    try:
        assert _dask_chunks(result) == (5, 8, 8)
        np.testing.assert_array_equal(result["tg"].values, written["tg"].values)
    finally:
        result.close()


def test_open_icechunk_dataset_keeps_whole_shards_along_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sharded store's time chunk is a multiple of the shard, sized from its inner chunks."""
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 5 * 256)
    _daily_store(tmp_path / "sharded.icechunk", encoding={"chunks": (1, 8, 8), "shards": (2, 16, 16)})
    result = open_icechunk_dataset(tmp_path / "sharded.icechunk")
    try:
        # 5 steps fit the target; rounded down to whole two-day shards
        assert _dask_chunks(result) == (4, 8, 8)
    finally:
        result.close()


def test_open_icechunk_dataset_merges_a_pyramid_level_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 5 * 256)
    _daily_store(tmp_path / "pyramid.icechunk", group="0")
    result = open_icechunk_dataset(tmp_path / "pyramid.icechunk")
    try:
        assert _dask_chunks(result) == (5, 8, 8)
    finally:
        result.close()


def test_open_icechunk_dataset_caps_the_time_chunk_at_the_series(tmp_path: Path) -> None:
    """The default target (16 MiB) far exceeds this tiny store, so one chunk holds every day."""
    _daily_store(tmp_path / "short.icechunk", days=3)
    result = open_icechunk_dataset(tmp_path / "short.icechunk")
    try:
        assert _dask_chunks(result) == (3, 8, 8)
    finally:
        result.close()


def test_open_icechunk_dataset_leaves_chunks_that_reach_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 256)
    _daily_store(tmp_path / "large.icechunk")
    result = open_icechunk_dataset(tmp_path / "large.icechunk")
    try:
        assert _dask_chunks(result) == (1, 8, 8)
    finally:
        result.close()


def test_open_icechunk_dataset_without_time_keeps_stored_chunks(tmp_path: Path) -> None:
    """A store with no time axis (an elevation layer, say) still opens dask-backed, as stored."""
    ds = xr.Dataset(
        {"elev": (["y", "x"], np.ones((16, 16), dtype="float32"))}, coords={"y": np.arange(16.0), "x": np.arange(16.0)}
    )
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "static.icechunk")))
    session = repo.writable_session("main")
    ds.to_zarr(session.store, mode="w", zarr_format=3, encoding={"elev": {"chunks": (8, 8)}})
    session.commit("seed")
    result = open_icechunk_dataset(tmp_path / "static.icechunk")
    try:
        assert result["elev"].chunks == ((8, 8), (8, 8))
    finally:
        result.close()


# ---------------------------------------------------------------------------
# _write_root_time_coordinate
# ---------------------------------------------------------------------------


def test_write_root_time_coordinate_skips_when_time_dimension_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "no-time.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs["sentinel"] = "kept"
    ds = xr.Dataset(
        {"pop_total": (["y", "x"], np.ones((2, 2), dtype="float32"))},
        coords={"y": [1.0, 2.0], "x": [3.0, 4.0]},
    )

    downloader._write_root_time_coordinate(zarr_path, ds, time_dim="t")

    root = zarr.open_group(str(zarr_path), mode="r", zarr_format=3)
    assert "t" not in root
    assert root.attrs["sentinel"] == "kept"


def test_write_root_time_coordinate_skips_when_time_coordinate_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "missing-time-coord.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs["sentinel"] = "kept"
    ds = xr.Dataset(
        {"pop_total": (["t", "y", "x"], np.ones((2, 2, 2), dtype="float32"))},
        coords={"y": [1.0, 2.0], "x": [3.0, 4.0]},
    )

    downloader._write_root_time_coordinate(zarr_path, ds, time_dim="t")

    root = zarr.open_group(str(zarr_path), mode="r", zarr_format=3)
    assert "t" not in root
    assert root.attrs["sentinel"] == "kept"


def test_write_root_time_coordinate_replaces_existing_root_time(tmp_path: Path) -> None:
    zarr_path = tmp_path / "existing-time.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs["sentinel"] = "kept"
    first = xr.Dataset(coords={"t": pd.date_range("2020-01-01", periods=3, freq="D")})
    second = xr.Dataset(coords={"t": pd.date_range("2020-02-01", periods=2, freq="D")})

    downloader._write_root_time_coordinate(zarr_path, first, time_dim="t")
    downloader._write_root_time_coordinate(zarr_path, second, time_dim="t")

    root_time = zarr.open_array(str(zarr_path / "t"), mode="r", zarr_format=3)
    assert root_time.shape == (2,)
    assert root_time.chunks == (2,)
    root = zarr.open_group(str(zarr_path), mode="r", zarr_format=3)
    assert root.attrs["sentinel"] == "kept"


# ---------------------------------------------------------------------------
# _coverage_from_dataset — WGS84 reprojection
# ---------------------------------------------------------------------------


def test_coverage_from_dataset_populates_spatial_wgs84_for_projected_crs() -> None:
    # Small UTM33N (EPSG:25833) bounding box covering south-central Norway.
    x = np.array([100_000.0, 200_000.0])  # easting metres
    y = np.array([6_500_000.0, 6_600_000.0])  # northing metres
    times = pd.date_range("2020-01-01", periods=1, freq="D")
    data = np.ones((1, len(y), len(x)))
    ds = xr.Dataset(
        {"temperature": (["time", "y", "x"], data)},
        coords={"time": times, "y": y, "x": x},
    )

    result = _coverage_from_dataset(ds=ds, period_type="daily", native_crs="EPSG:25833")

    wgs84 = result["coverage"]["spatial_wgs84"]
    assert wgs84 is not None
    # Reprojected bounds must be in WGS84 degree range.
    assert -180 <= wgs84["xmin"] <= 180
    assert -180 <= wgs84["xmax"] <= 180
    assert -90 <= wgs84["ymin"] <= 90
    assert -90 <= wgs84["ymax"] <= 90
    # Rough sanity check: UTM33N easting ~100–200 km, northing ~6500–6600 km.
    assert 7.0 < wgs84["xmin"] < 10.0
    assert 8.0 < wgs84["xmax"] < 12.0
    assert 58.0 < wgs84["ymin"] < 62.0
    assert 58.0 < wgs84["ymax"] < 62.0


def test_coverage_from_dataset_leaves_spatial_wgs84_none_for_wgs84() -> None:
    x = np.array([-10.0, -9.0])
    y = np.array([7.0, 8.0])
    times = pd.date_range("2020-01-01", periods=1, freq="D")
    data = np.ones((1, len(y), len(x)))
    ds = xr.Dataset(
        {"temperature": (["time", "y", "x"], data)},
        coords={"time": times, "y": y, "x": x},
    )

    result = _coverage_from_dataset(ds=ds, period_type="daily", native_crs="EPSG:4326")

    assert result["coverage"]["spatial_wgs84"] is None


def test_coverage_from_dataset_handles_non_temporal_ordinal_store() -> None:
    # A day-of-year climatology has a `dayofyear` axis, not a datetime one. Coverage
    # must report spatial extent with null temporal bounds rather than raising
    # "Unable to find time dimension" (which 500'd the publish step).
    x = np.array([-13.0, -11.0])
    y = np.array([7.0, 9.0])
    data = np.ones((366, len(y), len(x)), dtype="float32")
    ds = xr.Dataset(
        {"t2m": (["dayofyear", "y", "x"], data)},
        coords={"dayofyear": list(range(1, 367)), "y": y, "x": x},
    )

    result = _coverage_from_dataset(ds=ds, period_type="climatology", native_crs="EPSG:4326")

    assert result["has_data"] is True
    assert result["coverage"]["temporal"] == {"start": None, "end": None}
    assert result["coverage"]["spatial"] == {"xmin": -13.0, "ymin": 7.0, "xmax": -11.0, "ymax": 9.0}


def test_write_to_icechunk_store_preserves_native_crs_ignoring_instance_crs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Data is stored in its native CRS; the instance config CRS is never used.

    Regression for ERA5-Land (native WGS84) being mislabeled with the instance
    UTM CRS, which put degree coordinates off-map when rendered as metres.
    """
    from open_climate_service import config as api_config

    # Force a non-WGS84 instance CRS — the writer must ignore it entirely.
    monkeypatch.setattr(api_config, "get_crs", lambda: "EPSG:32633")

    # WGS84 dataset carrying its native CRS as the streaming store writes it.
    y = np.linspace(57.0, 72.5, 8)
    x = np.linspace(3.0, 32.0, 10)
    data = np.random.default_rng(0).random((1, len(y), len(x)), dtype="float32")
    ds = xr.Dataset(
        {"t2m": (("t", "y", "x"), data)},
        coords={"t": [np.datetime64("2024-01-01")], "y": y, "x": x},
    )
    ds.attrs["proj:code"] = "EPSG:4326"

    store = tmp_path / "native_crs.icechunk"
    downloader.write_to_icechunk_store(ds, store, commit_message="test")

    out = open_icechunk_dataset(str(store))
    assert out.attrs.get("proj:code") == "EPSG:4326"


def test_ensure_time_coordinate_chunking_rechunks_appended_store(tmp_path: Path) -> None:
    # Build a daily store the way streaming ingest does: append one timestep at a time,
    # which leaves the `t` coordinate chunked at size 1 (one chunk per day).
    store_path = tmp_path / "daily.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(store_path)))
    times = pd.date_range("2020-01-01", periods=40, freq="D")

    session = repo.writable_session("main")
    xr.Dataset(
        {"v": (("t", "y", "x"), np.zeros((1, 2, 2), "float32"))},
        coords={"t": times[:1], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    ).to_zarr(session.store, mode="w", zarr_format=3, encoding={"t": {"chunks": (1,)}})
    session.commit("seed")
    for i in range(1, len(times)):
        s = repo.writable_session("main")
        xr.Dataset(
            {"v": (("t", "y", "x"), np.full((1, 2, 2), i, "float32"))},
            coords={"t": times[i : i + 1], "y": [1.0, 2.0], "x": [1.0, 2.0]},
        ).to_zarr(s.store, append_dim="t", zarr_format=3)
        s.commit(f"append {i}")

    def arr(group: zarr.Group, name: str) -> zarr.Array:
        node = group[name]
        assert isinstance(node, zarr.Array)
        return node

    before = zarr.open_group(repo.readonly_session("main").store, mode="r")
    assert arr(before, "t").chunks == (1,)  # precondition: one chunk per timestep
    data_chunks = arr(before, "v").chunks

    changed = downloader.ensure_time_coordinate_chunking(store_path, "t")
    assert changed is True

    after = zarr.open_group(repo.readonly_session("main").store, mode="r")
    assert arr(after, "t").chunks == (40,)  # 40 < cap -> a single coordinate chunk
    assert arr(after, "v").chunks == data_chunks  # data variable untouched

    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    assert list(ds.t.values) == list(times)  # values + datetime decode preserved
    assert float(ds.v.isel(t=39, y=0, x=0)) == 39.0

    # idempotent: a second call is a no-op
    assert downloader.ensure_time_coordinate_chunking(store_path, "t") is False


def test_write_to_icechunk_store_handles_a_reversed_dask_chunk_tuple(tmp_path: Path) -> None:
    """A south-up, dask-backed source must still publish.

    `raster_contract.ensure_north_up` flips y with `isel(slice(None, None, -1))`, and that
    reverses the dask *chunk tuple* along with the data: a legal trailing remainder
    (10, 10, 5) becomes a leading one (5, 10, 10). Zarr permits the short chunk only last,
    so `to_zarr` refused the write with "Zarr requires uniform chunk sizes except for final
    chunk".

    Regression for the dekad -> monthly aggregation failing to publish CLMS GPP, whose grid
    (1949 x 1895, chunked 244) sits below the pyramid threshold and so takes the flat write
    path — the pyramid path materialises via `load()` and never saw this.
    """
    y = np.arange(25.0)  # ascending => south-up => ensure_north_up will reverse it
    x = np.arange(8.0)
    data = np.arange(25 * 8, dtype="float32").reshape(1, 25, 8)
    ds = xr.Dataset(
        {"gpp": (("t", "y", "x"), data)},
        coords={"t": [np.datetime64("2024-01-01")], "y": y, "x": x},
    )
    ds.attrs["proj:code"] = "EPSG:4326"
    ds = ds.chunk({"y": 10, "x": 8})
    source_chunks = ds["gpp"].chunks
    assert source_chunks is not None
    assert source_chunks[1] == (10, 10, 5), "precondition: a trailing short chunk"

    store = tmp_path / "reversed_chunks.icechunk"
    downloader.write_to_icechunk_store(ds, store, commit_message="test")

    out = open_icechunk_dataset(str(store))
    # Reversed to north-up, and the value at a given latitude has not moved with it.
    assert out["y"].values[0] > out["y"].values[-1]
    assert float(out["gpp"].isel(t=0).sel(y=0.0, x=0.0)) == pytest.approx(0.0)
    assert float(out["gpp"].isel(t=0).sel(y=24.0, x=7.0)) == pytest.approx(25 * 8 - 1)


def test_uniform_chunks_leaves_a_legal_trailing_remainder_alone() -> None:
    """Only the illegal shape is rewritten — rechunking a large array is not free."""
    ds = xr.Dataset(
        {"v": (("y", "x"), np.zeros((25, 8), dtype="float32"))},
        coords={"y": np.arange(25.0), "x": np.arange(8.0)},
    ).chunk({"y": 10, "x": 8})
    assert ds["v"].chunks == ((10, 10, 5), (8,))

    # Identity, not merely an equal result: a rechunk of a full-resolution grid is not free.
    assert downloader._uniform_chunks(ds) is ds

    reversed_ds = ds.isel(y=slice(None, None, -1))
    assert reversed_ds["v"].chunks == ((5, 10, 10), (8,)), "precondition: leading short chunk"
    assert downloader._uniform_chunks(reversed_ds)["v"].chunks == ((10, 10, 5), (8,))


def test_open_icechunk_dataset_caps_the_steps_for_a_small_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny grid would fit years in one chunk by size; the step cap keeps it parallel."""
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_MAX_STEPS", 7)
    # 256-byte chunks against the default 16 MiB target: bytes alone would allow every day
    _daily_store(tmp_path / "small.icechunk", days=20)
    result = open_icechunk_dataset(tmp_path / "small.icechunk")
    try:
        assert _dask_chunks(result) == (7, 8, 8)
    finally:
        result.close()


def test_open_icechunk_dataset_step_cap_keeps_whole_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_MAX_STEPS", 7)
    _daily_store(tmp_path / "sharded.icechunk", encoding={"chunks": (1, 8, 8), "shards": (2, 16, 16)})
    result = open_icechunk_dataset(tmp_path / "sharded.icechunk")
    try:
        assert _dask_chunks(result) == (6, 8, 8)
    finally:
        result.close()


def _two_variable_store(path: Path, shards: tuple[int, int]) -> None:
    """Two daily variables whose time shards differ, as a merged store might have."""
    days = 24
    coords = {"t": pd.date_range("2020-01-01", periods=days, freq="D"), "y": np.arange(16.0), "x": np.arange(16.0)}
    ds = xr.Dataset(
        {
            "a": (["t", "y", "x"], np.zeros((days, 16, 16), dtype="float32")),
            "b": (["t", "y", "x"], np.ones((days, 16, 16), dtype="float32")),
        },
        coords=coords,
    )
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    encoding = {
        "a": {"chunks": (1, 8, 8), "shards": (shards[0], 16, 16)},
        "b": {"chunks": (1, 8, 8), "shards": (shards[1], 16, 16)},
    }
    ds.to_zarr(session.store, mode="w", zarr_format=3, encoding=encoding)
    session.commit("seed")


def test_open_icechunk_dataset_keeps_every_variables_shards_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shards of 2 and 3 days: the one time chunk is a multiple of both, 6."""
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 10 * 256)
    _two_variable_store(tmp_path / "merged.icechunk", shards=(2, 3))
    result = open_icechunk_dataset(tmp_path / "merged.icechunk")
    try:
        for name in ("a", "b"):
            chunks = result[name].chunks
            assert chunks is not None and chunks[0][0] == 6
    finally:
        result.close()


def test_open_icechunk_dataset_keeps_stored_chunks_when_no_common_multiple_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.data_accessor.services import accessor

    monkeypatch.setattr(accessor, "READ_CHUNK_TARGET_BYTES", 10 * 256)
    _two_variable_store(tmp_path / "merged.icechunk", shards=(4, 3))  # common multiple 12 > 10 steps
    result = open_icechunk_dataset(tmp_path / "merged.icechunk")
    try:
        chunks = result["a"].chunks
        assert chunks is not None and chunks[0][0] == 1
    finally:
        result.close()
