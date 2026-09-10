"""Regression test for the managed-store multiscale pyramid write.

Guards against the topozarr API drift that broke publishing of large workflow
outputs (`'Pyramid' object has no attribute 'dt'`): exercises the full
`write_to_icechunk_store` pyramid path against the current `topozarr` API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("icechunk")
pytest.importorskip("topozarr")
pytest.importorskip("rioxarray")
zarr = pytest.importorskip("zarr")

from open_climate_service.data_manager.services.downloader import (  # noqa: E402
    _coarsen_native,
    _normalize_resampling_method,
    needs_pyramid,
    resampling_method_from_template,
    write_to_icechunk_store,
)


def _pyramid_sized_cube():
    # Larger than the pyramid threshold; integer var exercises the
    # no-data fill_value handling that used to be pinned via pyramid.encoding.
    ny = nx = 2100
    data = np.ones((1, ny, nx), dtype="uint8")
    return xr.Dataset(
        {"hotspot": (("t", "y", "x"), data)},
        coords={
            "t": np.array(["2020-01-01"], dtype="datetime64[ns]"),
            "y": np.linspace(-2.9, -1.0, ny),
            "x": np.linspace(28.8, 30.9, nx),
        },
    )


def test_write_to_icechunk_store_builds_pyramid(tmp_path: Path) -> None:
    import icechunk

    ds = _pyramid_sized_cube()
    assert needs_pyramid(ds)

    store_path = tmp_path / "pyramid.icechunk"
    # Must not raise (regression: topozarr Pyramid.write API, not pyramid.dt.*).
    write_to_icechunk_store(ds, store_path, crs="EPSG:4326")

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    store = repo.readonly_session("main").store
    root = zarr.open_group(store, mode="r")

    # Root carries the multiscales metadata, and the level groups exist.
    assert "multiscales" in dict(root.attrs)
    level_groups = {k for k, _ in root.groups()}
    assert {"0", "1"} <= level_groups

    # Full-resolution level reads back at the original size with the variable intact.
    with xr.open_zarr(store, group="0", consolidated=False, zarr_format=3) as lvl0:
        assert "hotspot" in lvl0.data_vars
        assert lvl0.sizes["y"] == 2100 and lvl0.sizes["x"] == 2100


def test_resampling_method_from_template() -> None:
    assert resampling_method_from_template(None) == "mean"
    assert resampling_method_from_template({}) == "mean"
    assert resampling_method_from_template({"ingestion": {"plugin": "x"}}) == "mean"
    assert resampling_method_from_template({"ingestion": {"resampling": "mode"}}) == "mode"
    assert resampling_method_from_template({"ingestion": {"resampling": "MAX"}}) == "max"
    # Unrecognised values fall back to mean rather than being passed downstream.
    assert resampling_method_from_template({"ingestion": {"resampling": "bilinear"}}) == "mean"
    # A display-block value is ignored — resampling is an ingestion concern.
    assert resampling_method_from_template({"display": {"resampling": "mode"}}) == "mean"


def test_normalize_resampling_method() -> None:
    # Case/whitespace are normalized, and unknown or missing values fall back to mean —
    # so a direct write_to_icechunk_store(pyramid_method=...) caller can't smuggle an
    # invalid CoarseningMethod through to topozarr.
    assert _normalize_resampling_method("MAX") == "max"
    assert _normalize_resampling_method("  mean ") == "mean"
    assert _normalize_resampling_method("Mode") == "mode"
    assert _normalize_resampling_method(None) == "mean"
    assert _normalize_resampling_method("bilinear") == "mean"


def _categorical_block_da():
    # 4x4 class-code grid. Each 2x2 block has a clear majority that differs from its
    # top-left cell, so mode and nearest give distinguishable results (and mean would
    # invent codes that don't exist, e.g. mean(99,10,10,10)=32.25).
    data = np.array(
        [
            [99, 10, 20, 20],
            [10, 10, 20, 30],
            [30, 30, 40, 41],
            [30, 31, 40, 40],
        ],
        dtype="uint8",
    )
    return xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(4.0), "x": np.arange(4.0)},
    )


def test_coarsen_native_mode_picks_block_majority() -> None:
    out = _coarsen_native(_categorical_block_da(), "x", "y", 2, "mode").transpose("y", "x")
    np.testing.assert_array_equal(out.values, np.array([[10, 20], [30, 40]], dtype="uint8"))


def test_nearest_is_delegated_to_topozarr() -> None:
    """`nearest` arrived upstream in topozarr 0.1.3, so OCS no longer computes it.

    It is composable — corner-of-corners equals corner-of-native — so level-from-previous
    (what topozarr does) and level-from-native (what OCS used to do) agree. Verified
    byte-identical across a real 3-level pyramid when the delegation was made.
    """
    from open_climate_service.data_manager.services.downloader import (
        _COMPOSABLE_METHODS,
        _NATIVE_RESAMPLE_METHODS,
    )

    assert "nearest" in _COMPOSABLE_METHODS
    assert "nearest" not in _NATIVE_RESAMPLE_METHODS
    # Still a valid template value — it just takes the upstream path now.
    assert _normalize_resampling_method("nearest") == "nearest"


def test_coarsen_native_only_handles_mode() -> None:
    """`mode` is the sole remaining local method; anything else is a programming error."""
    with pytest.raises(ValueError, match="unsupported native-resample method"):
        _coarsen_native(_categorical_block_da(), "x", "y", 2, "nearest")


def test_topozarr_nearest_keeps_the_top_left_cell() -> None:
    """Pin the upstream kernel's corner choice, since our equivalence argument rests on it.

    Also the last line of defence against a `topozarr` / `topozarr-core` version mismatch. That
    used to be reachable — 0.1.3 declared only `topozarr-core>=0.1.0` while the `nearest` kernel
    exists from core 0.1.2 — and it failed at ingest time rather than on import. 0.1.4 pins the
    pair exactly, so it should no longer be possible; this asserts that rather than assuming it.
    """
    from topozarr_core import block_reduce

    block = np.arange(16, dtype="int16").reshape(4, 4)
    out = np.asarray(block_reduce(block, (2, 2), "nearest", 0, True))
    np.testing.assert_array_equal(out, np.array([[0, 2], [8, 10]], dtype="int16"))


def _categorical_pyramid_cube():
    # Pyramid-sized grid of a few land-cover-style class codes (no 0 background).
    ny = nx = 2100
    codes = np.array([10, 20, 30, 40, 50], dtype="uint8")
    rng = np.random.default_rng(0)
    data = codes[rng.integers(0, len(codes), size=(ny, nx))]
    return xr.Dataset(
        {"landcover": (("y", "x"), data)},
        coords={"y": np.linspace(-2.9, -1.0, ny), "x": np.linspace(28.8, 30.9, nx)},
    ), set(int(c) for c in codes)


def test_write_to_icechunk_store_mode_keeps_class_codes(tmp_path: Path) -> None:
    import icechunk

    ds, codes = _categorical_pyramid_cube()
    assert needs_pyramid(ds)

    store_path = tmp_path / "landcover.icechunk"
    write_to_icechunk_store(ds, store_path, t_dim=None, crs="EPSG:4326", pyramid_method="mode")

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    store = repo.readonly_session("main").store

    # A coarsened level must contain only real class codes — never an averaged value
    # (which is exactly what the previous hardcoded "mean" produced for categorical data).
    with xr.open_zarr(store, group="1", consolidated=False, zarr_format=3) as lvl1:
        values = np.unique(lvl1["landcover"].values)
    assert set(int(v) for v in values) <= codes


def _uganda_gpp_shaped_cube(nt: int = 2):
    """A cube on the grid that motivated lowering the threshold: CLMS GPP over Uganda."""
    ny, nx = 1949, 1895
    return xr.Dataset(
        {"gpp": (("t", "y", "x"), np.zeros((nt, ny, nx), dtype="float32"))},
        coords={
            "t": np.array(["2024-01-01", "2024-01-11"], dtype="datetime64[ns]")[:nt],
            "y": np.linspace(4.29, -1.50, ny),
            "x": np.linspace(29.47, 35.11, nx),
        },
    )


def test_threshold_catches_the_grid_that_used_to_render_flat() -> None:
    """1949 x 1895 = 3.69 Mpx sat at 0.88x of the old 2048^2 bar, so no pyramid was built.

    Every frame then read the full grid at every zoom — 32 chunk requests and 14.8 MB per
    timestep to fill a few hundred screen pixels. Pinned to the real geometry rather than a
    round number so the constant cannot drift back above it unnoticed.
    """
    assert needs_pyramid(_uganda_gpp_shaped_cube())


def test_threshold_still_skips_a_grid_small_enough_to_send_whole() -> None:
    """A pyramid on a small grid is pure overhead: more metadata, no fewer bytes to draw."""
    ny = nx = 700  # 0.49 Mpx
    small = xr.Dataset(
        {"v": (("y", "x"), np.zeros((ny, nx), dtype="float32"))},
        coords={"y": np.linspace(1.0, 0.0, ny), "x": np.linspace(0.0, 1.0, nx)},
    )
    assert not needs_pyramid(small)


def test_pyramid_build_never_materialises_the_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """topozarr streams level 0 region by region, so the source must stay lazy.

    Loading it first defeated that and made peak memory scale with the store: measured on a
    2.13 GB cube, 7.50 GB peak loaded versus 1.19 GB streamed. Asserted by making `.load()`
    fail outright, which is the only way to prove it is not called somewhere in the path.
    """

    def explode(self: object, **kwargs: object) -> None:
        raise AssertionError("write_to_icechunk_store must not materialise the source")

    monkeypatch.setattr(xr.Dataset, "load", explode)
    monkeypatch.setattr(xr.DataArray, "load", explode)

    ds = _uganda_gpp_shaped_cube().chunk({"t": 1, "y": 244, "x": 474})
    assert ds["gpp"].chunks is not None
    assert needs_pyramid(ds)

    store_path = tmp_path / "streamed.icechunk"
    write_to_icechunk_store(ds, store_path, crs="EPSG:4326")

    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    root = zarr.open_group(repo.readonly_session("main").store, mode="r")
    assert "multiscales" in dict(root.attrs)
    assert {"0", "1"} <= {k for k, _ in root.groups()}


def _projected_pyramid_sized_cube():
    """A pyramid-sized cube on the British National Grid, in its native metres.

    The coordinates have to be real eastings/northings, not degrees: the write path replaces
    a projected CRS declared over coordinates that can only be degrees (see
    ``resolve_store_crs``), so a lon/lat grid labelled EPSG:27700 would be stored as
    EPSG:4326 and the test would assert nothing.
    """
    ny = nx = 2100
    return xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((1, ny, nx), dtype="float32"))},
        coords={
            "t": np.array(["2020-01-01"], dtype="datetime64[ns]"),
            "y": np.linspace(700000.0, 100000.0, ny),
            "x": np.linspace(100000.0, 600000.0, nx),
        },
    )


def test_pyramid_levels_carry_the_crs_definition_and_not_just_the_code(tmp_path: Path) -> None:
    """Every group that declares `proj:` must declare the whole CRS, root and levels alike.

    A client reads the `proj:` convention from the innermost source that declares any of it
    — the data array, then the level-0 group, then the root — and stops there. Level groups
    are written from the cube's own attrs, which carry a bare `proj:code`, so a definition
    written only at the root sits behind a level-0 code that shadows it. For a code proj4
    cannot resolve (EPSG:27700 here, unlike the UTM zones it ships) that leaves the CRS
    unresolved and the store rendered in the wrong place, exactly as if the root said
    nothing — see CLIM-833.
    """
    import icechunk
    from pyproj import CRS

    store_path = tmp_path / "national_grid.icechunk"
    write_to_icechunk_store(_projected_pyramid_sized_cube(), store_path, crs="EPSG:27700")

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    root = zarr.open_group(repo.readonly_session("main").store, mode="r")

    for name, attrs in [("root", dict(root.attrs)), ("level 0", dict(root["0"].attrs))]:
        assert attrs["proj:code"] == "EPSG:27700", name
        assert CRS.from_wkt(attrs["proj:wkt2"]) == CRS.from_epsg(27700), name
        assert CRS.from_json_dict(attrs["proj:projjson"]) == CRS.from_epsg(27700), name
