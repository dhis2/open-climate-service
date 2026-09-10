"""Appending to a store that an earlier sync promoted to a pyramid.

A pyramided store keeps its data variables in level groups and leaves the root with only the
time coordinate and ``spatial_ref``. Appending at the root therefore does not extend level 0 —
it creates a second, one-timestep variable beside the pyramid, after which the root cannot be
opened at all (``conflicting sizes for dimension 't'``) and nothing can repair it in place.

Latent until the pyramid threshold dropped to 1024^2: every store above the old 2048^2 bar was
`sync: kind: release` or `static`, so none of them appended. Uganda's dekadal GPP and Norway's
seNorge dailies do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
icechunk = pytest.importorskip("icechunk")
pytest.importorskip("topozarr")
pytest.importorskip("rioxarray")
zarr = pytest.importorskip("zarr")

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset  # noqa: E402
from open_climate_service.data_manager.services import downloader  # noqa: E402
from open_climate_service.streaming.store import (  # noqa: E402
    committed_data_group,
    read_committed_period_ids,
    read_committed_spatial_coords,
)


def _period(day: int, ny: int = 1500, nx: int = 1200):  # noqa: ANN202 — `xr` is an importorskip variable
    """One period on a grid over the 1024^2 threshold, so it gets a pyramid."""
    return xr.Dataset(
        {"gpp": (("t", "y", "x"), np.full((1, ny, nx), float(day), dtype="float32"))},
        coords={
            "t": np.array([f"2024-01-{day:02d}"], dtype="datetime64[ns]"),
            "y": np.linspace(4.0, -1.0, ny),
            "x": np.linspace(29.0, 35.0, nx),
        },
    )


def _pyramided_store(tmp_path: Path) -> tuple[Path, Any]:
    """A store with two committed periods, promoted to a pyramid as a sync would leave it."""
    store_path = tmp_path / "gpp.icechunk"
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(store_path)))
    session = repo.writable_session("main")
    _period(1).to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("p1")
    session = repo.writable_session("main")
    _period(2).to_zarr(session.store, append_dim="t", zarr_format=3)
    session.commit("p2")
    downloader.write_to_icechunk_store(
        open_icechunk_dataset(store_path), store_path, crs="EPSG:4326", commit_message="promote"
    )
    return store_path, repo


def test_committed_data_group_distinguishes_flat_from_pyramided(tmp_path: Path) -> None:
    flat = tmp_path / "flat.icechunk"
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(flat)))
    session = repo.writable_session("main")
    _period(1, ny=64, nx=64).to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("p1")
    assert committed_data_group(flat) is None

    pyramided, _ = _pyramided_store(tmp_path)
    assert committed_data_group(pyramided) == "0"


def test_appending_into_the_committed_group_keeps_the_store_readable(tmp_path: Path) -> None:
    """The regression: appending at the root instead left the store unopenable."""
    store_path, repo = _pyramided_store(tmp_path)

    session = repo.writable_session("main")
    _period(3).to_zarr(session.store, group=committed_data_group(store_path), append_dim="t", zarr_format=3)
    session.commit("p3")

    ds = open_icechunk_dataset(store_path)
    try:
        assert ds.sizes["t"] == 3
        assert [float(v) for v in ds["gpp"][:, 0, 0].values] == [1.0, 2.0, 3.0]
    finally:
        ds.close()


def test_appending_at_the_root_of_a_pyramided_store_would_break_it(tmp_path: Path) -> None:
    """Pins why `committed_data_group` exists, so the guard is not removed as redundant."""
    store_path, repo = _pyramided_store(tmp_path)

    session = repo.writable_session("main")
    _period(3).to_zarr(session.store, append_dim="t", zarr_format=3)  # no group= -> the root
    session.commit("bad append")

    with pytest.raises(ValueError, match="conflicting sizes for dimension"):
        open_icechunk_dataset(store_path)


def test_committed_periods_come_from_level_zero(tmp_path: Path) -> None:
    """The root time coordinate is only rebuilt at the end of a sync, so it lags mid-sync.

    Reading it would make resume re-append periods that are already committed.
    """
    store_path, repo = _pyramided_store(tmp_path)
    session = repo.writable_session("main")
    _period(3).to_zarr(session.store, group=committed_data_group(store_path), append_dim="t", zarr_format=3)
    session.commit("p3")

    root_t = zarr.open_group(repo.readonly_session("main").store, mode="r")["t"].shape[0]
    assert root_t == 2, "precondition: the root coordinate is stale until the rebuild"
    assert read_committed_period_ids(store_path, "daily") == {
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    }


def test_resetting_a_failed_append_restores_every_pyramid_level(tmp_path: Path) -> None:
    """The CLIM-899 rollback snapshot restores level 0, root, and overviews together."""
    store_path, repo = _pyramided_store(tmp_path)
    before = repo.lookup_branch("main")
    repo.create_branch("rollback-test", before)

    session = repo.writable_session("main")
    _period(3).to_zarr(session.store, group=committed_data_group(store_path), append_dim="t", zarr_format=3)
    session.commit("partial append")
    assert len(read_committed_period_ids(store_path, "daily")) == 3

    repo.reset_branch("main", before)
    repo.delete_branch("rollback-test")

    store = repo.readonly_session("main").store
    root = zarr.open_group(store, mode="r")
    counts = {"root": root["t"].shape[0]}
    for group in root.group_keys():
        level = xr.open_zarr(store, group=group)
        try:
            counts[group] = level.sizes["t"]
        finally:
            level.close()
    assert set(counts.values()) == {2}


def test_spatial_guard_is_active_on_a_pyramided_store(tmp_path: Path) -> None:
    """Reading the root returned None here, silently disabling the mirrored-raster check."""
    store_path, _ = _pyramided_store(tmp_path)

    stored = read_committed_spatial_coords(store_path)

    assert stored is not None
    assert stored["y"][0] > stored["y"][-1]  # north-up, per the publication contract
    assert len(stored["x"]) == 1200


def test_appending_does_not_un_declare_the_pyramid(tmp_path: Path) -> None:
    """`write_geozarr_attrs` runs on every commit and rebuilds `zarr_conventions` for a flat
    store, which used to drop the multiscales entry — leaving a store that advertised itself as
    flat while having no data variables at the root at all. If the run then aborted before the
    end-of-sync rebuild, that is how it stayed.
    """
    from open_climate_service.stac.media_types import attributes_declare_multiscales
    from open_climate_service.streaming.protocol import GridSpec
    from open_climate_service.streaming.store import write_geozarr_attrs

    store_path, repo = _pyramided_store(tmp_path)

    def declared() -> bool:
        attrs = dict(zarr.open_group(repo.readonly_session("main").store, mode="r").attrs)
        return attributes_declare_multiscales(attrs)

    assert declared(), "precondition: the built pyramid declares the convention"

    spec = GridSpec(
        shape=(1500, 1200),
        crs=4326,
        dtype=np.dtype("float32"),
        nodata=None,
        time_dim="t",
        x_dim="x",
        y_dim="y",
    )
    session = repo.writable_session("main")
    _period(3).to_zarr(session.store, group=committed_data_group(store_path), append_dim="t", zarr_format=3)
    write_geozarr_attrs(session.store, spec=spec, bbox=[29.0, -1.0, 35.0, 4.0])
    session.commit("append + attrs")

    assert declared(), "the pyramid must still be declared after an append"
    attrs = dict(zarr.open_group(repo.readonly_session("main").store, mode="r").attrs)
    names = [c.get("name") for c in attrs["zarr_conventions"] if isinstance(c, dict)]
    assert names.count("multiscales") == 1, f"declared once, not duplicated per commit: {names}"


def test_a_flat_store_does_not_gain_a_multiscales_declaration(tmp_path: Path) -> None:
    """The preservation must be conditional — a flat store claiming a pyramid is worse."""
    from open_climate_service.stac.media_types import attributes_declare_multiscales
    from open_climate_service.streaming.protocol import GridSpec
    from open_climate_service.streaming.store import write_geozarr_attrs

    flat = tmp_path / "flat.icechunk"
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(flat)))
    session = repo.writable_session("main")
    _period(1, ny=64, nx=64).to_zarr(session.store, mode="w", zarr_format=3)
    write_geozarr_attrs(
        session.store,
        spec=GridSpec(
            shape=(64, 64),
            crs=4326,
            dtype=np.dtype("float32"),
            nodata=None,
            time_dim="t",
            x_dim="x",
            y_dim="y",
        ),
        bbox=[29.0, -1.0, 35.0, 4.0],
    )
    session.commit("flat")

    attrs = dict(zarr.open_group(repo.readonly_session("main").store, mode="r").attrs)
    assert not attributes_declare_multiscales(attrs)
