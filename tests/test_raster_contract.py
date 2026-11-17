"""The published-store layout contract, enforced at the write boundary (CLIM-821).

Dimension naming, axis order and CRS consistency are properties of a published GeoZarr, not
of any one source. Enforced per-plugin they get forgotten one plugin at a time — that is how
datasets shipped on a ``time`` dim the map viewer could not find, and how Norway's WorldPop
store came to declare ``EPSG:32633`` over degree coordinates.

``y`` must descend (row 0 = north), because real consumers assume it and never check — the
thumbnail renderer's ``imshow(origin="upper")`` and OpenLayers' GeoZarr tile grid among them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from open_geo_service.shared.raster_contract import (  # noqa: E402
    ContractViolation,
    _axis_units,
    normalize_dim_layout,
    normalize_longitudes,
    normalize_y_direction,
    prepare_for_publication,
    published_contract_violations,
    resolve_store_crs,
    spatial_coords_match,
)

# Deciding that a CRS is projected needs a working PROJ database. When pyproj cannot read one
# (a polluted host PROJ_DATA is a common developer-machine failure — see
# `startup._ensure_proj_database`), `_axis_units` reports "unknown" and the CRS checks
# deliberately stand down rather than guessing. Skip the assertions that need a real answer.
_CRS_RESOLVABLE = _axis_units("EPSG:32633") == "linear"
_needs_proj = pytest.mark.skipif(not _CRS_RESOLVABLE, reason="pyproj cannot resolve EPSG:32633 here")


@pytest.fixture
def geoservice_logs(caplog: pytest.LogCaptureFixture) -> Any:
    """Capture warnings from the OGS package logger.

    ``startup.py`` sets ``propagate = False`` on it so the app owns its own handler, which
    means caplog's root handler never sees the records. Re-enable propagation for the test.
    """
    package_logger = logging.getLogger("open_geo_service")
    previous = package_logger.propagate
    package_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="open_geo_service"):
            yield caplog
    finally:
        package_logger.propagate = previous


def _cube(*, dims: tuple[str, ...], sizes: dict[str, int], coords: dict[str, Any] | None = None) -> Any:
    shape = tuple(sizes[d] for d in dims)
    data = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    return xr.Dataset({"v": (dims, data)}, coords=coords or {})


# --- invariant 1: the temporal dimension is named `t` -------------------------------------


@pytest.mark.parametrize("alias", ["time", "valid_time", "date", "time_counter"])
def test_temporal_aliases_are_renamed_to_t(alias: str) -> None:
    """The map viewer looks for `t`; a store shipping `time` renders with no time control."""
    ds = _cube(dims=(alias, "y", "x"), sizes={alias: 2, "y": 3, "x": 4})
    out = normalize_dim_layout(ds)
    assert "t" in out.dims
    assert alias not in out.dims


def test_an_existing_t_dim_is_left_alone() -> None:
    """A cube with both `t` and `time` must not have `time` collapsed onto it."""
    ds = _cube(dims=("t", "y", "x"), sizes={"t": 2, "y": 3, "x": 4})
    ds = ds.assign_coords(time=("t", np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")))
    out = normalize_dim_layout(ds)
    assert "t" in out.dims
    assert "time" in out.coords  # kept as an auxiliary coordinate, not renamed over `t`


# --- invariant 2: spatial dims are (…, y, x) ----------------------------------------------


def test_transposed_variable_is_reordered() -> None:
    ds = _cube(dims=("t", "x", "y"), sizes={"t": 2, "x": 4, "y": 3})
    out = normalize_dim_layout(ds)
    assert out["v"].dims == ("t", "y", "x")


def test_reordering_moves_data_not_just_labels() -> None:
    """A transpose that relabelled without moving values would silently mirror the raster."""
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")  # (x=2, y=3)
    ds = xr.Dataset({"v": (("x", "y"), values)}, coords={"x": [10.0, 11.0], "y": [50.0, 51.0, 52.0]})

    out = normalize_dim_layout(ds)

    assert out["v"].dims == ("y", "x")
    np.testing.assert_array_equal(out["v"].values, values.T)
    # The value at a given coordinate pair is unchanged — only its position in the array.
    assert float(out["v"].sel(x=11.0, y=52.0)) == float(ds["v"].sel(x=11.0, y=52.0)) == 6.0


def test_spatial_dims_end_up_last_even_with_several_leading_dims() -> None:
    ds = _cube(dims=("y", "t", "x", "age"), sizes={"y": 3, "t": 2, "x": 4, "age": 5})
    out = normalize_dim_layout(ds)
    assert out["v"].dims[-2:] == ("y", "x")
    assert set(out["v"].dims[:-2]) == {"t", "age"}


def test_conformant_dataset_is_returned_untouched() -> None:
    """The common case must not pay for a copy of the data."""
    ds = _cube(dims=("t", "y", "x"), sizes={"t": 2, "y": 3, "x": 4})
    out = normalize_dim_layout(ds)
    assert out["v"].dims == ("t", "y", "x")
    assert out["v"].values is ds["v"].values


def test_non_spatial_variables_are_ignored() -> None:
    """A scalar CRS holder has no y/x to order, and must survive normalisation."""
    ds = _cube(dims=("t", "x", "y"), sizes={"t": 2, "x": 4, "y": 3})
    ds["spatial_ref"] = xr.DataArray(0)
    out = normalize_dim_layout(ds)
    assert "spatial_ref" in out
    assert out["v"].dims == ("t", "y", "x")


def test_missing_spatial_dims_are_left_for_the_caller_to_report() -> None:
    """The orchestrator raises a clearer error about missing y/x than a transpose would."""
    ds = _cube(dims=("time", "station"), sizes={"time": 2, "station": 5})
    out = normalize_dim_layout(ds)
    assert "t" in out.dims  # the temporal rename still happens
    assert out["v"].dims == ("t", "station")


# --- invariant 3: geographic longitudes run -180…180 --------------------------------------


def test_longitudes_are_rolled_off_the_0_360_frame() -> None:
    ds = _cube(dims=("y", "x"), sizes={"y": 2, "x": 4}, coords={"y": [1.0, 2.0], "x": [0.0, 90.0, 270.0, 350.0]})
    out = normalize_longitudes(ds, crs="EPSG:4326")
    assert out["x"].values.tolist() == [-90.0, -10.0, 0.0, 90.0]  # rolled and re-sorted


def test_rolling_carries_the_data_with_the_coordinates() -> None:
    """Rolling without re-sorting the data would shift every value by half the globe."""
    ds = xr.Dataset(
        {"v": (("y", "x"), np.array([[1.0, 2.0, 3.0, 4.0]], dtype="float32"))},
        coords={"y": [0.0], "x": [0.0, 90.0, 270.0, 350.0]},
    )
    out = normalize_longitudes(ds, crs="EPSG:4326")
    assert out["v"].sel(x=-90.0, y=0.0).item() == 3.0  # was 270°
    assert out["v"].sel(x=0.0, y=0.0).item() == 1.0


def test_already_rolled_longitudes_are_untouched() -> None:
    ds = _cube(dims=("y", "x"), sizes={"y": 2, "x": 3}, coords={"y": [1.0, 2.0], "x": [-10.0, 0.0, 10.0]})
    out = normalize_longitudes(ds, crs="EPSG:4326")
    assert out["x"].values.tolist() == [-10.0, 0.0, 10.0]


@_needs_proj
def test_projected_eastings_are_never_rolled() -> None:
    """A UTM easting is legitimately far larger than 180 — rolling it would be destructive."""
    ds = _cube(
        dims=("y", "x"),
        sizes={"y": 2, "x": 3},
        coords={"y": [6450500.0, 6451500.0], "x": [-74500.0, 500000.0, 1119500.0]},
    )
    out = normalize_longitudes(ds, crs="EPSG:32633")
    assert out["x"].values.tolist() == [-74500.0, 500000.0, 1119500.0]


# --- invariant 4: the declared CRS matches the coordinates --------------------------------


def _degree_grid(declared: str | None = None) -> Any:
    # Norway's WorldPop extent, the store that reported this bug.
    ds = _cube(
        dims=("y", "x"),
        sizes={"y": 2, "x": 2},
        coords={"y": [71.1846, 57.9596], "x": [4.5037, 31.1679]},
    )
    if declared is not None:
        ds.attrs["proj:code"] = declared
    return ds


@_needs_proj
def test_the_data_crs_wins_over_a_projected_declaration() -> None:
    """A declaration is a guess *about* the data, so the data outranks it.

    Getting this precedence backwards is how a projected code ended up over lon/lat coordinates:
    the instance's `crs:` was consulted as though it knew better than the store.
    """
    ds = _degree_grid(declared="EPSG:4326")
    assert resolve_store_crs(ds, "EPSG:32633") == "EPSG:4326"


@_needs_proj
def test_a_projected_crs_over_degree_coordinates_is_refused(geoservice_logs: Any) -> None:
    """The reported failure: EPSG:32633 over lon/lat puts the store at the UTM origin.

    Reached here with no CRS on the data at all, so the projected declaration is what wins the
    precedence — and is then rejected by the coordinates it claims to describe.
    """
    ds = _degree_grid()

    resolved = resolve_store_crs(ds, "EPSG:32633")

    assert resolved == "EPSG:4326"
    assert "EPSG:32633" in geoservice_logs.text
    assert "±180/±90" in geoservice_logs.text  # the log names the evidence, not just the verdict


@_needs_proj
def test_a_grid_still_on_the_0_360_frame_is_recognised_as_geographic(geoservice_logs: Any) -> None:
    """Chicken-and-egg: the roll needs to know the axis is a longitude, but an unrolled axis
    exceeds 180. Testing longitude against 180 would make every 0…360 dataset look projected,
    so the check allows up to 360 and leans on latitude (±90) for the real evidence.
    """
    ds = _cube(
        dims=("y", "x"),
        sizes={"y": 2, "x": 3},
        coords={"y": [60.0, 59.0], "x": [0.0, 180.0, 350.0]},
    )
    assert resolve_store_crs(ds, "EPSG:32633") == "EPSG:4326"
    assert "EPSG:32633" in geoservice_logs.text


@_needs_proj
def test_an_unresolvable_crs_is_left_as_declared() -> None:
    """When we cannot tell whether a CRS is projected, stand down rather than guess."""
    ds = _degree_grid()
    assert resolve_store_crs(ds, "EPSG:999999") == "EPSG:999999"


@_needs_proj
def test_a_consistent_projected_crs_is_kept() -> None:
    """Only a contradiction is overridden — a real projected grid must survive untouched."""
    ds = _cube(
        dims=("y", "x"),
        sizes={"y": 2, "x": 2},
        coords={"y": [6450500.0, 7999500.0], "x": [-74500.0, 1119500.0]},
    )
    assert resolve_store_crs(ds, "EPSG:32633") == "EPSG:32633"


@_needs_proj
def test_a_sticky_wrong_crs_on_the_data_is_also_refused(geoservice_logs: Any) -> None:
    """A wrong `proj:code` is copied forward on every rewrite, so the data's CRS can be wrong too.

    Norway's WorldPop store is in exactly this state: root says EPSG:32633, level 0 says
    EPSG:4326. Data-wins precedence must not turn a stale bad code into a licence to keep it.
    """
    ds = _degree_grid(declared="EPSG:32633")

    assert resolve_store_crs(ds, "EPSG:32633") == "EPSG:4326"

    assert "EPSG:32633 is projected" in geoservice_logs.text


def test_no_declared_crs_uses_the_datasets_own() -> None:
    assert resolve_store_crs(_degree_grid(declared="EPSG:4326"), None) == "EPSG:4326"


def test_crs84_aliases_are_canonicalised() -> None:
    assert resolve_store_crs(_degree_grid(), "OGC:CRS84") == "EPSG:4326"


def test_the_instance_config_crs_is_never_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config CRS as a fallback is what created the mismatch in the first place."""
    from open_geo_service import config as api_config

    monkeypatch.setattr(api_config, "get_crs", lambda: "EPSG:32633")
    assert resolve_store_crs(_degree_grid(), None) == "EPSG:4326"


# --- the contract as a whole ---------------------------------------------------------------


def test_violations_lists_every_problem_at_once() -> None:
    ds = _cube(dims=("time", "x", "y"), sizes={"time": 2, "x": 4, "y": 3})
    ds = ds.assign_coords(x=[0.0, 90.0, 270.0, 350.0], y=[1.0, 2.0, 3.0])
    ds.attrs["proj:code"] = "EPSG:32633"

    kinds = {violation.kind for violation in published_contract_violations(ds)}

    assert ContractViolation.TEMPORAL_DIM_NAME in kinds
    assert ContractViolation.AXIS_ORDER in kinds
    if _CRS_RESOLVABLE:
        assert ContractViolation.CRS_CONTRADICTS_COORDS in kinds


def test_a_normalised_dataset_has_no_violations() -> None:
    ds = _cube(dims=("time", "x", "y"), sizes={"time": 2, "x": 4, "y": 3})
    ds = ds.assign_coords(x=[0.0, 90.0, 270.0, 350.0], y=[1.0, 2.0, 3.0])

    ds = normalize_dim_layout(ds)
    ds = normalize_y_direction(ds)
    ds = normalize_longitudes(ds, crs="EPSG:4326")
    ds.attrs["proj:code"] = resolve_store_crs(ds, "EPSG:4326")

    assert published_contract_violations(ds) == []


# --- invariant 3: y descends (row 0 = north) ----------------------------------------------


def test_south_up_rows_are_reversed() -> None:
    ds = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 3, "x": 2}, coords={"y": [58.0, 59.0, 60.0]})
    out = normalize_y_direction(ds)
    assert out["y"].values.tolist() == [60.0, 59.0, 58.0]


def test_north_up_rows_are_left_alone() -> None:
    """The common case: most sources are already north-up, so this must cost nothing."""
    ds = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 3, "x": 2}, coords={"y": [60.0, 59.0, 58.0]})
    out = normalize_y_direction(ds)
    assert out["y"].values.tolist() == [60.0, 59.0, 58.0]
    assert out["v"].values is ds["v"].values


def test_reversing_carries_the_data_with_the_coordinate() -> None:
    """Reversing the coordinate without the data would mirror the raster — the exact bug this
    invariant exists to prevent, so it must not be introduced while preventing it."""
    values = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype="float32")
    ds = xr.Dataset(
        {"v": (("t", "y", "x"), values)},
        coords={"t": [0], "y": [58.0, 59.0, 60.0], "x": [10.0, 11.0]},
    )

    out = normalize_y_direction(ds)

    # The value at a given latitude is unchanged; only its row index moved.
    assert out["v"].sel(y=58.0, x=10.0).item() == 1.0
    assert out["v"].sel(y=60.0, x=10.0).item() == 5.0
    # And row 0 now holds the northernmost row, which is what consumers assume.
    assert out["v"].values[0, 0].tolist() == [5.0, 6.0]


def test_an_ascending_y_is_a_contract_violation() -> None:
    """OpenLayers' GeoZarr source maps array row 0 to the north edge and cannot detect otherwise,
    so a south-up store renders mirrored with no signal available to it."""
    ascending = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 2, "x": 2}, coords={"y": [59.0, 60.0]})
    descending = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 2, "x": 2}, coords={"y": [60.0, 59.0]})

    assert [v.kind for v in published_contract_violations(ascending)] == [ContractViolation.Y_ASCENDS]
    assert published_contract_violations(descending) == []


def test_a_single_row_grid_has_no_direction_to_normalise() -> None:
    ds = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 1, "x": 2}, coords={"y": [60.0]})
    assert normalize_y_direction(ds)["y"].values.tolist() == [60.0]
    assert published_contract_violations(ds) == []


# --- the append guard ----------------------------------------------------------------------


def test_matching_coords_are_appendable() -> None:
    stored = np.array([60.0, 59.0, 58.0])
    assert spatial_coords_match(stored, np.array([60.0, 59.0, 58.0]), axis="y") is True


def test_a_reversed_axis_is_not_appendable() -> None:
    """The corruption case: same values, opposite order. Data would land mirrored."""
    stored = np.array([58.0, 59.0, 60.0])  # a store written before the contract
    assert spatial_coords_match(stored, np.array([60.0, 59.0, 58.0]), axis="y") is False


def test_a_rolled_axis_is_not_appendable() -> None:
    stored = np.array([0.0, 90.0, 270.0])  # 0…360 frame
    assert spatial_coords_match(stored, np.array([-90.0, 0.0, 90.0]), axis="x") is False


def test_a_float_round_trip_still_matches() -> None:
    """A coordinate can lose a last bit through Zarr; that must not fail an ingest."""
    stored = np.array([60.0, 59.95, 59.9])
    incoming = stored + 1e-12
    assert spatial_coords_match(stored, incoming, axis="y") is True


def test_a_different_length_is_not_appendable() -> None:
    assert spatial_coords_match(np.array([60.0, 59.0]), np.array([60.0, 59.0, 58.0]), axis="y") is False


def test_written_store_satisfies_the_contract(tmp_path: Path) -> None:
    """End to end through the real write path, on a dataset that violates every invariant."""
    import icechunk

    pytest.importorskip("rioxarray")
    from open_geo_service.data_manager.services.downloader import write_to_icechunk_store

    # (time, x, y) order, a `time` dim, 0…360 longitudes, and a projected CRS over degrees.
    ds = xr.Dataset(
        {"v": (("time", "x", "y"), np.ones((1, 4, 3), dtype="float32"))},
        coords={
            "time": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "x": [0.0, 10.0, 350.0, 355.0],
            "y": [60.0, 59.0, 58.0],
        },
    )

    store_path = tmp_path / "contract.icechunk"
    write_to_icechunk_store(ds, store_path, t_dim="time", crs="EPSG:32633")

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    written = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
    try:
        assert published_contract_violations(written) == []
        assert written["v"].dims == ("t", "y", "x")
        assert written["x"].values.min() < 0  # rolled off the 0…360 frame
        assert written.attrs["proj:code"] == "EPSG:4326"
    finally:
        written.close()


def test_ingest_refuses_to_append_to_a_pre_contract_store(tmp_path: Path) -> None:
    """A store written south-up must not receive north-up periods.

    An append writes along the time axis only, so the committed y coordinate stays ascending
    while the new rows arrive descending — the data would be stored mirrored. Failing with an
    actionable message beats silently corrupting the store, and the fix is a re-ingest.
    """
    pytest.importorskip("icechunk")
    from open_geo_service.streaming.orchestrator import _assert_appendable_axes
    from open_geo_service.streaming.store import open_or_create_repo

    store_path = tmp_path / "south_up.icechunk"
    ascending = xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((1, 3, 2), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": [58.0, 59.0, 60.0],
            "x": [10.0, 11.0],
        },
    )
    repo = open_or_create_repo(store_path)
    session = repo.writable_session("main")
    ascending.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("pre-contract store")

    normalised = normalize_y_direction(ascending)
    assert normalised["y"].values.tolist() == [60.0, 59.0, 58.0]

    with pytest.raises(RuntimeError, match="Re-ingest this dataset"):
        _assert_appendable_axes(store_path, normalised, x_dim="x", y_dim="y")


def test_ingest_appends_happily_to_a_conforming_store(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    from open_geo_service.streaming.orchestrator import _assert_appendable_axes
    from open_geo_service.streaming.store import open_or_create_repo

    store_path = tmp_path / "north_up.icechunk"
    descending = xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((1, 3, 2), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": [60.0, 59.0, 58.0],
            "x": [10.0, 11.0],
        },
    )
    repo = open_or_create_repo(store_path)
    session = repo.writable_session("main")
    descending.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("conforming store")

    _assert_appendable_axes(store_path, normalize_y_direction(descending), x_dim="x", y_dim="y")


def test_a_missing_store_is_not_a_blocker(tmp_path: Path) -> None:
    """Nothing committed yet means nothing to disagree with — the first write must not fail."""
    from open_geo_service.streaming.orchestrator import _assert_appendable_axes

    ds = _cube(dims=("t", "y", "x"), sizes={"t": 1, "y": 2, "x": 2}, coords={"y": [60.0, 59.0]})
    _assert_appendable_axes(tmp_path / "absent.icechunk", ds, x_dim="x", y_dim="y")


# --- the single entry point ----------------------------------------------------------------


@_needs_proj
def test_prepare_never_rolls_a_projected_axis(geoservice_logs: Any) -> None:
    """The defect this entry point exists to make unexpressible.

    Composed by hand, the streaming path passed the plugin's `crs` *class attribute* into the
    longitude roll while the resolved CRS sat unused a few lines away. With no attribute declared
    that argument is None, the projected-CRS guard is skipped, and an easting of 500000 rolls to
    ((500000 + 180) % 360) - 180 = -40 — the store silently destroyed. Going through
    prepare_for_publication, the roll cannot be reached before the CRS is resolved.
    """
    ds = xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((1, 2, 3), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": [7_000_000.0, 6_999_000.0],
            "x": [400_000.0, 500_000.0, 600_000.0],
        },
    )
    ds.attrs["proj:code"] = "EPSG:32633"

    prepared = prepare_for_publication(ds)  # no fallback_crs — the data must carry the decision

    assert prepared.crs == "EPSG:32633"
    assert prepared.dataset["x"].values.tolist() == [400_000.0, 500_000.0, 600_000.0]
    assert "Rolling" not in geoservice_logs.text


def test_prepare_applies_every_invariant_in_one_call() -> None:
    """A cube violating all four at once comes out conforming."""
    ds = xr.Dataset(
        {"v": (("time", "x", "y"), np.ones((1, 4, 3), dtype="float32"))},
        coords={
            "time": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "x": [0.0, 10.0, 350.0, 355.0],
            "y": [57.0, 58.0, 59.0],
        },
    )

    prepared = prepare_for_publication(ds, fallback_crs="EPSG:32633")
    out = prepared.dataset

    assert published_contract_violations(out) == []
    assert out["v"].dims == ("t", "y", "x")  # renamed and transposed
    assert out["y"].values.tolist() == [59.0, 58.0, 57.0]  # north-up
    assert out["x"].values.min() < 0  # rolled off 0…360
    assert prepared.crs == "EPSG:4326"  # the projected declaration refused by the coordinates


def test_prepare_leaves_a_conforming_cube_alone() -> None:
    ds = xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((1, 2, 2), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": [60.0, 59.0],
            "x": [10.0, 11.0],
        },
    )
    ds.attrs["proj:code"] = "EPSG:4326"

    prepared = prepare_for_publication(ds)

    assert prepared.crs == "EPSG:4326"
    assert prepared.dataset["v"].values is ds["v"].values
