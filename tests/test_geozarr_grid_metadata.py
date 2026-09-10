"""GeoZarr grid metadata: axis order, the affine, and the extent actually stored.

Direct-Zarr clients georeference from the store alone — no STAC, no catalog. Three things
they read were wrong or missing (CLIM-852), and each has its own failure signature:

* ``spatial:dimensions`` / ``spatial:shape`` are positional. A client takes the
  second-to-last entry as the row (y) axis and the last as the column (x) axis. Writing
  ``["x", "y"]`` / ``[width, height]`` transposes the grid.
* ``spatial:transform`` was never written for flat stores. Without it, viewers infer the grid
  from the coordinate arrays and assume EPSG:4326 while doing so — which puts a store in
  projected metres nowhere near its real location.
* ``spatial:bbox`` described the *requested* bbox. A source can return less than was asked
  for, so the declared extent claimed ground the store does not cover.

The affine assertions are pinned against CHIRPS' real numbers: the GeoTIFFs OCS ingests
carry their own GDAL ``GeoTransform``, so deriving one from the stored coordinates has an
independent right answer to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("icechunk")
pytest.importorskip("rioxarray")
zarr = pytest.importorskip("zarr")

from open_climate_service.shared.geozarr import (  # noqa: E402
    check_grid_description,
    gdal_geotransform,
    grid_geometry,
)
from open_climate_service.streaming.protocol import GridSpec  # noqa: E402
from open_climate_service.streaming.store import write_geozarr_attrs  # noqa: E402

# The real CHIRPS-over-Norway grid: 0.05° cells, and the source GeoTIFF's own GeoTransform
# is "2.9500027261674404 0.05000000074505806 0.0 60.0 0.0 -0.05000000074505806".
CHIRPS_X0 = 2.9750027265399694
CHIRPS_Y0 = 59.97499999962747
CHIRPS_STEP = 0.05000000074505806
CHIRPS_NX = 581
CHIRPS_NY = 60
# Norway asked for 72.5°N; CHIRPS stops at 60°N, so the store holds far less than the request.
CHIRPS_REQUESTED_BBOX = [3.0, 57.0, 32.0, 72.5]


def _chirps_coords() -> tuple[np.ndarray, np.ndarray]:
    x = CHIRPS_X0 + np.arange(CHIRPS_NX) * CHIRPS_STEP
    y = CHIRPS_Y0 - np.arange(CHIRPS_NY) * CHIRPS_STEP
    return x, y


def _cube(x: np.ndarray, y: np.ndarray, crs: str) -> Any:  # an xr.Dataset; xr is importorskip'd
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((1, len(y), len(x)), dtype="float32"))},
        coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]"), "y": y, "x": x},
    )
    return ds.rio.write_crs(crs)


def test_grid_geometry_matches_the_sources_own_geotransform() -> None:
    """The affine derived from stored cell centres reproduces CHIRPS' source GeoTransform."""
    x, y = _chirps_coords()
    geometry = grid_geometry(x, y)
    assert geometry is not None

    step_x, rot_x, origin_x, rot_y, step_y, origin_y = geometry["transform"]
    assert (rot_x, rot_y) == (0.0, 0.0)  # north-up, unrotated
    assert step_x == pytest.approx(CHIRPS_STEP)
    assert step_y == pytest.approx(-CHIRPS_STEP)  # rows run north→south
    # Pixel registration: the origin is the cell EDGE, half a step out from the first centre.
    assert origin_x == pytest.approx(2.9500027261674404)
    assert origin_y == pytest.approx(60.0)


def test_grid_geometry_shape_is_array_order() -> None:
    x, y = _chirps_coords()
    geometry = grid_geometry(x, y)
    assert geometry is not None
    assert geometry["shape"] == [CHIRPS_NY, CHIRPS_NX]  # (rows, columns), not (x, y)


def test_grid_geometry_bbox_covers_the_outer_cell_edges() -> None:
    x, y = _chirps_coords()
    geometry = grid_geometry(x, y)
    assert geometry is not None
    xmin, ymin, xmax, ymax = geometry["bbox"]
    assert (xmin, ymax) == pytest.approx((2.9500027261674404, 60.0))
    # Ordered [xmin, ymin, xmax, ymax] even though y descends through the array.
    assert ymin < ymax and xmin < xmax
    assert ymin == pytest.approx(60.0 - CHIRPS_NY * CHIRPS_STEP)


def test_grid_geometry_handles_ascending_y() -> None:
    """A south-up grid keeps its positive step — flipping the sign would mirror the image."""
    geometry = grid_geometry(np.array([10.0, 11.0, 12.0]), np.array([50.0, 51.0, 52.0]))
    assert geometry is not None
    assert geometry["transform"][4] == pytest.approx(1.0)
    assert geometry["transform"][5] == pytest.approx(49.5)
    assert geometry["bbox"][1] == pytest.approx(49.5)
    assert geometry["bbox"][3] == pytest.approx(52.5)


@pytest.mark.parametrize(
    "x,y",
    [
        ([1.0], [1.0, 2.0]),  # single column
        ([1.0, 2.0], [1.0]),  # single row
        ([], []),
        ([1.0, 1.0], [1.0, 2.0]),  # duplicated coordinate → zero step
    ],
)
def test_grid_geometry_refuses_to_guess_a_degenerate_grid(x: list[float], y: list[float]) -> None:
    """One cell on an axis leaves the cell size unknowable, so claim nothing rather than guess."""
    assert grid_geometry(np.array(x), np.array(y)) is None


def test_gdal_geotransform_reorders_the_coefficients() -> None:
    """GDAL wants originX stepX rotX originY rotY stepY — a different order to GeoZarr's."""
    assert gdal_geotransform([0.05, 0.0, 2.95, 0.0, -0.05, 60.0]) == "2.95 0.05 0.0 60.0 0.0 -0.05"


def _init_store(path: Path, ds: Any) -> str:
    store = str(path / "s.zarr")
    ds.to_zarr(store, mode="w", zarr_format=3, consolidated=False)
    return store


def test_streaming_store_declares_axes_in_array_order(tmp_path: Path) -> None:
    x, y = _chirps_coords()
    store = _init_store(tmp_path, _cube(x, y, "EPSG:4326"))
    spec = GridSpec(shape=(CHIRPS_NY, CHIRPS_NX), crs=4326, dtype=np.dtype("float32"))

    write_geozarr_attrs(store, spec=spec, bbox=CHIRPS_REQUESTED_BBOX)

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:shape"] == [CHIRPS_NY, CHIRPS_NX]


def test_streaming_store_writes_the_affine(tmp_path: Path) -> None:
    """Present at all, and matching the source's GeoTransform — this is what places a raster."""
    x, y = _chirps_coords()
    store = _init_store(tmp_path, _cube(x, y, "EPSG:4326"))
    spec = GridSpec(shape=(CHIRPS_NY, CHIRPS_NX), crs=4326, dtype=np.dtype("float32"))

    write_geozarr_attrs(store, spec=spec, bbox=CHIRPS_REQUESTED_BBOX)

    transform = dict(zarr.open_group(store, mode="r").attrs)["spatial:transform"]
    assert transform == pytest.approx([CHIRPS_STEP, 0.0, 2.9500027261674404, 0.0, -CHIRPS_STEP, 60.0])


def test_streaming_store_extent_describes_the_data_not_the_request(tmp_path: Path) -> None:
    """CHIRPS ends at 60°N; a store that claims 72.5°N stretches over 12° of missing ground."""
    x, y = _chirps_coords()
    store = _init_store(tmp_path, _cube(x, y, "EPSG:4326"))
    spec = GridSpec(shape=(CHIRPS_NY, CHIRPS_NX), crs=4326, dtype=np.dtype("float32"))

    write_geozarr_attrs(store, spec=spec, bbox=CHIRPS_REQUESTED_BBOX)

    bbox = dict(zarr.open_group(store, mode="r").attrs)["spatial:bbox"]
    assert bbox[3] == pytest.approx(60.0)
    assert bbox[3] != CHIRPS_REQUESTED_BBOX[3]


def test_streaming_store_falls_back_to_the_request_bbox_without_coordinates(tmp_path: Path) -> None:
    """First write, or a degenerate grid: keep the old behaviour rather than fail the ingest."""
    store = str(tmp_path / "empty.zarr")
    zarr.open_group(store, mode="w", zarr_format=3)
    spec = GridSpec(shape=(3, 3), crs=32633, dtype=np.dtype("float32"))

    write_geozarr_attrs(store, spec=spec, bbox=[0.0, 0.0, 1.0, 1.0])

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs["spatial:bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert "spatial:transform" not in attrs
    assert attrs["spatial:dimensions"] == ["y", "x"]  # axis order doesn't need coordinates


def test_streaming_store_stamps_the_gdal_geotransform(tmp_path: Path) -> None:
    """``rio.write_crs`` writes crs_wkt but not the transform, so we add it.

    Without it, whether a store carries a GeoTransform depends on whether its source
    happened to have one, and that attribute is what identifies a projected grid.
    """
    x = -74500.0 + np.arange(4) * 1000.0
    y = 7999500.0 - np.arange(3) * 1000.0
    store = _init_store(tmp_path, _cube(x, y, "EPSG:32633"))
    spec = GridSpec(shape=(3, 4), crs=32633, dtype=np.dtype("float32"))

    write_geozarr_attrs(store, spec=spec, bbox=[0.0, 0.0, 1.0, 1.0])

    root = zarr.open_group(store, mode="r")
    assert dict(root["spatial_ref"].attrs)["GeoTransform"] == "-75000.0 1000.0 0.0 8000000.0 0.0 -1000.0"
    assert dict(root["spatial_ref"].attrs)["grid_mapping_name"] == "transverse_mercator"


def _write_icechunk(tmp_path: Path, ds: Any, name: str, crs: str) -> Any:
    import icechunk

    from open_climate_service.data_manager.services.downloader import write_to_icechunk_store

    path = tmp_path / f"{name}.icechunk"
    write_to_icechunk_store(ds, path, t_dim="t", crs=crs)
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    return zarr.open_group(repo.readonly_session("main").store, mode="r")


def test_downloader_flat_store_grid_metadata(tmp_path: Path) -> None:
    """The batch/openEO write path must describe its grid the same way the streaming one does."""
    x, y = _chirps_coords()
    root = _write_icechunk(tmp_path, _cube(x, y, "EPSG:4326"), "flat", "EPSG:4326")

    attrs = dict(root.attrs)
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:shape"] == [CHIRPS_NY, CHIRPS_NX]
    assert attrs["spatial:transform"] == pytest.approx([CHIRPS_STEP, 0.0, 2.9500027261674404, 0.0, -CHIRPS_STEP, 60.0])
    assert attrs["spatial:bbox"][3] == pytest.approx(60.0)
    assert "GeoTransform" in dict(root["spatial_ref"].attrs)


def _pyramid_cube() -> Any:
    # Past the 2048x2048 pyramid threshold, and deliberately non-square so a transposed
    # shape would be visible rather than symmetric.
    ny, nx = 2100, 2400
    x = 28.8 + np.arange(nx) * 0.001
    y = -1.0 - np.arange(ny) * 0.001
    return _cube(x, y, "EPSG:4326")


def test_pyramid_root_agrees_with_topozarrs_level_0(tmp_path: Path) -> None:
    """We now write the root affine ourselves, overriding topozarr's — so it must match it.

    Same formula (pixel registration, origin on the cell edge), so a mismatch here would mean
    the root metadata contradicts the pyramid layout it sits next to.
    """
    root = _write_icechunk(tmp_path, _pyramid_cube(), "pyr", "EPSG:4326")
    attrs = dict(root.attrs)

    layout = attrs["multiscales"]["layout"]
    assert attrs["spatial:transform"] == pytest.approx(layout[0]["spatial:transform"])
    assert attrs["spatial:shape"] == list(layout[0]["spatial:shape"]) == [2100, 2400]
    assert attrs["spatial:dimensions"] == ["y", "x"]


def test_pyramid_levels_get_their_own_geotransform(tmp_path: Path) -> None:
    """Each level halves the resolution; one GeoTransform copied from level 0 would place
    every overview at twice its true size.

    Checked against topozarr's own per-level ``multiscales.layout`` affine rather than against
    the ``× 2**level`` rule we derive it from — otherwise this only restates our assumption.
    """
    root = _write_icechunk(tmp_path, _pyramid_cube(), "pyr", "EPSG:4326")
    layout = {entry["asset"]: entry["spatial:transform"] for entry in dict(root.attrs)["multiscales"]["layout"]}

    checked = 0
    for key in root.keys():
        group = root[key]
        if not isinstance(group, zarr.Group) or not key.isdigit():
            continue
        origin_x, step_x, rot_x, origin_y, rot_y, step_y = (
            float(v) for v in dict(group["spatial_ref"].attrs)["GeoTransform"].split()
        )
        expected_step_x, _, expected_origin_x, _, expected_step_y, expected_origin_y = layout[key]
        assert step_x == pytest.approx(expected_step_x), f"level {key} x step"
        assert step_y == pytest.approx(expected_step_y), f"level {key} y step"
        assert origin_x == pytest.approx(expected_origin_x), f"level {key} x origin"
        assert origin_y == pytest.approx(expected_origin_y), f"level {key} y origin"
        assert (rot_x, rot_y) == (0.0, 0.0)
        checked += 1

    assert checked == len(layout) >= 2  # every level covered, and there is more than one


# --- the write-time guard ---------------------------------------------------------------
#
# Getting the positional order wrong is silent: the store writes fine and every client that
# trusts the convention transposes the grid. 39 of 68 stores on one machine had it wrong for
# months before a viewer started reading the attributes. The guard refuses the write instead,
# because a wrong claim cannot be un-published.


def test_check_grid_description_accepts_a_correct_description() -> None:
    check_grid_description(
        {"spatial:dimensions": ["y", "x"], "spatial:shape": [60, 581]},
        y_dim="y",
        x_dim="x",
        y_size=60,
        x_size=581,
    )


def test_check_grid_description_rejects_x_first_dimensions() -> None:
    with pytest.raises(ValueError, match="array order"):
        check_grid_description(
            {"spatial:dimensions": ["x", "y"], "spatial:shape": [60, 581]},
            y_dim="y",
            x_dim="x",
            y_size=60,
            x_size=581,
        )


def test_check_grid_description_rejects_a_reversed_shape() -> None:
    with pytest.raises(ValueError, match=r"spatial:shape"):
        check_grid_description(
            {"spatial:dimensions": ["y", "x"], "spatial:shape": [581, 60]},
            y_dim="y",
            x_dim="x",
            y_size=60,
            x_size=581,
        )


def test_check_grid_description_catches_a_transposed_square_grid() -> None:
    """The case a shape check alone misses: [n, n] matches either way round."""
    with pytest.raises(ValueError, match="array order"):
        check_grid_description(
            {"spatial:dimensions": ["x", "y"], "spatial:shape": [1509, 1509]},
            y_dim="y",
            x_dim="x",
            y_size=1509,
            x_size=1509,
        )


def test_streaming_store_refuses_attrs_that_contradict_the_grid(tmp_path: Path) -> None:
    """Through the write path: a plugin's own attrs land last and must not slip past.

    `spec.attrs` is merged after the derived description, so the guard has to run after that
    merge — otherwise a plugin could reintroduce exactly the bug this exists to stop.
    """
    x, y = _chirps_coords()
    store = _init_store(tmp_path, _cube(x, y, "EPSG:4326"))
    spec = GridSpec(
        shape=(CHIRPS_NY, CHIRPS_NX),
        crs=4326,
        dtype=np.dtype("float32"),
        attrs={"spatial:dimensions": ["x", "y"]},
    )

    with pytest.raises(ValueError, match="array order"):
        write_geozarr_attrs(store, spec=spec, bbox=CHIRPS_REQUESTED_BBOX)
