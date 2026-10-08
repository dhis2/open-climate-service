"""Vector output from a spatial aggregation: the requested format, with real geometry.

The aggregation returns an xvec cube, as openEO describes one: the shapes on `geometry`, each
feature's id beside them as `feature_id`. The writers then honour the requested format, encode
the shapes where the format cannot hold objects, or raise; they never quietly swap the format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr
import xvec  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .xvec
from shapely.geometry import box

from open_climate_service.openeo import jobs
from open_climate_service.plugins.processes.aggregate_spatial_weighted import aggregate_spatial_weighted

_NORTH = [[0, 2], [4, 2], [4, 4], [0, 4], [0, 2]]
_SOUTH = [[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]


def _districts() -> dict[str, Any]:
    """Two stacked boxes over (0,0)-(4,4), labelled as an org-unit code would be."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "MW.N",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [_NORTH]},
            },
            {
                "type": "Feature",
                "id": "MW.S",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [_SOUTH]},
            },
        ],
    }


def _grid() -> xr.Dataset:
    """A 4x4 two-step cube whose rows are 30/20/10/0 from north to south."""
    rows = np.arange(4.0)[::-1] * 10
    data = np.tile(rows[:, None], (1, 4))[None, :, :].repeat(2, axis=0)
    return xr.Dataset(
        {"t2m": (("time", "y", "x"), data, {"units": "degC"})},
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            "y": np.arange(0.5, 4.5, 1.0)[::-1],
            "x": np.arange(0.5, 4.5, 1.0),
        },
    ).rio.write_crs("EPSG:4326")


def _mean(data: Any) -> float:
    return float(np.mean(data))


def _result() -> xr.Dataset:
    """The aggregation as a job hands it to the writers: a Dataset named after the variable."""
    result = aggregate_spatial_weighted(_grid(), _districts(), "mean")
    return result.to_dataset(name=result.name)


def _write(ds: xr.Dataset, results_dir: Path, fmt: str) -> Path:
    """`jobs._write_raster`, asserting it wrote something, returning the path."""
    written = jobs._write_raster(ds, results_dir, fmt)
    assert written is not None
    return Path(written)


# --- the shapes survive the aggregation --------------------------------------------------


def test_aggregation_returns_an_xvec_cube_with_the_feature_ids() -> None:
    result = aggregate_spatial_weighted(_grid(), _districts(), "mean")

    # A DataArray named after the input variable, as openeo-processes-dask returns them.
    assert isinstance(result, xr.DataArray)
    assert result.name == "t2m"
    # The shapes on `geometry` under xvec's index, which carries their CRS...
    assert type(result.xindexes["geometry"]).__name__ == "GeometryIndex"
    assert getattr(result.xindexes["geometry"], "crs").to_epsg() == 4326
    assert all(shape.geom_type == "Polygon" for shape in result.geometry.values)
    # ...and the ids beside them -- the DHIS2 and CHAP exports key on them.
    assert list(result.feature_id.values) == ["MW.N", "MW.S"]
    # north (30+20)/2, south (10+0)/2, for both time steps, read by id rather than by position.
    by_id = dict(zip(result.feature_id.values, result.transpose("geometry", "time").values.tolist(), strict=True))
    assert by_id == {"MW.N": [25.0, 25.0], "MW.S": [5.0, 5.0]}


# --- the requested format is honoured ----------------------------------------------------


def test_parquet_output_is_geoparquet_with_real_geometry(tmp_path: Path) -> None:
    """The bug: PARQUET on a vector cube silently wrote a Zarr store instead."""
    written = _write(_result(), tmp_path, "PARQUET")
    assert written.suffix == ".parquet"
    assert written.is_file()

    frame = gpd.read_parquet(written)
    assert len(frame) == 4  # two districts x two time steps
    assert sorted(frame.geom_type.unique()) == ["Polygon"]
    assert sorted(set(frame["feature_id"])) == ["MW.N", "MW.S"]
    assert [float(v) for v in frame.total_bounds] == [0.0, 0.0, 4.0, 4.0]


def test_geojson_output_carries_the_geometry_too(tmp_path: Path) -> None:
    written = _write(_result(), tmp_path, "GEOJSON")
    assert written.suffix == ".geojson"
    assert sorted(gpd.read_file(written).geom_type.unique()) == ["Polygon"]


def _projected_cube() -> xr.Dataset:
    """Two 1 km squares in UTM 33N, carried the way an xvec cube carries shapes.

    The aggregations produce this for a projected raster: the features are reprojected to the
    raster's CRS, and `_vector_crs` reads it off the GeometryIndex.
    """
    return xr.Dataset(
        {"t2m": (("geometry", "t"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "geometry": [box(500000, 6000000, 501000, 6001000), box(501000, 6000000, 502000, 6001000)],
            "t": pd.date_range("2024-01-01", periods=2),
        },
    ).xvec.set_geom_indexes("geometry", crs="EPSG:32633")


def test_geojson_output_is_reprojected_to_wgs84(tmp_path: Path) -> None:
    """RFC 7946 fixes GeoJSON to WGS 84, and the format has no CRS field to say otherwise.

    Written as-is, a projected cube produced a .geojson of eastings and northings that every
    reader takes for degrees.
    """
    frame = gpd.read_file(_write(_projected_cube(), tmp_path, "GEOJSON"))

    assert frame.crs is not None and frame.crs.to_epsg() == 4326
    assert [round(float(v), 3) for v in frame.total_bounds] == [15.0, 54.148, 15.031, 54.157]


def test_geoparquet_output_keeps_the_native_crs(tmp_path: Path) -> None:
    """The other half of the rule: GeoParquet records the CRS, so the coordinates stay as they are."""
    frame = gpd.read_parquet(_write(_projected_cube(), tmp_path, "PARQUET"))

    assert frame.crs is not None and frame.crs.to_epsg() == 32633
    assert [float(v) for v in frame.total_bounds] == [500000.0, 6000000.0, 502000.0, 6001000.0]


@pytest.mark.parametrize(("fmt", "suffix"), [("ZARR", ".zarr"), ("NETCDF", ".nc"), ("CSV", ".csv")])
def test_non_vector_formats_are_still_honoured_for_a_vector_cube(tmp_path: Path, fmt: str, suffix: str) -> None:
    """A vector cube asked for a non-vector format must not be diverted to vector output.

    The first attempt at the fix inverted the bug: ZARR on a vector cube returned GeoJSON.
    """
    results_dir = tmp_path / fmt
    results_dir.mkdir()
    assert _write(_result(), results_dir, fmt).suffix == suffix


@pytest.mark.parametrize("fmt", ["ZARR", "NETCDF"])
def test_zarr_and_netcdf_store_the_shapes_as_cf_geometry(tmp_path: Path, fmt: str) -> None:
    """Shapely objects cannot be written to either; CF geometry can, and decodes back with the CRS."""
    written = _write(_result(), tmp_path, fmt)
    stored = xr.open_zarr(written) if fmt == "ZARR" else xr.open_dataset(written)

    decoded = stored.load().xvec.decode_cf()
    assert getattr(decoded.xindexes["geometry"], "crs").to_epsg() == 4326
    assert [shape.bounds for shape in decoded.geometry.values] == [(0.0, 2.0, 4.0, 4.0), (0.0, 0.0, 4.0, 2.0)]
    assert list(decoded.feature_id.values) == ["MW.N", "MW.S"]


def test_a_vector_format_with_no_usable_geometry_raises(tmp_path: Path) -> None:
    """The failure that used to be swallowed at debug level must now surface.

    A cube with a geometry dimension but no shapes anywhere cannot produce GeoParquet, and a
    caller asking for it is asking for the shapes -- returning a Zarr directory instead left
    them with a file their reader could not open and no reason logged above debug.
    """
    result = _result().drop_indexes("geometry").assign_coords(geometry=["MW.N", "MW.S"])
    # ValueError specifically: it is what the sync route turns into a 400. Shapely's own parse
    # error would surface as a 500 and blame the server for a cube that has no shapes.
    with pytest.raises(ValueError, match="no usable geometry"):
        jobs._write_raster(result, tmp_path, "PARQUET")


def test_csv_needs_no_shapes(tmp_path: Path) -> None:
    """CSV is a vector format by listing only: it never carries geometry, so must not demand it."""
    result = _result().drop_indexes("geometry").assign_coords(geometry=["MW.N", "MW.S"])
    columns = _write(result, tmp_path, "CSV").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "geometry" in columns
    assert "t2m" in columns


def test_a_null_geometry_is_rejected_rather_than_borrowed(tmp_path: Path) -> None:
    """`pd.factorize` codes a null as -1, and `parsed[-1]` is the *last* polygon, not a missing one."""
    result = _result()
    shapes = result.geometry.values.astype(object)
    shapes[0] = None
    result = result.drop_indexes("geometry").assign_coords(geometry=shapes)
    with pytest.raises(ValueError, match="no geometry"):
        jobs._write_raster(result, tmp_path, "PARQUET")


def test_a_custom_target_dimension_is_still_a_vector_cube(tmp_path: Path) -> None:
    """A cube whose features are on `regions` rather than `geometry` is no less vector for it.

    The writer used to recognise a vector cube by the name `geometry` alone, so PARQUET on a
    `regions` cube reported it as a raster with no geometry.
    """
    result = _result().rename({"geometry": "regions"})
    assert "regions" in result.dims

    frame = gpd.read_parquet(_write(result, tmp_path, "PARQUET"))
    assert sorted(frame.geom_type.unique()) == ["Polygon"]
    assert sorted(set(frame["feature_id"])) == ["MW.N", "MW.S"]


# --- tabular output -------------------------------------------------------------------------


def test_csv_output_keys_on_the_feature_id_without_the_shapes(tmp_path: Path) -> None:
    """A table has no place for polygons: the CSV carries each feature's id and its values.

    The tabular exports identify their value column by elimination, so a stray column either
    becomes a bogus value column or makes the export refuse the cube.
    """
    header = _write(_result(), tmp_path, "CSV").read_text(encoding="utf-8").splitlines()[0]
    columns = header.split(",")
    assert "geometry" not in columns
    assert "feature_id" in columns
    assert "t2m" in columns


def test_dhis2_export_still_finds_its_single_value_column() -> None:
    """`feature_id` must not become an extra value-column candidate.

    `_select_dhis2_value_field` picks by elimination, so an unexcluded coordinate makes it
    refuse an otherwise valid cube.
    """
    frame = _result().to_dataframe().reset_index()
    assert jobs._select_dhis2_value_field(frame, "geometry", "time") == "t2m"


def test_chap_export_still_finds_its_value_columns() -> None:
    frame = _result().to_dataframe().reset_index()
    assert jobs._select_chap_value_fields(frame, "geometry", "time") == ["t2m"]


def test_every_renderer_excludes_the_same_non_value_columns() -> None:
    """The named-export plugin picks its value column by elimination too, from its own copy of
    the list. Three copies drifted apart once already, so the CHAP path accepted a cube the
    DHIS2JSON path refused. One shared set, one test."""
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
    from open_climate_service.exports.tabular import _build_dhis2_json_payload

    frame = _result().to_dataframe().reset_index()
    options = {
        "data_element_id": "DE123",
        "org_unit_field": "geometry",
        "period_field": "time",
        "period_type": "daily",
    }

    assert Dhis2ExportPlugin._candidate_value_fields(frame, None, "geometry", "time") == ["t2m"]
    payload = _build_dhis2_json_payload(frame, options)
    assert payload["dataValues"], "the plugin renderer refused a cube the CHAP path accepts"


def test_csv_of_a_raster_keeps_its_unlabelled_axes(tmp_path: Path) -> None:
    """Only a vector cube's shapeless dimension is dropped; a raster's positional axes stay."""
    raster = xr.Dataset({"t2m": (("y", "x"), np.arange(4.0).reshape(2, 2))})
    columns = _write(raster, tmp_path, "CSV").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert columns == ["y", "x", "t2m"]


# --- explicit ids from a GeoDataFrame or a vector cube --------------------------------------

_NAMED_DHIS2_EXPORT = {
    "process_graph": {
        "save": {
            "process_id": "save_result",
            "arguments": {"format": "DHIS2JSON", "options": {"export": "rainfall"}},
            "result": True,
        }
    }
}
_UIDS = ["ImspTQPwCqd", "O6uvpzGd5pu"]


def _frame(index: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(0, 2, 4, 4), box(0, 0, 4, 2)], index=index, crs="EPSG:4326")


@pytest.mark.parametrize(
    "geometries",
    [
        pytest.param(lambda: _frame(pd.Index(_UIDS)), id="geodataframe"),
        pytest.param(lambda: aggregate_spatial_weighted(_grid(), _frame(pd.Index(_UIDS)), "mean"), id="vector-cube"),
    ],
)
def test_a_named_dhis2_export_accepts_ids_a_frame_or_cube_carries(geometries: Any) -> None:
    from open_climate_service.shared.provenance import capture_execution

    with capture_execution(_NAMED_DHIS2_EXPORT) as evidence:
        result = aggregate_spatial_weighted(_grid(), geometries(), "mean")

    assert list(result.feature_id.values) == _UIDS
    assert evidence.features[-1]["ids_valid"] is True
    assert evidence.features[-1]["input_sha256"] is not None


def test_a_named_dhis2_export_still_refuses_positional_ids() -> None:
    """A GeoDataFrame's default 0, 1, 2 index is a position, not an org unit."""
    from open_climate_service.shared.provenance import capture_execution

    with capture_execution(_NAMED_DHIS2_EXPORT), pytest.raises(ValueError, match="explicit feature identifiers"):
        aggregate_spatial_weighted(_grid(), _frame(pd.RangeIndex(2)), "mean")


# --- the built-in aggregate_spatial, on a projected raster ------------------------------------


def _projected_grid() -> xr.DataArray:
    """A UTM 33N raster with a scalar grid-mapping coordinate, as a seNorge store has one."""
    raster = _grid().t2m.assign_coords(
        x=500000 + _grid().x.values * 1000, y=6000000 + _grid().y.values * 1000, UTM_Zone_33=0
    )
    return raster.rio.write_crs("EPSG:32633")


def test_the_builtin_keeps_the_crs_and_drops_the_grid_mapping() -> None:
    """openEO's own aggregate_spatial hands xvec a bare list of shapes, so its index had no CRS
    (GeoParquet then claimed WGS 84 for UTM metres), and it kept the raster's scalar grid-mapping
    coordinate, which the DHIS2 export then took for a second value column."""
    from open_climate_service.openeo import execution

    aggregate_spatial = execution._build_process_registry()["aggregate_spatial"].implementation
    districts = gpd.GeoDataFrame(
        geometry=[box(500000, 6002000, 504000, 6004000), box(500000, 6000000, 504000, 6002000)],
        index=pd.Index(["MW.N", "MW.S"]),
        crs="EPSG:32633",
    )

    def mean(data: Any, axis: Any = None, **_: Any) -> Any:
        return np.nanmean(data, axis=axis)

    result = aggregate_spatial(data=_projected_grid(), geometries=districts, reducer=mean)

    assert getattr(result.xindexes["geometry"], "crs").to_epsg() == 32633
    assert "UTM_Zone_33" not in result.coords and "spatial_ref" not in result.coords
    frame = result.to_dataset().to_dataframe().reset_index()
    assert jobs._select_dhis2_value_field(frame, "geometry", "time") == "t2m"
