"""Vector output from `aggregate_spatial`: the requested format, with real geometry.

`save_result(format="PARQUET")` on an aggregation result used to write `result.zarr`. The
geometry dimension carries feature *ids*, so parsing them as WKT threw; the failure was logged
at debug level and the raster writer took over, silently substituting the format. These tests
pin both halves of the fix: the shapes survive the aggregation, and the requested format is
honoured or an error is raised — never quietly swapped.
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

from open_climate_service.openeo import jobs
from open_climate_service.plugins.processes.aggregate_spatial import aggregate_spatial
from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD

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
    return aggregate_spatial(_grid(), _districts(), _mean)


def _write(ds: xr.Dataset, results_dir: Path, fmt: str) -> Path:
    """`jobs._write_raster`, asserting it wrote something, returning the path."""
    written = jobs._write_raster(ds, results_dir, fmt)
    assert written is not None
    return Path(written)


# --- the shapes survive the aggregation --------------------------------------------------


def test_aggregation_keeps_the_shapes_alongside_the_labels() -> None:
    result = _result()

    # The labels stay the feature ids -- the DHIS2 and CHAP exports key on them.
    assert list(result.geometry.values) == ["MW.N", "MW.S"]
    # ...and the shapes ride alongside as WKT.
    assert all(text.startswith("POLYGON") for text in result[GEOMETRY_WKT_COORD].values)
    # north (30+20)/2, south (10+0)/2, for both time steps. Selected by label rather than by
    # position: the assertion should not depend on how concat happens to order the dimensions.
    assert result.t2m.sel(geometry="MW.N").values.tolist() == [25.0, 25.0]
    assert result.t2m.sel(geometry="MW.S").values.tolist() == [5.0, 5.0]


# --- the requested format is honoured ----------------------------------------------------


def test_parquet_output_is_geoparquet_with_real_geometry(tmp_path: Path) -> None:
    """The bug: PARQUET on a vector cube silently wrote a Zarr store instead."""
    written = _write(_result(), tmp_path, "PARQUET")
    assert written.suffix == ".parquet"
    assert written.is_file()

    frame = gpd.read_parquet(written)
    assert len(frame) == 4  # two districts x two time steps
    assert sorted(frame.geom_type.unique()) == ["Polygon"]
    assert sorted(set(frame["geometry_id"])) == ["MW.N", "MW.S"]
    assert [float(v) for v in frame.total_bounds] == [0.0, 0.0, 4.0, 4.0]


def test_geojson_output_carries_the_geometry_too(tmp_path: Path) -> None:
    written = _write(_result(), tmp_path, "GEOJSON")
    assert written.suffix == ".geojson"
    assert sorted(gpd.read_file(written).geom_type.unique()) == ["Polygon"]


@pytest.mark.parametrize(("fmt", "suffix"), [("ZARR", ".zarr"), ("NETCDF", ".nc"), ("CSV", ".csv")])
def test_non_vector_formats_are_still_honoured_for_a_vector_cube(tmp_path: Path, fmt: str, suffix: str) -> None:
    """A vector cube asked for a non-vector format must not be diverted to vector output.

    The first attempt at the fix inverted the bug: ZARR on a vector cube returned GeoJSON.
    """
    results_dir = tmp_path / fmt
    results_dir.mkdir()
    assert _write(_result(), results_dir, fmt).suffix == suffix


def test_a_vector_format_with_no_usable_geometry_raises(tmp_path: Path) -> None:
    """The failure that used to be swallowed at debug level must now surface.

    A cube with a geometry dimension but no shapes anywhere cannot produce GeoParquet, and a
    caller asking for it is asking for the shapes -- returning a Zarr directory instead left
    them with a file their reader could not open and no reason logged above debug.
    """
    result = _result().drop_vars(GEOMETRY_WKT_COORD)
    # ValueError specifically: it is what the sync route turns into a 400. Shapely's own parse
    # error would surface as a 500 and blame the server for a cube that has no shapes.
    with pytest.raises(ValueError, match="no usable geometry"):
        jobs._write_raster(result, tmp_path, "PARQUET")


def test_csv_needs_no_shapes(tmp_path: Path) -> None:
    """CSV is a vector format by listing only: it never carries geometry, so must not demand it."""
    result = _result().drop_vars(GEOMETRY_WKT_COORD)
    columns = _write(result, tmp_path, "CSV").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "geometry" in columns
    assert "t2m" in columns


def test_a_null_geometry_is_rejected_rather_than_borrowed(tmp_path: Path) -> None:
    """`pd.factorize` codes a null as -1, and `parsed[-1]` is the *last* polygon, not a missing one."""
    result = _result()
    wkt = result[GEOMETRY_WKT_COORD].values.astype(object)
    wkt[0] = None
    result = result.assign_coords({GEOMETRY_WKT_COORD: ("geometry", wkt)})
    with pytest.raises(ValueError, match="no geometry"):
        jobs._write_raster(result, tmp_path, "PARQUET")


def test_a_custom_target_dimension_is_still_a_vector_cube(tmp_path: Path) -> None:
    """`aggregate_spatial(target_dimension="regions")` names the dimension; the cube is no less vector for it.

    The writer used to recognise a vector cube by the name `geometry` alone, so PARQUET on a
    `regions` cube reported it as a raster with no geometry.
    """
    result = aggregate_spatial(_grid(), _districts(), _mean, target_dimension="regions")
    assert "regions" in result.dims

    frame = gpd.read_parquet(_write(result, tmp_path, "PARQUET"))
    assert sorted(frame.geom_type.unique()) == ["Polygon"]
    # The label column keeps the dimension's name: nothing else is competing for `regions`.
    assert sorted(set(frame["regions"])) == ["MW.N", "MW.S"]
    assert "geometry_id" not in frame.columns


# --- the carrier must not leak into tabular output ----------------------------------------


def test_csv_output_does_not_leak_the_wkt_carrier_column(tmp_path: Path) -> None:
    """CSV keeps the columns it always had: the WKT coordinate is an internal carrier.

    The tabular exports identify their value column by elimination, so a stray coordinate
    either becomes a bogus value column or makes the export refuse the cube.
    """
    header = _write(_result(), tmp_path, "CSV").read_text(encoding="utf-8").splitlines()[0]
    columns = header.split(",")
    assert GEOMETRY_WKT_COORD not in columns
    # Exact, not a substring: "geometry" in header also matches "geometry_id", so the looser
    # assertion passed while the location column was silently renamed.
    assert "geometry" in columns, f"the labels are the location column; got {columns}"
    assert "t2m" in columns


def test_dhis2_export_still_finds_its_single_value_column() -> None:
    """The new coordinate must not become an extra value-column candidate.

    `_select_dhis2_value_field` picks by elimination, so an unexcluded coordinate made it
    refuse an otherwise valid cube with "found ['t2m', 'geometry_wkt']".
    """
    frame = _result().to_dataframe().reset_index()
    assert jobs._select_dhis2_value_field(frame, "geometry", "time") == "t2m"


def test_chap_export_still_finds_its_value_columns() -> None:
    frame = _result().to_dataframe().reset_index()
    assert jobs._select_chap_value_fields(frame, "geometry", "time") == ["t2m"]
