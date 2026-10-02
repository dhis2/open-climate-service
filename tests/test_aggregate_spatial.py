"""Tests for the aggregate_spatial zonal-statistics plugin process."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.processes.aggregate_spatial import (
    _make_reducer_caller,
    _parse_geometries,
    aggregate_spatial,
)


def _mean(data: np.ndarray) -> float:
    return float(np.mean(data))


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _grid(y_ascending: bool) -> xr.DataArray:
    """4x4 grid where each cell value encodes its coordinate: y*10 + x."""
    xv = np.array([0.0, 1.0, 2.0, 3.0])
    yv = xv.copy() if y_ascending else xv[::-1].copy()
    data = np.array([[yc * 10 + xc for xc in xv] for yc in yv])
    return xr.DataArray(data, dims=("y", "x"), coords={"y": yv, "x": xv}, name="v")


# ---------------------------------------------------------------------------
# _parse_geometries
# ---------------------------------------------------------------------------


def test_parse_geometries_feature_collection_uses_ids() -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "a", "geometry": _box(0, 0, 1, 1)},
            {"type": "Feature", "geometry": _box(0, 0, 1, 1)},
        ],
    }
    geoms, labels = _parse_geometries(fc)
    assert len(geoms) == 2
    assert labels == ["a", "1"]  # explicit id, then positional fallback


def test_parse_geometries_single_feature_and_geometry() -> None:
    _, labels_feat = _parse_geometries({"type": "Feature", "id": "x", "geometry": _box(0, 0, 1, 1)})
    assert labels_feat == ["x"]
    _, labels_geom = _parse_geometries(_box(0, 0, 1, 1))
    assert labels_geom == ["0"]


def test_parse_geometries_rejects_a_point() -> None:
    """No defined sampling semantics exist for a point yet (CLIM-785) -- refuse rather than
    silently rasterize to whichever pixel happens to contain it."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "facility-1", "geometry": _point(0.5, 0.5)},
        ],
    }
    with pytest.raises(ValueError, match="geometry 'facility-1' is a Point.*CLIM-785"):
        _parse_geometries(fc)


def test_parse_geometries_rejects_a_point_among_polygons() -> None:
    """A single bad geometry in an otherwise-valid hierarchy must name itself, not the batch."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "district-1", "geometry": _box(0, 0, 1, 1)},
            {"type": "Feature", "id": "facility-1", "geometry": _point(0.5, 0.5)},
        ],
    }
    with pytest.raises(ValueError, match="geometry 'facility-1' is a Point"):
        _parse_geometries(fc)


def test_parse_geometries_accepts_a_multipolygon() -> None:
    multi = {
        "type": "MultiPolygon",
        "coordinates": [_box(0, 0, 1, 1)["coordinates"], _box(2, 2, 3, 3)["coordinates"]],
    }
    geoms, labels = _parse_geometries({"type": "Feature", "id": "m", "geometry": multi})
    assert labels == ["m"]
    assert geoms[0].geom_type == "MultiPolygon"


def test_aggregate_spatial_rejects_a_point_before_touching_the_raster() -> None:
    da = _grid(y_ascending=True)
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "p", "geometry": _point(0.5, 0.5)}]}
    with pytest.raises(ValueError, match="geometry 'p' is a Point"):
        aggregate_spatial(da, fc, _mean)


# ---------------------------------------------------------------------------
# _make_reducer_caller
# ---------------------------------------------------------------------------


def test_reducer_caller_returns_nan_for_empty() -> None:
    call = _make_reducer_caller(lambda data: _mean(data), None)
    assert np.isnan(call(np.array([])))


def test_reducer_caller_forwards_context_when_supported() -> None:
    def reducer(data: np.ndarray, context: dict) -> float:
        return float(np.mean(data)) + context["offset"]

    call = _make_reducer_caller(reducer, {"offset": 100.0})
    assert call(np.array([1.0, 3.0])) == 102.0


def test_reducer_caller_skips_context_for_plain_reducer() -> None:
    # A reducer that only accepts data must not receive context.
    call = _make_reducer_caller(lambda data: _mean(data), {"offset": 100.0})
    assert call(np.array([1.0, 3.0])) == 2.0


# ---------------------------------------------------------------------------
# aggregate_spatial
# ---------------------------------------------------------------------------


def test_aggregate_spatial_dataarray_lower_left_box() -> None:
    """A box over the geographic lower-left selects cells (0,0),(1,0),(0,1),(1,1)."""
    da = _grid(y_ascending=True)
    out = aggregate_spatial(da, _box(-0.4, -0.4, 1.4, 1.4), _mean)
    # values {0, 1, 10, 11} -> mean 5.5
    assert float(out["v"].isel(geometry=0)) == 5.5


def test_aggregate_spatial_orientation_independent() -> None:
    """Same geographic box yields the same result for ascending and descending y."""
    asc = aggregate_spatial(_grid(y_ascending=True), _box(-0.4, -0.4, 1.4, 1.4), _mean)
    desc = aggregate_spatial(_grid(y_ascending=False), _box(-0.4, -0.4, 1.4, 1.4), _mean)
    assert float(asc["v"].isel(geometry=0)) == float(desc["v"].isel(geometry=0)) == 5.5


def test_aggregate_spatial_with_time_dimension() -> None:
    base = _grid(y_ascending=True)
    ds = xr.concat([base, base + 100], dim="t").assign_coords(t=[0, 1]).to_dataset(name="v")
    out = aggregate_spatial(ds, _box(-0.4, -0.4, 1.4, 1.4), _mean)
    assert list(out["t"].values) == [0, 1]
    np.testing.assert_allclose(out["v"].isel(geometry=0).values, [5.5, 105.5])


def test_aggregate_spatial_preserves_non_spatial_dimensions() -> None:
    cube1 = xr.concat([_grid(y_ascending=True), _grid(y_ascending=True) + 100], dim="t").assign_coords(t=[0, 1])
    cube2 = cube1 + 1000
    merged = xr.concat([cube1, cube2], dim="__cubes__").assign_coords(__cubes__=["cube1", "cube2"])
    out = aggregate_spatial(merged, _box(-0.4, -0.4, 1.4, 1.4), _mean)
    assert list(out["__cubes__"].values) == ["cube1", "cube2"]
    assert list(out["t"].values) == [0, 1]
    np.testing.assert_allclose(out["v"].sel(__cubes__="cube1", geometry="0").values, [5.5, 105.5])
    np.testing.assert_allclose(out["v"].sel(__cubes__="cube2", geometry="0").values, [1005.5, 1105.5])


def test_aggregate_spatial_multiple_geometries_labelled() -> None:
    da = _grid(y_ascending=True)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "lower_left", "geometry": _box(-0.4, -0.4, 1.4, 1.4)},
            {"type": "Feature", "id": "upper_right", "geometry": _box(1.6, 1.6, 3.4, 3.4)},
        ],
    }
    out = aggregate_spatial(da, fc, _mean)
    assert list(out["geometry"].values) == ["lower_left", "upper_right"]
    # upper-right box -> cells {22,23,32,33} -> mean 27.5
    assert float(out["v"].sel(geometry="upper_right")) == 27.5


def test_aggregate_spatial_empty_geometry_returns_nan() -> None:
    """A geometry that selects no pixels yields NaN rather than raising."""
    da = _grid(y_ascending=True)
    out = aggregate_spatial(da, _box(100.0, 100.0, 101.0, 101.0), _mean)
    assert np.isnan(float(out["v"].isel(geometry=0)))
