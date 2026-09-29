"""Tests for the aggregate_spatial zonal-statistics plugin process."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.processes.aggregate_spatial import (
    FRACTIONS_DIM,
    _make_reducer_caller,
    _parse_geometries,
    aggregate_spatial,
    reduce_by_method,
)
from open_climate_service.shared.vectors import RESAMPLING_ATTR

# A plain numpy statistic, which the weighted path recognises by identity.
_mean = np.mean


def _named(method: str) -> Any:
    """A reducer shaped like the one a workflow builds: reduce_by_method with a fixed method."""
    return partial(reduce_by_method, method=method)


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _features(*items: tuple[str, dict]) -> dict:
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "id": i, "geometry": g} for i, g in items]}


def _grid(y_ascending: bool) -> xr.DataArray:
    """4x4 grid of unit cells centred on 0..3, where each cell value encodes y*10 + x."""
    xv = np.array([0.0, 1.0, 2.0, 3.0])
    yv = xv.copy() if y_ascending else xv[::-1].copy()
    data = np.array([[yc * 10 + xc for xc in xv] for yc in yv])
    return xr.DataArray(data, dims=("y", "x"), coords={"y": yv, "x": xv}, name="v")


def _categorical(values: np.ndarray, resampling: str = "mode") -> xr.DataArray:
    """A 2x2 class grid centred on 0..1, declared categorical the way load_collection marks it."""
    da = xr.DataArray(values.astype("float64"), dims=("y", "x"), coords={"y": [0.0, 1.0], "x": [0.0, 1.0]}, name="lc")
    da.attrs[RESAMPLING_ATTR] = resampling
    return da


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


def test_parse_geometries_accepts_points_among_polygons() -> None:
    geoms, labels = _parse_geometries(_features(("district-1", _box(0, 0, 1, 1)), ("facility-1", _point(0.5, 0.5))))
    assert labels == ["district-1", "facility-1"]
    assert [g.geom_type for g in geoms] == ["Polygon", "Point"]


def test_parse_geometries_rejects_a_line_by_name() -> None:
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    with pytest.raises(ValueError, match="geometry 'road' is a LineString"):
        _parse_geometries(_features(("district-1", _box(0, 0, 1, 1)), ("road", line)))


def test_parse_geometries_accepts_a_multipolygon() -> None:
    multi = {
        "type": "MultiPolygon",
        "coordinates": [_box(0, 0, 1, 1)["coordinates"], _box(2, 2, 3, 3)["coordinates"]],
    }
    geoms, labels = _parse_geometries({"type": "Feature", "id": "m", "geometry": multi})
    assert labels == ["m"]
    assert geoms[0].geom_type == "MultiPolygon"


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
# Polygons: shape of the result
# ---------------------------------------------------------------------------


def test_aggregate_spatial_dataarray_lower_left_box() -> None:
    """A box symmetric over cells (0,0),(1,0),(0,1),(1,1) weights them equally."""
    da = _grid(y_ascending=True)
    out = aggregate_spatial(da, _box(-0.4, -0.4, 1.4, 1.4), _mean)
    # values {0, 1, 10, 11}, each 0.9 x 0.9 covered -> mean 5.5
    assert float(out["v"].isel(geometry=0)) == pytest.approx(5.5)


def test_aggregate_spatial_orientation_independent() -> None:
    """The same box gives the same weighted result whichever way the axes run."""
    box = _box(-0.5, -0.5, 0.75, 1.5)
    asc = aggregate_spatial(_grid(y_ascending=True), box, _mean)
    desc = aggregate_spatial(_grid(y_ascending=False), box, _mean)
    flipped_x = aggregate_spatial(_grid(y_ascending=True).isel(x=slice(None, None, -1)), box, _mean)
    expected = (0 + 10 + 0.25 * (1 + 11)) / 2.5
    for out in (asc, desc, flipped_x):
        assert float(out["v"].isel(geometry=0)) == pytest.approx(expected)


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
    fc = _features(("lower_left", _box(-0.4, -0.4, 1.4, 1.4)), ("upper_right", _box(1.6, 1.6, 3.4, 3.4)))
    out = aggregate_spatial(da, fc, _mean)
    assert list(out["geometry"].values) == ["lower_left", "upper_right"]
    # upper-right box -> cells {22,23,32,33} -> mean 27.5
    assert float(out["v"].sel(geometry="upper_right")) == pytest.approx(27.5)


# ---------------------------------------------------------------------------
# Polygons: area weighting
# ---------------------------------------------------------------------------


def test_zone_smaller_than_a_cell_takes_the_cell_it_lies_in() -> None:
    """The case the pixel-centre rule returns NaN for: no cell centre falls inside."""
    out = aggregate_spatial(_grid(y_ascending=True), _box(2.1, 1.1, 2.3, 1.3), _mean)
    assert float(out["v"].isel(geometry=0)) == pytest.approx(12.0)


def test_partly_covered_cells_count_by_the_fraction_covered() -> None:
    """One whole cell plus a quarter of its neighbour: the neighbour weighs a quarter."""
    # cell (x=0, y=0) = 0 fully, cell (x=1, y=0) = 1 for x in [0.5, 0.75] -> 0.25 covered
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 0.75, 0.5), _mean)
    assert float(out["v"].isel(geometry=0)) == pytest.approx((0 * 1.0 + 1 * 0.25) / 1.25)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("sum", 0 * 1.0 + 1 * 0.25),  # a flux over the covered area
        ("min", 0.0),
        ("max", 1.0),  # any covered cell counts
        ("median", 0.0),  # the whole cell outweighs the quarter
        ("mean", 0.25 / 1.25),
    ],
)
def test_named_reducers_are_weighted(method: str, expected: float) -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 0.75, 0.5), _named(method))
    assert float(out["v"].isel(geometry=0)) == pytest.approx(expected)


def test_numpy_statistics_are_recognised() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 0.75, 0.5), np.sum)
    assert float(out["v"].isel(geometry=0)) == pytest.approx(0.25)


def test_a_reducer_is_not_recognised_by_its_values() -> None:
    """A function that happens to compute a sum is still run as given, over pixel centres."""
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 0.75, 0.5), lambda data: float(np.sum(data)))
    assert float(out["v"].isel(geometry=0)) == 0.0  # only cell (0, 0) has its centre inside


def test_missing_cells_are_left_out_of_the_weights() -> None:
    da = _grid(y_ascending=True)
    da[0, 1] = np.nan  # cell (x=1, y=0)
    out = aggregate_spatial(da, _box(-0.5, -0.5, 1.5, 0.5), _mean)
    assert float(out["v"].isel(geometry=0)) == pytest.approx(0.0)


def test_unknown_reducer_falls_back_to_pixel_centres(caplog: pytest.LogCaptureFixture) -> None:
    """A reducer the weighted path cannot express keeps the openEO specification's rule."""

    def spread(data: np.ndarray) -> float:
        return float(np.max(data) - np.min(data))

    fc = _features(("big", _box(-0.4, -0.4, 1.4, 1.4)), ("tiny", _box(2.1, 1.1, 2.3, 1.3)))
    with caplog.at_level(logging.INFO):
        out = aggregate_spatial(_grid(y_ascending=True), fc, spread)
    assert float(out["v"].sel(geometry="big")) == 11.0
    assert np.isnan(float(out["v"].sel(geometry="tiny")))
    assert "pixel-centre selection" in caplog.text


def test_zones_that_capture_nothing_are_named_in_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    fc = _features(("inside", _box(0, 0, 1, 1)), ("far_away", _box(100.0, 100.0, 101.0, 101.0)))
    with caplog.at_level(logging.WARNING):
        out = aggregate_spatial(_grid(y_ascending=True), fc, _mean)
    assert np.isnan(float(out["v"].sel(geometry="far_away")))
    assert "1 of 2 geometries captured no data" in caplog.text
    assert "far_away" in caplog.text


# ---------------------------------------------------------------------------
# Categorical data
# ---------------------------------------------------------------------------


def test_mean_over_categorical_data_becomes_the_area_weighted_majority(caplog: pytest.LogCaptureFixture) -> None:
    # classes: row y=0 -> [10, 20], row y=1 -> [20, 90]; averaging them would give 35
    da = _categorical(np.array([[10, 20], [20, 90]]))
    with caplog.at_level(logging.WARNING):
        out = aggregate_spatial(da, _box(-0.5, -0.5, 1.5, 1.5), _named("mean"))
    assert float(out["lc"].isel(geometry=0)) == 20.0  # class 20 covers half the zone
    assert "majority class" in caplog.text


def test_majority_follows_covered_area_not_cell_count() -> None:
    # in row y=0, the class-10 cell is covered 0.1 and the class-30 cell fully
    da = _categorical(np.array([[10, 30], [10, 30]]))
    out = aggregate_spatial(da, _box(0.4, -0.5, 1.5, 0.5), _named("majority"))
    assert float(out["lc"].isel(geometry=0)) == 30.0


def test_fractions_give_each_class_its_share_of_the_zone() -> None:
    da = _categorical(np.array([[10, 20], [20, 30]]))
    out = aggregate_spatial(
        da,
        _features(("whole", _box(-0.5, -0.5, 1.5, 1.5)), ("corner", _box(-0.5, -0.5, 0.5, 0.5))),
        _named("fractions"),
    )
    assert list(out[FRACTIONS_DIM].values) == [10.0, 20.0, 30.0]
    np.testing.assert_allclose(out["lc"].sel(geometry="whole").values, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(out["lc"].sel(geometry="corner").values, [1.0, 0.0, 0.0])


def test_max_over_a_presence_mask_is_kept() -> None:
    da = _categorical(np.array([[0, 1], [0, 0]]), resampling="max")
    out = aggregate_spatial(da, _box(-0.5, -0.5, 1.5, 1.5), _named("max"))
    assert float(out["lc"].isel(geometry=0)) == 1.0


def test_fractions_standalone_are_refused() -> None:
    with pytest.raises(ValueError, match="only works in aggregate_spatial"):
        reduce_by_method(np.array([1.0, 2.0]), method="fractions")


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


def test_point_between_centres_is_interpolated_bilinearly() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _point(1.5, 2.5), _mean)
    assert float(out["v"].isel(geometry=0)) == pytest.approx(26.5)


def test_point_with_near_takes_the_containing_cell() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _point(1.6, 2.4), _mean, method="near")
    assert float(out["v"].isel(geometry=0)) == 22.0


def test_point_on_categorical_data_takes_the_nearest_cell() -> None:
    da = _categorical(np.array([[10, 20], [20, 30]]))
    out = aggregate_spatial(da, _point(0.6, 0.9), _named("mean"))
    assert float(out["lc"].isel(geometry=0)) == 30.0


def test_interpolation_over_categorical_data_warns(caplog: pytest.LogCaptureFixture) -> None:
    da = _categorical(np.array([[10, 20], [20, 30]]))
    with caplog.at_level(logging.WARNING):
        aggregate_spatial(da, _point(0.5, 0.5), _named("max"), method="bilinear")
    assert "interpolation requested over categorical data" in caplog.text


def test_point_near_the_grid_edge_uses_its_cell() -> None:
    """Within half a cell of the edge there is nothing to interpolate towards."""
    out = aggregate_spatial(_grid(y_ascending=True), _point(3.3, 0.0), _mean)
    assert float(out["v"].isel(geometry=0)) == 3.0


def test_point_beside_a_missing_cell_uses_its_cell() -> None:
    """A coastal facility next to a sea cell keeps the land value rather than becoming NaN."""
    da = _grid(y_ascending=True)
    da[1, 2] = np.nan  # (x=2, y=1)
    out = aggregate_spatial(da, _point(1.2, 1.2), _mean)
    assert float(out["v"].isel(geometry=0)) == 11.0


def test_point_outside_the_grid_is_nan() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _point(50.0, 50.0), _mean)
    assert np.isnan(float(out["v"].isel(geometry=0)))


def test_unknown_point_method_is_refused() -> None:
    with pytest.raises(ValueError, match="method 'spline' is not supported"):
        aggregate_spatial(_grid(y_ascending=True), _point(1.0, 1.0), _mean, method="spline")


def test_points_and_polygons_keep_their_input_order() -> None:
    fc = _features(("facility", _point(3.0, 3.0)), ("district", _box(-0.4, -0.4, 1.4, 1.4)))
    out = aggregate_spatial(_grid(y_ascending=True).expand_dims(t=[0, 1]), fc, _mean)
    assert list(out["geometry"].values) == ["facility", "district"]
    np.testing.assert_allclose(out["v"].sel(geometry="facility").values, [33.0, 33.0])
    np.testing.assert_allclose(out["v"].sel(geometry="district").values, [5.5, 5.5])


def test_fractions_over_points_are_refused() -> None:
    da = _categorical(np.array([[10, 20], [20, 30]]))
    with pytest.raises(ValueError, match="fractions apply to polygons"):
        aggregate_spatial(da, _point(0.0, 0.0), _named("fractions"))


def test_unknown_method_is_refused_for_polygons_too() -> None:
    with pytest.raises(ValueError, match="method 'bilinaer' is not supported"):
        aggregate_spatial(_grid(y_ascending=True), _box(0, 0, 1, 1), _mean, method="bilinaer")


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def test_median_of_equal_weights_is_the_midpoint_like_numpy() -> None:
    """Two whole cells of 0 and 1: np.median gives 0.5, and so does the weighted median."""
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 1.5, 0.5), _named("median"))
    assert float(out["v"].isel(geometry=0)) == pytest.approx(0.5)


def _two_class_bands() -> xr.Dataset:
    a = _categorical(np.array([[10, 10], [20, 20]])).rename("a")
    b = _categorical(np.array([[20, 30], [20, 30]])).rename("b")
    return xr.merge([a, b], combine_attrs="drop_conflicts")


def test_fractions_share_one_class_axis_with_zero_for_absent_classes() -> None:
    out = aggregate_spatial(_two_class_bands(), _box(-0.5, -0.5, 1.5, 1.5), _named("fractions"))
    assert list(out[FRACTIONS_DIM].values) == [10.0, 20.0, 30.0]
    np.testing.assert_allclose(out["a"].isel(geometry=0).values, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(out["b"].isel(geometry=0).values, [0.0, 0.5, 0.5])


def test_fractions_of_a_zone_with_only_missing_cells_are_nan() -> None:
    da = _categorical(np.array([[np.nan, 20], [20, 30]]))
    fc = _features(("missing", _box(-0.5, -0.5, 0.5, 0.5)), ("whole", _box(-0.5, -0.5, 1.5, 1.5)))
    out = aggregate_spatial(da, fc, _named("fractions"))
    assert np.isnan(out["lc"].sel(geometry="missing").values).all()
    np.testing.assert_allclose(out["lc"].sel(geometry="whole").values, [2 / 3, 1 / 3])


def test_fractions_with_no_valid_cell_anywhere_return_an_empty_class_axis() -> None:
    da = _categorical(np.full((2, 2), np.nan))
    out = aggregate_spatial(da, _box(-0.5, -0.5, 1.5, 1.5), _named("fractions"))
    assert out["lc"].sizes[FRACTIONS_DIM] == 0


def test_fractions_refuse_continuous_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.processes import aggregate_spatial as module

    monkeypatch.setattr(module, "MAX_FRACTION_CLASSES", 10)
    with pytest.raises(ValueError, match="16 distinct values.*not continuous data"):
        aggregate_spatial(_grid(y_ascending=True), _box(-0.5, -0.5, 3.5, 3.5), _named("fractions"))


def test_cubic_on_a_grid_too_small_is_a_clear_error() -> None:
    da = _categorical(np.array([[10, 20], [20, 30]]), resampling="mean")
    with pytest.raises(ValueError, match="cubic interpolation needs at least 4 cells"):
        aggregate_spatial(da, _point(0.5, 0.5), _mean, method="cubic")


def test_cubic_on_a_large_enough_grid_interpolates() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _point(1.5, 1.5), _mean, method="cubic")
    assert float(out["v"].isel(geometry=0)) == pytest.approx(16.5)


def test_standalone_majority_leaves_missing_values_out() -> None:
    assert reduce_by_method(np.array([np.nan, np.nan, 1.0]), method="majority") == 1.0
    assert np.isnan(reduce_by_method(np.array([np.nan]), method="majority"))


def test_categorical_is_decided_per_variable() -> None:
    """Land cover beside temperature: the codes take the majority, the temperature its mean."""
    temperature = _grid(y_ascending=True).isel(x=slice(0, 2), y=slice(0, 2)).rename("temp")
    land_cover = _categorical(np.array([[10, 20], [20, 90]]))
    cube = xr.merge([temperature, land_cover], combine_attrs="drop_conflicts")
    polygon = aggregate_spatial(cube, _box(-0.5, -0.5, 1.5, 1.5), _named("mean"))
    assert float(polygon["temp"].isel(geometry=0)) == pytest.approx(5.5)
    assert float(polygon["lc"].isel(geometry=0)) == 20.0
    point = aggregate_spatial(cube, _point(0.6, 0.9), _named("mean"))
    assert float(point["temp"].isel(geometry=0)) == pytest.approx(9.6)  # bilinear
    assert float(point["lc"].isel(geometry=0)) == 90.0  # nearest cell


# ---------------------------------------------------------------------------
# Reading in blocks
# ---------------------------------------------------------------------------


def _series(days: int = 7) -> xr.DataArray:
    base = _grid(y_ascending=True)
    return xr.concat([base + 100 * i for i in range(days)], dim="t").assign_coords(t=np.arange(days))


@pytest.mark.parametrize(
    ("cube", "reducer"),
    [
        (_series(), _named("mean")),
        (_series(), _named("median")),
        (_series(), lambda data: float(np.max(data) - np.min(data))),  # pixel-centre fallback
        (xr.concat([_categorical(np.array([[10, 20], [20, 30]]))] * 5, dim="t"), _named("fractions")),
    ],
)
def test_blocked_reads_give_the_same_result(monkeypatch: pytest.MonkeyPatch, cube: xr.DataArray, reducer: Any) -> None:
    """One time step per block gives exactly what one block for the whole series gives."""
    from open_climate_service.plugins.processes import aggregate_spatial as module

    fc = _features(("a", _box(-0.4, -0.4, 1.4, 1.4)), ("b", _box(0.5, 0.5, 1.2, 1.2)))
    whole = aggregate_spatial(cube, fc, reducer)
    monkeypatch.setattr(module, "READ_BLOCK_BYTES", 1)
    blocked = aggregate_spatial(cube, fc, reducer)
    xr.testing.assert_allclose(whole, blocked)


# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------


def _utm_cube() -> xr.DataArray:
    """A 1 km UTM 33N grid over Oslo, values 1 (south-west of 262 km E / 6650 km N) and 2."""
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]

    x = np.arange(255_500.0, 270_000.0, 1_000.0)
    y = np.arange(6_660_500.0, 6_640_000.0, -1_000.0)
    values = np.where((x[None, :] < 262_000) & (y[:, None] < 6_650_000), 1.0, 2.0)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name="tg")
    return da.rio.write_crs("EPSG:32633")


def test_lon_lat_geometries_are_reprojected_into_a_projected_cube() -> None:
    """GeoJSON is WGS 84; a UTM cube must still find the zone, as load_features hands it over."""
    oslo = _box(10.70, 59.88, 10.80, 59.93)  # lon/lat, inside the grid once reprojected
    out = aggregate_spatial(_utm_cube(), oslo, _mean)
    assert not np.isnan(float(out["tg"].isel(geometry=0)))


def test_geometries_already_in_the_cube_crs_are_left_alone() -> None:
    out = aggregate_spatial(_utm_cube(), _box(256_000, 6_641_000, 258_000, 6_643_000), _mean)
    assert float(out["tg"].isel(geometry=0)) == 1.0


def test_a_cube_without_a_crs_takes_geometries_as_given() -> None:
    out = aggregate_spatial(_grid(y_ascending=True), _box(-0.4, -0.4, 1.4, 1.4), _mean)
    assert float(out["v"].isel(geometry=0)) == pytest.approx(5.5)


def test_the_result_keeps_the_supplied_lon_lat_shapes() -> None:
    """The vector writers read `geometry_wkt` as the request's GeoJSON, so it stays WGS 84."""
    from shapely import wkt

    from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD

    oslo = _box(10.70, 59.88, 10.80, 59.93)
    out = aggregate_spatial(_utm_cube(), oslo, _mean)
    minx, miny, maxx, maxy = wkt.loads(str(out[GEOMETRY_WKT_COORD].values[0])).bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((10.70, 59.88, 10.80, 59.93))
