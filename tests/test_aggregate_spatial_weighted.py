from datetime import date
from typing import Any

import numpy as np
import pytest
import shapely
import xarray as xr

from open_climate_service.plugins.processes.aggregate_spatial_weighted import (
    aggregate_spatial_weighted,
)


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }


def _flat_grid(y_ascending: bool) -> xr.DataArray:
    """A flat 4x4 grid of unit cells centred on 0..3, where all cell values are 1."""
    # NOTE: xy coords are centerpoints so we offset by 0.5
    xv = np.array([0.0, 1.0, 2.0, 3.0]) + 0.5
    yv = xv.copy() if y_ascending else xv[::-1].copy()
    data = np.array([[1 for _ in xv] for _ in yv])
    return xr.DataArray(data, dims=("y", "x"), coords={"y": yv, "x": xv}, name="v")


# Test input geometries


def test_aggregate_spatial_single_geometry():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 1x1 pixels at partial pixels, but only one-fourth inside
    geom = _box(-0.5, -0.5, 0.5, 0.5)

    # weighted sum = polygon area / 4
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert out.sizes["geometry"] == 1
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area / 4)


def test_aggregate_spatial_feature_collection():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # feature collection
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"coverage": "one-fourth"},
                "geometry": _box(-0.5, -0.5, 0.5, 0.5),
            },
            {
                "type": "Feature",
                "properties": {"coverage": "one-half"},
                "geometry": _box(-0.5, 0, 0.5, 1),
            },
        ],
    }

    # weighted sum
    out = aggregate_spatial_weighted(da, collection, "sum")
    assert out.sizes["geometry"] == 2
    assert float(out.isel(geometry=0).item()) == pytest.approx(
        shapely.geometry.shape(collection["features"][0]["geometry"]).area / 4
    )
    assert float(out.isel(geometry=1).item()) == pytest.approx(
        shapely.geometry.shape(collection["features"][1]["geometry"]).area / 2
    )


# Test weighted sum and mean on uniform grid of only 1s


def test_aggregate_spatial_uniform_1x1_aligned():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 1x1 pixels aligned with pixels
    geom = _box(0, 0, 1, 1)

    # weighted sum = polygon area
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_2x2_aligned():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 2x2 pixels aligned with pixels
    geom = _box(0, 0, 2, 2)

    # weighted sum = polygon area
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_2x2_aligned_half_inside():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 2x2 pixels aligned with pixels, but only half inside
    geom = _box(-1, 0, 1, 2)

    # weighted sum = polygon area / 2
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area / 2)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_1x1_partial_half_inside():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 1x1 pixels at partial pixels, but only half inside
    geom = _box(-0.5, 0, 0.5, 1)

    # weighted sum = polygon area / 2
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area / 2)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_1x1_partial_fourth_inside():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 1x1 pixels at partial pixels, but only one-fourth inside
    geom = _box(-0.5, -0.5, 0.5, 0.5)

    # weighted sum = polygon area / 4
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area / 4)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_tiny_inside_one_pixel():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 0.1x0.1 pixels (one tenth area) entirely inside one pixel corner
    geom = _box(0.1, 0.1, 0.2, 0.2)

    # weighted sum = polygon area
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(shapely.geometry.shape(geom).area)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


def test_aggregate_spatial_uniform_realistic_poly():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # "realistic" irregular polygon that is both inside and outside
    # and covers several pixels
    geom = shapely.geometry.Polygon(
        [
            (-1.2, -1.1),
            (2.7, 0.4),
            (1.3, 2.6),
        ]
    )

    # weighted sum = intersecting polygon area
    out = aggregate_spatial_weighted(da, geom.__geo_interface__, "sum")
    raster_extent = shapely.geometry.box(*da.rio.bounds())
    expected = geom.intersection(raster_extent).area
    assert float(out.isel(geometry=0).item()) == pytest.approx(expected)

    # weighted mean is always 1
    out = aggregate_spatial_weighted(da, geom.__geo_interface__, "mean")
    assert float(out.isel(geometry=0).item()) == pytest.approx(1)


# Test non results


def test_aggregate_spatial_fully_outside():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # exactly 1x1 pixels aligned with pixels
    geom = _box(-2, -2, -1, -1)

    # returns geometry even if outside
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert out.sizes["geometry"] == 1

    # weighted sum for nomatch = 0
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert float(out.isel(geometry=0).item()) == pytest.approx(0)

    # weighted mean for nomatch = nan
    out = aggregate_spatial_weighted(da, geom, "mean")
    assert np.isnan(out.isel(geometry=0))


def test_aggregate_spatial_mixed_geometry_types():
    # uniform flat grid with only 1s
    da = _flat_grid(y_ascending=True)

    # polygon, linestring, point
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": _box(-0.5, -0.5, 0.5, 0.5),  # one-fourth overlap
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [(1, 1), (2, 2), (3, 1)]},
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": (1, 1)},
            },
        ],
    }

    # exactextract should raise error for unsupported geometry types (points, lines)
    with pytest.raises(ValueError):
        aggregate_spatial_weighted(da, collection, "sum")


# Test return structure


def test_aggregate_spatial_with_time_dimension() -> None:
    geom = _box(-0.5, -0.5, 0.5, 0.5)  # covering one-fourth pixel
    base = _flat_grid(y_ascending=True)
    years = [2025, 2026]
    t_coords = [date(year=y, month=1, day=1) for y in years]
    ds = xr.concat([base, base + 100], dim="t").assign_coords(t=t_coords).to_dataset(name="v")
    out = aggregate_spatial_weighted(ds, geom, "sum")
    assert isinstance(out, xr.DataArray)
    assert list(out["t"].values) == t_coords
    np.testing.assert_allclose(out.isel(geometry=0).values, [1 / 4, (1 + 100) / 4])


def test_aggregate_spatial_preserves_non_spatial_dimensions() -> None:
    da = xr.concat(
        [
            _flat_grid(y_ascending=True),
            _flat_grid(y_ascending=True) * 2,
            _flat_grid(y_ascending=True) * 3,
        ],
        dim="bands",
    ).assign_coords(bands=["r", "g", "b"])
    geom = _box(-0.5, -0.5, 0.5, 0.5)  # covering one-fourth pixel
    out = aggregate_spatial_weighted(da, geom, "sum")
    assert isinstance(out, xr.DataArray)
    assert list(out["bands"].values) == ["r", "g", "b"]
    assert float(out.isel(geometry=0).sel(bands="r").item()) == pytest.approx(0.25)
    assert float(out.isel(geometry=0).sel(bands="g").item()) == pytest.approx(0.5)
    assert float(out.isel(geometry=0).sel(bands="b").item()) == pytest.approx(0.75)


def _series(steps: int = 6) -> xr.DataArray:
    """A 4x4 raster over `steps` days, each day a different constant, chunked two days deep."""
    data = np.arange(steps, dtype="float64")[:, None, None] * np.ones((steps, 4, 4))
    da = xr.DataArray(
        data,
        dims=("t", "y", "x"),
        coords={"t": np.arange(steps), "y": np.arange(3.5, -0.5, -1), "x": np.arange(0.5, 4.5, 1)},
        name="tg",
    )
    return da.rio.write_crs("EPSG:4326").chunk({"t": 2})


def test_a_long_series_is_read_in_blocks_and_gives_the_same_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.processes import aggregate_spatial_weighted as module

    geom = _box(0.0, 0.0, 2.0, 2.0)
    whole = aggregate_spatial_weighted(_series(), geom, "mean")
    # One day is 128 bytes; a 300-byte bound reads two days at a time, one chunk per block.
    monkeypatch.setattr(module, "READ_BLOCK_BYTES", 300)
    assert module._blocks(_series())[1] == [slice(0, 2), slice(2, 4), slice(4, 6)]

    blocked = aggregate_spatial_weighted(_series(), geom, "mean")
    assert blocked.sizes["t"] == 6
    assert blocked.isel(geometry=0).values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    xr.testing.assert_identical(blocked, whole)


def test_blocks_group_chunks_and_split_one_that_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.processes import aggregate_spatial_weighted as module

    # A day is 128 bytes, so 700 bytes holds five days.
    monkeypatch.setattr(module, "READ_BLOCK_BYTES", 700)
    assert module._blocks(_series(12).chunk({"t": 2}))[1] == [slice(0, 4), slice(4, 8), slice(8, 12)]
    assert module._blocks(_series(12).chunk({"t": 12}))[1] == [slice(0, 5), slice(5, 10), slice(10, 12)]
    assert module._blocks(_series().isel(t=0))[1] == [slice(None)]
