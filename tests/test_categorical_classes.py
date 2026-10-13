"""Categorical datasets (CLIM-895): declared classes, and zonal statistics per class.

Grids are small and synthetic so every expected number can be worked out by hand: unit cells,
classes laid out in halves or quadrants, and boxes that cover whole cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.data_manager.services.downloader import resampling_method_from_template
from open_climate_service.plugins.processes.aggregate_spatial_classes import aggregate_spatial_classes
from open_climate_service.plugins.processes.aggregate_spatial_weighted import aggregate_spatial_weighted
from open_climate_service.shared.cf import cf_attrs_from_template
from open_climate_service.shared.classes import (
    classification_from_template,
    cube_classes,
    stac_classes,
    stamp_classes,
)

HEAT_LEVELS: dict[str, Any] = {
    "id": "heat_levels",
    "classes": [
        {"value": 0, "label": "No warning", "color": "#ffffb2"},
        {"value": 1, "label": "Caution (35-38 °C)", "color": "#fecc5c"},
        {"value": 2, "label": "Warning (38-40 °C)", "color": "#fd8d3c"},
        {"value": 3, "label": "Danger (40 °C and above)", "color": "#e31a1c"},
    ],
    "display": {"nodata": 255},
}


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    ring = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    return {"type": "Polygon", "coordinates": [ring]}


def _grid(values: list[list[int]], *, cell: float = 1.0, name: str = "level") -> xr.DataArray:
    """A grid of cells `cell` wide from (0, 0), rows listed north to south, in WGS 84."""
    rows, cols = len(values), len(values[0])
    x = (np.arange(cols) + 0.5) * cell
    y = ((np.arange(rows) + 0.5) * cell)[::-1]
    raster = xr.DataArray(np.array(values), dims=("y", "x"), coords={"y": y, "x": x}, name=name)
    return raster.rio.write_crs("EPSG:4326")


def _heat(values: list[list[int]], **kwargs: Any) -> xr.DataArray:
    return stamp_classes(_grid(values, **kwargs), "heat_levels", classification_from_template(HEAT_LEVELS))


# --- declaring classes -------------------------------------------------------------------------


def test_a_template_declares_classes_with_labels_colours_and_a_nodata_code() -> None:
    classification = classification_from_template(HEAT_LEVELS)

    assert classification is not None
    assert classification.values == [0, 1, 2, 3]
    assert classification.labels[1] == "Caution (35-38 °C)"
    assert classification.nodata == 255
    assert classification_from_template({"id": "t2m"}) is None


@pytest.mark.parametrize(
    ("classes", "nodata", "message"),
    [
        ([{"value": 0, "label": "A"}, {"value": 0, "label": "B"}], None, "more than once"),
        ([{"value": 1.5, "label": "A"}], None, "integer code"),
        ([{"value": 1, "label": " "}], None, "non-empty"),
        ([{"value": 1, "label": "A", "color": "red"}], None, "hex colour"),
        ([{"value": 0, "label": "A"}], 0, "also declared as a class"),
        ([], None, "non-empty list"),
    ],
)
def test_a_malformed_declaration_is_refused(classes: list[Any], nodata: int | None, message: str) -> None:
    template: dict[str, Any] = {"classes": classes, "display": {"nodata": nodata}}

    with pytest.raises(ValueError, match=message):
        classification_from_template(template)


def test_the_classes_reach_the_store_as_cf_flags() -> None:
    attrs = cf_attrs_from_template(HEAT_LEVELS)

    assert attrs["flag_values"] == [0, 1, 2, 3]
    assert attrs["flag_meanings"] == "No_warning Caution_35-38_C Warning_38-40_C Danger_40_C_and_above"


def test_the_classes_reach_stac_with_the_nodata_code_marked() -> None:
    entries = stac_classes(classification_from_template(HEAT_LEVELS))  # type: ignore[arg-type]

    assert entries[1] == {
        "value": 1,
        "name": "Caution_35-38_C",
        "description": "Caution (35-38 °C)",
        "color_hint": "FECC5C",
    }
    assert entries[-1] == {"value": 255, "name": "nodata", "nodata": True}


def test_a_categorical_dataset_is_pyramided_by_mode_unless_it_says_otherwise() -> None:
    assert resampling_method_from_template(HEAT_LEVELS) == "mode"
    assert resampling_method_from_template({**HEAT_LEVELS, "ingestion": {"resampling": "max"}}) == "max"
    assert resampling_method_from_template({"id": "t2m"}) == "mean"


# --- statistics per class ----------------------------------------------------------------------


def test_share_is_the_fraction_of_each_geometry_in_each_class() -> None:
    # West half level 0, east half level 2; the nodata cell (255) counts towards no class.
    heat = _heat([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 255], [0, 0, 2, 2]])

    out = aggregate_spatial_classes(heat, _box(0, 0, 4, 4), "share")

    shares = dict(zip(out["class"].values.tolist(), out.isel(geometry=0).values.tolist(), strict=True))
    assert shares[0] == pytest.approx(8 / 15)
    assert shares[2] == pytest.approx(7 / 15)
    assert shares[1] == pytest.approx(0) and shares[3] == pytest.approx(0)
    assert out["class_label"].values.tolist() == [
        "No_warning",
        "Caution_35-38_C",
        "Warning_38-40_C",
        "Danger_40_C_and_above",
    ]


def test_majority_is_the_class_covering_most_of_each_geometry() -> None:
    heat = _heat([[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 1, 3], [1, 1, 1, 1]])

    out = aggregate_spatial_classes(
        heat,
        {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": "whole", "properties": {}, "geometry": _box(0, 0, 4, 4)},
                {"type": "Feature", "id": "north_east", "properties": {}, "geometry": _box(2, 2, 4, 4)},
            ],
        },
        "majority",
    )

    assert out.values.tolist() == [1, 3]
    assert out["feature_id"].values.tolist() == ["whole", "north_east"]


def test_total_sums_people_per_level_on_the_population_grid() -> None:
    """The Malawi question: how many people live at each heat level, per district.

    Heat on a coarse 2-degree grid, people on a 1-degree grid, as WorldPop at 100 m sits under
    a hazard at several km. Each person counts towards the level of the hazard cell above them.
    """
    heat = _heat([[0, 3], [1, 2]], cell=2.0)  # quadrants, north row first
    people = _grid([[10, 20, 30, 40], [50, 60, 70, 80], [1, 2, 3, 4], [5, 6, 7, 8]], name="population")

    out = aggregate_spatial_classes(heat, _box(0, 0, 4, 4), "total", weights=people)

    totals = dict(zip(out["class"].values.tolist(), out.isel(geometry=0).values.tolist(), strict=True))
    assert totals == pytest.approx({0: 10 + 20 + 50 + 60, 3: 30 + 40 + 70 + 80, 1: 1 + 2 + 5 + 6, 2: 3 + 4 + 7 + 8})
    assert out.name == "level"


def test_total_keeps_the_time_axis_of_the_levels() -> None:
    heat = stamp_classes(
        xr.concat([_grid([[0, 0], [0, 0]], cell=2.0), _grid([[2, 2], [2, 2]], cell=2.0)], dim="t").assign_coords(
            t=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
        ),
        "heat_levels",
        classification_from_template(HEAT_LEVELS),
    )
    people = _grid([[1, 1, 1, 1]] * 4, name="population")

    out = aggregate_spatial_classes(heat, _box(0, 0, 4, 4), "total", weights=people)

    assert out.sel({"class": 0}).isel(geometry=0).values.tolist() == pytest.approx([16, 0])
    assert out.sel({"class": 2}).isel(geometry=0).values.tolist() == pytest.approx([0, 16])


def test_total_needs_weights_and_refuses_a_weight_series() -> None:
    heat = _heat([[0, 1], [2, 3]], cell=2.0)
    series = xr.concat([_grid([[1] * 4] * 4), _grid([[1] * 4] * 4)], dim="year")

    with pytest.raises(ValueError, match="needs `weights`"):
        aggregate_spatial_classes(heat, _box(0, 0, 4, 4), "total")
    with pytest.raises(ValueError, match="must be one grid"):
        aggregate_spatial_classes(heat, _box(0, 0, 4, 4), "total", weights=series)


def test_a_continuous_cube_is_refused_by_the_class_process() -> None:
    with pytest.raises(ValueError, match="'t2m' declares no classes"):
        aggregate_spatial_classes(_grid([[1, 2], [3, 4]], name="t2m"), _box(0, 0, 2, 2), "share")


def test_a_categorical_dataset_is_never_averaged() -> None:
    """Land cover 10 and 50 average to a third class; the weighted aggregation refuses instead."""
    heat = _heat([[0, 3], [0, 3]])

    with pytest.raises(ValueError, match="'heat_levels' is a categorical dataset, so 'mean'"):
        aggregate_spatial_weighted(heat, _box(0, 0, 2, 2), "mean")


def test_loading_a_collection_stamps_its_declared_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the load path, so a store written before its classes were declared counts too."""
    from open_climate_service.data_registry.services import datasets as registry
    from open_climate_service.openeo.execution import stamp_declared_cadence

    monkeypatch.setattr(registry, "get_dataset", lambda _id: {**HEAT_LEVELS, "period_type": "daily"})
    cube = stamp_declared_cadence(_grid([[0, 1], [2, 3]]), "heat_levels")

    assert cube_classes(cube) == (
        [0, 1, 2, 3],
        ["No_warning", "Caution_35-38_C", "Warning_38-40_C", "Danger_40_C_and_above"],
    )
    assert cube.attrs["ocs_nodata"] == 255


@pytest.mark.parametrize("template", [HEAT_LEVELS, {"id": "heat_levels"}])
def test_a_published_collection_carries_its_classes_in_stac(
    client: Any, monkeypatch: pytest.MonkeyPatch, template: dict[str, Any]
) -> None:
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.shared.classes import STAC_CLASSIFICATION_EXTENSION
    from open_climate_service.stac import services as stac_services

    from .test_stac import _artifact, _minimal_xstac_payload

    raster = _artifact(artifact_id="a1", dataset_id="heat_levels")
    monkeypatch.setattr(ingestion_services, "stac_eligible_artifacts_by_dataset", lambda: {"heat_levels": raster})
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", lambda **_: _minimal_xstac_payload())
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {**template, "period_type": "daily"})
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {})

    doc = client.get("/stac/collections/heat_levels").json()

    if "classes" in template:
        classes = doc["assets"]["zarr"]["classification:classes"]
        assert [entry["value"] for entry in classes] == [0, 1, 2, 3, 255]
        assert STAC_CLASSIFICATION_EXTENSION in doc["stac_extensions"]
    else:
        assert "classification:classes" not in doc["assets"]["zarr"]
        assert STAC_CLASSIFICATION_EXTENSION not in doc["stac_extensions"]
