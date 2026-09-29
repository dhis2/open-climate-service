"""aggregate_spatial through a real process graph: load_collection, the reducer openEO builds,
and the provenance a named export checks (CLIM-785)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.openeo import execution


def _cube() -> xr.Dataset:
    """A 2x2 grid of unit cells centred on 0..1, one time step, values 10, 20 / 20, 90."""
    ds = xr.Dataset(
        {"v": (("t", "y", "x"), np.array([[[10.0, 20.0], [20.0, 90.0]]]))},
        coords={"t": np.array(["2025-01-01"], dtype="datetime64[ns]"), "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    ds.attrs["proj:code"] = "EPSG:4326"
    return ds


@pytest.fixture()
def dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A published dataset whose template the test can make categorical."""
    template: dict[str, Any] = {"id": "cube", "variable": "v"}
    monkeypatch.setattr(execution, "_get_published_artifact", lambda _id: object())
    monkeypatch.setattr(execution, "_open_artifact", lambda _a: _cube())
    monkeypatch.setattr(
        "open_climate_service.data_registry.services.datasets.get_dataset",
        lambda dataset_id: template if dataset_id == "cube" else None,
    )
    return template


def _run(geometry: dict, method: str) -> Any:
    graph = {
        "load": {"process_id": "load_collection", "arguments": {"id": "cube"}},
        "zonal": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "load"},
                "geometries": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "id": "z", "geometry": geometry}],
                },
                "reducer": {
                    "process_graph": {
                        "r": {
                            "process_id": "reduce_by_method",
                            "arguments": {"data": {"from_parameter": "data"}, "method": method},
                            "result": True,
                        }
                    }
                },
            },
        },
        "save": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "zonal"}, "format": "JSON"},
            "result": True,
        },
    }
    return execution.run_process_graph({"process_graph": graph})


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}


def test_sub_cell_zone_gets_a_value_through_the_graph(dataset: dict[str, Any]) -> None:
    envelope = _run(_box(0.1, 0.1, 0.2, 0.2), "mean")
    assert float(envelope.data["v"].sel(geometry="z").item()) == 10.0
    assert envelope.provenance["spatial_aggregations"] == ["mean"]


def test_categorical_template_aggregates_by_majority_through_the_graph(dataset: dict[str, Any]) -> None:
    dataset["ingestion"] = {"resampling": "mode"}
    envelope = _run(_box(-0.5, -0.5, 1.5, 1.5), "mean")
    assert float(envelope.data["v"].sel(geometry="z").item()) == 20.0  # not the mean, 35
    # The export's declared aggregation is checked against what ran, which was a majority.
    assert envelope.provenance["spatial_aggregations"] == ["majority"]


def test_continuous_template_keeps_the_weighted_mean(dataset: dict[str, Any]) -> None:
    envelope = _run(_box(-0.5, -0.5, 1.5, 1.5), "mean")
    assert float(envelope.data["v"].sel(geometry="z").item()) == 35.0


def test_point_is_interpolated_through_the_graph(dataset: dict[str, Any]) -> None:
    envelope = _run({"type": "Point", "coordinates": [0.5, 0.5]}, "mean")
    assert float(envelope.data["v"].sel(geometry="z").item()) == pytest.approx(35.0)
    # Sampled, not reduced: an export declaring `mean` must not accept it as one.
    assert envelope.provenance["spatial_aggregations"] == [None]


def _run_with(geometry: dict, reducer_graph: dict) -> Any:
    graph = {
        "load": {"process_id": "load_collection", "arguments": {"id": "cube"}},
        "zonal": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "load"},
                "geometries": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "id": "z", "geometry": geometry}],
                },
                "reducer": {"process_graph": reducer_graph},
            },
        },
        "save": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "zonal"}, "format": "JSON"},
            "result": True,
        },
    }
    return execution.run_process_graph({"process_graph": graph})


_DATA = {"from_parameter": "data"}
_SUB_CELL = _box(0.1, 0.1, 0.2, 0.2)
_WHOLE = _box(-0.5, -0.5, 1.5, 1.5)


def test_openeo_mean_is_weighted(dataset: dict[str, Any]) -> None:
    graph = {"m": {"process_id": "mean", "arguments": {"data": _DATA}, "result": True}}
    envelope = _run_with(_SUB_CELL, graph)
    assert float(envelope.data["v"].sel(geometry="z").item()) == 10.0
    assert envelope.provenance["spatial_aggregations"] == ["mean"]


def test_a_named_reduction_followed_by_more_work_runs_as_given(dataset: dict[str, Any]) -> None:
    """reduce_by_method(mean) times 2 is not a mean; the multiplication must not be skipped."""
    graph = {
        "m": {"process_id": "reduce_by_method", "arguments": {"data": _DATA, "method": "mean"}},
        "x": {"process_id": "multiply", "arguments": {"x": {"from_node": "m"}, "y": 2}, "result": True},
    }
    whole = _run_with(_WHOLE, graph)
    assert float(whole.data["v"].sel(geometry="z").item()) == 70.0
    # Pixel-centre, as the specification has it: a sub-cell zone captures no centre.
    assert np.isnan(float(_run_with(_SUB_CELL, graph).data["v"].sel(geometry="z").item()))
    assert whole.provenance["spatial_aggregations"] == [None]


def test_a_statistic_capped_by_another_step_runs_as_given(dataset: dict[str, Any]) -> None:
    graph = {
        "m": {"process_id": "mean", "arguments": {"data": _DATA}},
        "c": {"process_id": "clip", "arguments": {"x": {"from_node": "m"}, "min": 0, "max": 10}, "result": True},
    }
    envelope = _run_with(_WHOLE, graph)
    assert float(envelope.data["v"].sel(geometry="z").item()) == 10.0  # the mean 35, capped


def test_mean_counting_missing_values_is_not_weighted(dataset: dict[str, Any]) -> None:
    graph = {"m": {"process_id": "mean", "arguments": {"data": _DATA, "ignore_nodata": False}, "result": True}}
    envelope = _run_with(_SUB_CELL, graph)
    assert np.isnan(float(envelope.data["v"].sel(geometry="z").item()))


def _with_missing_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cube with its 90 cell missing, so a whole-grid zone mixes valid and missing cells."""
    cube = _cube()
    cube["v"][0, 1, 1] = np.nan
    monkeypatch.setattr(execution, "_open_artifact", lambda _a: cube)


def test_mean_counting_missing_values_propagates_a_missing_cell(
    dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_missing_cell(monkeypatch)
    graph = {"m": {"process_id": "mean", "arguments": {"data": _DATA, "ignore_nodata": False}, "result": True}}
    assert np.isnan(float(_run_with(_WHOLE, graph).data["v"].sel(geometry="z").item()))


def test_a_graph_ignoring_missing_values_still_skips_them(
    dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_missing_cell(monkeypatch)
    graph = {
        "m": {"process_id": "mean", "arguments": {"data": _DATA}},
        "x": {"process_id": "multiply", "arguments": {"x": {"from_node": "m"}, "y": 2}, "result": True},
    }
    assert float(_run_with(_WHOLE, graph).data["v"].sel(geometry="z").item()) == pytest.approx(2 * 50 / 3)
