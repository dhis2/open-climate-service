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
