"""Tests for the built-in org-unit aggregation workflows and UDP parameter substitution."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.process_registry import Process

from open_climate_service.openeo import jobs
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.openeo.execution import (
    _augment_with_workflows,
    _build_process_registry,
    _resolve_workflow_parameters,
)

# ---------------------------------------------------------------------------
# _resolve_workflow_parameters
# ---------------------------------------------------------------------------


def test_resolve_workflow_parameters_substitutes_declared_refs() -> None:
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": {"from_parameter": "dataset_id"}}},
        "save": {
            "process_id": "save_result",
            "arguments": {"options": {"period_type": {"from_parameter": "period_type"}}},
        },
    }
    out = _resolve_workflow_parameters(pg, {"dataset_id": "ds1", "period_type": "month"})
    assert out["load"]["arguments"]["id"] == "ds1"
    assert out["save"]["arguments"]["options"]["period_type"] == "month"


def test_resolve_workflow_parameters_leaves_callback_refs_untouched() -> None:
    # A reducer's `data` callback parameter is not a workflow parameter and must be left
    # for the engine to resolve at call time.
    pg = {"mean": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}}}
    out = _resolve_workflow_parameters(pg, {"dataset_id": "ds1"})
    assert out["mean"]["arguments"]["data"] == {"from_parameter": "data"}


# ---------------------------------------------------------------------------
# End-to-end workflow execution (UDP path) with a mocked load_collection
# ---------------------------------------------------------------------------


def _overlay_with_mock_dataset() -> Any:
    ds = xr.Dataset(
        {"tp": (("t", "y", "x"), np.arange(2 * 4 * 5, dtype="float32").reshape(2, 4, 5))},
        coords={
            "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    reg = _build_process_registry()
    reg["load_collection"] = Process(spec={}, implementation=lambda id=None, temporal_extent=None, **k: ds["tp"])
    return _augment_with_workflows(reg)


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]],
    }


_GEOMETRIES = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": "OU_A", "geometry": _box(0.5, 0.5, 2.5, 2.5)},
        {"type": "Feature", "id": "OU_B", "geometry": _box(3.5, 2.5, 5.5, 4.5)},
    ],
}


def test_both_workflows_registered_as_processes() -> None:
    ids = {wf.id for wf in workflow_store.list_workflows().processes}
    assert {"aggregate_to_dhis2_json", "aggregate_to_chap_csv"} <= ids


def test_aggregate_to_org_units_dhis2json() -> None:
    overlay = _overlay_with_mock_dataset()
    # Mapping and period type come from the named export, so the workflow forwards only its ID.
    envelope = overlay["aggregate_to_dhis2_json"].implementation(
        dataset_id="my_dataset",
        temporal_extent=["2025-01-01", "2025-02-28"],
        geometries=_GEOMETRIES,
        export="rain-monthly",
    )
    assert envelope.format == "DHIS2JSON"
    assert envelope.options == {"export": "rain-monthly"}
    frame = envelope.data.to_dataframe().reset_index()
    assert set(frame["geometry"]) == {"OU_A", "OU_B"}
    assert len(frame) == 4


def test_aggregate_to_org_units_chap_csv() -> None:
    overlay = _overlay_with_mock_dataset()
    envelope = overlay["aggregate_to_chap_csv"].implementation(
        dataset_id="my_dataset",
        temporal_extent=["2025-01-01", "2025-02-28"],
        geometries=_GEOMETRIES,
        period_type="month",
    )
    assert envelope.format == "CHAPCSV"
    frame = jobs._build_chap_csv_frame(envelope.data.to_dataframe().reset_index(), envelope.options)
    rows = list(csv.DictReader(StringIO(frame.to_csv(index=False))))
    assert list(frame.columns) == ["time_period", "location", "tp"]
    assert {r["location"] for r in rows} == {"OU_A", "OU_B"}
    assert {r["time_period"] for r in rows} == {"202501", "202502"}


# OU_A covers pixels with t=0 values [0, 1, 5, 6] -> mean 3, sum 12, min 0, max 6.
@pytest.mark.parametrize(("method", "expected"), [("mean", 3.0), ("sum", 12.0), ("min", 0.0), ("max", 6.0)])
def test_method_selects_reducer(method: str, expected: float) -> None:
    overlay = _overlay_with_mock_dataset()
    envelope = overlay["aggregate_to_dhis2_json"].implementation(
        dataset_id="my_dataset",
        temporal_extent=["2025-01-01", "2025-02-28"],
        geometries=_GEOMETRIES,
        export="rain-monthly",
        method=method,
    )
    frame = envelope.data.to_dataframe().reset_index()
    ou_a_jan = frame[(frame["geometry"] == "OU_A") & (frame["t"] == np.datetime64("2025-01-01"))]["tp"]
    assert ou_a_jan.tolist() == [expected]


def test_builtin_workflows_use_literal_process_ids() -> None:
    """openEO requires `process_id` to be a literal string; a `{from_parameter}` process_id
    is non-portable (standard validators reject it) even though OCS resolves it at runtime.
    Guard against reintroducing the pattern in any built-in workflow — use a process that
    takes the choice as an argument (e.g. reduce_by_method) instead."""
    import importlib.resources
    import json

    def _non_literal_process_ids(node: Any, path: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            if isinstance(node.get("process_id"), dict):
                found.append(path or "<root>")
            for key, value in node.items():
                found += _non_literal_process_ids(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                found += _non_literal_process_ids(value, f"{path}[{i}]")
        return found

    pkg = importlib.resources.files("open_climate_service") / "plugins" / "workflows"
    checked = 0
    for resource in pkg.iterdir():
        if not resource.name.endswith(".json"):
            continue
        checked += 1
        workflow = json.loads(resource.read_text(encoding="utf-8"))
        offenders = _non_literal_process_ids(workflow)
        assert not offenders, f"{resource.name} has non-literal process_id(s) at: {offenders}"
    assert checked > 0  # ensure we actually scanned the packaged workflows
