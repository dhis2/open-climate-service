"""Spatial aggregation as the openEO Web Editor builds it.

The editor fills every optional parameter it shows, so `aggregate_spatial` arrives with
`target_dimension: null` and `context: null`, and a reducer it builds is a callback, a child
process graph. It reads `/processes` to decide what to offer for each parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.schemas import OpenEOJobStatus
from tests.test_dhis2_export_workflow import (
    _DATA_ELEMENT,
    _OU_A,
    _OU_B,
    _geometries,
    _run,
    instance,  # noqa: F401  # pyright: ignore[reportUnusedImport]  # the fixture, used by name
)

_MEAN_CALLBACK = {
    "process_graph": {
        "mean1": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True}
    }
}


def _graph(process_id: str, reducer: Any, org_unit_field: str = "geometry", **arguments: Any) -> dict[str, Any]:
    return {
        "process_graph": {
            "load": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "rain_monthly",
                    "spatial_extent": None,
                    "temporal_extent": ["2025-01-01", "2025-02-28"],
                    "bands": None,
                },
            },
            "zonal": {
                "process_id": process_id,
                "arguments": {
                    "data": {"from_node": "load"},
                    "geometries": _geometries(),
                    "reducer": reducer,
                    **arguments,
                },
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {"from_node": "zonal"},
                    "format": "DHIS2JSON",
                    "options": {
                        "data_element_id": _DATA_ELEMENT,
                        "org_unit_field": org_unit_field,
                        "period_type": "month",
                    },
                },
                "result": True,
            },
        }
    }


def _org_units(service: openeo_jobs.OpenEOJobService, job_id: str, graph: dict[str, Any]) -> set[str]:
    record = _run(service, job_id, graph)
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    values = json.loads(Path(str((record.usage or {})["output_path"])).read_text())["dataValues"]
    return {value["orgUnit"] for value in values}


@pytest.mark.parametrize("target_dimension", [None, "regions"])
def test_aggregate_spatial_takes_the_editors_optional_parameters(
    instance: openeo_jobs.OpenEOJobService,  # noqa: F811
    target_dimension: str | None,
) -> None:
    # A graph that names the dimension names it in the export too.
    graph = _graph(
        "aggregate_spatial",
        _MEAN_CALLBACK,
        org_unit_field=target_dimension or "geometry",
        target_dimension=target_dimension,
        context=None,
    )

    assert _org_units(instance, f"core-{target_dimension}", graph) == {_OU_A, _OU_B}


def _values(service: openeo_jobs.OpenEOJobService, job_id: str, graph: dict[str, Any]) -> dict[tuple[str, str], float]:
    record = _run(service, job_id, graph)
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    values = json.loads(Path(str((record.usage or {})["output_path"])).read_text())["dataValues"]
    return {(value["orgUnit"], value["period"]): float(value["value"]) for value in values}


def test_aggregate_spatial_accepts_a_context(instance: openeo_jobs.OpenEOJobService) -> None:  # noqa: F811
    """openEO's `aggregate_spatial` takes a `context`, and a graph that gives one runs, as on main.

    It is bound to the reducer the way openeo-processes-dask's own reducers receive it. That a
    callback cannot read it with `from_parameter` is a limitation of the process graph parser,
    which upstream's `reduce_dimension` shares.
    """
    plain = _values(instance, "core-plain", _graph("aggregate_spatial", _MEAN_CALLBACK))
    with_context = _values(instance, "core-context", _graph("aggregate_spatial", _MEAN_CALLBACK, context={"k": 1}))

    assert with_context == plain


_SUM_CALLBACK = {
    "process_graph": {"sum1": {"process_id": "sum", "arguments": {"data": {"from_parameter": "data"}}, "result": True}}
}


@pytest.mark.parametrize(("reducer", "method"), [(_MEAN_CALLBACK, "mean"), (_SUM_CALLBACK, "sum")])
def test_aggregate_spatial_records_the_method_its_reducer_ran(
    instance: openeo_jobs.OpenEOJobService,  # noqa: F811
    reducer: dict[str, Any],
    method: str,
) -> None:
    """The reducer runs inside the recording scope, not in a worker process that cannot see it."""
    result = execution.run_process_graph(_graph("aggregate_spatial", reducer))

    assert result.provenance["spatial_aggregations"] == [method]


def test_a_named_export_refuses_a_spatial_method_it_does_not_declare(
    instance: openeo_jobs.OpenEOJobService,  # noqa: F811
) -> None:
    """`rain-monthly` declares `aggregation: mean`; a graph that sums is refused, not exported."""
    graph = _graph("aggregate_spatial", _SUM_CALLBACK)
    graph["process_graph"]["save"]["arguments"]["options"] = {"export": "rain-monthly"}
    record = _run(instance, "core-sum-named", graph)

    assert record.status == OpenEOJobStatus.ERROR
    assert "does not match the executed spatial aggregation 'sum'" in str(record.error_message)


def test_aggregate_spatial_weighted_takes_a_named_reducer(instance: openeo_jobs.OpenEOJobService) -> None:  # noqa: F811
    assert _org_units(instance, "weighted", _graph("aggregate_spatial_weighted", "mean")) == {_OU_A, _OU_B}


def test_aggregate_spatial_weighted_refuses_a_reducer_process_by_name(
    instance: openeo_jobs.OpenEOJobService,  # noqa: F811
) -> None:
    record = _run(instance, "weighted-callback", _graph("aggregate_spatial_weighted", _MEAN_CALLBACK))

    assert record.status == OpenEOJobStatus.ERROR
    assert "reducer must be one of mean, sum, min, max, median, by name" in str(record.error_message)


def test_the_editor_is_offered_named_statistics_for_the_weighted_reducer(client: TestClient) -> None:
    processes = {p["id"]: p for p in client.get("/processes").json()["processes"]}
    parameters = {p["name"]: p for p in processes["aggregate_spatial_weighted"]["parameters"]}

    assert parameters["reducer"]["schema"] == {"type": "string", "enum": ["mean", "sum", "min", "max", "median"]}
    assert parameters["data"]["schema"]["subtype"] == "datacube"
    assert {schema["subtype"] for schema in parameters["geometries"]["schema"]} == {"datacube", "geojson"}
