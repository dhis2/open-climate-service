"""Feature ids survive spatial aggregation as `feature_id`, and the exports read them (CLIM-1355).

openEO labels a vector cube's geometry dimension with the geometries and leaves the feature id to
the backend (openeo-processes#466); OCS keeps it in a `feature_id` coordinate. The graphs below run
as jobs with two aggregations: OCS's own `aggregate_spatial`, which labels the dimension with the
ids, and the openEO reference implementation from openeo-processes-dask, which labels it with the
shapes, as CLIM-785's processes do. Both must export the same org units and locations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

gpd = pytest.importorskip("geopandas")
pytest.importorskip("xvec")

from openeo_pg_parser_networkx.process_registry import Process  # noqa: E402
from shapely.geometry import box  # noqa: E402

from open_climate_service.exports.delivery_input import lease_export_input  # noqa: E402
from open_climate_service.exports.service import render_named_export  # noqa: E402
from open_climate_service.openeo import execution  # noqa: E402
from open_climate_service.openeo import jobs as openeo_jobs  # noqa: E402
from open_climate_service.openeo.schemas import OpenEOJobStatus  # noqa: E402
from open_climate_service.shared.vectors import (  # noqa: E402
    FEATURE_ID_COORD,
    attach_feature_ids,
    geometries_to_frame,
)
from tests.test_dhis2_export_workflow import (  # noqa: E402
    _DATA_ELEMENT,
    _OU_A,
    _OU_B,
    _geometries,
    _process,
    _run,
    instance,  # noqa: F401  # pyright: ignore[reportUnusedImport]  # the fixture, used by name
)


def _openeo_aggregate_spatial(data: Any, geometries: Any, reducer: Any, **_: Any) -> Any:
    """openeo-processes-dask's `aggregate_spatial`, keeping the ids the way CLIM-785 does."""
    from openeo_processes_dask.process_implementations.cubes.aggregate import aggregate_spatial as reference

    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326")
    frame = geometries_to_frame(geometries)
    # Annotated as openEO's VectorCube union; for a raster input it is a DataArray.
    result = cast(xr.DataArray, reference(data=data, geometries=frame, reducer=reducer))
    return attach_feature_ids(result, frame.index, "geometry")


@pytest.fixture(params=["ocs", "openeo"])
def service(
    request: pytest.FixtureRequest,
    instance: openeo_jobs.OpenEOJobService,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> openeo_jobs.OpenEOJobService:
    """The job service, aggregating with OCS's process or with the openEO reference one."""
    if request.param == "openeo":
        registry = execution._build_process_registry()
        overlay = execution._RegistryOverlay(
            registry, {"aggregate_spatial": Process(spec={}, implementation=_openeo_aggregate_spatial)}
        )
        monkeypatch.setattr(execution, "_build_process_registry", lambda: overlay)
    return instance


def _graph(fmt: str, options: dict[str, Any]) -> dict[str, Any]:
    return {
        "process_graph": {
            "load": {
                "process_id": "load_collection",
                "arguments": {"id": "rain_monthly", "temporal_extent": ["2025-01-01", "2025-02-28"]},
            },
            "zonal": {
                "process_id": "aggregate_spatial",
                "arguments": {
                    "data": {"from_node": "load"},
                    "geometries": _geometries(),
                    "reducer": {
                        "process_graph": {
                            "mean": {
                                "process_id": "mean",
                                "arguments": {"data": {"from_parameter": "data"}},
                                "result": True,
                            }
                        }
                    },
                },
            },
            "save": {
                "process_id": "save_result",
                "arguments": {"data": {"from_node": "zonal"}, "format": fmt, "options": options},
                "result": True,
            },
        }
    }


def _output(service: openeo_jobs.OpenEOJobService, job_id: str, graph: dict[str, Any]) -> Path:
    record = _run(service, job_id, graph)
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    return Path(str((record.usage or {})["output_path"]))


def test_dhis2_json_exports_the_feature_ids(service: openeo_jobs.OpenEOJobService) -> None:
    options = {"data_element_id": _DATA_ELEMENT, "org_unit_field": "geometry", "period_type": "month"}
    values = json.loads(_output(service, "dhis2-json", _graph("DHIS2JSON", options)).read_text())["dataValues"]

    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}
    assert len(values) == 4


def test_chap_csv_exports_the_feature_ids(service: openeo_jobs.OpenEOJobService) -> None:
    options = {"location_field": "geometry", "period_type": "monthly"}
    with _output(service, "chap-csv", _graph("CHAPCSV", options)).open() as handle:
        rows = list(csv.DictReader(handle))

    assert {row["location"] for row in rows} == {_OU_A, _OU_B}
    assert len(rows) == 4


def test_a_named_dhis2_export_reads_the_feature_ids(instance: openeo_jobs.OpenEOJobService) -> None:  # noqa: F811
    record = _run(instance, "named", _process(_geometries()))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message

    with lease_export_input("rain-monthly", "named") as verified:
        values = json.loads(verified.content)["dataValues"]

    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}


def test_a_named_dhis2_export_reads_ids_beside_shape_labels(instance: openeo_jobs.OpenEOJobService) -> None:  # noqa: F811
    """The cube openEO's aggregation returns: shapes on the dimension, ids in `feature_id`.

    Rendered directly rather than through the workflow, whose `reduce_by_method` reducer reduces
    every dimension and so does not fit the reference implementation; that is CLIM-785's to settle.
    """
    cube = xr.Dataset(
        {"tp": (("t", "geometry"), [[3.0, 16.0], [23.0, 36.0]])},
        coords={
            "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
            "geometry": [box(0.5, 0.5, 2.5, 2.5), box(3.5, 2.5, 5.5, 4.5)],
        },
        attrs={"period_type": "monthly"},
    )
    cube = attach_feature_ids(cube.xvec.set_geom_indexes("geometry", crs=4326), [_OU_A, _OU_B], "geometry")

    _, rendered = render_named_export(cube, "DHIS2JSON", {"export": "rain-monthly"})

    values = json.loads(rendered.content)["dataValues"]
    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}
    assert len(values) == 4


def test_a_geodataframe_keeps_its_index_as_the_feature_ids() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2)], index=["WEST", "EAST"], crs=4326)

    parsed = geometries_to_frame(frame)

    assert parsed.index.tolist() == ["WEST", "EAST"]
    assert parsed.index.name == FEATURE_ID_COORD
    assert parsed.crs == frame.crs


def test_a_vector_cube_keeps_its_feature_ids() -> None:
    cube = xr.Dataset(coords={"geometry": [box(0, 0, 2, 2), box(2, 0, 4, 2)]})
    cube = attach_feature_ids(cube.xvec.set_geom_indexes("geometry", crs=4326), ["WEST", "EAST"], "geometry")

    parsed = geometries_to_frame(cube)

    assert parsed.index.tolist() == ["WEST", "EAST"]
    assert parsed.geometry.tolist() == [box(0, 0, 2, 2), box(2, 0, 4, 2)]


def test_a_feature_without_an_id_gets_its_position() -> None:
    assert geometries_to_frame(_geometries(first_id=None)).index.tolist() == ["0", _OU_B]
    assert geometries_to_frame(_geometries()["features"][0]["geometry"]).index.tolist() == ["0"]
