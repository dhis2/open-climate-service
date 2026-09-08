"""Tests for the openEO process graph execution layer."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service.ingestions.schemas import ArtifactFormat
from open_climate_service.openeo.execution import (
    SaveResultEnvelope,
    _augment_with_workflows,
    _bbox_to_dict,
    _ensure_crs,
    _load_collection_impl,
    _RegistryOverlay,
    _temporal_to_list,
    run_process_graph,
)
from open_climate_service.openeo.jobs import (
    OpenEOJobService,
    _build_chap_csv_frame,
    _build_dhis2_json_payload,
    _derive_coverage,
    _derive_variable,
    _infer_period_type,
    _recover_temporal_from_attrs,
    _result_assets,
    _to_dhis2_period_string,
    _to_dhis2_value_string,
    _write_dataset_tabular_export,
)
from open_climate_service.openeo.schemas import OpenEOJobCreate, OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now

# ---------------------------------------------------------------------------
# _bbox_to_dict
# ---------------------------------------------------------------------------


def test_bbox_to_dict_none_returns_none() -> None:
    assert _bbox_to_dict(None) is None


def test_bbox_to_dict_plain_dict_lowercases_keys() -> None:
    result = _bbox_to_dict({"West": 1.0, "East": 2.0, "South": 3.0, "North": 4.0})
    assert result == {"west": 1.0, "east": 2.0, "south": 3.0, "north": 4.0}


def test_bbox_to_dict_pydantic_object() -> None:
    bbox = MagicMock()
    bbox.west, bbox.south, bbox.east, bbox.north = 1.0, 2.0, 3.0, 4.0
    result = _bbox_to_dict(bbox)
    assert result == {"west": 1.0, "south": 2.0, "east": 3.0, "north": 4.0}


# ---------------------------------------------------------------------------
# _temporal_to_list
# ---------------------------------------------------------------------------


def test_temporal_to_list_none_returns_none() -> None:
    assert _temporal_to_list(None) is None


def test_temporal_to_list_plain_list_strips_tz() -> None:
    result = _temporal_to_list(["2020-01-01T00:00:00Z", "2023-01-01T00:00:00+00:00"])
    assert result == ["2020-01-01T00:00:00", "2023-01-01T00:00:00"]


def test_temporal_to_list_none_element_preserved() -> None:
    result = _temporal_to_list(["2020-01-01", None])
    assert result == ["2020-01-01", None]


def test_temporal_to_list_temporal_interval_object() -> None:
    from openeo_pg_parser_networkx.pg_schema import TemporalInterval

    ti = TemporalInterval.model_validate(["2020-01-01", "2023-06-15"])
    result = _temporal_to_list(ti)
    assert result is not None
    assert result[0] is not None and "2020-01-01" in result[0]
    assert result[1] is not None and "2023-06-15" in result[1]
    # No timezone suffix
    for v in result:
        if v is not None:
            assert "Z" not in v and "+00:00" not in v


# ---------------------------------------------------------------------------
# _RegistryOverlay
# ---------------------------------------------------------------------------


def _make_process(impl: Any) -> Any:
    from openeo_pg_parser_networkx.process_registry import Process

    return Process(spec={}, implementation=impl)


def test_registry_overlay_falls_back_to_base() -> None:
    base = {"add": _make_process(lambda x, y: x + y)}
    overlay = _RegistryOverlay(base, {})
    assert overlay["add"] is base["add"]


def test_registry_overlay_udp_shadows_base() -> None:
    udp_proc = _make_process(lambda: "udp")
    base = {"foo": _make_process(lambda: "base")}
    overlay = _RegistryOverlay(base, {"foo": udp_proc})
    assert overlay["foo"] is udp_proc


def test_registry_overlay_tuple_key_uses_name() -> None:
    proc = _make_process(lambda: None)
    base: dict[Any, Any] = {}
    overlay = _RegistryOverlay(base, {"bar": proc})
    assert overlay[("predefined", "bar")] is proc


# ---------------------------------------------------------------------------
# _ensure_crs / load_collection CRS tagging (regression for #243)
# ---------------------------------------------------------------------------


def _streaming_style_cube() -> xr.Dataset:
    """A cube as written by the streaming engine: x/y coords, CRS only in GeoZarr attrs."""
    ds = xr.Dataset(
        {"pop": (("t", "y", "x"), np.arange(20, dtype="float32").reshape(1, 4, 5))},
        coords={
            "t": np.array(["2021-01-01"], dtype="datetime64[ns]"),
            "y": np.linspace(10, 7, 4),
            "x": np.linspace(-13, -10, 5),
        },
    )
    ds.attrs["proj:code"] = "EPSG:4326"  # GeoZarr root attr only — no spatial_ref coord
    return ds


def test_ensure_crs_tags_untagged_streaming_cube() -> None:
    ds = _streaming_style_cube()
    import rioxarray  # noqa: F401  # activate .rio  # pyright: ignore[reportUnusedImport]

    assert ds.rio.crs is None  # odc.geo would see a non-georegistered array

    tagged = _ensure_crs(ds)

    assert "spatial_ref" in tagged.coords
    assert tagged.rio.crs is not None
    assert tagged.rio.crs.to_epsg() == 4326


def test_ensure_crs_is_idempotent_and_preserves_existing_crs() -> None:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]

    ds = _streaming_style_cube().rio.write_crs("EPSG:32633")  # downloader-path style
    # proj:code says 4326, but an already-written CRS must win (no override).
    result = _ensure_crs(ds)
    assert result.rio.crs.to_epsg() == 32633


def test_load_collection_returns_georegistered_cube(monkeypatch: pytest.MonkeyPatch) -> None:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]

    monkeypatch.setattr("open_climate_service.openeo.execution._get_published_artifact", lambda _id: object())
    monkeypatch.setattr("open_climate_service.openeo.execution._open_artifact", lambda _a: _streaming_style_cube())

    cube = _load_collection_impl("pop_collection")

    # The returned DataArray must carry a CRS so odc-based processes
    # (resample_cube_spatial) can georegister it.
    assert cube.rio.crs is not None
    assert cube.rio.crs.to_epsg() == 4326


def test_load_collection_empty_temporal_extent_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # The mock cube only covers 2021; a 2030 extent selects zero timesteps.
    monkeypatch.setattr("open_climate_service.openeo.execution._get_published_artifact", lambda _id: object())
    monkeypatch.setattr("open_climate_service.openeo.execution._open_artifact", lambda _a: _streaming_style_cube())

    with pytest.raises(HTTPException) as excinfo:
        _load_collection_impl("pop_collection", temporal_extent=["2030-01-01", "2030-12-31"])

    # Fails early with an actionable message instead of letting an empty cube
    # reach reduce_dimension (which would raise a cryptic ndim=0 / broadcast error).
    assert excinfo.value.status_code == 400
    detail = str(excinfo.value.detail)
    assert "pop_collection" in detail
    assert "2030" in detail
    assert "2021" in detail  # reports the available coverage


def test_load_collection_overlapping_temporal_extent_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_climate_service.openeo.execution._get_published_artifact", lambda _id: object())
    monkeypatch.setattr("open_climate_service.openeo.execution._open_artifact", lambda _a: _streaming_style_cube())

    cube = _load_collection_impl("pop_collection", temporal_extent=["2021-01-01", "2021-12-31"])

    assert cube.sizes["t"] == 1


# ---------------------------------------------------------------------------
# _augment_with_workflows
# ---------------------------------------------------------------------------


def test_augment_with_workflows_returns_base_when_no_udps(monkeypatch: pytest.MonkeyPatch) -> None:
    import open_climate_service.openeo.workflows as workflow_module

    monkeypatch.setattr(workflow_module, "list_workflows", lambda: MagicMock(processes=[]))
    base = object()
    result = _augment_with_workflows(base)
    assert result is base


def test_augment_with_workflows_registers_udp(monkeypatch: pytest.MonkeyPatch) -> None:
    import open_climate_service.openeo.workflows as workflow_module

    udp = MagicMock()
    udp.id = "my_udp"
    udp.process_graph = {
        "result": {"process_id": "save_result", "arguments": {"data": 42, "format": "Zarr"}, "result": True}
    }
    monkeypatch.setattr(workflow_module, "list_workflows", lambda: MagicMock(processes=[udp]))

    from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry

    base = ProcessRegistry()
    base["save_result"] = Process(spec={}, implementation=lambda data, **kw: data)

    overlay = _augment_with_workflows(base)
    assert isinstance(overlay, _RegistryOverlay)
    assert "my_udp" in overlay._udps


# ---------------------------------------------------------------------------
# _persist_result — DataArray and GeoDataFrame handling
# ---------------------------------------------------------------------------


@pytest.fixture()
def job_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OpenEOJobService:
    monkeypatch.setattr("open_climate_service.openeo.jobs._JOBS_DIR", tmp_path)
    return OpenEOJobService(max_workers=1)


def _sample_dataarray() -> xr.DataArray:
    return xr.DataArray(
        np.ones((3, 4, 5), dtype=np.float32),
        dims=["t", "y", "x"],
        name="temperature",
    )


def test_persist_result_dataarray_writes_zarr(job_service: OpenEOJobService, tmp_path: Path) -> None:
    da = _sample_dataarray()
    output_path = job_service._persist_result("job-1", da)

    assert output_path is not None
    assert output_path.endswith(".zarr")
    ds = xr.open_zarr(output_path)
    assert "temperature" in ds


def test_persist_result_dataset_writes_zarr(job_service: OpenEOJobService) -> None:
    ds = _sample_dataarray().to_dataset(name="ta")
    output_path = job_service._persist_result("job-2", ds)

    assert output_path is not None
    assert output_path.endswith(".zarr")


def test_persist_result_unsupported_type_raises(job_service: OpenEOJobService) -> None:
    with pytest.raises(TypeError, match="Unsupported result type"):
        job_service._persist_result("job-3", {"value": 42})


def test_persist_result_geodataframe_writes_geojson(job_service: OpenEOJobService) -> None:
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"value": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    output_path = job_service._persist_result("job-4", gdf)

    assert output_path is not None
    assert output_path.endswith(".geojson")


def test_persist_result_dataset_writes_dhis2_json(job_service: OpenEOJobService) -> None:
    ds = xr.Dataset(
        {
            "precip": xr.DataArray(
                np.array([[1.5, np.nan], [2.75, 3.0]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            )
        }
    )

    output_path = job_service._persist_result(
        "job-dhis2",
        SaveResultEnvelope(
            ds,
            "DHIS2JSON",
            {
                "data_element_id": "DE_123",
                "org_unit_field": "geometry",
                "period_type": "monthly",
            },
        ),
    )

    assert output_path is not None
    assert output_path.endswith("result.json")
    payload = json.loads(Path(output_path).read_text())
    assert payload == {
        "dataValues": [
            {"dataElement": "DE_123", "orgUnit": "OU_1", "period": "202401", "value": "1.5"},
            {"dataElement": "DE_123", "orgUnit": "OU_1", "period": "202402", "value": "2.75"},
            {"dataElement": "DE_123", "orgUnit": "OU_2", "period": "202402", "value": "3"},
        ]
    }


def test_persist_result_dataset_writes_chap_csv(job_service: OpenEOJobService) -> None:
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.array([[28.4, 24.1], [29.0, 25.5]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            ),
            "precipitation": xr.DataArray(
                np.array([[12.1, 8.3], [np.nan, 9.4]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            ),
        }
    )

    output_path = job_service._persist_result("job-chap", SaveResultEnvelope(ds, "CHAPCSV"))

    assert output_path is not None
    assert output_path.endswith("result.csv")
    rows = list(csv.DictReader(StringIO(Path(output_path).read_text())))
    assert rows == [
        {"time_period": "202401", "location": "OU_1", "temperature": "28.4", "precipitation": "12.1"},
        {"time_period": "202401", "location": "OU_2", "temperature": "24.1", "precipitation": "8.3"},
        {"time_period": "202402", "location": "OU_1", "temperature": "29", "precipitation": ""},
        {"time_period": "202402", "location": "OU_2", "temperature": "25.5", "precipitation": "9.4"},
    ]


def test_openeo_job_service_create_execute_and_get_results(
    job_service: OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process: {"ignored": True},
    )
    monkeypatch.setattr(
        job_service,
        "_persist_result",
        lambda job_id, result: f"/tmp/{job_id}/result.geojson",
    )

    record = job_service.create_job(
        OpenEOJobCreate(
            process={"process_graph": {"result": {"process_id": "constant", "arguments": {"x": 1}, "result": True}}}
        )
    )

    job_service._execute(record.id)
    results = job_service.get_results(record.id)

    assert results.id == record.id
    assert results.assets["result"]["href"].endswith("result.geojson")


# ---------------------------------------------------------------------------
# _result_assets
# ---------------------------------------------------------------------------


def _record(output_path: str | None) -> OpenEOJobRecord:
    return OpenEOJobRecord(
        id="job-1",
        status=OpenEOJobStatus.FINISHED,
        created=utc_now(),
        updated=utc_now(),
        usage={"output_path": output_path} if output_path else {},
    )


def test_result_assets_zarr() -> None:
    assets = _result_assets(_record("/some/path/result.zarr"))
    assert assets["result"]["type"] == "application/x-zarr"
    assert assets["result"]["href"].endswith("result.zarr/")


def test_result_assets_geojson() -> None:
    assets = _result_assets(_record("/some/path/result.geojson"))
    assert assets["result"]["type"] == "application/geo+json"
    assert assets["result"]["href"].endswith("result.geojson")


def test_result_assets_csv() -> None:
    assets = _result_assets(_record("/some/path/result.csv"))
    assert assets["result"]["type"] == "text/csv"
    assert assets["result"]["href"].endswith("result.csv")


def test_result_assets_json() -> None:
    assets = _result_assets(_record("/some/path/result.json"))
    assert assets["result"]["type"] == "application/json"
    assert assets["result"]["href"].endswith("result.json")


def test_result_assets_none_output_returns_empty() -> None:
    assert _result_assets(_record(None)) == {}


def test_result_assets_managed_dataset_exposes_links(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unpublished: dataset + zarr links, no STAC.
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.latest_published_zarr_artifacts_by_dataset",
        lambda: {},
    )
    assets = _result_assets(_record("managed://my_aggregate"))
    assert assets["dataset"]["href"] == "/datasets/my_aggregate"
    assert assets["zarr"]["href"] == "/zarr/my_aggregate"
    assert "stac" not in assets

    # Published: STAC collection link is added.
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.latest_published_zarr_artifacts_by_dataset",
        lambda: {"my_aggregate": object()},
    )
    published = _result_assets(_record("managed://my_aggregate"))
    assert published["stac"]["href"] == "/stac/collections/my_aggregate"


def test_download_result_file_serves_geojson_with_geojson_media_type(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("open_climate_service.openeo.jobs._JOBS_DIR", tmp_path)
    results_dir = tmp_path / "job-1" / "results"
    results_dir.mkdir(parents=True)
    geojson_path = results_dir / "result.geojson"
    geojson_path.write_text('{"type":"FeatureCollection","features":[]}')

    response = client.get("/jobs/job-1/results/result.geojson")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")


def test_download_result_file_serves_json_with_json_media_type(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("open_climate_service.openeo.jobs._JOBS_DIR", tmp_path)
    results_dir = tmp_path / "job-1" / "results"
    results_dir.mkdir(parents=True)
    json_path = results_dir / "result.json"
    json_path.write_text('{"dataValues":[]}')

    response = client.get("/jobs/job-1/results/result.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_create_job_uses_explicit_title_when_provided(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        json={
            "title": "My custom job",
            "process": {"process_graph": {"result": {"process_id": "constant", "arguments": {"x": 1}, "result": True}}},
        },
    )
    assert response.status_code == 201
    assert response.json()["title"] == "My custom job"


def test_create_job_derives_title_from_load_collection(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        json={
            "process": {
                "process_graph": {
                    "load": {
                        "process_id": "load_collection",
                        "arguments": {
                            "id": "chirps3_precipitation_daily",
                            "temporal_extent": ["2023-01-01", "2023-12-31"],
                        },
                    },
                    "result": {
                        "process_id": "save_result",
                        "arguments": {"data": {"from_node": "load"}, "format": "GTiff"},
                        "result": True,
                    },
                }
            }
        },
    )
    assert response.status_code == 201
    assert response.json()["title"] == "chirps3_precipitation_daily 2023-01-01–2023-12-31"


def test_create_job_title_is_none_when_no_load_collection(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        json={
            "process": {"process_graph": {"result": {"process_id": "constant", "arguments": {"x": 1}, "result": True}}}
        },
    )
    assert response.status_code == 201
    assert response.json()["title"] is None


def test_create_job_does_not_advertise_missing_logs_endpoint(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        json={
            "process": {"process_graph": {"result": {"process_id": "constant", "arguments": {"x": 1}, "result": True}}}
        },
    )

    assert response.status_code == 201
    links = response.json()["links"]
    assert all(link["rel"] != "logs" for link in links)


def test_put_udp_rejects_predefined_process_id(client: TestClient) -> None:
    response = client.put(
        "/process_graphs/load_collection",
        json={"summary": "Bad override", "process_graph": {}},
    )

    assert response.status_code == 400
    assert "conflicts with a predefined process" in response.json()["detail"]


def test_run_process_graph_maps_invalid_graph_errors_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGraph:
        def __init__(self, process_graph: dict[str, Any]) -> None:
            self.process_graph = process_graph

        def to_callable(self, registry: Any) -> Any:
            def _runner() -> Any:
                raise ValueError("unknown process id")

            return _runner

    monkeypatch.setattr("openeo_pg_parser_networkx.OpenEOProcessGraph", FakeGraph)

    with pytest.raises(Exception) as exc_info:
        run_process_graph({"process_graph": {"result": {"process_id": "missing", "result": True}}})

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 400
    assert "Invalid process graph" in str(getattr(exc, "detail", exc))


def test_run_process_graph_keeps_runtime_failures_as_500(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGraph:
        def __init__(self, process_graph: dict[str, Any]) -> None:
            self.process_graph = process_graph

        def to_callable(self, registry: Any) -> Any:
            def _runner() -> Any:
                raise RuntimeError("boom")

            return _runner

    monkeypatch.setattr("openeo_pg_parser_networkx.OpenEOProcessGraph", FakeGraph)

    with pytest.raises(Exception) as exc_info:
        run_process_graph({"process_graph": {"result": {"process_id": "add", "result": True}}})

    exc = exc_info.value
    assert getattr(exc, "status_code", None) == 500
    assert "Process graph execution failed" in str(getattr(exc, "detail", exc))


def test_result_route_rejects_synchronous_zarr_datacube(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process, request=None: xr.Dataset(
            {"temperature": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])}
        ),
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "load_collection", "result": True}}},
    )

    assert response.status_code == 400
    assert "do not support ZARR output" in response.json()["detail"]


@pytest.mark.parametrize(
    ("fmt", "expected_fragment"),
    [
        ("GEOJSON", "describes vector features"),
        ("PARQUET", "describes vector features"),
        ("JSON", "Unsupported output format"),
    ],
)
def test_result_route_rejects_formats_a_raster_cube_cannot_produce(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fmt: str,
    expected_fragment: str,
) -> None:
    """A raster cube asked for a vector or unknown format must 4xx, not 500.

    These used to fall through to the Zarr default, writing a `result.zarr` directory that the
    route then tried to read as a file: `IsADirectoryError`, surfaced as 500 (CLIM-909).
    """
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process, request=None: SaveResultEnvelope(
            xr.Dataset({"temperature": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])}),
            fmt,
        ),
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 400
    assert expected_fragment in response.json()["detail"]


@pytest.mark.parametrize("fmt", ["NETCDF", "GTIFF", "CSV"])
def test_result_route_still_serves_raster_formats(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fmt: str,
) -> None:
    """The refusal above must not catch formats a raster cube genuinely produces."""
    ds = xr.Dataset(
        {"temperature": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process, request=None: SaveResultEnvelope(ds, fmt),
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200, response.text
    assert response.content


def test_result_route_returns_geojson_payload_for_synchronous_vector_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"value": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")

    def return_geojson_result(*args: object, **kwargs: object) -> SaveResultEnvelope:
        del args, kwargs
        return SaveResultEnvelope(gdf, "GEOJSON")

    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        return_geojson_result,
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 2
    assert sorted(feature["properties"]["value"] for feature in payload["features"]) == [1.0, 2.0]


def test_result_route_reprojects_geojson_payload_to_wgs84(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"value": [1.0]}, geometry=[Point(111319.49079327357, 111325.1428663851)], crs="EPSG:3857")

    def return_geojson_result(*args: object, **kwargs: object) -> SaveResultEnvelope:
        del args, kwargs
        return SaveResultEnvelope(gdf, "GEOJSON")

    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        return_geojson_result,
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200
    payload = response.json()
    coords = payload["features"][0]["geometry"]["coordinates"]
    assert coords[0] == pytest.approx(1.0, rel=0, abs=1e-6)
    assert coords[1] == pytest.approx(1.0, rel=0, abs=1e-6)


def test_result_route_returns_chap_csv_payload(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.array([[28.4, 24.1]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            ),
            "precipitation": xr.DataArray(
                np.array([[12.1, 8.3]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            ),
        }
    )

    def return_chap_result(*args: object, **kwargs: object) -> SaveResultEnvelope:
        del args, kwargs
        return SaveResultEnvelope(ds, "CHAPCSV", {"period_type": "monthly"})

    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        return_chap_result,
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(StringIO(response.text)))
    assert rows == [
        {"time_period": "202401", "location": "OU_1", "temperature": "28.4", "precipitation": "12.1"},
        {"time_period": "202401", "location": "OU_2", "temperature": "24.1", "precipitation": "8.3"},
    ]


def test_result_route_returns_dhis2_json_payload(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {
            "precip": xr.DataArray(
                np.array([[1.5, np.nan], [2.75, 3.0]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1", "OU_2"],
                },
            )
        }
    )

    def return_dhis2_result(*args: object, **kwargs: object) -> SaveResultEnvelope:
        del args, kwargs
        return SaveResultEnvelope(
            ds,
            "DHIS2JSON",
            {
                "data_element_id": "DE_123",
                "org_unit_field": "geometry",
                "period_type": "monthly",
                "category_option_combo": "COC_456",
            },
        )

    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        return_dhis2_result,
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "dataValues": [
            {
                "dataElement": "DE_123",
                "orgUnit": "OU_1",
                "period": "202401",
                "value": "1.5",
                "categoryOptionCombo": "COC_456",
            },
            {
                "dataElement": "DE_123",
                "orgUnit": "OU_1",
                "period": "202402",
                "value": "2.75",
                "categoryOptionCombo": "COC_456",
            },
            {
                "dataElement": "DE_123",
                "orgUnit": "OU_2",
                "period": "202402",
                "value": "3",
                "categoryOptionCombo": "COC_456",
            },
        ]
    }


def test_result_route_returns_400_for_missing_dhis2_option(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {
            "precip": xr.DataArray(
                np.array([[1.5]], dtype=np.float32),
                dims=["t", "geometry"],
                coords={
                    "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                    "geometry": ["OU_1"],
                },
            )
        }
    )

    def return_invalid_result(*args: object, **kwargs: object) -> SaveResultEnvelope:
        del args, kwargs
        return SaveResultEnvelope(
            ds,
            "DHIS2JSON",
            {
                "org_unit_field": "geometry",
                "period_type": "monthly",
            },
        )

    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        return_invalid_result,
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 400
    assert "Missing required export option 'data_element_id'" in response.json()["detail"]


def test_dhis2_json_payload_omits_nulls_and_formats_values() -> None:
    payload = _build_dhis2_json_payload(
        [
            {"t": "2024-01-01", "geometry": "OU_1", "precip": 1e-05},
            {"t": "2024-01-02", "geometry": "OU_2", "precip": None},
        ],
        {
            "data_element_id": "DE_123",
            "org_unit_field": "geometry",
            "period_type": "daily",
        },
    )

    assert payload == {
        "dataValues": [
            {
                "dataElement": "DE_123",
                "orgUnit": "OU_1",
                "period": "20240101",
                "value": "0.00001",
            }
        ]
    }


def test_dhis2_json_payload_requires_org_unit_field() -> None:
    with pytest.raises(ValueError, match="Missing required export option 'org_unit_field'"):
        _build_dhis2_json_payload(
            [{"t": "2024-01-01", "precip": 1.5}],
            {"data_element_id": "DE_123", "period_type": "daily"},
        )


def test_dhis2_json_payload_requires_period_field_column() -> None:
    with pytest.raises(ValueError, match="Missing period field 't' in aggregated result"):
        _build_dhis2_json_payload(
            [{"geometry": "OU_1", "precip": 1.5}],
            {
                "data_element_id": "DE_123",
                "org_unit_field": "geometry",
                "period_type": "daily",
            },
        )


def test_dhis2_json_payload_rejects_multiple_value_columns() -> None:
    with pytest.raises(ValueError, match="requires exactly one value column"):
        _build_dhis2_json_payload(
            [{"t": "2024-01-01", "geometry": "OU_1", "precip": 1.5, "temp": 22.0}],
            {
                "data_element_id": "DE_123",
                "org_unit_field": "geometry",
                "period_type": "daily",
            },
        )


def test_dhis2_period_string_supports_weekly_and_quarterly() -> None:
    assert _to_dhis2_period_string("2024-02-01", "monthly") == "202402"
    assert _to_dhis2_period_string("2024-02-01", "weekly").startswith("2024W")
    assert _to_dhis2_period_string("2024-05-01", "quarterly") == "2024Q2"


def test_dhis2_period_string_accepts_existing_dhis2_strings_with_period_type() -> None:
    assert _to_dhis2_period_string("2024Q2", "quarterly") == "2024Q2"
    assert _to_dhis2_period_string("2024W05", "weekly") == "2024W05"


def test_dhis2_period_string_rejects_ambiguous_string_without_period_type() -> None:
    with pytest.raises(ValueError, match="Ambiguous period value"):
        _to_dhis2_period_string("2024-02-01")


def test_dhis2_period_string_rejects_unsupported_period_type() -> None:
    with pytest.raises(ValueError, match="Unsupported period_type 'decadal'"):
        _to_dhis2_period_string("2024-02-01", "decadal")


def test_dhis2_value_string_avoids_scientific_notation() -> None:
    assert _to_dhis2_value_string(1e-05) == "0.00001"


def test_dhis2_value_string_formats_numpy_scalars() -> None:
    assert _to_dhis2_value_string(np.float32(1e-05)) == "0.00001"
    assert _to_dhis2_value_string(np.int64(42)) == "42"


def test_dhis2_value_string_formats_boolean_scalars() -> None:
    assert _to_dhis2_value_string(True) == "true"
    assert _to_dhis2_value_string(np.bool_(False)) == "false"


def test_build_chap_csv_frame_requires_location_field() -> None:
    with pytest.raises(ValueError, match="Missing location field 'geometry' in aggregated result"):
        _build_chap_csv_frame(
            [{"t": "2024-01-01", "temperature": 1.5}],
            {"period_type": "daily"},
        )


def test_build_chap_csv_frame_missing_default_geometry_field_guides_geodataframe_inputs() -> None:
    with pytest.raises(ValueError, match="set save_result option 'location_field' explicitly"):
        _build_chap_csv_frame(
            [{"t": "2024-01-01", "temperature": 1.5}],
            {"period_type": "daily"},
        )


def test_build_chap_csv_frame_requires_period_field() -> None:
    with pytest.raises(ValueError, match="Missing period field 't' in aggregated result"):
        _build_chap_csv_frame(
            [{"geometry": "OU_1", "temperature": 1.5}],
            {"period_type": "daily"},
        )


def test_build_chap_csv_frame_includes_row_context_for_null_location() -> None:
    with pytest.raises(ValueError, match=r"Null location value in field 'geometry' at row 0"):
        _build_chap_csv_frame(
            [{"t": "2024-01-01", "geometry": None, "temperature": 1.5}],
            {"period_type": "daily"},
        )


def test_build_chap_csv_frame_requires_at_least_one_value_column() -> None:
    with pytest.raises(ValueError, match="requires at least one value column"):
        _build_chap_csv_frame(
            [{"t": "2024-01-01", "geometry": "OU_1"}],
            {"period_type": "daily"},
        )


def test_build_chap_csv_frame_pivots_merged_cubes_wide() -> None:
    frame = _build_chap_csv_frame(
        [
            {"t": "2024-01-01", "geometry": "OU_1", "__cubes__": "cube1", "tp": 1.5},
            {"t": "2024-01-01", "geometry": "OU_1", "__cubes__": "cube2", "tp": 2.5},
            {"t": "2024-02-01", "geometry": "OU_1", "__cubes__": "cube1", "tp": 3.5},
            {"t": "2024-02-01", "geometry": "OU_1", "__cubes__": "cube2", "tp": 4.5},
        ],
        {"period_type": "monthly"},
    )
    assert list(frame.columns) == ["time_period", "location", "cube1", "cube2"]
    assert frame.to_dict(orient="records") == [
        {"time_period": "202401", "location": "OU_1", "cube1": "1.5", "cube2": "2.5"},
        {"time_period": "202402", "location": "OU_1", "cube1": "3.5", "cube2": "4.5"},
    ]


def test_build_chap_csv_frame_renames_merged_cubes_with_explicit_labels() -> None:
    frame = _build_chap_csv_frame(
        [
            {"t": "2024-01-01", "geometry": "OU_1", "__cubes__": "cube1", "tp": 1.5},
            {"t": "2024-01-01", "geometry": "OU_1", "__cubes__": "cube2", "tp": 2.5},
        ],
        {"period_type": "monthly", "cube_labels": {"cube1": "tp", "cube2": "t2m"}},
    )
    assert list(frame.columns) == ["time_period", "location", "tp", "t2m"]
    assert frame.to_dict(orient="records") == [
        {"time_period": "202401", "location": "OU_1", "tp": "1.5", "t2m": "2.5"},
    ]


def test_process_registry_builds_with_full_server_impl_stack() -> None:
    """Guard against the [server] curated dep list drifting from upstream (#288).

    openeo-processes-dask eagerly imports its whole implementation stack (xvec, odc,
    dask_geopandas, planetary_computer, pystac_client, stac_validator, …) at module load.
    Importing it here fails if any of those are missing from the [server] extra, and
    building the registry confirms the standard + backend processes are all present.
    A new openeo-processes-dask release that adds an eager import we don't declare would
    fail this test in CI — before it reaches an instance as a mid-job ModuleNotFoundError.
    """
    import importlib

    # Triggers the eager import of the full implementation stack.
    importlib.import_module("openeo_processes_dask.process_implementations")

    from open_climate_service.openeo.execution import _build_process_registry

    predefined = _build_process_registry()[("predefined", None)]
    assert predefined  # registry built and non-empty
    expected = ("spi", "load_collection", "save_result", "ndvi", "reduce_dimension", "aggregate_temporal_period")
    for process_id in expected:
        assert process_id in predefined, f"expected process '{process_id}' missing from the registry"


def test_registry_build_missing_server_dep_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing [server] impl dep yields one clear, actionable message, not a deep
    ModuleNotFoundError mid-job (#289)."""
    import importlib

    import open_climate_service.openeo.execution as ex

    monkeypatch.setattr(ex, "_registry", None)  # bypass the singleton so the import runs
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openeo_processes_dask.process_implementations":
            raise ModuleNotFoundError("No module named 'xvec'", name="xvec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(ex.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match=r"missing server dependency 'xvec'"):
        ex._build_process_registry()


def test_merge_cubes_wrapper_preserves_named_dataarrays_on_cube_axis() -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    reg = _build_process_registry()
    merge = reg["merge_cubes"].implementation
    cube1 = xr.DataArray(
        np.ones((2, 2, 2), dtype=np.float32),
        dims=("t", "y", "x"),
        coords={"t": [0, 1], "y": [0, 1], "x": [0, 1]},
        name="tp",
    )
    cube2 = xr.DataArray(
        np.full((2, 2, 2), 2.0, dtype=np.float32),
        dims=("t", "y", "x"),
        coords={"t": [0, 1], "y": [0, 1], "x": [0, 1]},
        name="t2m",
    )
    merged = merge(cube1=cube1, cube2=cube2)
    assert isinstance(merged, xr.DataArray)
    assert "__cubes__" in merged.dims
    assert list(merged["__cubes__"].values) == ["tp", "t2m"]


def test_merge_cubes_wrapper_appends_third_named_predictor() -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    reg = _build_process_registry()
    merge = reg["merge_cubes"].implementation
    coords = {
        "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[D]"),
        "geometry": ["DISTRICT_A", "DISTRICT_B"],
    }
    precipitation = xr.DataArray(
        np.array([[100.0, 80.0], [120.0, 90.0]], dtype=np.float32),
        dims=("t", "geometry"),
        coords=coords,
        name="precip",
    )
    temperature = xr.DataArray(
        np.array([[25.0, 24.0], [26.0, 25.0]], dtype=np.float32),
        dims=("t", "geometry"),
        coords=coords,
        name="t2m",
    )
    population = xr.DataArray(
        np.array([[10_000.0, 20_000.0], [10_000.0, 20_000.0]], dtype=np.float32),
        dims=("t", "geometry"),
        coords=coords,
        name="pop_total",
    )

    climate = merge(cube1=precipitation, cube2=temperature)
    merged = merge(cube1=climate, cube2=population)

    assert isinstance(merged, xr.DataArray)
    assert list(merged["__cubes__"].values) == ["precip", "t2m", "pop_total"]
    xr.testing.assert_equal(merged.sel(__cubes__="pop_total", drop=True), population)


def test_merge_cubes_wrapper_combines_three_zonal_datasets_as_chap_csv() -> None:
    """Single-variable Datasets mirror aggregate_spatial's return type."""
    from open_climate_service.openeo.execution import _build_process_registry

    reg = _build_process_registry()
    merge = reg["merge_cubes"].implementation
    coords = {
        "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[D]"),
        "geometry": ["DISTRICT_A"],
    }
    precipitation = xr.Dataset(
        {"precip": (("t", "geometry"), np.array([[100.0], [120.0]], dtype=np.float32))},
        coords=coords,
    )
    temperature = xr.Dataset(
        {"t2m": (("t", "geometry"), np.array([[25.0], [26.0]], dtype=np.float32))},
        coords=coords,
    )
    population = xr.Dataset(
        {"pop_total": (("t", "geometry"), np.array([[10_000.0], [10_000.0]], dtype=np.float32))},
        coords=coords,
    )

    climate = merge(cube1=precipitation, cube2=temperature)
    merged = merge(cube1=climate, cube2=population)
    frame = _build_chap_csv_frame(merged.to_dataframe().reset_index(), {"period_type": "monthly"})

    assert frame.to_dict(orient="records") == [
        {
            "time_period": "202501",
            "location": "DISTRICT_A",
            "precip": "100",
            "t2m": "25",
            "pop_total": "10000",
        },
        {
            "time_period": "202502",
            "location": "DISTRICT_A",
            "precip": "120",
            "t2m": "26",
            "pop_total": "10000",
        },
    ]


def test_merge_cubes_wrapper_rejects_misaligned_predictor_indexes() -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    reg = _build_process_registry()
    merge = reg["merge_cubes"].implementation
    precipitation = xr.DataArray(
        np.ones((2, 1), dtype=np.float32),
        dims=("t", "geometry"),
        coords={"t": [0, 1], "geometry": ["DISTRICT_A"]},
        name="precip",
    )
    temperature = precipitation.rename("t2m")
    population = xr.DataArray(
        np.ones((2, 1), dtype=np.float32),
        dims=("t", "geometry"),
        coords={"t": [0, 2], "geometry": ["DISTRICT_A"]},
        name="pop_total",
    )

    climate = merge(cube1=precipitation, cube2=temperature)
    with pytest.raises(ValueError, match="cannot align objects with join='exact'"):
        merge(cube1=climate, cube2=population)


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_cubes_wrapper_combines_two_named_predictor_groups(reverse: bool) -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    merge = _build_process_registry()["merge_cubes"].implementation
    predictors = [
        xr.DataArray([float(index)], dims="t", coords={"t": [0]}, name=name)
        for index, name in enumerate(["precip", "t2m", "pop_total", "elevation"])
    ]
    left = merge(cube1=predictors[0], cube2=predictors[1])
    right = merge(cube1=predictors[2], cube2=predictors[3])
    merged = merge(cube1=right, cube2=left) if reverse else merge(cube1=left, cube2=right)

    expected = predictors[2:] + predictors[:2] if reverse else predictors
    assert list(merged["__cubes__"].values) == [cube.name for cube in expected]
    for cube in predictors:
        xr.testing.assert_equal(merged.sel(__cubes__=cube.name, drop=True), cube)


@pytest.mark.parametrize("missing", ["dimension", "index"])
def test_merge_cubes_wrapper_does_not_broadcast_missing_predictor_axes(missing: str) -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    merge = _build_process_registry()["merge_cubes"].implementation
    cube = xr.DataArray(
        np.ones((2, 1)),
        dims=("t", "geometry"),
        coords={"t": [0, 1], "geometry": ["DISTRICT_A"]},
        name="precip",
    )
    climate = merge(cube1=cube, cube2=cube.rename("t2m"))
    population = cube.rename("pop_total")
    if missing == "dimension":
        population = population.isel(t=0, drop=True)
    else:
        population = population.drop_indexes("t")
    with pytest.raises(ValueError, match="Named predictors must have the same"):
        merge(cube1=climate, cube2=population)


@pytest.mark.parametrize("dimension, labels", [("geometry", ["B", "A"]), ("y", [2.0, 1.0])])
@pytest.mark.parametrize("noise", [0.0, 1e-9])
def test_merge_cubes_wrapper_aligns_reordered_predictors(dimension: str, labels: list[Any], noise: float) -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    merge = _build_process_registry()["merge_cubes"].implementation
    cube = xr.DataArray([10.0, 20.0], dims=dimension, coords={dimension: labels}, name="precip")
    stacked = merge(cube1=cube, cube2=cube.rename("t2m").copy(deep=True))
    third = cube.rename("population").isel({dimension: [1, 0]})
    if dimension == "y":
        third = third.assign_coords({dimension: third[dimension] + noise})
    before = third.copy(deep=True)
    result = merge(cube1=stacked, cube2=third)
    xr.testing.assert_equal(result.sel(__cubes__="population", drop=True), cube.rename("population"))
    xr.testing.assert_identical(third, before)
    assert result.chunksizes["__cubes__"] == (3,)


@pytest.mark.parametrize("offset", [1e-9, 1e-4])
def test_merge_cubes_wrapper_uses_upstream_coordinate_tolerance(offset: float) -> None:
    from open_climate_service.openeo.execution import _build_process_registry

    merge = _build_process_registry()["merge_cubes"].implementation
    cube = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [10.0, 11.0]}, name="precip")
    stacked = merge(cube1=cube, cube2=cube.rename("t2m").copy(deep=True))
    third = cube.rename("population").assign_coords(x=cube.x + offset)
    if offset > 1e-6:
        with pytest.raises(ValueError, match="cannot align objects with join='exact'"):
            merge(cube1=stacked, cube2=third)
    else:
        result = merge(cube1=stacked, cube2=third)
        xr.testing.assert_equal(result.sel(__cubes__="population", drop=True), cube.rename("population"))


@pytest.mark.parametrize("grouped", [False, True])
def test_merge_cubes_wrapper_delegates_disjoint_resolver_and_context(grouped: bool) -> None:
    from open_climate_service.openeo.execution import _make_named_merge_cubes

    cube = xr.DataArray([[1.0], [2.0]], dims=("__cubes__", "t"), coords={"__cubes__": ["precip", "t2m"], "t": [0]})
    other = xr.DataArray([3.0], dims="t", coords={"t": [0]}, name="population")
    if grouped:
        other = other.expand_dims(__cubes__=["population"])
    captured: dict[str, Any] = {}
    reduced = other.sum()

    def original(**kwargs: Any) -> xr.DataArray:
        captured.update(kwargs)
        return reduced

    resolver = object()
    context = {"scale": 2}
    result = _make_named_merge_cubes(original)(cube1=cube, cube2=other, overlap_resolver=resolver, context=context)
    assert captured == {"cube1": cube, "cube2": other, "overlap_resolver": resolver, "context": context}
    assert result is reduced


def test_merge_cubes_wrapper_delegates_overlapping_labels_and_resolver() -> None:
    from open_climate_service.openeo.execution import _make_named_merge_cubes

    cube = xr.DataArray(
        [[1.0], [2.0]],
        dims=("__cubes__", "t"),
        coords={"__cubes__": ["precip", "t2m"], "t": [0]},
        name="precip",
    )
    other = cube.rename("other")
    captured: dict[str, Any] = {}

    def original(**kwargs: Any) -> xr.DataArray:
        captured.update(kwargs)
        return cube

    resolver = object()
    result = _make_named_merge_cubes(original)(cube1=cube, cube2=other, overlap_resolver=resolver)
    assert captured["cube1"] is cube
    assert captured["cube2"] is other
    assert captured["overlap_resolver"] is resolver
    assert result is cube
    assert list(result["__cubes__"].values) == ["precip", "t2m"]


def test_dhis2_period_string_accepts_existing_monthly_string() -> None:
    assert _to_dhis2_period_string("202401") == "202401"


def test_build_chap_csv_frame_rejects_ambiguous_period_without_period_type() -> None:
    with pytest.raises(ValueError, match="Ambiguous period value"):
        _build_chap_csv_frame(
            [{"t": "2024-01-01", "geometry": "OU_1", "temperature": 1.5}],
            {},
        )


def test_dhis2_period_string_rejects_mismatched_existing_period_type() -> None:
    with pytest.raises(ValueError, match="appears to be monthly, but period_type=daily"):
        _to_dhis2_period_string("202401", "daily")


def test_dhis2_period_string_wraps_parse_failure_with_context() -> None:
    with pytest.raises(ValueError, match=r"Could not parse period value 'not-a-date' for period_type=daily"):
        _to_dhis2_period_string("not-a-date", "daily")


def test_write_dataset_tabular_export_does_not_infer_unsupported_hourly_period_type(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {"temperature": (("t", "geometry"), np.array([[1.0], [2.0]], dtype=np.float32))},
        coords={
            "t": np.array(["2024-01-01T00:00:00", "2024-01-01T01:00:00"], dtype="datetime64[ns]"),
            "geometry": ["OU_1"],
        },
    )

    with pytest.raises(ValueError, match="Ambiguous period value"):
        _write_dataset_tabular_export(ds, tmp_path, "CHAPCSV", {})


@pytest.mark.parametrize("period_type", [None, ""])
def test_write_dataset_tabular_export_treats_blank_period_type_as_missing(
    tmp_path: Path, period_type: str | None
) -> None:
    ds = xr.Dataset(
        {"temperature": (("t", "geometry"), np.array([[1.0], [2.0]], dtype=np.float32))},
        coords={
            "t": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]"),
            "geometry": ["OU_1"],
        },
    )

    output_path = _write_dataset_tabular_export(ds, tmp_path, "CHAPCSV", {"period_type": period_type})

    assert output_path is not None
    rows = list(csv.DictReader(StringIO(Path(output_path).read_text())))
    assert rows == [
        {"time_period": "202401", "location": "OU_1", "temperature": "1"},
        {"time_period": "202402", "location": "OU_1", "temperature": "2"},
    ]


# ---------------------------------------------------------------------------
# _write_managed_zarr — managed Icechunk / pyramid store via save_result
# ---------------------------------------------------------------------------


def _small_dataset() -> xr.Dataset:
    """A dataset small enough to stay below the pyramid pixel threshold."""
    t = np.array(["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    return xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((3, 4, 5), dtype=np.float32))},
        coords={"t": t, "y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]},
    )


def _stub_get_dataset(dataset_id: str) -> dict[str, Any]:
    """Minimal dataset template stub for tests that call _write_managed_zarr."""
    return {"id": dataset_id}


def test_persist_result_writes_icechunk_when_dataset_id_in_options(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _small_dataset()
    envelope = SaveResultEnvelope(ds, "Zarr", {"dataset_id": "my_aggregate"})

    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _stub_get_dataset)
    registered: list[Any] = []
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: registered.append((record, publish)) or record,
    )

    result = job_service._persist_result("job-managed", envelope)

    assert result == "managed://my_aggregate"
    assert (tmp_path / "my_aggregate.icechunk").exists()
    assert len(registered) == 1
    record, publish_flag = registered[0]
    assert record.dataset_id == "my_aggregate"
    assert record.format == ArtifactFormat.ICECHUNK
    assert publish_flag is True  # publish defaults to True when key is absent


def test_persist_result_stamps_cf_attrs_from_template(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CF attributes from the dataset template are persisted onto the managed store variable (#280)."""
    from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

    def _cf_template(dataset_id: str) -> dict[str, Any]:
        return {
            "id": dataset_id,
            "units": "mm",
            "standard_name": "lwe_thickness_of_precipitation_amount",
            "cell_methods": "time: sum",
        }

    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _cf_template)
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: record,
    )

    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": "cf_aggregate", "publish": False})
    job_service._persist_result("job-cf", envelope)

    written = open_icechunk_dataset(str(tmp_path / "cf_aggregate.icechunk"))
    try:
        attrs = written["precip"].attrs
        assert attrs.get("units") == "mm"
        assert attrs.get("standard_name") == "lwe_thickness_of_precipitation_amount"
        assert attrs.get("cell_methods") == "time: sum"
    finally:
        written.close()


def test_persist_result_does_not_publish_when_publish_false(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": "ds", "publish": False})
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _stub_get_dataset)
    registered: list[Any] = []
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: registered.append((record, publish)) or record,
    )

    job_service._persist_result("job-nopub", envelope)

    _, publish_flag = registered[0]
    assert publish_flag is False


def test_persist_result_publishes_to_stac_when_publish_true(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": "ds", "publish": True})
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _stub_get_dataset)
    registered: list[Any] = []
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: registered.append((record, publish)) or record,
    )

    job_service._persist_result("job-pub", envelope)

    _, publish_flag = registered[0]
    assert publish_flag is True


def test_persist_result_uses_icechunk_when_pyramid_needed(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "open_climate_service.data_manager.services.downloader._needs_pyramid",
        lambda ds, x, y: True,
    )
    monkeypatch.setattr(
        "open_climate_service.data_manager.services.downloader._pyramid_levels",
        lambda ds, x, y: 2,
    )

    class _FakePyramidDt:
        attrs: dict[str, Any] = {}

        def to_zarr(self, path: Any, **kwargs: Any) -> None:
            pass  # path is an IcechunkStore, not a filesystem path

        def close(self) -> None:
            pass

    class _FakePyramid:
        dt = _FakePyramidDt()
        encoding: dict[str, Any] = {}

    monkeypatch.setattr("topozarr.coarsen.create_pyramid", lambda *a, **kw: _FakePyramid())
    monkeypatch.setattr(
        "open_climate_service.data_manager.services.downloader._write_root_time_coordinate",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _stub_get_dataset)
    registered: list[Any] = []
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: registered.append((record, publish)) or record,
    )

    ds = _small_dataset()
    ds.attrs["proj:code"] = "EPSG:4326"
    envelope = SaveResultEnvelope(ds, "Zarr", {"dataset_id": "big_dataset"})

    job_service._persist_result("job-pyramid", envelope)

    record, _ = registered[0]
    assert record.format == ArtifactFormat.ICECHUNK
    assert record.path == str(tmp_path / "big_dataset.icechunk")


def test_persist_result_falls_through_to_ephemeral_zarr_without_dataset_id(
    job_service: OpenEOJobService,
) -> None:
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {})
    output_path = job_service._persist_result("job-ephemeral", envelope)

    assert output_path is not None
    assert output_path.endswith(".zarr")


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a/b", "../../secret", "/abs/path"])
def test_persist_result_rejects_dataset_id_with_path_separators(job_service: OpenEOJobService, bad_id: str) -> None:
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": bad_id})
    with pytest.raises(ValueError, match="Invalid dataset_id"):
        job_service._persist_result("job-traversal", envelope)


def test_persist_result_rejects_unknown_dataset_id(
    job_service: OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service import config as api_config

    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", lambda _id: None)
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": "no_such_dataset"})
    with pytest.raises(ValueError, match="plugins_dir is not configured"):
        job_service._persist_result("job-unknown", envelope)


def test_persist_result_auto_registers_missing_template(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service import config as api_config
    from open_climate_service.data_registry.services import datasets as dataset_registry

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")

    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    registered: list[Any] = []
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: registered.append((record, publish)) or record,
    )

    ds = _small_dataset().isel(t=0, drop=True)
    envelope = SaveResultEnvelope(
        ds,
        "Zarr",
        {
            "dataset_id": "worldpop_population_change_autogen",
            "source_dataset_id": "worldpop_population_global2_R2025A_100m",
            "variable": "pop_change",
            "publish": False,
        },
    )

    result = job_service._persist_result("job-auto-template", envelope)

    assert result == "managed://worldpop_population_change_autogen"
    template = dataset_registry.get_dataset("worldpop_population_change_autogen")
    assert template is not None
    assert template["sync"]["kind"] == "static"
    assert template["variable"] == "pop_change"
    assert template["period_type"] == "yearly"
    assert template["units"] == "people"
    assert template["display"]["colormap"] == "RdBu"
    assert template["display"]["range"] == [-1.0, 1.0]

    record, publish_flag = registered[0]
    assert record.dataset_id == "worldpop_population_change_autogen"
    assert record.period_type == "yearly"
    assert publish_flag is False


def test_persist_result_reloads_template_after_concurrent_create(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service import config as api_config
    from open_climate_service.data_registry.services import datasets as dataset_registry

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")

    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)

    real_get_dataset = dataset_registry.get_dataset
    real_write_dataset_template = dataset_registry.write_dataset_template
    get_calls = {"count": 0}

    def _get_dataset(dataset_id: str) -> dict[str, Any] | None:
        if dataset_id == "worldpop_population_change_autogen":
            get_calls["count"] += 1
            if get_calls["count"] == 1:
                return None
        return real_get_dataset(dataset_id)

    def _write_dataset_template(template: dict[str, Any], *, overwrite: bool = False) -> Path:
        real_write_dataset_template(template, overwrite=overwrite)
        raise FileExistsError("simulated concurrent create")

    monkeypatch.setattr(dataset_registry, "get_dataset", _get_dataset)
    monkeypatch.setattr(dataset_registry, "write_dataset_template", _write_dataset_template)
    monkeypatch.setattr(
        "open_climate_service.ingestions.services.register_artifact_record",
        lambda record, publish: record,
    )

    ds = _small_dataset().isel(t=0, drop=True)
    envelope = SaveResultEnvelope(
        ds,
        "Zarr",
        {
            "dataset_id": "worldpop_population_change_autogen",
            "source_dataset_id": "worldpop_population_global2_R2025A_100m",
            "variable": "pop_change",
            "publish": False,
        },
    )

    result = job_service._persist_result("job-auto-template-race", envelope)

    assert result == "managed://worldpop_population_change_autogen"
    template = real_get_dataset("worldpop_population_change_autogen")
    assert template is not None


def test_persist_result_rejects_non_boolean_publish_option(
    job_service: OpenEOJobService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("open_climate_service.data_registry.services.datasets.get_dataset", _stub_get_dataset)
    envelope = SaveResultEnvelope(_small_dataset(), "Zarr", {"dataset_id": "ds", "publish": "false"})
    with pytest.raises(ValueError, match="'publish' option must be a boolean"):
        job_service._persist_result("job-bad-publish", envelope)


# ---------------------------------------------------------------------------
# _derive_variable
# ---------------------------------------------------------------------------


def test_derive_variable_from_options() -> None:
    ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])})
    assert _derive_variable(ds, {"variable": "a"}) == "a"


def test_derive_variable_rejects_nonexistent_variable() -> None:
    ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])})
    with pytest.raises(ValueError, match="not found in dataset"):
        _derive_variable(ds, {"variable": "temperature"})


def test_derive_variable_from_sole_data_var() -> None:
    ds = xr.Dataset({"precip": ("x", [1.0])})
    assert _derive_variable(ds, {}) == "precip"


def test_derive_variable_raises_for_multiple_vars_without_option() -> None:
    ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])})
    with pytest.raises(ValueError, match="multiple variables"):
        _derive_variable(ds, {})


# ---------------------------------------------------------------------------
# _infer_period_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("timedelta_days", "expected"),
    [
        (1, "daily"),
        (7, "weekly"),
        (30, "monthly"),
        (365, "yearly"),
    ],
)
def test_infer_period_type(timedelta_days: int, expected: str) -> None:
    t = np.arange(3).astype("timedelta64[D]") * timedelta_days + np.datetime64("2025-01-01", "D")
    ds = xr.Dataset({"v": ("t", [1.0, 2.0, 3.0])}, coords={"t": t})
    assert _infer_period_type(ds, "t") == expected


def _axis(ids: list[str]) -> xr.Dataset:
    t = np.array(ids, dtype="datetime64[ns]")
    return xr.Dataset({"v": ("t", np.arange(len(ids), dtype="float64"))}, coords={"t": t})


def test_infer_period_type_reads_a_real_dekadal_axis_as_dekadal() -> None:
    """The 32-day monthly bucket used to swallow dekads, mislabelling them monthly."""
    from open_climate_service.shared.time import dekad_period_ids

    assert _infer_period_type(_axis(dekad_period_ids("2026-01-01", "2026-12-31")), "t") == "dekadal"


def test_infer_period_type_recognises_a_short_dekadal_axis_across_a_month_boundary() -> None:
    """Feb 21 -> Mar 1 is 8 days, so any interval rule reads it as weekly.

    Dekads are defined by *starting* on the 1st, 11th or 21st, which is exact where a median
    is a guess.
    """
    assert _infer_period_type(_axis(["2026-02-21", "2026-03-01"]), "t") == "dekadal"


def test_infer_period_type_recognises_a_dekadal_axis_with_missing_dekads() -> None:
    """A gap is a real state — `aggregate_dekads` warns about it — and its median is 15.5 days."""
    assert _infer_period_type(_axis(["2026-01-01", "2026-01-11", "2026-02-01"]), "t") == "dekadal"


def test_infer_period_type_does_not_call_unrelated_ten_day_data_dekadal() -> None:
    """A regular 10-day series on other days of the month is not a dekadal cadence."""
    stride = [
        str(d)[:10]
        for d in np.arange(np.datetime64("2026-01-03"), np.datetime64("2026-03-01"), np.timedelta64(10, "D"))
    ]
    assert _infer_period_type(_axis(stride), "t") != "dekadal"


def test_infer_period_type_does_not_call_a_monthly_axis_dekadal() -> None:
    """Every month starts on the 1st, so the day-of-month test alone would accept it — and a
    monthly-on-the-11th axis is equally a subset of the dekad start days."""
    assert _infer_period_type(_axis([f"2026-{m:02d}-01" for m in range(1, 13)]), "t") == "monthly"
    assert _infer_period_type(_axis([f"2026-{m:02d}-11" for m in range(1, 13)]), "t") == "monthly"


def test_infer_period_type_still_separates_weekly_from_dekadal() -> None:
    weekly = [
        str(d)[:10] for d in np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-04-01"), np.timedelta64(7, "D"))
    ]
    assert _infer_period_type(_axis(weekly), "t") == "weekly"


def test_infer_period_type_returns_none_for_single_timestep() -> None:
    ds = xr.Dataset({"v": ("t", [1.0])}, coords={"t": np.array(["2025-01-01"], dtype="datetime64[D]")})
    assert _infer_period_type(ds, "t") is None


def test_infer_period_type_does_not_infer_quarterly() -> None:
    """Quarterly is in the STAC step map but unimplemented for ingest and coverage.

    `datetime_to_period_string` raises on it and `numpy_datetime_to_period_string` KeyErrors,
    so inferring it attached a cadence that failed the moment the artifact was written — and
    once it is rejected at registration, auto-registering a quarterly openEO result would fail
    outright. None is legal here: a managed output is static, and a static template may carry
    no cadence.
    """
    t = np.arange(3).astype("timedelta64[D]") * 90 + np.datetime64("2025-01-01", "D")
    ds = xr.Dataset({"v": ("t", [1.0, 2.0, 3.0])}, coords={"t": t})
    assert _infer_period_type(ds, "t") is None


def test_infer_period_type_returns_none_for_bimonthly_spacing() -> None:
    t = np.arange(3).astype("timedelta64[D]") * 60 + np.datetime64("2025-01-01", "D")
    ds = xr.Dataset({"v": ("t", [1.0, 2.0, 3.0])}, coords={"t": t})
    assert _infer_period_type(ds, "t") is None


# ---------------------------------------------------------------------------
# _derive_coverage
# ---------------------------------------------------------------------------


def test_derive_coverage_falls_back_to_reduced_dimensions_attrs() -> None:
    """After reduce_dimension the time dim is gone; recover range from attrs."""
    ds = xr.Dataset(
        {"temperature": (("y", "x"), np.ones((3, 4)))},
        coords={"y": [10.0, 20.0, 30.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    # Simulate what openeo-processes-dask injects on each variable after reduce_dimension
    ds["temperature"].attrs["reduced_dimensions_min_values"] = {"t": np.datetime64("2018-01-01")}
    ds["temperature"].attrs["reduced_dimensions_max_values"] = {"t": np.datetime64("2018-03-01")}

    coverage = _derive_coverage(ds, "x", "y", None)
    assert coverage.temporal.start == "2018-01-01"
    assert coverage.temporal.end == "2018-03-01"


def test_recover_temporal_from_attrs_returns_none_when_no_attrs() -> None:
    # A non-temporal output (e.g. a day-of-year/month climatology) genuinely has no
    # temporal extent, so the fallback is (None, None) — matching the None coverage
    # convention and the Optional CoverageTemporal fields used for normals.
    ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
    start, end = _recover_temporal_from_attrs(ds)
    assert start is None
    assert end is None


def test_recover_temporal_from_attrs_reads_from_dataset_attrs() -> None:
    ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
    ds.attrs["reduced_dimensions_min_values"] = {"t": np.datetime64("2020-06-01")}
    ds.attrs["reduced_dimensions_max_values"] = {"t": np.datetime64("2020-08-31")}
    start, end = _recover_temporal_from_attrs(ds)
    assert start == "2020-06-01"
    assert end == "2020-08-31"


def test_derive_coverage_returns_spatial_and_temporal_from_dataset() -> None:
    t = np.array(["2025-01-01", "2025-01-03"], dtype="datetime64[D]")
    ds = xr.Dataset(
        {"v": (("t", "y", "x"), np.ones((2, 3, 4)))},
        coords={"t": t, "y": [10.0, 20.0, 30.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    coverage = _derive_coverage(ds, "x", "y", "t")

    assert coverage.spatial.xmin == pytest.approx(1.0)
    assert coverage.spatial.xmax == pytest.approx(4.0)
    assert coverage.spatial.ymin == pytest.approx(10.0)
    assert coverage.spatial.ymax == pytest.approx(30.0)
    assert coverage.temporal.start == "2025-01-01"
    assert coverage.temporal.end == "2025-01-03"


def _reduced_dataset() -> Any:
    """A dataset shaped like `reduce_dimension` output: dict-valued bookkeeping attrs."""
    ds = xr.Dataset(
        {"precip": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    ds.attrs["reduced_dimensions_min_values"] = {"t": np.datetime64("2025-01-01")}
    ds.attrs["units"] = "mm/d"
    ds["precip"].attrs["reduced_dimensions_max_values"] = {"t": np.datetime64("2025-02-01")}
    return ds


@pytest.mark.parametrize("fmt", ["NETCDF", "ZARR"])
def test_write_raster_scrubs_attrs_no_writer_can_encode(tmp_path: Path, fmt: str) -> None:
    """netCDF rejects a dict attr outright; Zarr rejects it as non-JSON (CLIM-825)."""
    from open_climate_service.openeo.jobs import _write_raster

    output = _write_raster(_reduced_dataset(), tmp_path, fmt)

    assert output is not None
    assert Path(output).exists()


# JSON-serializability is Zarr's contract, not netCDF's, and they disagree both ways. Each case
# below records what `to_netcdf` actually does, so the filter is measured rather than assumed.
_NETCDF_ATTR_CASES = [
    ("dict_of_datetime64", {"t": np.datetime64("2025-01-01")}, False),
    ("dict_of_str", {"t": "2025-01-01"}, False),
    ("list_of_dict", [{"a": 1}], False),
    ("none", None, False),
    ("bool", True, False),
    ("ndarray", np.array([1.0, 2.0], dtype="float32"), True),
    ("numpy_scalar", np.float32(0.5), True),
    ("str", "mm/d", True),
    ("int", 3, True),
    ("list_of_str", ["a", "b"], True),
]


@pytest.mark.parametrize(("label", "value", "netcdf_can_encode"), _NETCDF_ATTR_CASES)
def test_netcdf_attr_filter_matches_what_the_writer_accepts(
    tmp_path: Path, label: str, value: object, netcdf_can_encode: bool
) -> None:
    """The filter must keep exactly what netCDF can write — no more, no less."""
    from open_climate_service.openeo.jobs import _netcdf_safe_attrs

    ds = xr.Dataset({"v": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])})
    ds.attrs["probe"] = value

    kept = "probe" in _netcdf_safe_attrs(ds).attrs
    assert kept == netcdf_can_encode, f"{label}: filter kept={kept}, writer accepts={netcdf_can_encode}"

    # And the writer agrees, so this table cannot drift from reality unnoticed.
    if netcdf_can_encode:
        _netcdf_safe_attrs(ds).to_netcdf(tmp_path / f"{label}.nc")
    else:
        with pytest.raises(TypeError):
            ds.to_netcdf(tmp_path / f"{label}-raw.nc")


def test_write_raster_netcdf_drops_a_json_safe_dict_attr(tmp_path: Path) -> None:
    """A dict of plain strings survives a JSON scrub but still breaks to_netcdf."""
    from open_climate_service.openeo.jobs import _write_raster

    ds = _reduced_dataset()
    ds.attrs["json_safe_dict"] = {"t": "2025-01-01"}

    output = _write_raster(ds, tmp_path, "NETCDF")

    assert output is not None
    reopened = xr.open_dataset(output)
    try:
        assert "json_safe_dict" not in reopened.attrs
    finally:
        reopened.close()


def test_write_raster_netcdf_keeps_array_attrs_a_json_scrub_would_drop(tmp_path: Path) -> None:
    """netCDF writes arrays and numpy scalars happily; the JSON scrub would discard them."""
    from open_climate_service.openeo.jobs import _write_raster

    ds = _reduced_dataset()
    ds.attrs["valid_range"] = np.array([0.0, 100.0], dtype="float32")
    ds.attrs["scale_factor"] = np.float32(0.1)

    output = _write_raster(ds, tmp_path, "NETCDF")

    assert output is not None
    reopened = xr.open_dataset(output)
    try:
        assert list(reopened.attrs["valid_range"]) == [0.0, 100.0]
        assert reopened.attrs["scale_factor"] == pytest.approx(0.1)
    finally:
        reopened.close()


def test_write_raster_keeps_attrs_the_writer_can_encode(tmp_path: Path) -> None:
    """The scrub must drop only what cannot be written, not all metadata."""
    from open_climate_service.openeo.jobs import _write_raster

    output = _write_raster(_reduced_dataset(), tmp_path, "NETCDF")
    assert output is not None

    reopened = xr.open_dataset(output)
    try:
        assert reopened.attrs["units"] == "mm/d"
        assert "reduced_dimensions_min_values" not in reopened.attrs
        assert "reduced_dimensions_max_values" not in reopened["precip"].attrs
    finally:
        reopened.close()


def test_result_route_serves_netcdf_after_reduce_dimension(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported symptom: a graph ending in reduce_dimension, exported as NETCDF, 500d."""
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process, request=None: SaveResultEnvelope(_reduced_dataset(), "NETCDF"),
    )

    response = client.post(
        "/result",
        json={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
    )

    assert response.status_code == 200, response.text
    assert response.content


def test_batch_job_with_unwritable_format_errors_without_a_result_asset(
    job_service: OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch half of CLIM-909: the job must fail, not finish with mislabelled output.

    Before the fix `_write_raster` fell back to Zarr, so the job wrote `result.zarr`, was marked
    FINISHED, and advertised it as the requested format — quieter than the synchronous 500 and
    harder to notice.
    """
    ds = xr.Dataset(
        {"temperature": xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=["y", "x"])},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        "open_climate_service.openeo.execution.run_process_graph",
        lambda process: SaveResultEnvelope(ds, "GEOJSON"),
    )

    record = job_service.create_job(
        OpenEOJobCreate(
            process={"process_graph": {"result": {"process_id": "save_result", "result": True}}},
        )
    )
    job_service._execute(record.id)

    failed = job_service.get_job_or_404(record.id)
    assert failed.status == OpenEOJobStatus.ERROR
    assert "describes vector features" in (failed.error_message or "")
    assert not (failed.usage or {}).get("output_path")

    with pytest.raises(HTTPException) as exc_info:
        job_service.get_results(record.id)
    assert exc_info.value.status_code == 424


@pytest.mark.parametrize(
    ("units", "expected_colormap"),
    [("degC", "rdbu_r"), ("K", "rdbu_r"), ("mm/d", "RdBu"), ("%", "RdBu")],
)
def test_derived_anomaly_colormap_runs_the_way_the_variable_is_read(units: str, expected_colormap: str) -> None:
    """A diverging scale is not symmetric in meaning: warm is red, wet is blue.

    Defaulting every signed output to one direction rendered a temperature anomaly with warm
    in blue, or precipitation with wet in red — plausible-looking maps that invert the reading.
    """
    from open_climate_service.openeo.jobs import _derive_display_config

    ds = xr.Dataset(
        {"value": xr.DataArray(np.array([[-1.0, 1.0]], dtype=np.float32), dims=("y", "x"))},
    )
    ds["value"].attrs["units"] = units

    display = _derive_display_config(ds, "value", "derived_anomaly", {}, None, "Anomaly")

    assert display["colormap"] == expected_colormap
