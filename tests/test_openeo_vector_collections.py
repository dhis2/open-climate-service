"""Vector datasets as openEO collections, loaded with `load_collection` (CLIM-1326)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.openeo import execution
from open_climate_service.stac import services as stac_services

DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.3.0/schema.json"
REGIONS_TEMPLATE = {"id": "regions", "name": "Regions", "id_property": "code"}


def _box(code: str, xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    ring = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    return {"type": "Feature", "properties": {"code": code}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "vectors"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    monkeypatch.setattr(stac_services, "_clear_xstac_collection_cache", lambda: None, raising=False)
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [REGIONS_TEMPLATE])
    feature_templates.reset_feature_template_caches()
    return root


@pytest.fixture
def regions(monkeypatch: pytest.MonkeyPatch) -> ArtifactRecord:
    """Two published polygons over a 4 x 4 degree area, stored as real GeoParquet."""
    record = feature_services.refresh_feature_collection(
        template=REGIONS_TEMPLATE,
        features={"type": "FeatureCollection", "features": [_box("WEST", 0, 0, 2, 4), _box("EAST", 2, 0, 4, 4)]},
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)
    return record


def _raster_record(dataset_id: str = "rain") -> ArtifactRecord:
    path = f"/tmp/{dataset_id}.icechunk"
    return ArtifactRecord(
        artifact_id="r1",
        dataset_id=dataset_id,
        dataset_name="Rain",
        variable="tp",
        period_type="monthly",
        format=ArtifactFormat.ICECHUNK,
        path=path,
        asset_paths=[path],
        variables=["tp"],
        request_scope=ArtifactRequestScope(start="2025-01", end="2025-02"),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=0.0, ymin=0.0, xmax=4.0, ymax=4.0),
            temporal=CoverageTemporal(start="2025-01-01", end="2025-02-01"),
        ),
        created_at=datetime(2025, 3, 1, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=dataset_id),
    )


def _rain_cube(_artifact: Any) -> xr.Dataset:
    """Two months on a 1-degree grid: the west half is 1.0, the east half 3.0."""
    values = np.ones((2, 4, 4), dtype="float32")
    values[:, :, 2:] = 3.0
    ds = xr.Dataset(
        {"tp": (("t", "y", "x"), values)},
        coords={
            "t": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
            "y": [3.5, 2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
    )
    return ds.rio.write_crs("EPSG:4326")


# --- listing and the collection document ----------------------------------------------------


def test_a_published_vector_dataset_is_listed_as_an_openeo_collection(
    client: TestClient, regions: ArtifactRecord
) -> None:
    listed = client.get("/collections").json()["collections"]

    assert [collection["id"] for collection in listed] == ["regions"]


def test_the_collection_document_describes_a_vector_cube(client: TestClient, regions: ArtifactRecord) -> None:
    collection = client.get("/collections/regions").json()

    assert collection["cube:dimensions"] == {
        "geometry": {
            "type": "geometry",
            "axes": ["x", "y"],
            "bbox": [0.0, 0.0, 4.0, 4.0],
            "reference_system": 4326,
            "geometry_types": ["Polygon"],
        }
    }
    assert DATACUBE_EXTENSION in collection["stac_extensions"]
    # The feature properties stay where STAC puts them, and no raster fields are invented.
    assert {column["name"] for column in collection["table:columns"]} >= {"code"}
    assert "cube:variables" not in collection
    assert collection["extent"]["temporal"]["interval"] == [[None, None]]
    assert all("/stac/collections" not in link["href"] for link in collection["links"])


def test_the_listing_and_the_detail_route_agree(client: TestClient, regions: ArtifactRecord) -> None:
    listed = client.get("/collections").json()["collections"][0]

    assert listed["cube:dimensions"] == client.get("/collections/regions").json()["cube:dimensions"]


def test_the_collection_document_is_built_from_the_record_openeo_selected(
    client: TestClient, regions: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Looking the record up again could return a newer one mid-refresh and mix two into one document."""

    def must_not_look_up(*args: Any) -> Any:
        raise AssertionError("the openEO collection looked its record up again")

    monkeypatch.setattr(stac_services, "build_collection", must_not_look_up)

    assert client.get("/collections/regions").status_code == 200
    assert client.get("/collections").json()["collections"][0]["id"] == "regions"


def test_an_unpublished_vector_dataset_is_not_an_openeo_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_services.refresh_feature_collection(
        template=REGIONS_TEMPLATE,
        features={"type": "FeatureCollection", "features": [_box("WEST", 0, 0, 2, 4)]},
        publish=False,
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert client.get("/collections").json()["collections"] == []
    assert client.get("/collections/regions").status_code == 404


def test_three_dimensional_geometry_types_are_named_by_their_geojson_type(
    client: TestClient, regions: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GeoParquet may declare "Polygon Z"; the openEO API enumerates only GeoJSON type names."""
    from open_climate_service.features import store as feature_store

    monkeypatch.setattr(feature_store, "stored_geometry_types", lambda _record: ["Polygon Z", "Point"])

    dimension = client.get("/collections/regions").json()["cube:dimensions"]["geometry"]

    assert dimension["geometry_types"] == ["Point", "Polygon"]


# --- load_collection --------------------------------------------------------------------------


@pytest.fixture
def rain_and_regions(monkeypatch: pytest.MonkeyPatch, regions: ArtifactRecord) -> None:
    """A published raster next to the published feature collection; only the raster's store is faked."""
    published = {**ingestion_services.openeo_collection_artifacts_by_dataset(), "rain": _raster_record()}
    monkeypatch.setattr(ingestion_services, "openeo_collection_artifacts_by_dataset", lambda: published)
    monkeypatch.setattr(execution, "_open_artifact", _rain_cube)


def test_load_collection_loads_a_vector_collection_as_load_features_does(rain_and_regions: None) -> None:
    from open_climate_service.plugins.processes.load_features import load_features

    loaded = execution._load_collection_impl("regions")

    assert loaded == load_features("regions")
    assert [feature["id"] for feature in loaded["features"]] == ["WEST", "EAST"]


def test_load_collection_takes_the_arguments_the_openeo_editor_builds(rain_and_regions: None) -> None:
    """Dragging a collection into the editor's model gives its bbox, `temporal_extent: null` and `bands: null`."""
    loaded = execution._load_collection_impl(
        "regions",
        spatial_extent={"west": 0.0, "south": 0.0, "east": 4.0, "north": 4.0},
        temporal_extent=None,
        bands=None,
    )

    assert len(loaded["features"]) == 2


def test_load_collection_narrows_a_vector_collection_by_its_spatial_extent(rain_and_regions: None) -> None:
    loaded = execution._load_collection_impl(
        "regions", spatial_extent={"west": 0.1, "south": 0.1, "east": 1.0, "north": 1.0}
    )

    assert [feature["id"] for feature in loaded["features"]] == ["WEST"]


@pytest.mark.parametrize("bands", [["code"], [], "code"])
def test_load_collection_refuses_bands_for_a_vector_collection(rain_and_regions: None, bands: Any) -> None:
    """Any `bands` is refused, also an empty list or a malformed value, not only a selection."""
    with pytest.raises(HTTPException) as refused:
        execution._load_collection_impl("regions", bands=bands)

    assert refused.value.status_code == 400
    assert "has no bands" in str(refused.value.detail)


@pytest.mark.parametrize(
    "extent",
    ["0,0,4,4", 5, [0, 0, 4, 4], {"west": "a", "south": 0, "east": 4, "north": 4}, {"west": 0}],
)
def test_a_malformed_spatial_extent_is_a_client_error_naming_the_argument(rain_and_regions: None, extent: Any) -> None:
    """`load_collection` has no process spec to reject these, so the vector path must, by name."""
    graph = {
        "process_graph": {
            "zones": {
                "process_id": "load_collection",
                "arguments": {"id": "regions", "spatial_extent": extent},
                "result": True,
            }
        }
    }

    with pytest.raises(HTTPException) as refused:
        execution.run_process_graph(graph)

    assert refused.value.status_code == 400
    assert "load_collection: spatial_extent" in str(refused.value.detail)
    assert "load_features" not in str(refused.value.detail)


def test_load_collection_loads_a_published_collection_whose_template_is_removed(
    rain_and_regions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What `/collections` advertises stays loadable after the instance drops its feature template."""
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [])
    feature_templates.reset_feature_template_caches()

    loaded = execution._load_collection_impl("regions")

    assert [feature["id"] for feature in loaded["features"]] == ["WEST", "EAST"]


def test_a_vector_collection_feeds_aggregate_spatial_through_a_process_graph(rain_and_regions: None) -> None:
    graph = {
        "process_graph": {
            "rain": {
                "process_id": "load_collection",
                "arguments": {"id": "rain", "temporal_extent": ["2025-01-01", "2025-02-28"]},
            },
            "zones": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "regions",
                    "spatial_extent": {"west": 0.0, "south": 0.0, "east": 4.0, "north": 4.0},
                    "temporal_extent": None,
                },
            },
            "zonal": {
                "process_id": "aggregate_spatial",
                "arguments": {
                    "data": {"from_node": "rain"},
                    "geometries": {"from_node": "zones"},
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
                "result": True,
            },
        }
    }

    result = execution.run_process_graph(graph)

    data = result if isinstance(result, xr.DataArray) else next(iter(result.data_vars.values()))
    means = data.isel(t=0).to_series()
    assert means.to_dict() == {"WEST": 1.0, "EAST": 3.0}


def test_the_python_client_graph_runs(rain_and_regions: None) -> None:
    """Exactly the graph the openEO Python client builds for `aggregate_spatial(geometries=<load_collection>)`."""
    openeo_datacube = pytest.importorskip("openeo.rest.datacube")
    rain = openeo_datacube.DataCube.load_collection(
        "rain", connection=None, temporal_extent=["2025-01-01", "2025-02-28"], fetch_metadata=False
    )
    regions = openeo_datacube.DataCube.load_collection("regions", connection=None, fetch_metadata=False)
    graph = rain.aggregate_spatial(geometries=regions, reducer="mean").flat_graph()

    result = execution.run_process_graph({"process_graph": graph})

    data = result if isinstance(result, xr.DataArray) else next(iter(result.data_vars.values()))
    assert data.isel(t=1).to_series().to_dict() == {"WEST": 1.0, "EAST": 3.0}


def test_raster_collections_load_as_before(rain_and_regions: None) -> None:
    cube = execution._load_collection_impl("rain", temporal_extent=["2025-01-01", "2025-01-31"])

    assert isinstance(cube, xr.DataArray)
    assert cube.sizes["t"] == 1


# --- the map viewer ---------------------------------------------------------------------------


def test_the_map_viewer_leaves_vector_collections_out_of_its_list(client: TestClient) -> None:
    """It draws rasters only, and fills its list from /collections, which now holds vectors too."""
    page = client.get("/map").text

    assert "function isVectorCollection(col)" in page
    assert 'dim?.type === "geometry"' in page
    assert "all.filter((col) => !isVectorCollection(col))" in page


def test_the_map_viewer_tells_a_vector_dataset_from_an_unpublished_one(client: TestClient) -> None:
    """A vector-only instance, or a deep link to a vector, is not reported as nothing published."""
    page = client.get("/map").text

    assert "vectorCollectionIds = new Set(all.filter(isVectorCollection)" in page
    assert "No published raster datasets found. The map shows raster datasets only." in page
    assert "vectorCollectionIds.has(requested)" in page
    assert "is a vector dataset; the map shows raster datasets only." in page
