"""GeoParquet as an artifact format: the record shape, the gates, and the branches (CLIM-1067)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from open_climate_service.ingestions import services
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    DatasetItemType,
    FeatureDetail,
    PublicationStatus,
)
from open_climate_service.openeo import execution as openeo_execution
from open_climate_service.stac import services as stac_services

DISTRICTS_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}


def _feature(code: str, coordinates: list[list[list[float]]]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {"type": "Polygon", "coordinates": coordinates},
    }


def _feature_collection() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            _feature("SL-W", [[[-13.5, 6.9], [-12.0, 6.9], [-12.0, 8.0], [-13.5, 8.0], [-13.5, 6.9]]]),
            _feature("SL-N", [[[-12.5, 8.5], [-10.1, 8.5], [-10.1, 10.0], [-12.5, 10.0], [-12.5, 8.5]]]),
        ],
    }


def _feature_artifact(
    *,
    artifact_id: str = "f1",
    dataset_id: str = "districts",
    status: PublicationStatus = PublicationStatus.PUBLISHED,
    path: str = "/tmp/districts.parquet",
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        dataset_name="District boundaries",
        variable=None,
        period_type=None,
        format=ArtifactFormat.GEOPARQUET,
        path=path,
        asset_paths=[path],
        variables=[],
        request_scope=ArtifactRequestScope(start=None, end=None, bbox=(-13.5, 6.9, -10.1, 10.0)),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=-13.5, ymin=6.9, xmax=-10.1, ymax=10.0),
            temporal=CoverageTemporal(start=None, end=None),
        ),
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(status=status, collection_id=dataset_id),
        features=FeatureDetail(
            id_property="orgUnitCode", feature_count=2, primary_geometry="geometry", crs="EPSG:4326"
        ),
    )


def _raster_artifact(*, artifact_id: str = "r1", dataset_id: str = "chirps3_precipitation_daily") -> ArtifactRecord:
    path = f"/tmp/{dataset_id}.icechunk"
    return ArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        period_type="daily",
        format=ArtifactFormat.ICECHUNK,
        path=path,
        asset_paths=[path],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-10", bbox=(1.0, 2.0, 3.0, 4.0)),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-10"),
        ),
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=dataset_id),
    )


@pytest.fixture(autouse=True)
def materialized_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "_artifact_storage_exists", lambda _: True)


# --- the record shape --------------------------------------------------------------------


def test_feature_detail_cannot_exist_without_an_id_property() -> None:
    """The one value whose loss does not raise, so the model is where it is made to."""
    with pytest.raises(ValidationError):
        FeatureDetail(feature_count=2, primary_geometry="geometry")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        FeatureDetail(id_property="", feature_count=2, primary_geometry="geometry", crs="EPSG:4326")


def test_a_raster_record_carries_no_feature_detail() -> None:
    """Nested rather than flattened, so a raster does not hold three null vector fields."""
    assert _raster_artifact().features is None


def test_feature_detail_requires_an_explicit_crs() -> None:
    """ADR 0002 decision 9: every stored collection carries one, never assumed at read time."""
    with pytest.raises(ValidationError):
        FeatureDetail(id_property="orgUnitCode", feature_count=2, primary_geometry="geometry")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"id_property": " orgUnitCode "}, "whitespace", id="padded_id_property"),
        pytest.param({"id_property": "   "}, "not blank space", id="blank_id_property"),
        pytest.param({"primary_geometry": " geometry "}, "whitespace", id="padded_geometry_column"),
        pytest.param({"primary_geometry": "   "}, "not blank space", id="blank_geometry_column"),
        pytest.param({"crs": "   "}, "invalid crs", id="blank_crs"),
        pytest.param({"crs": "WGS 84"}, "invalid crs", id="crs_that_is_a_label_not_a_code"),
        pytest.param({"crs": "EPSG"}, "invalid crs", id="crs_with_no_code"),
    ],
)
def test_feature_detail_refuses_a_value_no_reader_could_use(overrides: dict[str, Any], expected: str) -> None:
    """Enforced on the model, so a record loaded from disk gets the same guarantee as a new one.

    A padded column name is the quiet failure: it matches no property, so every lookup misses
    and the identifier reads as absent rather than wrong.
    """
    fields: dict[str, Any] = {
        "id_property": "orgUnitCode",
        "feature_count": 2,
        "primary_geometry": "geometry",
        "crs": "EPSG:4326",
    }
    fields.update(overrides)

    with pytest.raises(ValidationError, match=expected):
        FeatureDetail(**fields)


@pytest.mark.parametrize("declared", ["OGC:CRS84", "CRS84", "urn:ogc:def:crs:OGC:1.3:CRS84", "4326", " EPSG:4326 "])
def test_feature_detail_stores_one_spelling_of_wgs84(declared: str) -> None:
    """Canonicalized at the boundary, so two records naming WGS 84 do not compare unequal."""
    detail = FeatureDetail(id_property="orgUnitCode", feature_count=2, primary_geometry="geometry", crs=declared)

    assert detail.crs == "EPSG:4326"


def _record_fields(**overrides: Any) -> dict[str, Any]:
    fields = _raster_artifact().model_dump()
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {"format": ArtifactFormat.GEOPARQUET, "features": None, "variable": None},
            "must carry a 'features' detail",
            id="geoparquet_without_feature_detail",
        ),
        pytest.param(
            {
                "format": ArtifactFormat.GEOPARQUET,
                "features": {
                    "id_property": "orgUnitCode",
                    "feature_count": 2,
                    "primary_geometry": "geometry",
                    "crs": "EPSG:4326",
                },
            },
            "properties rather than a measured variable",
            id="geoparquet_claiming_a_raster_variable",
        ),
        pytest.param(
            {
                "features": {
                    "id_property": "orgUnitCode",
                    "feature_count": 2,
                    "primary_geometry": "geometry",
                    "crs": "EPSG:4326",
                }
            },
            "must not carry feature detail",
            id="zarr_carrying_feature_detail",
        ),
        pytest.param({"variable": None}, "must name the raster variable", id="raster_without_a_variable"),
        pytest.param({"variable": "   "}, "must name the raster variable", id="raster_with_a_blank_variable"),
    ],
)
def test_a_record_shape_that_contradicts_its_format_is_refused(overrides: dict[str, Any], expected: str) -> None:
    """The invariant is enforced on the model, not left to each construction site.

    `variable` was relaxed to optional for feature collections alone, so a raster keeps the
    requirement it has always had — and a GeoParquet cannot exist without an `id_property`,
    which is the whole point of nesting it in a required submodel.
    """
    with pytest.raises(ValidationError, match=expected):
        ArtifactRecord(**_record_fields(**overrides))


def test_the_records_this_release_actually_writes_are_accepted() -> None:
    """The complement of the parametrized refusals: both valid shapes still construct."""
    assert _raster_artifact().variable == "precip"
    assert _feature_artifact().features is not None


# --- the vector entry point --------------------------------------------------------------


def _tmp_record_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    store_path = tmp_path / "districts.parquet"
    store_path.write_bytes(b"PAR1")
    return store_path


def test_create_feature_artifact_registers_and_publishes_a_geoparquet_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="EPSG:4326",
        bbox=[-13.5, 6.9, -10.1, 10.0],
    )

    assert record.format == ArtifactFormat.GEOPARQUET
    assert record.publication.status == PublicationStatus.PUBLISHED
    assert record.features == FeatureDetail(
        id_property="orgUnitCode", feature_count=2, primary_geometry="geometry", crs="EPSG:4326"
    )
    # A boundary set measures nothing and has no temporal axis; both fields are optional so
    # this record does not have to invent values for them.
    assert record.variable is None
    assert record.period_type is None
    assert record.coverage.temporal == CoverageTemporal(start=None, end=None)
    # GeoJSON is WGS 84 by RFC 7946, so `spatial` already is the WGS 84 extent.
    assert record.coverage.spatial_wgs84 is None
    assert [stored.artifact_id for stored in services._load_records()] == [record.artifact_id]


def test_create_feature_artifact_derives_the_extent_from_the_geometries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Computed from the shapes, not read from an advertised bbox that nothing keeps in step."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    collection = _feature_collection()
    collection["bbox"] = [0.0, 0.0, 1.0, 1.0]

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=collection,
        store_path=store_path,
        crs="EPSG:4326",
    )

    assert record.coverage.spatial == CoverageSpatial(xmin=-13.5, ymin=6.9, xmax=-10.1, ymax=10.0)


def test_create_feature_artifact_reads_every_geojson_geometry_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One coordinate walk covers Point through GeometryCollection, elevation included."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"orgUnitCode": "clinic-1"},
                "geometry": {"type": "Point", "coordinates": [-11.0, 8.0, 42.0]},
            },
            {
                "type": "Feature",
                "properties": {"orgUnitCode": "mixed"},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "MultiPoint", "coordinates": [[-13.0, 6.0], [-10.0, 7.0]]},
                        {"type": "LineString", "coordinates": [[-12.0, 9.0], [-11.0, 11.0]]},
                    ],
                },
            },
        ],
    }

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=collection,
        store_path=store_path,
        crs="EPSG:4326",
    )

    assert record.coverage.spatial == CoverageSpatial(xmin=-13.0, ymin=6.0, xmax=-10.0, ymax=11.0)
    assert record.features is not None
    assert record.features.feature_count == 2


def test_create_feature_artifact_names_the_geometry_column_the_writer_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="EPSG:4326",
        primary_geometry="boundary",
    )

    assert record.features is not None
    assert record.features.primary_geometry == "boundary"


def test_create_feature_artifact_refreshes_a_collection_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No version history: a refresh replaces the record rather than appending one."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    first = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE, features=_feature_collection(), store_path=store_path, crs="EPSG:4326"
    )
    grown = _feature_collection()
    grown["features"].append(  # type: ignore[union-attr]
        _feature("SL-E", [[[-11.5, 7.0], [-10.5, 7.0], [-10.5, 8.0], [-11.5, 8.0], [-11.5, 7.0]]])
    )

    second = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE, features=grown, store_path=store_path, crs="EPSG:4326"
    )

    assert second.artifact_id == first.artifact_id
    assert second.features is not None
    assert second.features.feature_count == 3
    assert len(services._load_records()) == 1


@pytest.mark.parametrize(
    ("template", "features", "expected"),
    [
        pytest.param(
            {"id": "districts", "name": "District boundaries"},
            _feature_collection(),
            "id_property",
            id="template_without_an_id_property",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            {"type": "Feature", "properties": {}, "geometry": None},
            "FeatureCollection",
            id="payload_that_is_not_a_feature_collection",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            {"type": "FeatureCollection", "features": []},
            "empty",
            id="provider_that_returned_nothing",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": None}]},
            "no usable geometry",
            id="features_without_geometry",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            {"type": "FeatureCollection", "features": [_feature("SL-W", [[[0.0, 0.0]]]), "not-a-feature"]},
            "malformed members at 1",
            id="member_that_is_not_an_object",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            {
                "type": "FeatureCollection",
                "features": [
                    _feature("SL-W", [[[0.0, 0.0]]]),
                    {"type": "NotAFeature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}},
                ],
            },
            "malformed members at 1",
            id="object_that_does_not_say_it_is_a_feature",
        ),
        pytest.param(
            {**DISTRICTS_TEMPLATE, "id_property": " orgUnitCode "},
            _feature_collection(),
            "without padding",
            id="template_field_with_padding",
        ),
    ],
)
def test_create_feature_artifact_refuses_input_that_cannot_produce_a_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template: dict[str, object],
    features: dict[str, Any],
    expected: str,
) -> None:
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=expected):
        services.create_feature_artifact(template=template, features=features, store_path=store_path, crs="EPSG:4326")

    assert not services.ARTIFACTS_INDEX_PATH.exists() or services._load_records() == []


def test_create_feature_artifact_rejects_a_malformed_member_rather_than_dropping_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A filtered member would leave feature_count describing neither the input nor the store."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    collection = _feature_collection()
    collection["features"].insert(1, ["not", "a", "feature"])

    with pytest.raises(ValueError, match="malformed members at 1"):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE, features=collection, store_path=store_path, crs="EPSG:4326"
        )


def test_create_feature_artifact_canonicalizes_the_crs84_spelling_geojson_uses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One CRS, stored in one spelling, whichever alias the provider declared."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="OGC:CRS84",
    )

    assert record.features is not None
    assert record.features.crs == "EPSG:4326"


def test_create_feature_artifact_refuses_a_reprojected_store_it_cannot_describe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The extent comes from GeoJSON, which is WGS 84; recording it against a projected store
    is the silent mismatch ADR 0002 decision 9 exists to prevent."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="CLIM-1068"):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_feature_collection(),
            store_path=store_path,
            crs="EPSG:3857",
        )


# --- the gates ---------------------------------------------------------------------------


def test_neither_catalogue_admits_a_feature_collection_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposure lands atomically with the collection document in CLIM-1069, not before.

    The record exists, is published, and is listed under `/datasets` — but STAC would have to
    advertise a collection URL whose document this build cannot render, so the gate stays shut.
    """
    monkeypatch.setattr(
        services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_raster_artifact(), _feature_artifact()]),
    )

    assert list(services.stac_eligible_artifacts_by_dataset()) == ["chirps3_precipitation_daily"]
    assert list(services.latest_published_raster_artifacts_by_dataset()) == ["chirps3_precipitation_daily"]
    assert ArtifactFormat.GEOPARQUET not in services.LOADABLE_RASTER_FORMATS


def test_an_unpublished_feature_collection_reaches_neither_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_feature_artifact(status=PublicationStatus.UNPUBLISHED)]),
    )

    assert services.stac_eligible_artifacts_by_dataset() == {}


def test_no_surface_advertises_a_collection_url_that_would_not_resolve(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalogue, the openEO listing and the dataset links all agree it is not there."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    catalog = client.get("/stac/catalog.json").json()
    dataset = client.get("/datasets/districts").json()

    assert all("districts" not in link["href"] for link in catalog["links"])
    assert client.get("/stac/collections/districts").status_code == 404
    assert client.get("/collections/districts").status_code == 404
    assert client.get("/collections").json()["collections"] == []
    assert {link["rel"] for link in dataset["links"]} == {"self"}


# --- the format branches -----------------------------------------------------------------


def test_stac_collection_builder_is_never_reached_for_a_feature_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain 404, which is true, rather than a 501 behind a link the catalogue advertised."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    assert client.get("/stac/collections/districts").status_code == 404


def test_openeo_refuses_to_open_a_feature_collection_as_a_datacube() -> None:
    """Enumerated formats, so this never falls through to open_zarr_dataset on a parquet file."""
    with pytest.raises(HTTPException) as excinfo:
        openeo_execution._open_artifact(_feature_artifact())

    assert excinfo.value.status_code == 409
    assert "raster datacube" in excinfo.value.detail


def test_zarr_route_does_not_serve_a_feature_collection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The raster gate is what the store routes read, so a feature collection is simply not there."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    assert client.get("/zarr/districts/zarr.json").status_code == 404


@pytest.mark.parametrize("plan_only", [True, False])
def test_raster_sync_refuses_a_feature_collection(monkeypatch: pytest.MonkeyPatch, plan_only: bool) -> None:
    """Sync is period arithmetic; a collection with no temporal axis refreshes via its provider."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    with pytest.raises(HTTPException) as excinfo:
        if plan_only:
            services.plan_sync_dataset(dataset_id="districts", end=None)
        else:
            services.sync_dataset(dataset_id="districts", end=None, publish=True)

    assert excinfo.value.status_code == 409
    assert "feature collection" in excinfo.value.detail


# --- itemType on the dataset record -------------------------------------------------------


def test_datasets_discriminate_feature_collections_from_rasters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "_load_records", lambda: [_raster_artifact(), _feature_artifact()])
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda dataset_id: {"period_type": "daily", "units": "mm"} if dataset_id != "districts" else None,
    )

    by_id = {dataset.dataset_id: dataset for dataset in services.list_datasets().items}

    assert by_id["chirps3_precipitation_daily"].item_type == DatasetItemType.COVERAGE
    assert by_id["chirps3_precipitation_daily"].variable == "precip"
    assert by_id["chirps3_precipitation_daily"].period_type == "daily"
    assert by_id["districts"].item_type == DatasetItemType.FEATURE
    assert by_id["districts"].variable is None
    assert by_id["districts"].period_type is None


def test_dataset_list_response_spells_the_discriminator_as_ogc_does(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`itemType` is OGC API - Features' own field name, so the wire form keeps its camelCase."""
    monkeypatch.setattr(services, "_load_records", lambda: [_raster_artifact(), _feature_artifact()])
    monkeypatch.setattr(services.registry_datasets, "get_dataset", lambda _: None)

    payload = client.get("/datasets").json()

    assert {item["dataset_id"]: item["itemType"] for item in payload["items"]} == {
        "chirps3_precipitation_daily": "coverage",
        "districts": "feature",
    }
    assert all("item_type" not in item for item in payload["items"])


def test_dataset_detail_omits_variable_and_period_type_for_a_feature_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "_load_records", lambda: [_feature_artifact()])
    monkeypatch.setattr(services.registry_datasets, "get_dataset", lambda _: None)

    payload = client.get("/datasets/districts").json()

    assert payload["itemType"] == "feature"
    assert payload["variable"] is None
    assert payload["period_type"] is None
    assert [version["format"] for version in payload["versions"]] == ["geoparquet"]


def test_dataset_links_offer_neither_zarr_nor_stac_for_a_feature_collection() -> None:
    """The links track the gates exactly, so `/datasets` never points at a 404."""
    links = services._dataset_links("districts", _feature_artifact())

    assert [link.rel for link in links] == ["self"]


def test_stac_collection_builder_still_serves_a_raster(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every raster path through the builder is untouched by the feature collection work."""
    raster = _raster_artifact()
    monkeypatch.setattr(services, "stac_eligible_artifacts_by_dataset", lambda: {"chirps3_precipitation_daily": raster})
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "chirps3_precipitation_daily",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {},
            "cube:variables": {},
            "assets": {"zarr": {}},
        },
    )
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {})

    response = client.get("/stac/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    assert response.json()["id"] == "chirps3_precipitation_daily"
