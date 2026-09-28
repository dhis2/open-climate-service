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
from shapely.geometry import GeometryCollection, Point, Polygon, mapping

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

BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

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


@pytest.mark.parametrize(
    "declared",
    ["OGC:CRS84", "CRS84", "urn:ogc:def:crs:OGC:1.3:CRS84", "4326", " EPSG:4326 ", "epsg:4326", "ePsG:4326"],
)
def test_feature_detail_stores_one_spelling_of_wgs84(declared: str) -> None:
    """Canonicalized at the boundary, so two records naming WGS 84 do not compare unequal."""
    detail = FeatureDetail(id_property="orgUnitCode", feature_count=2, primary_geometry="geometry", crs=declared)

    assert detail.crs == "EPSG:4326"


def test_feature_detail_uppercases_the_authority_but_leaves_the_code_alone() -> None:
    """The authority is a register name and case is not part of it; the code is its token."""
    detail = FeatureDetail(id_property="orgUnitCode", feature_count=2, primary_geometry="geometry", crs="esri:102008")

    assert detail.crs == "ESRI:102008"


_FEATURE_DETAIL_FIELDS: dict[str, Any] = {
    "id_property": "orgUnitCode",
    "feature_count": 2,
    "primary_geometry": "geometry",
    "crs": "EPSG:4326",
}


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
                "features": _FEATURE_DETAIL_FIELDS,
            },
            "properties rather than a measured variable",
            id="geoparquet_claiming_a_raster_variable",
        ),
        pytest.param(
            {"features": _FEATURE_DETAIL_FIELDS},
            "must not carry feature detail",
            id="zarr_carrying_feature_detail",
        ),
        pytest.param(
            {
                "format": ArtifactFormat.GEOPARQUET,
                "variable": None,
                "variables": ["precip"],
                "features": _FEATURE_DETAIL_FIELDS,
            },
            "rather than data variables",
            id="geoparquet_listing_data_variables",
        ),
        pytest.param(
            {
                "format": ArtifactFormat.GEOPARQUET,
                "variable": None,
                "variables": [],
                "period_type": "daily",
                "features": _FEATURE_DETAIL_FIELDS,
            },
            "no period axis",
            id="geoparquet_claiming_a_period_axis",
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
    import geopandas as gpd

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    store_path = tmp_path / "districts.parquet"
    gpd.GeoDataFrame({"orgUnitCode": ["SL-W"]}, geometry=[Point(-13.5, 6.9)], crs="EPSG:4326").to_parquet(store_path)
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


def test_create_feature_artifact_accepts_shapely_mapping_sequences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    geometry = mapping(GeometryCollection([Point(-13.0, 6.0), Polygon([(0, 0), (2, 0), (2, 3), (0, 0)])]))
    collection = {
        "type": "FeatureCollection",
        "features": ({"type": "Feature", "properties": {"orgUnitCode": "SL-W"}, "geometry": geometry},),
    }

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE, features=collection, store_path=store_path, crs="EPSG:4326"
    )

    assert record.coverage.spatial == CoverageSpatial(xmin=-13.0, ymin=0.0, xmax=2.0, ymax=6.0)


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
            "unlocated feature at 0",
            id="unlocated_feature",
        ),
        pytest.param(
            DISTRICTS_TEMPLATE,
            [{"type": "Feature", "properties": {}, "geometry": None}],
            "must be a GeoJSON FeatureCollection object, got list",
            id="payload_that_is_not_even_a_mapping",
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


def test_create_feature_artifact_refuses_a_store_path_with_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A record pointing at nothing registers loudly and then vanishes from every listing.

    `_materialized_records` drops a record whose backing storage is missing, so without this
    guard the collection would be published and then absent, with a stale-artifact warning as
    the only trace.
    """
    _tmp_record_store(monkeypatch, tmp_path)
    missing = tmp_path / "never-written.parquet"

    with pytest.raises(ValueError, match="no GeoParquet file at"):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_feature_collection(),
            store_path=missing,
            crs="EPSG:4326",
        )

    assert not services.ARTIFACTS_INDEX_PATH.exists() or services._load_records() == []


def test_create_feature_artifact_refuses_a_directory_in_place_of_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One GeoParquet file; the partitioned form is not what this release writes or reads."""
    _tmp_record_store(monkeypatch, tmp_path)
    partitioned = tmp_path / "districts.parquet.d"
    partitioned.mkdir()

    with pytest.raises(ValueError, match="no GeoParquet file at"):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_feature_collection(),
            store_path=partitioned,
            crs="EPSG:4326",
        )


def test_create_feature_artifact_records_a_projected_store_with_both_extents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A projected store was refused until CLIM-1068 gave it a CRS-correct reader; now it registers.

    The record keeps the raster convention: `spatial` is the extent in the store's own CRS and
    `spatial_wgs84` is the WGS 84 one the GeoJSON gave. Recording the WGS 84 extent as `spatial`
    is the silent mismatch ADR 0002 decision 9 exists to prevent.
    """
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    import geopandas as gpd

    gpd.read_parquet(store_path).to_crs("EPSG:3857").to_parquet(store_path)

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="EPSG:3857",
    )

    assert record.features is not None
    assert record.features.crs == "EPSG:3857"
    assert record.coverage.spatial_wgs84 == CoverageSpatial(xmin=-13.5, ymin=6.9, xmax=-10.1, ymax=10.0)
    # Web Mercator metres, so the projected extent is far outside the degree range it came from.
    assert record.coverage.spatial.xmin < -1_000_000
    assert record.coverage.spatial.ymax > 1_000_000


def test_create_feature_artifact_still_refuses_a_crs_that_is_not_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Widening which CRSs are accepted did not stop the field being required and checked."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="authority code"):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_feature_collection(),
            store_path=store_path,
            crs="WGS 84",
        )


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        pytest.param([-13.5, 6.9, -10.1, 10.0, 0.0, 120.0], "6-element bbox", id="three_dimensional_bbox"),
        pytest.param([-13.5, 6.9, -10.1], "3-element bbox", id="too_few_numbers"),
    ],
)
def test_create_feature_artifact_refuses_a_bbox_that_is_not_four_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bbox: list[float], expected: str
) -> None:
    """Truncating a 3D bbox would record minx, miny, minz, maxx — plausible and wrong."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=expected):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_feature_collection(),
            store_path=store_path,
            crs="EPSG:4326",
            bbox=bbox,
        )


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        pytest.param(
            {"type": "Polygon", "coordinates": [-13.5, 6.9]},
            "malformed Polygon",
            id="polygon_carrying_a_bare_position",
        ),
        pytest.param(
            {"type": "Point", "coordinates": [[[-13.5, 6.9]]]},
            "malformed Point",
            id="point_carrying_a_ring",
        ),
        pytest.param(
            {"type": "Bogus", "coordinates": [-13.5, 6.9]},
            "geometry type 'Bogus'",
            id="type_that_is_not_a_geojson_geometry",
        ),
        pytest.param(
            {"type": "Point", "coordinates": ["-13.5", "6.9"]},
            "must hold numbers",
            id="position_of_strings",
        ),
        pytest.param(
            {"type": "Point", "coordinates": [-13.5]},
            "at least two numbers",
            id="position_of_one_number",
        ),
        pytest.param(
            {"type": "Polygon", "coordinates": []},
            "non-empty array",
            id="polygon_with_no_rings",
        ),
        pytest.param(
            {"type": "GeometryCollection", "geometries": []},
            "declares no 'geometries' array",
            id="empty_geometry_collection",
        ),
    ],
)
def test_create_feature_artifact_refuses_a_geometry_it_cannot_interpret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, geometry: dict[str, Any], expected: str
) -> None:
    """The extent is what the catalogue publishes, so a shape OCS cannot read has to fail here.

    A Polygon holding a bare position is the case a numbers-only walker accepts: it yields one
    point and registers an area dataset whose extent is a dot.
    """
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    collection = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"orgUnitCode": "SL-W"}, "geometry": geometry}],
    }

    with pytest.raises(ValueError, match=expected):
        services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE, features=collection, store_path=store_path, crs="EPSG:4326"
        )


def test_create_feature_artifact_accepts_every_geometry_type_at_its_own_nesting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The complement of the refusals: each RFC 7946 type parses at the depth it declares."""
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    geometries = [
        {"type": "Point", "coordinates": [0.0, 0.0]},
        {"type": "MultiPoint", "coordinates": [[1.0, 1.0]]},
        {"type": "LineString", "coordinates": [[2.0, 2.0], [3.0, 3.0]]},
        {"type": "MultiLineString", "coordinates": [[[4.0, 4.0], [5.0, 5.0]]]},
        {"type": "Polygon", "coordinates": [[[6.0, 6.0], [7.0, 6.0], [7.0, 7.0], [6.0, 6.0]]]},
        {"type": "MultiPolygon", "coordinates": [[[[8.0, 8.0], [9.0, 8.0], [9.0, 9.0], [8.0, 8.0]]]]},
    ]
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"orgUnitCode": str(index)}, "geometry": geometry}
            for index, geometry in enumerate(geometries)
        ],
    }

    record = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE, features=collection, store_path=store_path, crs="EPSG:4326"
    )

    assert record.coverage.spatial == CoverageSpatial(xmin=0.0, ymin=0.0, xmax=9.0, ymax=9.0)
    assert record.features is not None
    assert record.features.feature_count == len(geometries)


def test_a_refresh_replaces_the_collection_even_when_the_extract_window_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A feature collection has no version history, so its identity cannot be the request scope.

    Matching on request scope would let a widened bbox append a second record: the collection
    would gain a version, and the older record would keep the publication state.
    """
    store_path = _tmp_record_store(monkeypatch, tmp_path)
    first = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="EPSG:4326",
        bbox=[-13.5, 6.9, -10.1, 10.0],
    )

    second = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE,
        features=_feature_collection(),
        store_path=store_path,
        crs="EPSG:4326",
        bbox=[-20.0, 0.0, 0.0, 20.0],
    )
    third = services.create_feature_artifact(
        template=DISTRICTS_TEMPLATE, features=_feature_collection(), store_path=store_path, crs="EPSG:4326"
    )

    assert second.artifact_id == first.artifact_id
    assert third.artifact_id == first.artifact_id
    assert third.request_scope.bbox is None
    assert len(services._load_records()) == 1


def test_a_raster_still_keeps_one_record_per_request_scope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The feature identity rule is narrow: raster version history is untouched by it."""
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    january = _raster_artifact(artifact_id="r1")
    february = _raster_artifact(artifact_id="r2")
    february.request_scope.start = "2026-02-01"

    services.register_artifact_record(january, publish=False)
    services.register_artifact_record(february, publish=False)

    assert len(services._load_records()) == 2


def test_raster_overwrite_does_not_replace_feature_with_same_dataset_and_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tmp_record_store(monkeypatch, tmp_path)
    feature = _feature_artifact()
    raster = _raster_artifact(dataset_id=feature.dataset_id)
    raster.request_scope = feature.request_scope.model_copy(deep=True)
    services.register_artifact_record(feature, publish=False)

    stored_raster = services.register_artifact_record(raster, publish=False)

    records = services._load_records()
    assert len(records) == 2
    assert stored_raster.artifact_id == raster.artifact_id
    assert records[0] == feature
    assert records[1].format == ArtifactFormat.ICECHUNK


# --- the gates ---------------------------------------------------------------------------


def test_stac_admits_a_feature_collection_and_the_raster_gate_still_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The divergence the gates were split for, now that STAC has a document for both formats.

    STAC describes what exists; openEO advertises what `load_collection` can consume. A feature
    collection is the first artifact where those differ.
    """
    monkeypatch.setattr(
        services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_raster_artifact(), _feature_artifact()]),
    )

    assert sorted(services.stac_eligible_artifacts_by_dataset()) == ["chirps3_precipitation_daily", "districts"]
    assert list(services.latest_published_raster_artifacts_by_dataset()) == ["chirps3_precipitation_daily"]
    assert ArtifactFormat.GEOPARQUET not in services.LOADABLE_RASTER_FORMATS
    assert ArtifactFormat.GEOPARQUET in services.CATALOGUED_FORMATS


def test_an_unpublished_feature_collection_reaches_neither_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_feature_artifact(status=PublicationStatus.UNPUBLISHED)]),
    )

    assert services.stac_eligible_artifacts_by_dataset() == {}


def test_the_catalogue_advertises_a_feature_collection_and_openeo_does_not(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every surface agrees: in STAC and `/datasets`, absent from openEO."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    catalog = client.get("/stac/catalog.json").json()
    dataset = client.get("/datasets/districts").json()

    assert any(link["href"].endswith("/stac/collections/districts") for link in catalog["links"])
    assert {link["rel"] for link in dataset["links"]} == {"self", "stac", "features"}
    assert client.get("/collections/districts").status_code == 404
    assert client.get("/collections").json()["collections"] == []


def test_an_unpublished_feature_collection_is_advertised_nowhere(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening the gate did not weaken the publication check it sits behind."""
    monkeypatch.setattr(
        services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_feature_artifact(status=PublicationStatus.UNPUBLISHED)]),
    )

    catalog = client.get("/stac/catalog.json").json()

    assert all("districts" not in link["href"] for link in catalog["links"])
    assert client.get("/stac/collections/districts").status_code == 404
    assert {link["rel"] for link in client.get("/datasets/districts").json()["links"]} == {"self"}


# --- the format branches -----------------------------------------------------------------


def test_the_feature_builder_needs_no_datacube_machinery(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatched by format: the xstac path is not merely unused for a collection, it is unreachable.

    Every raster helper is replaced with something that raises, so a collection document that
    still touched one would fail loudly rather than quietly produce datacube fields.
    """

    def unreachable(*_: object, **__: object) -> object:
        raise AssertionError("the raster builder must not be reached for a feature collection")

    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", unreachable)
    monkeypatch.setattr(stac_services, "_open_published_store", unreachable)
    monkeypatch.setattr(stac_services, "_zarr_media_type", unreachable)

    response = client.get("/stac/collections/districts")

    assert response.status_code == 200
    assert response.json()["id"] == "districts"


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


@pytest.mark.parametrize(
    "refused",
    [fmt for fmt in ArtifactFormat if fmt is not ArtifactFormat.NETCDF],
    ids=lambda fmt: str(fmt),
)
def test_download_route_refuses_every_format_it_cannot_describe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, refused: ArtifactFormat
) -> None:
    """The allowlist itself, not just the format that exposed it.

    GeoParquet is what broke the old denylist — Parquet bytes came back as
    application/x-netcdf named .nc — but the fix was to name the one format the response is
    built for, so this enumerates `ArtifactFormat` and holds every other member to a refusal.
    Adding a format to the enum without a branch here fails this test rather than shipping a
    wrong media type.
    """
    artifact = _feature_artifact() if refused is ArtifactFormat.GEOPARQUET else _raster_artifact()
    artifact.format = refused
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[artifact]))

    response = client.get(f"/datasets/{services.managed_dataset_id_for(artifact)}/download")

    assert response.status_code == 409
    assert str(refused) in response.json()["detail"]


def test_download_route_still_serves_a_netcdf_artifact(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one format the response is actually built for keeps working."""
    stored = tmp_path / "legacy.nc"
    stored.write_bytes(b"CDF\x01")
    netcdf = _raster_artifact(artifact_id="n1", dataset_id="legacy_netcdf")
    netcdf.format = ArtifactFormat.NETCDF
    netcdf.path = str(stored)
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[netcdf]))

    response = client.get("/datasets/legacy_netcdf/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-netcdf"


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


def test_a_stale_raster_template_cannot_lend_a_feature_collection_a_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record's format decides whether a period axis exists; the template only names one.

    `source_dataset` is resolved by dataset id, so a feature collection whose id also names a
    raster template would otherwise publish `itemType: "feature"` beside `period_type: "daily"`.
    """
    monkeypatch.setattr(services, "_load_records", lambda: [_feature_artifact()])
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "districts", "period_type": "daily", "units": "mm"},
    )

    dataset = services.list_datasets().items[0]

    assert dataset.item_type == DatasetItemType.FEATURE
    assert dataset.period_type is None


def test_a_raster_with_no_declared_period_type_still_reports_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nullable change was for feature collections; it must not alter a raster response."""
    raster = _raster_artifact()
    raster.period_type = None
    monkeypatch.setattr(services, "_load_records", lambda: [raster, _feature_artifact()])
    monkeypatch.setattr(services.registry_datasets, "get_dataset", lambda _: None)

    by_id = {dataset.dataset_id: dataset for dataset in services.list_datasets().items}

    assert by_id["chirps3_precipitation_daily"].period_type == "unknown"
    assert by_id["districts"].period_type is None


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


def test_dataset_links_offer_stac_and_features_but_not_zarr_for_a_feature_collection() -> None:
    """The links track the gates exactly, so `/datasets` never points at a 404 nor withholds a URL."""
    published = _feature_artifact()
    links = services._dataset_links("districts", published, published=published)

    assert {link.rel for link in links} == {"self", "stac", "features"}
    assert any(link.rel == "stac" and link.href == "/stac/collections/districts" for link in links)
    assert any(link.rel == "features" and link.href == "/features/districts" for link in links)


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


# --- the server-rendered pages ------------------------------------------------------------


def test_the_rendered_pages_show_absences_rather_than_python_nulls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature collection has no variable, period type or temporal extent to show.

    Checked on the dataset list and a dataset's own page, which are where a collection appears
    now that the root is an overview and the `/manage` console is gone.
    """
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    for path in ("/datasets", "/datasets/districts"):
        body = client.get(path, headers={"Accept": BROWSER_ACCEPT}).text

        assert "District boundaries" in body, path
        assert "None" not in body, path


def test_a_feature_collection_is_not_offered_a_sync(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_refuse_non_raster_sync` always 409s a feature collection, so the control is not drawn."""
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_feature_artifact()]))

    body = client.get("/datasets/districts", headers={"Accept": BROWSER_ACCEPT}).text

    assert 'id="sync-form"' not in body
    assert "Start sync" not in body


def test_a_raster_is_still_offered_a_sync(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "list_artifacts", lambda: SimpleNamespace(items=[_raster_artifact()]))

    body = client.get("/datasets/chirps3_precipitation_daily", headers={"Accept": BROWSER_ACCEPT}).text

    assert "Start sync" in body
