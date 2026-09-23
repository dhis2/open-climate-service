"""Feature collections in STAC with the table extension, and out of openEO (CLIM-1069)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pystac
import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    ArtifactVersion,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.shared.geoparquet import PARQUET_MEDIA_TYPE
from open_climate_service.stac import services as stac_services

DATACUBE_FIELDS = ("cube:dimensions", "cube:variables")
ZARR_EXTENSION = "https://stac-extensions.github.io/zarr/v1.1.0/schema.json"
DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.3.0/schema.json"
TABLE_EXTENSION = "https://stac-extensions.github.io/table/v1.2.0/schema.json"


def _feature(code: str, level: int, geometry: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "properties": {"orgUnitCode": code, "level": level}, "geometry": geometry}


def _collection_payload() -> dict[str, Any]:
    """Deliberately mixed geometry: primary_geometry names a column, not a single type."""
    return {
        "type": "FeatureCollection",
        "features": [
            _feature(
                "SL-W",
                2,
                {
                    "type": "Polygon",
                    "coordinates": [[[-13.5, 6.9], [-12.0, 6.9], [-12.0, 8.0], [-13.5, 8.0], [-13.5, 6.9]]],
                },
            ),
            _feature("SL-CLINIC", 4, {"type": "Point", "coordinates": [-10.5, 9.5]}),
        ],
    }


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "features"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    monkeypatch.setattr(stac_services, "_clear_xstac_collection_cache", lambda: None, raising=False)
    return root


@pytest.fixture
def districts(monkeypatch: pytest.MonkeyPatch) -> ArtifactRecord:
    """A real GeoParquet on disk, registered the way a provider run registers one."""
    from open_climate_service.features import services as feature_services

    record = feature_services.refresh_feature_collection(
        template={"id": "districts", "name": "District boundaries", "id_property": "orgUnitCode"},
        features=_collection_payload(),
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)
    return record


def _raster(dataset_id: str = "chirps3_precipitation_daily") -> ArtifactRecord:
    path = f"/tmp/{dataset_id}.icechunk"
    return ArtifactRecord(
        artifact_id="r1",
        dataset_id=dataset_id,
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        period_type="daily",
        format=ArtifactFormat.ICECHUNK,
        path=path,
        asset_paths=[path],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-10"),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-10"),
        ),
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=dataset_id),
    )


def _declare_template(monkeypatch: pytest.MonkeyPatch, template: dict[str, Any] | None) -> None:
    """Declare the metadata source explicitly.

    `plugins/features/` is CLIM-926's registry and does not exist yet, so the feature path reads
    through the same dataset registry the raster path does. Steering that lookup is how this
    exercises declared licence and attribution without inventing a second registry here.
    """
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: template)


# --- the collection document ---------------------------------------------------------------


def test_a_published_feature_collection_appears_in_the_stac_catalogue(
    client: TestClient, districts: ArtifactRecord
) -> None:
    catalog = client.get("/stac/catalog.json").json()

    children = [link for link in catalog["links"] if link["rel"] == "child"]
    assert [link["title"] for link in children] == ["District boundaries"]
    assert children[0]["href"].endswith("/stac/collections/districts")


def test_the_collection_declares_the_table_extension_from_the_stored_data(
    client: TestClient, districts: ArtifactRecord
) -> None:
    doc = client.get("/stac/collections/districts").json()

    assert TABLE_EXTENSION in doc["stac_extensions"]
    assert doc["table:row_count"] == 2
    assert doc["table:primary_geometry"] == "geometry"
    # Read from the stored file's schema, so the provider's own columns appear with their types.
    by_name = {column["name"]: column["type"] for column in doc["table:columns"]}
    assert by_name["orgUnitCode"] == "string"
    assert by_name["level"] == "int64"
    assert "geometry" in by_name


def test_the_covering_bbox_column_is_not_advertised_as_data(client: TestClient, districts: ArtifactRecord) -> None:
    """Bookkeeping this service writes to make windowed reads cheap, not a provider's column."""
    doc = client.get("/stac/collections/districts").json()

    assert "bbox" not in {column["name"] for column in doc["table:columns"]}


def test_a_mixed_geometry_collection_is_described_without_flattening_it(
    client: TestClient, districts: ArtifactRecord
) -> None:
    """`primary_geometry` names a column; one column may hold points and polygons together."""
    doc = client.get("/stac/collections/districts").json()

    assert doc["table:primary_geometry"] == "geometry"
    assert doc["table:row_count"] == 2


def test_no_datacube_or_zarr_fields_are_emitted_for_a_feature_collection(
    client: TestClient, districts: ArtifactRecord
) -> None:
    doc = client.get("/stac/collections/districts").json()

    assert DATACUBE_EXTENSION not in doc["stac_extensions"]
    assert ZARR_EXTENSION not in doc["stac_extensions"]
    for field in DATACUBE_FIELDS:
        assert field not in doc
    assert set(doc["assets"]) == {"data"}
    assert "renders" not in doc


def test_the_feature_collection_reports_its_own_extent_and_crs(client: TestClient, districts: ArtifactRecord) -> None:
    doc = client.get("/stac/collections/districts").json()

    assert doc["extent"]["spatial"]["bbox"] == [[-13.5, 6.9, -10.5, 9.5]]
    assert doc["proj:code"] == "EPSG:4326"


def test_a_projected_collection_reports_the_crs_it_is_stored_in(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extent stays WGS 84 as STAC requires; `proj:code` says where the geometry really is."""
    from open_climate_service.features import services as feature_services

    feature_services.refresh_feature_collection(
        template={"id": "districts", "name": "District boundaries", "id_property": "orgUnitCode"},
        features=_collection_payload(),
        store_crs="EPSG:3857",
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    doc = client.get("/stac/collections/districts").json()

    assert doc["proj:code"] == "EPSG:3857"
    assert doc["extent"]["spatial"]["bbox"][0][0] == pytest.approx(-13.5)


def test_a_static_collection_declares_no_temporal_extent(client: TestClient, districts: ArtifactRecord) -> None:
    """A release version is a release identifier, not an instant, so none is invented from it."""
    stored = ingestion_services._load_records()[0]
    stored.version = ArtifactVersion(value="2026-08-19.0", authority="overture")
    ingestion_services._save_records([stored])

    doc = client.get("/stac/collections/districts").json()

    assert doc["extent"]["temporal"]["interval"] == [[None, None]]


def test_the_collection_carries_the_standard_link_set(client: TestClient, districts: ArtifactRecord) -> None:
    doc = client.get("/stac/collections/districts").json()

    by_rel = {link["rel"]: link["href"] for link in doc["links"]}
    assert by_rel["self"].endswith("/stac/collections/districts")
    assert by_rel["root"].endswith("/stac/catalog.json")
    assert by_rel["parent"].endswith("/stac/catalog.json")
    assert by_rel["alternate"].endswith("/features/districts")


def test_the_generated_collection_validates_as_stac(client: TestClient, districts: ArtifactRecord) -> None:
    """Parsed and checked by pystac, so the document is a Collection rather than a lookalike."""
    doc = client.get("/stac/collections/districts").json()

    collection = pystac.Collection.from_dict(doc, migrate=False, preserve_dict=True)

    assert collection.id == "districts"
    assert collection.stac_extensions is not None
    assert TABLE_EXTENSION in collection.stac_extensions
    from pystac.extensions.table import TableExtension

    table = TableExtension.ext(collection)
    assert table.row_count == 2
    assert table.primary_geometry == "geometry"


def test_the_collection_renders_the_fields_stac_browser_navigates_by(
    client: TestClient, districts: ArtifactRecord
) -> None:
    """CLIM-853: STAC Browser needs a typed, titled, self-describing document to render a page.

    It reads `type`, `id`, `stac_version`, `title`, `description`, `license`, `extent` and the
    self/root links; a collection missing any of them renders as an error or an untitled stub.
    """
    doc = client.get("/stac/collections/districts").json()

    assert doc["type"] == "Collection"
    assert doc["stac_version"] == stac_services.STAC_VERSION
    for field in ("id", "title", "description", "license", "extent", "links", "assets"):
        assert doc.get(field), f"STAC Browser needs a non-empty {field}"
    assert {link["rel"] for link in doc["links"]} >= {"self", "root"}


# --- licence and attribution ----------------------------------------------------------------


def test_a_declared_licence_and_attribution_reach_the_collection(
    client: TestClient, districts: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    _declare_template(
        monkeypatch,
        {
            "license": "CC-BY-4.0",
            "description": "District boundaries from the national hierarchy.",
            "providers": [
                {"name": "Ministry of Health", "roles": ["producer", "licensor"], "url": "https://moh.example"}
            ],
        },
    )

    doc = client.get("/stac/collections/districts").json()

    assert doc["license"] == "CC-BY-4.0"
    assert doc["description"] == "District boundaries from the national hierarchy."
    assert doc["providers"] == [
        {"name": "Ministry of Health", "url": "https://moh.example", "roles": ["producer", "licensor"]}
    ]


def test_a_licence_with_only_a_url_travels_as_a_link(
    client: TestClient, districts: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`license` can only say `other`, so without the link a client learns nothing about the terms."""
    _declare_template(
        monkeypatch, {"license": {"name": "Copernicus Licence", "url": "https://apps.ecmwf.int/licences/copernicus"}}
    )

    doc = client.get("/stac/collections/districts").json()

    assert doc["license"] == "other"
    licence_links = [link for link in doc["links"] if link["rel"] == "license"]
    assert licence_links == [
        {
            "rel": "license",
            "href": "https://apps.ecmwf.int/licences/copernicus",
            "type": "text/html",
            "title": "Copernicus Licence",
        }
    ]


def test_an_undeclared_licence_is_not_invented(
    client: TestClient, districts: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`other` rather than something that reads as permissive, and no providers out of nowhere."""
    _declare_template(monkeypatch, None)

    doc = client.get("/stac/collections/districts").json()

    assert doc["license"] == "other"
    assert "providers" not in doc
    assert not [link for link in doc["links"] if link["rel"] == "license"]


# --- the GeoParquet asset --------------------------------------------------------------------


def test_the_data_asset_advertises_the_settled_media_type(client: TestClient, districts: ArtifactRecord) -> None:
    doc = client.get("/stac/collections/districts").json()

    assert doc["assets"]["data"]["type"] == PARQUET_MEDIA_TYPE == "application/x-parquet"
    assert doc["assets"]["data"]["roles"] == ["data"]


def test_the_advertised_asset_url_serves_the_stored_file(client: TestClient, districts: ArtifactRecord) -> None:
    """The href a STAC client follows has to return the bytes the collection describes."""
    doc = client.get("/stac/collections/districts").json()
    href = doc["assets"]["data"]["href"]

    response = client.get(href.replace("http://testserver", ""))

    assert response.status_code == 200
    assert response.headers["content-type"] == PARQUET_MEDIA_TYPE
    assert response.content[:4] == b"PAR1"
    assert response.content == Path(str(districts.path)).read_bytes()


def test_the_asset_route_resolves_through_the_record_not_the_directory(
    client: TestClient, districts: ArtifactRecord, isolated_store: Path
) -> None:
    """A file nothing registered is not a collection, so it cannot be fetched through this route."""
    from open_climate_service.features import store

    written, _count, _geometry = store.write_feature_collection(
        dataset_id="unregistered", features=_collection_payload(), id_property="orgUnitCode"
    )

    assert written.is_file()
    assert written.parent == isolated_store
    assert client.get("/features/unregistered/data.parquet").status_code == 404


def test_the_asset_route_refuses_an_unpublished_collection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAC advertises only published collections, so this must not hand out what it withholds."""
    from open_climate_service.features import services as feature_services

    feature_services.refresh_feature_collection(
        template={"id": "districts", "name": "District boundaries", "id_property": "orgUnitCode"},
        features=_collection_payload(),
        publish=False,
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert client.get("/features/districts/data.parquet").status_code == 404


# --- the openEO separation --------------------------------------------------------------------


def test_a_feature_collection_stays_out_of_the_openeo_catalogue(client: TestClient, districts: ArtifactRecord) -> None:
    """`load_collection` cannot consume one, so advertising it would offer an unusable dataset."""
    assert client.get("/collections").json()["collections"] == []
    assert client.get("/collections/districts").status_code == 404


def test_the_raster_gate_is_unchanged_by_the_wider_stac_gate(
    monkeypatch: pytest.MonkeyPatch, districts: ArtifactRecord
) -> None:
    raster = _raster()
    stored = ingestion_services._load_records()
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[*stored, raster]))

    assert sorted(ingestion_services.stac_eligible_artifacts_by_dataset()) == [
        "chirps3_precipitation_daily",
        "districts",
    ]
    assert list(ingestion_services.latest_published_raster_artifacts_by_dataset()) == ["chirps3_precipitation_daily"]


# --- the raster catalogue is untouched ----------------------------------------------------------


def test_a_raster_collection_still_builds_with_its_datacube_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    raster = _raster()
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[raster]))
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "chirps3_precipitation_daily",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {"time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}},
            "cube:variables": {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}},
            "assets": {"zarr": {}},
        },
    )
    _declare_template(monkeypatch, {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {})

    doc = client.get("/stac/collections/chirps3_precipitation_daily").json()

    assert DATACUBE_EXTENSION in doc["stac_extensions"]
    assert ZARR_EXTENSION in doc["stac_extensions"]
    assert TABLE_EXTENSION not in doc["stac_extensions"]
    assert "cube:dimensions" in doc
    assert "zarr" in doc["assets"]


# --- the parquet media type, settled once -------------------------------------------------------


def test_the_openeo_job_result_path_emits_the_settled_parquet_media_type(tmp_path: Path) -> None:
    """The other place OCS names Parquet. It used to say `application/vnd.apache.parquet`.

    Two spellings for one format meant a client reading a STAC asset and a client reading a job
    result saw different types for the same bytes, so both now read the shared constant.
    """
    from open_climate_service.openeo.jobs import _VECTOR_FORMATS, _result_assets
    from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
    from open_climate_service.shared.time import utc_now

    output = tmp_path / "result.parquet"
    output.write_bytes(b"PAR1")
    record = OpenEOJobRecord(
        id="job-1", status=OpenEOJobStatus.FINISHED, created=utc_now(), usage={"output_path": str(output)}
    )

    assert _result_assets(record)["result"]["type"] == PARQUET_MEDIA_TYPE
    assert _VECTOR_FORMATS["PARQUET"][1] == PARQUET_MEDIA_TYPE


def test_no_module_still_advertises_the_old_parquet_spelling() -> None:
    """One constant, so the catalogue and the job-result path cannot drift apart again."""
    import open_climate_service.openeo.jobs as jobs
    import open_climate_service.openeo.routes as openeo_routes

    for module in (jobs, openeo_routes):
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        assert "vnd.apache.parquet" not in source, f"{module.__name__} still names the old media type"


# --- ids that would produce links that lie --------------------------------------------------


@pytest.mark.parametrize(
    "rejected",
    [
        "districts?west",
        "districts#west",
        "districts west",
        "a%2e",
        "../escape",
        "a/b",
        ".hidden",
        "",
        "  x",
        "districts\n",
    ],
    ids=[
        "query",
        "fragment",
        "space",
        "percent",
        "traversal",
        "slash",
        "dot",
        "empty",
        "padded",
        "newline",
    ],
)
def test_a_collection_id_that_could_not_be_a_url_segment_is_refused(rejected: str) -> None:
    """An id that needs escaping to be usable is an id nobody should be able to create.

    `districts?west` is the one that motivated this: as a path it silently becomes
    `/features/districts` plus a query string, so the link resolves somewhere else and nothing
    reports an error.
    """
    from open_climate_service.features import store

    with pytest.raises(ValueError, match="invalid feature collection id"):
        store.validate_collection_id(rejected)


@pytest.mark.parametrize("accepted", ["districts", "worldpop_population_global2_100m", "a.b-c_1", "A1"])
def test_an_ordinary_collection_id_is_still_accepted(accepted: str) -> None:
    """The allowlist has to admit the ids real templates use, or it is the wrong allowlist."""
    from open_climate_service.features import store

    assert store.validate_collection_id(accepted) == accepted
    assert store.new_collection_file(accepted).name.startswith(f"{accepted}.")
    assert store.new_collection_file(accepted).name.endswith(".parquet")


def test_every_generated_link_escapes_the_id_as_one_segment() -> None:
    """Second line of defence, for a record written before the ids were constrained.

    Escaping is checked at the URL builder rather than through the store, precisely because the
    store can no longer produce such an id.
    """
    from urllib.parse import urlsplit

    from open_climate_service.shared.urls import path_segment

    assert path_segment("districts?west") == "districts%3Fwest"
    assert path_segment("districts#west") == "districts%23west"
    assert path_segment("a/b") == "a%2Fb"

    url = f"http://testserver/features/{path_segment('districts?west')}/data.parquet"
    parts = urlsplit(url)

    assert parts.query == ""
    assert parts.fragment == ""
    assert parts.path == "/features/districts%3Fwest/data.parquet"


# --- dataset links follow the artifact the routes resolve -------------------------------------


def test_dataset_links_track_stac_when_a_newer_artifact_is_unpublished(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAC resolves the latest *published* artifact; the links must resolve the same one.

    An ingest with `publish: false` over an already-published dataset leaves the catalogue
    serving the earlier record. Keying the links on the newest artifact instead made `/datasets`
    withhold links to a collection STAC was still advertising.
    """
    older = _raster()
    newer = older.model_copy(
        update={
            "artifact_id": "a2",
            "created_at": older.created_at + timedelta(days=1),
            "request_scope": ArtifactRequestScope(start="2026-02-01", end="2026-02-10"),
            "publication": ArtifactPublication(status=PublicationStatus.UNPUBLISHED),
        }
    )
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[older, newer]))
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert list(ingestion_services.stac_eligible_artifacts_by_dataset()) == ["chirps3_precipitation_daily"]
    links = {link["rel"] for link in client.get("/datasets/chirps3_precipitation_daily").json()["links"]}
    assert "stac" in links
    assert "zarr" in links, "the zarr route resolves the latest published artifact too"


def test_dataset_links_stay_absent_when_nothing_is_published(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction: no published artifact means no catalogue links, and no STAC entry."""
    unpublished = _raster().model_copy(
        update={"publication": ArtifactPublication(status=PublicationStatus.UNPUBLISHED)}
    )
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[unpublished]))
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert ingestion_services.stac_eligible_artifacts_by_dataset() == {}
    links = {link["rel"] for link in client.get("/datasets/chirps3_precipitation_daily").json()["links"]}
    assert links == {"self"}


def test_feature_dataset_links_track_stac_when_a_newer_artifact_is_unpublished(
    client: TestClient, districts: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule for a feature collection, whose `features` link is gated identically."""
    newer = districts.model_copy(
        update={
            "artifact_id": "f2",
            "created_at": districts.created_at + timedelta(days=1),
            "publication": ArtifactPublication(status=PublicationStatus.UNPUBLISHED),
        }
    )
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[districts, newer]))

    assert list(ingestion_services.stac_eligible_artifacts_by_dataset()) == ["districts"]
    links = {link["rel"] for link in client.get("/datasets/districts").json()["links"]}
    assert {"stac", "features"} <= links


# --- the advertised asset stays reachable -------------------------------------------------------


def test_the_asset_route_serves_the_artifact_stac_advertises(
    client: TestClient, districts: ArtifactRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalogue advertises the newest *published* collection; the asset must resolve the same one.

    Taking the newest record first and then testing publication 404s the very asset the
    collection document points at.
    """
    newer_unpublished = districts.model_copy(
        update={
            "artifact_id": "f2",
            "created_at": districts.created_at + timedelta(days=1),
            "publication": ArtifactPublication(status=PublicationStatus.UNPUBLISHED),
        }
    )
    monkeypatch.setattr(
        ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[districts, newer_unpublished])
    )

    doc = client.get("/stac/collections/districts").json()
    href = doc["assets"]["data"]["href"].replace("http://testserver", "")
    response = client.get(href)

    assert response.status_code == 200
    assert response.content == Path(str(districts.path)).read_bytes()


# --- an id that cannot route is refused where it enters ------------------------------------------


@pytest.mark.parametrize("rejected", ["a/b", "a?b", "a b", "a\nb", "", ".hidden"])
def test_a_dataset_template_id_that_cannot_be_a_url_segment_is_refused(rejected: str) -> None:
    """A raster template id becomes a catalogue path segment too, so it is held to the same shape.

    `a/b` is the case escaping cannot rescue: ASGI decodes the path before routing, so `%2F`
    splits into two segments and no single-segment route matches.
    """
    from open_climate_service.data_registry.services.datasets import _validate_dataset_template

    template = {"id": rejected, "sync": {"kind": "static"}}

    with pytest.raises(ValueError):
        _validate_dataset_template(template, source="test.yaml")


def test_every_built_in_template_id_is_already_routable() -> None:
    """The allowlist has to admit the templates that ship, or it is the wrong allowlist."""
    from open_climate_service.data_registry.services.datasets import list_datasets
    from open_climate_service.shared.urls import is_segment_safe_id

    templates = list_datasets()

    assert templates
    assert [t["id"] for t in templates if not is_segment_safe_id(str(t["id"]))] == []


def test_a_percent_encoded_slash_does_not_route_back(client: TestClient, districts: ArtifactRecord) -> None:
    """Why the id is constrained rather than merely escaped: `%2F` decodes before routing."""
    from open_climate_service.shared.urls import path_segment

    assert path_segment("a/b") == "a%2Fb"
    assert client.get("/features/a%2Fb").status_code == 404


def test_a_trailing_newline_is_not_a_safe_segment() -> None:
    """`$` also matches before a trailing newline, so this needs `fullmatch` rather than `match`."""
    from open_climate_service.shared.urls import is_segment_safe_id

    assert not is_segment_safe_id("districts\n")
    assert is_segment_safe_id("districts")


# --- the CRS contract holds on every path ---------------------------------------------------------


def test_an_unknown_crs_is_refused_even_when_no_transform_is_needed() -> None:
    """The identity shortcut must not skip the validation this helper promises."""
    from open_climate_service.shared.crs import transform_bbox

    with pytest.raises(ValueError, match="is not a CRS this service can resolve"):
        transform_bbox((0.0, 0.0, 1.0, 1.0), source="EPSG:999999", target="EPSG:999999")


def test_a_non_finite_bbox_is_refused_even_when_no_transform_is_needed() -> None:
    """A window of infinities silently matches everything, so it is refused before it is used."""
    from open_climate_service.shared.crs import transform_bbox

    with pytest.raises(ValueError, match="not a finite box"):
        transform_bbox((0.0, 0.0, float("inf"), 1.0), source="EPSG:4326", target="EPSG:4326")


def test_a_same_crs_bbox_still_passes_through_unchanged() -> None:
    """The shortcut still exists; it just no longer skips the checks."""
    from open_climate_service.shared.crs import transform_bbox

    assert transform_bbox((-13.5, 6.9, -10.1, 10.0), source="epsg:4326", target="EPSG:4326") == (
        -13.5,
        6.9,
        -10.1,
        10.0,
    )


# --- omitted CRS is a statement; an explicit null is not ------------------------------------------


def test_an_omitted_crs_reads_as_the_geoparquet_default(tmp_path: Path) -> None:
    from open_climate_service.shared import geoparquet

    written = _write_parquet(tmp_path / "omitted.parquet", crs="EPSG:4326", drop_crs_key=True)

    assert geoparquet.stored_crs(written) == "EPSG:4326"


def test_an_explicit_null_crs_reads_as_unknown_not_as_wgs84(tmp_path: Path) -> None:
    """A file of undefined coordinates must not register as degrees and then be windowed as degrees."""
    from open_climate_service.shared import geoparquet

    written = _write_parquet(tmp_path / "unknown.parquet", crs=None)

    assert geoparquet.stored_crs(written) is None


def _write_parquet(path: Path, *, crs: str | None, drop_crs_key: bool = False) -> Path:
    """Write a GeoParquet, optionally removing the `crs` key entirely rather than nulling it."""
    import json

    import geopandas as gpd
    import pyarrow.parquet as pq
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame({"code": ["a"]}, geometry=[Point(0, 0)], crs=crs)
    frame.to_parquet(path, write_covering_bbox=True, schema_version="1.1.0")
    if not drop_crs_key:
        return path
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    geo = json.loads(metadata[b"geo"])
    geo["columns"][geo["primary_column"]].pop("crs", None)
    metadata[b"geo"] = json.dumps(geo).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path
