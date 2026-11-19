"""A table dataset's licence and providers from its sources, its STAC collection, README and page (CLIM-1555)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from open_geo_service.data_registry.services import datasets as registry_datasets
from open_geo_service.features import templates as feature_templates
from open_geo_service.ingestions.schemas import ArtifactProvenance
from open_geo_service.tables import store
from tests.test_table_datasets import _row, stores  # noqa: F401  # pyright: ignore[reportUnusedImport]  (autouse)

PRECIP = {
    "id": "precip",
    "name": "Precipitation",
    "source": "CHIRPS3",
    "source_url": "https://www.chc.ucsb.edu/data/chirps3",
    "license": "CC-BY-4.0",
    "providers": [{"name": "UCSB Climate Hazards Center", "url": "https://www.chc.ucsb.edu/data/chirps"}],
}
COPERNICUS_LICENCE = "https://apps.ecmwf.int/datasets/licences/copernicus/"
TEMPERATURE = {
    "id": "temperature",
    "name": "Temperature",
    "license": {"name": "Licence to Use Copernicus Products", "url": COPERNICUS_LICENCE},
    "providers": [{"name": "ECMWF", "url": "https://www.ecmwf.int"}],
}
DISTRICTS = {
    "id": "districts",
    "name": "Districts",
    "license": "ODbL-1.0",
    "providers": [{"name": "Overture Maps", "url": "https://overturemaps.org", "roles": ["producer", "licensor"]}],
}


@pytest.fixture(autouse=True)
def templates(monkeypatch: pytest.MonkeyPatch) -> None:
    rasters = {template["id"]: template for template in (PRECIP, TEMPERATURE)}
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: rasters.get(dataset_id))
    monkeypatch.setattr(
        feature_templates, "get_feature_template", lambda dataset_id: DISTRICTS if dataset_id == "districts" else None
    )


def _computed(*inputs: str, dataset_id: str = "precip_districts") -> Any:
    """A table written by a job that read *inputs*, two months of values for both districts."""
    provenance = ArtifactProvenance(
        producer_job_id="job-1",
        execution={"process_sha256": "f" * 64, "sources": [{"collection_id": input_id} for input_id in inputs]},
    )
    rows = [
        _row(feature, value, period_start=period, unit="mm", coverage=1.0, feature_name=f"District {feature}")
        for feature, period, value in (
            ("D1", "2026-01-01", 10.0),
            ("D1", "2026-02-01", 12.0),
            ("D2", "2026-01-01", 20.0),
            ("D2", "2026-02-01", 22.0),
        )
    ]
    return store.write_rows(
        dataset_id=dataset_id,
        collection_id="districts",
        rows=rows,
        dataset_name="Precipitation by district",
        period_type="monthly",
        written_at=datetime(2026, 3, 1, tzinfo=UTC),
        provenance=provenance,
    )


def _listed(client: TestClient, dataset_id: str = "precip_districts") -> dict[str, Any]:
    return next(item for item in client.get("/datasets").json()["items"] if item["dataset_id"] == dataset_id)


def test_a_computed_table_takes_the_licences_and_providers_of_its_sources(client: TestClient) -> None:
    """The boundaries count: each row names a feature, so their licence binds the table too."""
    _computed("precip", "districts")

    listed = _listed(client)
    assert listed["license"] == "CC-BY-4.0 AND ODbL-1.0"
    assert listed["license_url"] is None
    assert listed["source"] == "CHIRPS3, Overture Maps"
    assert [provider["name"] for provider in listed["providers"]] == ["UCSB Climate Hazards Center", "Overture Maps"]
    assert listed["providers"][1]["roles"] == ["producer", "licensor"]


def test_one_source_licence_is_the_tables_licence(client: TestClient) -> None:
    _computed("precip")
    listed = _listed(client)
    assert listed["license"] == "CC-BY-4.0"
    assert [provider["name"] for provider in listed["providers"]] == ["UCSB Climate Hazards Center"]


def test_a_source_licence_without_an_spdx_identifier_makes_the_table_other_and_links_it(client: TestClient) -> None:
    _computed("temperature", "districts")

    assert _listed(client)["license"] == "other"
    collection = client.get("/stac/precip_districts/collection.json").json()
    assert collection["license"] == "other"
    assert [link["href"] for link in collection["links"] if link["rel"] == "license"] == [COPERNICUS_LICENCE]
    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert "Licence to Use Copernicus Products, ODbL-1.0" in page


def test_a_source_that_declares_no_licence_leaves_the_table_undeclared(client: TestClient) -> None:
    _computed("unknown_raster", "districts")
    assert _listed(client)["license"] == "other"


def test_a_derived_source_without_a_licence_is_followed_to_the_datasets_it_was_computed_from(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As `via` links are: a workflow output whose template declares no licence takes its inputs'."""
    from open_geo_service.ingestions import services as ingestion_services

    normal = _computed("precip", dataset_id="precip_normal")
    real = ingestion_services.recorded_inputs
    monkeypatch.setattr(
        ingestion_services,
        "recorded_inputs",
        lambda record: [{"collection_id": "precip"}] if record.artifact_id == normal.artifact_id else real(record),
    )
    _computed("precip_normal", "districts")

    assert _listed(client)["license"] == "CC-BY-4.0 AND ODbL-1.0"


def test_the_stac_collection_describes_the_table_and_meets_the_portolan_schema(client: TestClient) -> None:
    record = _computed("precip", "districts")

    catalog = client.get("/stac/catalog.json").json()
    assert "http://testserver/stac/precip_districts/collection.json" in [
        link["href"] for link in catalog["links"] if link["rel"] == "child"
    ]
    assert {"rel": "stac", "href": "/stac/precip_districts/collection.json", "title": "STAC collection"} in _listed(
        client
    )["links"]

    collection = client.get("/stac/precip_districts/collection.json").json()
    assert collection["license"] == "CC-BY-4.0 AND ODbL-1.0"
    assert collection["extent"]["spatial"]["bbox"] == [[30.0, -2.0, 32.0, 0.0]]
    assert collection["extent"]["temporal"]["interval"] == [["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]]
    assert collection["table:row_count"] == 4
    columns = {column["name"]: column for column in collection["table:columns"]}
    assert list(columns) == store.SCHEMA.names
    assert all(column["description"] and column["type"] for column in columns.values())
    assert columns["value"]["unit"] == "mm"
    assert columns["period_start"]["type"] == "date32[day]"

    data = collection["assets"]["data"]
    assert data["href"] == "http://testserver/tables/precip_districts/data.parquet"
    assert data["roles"] == ["data"]
    path = Path(record.asset_paths[0])
    assert data["file:size"] == path.stat().st_size
    assert data["file:checksum"].startswith("1220")

    links = collection["links"]
    assert [link["href"] for link in links if link["rel"] == "derived_from"] == [
        "http://testserver/stac/precip/collection.json",
        "http://testserver/stac/districts/collection.json",
    ]
    assert [link["href"] for link in links if link["rel"] == "via"] == [PRECIP["source_url"]]
    assert {link["rel"] for link in links} >= {"self", "root", "parent", "describedby", "agents"}
    assert [(provider["name"], provider.get("roles")) for provider in collection["providers"]] == [
        ("UCSB Climate Hazards Center", ["producer"]),
        ("Overture Maps", ["producer", "licensor"]),
        ("Open Geo Service", ["processor"]),
        ("Open Geo Service", ["host"]),
    ]

    # Over https, as an instance is served: Portolan refuses an http asset href without an https alternate.
    served = TestClient(client.app, base_url="https://testserver").get("/stac/precip_districts/collection.json").json()
    schemas = Path(__file__).parent / "data"
    jsonschema.validate(served, json.loads((schemas / "portolan_v0.3.0.json").read_text()))
    jsonschema.validate(served, json.loads((schemas / "stac_processing_v1.2.0.json").read_text()))


def test_the_data_asset_serves_the_tables_file(client: TestClient) -> None:
    _computed("precip")
    response = client.get("/tables/precip_districts/data.parquet")
    assert response.status_code == 200
    assert pq.read_table(io.BytesIO(response.content)).num_rows == 4
    assert client.head("/tables/precip_districts/data.parquet").status_code == 200
    assert client.get("/tables/nothing/data.parquet").status_code == 404


def test_the_readme_documents_the_columns_and_joins_the_table_to_its_features(client: TestClient) -> None:
    _computed("precip", "districts")

    readme = client.get("/stac/precip_districts/README.md").text
    assert "- CC-BY-4.0\n- ODbL-1.0" in readme
    assert "Monthly periods from 2026-01 to 2026-02." in readme
    assert "| `value` | double |" in readme
    assert "JOIN 'http://testserver/features/districts/data.parquet' AS f" in readme
    assert "ON t.collection_id = 'districts'" in readme
    assert 'CAST(f."orgUnitCode" AS VARCHAR) = t.feature_id' in readme

    agents = client.get("/stac/precip_districts/AGENTS.md").text
    assert "http://testserver/tables/precip_districts/data.parquet" in agents
    assert "(table)" in client.get("/stac/README.md").text


def test_the_dataset_page_shows_the_latest_rows_and_charts_a_feature(client: TestClient) -> None:
    _computed("precip", "districts")

    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    # One reducer for every row is said once, in the heading, not in a column.
    assert "Latest period, 2026-02 (mean)" in page
    assert "Value of" not in page and ">Coverage</th>" in page
    assert '<td title="D1">District D1</td>' in page
    assert "District D1" in page and "12" in page
    assert 'id="table-chart"' in page and 'data-rows-url="/tables/precip_districts/rows"' in page
    assert '<option value="D2" data-collection="districts">District D2</option>' in page
    assert "Open in map viewer" not in page
    assert "UCSB Climate Hazards Center, Overture Maps" in page

    # What the chart reads: one feature's rows.
    rows = client.get("/tables/precip_districts/rows", params={"feature": "D2"}).json()["items"]
    assert [(row["feature_id"], row["period"], row["value"]) for row in rows] == [
        ("D2", "2026-01", 20.0),
        ("D2", "2026-02", 22.0),
    ]


def test_rows_computed_from_other_datasets_are_refused() -> None:
    """The record keeps the latest write's sources only, so the stored rows would lose theirs."""
    _computed("precip", "districts")
    with pytest.raises(ValueError, match=r"was computed from \['districts', 'precip'\]"):
        _computed("temperature", "districts")
    _computed("precip", "districts")


def test_a_feature_id_in_two_collections_is_two_features_on_the_page(client: TestClient) -> None:
    from open_geo_service.features import store as feature_store
    from open_geo_service.ingestions import services as ingestion_services
    from tests.test_table_datasets import _box

    regions = {"type": "FeatureCollection", "features": [_box("D1", 30, -2)]}
    path, _count, geometry = feature_store.write_feature_collection(
        dataset_id="regions", features=regions, id_property="orgUnitCode"
    )
    ingestion_services.create_feature_artifact(
        template={"id": "regions", "name": "Regions", "id_property": "orgUnitCode"},
        features=regions,
        store_path=path,
        crs=feature_store.WGS84,
        primary_geometry=geometry,
    )
    _computed("precip", "districts")
    store.write_rows(
        dataset_id="precip_districts",
        collection_id="regions",
        rows=[_row("D1", 5.0, period_start="2026-02-01", unit="mm", feature_name="Region D1")],
    )

    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert '<option value="D1" data-collection="districts">District D1 (districts)</option>' in page
    assert '<option value="D1" data-collection="regions">Region D1 (regions)</option>' in page
    rows = client.get("/tables/precip_districts/rows", params={"collection": "regions", "feature": "D1"}).json()
    assert [row["value"] for row in rows["items"]] == [5.0]


def test_the_preview_shows_a_column_only_where_it_tells_rows_apart(client: TestClient) -> None:
    """Pipeline rows record no coverage, and two reducers per feature need naming per row."""
    store.write_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=[_row("D1", 1.0), _row("D1", 3.0, reducer="max")],
        period_type="monthly",
    )
    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert "Latest period, 2026-01</h4>" in page
    assert ">Value of</th>" in page and ">Coverage</th>" not in page


def test_the_listing_filters_tables_and_draws_no_map_placeholder_for_one(client: TestClient) -> None:
    _computed("precip", "districts")
    listing = client.get("/datasets", params={"f": "html"}).text
    assert '<option value="table">Table</option>' in listing
    tile = listing.split('data-kind="table"', 1)[1].split("</article>", 1)[0]
    assert "thumb-table" in tile and "No preview yet" not in tile
