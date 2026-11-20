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
    assert [link["href"] for link in collection["links"] if link["rel"] == "license"] == [
        COPERNICUS_LICENCE,
        "https://spdx.org/licenses/ODbL-1.0.html",
    ]
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
    # STAC and Portolan take an SPDX identifier or `other`, not an expression: `other`, and a link to each.
    assert collection["license"] == "other"
    assert [(link["href"], link["title"]) for link in collection["links"] if link["rel"] == "license"] == [
        ("https://spdx.org/licenses/CC-BY-4.0.html", "CC-BY-4.0"),
        ("https://spdx.org/licenses/ODbL-1.0.html", "ODbL-1.0"),
    ]
    assert collection["extent"]["spatial"]["bbox"] == [[30.0, -2.0, 32.0, 0.0]]
    assert collection["extent"]["temporal"]["interval"] == [["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]]
    assert collection["table:row_count"] == 4
    columns = {column["name"]: column for column in collection["table:columns"]}
    assert list(columns) == store.schema_of(record).names
    assert all(column["description"] and column["type"] for column in columns.values())
    assert columns["value"]["unit"] == "mm"
    assert columns["period_start"]["type"] == "date32[day]"

    data = collection["assets"]["data"]
    assert data["href"] == "http://testserver/tables/precip_districts/data.parquet"
    assert data["roles"] == ["data"]
    assert data["description"]
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
    assert """ON t."collection_id" = 'districts'""" in readme
    assert 'CAST(f."orgUnitCode" AS VARCHAR) = t."feature_id"' in readme

    agents = client.get("/stac/precip_districts/AGENTS.md").text
    assert "http://testserver/tables/precip_districts/data.parquet" in agents
    assert "(table)" in client.get("/stac/README.md").text


def test_the_dataset_page_shows_each_features_latest_value(client: TestClient) -> None:
    _computed("precip", "districts")

    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    # One period and one reducer for every row are said once, in the heading, not in a column.
    assert "Latest value of each feature, 2026-02 (mean)</h3>" in page
    assert (
        ">Period</th>" not in page
        and "Value of" not in page
        and ">Coverage</th>" in page
        and '<th scope="col" class="numeric">Value (mm)</th>' in page
    )
    assert (
        '<td title="D1">District D1</td>' in page
        and '<td class="numeric">12</td>' in page
        and '<td class="numeric">22</td>' in page
    )
    assert '<td class="numeric">10</td>' not in page
    assert "Open in map viewer" not in page
    assert "UCSB Climate Hazards Center, Overture Maps" in page
    # No template to take prose from: what it holds and what it was computed from, as in STAC.
    description = (
        "Precipitation by district: values per feature and monthly period, computed from Precipitation and Districts."
    )
    assert description in page
    assert client.get("/stac/precip_districts/collection.json").json()["description"] == description

    rows = client.get("/tables/precip_districts/rows", params={"feature": "D2"}).json()["items"]
    assert [(row["feature_id"], row["period"], row["value"]) for row in rows] == [
        ("D2", "2026-01", 20.0),
        ("D2", "2026-02", 22.0),
    ]


def test_a_feature_whose_newest_value_is_older_still_shows_it(client: TestClient) -> None:
    """Latest per feature, not the dataset's last period, which would leave the feature out."""
    store.write_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=[_row("D1", 1.0, period_start="2026-02-01"), _row("D2", 2.0)],
        period_type="monthly",
    )
    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert "Latest value of each feature (mean)</h3>" in page
    assert ">Period</th>" in page and "<td>2026-01</td>" in page and "<td>2026-02</td>" in page


def test_the_page_shows_at_most_ten_latest_values(client: TestClient) -> None:
    store.write_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=[
            _row(feature, 1.0, reducer=f"r{index:02d}", period_start=period)
            for index in range(12)
            for feature, period in (("D1", "2026-02-01"), ("D2", "2026-01-01"))
        ],
        period_type="monthly",
    )
    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert page.count('<td title="D1') == 10
    assert "14 more latest values." in page
    # The rows shown share a period, but D2's latest is older, so the heading names none.
    assert "Latest value of each feature</h3>" in page and ">Period</th>" in page


def test_rows_computed_from_other_datasets_are_refused() -> None:
    """The record keeps the latest write's sources only, so the stored rows would lose theirs."""
    _computed("precip", "districts")
    with pytest.raises(ValueError, match=r"was computed from \['districts', 'precip'\]"):
        _computed("temperature", "districts")
    _computed("precip", "districts")


def test_the_preview_shows_a_column_only_where_it_tells_rows_apart(client: TestClient) -> None:
    """Pipeline rows record no coverage, and two reducers per feature need naming per row."""
    store.write_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=[_row("D1", 1.0), _row("D1", 3.0, reducer="max")],
        period_type="monthly",
    )
    page = client.get("/datasets/precip_districts", params={"f": "html"}).text
    assert "Latest value of each feature, 2026-01</h3>" in page
    assert ">Value of</th>" in page and ">Coverage</th>" not in page


def test_the_listing_filters_tables_and_draws_no_map_placeholder_for_one(client: TestClient) -> None:
    _computed("precip", "districts")
    listing = client.get("/datasets", params={"f": "html"}).text
    assert '<option value="table">Table</option>' in listing
    tile = listing.split('data-kind="table"', 1)[1].split("</article>", 1)[0]
    assert "thumb-table" in tile and "No preview yet" not in tile


def test_every_tile_names_its_kind(client: TestClient) -> None:
    _computed("precip", "districts")
    listing = client.get("/datasets", params={"f": "html"}).text
    for kind in ("table", "vector"):
        tile = listing.split(f'data-kind="{kind}"', 1)[1].split("</article>", 1)[0]
        assert f'<span class="chip">{kind}</span>' in tile


def test_a_dry_run_refuses_rows_computed_from_other_datasets() -> None:
    """The dry run gives the delivery's verdict, so it refuses the source change the write would."""
    from types import SimpleNamespace

    from open_geo_service.exports.table import plugin
    from tests.test_table_datasets import _MAPPING, _cube

    _computed("precip", "districts")
    rendered = plugin.render(_cube([1.0, 2.0]), plugin.validate_mapping(dict(_MAPPING)))
    context = SimpleNamespace(
        source_job_id="job-2",
        source_provenance={"sources": [{"collection_id": "temperature"}, {"collection_id": "districts"}]},
    )
    report = plugin.send(rendered.content, "tables", dry_run=True, context=context)  # type: ignore[arg-type]
    assert report.outcome == "rejected" and report.ignored == 2
    assert report.message is not None and "was computed from ['districts', 'precip']" in report.message


STATION_SCHEMA = """
- id: station_daily
  name: Daily station observations
  columns:
    - {name: station, type: string, required: true, nullable: false, description: Id of the station.}
    - {name: day, type: date, required: true, nullable: false, represents: period_start, description: Day observed.}
    - {name: rainfall, type: double, unit: mm, represents: value, description: Rainfall over the day.}
  key: [station, day]
"""


def test_a_table_follows_the_schema_it_declares_and_skips_what_its_roles_do_not_provide(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A schema in the instance's plugins_dir: a period and a value, but no feature, so no collection to join."""
    from open_geo_service import config as api_config

    folder = tmp_path / "plugins" / "tables" / "schemas"
    folder.mkdir(parents=True)
    (folder / "stations.yaml").write_text(STATION_SCHEMA, encoding="utf-8")
    config = api_config.get_config()
    monkeypatch.setattr(api_config, "get_config", lambda: {**config, "plugins_dir": str(tmp_path / "plugins")})

    def write(rows: list[dict[str, Any]], **arguments: Any) -> Any:
        return store.write_rows(dataset_id="stations", rows=rows, dataset_name="Station rainfall", **arguments)

    record = write(
        [{"station": "S1", "day": "2026-01-02", "rainfall": 4.0}, {"station": "S1", "day": "2026-01-01"}],
        schema="station_daily",
        period_type="daily",
    )
    assert record.table_schema == "station_daily"
    with pytest.raises(ValueError, match=r"unknown columns \['feature_id', 'period_start', 'reducer', 'value'\]"):
        write([_row("D1", 1.0)])
    with pytest.raises(ValueError, match="has schema 'station_daily'; it cannot take rows of schema 'aggregation'"):
        write([{"station": "S1", "day": "2026-01-03"}], schema="aggregation")
    with pytest.raises(ValueError, match="has no feature column"):
        write([{"station": "S1", "day": "2026-01-03"}], collection_id="districts")

    rows = client.get("/tables/stations/rows").json()["items"]
    assert [(row["station"], row["period"], row["day"], row["period_end"], row["rainfall"]) for row in rows] == [
        ("S1", "2026-01-01", "2026-01-01", "2026-01-01", None),
        ("S1", "2026-01-02", "2026-01-02", "2026-01-02", 4.0),
    ]
    header = client.get("/tables/stations/rows", params={"format": "csv"}).text.splitlines()[0]
    assert header == "station,period,day,period_end,rainfall,written_at"
    assert client.get("/tables/stations/rows", params={"feature": "S1"}).status_code == 400
    assert client.get("/tables/stations/rows", params={"reducer": "mean"}).status_code == 400

    collection = client.get("/stac/stations/collection.json").json()
    assert collection["description"] == "Station rainfall: values per daily period."
    assert [column["name"] for column in collection["table:columns"]] == ["station", "day", "rainfall", "written_at"]
    assert collection["table:columns"][2]["unit"] == "mm"
    assert collection["extent"]["temporal"]["interval"] == [["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]]
    assert collection["extent"]["spatial"]["bbox"] == [[-180.0, -90.0, 180.0, 90.0]]

    readme = client.get("/stac/stations/README.md").text
    assert "Daily periods from 2026-01-01 to 2026-01-02." in readme and "Joining" not in readme
    agents = client.get("/stac/stations/AGENTS.md").text
    assert "one per key: `station`, `day`" in agents and "GeoParquet" not in agents
    page = client.get("/datasets/stations", params={"f": "html"}).text
    assert "Station rainfall" in page and "Latest value of each feature" not in page
