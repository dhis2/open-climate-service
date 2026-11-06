"""Table datasets: rows appended as Parquet, listed under /datasets and served as JSON and CSV (CLIM-1553)."""

from __future__ import annotations

import csv
import io
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import store as feature_store
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.tables import store

FIRST = datetime(2026, 3, 1, tzinfo=UTC)
LATER = datetime(2026, 4, 1, tzinfo=UTC)


def _box(code: str, west: float, south: float) -> dict[str, Any]:
    east, north = west + 1, south + 1
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
    }


@pytest.fixture(autouse=True)
def stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Temporary feature, table and record stores, with a registered 'districts' collection."""
    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "vectors")
    monkeypatch.setattr(api_config, "get_tables_root", lambda: tmp_path / "tables")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    collection = {"type": "FeatureCollection", "features": [_box("D1", 30, -2), _box("D2", 31, -1)]}
    path, _count, geometry = feature_store.write_feature_collection(
        dataset_id="districts", features=collection, id_property="orgUnitCode"
    )
    ingestion_services.create_feature_artifact(
        template={"id": "districts", "name": "Districts", "id_property": "orgUnitCode"},
        features=collection,
        store_path=path,
        crs=feature_store.WGS84,
        primary_geometry=geometry,
    )
    return tmp_path / "tables"


def _row(feature_id: str, value: float | None, **extra: Any) -> dict[str, Any]:
    return {"feature_id": feature_id, "period": "2026-01-01", "value": value, "reducer": "mean", **extra}


def _write(rows: list[dict[str, Any]], written_at: datetime = FIRST) -> Any:
    return store.write_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=rows,
        dataset_name="Precipitation by district",
        period_type="monthly",
        written_at=written_at,
    )


def test_written_rows_are_listed_as_a_table_dataset_and_served_as_json_and_csv(client: TestClient) -> None:
    _write([_row("D1", 12.5, coverage=1.0, unit="mm"), _row("D2", None, coverage=0.0, unit="mm")])

    listed = {item["dataset_id"]: item for item in client.get("/datasets").json()["items"]}
    table = listed["precip_districts"]
    assert table["itemType"] == "table"
    assert table["period_type"] == "monthly"
    assert table["extent"]["temporal"] == {"start": "2026-01-01", "end": "2026-01-01"}
    assert table["extent"]["spatial"] == {"xmin": 30.0, "ymin": -2.0, "xmax": 32.0, "ymax": 0.0}
    assert {"rel": "rows", "href": "/tables/precip_districts/rows", "title": "Table rows"} in table["links"]

    rows = client.get("/tables/precip_districts/rows").json()["items"]
    assert [(row["feature_id"], row["value"], row["period"]) for row in rows] == [
        ("D1", 12.5, "2026-01-01"),
        ("D2", None, "2026-01-01"),
    ]
    assert rows[0]["collection_id"] == "districts"
    assert rows[0]["dataset_id"] == "precip_districts"

    response = client.get("/tables/precip_districts/rows", params={"format": "csv"})
    assert response.headers["content-type"].startswith("text/csv")
    parsed = list(csv.DictReader(io.StringIO(response.text)))
    assert [(row["feature_id"], row["value"]) for row in parsed] == [("D1", "12.5"), ("D2", "")]


def test_a_later_write_of_the_same_key_is_current_and_the_first_is_history(client: TestClient) -> None:
    _write([_row("D1", 10.0)], written_at=FIRST)
    _write([_row("D1", 11.0)], written_at=LATER)

    current = client.get("/tables/precip_districts/rows").json()["items"]
    assert [(row["value"], row["written_at"]) for row in current] == [(11.0, "2026-04-01T00:00:00+00:00")]

    history = client.get("/tables/precip_districts/rows", params={"history": "true"}).json()["items"]
    assert [row["value"] for row in history] == [10.0, 11.0]

    # Still one record, however many writes.
    assert (
        len([item for item in client.get("/datasets").json()["items"] if item["dataset_id"] == "precip_districts"]) == 1
    )


def test_null_key_columns_still_identify_one_value() -> None:
    """`weights` and the forecast columns are NULL for most rows; NULL keys must still match."""
    _write([_row("D1", 1.0)], written_at=FIRST)
    record = _write([_row("D1", 2.0)], written_at=LATER)
    assert store.read_rows(record).column("value").to_pylist() == [2.0]


def test_rows_are_filtered_by_collection_period_reducer_and_weights() -> None:
    _write(
        [
            _row("D1", 1.0),
            _row("D1", 2.0, period="2026-02-01"),
            _row("D1", 3.0, reducer="max"),
            _row("D1", 4.0, weights="population"),
        ]
    )
    record = ingestion_services.published_table_artifacts_by_dataset()["precip_districts"]
    values = lambda **kwargs: sorted(store.read_rows(record, **kwargs).column("value").to_pylist())  # noqa: E731
    assert values(start=date(2026, 2, 1)) == [2.0]
    assert values(end=date(2026, 1, 31)) == [1.0, 3.0, 4.0]
    assert values(reducer="max") == [3.0]
    assert values(weights="population") == [4.0]
    assert values(collection_id="districts") == [1.0, 2.0, 3.0, 4.0]
    assert values(collection_id="other") == []


def test_a_reader_during_a_write_sees_the_files_before_it(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _write([_row("D1", 10.0)], written_at=FIRST)
    seen: list[list[float | None]] = []
    rename = store.os.replace

    def read_then_rename(source: Any, target: Any) -> None:
        # The new file is fully written under its temporary name, and not yet visible.
        assert Path(source).stat().st_size > 0
        seen.append(store.read_rows(record, history=True).column("value").to_pylist())
        rename(source, target)

    monkeypatch.setattr(store.os, "replace", read_then_rename)
    _write([_row("D1", 11.0)], written_at=LATER)

    assert seen == [[10.0]]
    assert store.read_rows(record, history=True).column("value").to_pylist() == [10.0, 11.0]


def test_a_failed_write_leaves_no_file(monkeypatch: pytest.MonkeyPatch, stores: Path) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(OSError):
        _write([_row("D1", 10.0)])
    assert list((stores / "precip_districts").rglob("*")) == [stores / "precip_districts" / "districts"]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row("D9", 1.0), "feature 'D9' is not in feature collection 'districts'"),
        (_row("D1", 0.0, coverage=0.0), "has coverage 0"),
        ({"feature_id": "D1", "period": "2026-01-01"}, "missing required columns ['value']"),
        (_row("D1", 1.0, colour="red"), "unknown columns ['colour']"),
        (_row("D1", 1.0, period="2026-01"), "period must be a date"),
        (_row("D1", math.inf), "value must be finite"),
        (_row("D1", 1.0, coverage=1.5), "coverage must be between 0 and 1"),
        (_row("D1", 1.0, dataset_id="other"), "does not match"),
    ],
)
def test_a_refused_row_writes_nothing(row: dict[str, Any], message: str, stores: Path) -> None:
    pattern = message.replace("[", r"\[").replace("]", r"\]")
    with pytest.raises(ValueError, match=pattern):
        _write([_row("D2", 5.0), row])
    # A dry run gets the same verdict.
    with pytest.raises(ValueError, match=pattern):
        store.check_rows(dataset_id="precip_districts", collection_id="districts", rows=[_row("D2", 5.0), row])
    assert not (stores / "precip_districts").exists()
    assert "precip_districts" not in ingestion_services.group_datasets()


def test_checking_rows_returns_the_table_a_write_would_append_and_writes_nothing(stores: Path) -> None:
    table = store.check_rows(
        dataset_id="precip_districts",
        collection_id="districts",
        rows=[_row("D1", 12.5, coverage=1.0)],
        written_at=FIRST,
    )
    assert table.schema == store.SCHEMA
    assert table.to_pylist()[0] | {"written_at": None} == {
        **dict.fromkeys(store.SCHEMA.names),
        "collection_id": "districts",
        "dataset_id": "precip_districts",
        "feature_id": "D1",
        "period": date(2026, 1, 1),
        "value": 12.5,
        "coverage": 1.0,
        "reducer": "mean",
    }
    assert table.column("written_at").to_pylist() == [FIRST]
    assert not stores.exists()
    assert "precip_districts" not in ingestion_services.group_datasets()


def test_an_unknown_collection_and_a_repeated_key_are_refused() -> None:
    with pytest.raises(ValueError, match="feature collection 'regions' is not registered"):
        store.write_rows(dataset_id="t", collection_id="regions", rows=[_row("D1", 1.0)])
    with pytest.raises(ValueError, match="same key"):
        _write([_row("D1", 1.0), _row("D1", 2.0)])


def test_nan_is_stored_as_null() -> None:
    record = _write([_row("D1", math.nan)])
    assert store.read_rows(record).column("value").to_pylist() == [None]


def test_an_unknown_or_unpublished_table_is_not_served(client: TestClient) -> None:
    assert client.get("/tables/nothing/rows").status_code == 404
    store.write_rows(dataset_id="draft", collection_id="districts", rows=[_row("D1", 1.0)], publish=False)
    assert client.get("/tables/draft/rows").status_code == 404
    assert client.get("/tables/districts/rows").status_code == 404


def test_the_dataset_pages_render_a_table_dataset(client: TestClient) -> None:
    _write([_row("D1", 1.0)])
    listing = client.get("/datasets", params={"f": "html"})
    assert listing.status_code == 200
    assert 'data-kind="table"' in listing.text
    assert client.get("/datasets/precip_districts").json()["itemType"] == "table"
    assert client.get("/datasets/precip_districts", params={"f": "html"}).status_code == 200
    assert client.get("/", headers={"Accept": "text/html"}).status_code == 200
