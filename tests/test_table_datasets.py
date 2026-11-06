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


# --- the `table` pipeline destination (CLIM-1554) -------------------------------------------


def _cube(values: list[float], reducer: str | None = "mean") -> Any:
    import numpy as np
    import xarray as xr

    from open_climate_service.shared.vectors import SPATIAL_REDUCER_ATTR

    cube = xr.DataArray(
        np.array(values, dtype="float64").reshape(1, 2),
        dims=("t", "geometry"),
        coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]"), "feature_id": ("geometry", ["D1", "D2"])},
        name="tp",
    )
    if reducer:
        cube.attrs[SPATIAL_REDUCER_ATTR] = reducer
    return cube


_MAPPING = {"dataset_id": "precip_districts", "collection_id": "districts", "name": "Precipitation", "unit": "mm"}


def test_the_destination_renders_rows_and_delivers_them_through_the_writer() -> None:
    from open_climate_service.exports.table import plugin

    rendered = plugin.render(_cube([1.5, math.nan]), plugin.validate_mapping(dict(_MAPPING)))
    assert rendered.record_count == 2 and rendered.periods == ("2026-01-01",)

    report = plugin.send(rendered.content, "tables")
    assert report.outcome == "success" and report.imported == 2
    record = ingestion_services.published_table_artifacts_by_dataset()["precip_districts"]
    assert record.dataset_name == "Precipitation"
    rows = store.read_rows(record).to_pylist()
    assert [(row["feature_id"], row["value"], row["reducer"], row["unit"]) for row in rows] == [
        ("D1", 1.5, "mean", "mm"),
        ("D2", None, "mean", "mm"),
    ]

    # Sent again, as an operator may after an unconfirmed delivery: nothing new in current or history.
    assert plugin.send(rendered.content, "tables").outcome == "success"
    assert len(store.read_rows(record, history=True)) == 2


def test_a_dry_run_send_checks_without_writing_and_a_refusal_is_a_rejected_report(
    stores: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.exports.table import plugin

    rendered = plugin.render(_cube([1.0, 2.0]), plugin.validate_mapping(dict(_MAPPING)))
    assert plugin.send(rendered.content, "tables", dry_run=True).outcome == "dry_run"
    assert not stores.exists()

    # The collection loses a feature between render and delivery: the writer refuses, and says why.
    monkeypatch.setattr(store, "_collection_feature_ids", lambda collection_id: {"D1"})
    report = plugin.send(rendered.content, "tables")
    assert report.outcome == "rejected" and report.imported == 0 and report.ignored == 2
    assert report.message is not None and "feature 'D2' is not in feature collection" in report.message
    assert not stores.exists()


def test_the_destination_refuses_a_result_with_more_than_one_value_per_feature_and_period() -> None:
    from open_climate_service.exports.table import plugin

    cube = _cube([1.0, 2.0]).to_dataset()
    cube["other"] = cube["tp"] * 2
    with pytest.raises(ValueError, match="one value per feature and period"):
        plugin.render(cube, plugin.validate_mapping(dict(_MAPPING)))


def test_destination_checks_name_a_held_dataset_id_and_a_missing_collection() -> None:
    from open_climate_service.exports.base import DestinationContext
    from open_climate_service.exports.table import plugin

    context = DestinationContext(export_id="p", datasets=(), feature_collections=("regions",))
    checks = {check.id: check for check in plugin.check_destination({"dataset_id": "districts"}, context)}
    assert (
        checks["table_dataset"].status == "fail" and "held by a geoparquet dataset" in checks["table_dataset"].message
    )
    assert checks["feature_collection"].status == "fail" and "'regions'" in checks["feature_collection"].message

    context = DestinationContext(export_id="p", datasets=(), feature_collections=("districts",))
    checks = {check.id: check for check in plugin.check_destination({"dataset_id": "precip_districts"}, context)}
    assert {check.status for check in checks.values()} == {"pass"}
    bad = plugin.check_destination({"dataset_id": "no spaces"}, context)
    assert [check.status for check in bad] == ["fail"]


def test_a_dataset_keeps_one_period_type_and_its_files_carry_it(stores: Path) -> None:
    import pyarrow.parquet as pq

    _write([_row("D1", 1.0)], written_at=FIRST)  # monthly
    # Rows with another cadence are refused, by the writer and by a dry run alike.
    for attempt in (
        lambda: store.write_rows(
            dataset_id="precip_districts", collection_id="districts", rows=[_row("D2", 2.0)], period_type="daily"
        ),
        lambda: store.check_rows(
            dataset_id="precip_districts", collection_id="districts", rows=[_row("D2", 2.0)], period_type="daily"
        ),
    ):
        with pytest.raises(ValueError, match="'precip_districts' is monthly; it cannot take daily rows"):
            attempt()
    # A write that names no period type keeps the dataset's.
    store.write_rows(dataset_id="precip_districts", collection_id="districts", rows=[_row("D2", 2.0)], written_at=LATER)

    files = sorted((stores / "precip_districts" / "districts").glob("*.parquet"))
    assert len(files) == 2
    assert {pq.read_schema(path).metadata[store.PERIOD_TYPE_KEY] for path in files} == {b"monthly"}
    assert ingestion_services.published_table_artifacts_by_dataset()["precip_districts"].period_type == "monthly"


def test_a_destination_payload_for_another_period_type_is_rejected(stores: Path) -> None:
    from open_climate_service.exports.table import plugin

    _write([_row("D1", 1.0)])  # monthly
    with pytest.raises(ValueError, match="is monthly; it cannot take daily rows"):
        plugin.render(_cube([1.0, 2.0]), plugin.validate_mapping({**_MAPPING, "period_type": "daily"}))


@pytest.mark.parametrize(
    ("period_type", "day", "start"),
    [
        ("daily", date(2025, 1, 31), date(2025, 1, 31)),
        ("weekly", date(2025, 1, 5), date(2024, 12, 30)),  # a Sunday label is the ISO week from Monday
        ("dekadal", date(2025, 1, 20), date(2025, 1, 11)),
        ("monthly", date(2025, 1, 31), date(2025, 1, 1)),
        ("quarterly", date(2025, 6, 30), date(2025, 4, 1)),
        ("yearly", date(2025, 12, 31), date(2025, 1, 1)),
        (None, date(2025, 1, 31), date(2025, 1, 31)),
    ],
)
def test_a_period_is_stored_as_its_start(period_type: str | None, day: date, start: date) -> None:
    assert store.period_start(day, period_type) == start


def test_a_month_labelled_by_its_last_day_is_the_same_value_as_its_first() -> None:
    _write([_row("D1", 10.0, period="2026-01-31")], written_at=FIRST)
    record = _write([_row("D1", 11.0, period="2026-01-01")], written_at=LATER)
    current = store.read_rows(record).to_pylist()
    assert [(row["period"], row["value"]) for row in current] == [(date(2026, 1, 1), 11.0)]


def test_a_graph_that_aggregates_daily_data_to_months_writes_monthly_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real temporal and spatial aggregation: pandas labels each month by its last day."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from openeo_pg_parser_networkx.process_registry import Process

    from open_climate_service.exports.table import plugin
    from open_climate_service.openeo import execution
    from open_climate_service.shared.provenance import record_source
    from open_climate_service.shared.time import stamp_cadence

    days = pd.date_range("2026-01-01", "2026-02-28", freq="D")

    def load_collection(id: str | None = None, **_: Any) -> xr.DataArray:
        record_source(str(id), type("Artifact", (), {"artifact_id": "a", "source_dataset_id": id, "path": None})())
        cube = xr.DataArray(
            np.ones((len(days), 2, 2)),
            dims=("t", "y", "x"),
            coords={"t": days.values, "y": [-0.5, -1.5], "x": [30.5, 31.5]},
            name="precip",
        ).rio.write_crs("EPSG:4326")
        return stamp_cadence(cube, "daily")

    mocked = execution._RegistryOverlay(
        execution._build_process_registry(), {"load_collection": Process(spec={}, implementation=load_collection)}
    )
    monkeypatch.setattr(execution, "_build_process_registry", lambda: mocked)
    collection = {
        "type": "FeatureCollection",
        "features": [{**_box("D1", 30, -2), "id": "D1"}, {**_box("D2", 31, -1), "id": "D2"}],
    }
    graph = {
        "process_graph": {
            "load": {"process_id": "load_collection", "arguments": {"id": "rain_daily"}},
            "monthly": {
                "process_id": "aggregate_temporal_period",
                "arguments": {
                    "data": {"from_node": "load"},
                    "period": "month",
                    "reducer": {
                        "process_graph": {
                            "r": {
                                "process_id": "sum",
                                "arguments": {"data": {"from_parameter": "data"}},
                                "result": True,
                            }
                        }
                    },
                },
            },
            "zonal": {
                "process_id": "aggregate_spatial_weighted",
                "arguments": {"data": {"from_node": "monthly"}, "geometries": collection, "reducer": "mean"},
                "result": True,
            },
        }
    }
    result = execution.run_process_graph(graph)
    assert [str(value)[:10] for value in result["t"].values] == ["2026-01-31", "2026-02-28"]

    # The source dataset is daily, as the pipeline compiled it; the result says it is monthly.
    rendered = plugin.render(result, plugin.validate_mapping({**_MAPPING, "period_type": "daily"}))
    assert rendered.periods == ("2026-01-01", "2026-02-01")
    plugin.send(rendered.content, "tables")
    record = ingestion_services.published_table_artifacts_by_dataset()["precip_districts"]
    assert record.period_type == "monthly"
    rows = store.read_rows(record).to_pylist()
    assert {(row["period"], row["value"]) for row in rows if row["feature_id"] == "D1"} == {
        (date(2026, 1, 1), 31.0),
        (date(2026, 2, 1), 28.0),
    }
