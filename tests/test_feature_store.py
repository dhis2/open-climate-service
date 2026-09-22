"""The feature store, the GeoParquet reader, identity validation and GET /features (CLIM-1068)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.features import store
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
)
from open_climate_service.shared.features import validate_feature_ids

DISTRICTS_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}


def _box(code: str, west: float, south: float, east: float, north: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
    }


WEST = _box("SL-W", -13.5, 6.9, -12.0, 8.0)
EAST = _box("SL-E", -11.0, 8.5, -10.1, 10.0)


def _collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features or (WEST, EAST))}


@pytest.fixture(autouse=True)
def feature_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the store at a temporary directory, and keep records out of the real index."""
    root = tmp_path / "features"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    return root


def _register(
    *,
    features: dict[str, Any] | None = None,
    crs: str = store.WGS84,
    dataset_id: str = "districts",
    publish: bool = True,
) -> ArtifactRecord:
    """Write a collection to the store and register it, the way a provider run will."""
    collection = features if features is not None else _collection()
    template = {**DISTRICTS_TEMPLATE, "id": dataset_id}
    path, _count, geometry = store.write_feature_collection(
        dataset_id=dataset_id,
        features=collection,
        id_property="orgUnitCode",
        store_crs=crs,
    )
    return ingestion_services.create_feature_artifact(
        template=template,
        features=collection,
        store_path=path,
        crs=crs,
        primary_geometry=geometry,
        publish=publish,
    )


# --- identity validation ------------------------------------------------------------------


def test_the_identifier_is_read_from_properties_when_a_collection_names_one() -> None:
    """What the openEO spec guarantees survives aggregation, and what a frame keeps as a column."""
    assert validate_feature_ids(_collection(), id_property="orgUnitCode") == ["SL-W", "SL-E"]


def test_a_top_level_id_is_read_for_a_hand_made_call() -> None:
    """An inline FeatureCollection in a process graph has no template to name a property."""
    inline = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": "district-1", "geometry": {"type": "Point", "coordinates": [0, 0]}}],
    }

    assert validate_feature_ids(inline) == ["district-1"]


def test_a_named_property_is_not_satisfied_by_a_top_level_id() -> None:
    """The two modes do not overlap, because a fallback here would pass on an unstorable collection.

    `GeoDataFrame.from_features` keeps `properties` as columns and drops the top-level `id`, so
    accepting one would write a file with no identifier column while the record named one — a
    loss nothing downstream raises on, it just pushes values against nothing.
    """
    inline = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "district-1",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
        ],
    }

    with pytest.raises(ValueError, match="so the provider must put it there"):
        validate_feature_ids(inline, id_property="orgUnitCode")


def _identified(value: object, *, id_property: str | None) -> dict[str, Any]:
    """Return a Feature carrying *value* wherever the mode under test reads identifiers from."""
    base: dict[str, Any] = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}}
    if id_property is None:
        base["id"] = value
    else:
        base["properties"] = {id_property: value}
    return base


@pytest.mark.parametrize("id_property", [None, "orgUnitCode"], ids=["top_level_id", "property_path"])
def test_a_null_identifier_fails_naming_the_feature(id_property: str | None) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [_identified("SL-W", id_property=id_property), _identified(None, id_property=id_property)],
    }

    with pytest.raises(ValueError, match="Feature 1 has no usable"):
        validate_feature_ids(collection, id_property=id_property)


@pytest.mark.parametrize("id_property", [None, "orgUnitCode"], ids=["top_level_id", "property_path"])
def test_a_blank_identifier_fails_naming_the_feature(id_property: str | None) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [_identified("SL-W", id_property=id_property), _identified("   ", id_property=id_property)],
    }

    with pytest.raises(ValueError, match="Feature 1 has a blank"):
        validate_feature_ids(collection, id_property=id_property)


@pytest.mark.parametrize("id_property", [None, "orgUnitCode"], ids=["top_level_id", "property_path"])
def test_a_duplicate_identifier_fails_naming_both_features(id_property: str | None) -> None:
    """Silently wrong, not visibly missing: two features push against one org unit."""
    collection = {
        "type": "FeatureCollection",
        "features": [_identified("same", id_property=id_property), _identified("same", id_property=id_property)],
    }

    with pytest.raises(ValueError, match="Feature 1 repeats .*'same', first seen at feature 0"):
        validate_feature_ids(collection, id_property=id_property)


def test_the_store_refuses_to_write_a_collection_with_broken_identity() -> None:
    """Validated before anything is written, so a bad collection leaves no file behind."""
    duplicated = _collection(WEST, _box("SL-W", -11.0, 8.5, -10.1, 10.0))

    with pytest.raises(ValueError, match="repeats"):
        store.write_feature_collection(dataset_id="districts", features=duplicated, id_property="orgUnitCode")

    assert not store.feature_store_path("districts").exists()


@pytest.mark.parametrize(
    "members",
    [
        pytest.param(lambda: [WEST, EAST], id="list"),
        pytest.param(lambda: (WEST, EAST), id="tuple"),
    ],
)
def test_a_collection_may_carry_its_features_in_any_sequence(members: Any) -> None:
    """A tuple is what a provider that built its features with a comprehension hands over.

    Registration accepted one before the writer did, so a collection could be recorded and then
    fail to store. The three checks on this path now agree about what a collection is.
    """
    collection = {"type": "FeatureCollection", "features": members()}

    assert validate_feature_ids(collection, id_property="orgUnitCode") == ["SL-W", "SL-E"]
    path, count, _geometry = store.write_feature_collection(
        dataset_id="districts", features=collection, id_property="orgUnitCode"
    )

    assert count == 2
    assert path.is_file()


# --- the store ----------------------------------------------------------------------------


def test_the_stored_file_carries_the_identifier_column_the_record_names() -> None:
    """The one loss that raises nothing later: a record naming an id_property no read can find."""
    import geopandas as gpd

    record = _register()

    assert record.features is not None
    assert record.features.id_property in gpd.read_parquet(str(record.path)).columns


def test_a_write_refuses_when_the_identifier_column_would_not_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checked against the frame, not only the input, since conversion decides what is a column."""
    import geopandas as gpd

    original = gpd.GeoDataFrame.from_features

    def drop_the_identifier(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs).drop(columns=["orgUnitCode"])

    monkeypatch.setattr(gpd.GeoDataFrame, "from_features", staticmethod(drop_the_identifier))

    with pytest.raises(ValueError, match="no such column survived conversion"):
        store.write_feature_collection(dataset_id="districts", features=_collection(), id_property="orgUnitCode")


def test_concurrent_writes_of_one_collection_do_not_share_a_staging_file() -> None:
    """Two refreshes overlap until the replace, so a shared staging name loses one of them.

    With one name per collection, each write lands in the other's in-progress file and the
    loser's cleanup deletes the winner's — so the barrier here forces both to be mid-flight at
    once, which is exactly when that happened.
    """
    import threading

    import geopandas as gpd

    barrier = threading.Barrier(2, timeout=30)
    original = gpd.GeoDataFrame.to_parquet
    staging_paths: list[str] = []
    lock = threading.Lock()

    def to_parquet(self: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        with lock:
            staging_paths.append(str(path))
        result = original(self, path, *args, **kwargs)
        barrier.wait()
        return result

    gpd.GeoDataFrame.to_parquet = to_parquet  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def write() -> None:
        try:
            store.write_feature_collection(dataset_id="districts", features=_collection(), id_property="orgUnitCode")
        except BaseException as exc:  # noqa: BLE001 - reported through the assertion below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=write) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        gpd.GeoDataFrame.to_parquet = original  # type: ignore[method-assign]

    assert errors == []
    assert len(set(staging_paths)) == 2
    assert store.feature_store_path("districts").is_file()
    assert not list(store.feature_store_path("districts").parent.glob("*.writing"))


def test_a_written_collection_round_trips_through_the_reader() -> None:
    record = _register()

    frame = store.read_feature_collection(record)

    assert sorted(frame["orgUnitCode"]) == ["SL-E", "SL-W"]
    assert record.features is not None
    assert record.features.feature_count == 2


def test_a_bbox_read_does_not_read_rows_outside_the_bbox() -> None:
    """The acceptance criterion: a window returns what it covers and nothing else."""
    record = _register()

    western = store.read_feature_collection(record, bbox=(-13.6, 6.8, -12.5, 7.5))
    eastern = store.read_feature_collection(record, bbox=(-10.9, 8.6, -10.2, 9.9))

    assert list(western["orgUnitCode"]) == ["SL-W"]
    assert list(eastern["orgUnitCode"]) == ["SL-E"]


def test_the_written_file_carries_the_covering_bbox_a_pushdown_needs() -> None:
    """What makes a windowed read skip row groups rather than decode and discard them.

    The returned rows would look the same either way, so the structural claim is checked here:
    GeoParquet 1.1's `covering.bbox` is present, which is what the reader's bbox argument uses.
    """
    import json

    import pyarrow.parquet as pq

    record = _register()

    metadata = pq.read_schema(str(record.path)).metadata or {}
    geo = json.loads(metadata[b"geo"].decode("utf-8"))
    column = geo["columns"][geo["primary_column"]]

    assert geo["version"].startswith("1.1")
    assert "bbox" in column["covering"]


def test_a_projected_collection_is_windowed_against_a_wgs84_bbox() -> None:
    """The bbox crosses CRSs, not the rows; the rows stay in the CRS they were written in."""
    record = _register(crs="EPSG:3857")

    assert record.features is not None
    assert record.features.crs == "EPSG:3857"
    western = store.read_feature_collection(record, bbox=(-13.6, 6.8, -12.5, 7.5))

    assert list(western["orgUnitCode"]) == ["SL-W"]
    # Stored in metres, so the read really did come back projected rather than reprojected.
    assert western.crs is not None
    assert western.crs.to_epsg() == 3857
    assert western.total_bounds[0] < -1_000_000


def test_a_bbox_declared_in_the_stores_own_crs_is_not_transformed_twice() -> None:
    record = _register(crs="EPSG:3857")

    western = store.read_feature_collection(
        record, bbox=(-1_520_000.0, 760_000.0, -1_390_000.0, 840_000.0), bbox_crs="EPSG:3857"
    )

    assert list(western["orgUnitCode"]) == ["SL-W"]


def test_an_unqualified_read_above_the_limit_is_refused_and_names_it() -> None:
    record = _register()

    with pytest.raises(ValueError, match="unqualified-read limit of 1"):
        store.read_feature_collection(record, max_unqualified_read=1)


def test_the_limit_does_not_apply_to_a_windowed_read_or_to_a_deliberate_one() -> None:
    """The guard catches the caller that forgot a bbox, not the one that means to read it all."""
    record = _register()

    windowed = store.read_feature_collection(record, bbox=(-13.6, 6.8, -12.5, 7.5), max_unqualified_read=1)
    deliberate = store.read_feature_collection(record, max_unqualified_read=None)

    assert list(windowed["orgUnitCode"]) == ["SL-W"]
    assert len(deliberate) == 2


def test_the_default_limit_sits_above_a_country_hierarchy() -> None:
    """A backstop, not a capacity: it must not refuse the collections this is built for."""
    assert store.UNQUALIFIED_READ_LIMIT >= 5000


def test_a_write_replaces_the_previous_collection_atomically() -> None:
    _register()
    grown = _collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))

    path, count, _geometry = store.write_feature_collection(
        dataset_id="districts", features=grown, id_property="orgUnitCode"
    )

    assert count == 3
    assert not list(path.parent.glob("*.writing"))


def test_a_failed_write_leaves_no_partial_file() -> None:
    """A reader never sees a half-written collection, and a failure keeps the previous one."""
    first = _register()
    assert first.features is not None

    with pytest.raises(ValueError, match="repeats"):
        store.write_feature_collection(
            dataset_id="districts",
            features=_collection(WEST, _box("SL-W", 0.0, 0.0, 1.0, 1.0)),
            id_property="orgUnitCode",
        )

    assert not list(store.feature_store_path("districts").parent.glob("*.writing"))
    assert len(store.read_feature_collection(first)) == 2


def test_a_feature_without_geometry_is_refused_before_the_write() -> None:
    """A null-geometry feature is stored and counted but invisible to every bbox read."""
    null_geometry = {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": None}

    with pytest.raises(ValueError, match="no geometry"):
        store.write_feature_collection(
            dataset_id="districts", features=_collection(WEST, null_geometry), id_property="orgUnitCode"
        )

    assert not store.feature_store_path("districts").exists()


@pytest.mark.parametrize(
    ("feature", "error"),
    [
        pytest.param(
            {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": {"type": "Polygon", "coordinates": []}},
            "no geometry at index 1",
            id="empty_polygon",
        ),
        pytest.param(
            {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": {"type": "Point"}},
            "malformed GeoJSON geometry",
            id="missing_coordinates",
        ),
        pytest.param(
            {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "properties": {"orgUnitCode": "SL-X", "geometry": "label"}},
            "property named 'geometry'",
            id="geometry_property",
        ),
    ],
)
def test_unstorable_geometry_is_refused_with_collection_context(feature: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=f"feature collection 'districts'.*{error}"):
        store.write_feature_collection(
            dataset_id="districts", features=_collection(WEST, feature), id_property="orgUnitCode"
        )

    assert not store.feature_store_path("districts").exists()


def test_a_property_named_bbox_is_refused_with_a_rename() -> None:
    """The covering-bbox column the store writes would overwrite a `bbox` property."""
    with_bbox = {
        **_box("SL-W", -13.5, 6.9, -12.0, 8.0),
        "properties": {"orgUnitCode": "SL-W", "bbox": [0, 0, 1, 1]},
    }

    with pytest.raises(ValueError, match="property named 'bbox'"):
        store.write_feature_collection(
            dataset_id="districts", features=_collection(with_bbox), id_property="orgUnitCode"
        )

    assert not store.feature_store_path("districts").exists()


@pytest.mark.parametrize(
    "dataset_id",
    ["", " ", " districts", "districts ", "a\x00b", "../escape", "a/b", ".hidden"],
    ids=["empty", "blank", "leading_space", "trailing_space", "nul", "traversal", "slash", "dot"],
)
def test_a_collection_id_cannot_escape_the_store_directory(dataset_id: str) -> None:
    with pytest.raises(ValueError, match="invalid feature collection id"):
        store.feature_store_path(dataset_id)


def test_reading_a_record_whose_file_is_gone_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _register()
    Path(str(record.path)).unlink()

    with pytest.raises(ValueError, match="no file is there"):
        store.read_feature_collection(record)


def test_reading_a_raster_record_as_features_is_refused() -> None:
    raster = ArtifactRecord(
        artifact_id="r1",
        dataset_id="chirps3_precipitation_daily",
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        period_type="daily",
        format=ArtifactFormat.ICECHUNK,
        path="/tmp/chirps.icechunk",
        asset_paths=["/tmp/chirps.icechunk"],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-10"),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-10"),
        ),
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(),
    )

    with pytest.raises(ValueError, match="not a feature collection"):
        store.read_feature_collection(raster)


def test_geometry_types_are_read_from_the_file_footer() -> None:
    record = _register()

    assert store.stored_geometry_types(record) == ["Polygon"]


# --- write and register as one operation --------------------------------------------------


def test_a_refresh_writes_and_registers_through_one_door() -> None:
    record = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert record.features is not None
    assert record.features.feature_count == 2
    assert store.feature_store_path("districts").is_file()
    assert [r.artifact_id for r in ingestion_services._load_records()] == [record.artifact_id]


@pytest.mark.parametrize(
    "geometry",
    [None, {"type": "Polygon", "coordinates": []}],
    ids=["null", "empty"],
)
def test_a_refresh_refuses_bad_features_before_copying_the_previous_aside(geometry: object) -> None:
    """The write's refusals fire before the backup copy, so the previous collection is untouched."""
    import geopandas as gpd

    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    bad = _collection(WEST, {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": geometry})
    with pytest.raises(ValueError, match="no geometry"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=bad)

    assert not list(store.feature_store_path("districts").parent.glob("*.previous"))
    assert len(gpd.read_parquet(store.feature_store_path("districts"))) == 2


def test_a_failed_registration_leaves_the_previous_collection_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apart, these two steps leave the old record describing the new file — and nothing raises.

    The record stays valid and the file stays present, so `feature_count`, `extent` and `crs`
    all describe bytes that are gone.
    """
    import geopandas as gpd

    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    grown = _collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))

    def refuse(**_: Any) -> None:
        raise RuntimeError("records.json is unwritable")

    monkeypatch.setattr(ingestion_services, "create_feature_artifact", refuse)

    with pytest.raises(RuntimeError, match="unwritable"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=grown)

    stored = gpd.read_parquet(str(store.feature_store_path("districts")))
    assert len(stored) == 2, "the file must still be the one the surviving record describes"
    assert first.features is not None
    assert first.features.feature_count == len(stored)
    assert not list(store.feature_store_path("districts").parent.glob("*.previous"))


def test_a_failed_first_registration_leaves_no_collection_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no previous version to restore, the unregistered file is removed rather than left."""

    def refuse(**_: Any) -> None:
        raise RuntimeError("records.json is unwritable")

    monkeypatch.setattr(ingestion_services, "create_feature_artifact", refuse)

    with pytest.raises(RuntimeError, match="unwritable"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert not store.feature_store_path("districts").exists()


def test_publication_failure_restores_the_previous_file_and_record(monkeypatch: pytest.MonkeyPatch) -> None:
    import geopandas as gpd

    first = feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(), publish=False
    )
    grown = _collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))

    def fail_after_upsert(_artifact_id: str) -> ArtifactRecord:
        stored = ingestion_services._load_records()[0]
        assert stored.features is not None
        assert stored.features.feature_count == 3
        raise RuntimeError("publication failed")

    monkeypatch.setattr(ingestion_services, "publish_artifact_record", fail_after_upsert)
    with pytest.raises(RuntimeError, match="publication failed"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=grown)

    assert ingestion_services._load_records() == [first]
    assert len(gpd.read_parquet(store.feature_store_path("districts"))) == 2
    assert not list(store.feature_store_path("districts").parent.glob("*.previous"))


def test_failed_first_publication_removes_the_new_file_and_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_after_upsert(_artifact_id: str) -> ArtifactRecord:
        assert len(ingestion_services._load_records()) == 1
        raise RuntimeError("publication failed")

    monkeypatch.setattr(ingestion_services, "publish_artifact_record", fail_after_upsert)
    with pytest.raises(RuntimeError, match="publication failed"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert ingestion_services._load_records() == []
    assert not store.feature_store_path("districts").exists()


def test_the_collection_lock_is_still_held_while_the_record_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration runs inside the lock, not after it — which is the whole point of the fix.

    Asserted structurally rather than by racing threads: with the lock in place a second thread
    cannot enter the critical section, so no interleaving test can distinguish the fixed code
    from the broken code by timing alone. What is checkable is that the lock is held at the
    moment the record is written, since that is when a second refresh would otherwise stamp its
    numbers onto the file this one just replaced.
    """
    held: list[bool] = []
    original = ingestion_services.create_feature_artifact

    def observe(**kwargs: Any) -> Any:
        held.append(store.collection_lock("districts").locked())
        return original(**kwargs)

    monkeypatch.setattr(ingestion_services, "create_feature_artifact", observe)

    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert held == [True]
    assert not store.collection_lock("districts").locked()


def test_concurrent_refreshes_leave_the_file_and_its_record_agreeing() -> None:
    """The end state both reviewers care about: the record describes the bytes on disk."""
    import threading

    import geopandas as gpd

    small = _collection()
    large = _collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    errors: list[BaseException] = []

    def refresh(payload: dict[str, Any]) -> None:
        try:
            feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=payload)
        except BaseException as exc:  # noqa: BLE001 - surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=refresh, args=(payload,)) for payload in (small, large)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    records = [r for r in ingestion_services._load_records() if r.features is not None]
    assert len(records) == 1
    stored = gpd.read_parquet(str(store.feature_store_path("districts")))
    assert records[0].features is not None
    assert records[0].features.feature_count == len(stored)


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        pytest.param({"name": "D", "id_property": "orgUnitCode"}, "non-empty 'id'", id="no_id"),
        pytest.param(
            {"id": " districts ", "name": "D", "id_property": "orgUnitCode"},
            "invalid feature collection id",
            id="surrounding_whitespace",
        ),
        pytest.param({"id": "districts", "name": "D"}, "non-empty 'id_property'", id="no_id_property"),
    ],
)
def test_a_refresh_refuses_a_template_that_cannot_identify_its_features(
    template: dict[str, Any], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        feature_services.refresh_feature_collection(template=template, features=_collection())


# --- the declared CRS is checked against the file -----------------------------------------


def test_registering_a_file_as_a_crs_it_does_not_store_is_refused() -> None:
    """Declaring a degrees file as metres publishes wrong coverage and windows every later read."""
    path, _count, geometry = store.write_feature_collection(
        dataset_id="districts", features=_collection(), id_property="orgUnitCode"
    )

    with pytest.raises(ValueError, match="stores EPSG:4326"):
        ingestion_services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE,
            features=_collection(),
            store_path=path,
            crs="EPSG:3857",
            primary_geometry=geometry,
        )


def test_the_crs_check_accepts_a_file_that_agrees_with_the_declaration() -> None:
    record = _register(crs="EPSG:3857")

    assert record.features is not None
    assert record.features.crs == "EPSG:3857"


def test_a_file_without_readable_geoparquet_metadata_is_refused(tmp_path: Path) -> None:
    from open_climate_service.shared import geoparquet

    plain = tmp_path / "plain.parquet"
    plain.write_bytes(b"PAR1")

    assert geoparquet.stored_crs(plain) is None
    with pytest.raises(ValueError, match="no readable GeoParquet CRS"):
        ingestion_services.create_feature_artifact(
            template=DISTRICTS_TEMPLATE, features=_collection(), store_path=plain, crs="EPSG:3857"
        )


# --- GET /features ------------------------------------------------------------------------


def test_features_lists_a_registered_collection(client: TestClient) -> None:
    record = _register()

    payload = client.get("/features").json()

    assert payload["kind"] == "FeatureCollectionList"
    assert [item["id"] for item in payload["items"]] == ["districts"]
    listed = payload["items"][0]
    assert listed["name"] == "District boundaries"
    assert listed["id_property"] == "orgUnitCode"
    assert listed["feature_count"] == 2
    assert listed["geometry_types"] == ["Polygon"]
    assert listed["crs"] == "EPSG:4326"
    assert listed["primary_geometry"] == record.features.primary_geometry if record.features else False


def test_a_file_in_the_store_directory_with_no_record_does_not_appear(
    client: TestClient, feature_store_root: Path
) -> None:
    """The acceptance criterion: a record is what makes a collection exist, not a file on disk."""
    _register()
    # Written through the store, so it is a real GeoParquet in the real location — and still
    # invisible, because nothing registered it.
    store.write_feature_collection(dataset_id="unregistered", features=_collection(), id_property="orgUnitCode")

    payload = client.get("/features").json()

    assert (feature_store_root / "unregistered.parquet").is_file()
    assert [item["id"] for item in payload["items"]] == ["districts"]
    assert client.get("/features/unregistered").status_code == 404


def test_features_detail_returns_one_collection(client: TestClient) -> None:
    _register()

    response = client.get("/features/districts")

    assert response.status_code == 200
    assert response.json()["id"] == "districts"


def test_features_detail_404s_for_an_unknown_collection(client: TestClient) -> None:
    assert client.get("/features/nope").status_code == 404


def test_features_lists_an_unpublished_collection(client: TestClient) -> None:
    """`/features` is the inventory of what is held; publication decides what is advertised."""
    _register(publish=False)

    payload = client.get("/features").json()

    assert [item["id"] for item in payload["items"]] == ["districts"]


def test_features_ignores_raster_artifacts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _register()
    raster = record.model_copy(
        update={
            "artifact_id": "r1",
            "dataset_id": "chirps3_precipitation_daily",
            "format": ArtifactFormat.ICECHUNK,
            "variable": "precip",
            "features": None,
        }
    )
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[record, raster]))

    assert [item["id"] for item in client.get("/features").json()["items"]] == ["districts"]


def test_features_reports_the_licence_and_prose_a_template_declares(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sourced through the existing registry, so these populate when CLIM-926 adds templates."""
    _register()
    monkeypatch.setattr(
        feature_services.registry_datasets,
        "get_dataset",
        lambda _: {
            "license": "CC-BY-4.0",
            "description": "  District boundaries from the national hierarchy.\n",
            "attribution": "Ministry of Health",
        },
    )

    listed = client.get("/features/districts").json()

    assert listed["license"] == "CC-BY-4.0"
    assert listed["description"] == "District boundaries from the national hierarchy."
    assert listed["attribution"] == "Ministry of Health"


def test_features_reports_other_when_no_template_declares_a_licence(client: TestClient) -> None:
    """Never absent, and never something that reads as permissive."""
    _register()

    listed = client.get("/features/districts").json()

    assert listed["license"] == "other"
    assert listed["description"] is None
    assert listed["attribution"] is None


def test_features_reports_the_release_a_collection_was_cut_from(client: TestClient) -> None:
    _register()
    stored = ingestion_services._load_records()[0]
    stored.version = ArtifactVersion(value="2026-08-19.0", authority="overture")
    ingestion_services._save_records([stored])

    listed = client.get("/features/districts").json()

    assert listed["version"] == {"value": "2026-08-19.0", "authority": "overture"}


def test_features_reports_a_projected_collection_in_its_own_crs(client: TestClient) -> None:
    _register(crs="EPSG:3857")

    listed = client.get("/features/districts").json()

    assert listed["crs"] == "EPSG:3857"
    assert listed["extent"]["spatial_wgs84"]["xmin"] == pytest.approx(-13.5)
    assert listed["extent"]["spatial"]["xmin"] < -1_000_000


def test_the_newest_record_represents_a_refreshed_collection(client: TestClient) -> None:
    """A refresh replaces in place, so one collection is listed however many times it ran."""
    _register()
    _register(features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5)))

    payload = client.get("/features").json()

    assert [item["id"] for item in payload["items"]] == ["districts"]
    assert payload["items"][0]["feature_count"] == 3


def test_features_is_registered_in_the_openapi_surface(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/features" in paths
    assert "/features/{collection_id}" in paths


# --- error contracts shared with callers --------------------------------------------------


def test_an_unknown_crs_code_is_a_value_error_not_a_pyproj_error() -> None:
    """An authority-shaped code that no register knows is the ordinary typo, not a fault.

    Every caller of `transform_bbox` promises its own callers ValueError, so a leaked
    `CRSError` would surface as an internal error for a mistake a user can fix.
    """
    from open_climate_service.shared.crs import transform_bbox

    with pytest.raises(ValueError, match="cannot transform a bbox"):
        transform_bbox((-13.5, 6.9, -10.1, 10.0), source="EPSG:4326", target="EPSG:999999")


def test_a_read_with_an_unknown_bbox_crs_reports_a_value_error() -> None:
    record = _register()

    with pytest.raises(ValueError, match="cannot transform a bbox"):
        store.read_feature_collection(record, bbox=(-13.6, 6.8, -12.5, 7.5), bbox_crs="EPSG:999999")


def test_provenance_fingerprints_a_tuple_backed_collection() -> None:
    """The recorder accepts what the validator accepts, or an execution loses its fingerprint."""
    from open_climate_service.shared.provenance import capture_execution, record_features

    pt = {"type": "Point", "coordinates": [0.0, 0.0]}
    collection = {
        "type": "FeatureCollection",
        "features": ({"type": "Feature", "id": "a", "geometry": pt},),
    }

    with capture_execution({}) as evidence:
        record_features(collection)

    assert len(evidence.features) == 1
    assert evidence.features[0]["feature_count"] == 1
    assert evidence.features[0]["ids_valid"] is True


def test_provenance_agrees_with_the_validator_about_an_integer_id() -> None:
    """One rule, one answer: identity validation and the manifest cannot disagree."""
    from open_climate_service.shared.provenance import capture_execution, record_features

    pt = {"type": "Point", "coordinates": [0.0, 0.0]}
    collection = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": 7, "geometry": pt}]}

    assert validate_feature_ids(collection) == ["7"]
    with capture_execution({}) as evidence:
        record_features(collection)

    assert evidence.features[0]["ids_valid"] is True


def test_provenance_still_reports_broken_identity_as_invalid() -> None:
    from open_climate_service.shared.provenance import capture_execution, record_features

    pt = {"type": "Point", "coordinates": [0.0, 0.0]}
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "same", "geometry": pt},
            {"type": "Feature", "id": "same", "geometry": pt},
        ],
    }

    with capture_execution({}) as evidence:
        record_features(collection)

    assert evidence.features[0]["ids_valid"] is False
