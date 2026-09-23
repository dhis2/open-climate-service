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


def _collection_files(dataset_id: str = "districts") -> list[Path]:
    """Every stored file belonging to this collection, current and superseded alike."""
    return store.superseded_collection_files(dataset_id, keep=None)


def _current_file(dataset_id: str = "districts") -> Path:
    """The file the collection's newest record points at — the only one a reader resolves."""
    records = [
        record
        for record in ingestion_services._load_records()
        if record.dataset_id == dataset_id and record.features is not None
    ]
    assert records, f"no registered record for '{dataset_id}'"
    return Path(str(max(records, key=lambda record: record.created_at).path))


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

    assert _collection_files() == []


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


def test_a_write_never_touches_the_file_the_current_record_names() -> None:
    """The structural fix: a write lands on a brand-new path, never on an existing file.

    Before this, a refresh replaced one stable file in place, so a reader that resolved the old
    record just before a refresh committed could read bytes a concurrent write had already
    overwritten -- and a crash in that same window made the mismatch between record and file
    permanent. Writing to a fresh path every time removes the window rather than narrowing it:
    there is no file an existing record points at that a write can ever touch.
    """
    first = _register()
    first_path = Path(str(first.path))
    first_bytes = first_path.read_bytes()

    store.write_feature_collection(
        dataset_id="districts",
        features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5)),
        id_property="orgUnitCode",
    )

    assert first_path.is_file()
    assert first_path.read_bytes() == first_bytes


def test_a_refresh_switches_the_record_to_the_new_file_and_keeps_the_old_one_reachable() -> None:
    """The record is switched to the new file; the superseded file is *not* deleted on the spot.

    So resolving record-then-path -- from either the record just before a refresh or the one
    just after -- always yields bytes that record actually describes. Goes through
    `refresh_feature_collection` rather than `_register`'s direct calls, because pruning is
    that function's job, not the raw writer's.

    The old file surviving this refresh is the whole point of the grace period: a reader that
    resolved `first` a moment before this refresh landed must still find its file intact.
    """
    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    first_path = Path(str(first.path))

    second = feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )

    assert second.artifact_id == first.artifact_id
    assert Path(str(second.path)) != first_path
    assert first_path.is_file(), "a file must survive the refresh cycle in which it is superseded"
    assert Path(str(second.path)).is_file()


def test_a_reader_that_resolved_the_old_record_can_still_open_its_file_after_a_refresh() -> None:
    """The gap the second PR#400 review round found: pruning must not race a reader that has
    already resolved the old record and has not yet opened its file.

    No threads. The sequencing that matters is structural, not literal concurrency: record
    lookup happens, *then* a refresh runs to completion (write, register, prune), *then* the
    file is opened -- and the grace period is what makes that ordering safe regardless of how
    much real time separates the two.
    """
    old_record = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    # A reader looked up the collection here and captured `old_record`, including its path --
    # but has not opened the file yet.

    # While the reader is paused, a refresh happens and completes end to end.
    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )

    # The reader resumes and opens the file the record it already holds names.
    frame = store.read_feature_collection(old_record)

    assert len(frame) == 2


def test_a_superseded_file_is_removed_once_it_has_aged_past_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grace period does end: a file only outlives its record for so long.

    Grace is set to zero so a second prune call -- not the one that first marks the file, the
    next one after it -- treats any already-marked file as old enough. This is deterministic:
    no sleeping, no timing assumptions, just "has a marker been written on some earlier call".
    """
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    first_path = Path(str(first.path))

    # First subsequent refresh: `first_path` is seen as superseded for the first time and only
    # marked, never deleted on the same call that notices it.
    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )
    assert first_path.is_file(), "never deleted on the same call that first marks it, whatever the grace period"

    # Second subsequent refresh: the marker from the previous call is now old enough (grace=0).
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert not first_path.exists()
    assert not first_path.with_name(first_path.name + ".superseded").exists()


def test_an_unmarked_file_is_never_deleted_in_the_same_call_that_marks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with grace at zero, marking and deleting never happen in the same prune call.

    If they did, the fix would collapse back to the immediate-deletion bug this closes: the
    marker exists only to force a file to survive at least one full refresh cycle before
    deletion is even considered.
    """
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    first_path = Path(str(first.path))

    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )

    assert first_path.is_file()


def test_prune_removes_a_marker_orphaned_by_a_partial_earlier_delete() -> None:
    """A marker whose file is already gone is swept up rather than left forever."""
    root = api_config.get_features_root()
    root.mkdir(parents=True, exist_ok=True)
    orphan = root / f"districts.{'a' * 32}.parquet.superseded"
    orphan.touch()

    store.prune_superseded_files("districts", keep=root / "districts.does-not-exist.parquet")

    assert not orphan.exists()


def test_a_marker_within_a_positive_grace_period_is_not_yet_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the inequality at a genuine positive value, not only via grace forced to zero.

    Every other grace-boundary test monkeypatches SUPERSEDED_FILE_GRACE_SECONDS to 0, which
    makes `marker_age < 0` always false and never actually compares a positive marker_age
    against a positive grace -- mutating that `<` to `<=` still passes every one of them. This
    controls `time.time()` directly instead, so "marked, but still within grace" is taken at a
    real, positive age, with no sleeping and the real default grace value left untouched.
    """
    # `store.time` is the stdlib `time` module itself, so patching its `time` attribute mutates
    # that module globally -- capturing the real function first (not a reference to the module)
    # is what keeps the fallback from calling the patched version of itself.
    real_time_time = store.time.time
    now: dict[str, float | None] = {"value": None}
    monkeypatch.setattr(store.time, "time", lambda: now["value"] if now["value"] is not None else real_time_time())

    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    first_path = Path(str(first.path))

    # Marks first_path as superseded for the first time; its marker's mtime is real wall-clock
    # time, set by Path.touch() rather than by the patched time.time().
    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )
    marker = first_path.with_name(first_path.name + ".superseded")
    marked_at = marker.stat().st_mtime

    # Force the *comparison* to see 10 real seconds elapsed -- well under the real 60s default.
    now["value"] = marked_at + 10.0
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert first_path.is_file(), "10s < the real, unpatched 60s default grace period"

    # Now push the same comparison past the real default and confirm it is finally removed.
    now["value"] = marked_at + store.SUPERSEDED_FILE_GRACE_SECONDS + 1.0
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert not first_path.exists(), "past the real default grace period, the file is removed"


def test_a_marker_exactly_at_the_grace_boundary_is_treated_as_due(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary itself, pinned in the direction the code actually implements.

    `if marker_age < SUPERSEDED_FILE_GRACE_SECONDS: continue` treats "not yet due" as *strictly*
    younger than the grace period, so a marker exactly that old is eligible for deletion, not
    protected by one more cycle. A `<=` in place of `<` would pass every other test in this file
    (a wall-clock tie is never hit by accident) but flips the outcome of exactly this check.
    """
    real_time_time = store.time.time
    now: dict[str, float | None] = {"value": None}
    monkeypatch.setattr(store.time, "time", lambda: now["value"] if now["value"] is not None else real_time_time())

    first = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())
    first_path = Path(str(first.path))

    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )
    marker = first_path.with_name(first_path.name + ".superseded")
    marked_at = marker.stat().st_mtime

    now["value"] = marked_at + store.SUPERSEDED_FILE_GRACE_SECONDS
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert not first_path.exists(), "a marker exactly grace-seconds old is due for deletion, not protected"


def test_a_reader_that_pauses_past_the_grace_period_can_still_lose_its_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented, accepted boundary of this design, pinned as a visible assertion.

    `SUPERSEDED_FILE_GRACE_SECONDS`'s own docstring says this plainly: the grace period is a
    heuristic window bounded by wall-clock time and refresh count, not a hard guarantee: a
    reader that pauses longer than the grace period while two or more further refreshes run
    its collection's prune calls can still have its file deleted out from under it. Nothing
    short of reference counting -- which this module does not have -- closes this fully. This
    test exists so that limit is an intentional, checked fact rather than a silent assumption.
    """
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    old_record = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    # First further refresh marks the old file as superseded; the second, with grace at zero,
    # finds it already marked and deletes it -- the reader "paused" through both.
    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=_collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    )
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    with pytest.raises(ValueError, match="the record and the store disagree"):
        store.read_feature_collection(old_record)


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
    # Two writes, two versions. Neither was registered, so nothing pruned either: a write only
    # ever adds a file, and the record is what makes one of them the collection.
    assert len(_collection_files()) == 2
    assert not list(api_config.get_features_root().glob("*.writing"))


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

    assert not list(api_config.get_features_root().glob("*.writing"))
    assert len(store.read_feature_collection(first)) == 2


def test_a_feature_without_geometry_is_refused_before_the_write() -> None:
    """A null-geometry feature is stored and counted but invisible to every bbox read."""
    null_geometry = {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": None}

    with pytest.raises(ValueError, match="no geometry"):
        store.write_feature_collection(
            dataset_id="districts", features=_collection(WEST, null_geometry), id_property="orgUnitCode"
        )

    assert _collection_files() == []


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

    assert _collection_files() == []


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

    assert _collection_files() == []


@pytest.mark.parametrize(
    "dataset_id",
    ["", " ", " districts", "districts ", "a\x00b", "../escape", "a/b", ".hidden"],
    ids=["empty", "blank", "leading_space", "trailing_space", "nul", "traversal", "slash", "dot"],
)
def test_a_collection_id_cannot_escape_the_store_directory(dataset_id: str) -> None:
    with pytest.raises(ValueError, match="invalid feature collection id"):
        store.validate_collection_id(dataset_id)


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
    assert len(_collection_files()) == 1
    assert [r.artifact_id for r in ingestion_services._load_records()] == [record.artifact_id]


@pytest.mark.parametrize(
    "geometry",
    [None, {"type": "Polygon", "coordinates": []}],
    ids=["null", "empty"],
)
def test_a_refresh_refuses_bad_features_before_writing_anything(geometry: object) -> None:
    """The write's refusals fire before the backup copy, so the previous collection is untouched."""
    import geopandas as gpd

    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    bad = _collection(WEST, {**_box("SL-X", -13.0, 6.0, -12.5, 7.0), "geometry": geometry})
    with pytest.raises(ValueError, match="no geometry"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=bad)

    assert len(_collection_files()) == 1, "the superseded version is removed once the record is durable"
    assert len(gpd.read_parquet(_current_file())) == 2


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

    stored = gpd.read_parquet(str(_current_file()))
    assert len(stored) == 2, "the file must still be the one the surviving record describes"
    assert first.features is not None
    assert first.features.feature_count == len(stored)
    # The abandoned write is marked as stale, not deleted on the spot: the cleanup path is
    # shared with cases where a caller could have resolved a record naming it, and it must get
    # the same grace period as any other superseded file (see the dedicated reader-race test
    # below for the case where a record actually did exist).
    assert len(_collection_files()) == 2, "the abandoned write is marked stale, not deleted immediately"


def test_a_failed_first_registration_marks_rather_than_deletes_the_unregistered_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no previous version to restore, the file is marked stale, not deleted on the spot.

    Immediate deletion here would be the exact race the fix closes for the steady-state case,
    just reached with no prior record to roll back to: `keep=None` still routes through the
    same mark-then-delete scheme rather than an unconditional unlink.
    """

    def refuse(**_: Any) -> None:
        raise RuntimeError("records.json is unwritable")

    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(ingestion_services, "create_feature_artifact", refuse)
        with pytest.raises(RuntimeError, match="unwritable"):
            feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert len(_collection_files()) == 1, "marked stale, not deleted immediately"

    # It is not left forever either: a later successful refresh's own prune call finds the
    # marker already present and, once old enough, removes it. `refuse` is out of scope again
    # here (the `with` block above restored it), so this refresh runs for real.
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert len(_collection_files()) == 1, "the abandoned file is gone; only the successful write remains"


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
    assert len(gpd.read_parquet(_current_file())) == 2
    # The record that briefly named the "grown" file is durable before publication is attempted
    # (register_artifact_record upserts, then publishes), so a caller could have resolved it in
    # that window and be holding its path. Marking rather than deleting is what keeps that
    # caller's read from racing this rollback.
    assert len(_collection_files()) == 2, "the rolled-back file is marked stale, not deleted immediately"


def test_a_reader_that_resolved_the_record_before_a_publication_failure_can_still_open_its_file() -> None:
    """The high-severity gap the adversarial review round found: rollback used to bypass the
    grace period entirely with a raw unlink, reopening the exact race the fix otherwise closes --
    just reachable from the failure path instead of a normal refresh.

    `register_artifact_record` upserts the new record durably before `publish_artifact_record`
    runs, so a caller resolving the collection in that window (`GET /features` does not filter
    on publication) is handed a record naming the not-yet-published file. If publication then
    fails, that file must still be there when the caller gets around to opening it.
    """
    import pytest as _pytest

    captured: dict[str, Any] = {}

    def fail_after_upsert(_artifact_id: str) -> ArtifactRecord:
        captured["record"] = feature_services.registered_collections()["districts"]
        raise RuntimeError("publication failed")

    with _pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ingestion_services, "publish_artifact_record", fail_after_upsert)
        with pytest.raises(RuntimeError, match="publication failed"):
            feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    # The reader resumes, after the rollback has already run, and opens the file the record it
    # captured names.
    frame = store.read_feature_collection(captured["record"])

    assert len(frame) == 2


def test_failed_first_publication_marks_rather_than_deletes_the_new_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_upsert(_artifact_id: str) -> ArtifactRecord:
        assert len(ingestion_services._load_records()) == 1
        raise RuntimeError("publication failed")

    monkeypatch.setattr(ingestion_services, "publish_artifact_record", fail_after_upsert)
    with pytest.raises(RuntimeError, match="publication failed"):
        feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert ingestion_services._load_records() == []
    assert len(_collection_files()) == 1, "marked stale, not deleted immediately"


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
    stored = gpd.read_parquet(str(_current_file()))
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

    assert len(_collection_files("unregistered")) == 1
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


def test_features_derives_attribution_from_the_template_providers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same provider declaration that reaches STAC must also reach /features."""
    _register()
    monkeypatch.setattr(
        feature_services.registry_datasets,
        "get_dataset",
        lambda _: {
            "providers": [
                {"name": " OpenStreetMap contributors ", "roles": ["licensor"]},
                {"name": "National Mapping Agency", "roles": ["producer"]},
                {"roles": ["host"]},
            ]
        },
    )

    listed = client.get("/features/districts").json()

    assert listed["attribution"] == "OpenStreetMap contributors; National Mapping Agency"


def test_explicit_attribution_takes_precedence_over_provider_names(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template can supply intentional prose when provider names alone are insufficient."""
    _register()
    monkeypatch.setattr(
        feature_services.registry_datasets,
        "get_dataset",
        lambda _: {
            "attribution": "Contains OpenStreetMap data",
            "providers": [{"name": "OpenStreetMap contributors", "roles": ["licensor"]}],
        },
    )

    listed = client.get("/features/districts").json()

    assert listed["attribution"] == "Contains OpenStreetMap data"


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

    with pytest.raises(ValueError, match="EPSG:999999' is not a CRS this service can resolve"):
        transform_bbox((-13.5, 6.9, -10.1, 10.0), source="EPSG:4326", target="EPSG:999999")


def test_a_read_with_an_unknown_bbox_crs_reports_a_value_error() -> None:
    record = _register()

    with pytest.raises(ValueError, match="is not a CRS this service can resolve"):
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
