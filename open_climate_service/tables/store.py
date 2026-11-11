"""The table store: current values per feature and period, one Parquet file per dataset (CLIM-1553).

One folder per table dataset under `config.get_tables_root`, holding the dataset's one file of
current values. No history is kept: a revised value replaces its row, and a value written again
unchanged keeps the row and its `written_at`, so `written_at` says when each value last changed.
A write for a new version of a feature collection drops that collection's rows of other versions,
so values for two boundary versions never mix.

Every write produces a new file, written under a temporary name and renamed, holding the merged
rows. The dataset's record names it in `asset_paths`, and readers read only that: the file
becomes the dataset when the record naming it is registered. The previous file stays for a grace
period, for a reader that resolved the previous record and has not opened its file yet.
A reader sees the dataset before a write or after it, and a write whose record fails leaves the
dataset as it was. The one file is what a static catalog serves as the dataset's data.

`write_rows` is the one writer. It validates rows against the feature collection they report
on, merges them, and registers the dataset's record, which is what lists it under `/datasets`
as `itemType: table`. `check_rows` gives the same verdict without writing, for a dry run.
`read_rows` reads them back with pyarrow.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

from open_climate_service import config as api_config
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
)
from open_climate_service.shared.persistence import index_lock
from open_climate_service.shared.urls import is_segment_safe_id

logger = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("feature_id", pa.string(), nullable=False),
        pa.field("feature_name", pa.string()),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("period_start", pa.date32(), nullable=False),
        pa.field("class", pa.string()),
        pa.field("reducer", pa.string()),
        pa.field("weights", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("coverage", pa.float64()),
        pa.field("unit", pa.string()),
        pa.field("reference_time", pa.timestamp("us", tz="UTC")),
        pa.field("lead_time", pa.duration("s")),
        pa.field("member", pa.int32()),
        pa.field("collection_version", pa.string()),
        pa.field("written_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
"""One row per feature, period and variable: the CLIM-1551 data model.

`class` is the category a per-class value belongs to (people per hazard class, statistics per
land cover class), NULL for a value that has none. Classes of one computation stay in one
dataset, so their rows share a reducer, weights and unit and can be summed back to a total.
"""

KEY = (
    "collection_id",
    "collection_version",
    "feature_id",
    "dataset_id",
    "period_start",
    "class",
    "reducer",
    "weights",
    "reference_time",
    "lead_time",
    "member",
)
"""The columns that identify one value: a write with the same key replaces the stored row."""

PERIOD_TYPE_KEY = b"ocs:period_type"
"""Parquet metadata key holding the dataset's period type, so a reader of the files alone knows
whether `period_start` starts a day, a month or a year."""

_WRITABLE = frozenset(SCHEMA.names) - {"written_at"}
_REQUIRED = ("feature_id", "period_start", "value")


def _dataset_lock(dataset_id: str) -> AbstractContextManager[None]:
    """Serialize writes to one dataset across processes, so no write's record drops another's file.

    The lock file sits beside the records index, so a refused write leaves the table store as it was.
    """
    from open_climate_service.ingestions import services as ingestion_services

    _require_id(dataset_id, "table dataset")
    return index_lock(Path(ingestion_services.ARTIFACTS_DIR) / "locks" / f"table-{dataset_id}")


def write_rows(
    *,
    dataset_id: str,
    collection_id: str,
    rows: Iterable[Mapping[str, Any]],
    dataset_name: str | None = None,
    period_type: str | None = None,
    written_at: datetime | None = None,
    provenance: ArtifactProvenance | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """Check *rows* with `check_rows`, merge them into the dataset's file and register the dataset.

    Raises ValueError before anything is written when any row is refused. *provenance* is how
    the rows were produced, for computed rows; the record keeps the latest write's. *publish*
    applies to a new dataset only: a later write keeps the dataset's publication, whatever it is.
    """
    stamp = _utc(written_at or datetime.now(UTC), "written_at")

    from open_climate_service.ingestions import services as ingestion_services

    # Checked under the lock, so two first writes cannot give one dataset two period types or units.
    with _dataset_lock(dataset_id):
        table = check_rows(
            dataset_id=dataset_id, collection_id=collection_id, rows=rows, period_type=period_type, written_at=stamp
        )
        existing = _existing_record(dataset_id)
        period_type = period_type or (existing.period_type if existing else None)
        merged = _merged(_read_file(existing), table, collection_id, period_type)
        written = _write_file(_dataset_root(dataset_id), merged, stamp)
        try:
            units = table.column("unit").drop_null().unique().to_pylist()
            record = ArtifactRecord(
                artifact_id=str(uuid4()),
                dataset_id=dataset_id,
                dataset_name=dataset_name or (existing.dataset_name if existing else dataset_id),
                period_type=period_type,
                format=ArtifactFormat.PARQUET,
                path=str(_dataset_root(dataset_id)),
                asset_paths=[str(written)],
                units=units[0] if units else (existing.units if existing else None),
                size_bytes=written.stat().st_size,
                request_scope=ArtifactRequestScope(),
                coverage=_coverage(merged, period_type),
                created_at=datetime.now(UTC),
                provenance=provenance or (existing.provenance if existing else None),
            )
            registered = ingestion_services.register_artifact_record(record, publish=False)
            if publish and existing is None:
                try:
                    registered = ingestion_services.publish_artifact_record(registered.artifact_id)
                except BaseException:
                    # The record is stored before it is published; a dataset that failed to
                    # publish has no record, as before the write. It was never served, so its
                    # file can go at once.
                    _remove_records(dataset_id)
                    raise
        except BaseException:
            # No record names the file, so no reader sees it; remove it rather than leave it.
            written.unlink(missing_ok=True)
            raise
        _prune_superseded(_dataset_root(dataset_id), keep=written)
        return registered


def check_rows(
    *,
    dataset_id: str,
    collection_id: str,
    rows: Iterable[Mapping[str, Any]],
    period_type: str | None = None,
    written_at: datetime | None = None,
) -> pa.Table:
    """Return *rows* as the table `write_rows` would merge in, or raise ValueError; nothing is written.

    For a dry run, which needs the writer's verdict without the write. Every row names a
    `feature_id` of the registered collection, a `period_start` and a `value`; the other `SCHEMA`
    columns are optional. `dataset_id` and `collection_id` come from the arguments, and a row
    that names others is refused. A missing value is None (NaN is read as None); a value with
    `coverage` 0 is refused, because a unit without valid cells has no value, and 0 would be
    exported as a measurement. All rows share one `written_at`, now unless given. A dataset id
    a raster or feature collection already holds is refused, so the two never share a listing.

    A dataset has one period type, the first one written. A *period_type* that differs from it is
    refused; None keeps it. The table carries it as Parquet metadata (`PERIOD_TYPE_KEY`), and each
    `period_start` is stored as the first day of the period that contains it (`start_of_period`).
    It has one unit too, the first one written: rows in another unit are refused.
    """
    _require_id(dataset_id, "table dataset")
    _require_id(collection_id, "feature collection")
    require_table_id(dataset_id)
    period_type = _dataset_period_type(dataset_id, period_type)
    existing = _existing_record(dataset_id)
    unit = existing.units if existing else None
    feature_ids = _collection_feature_ids(collection_id)
    stamp = _utc(written_at or datetime.now(UTC), "written_at")
    columns: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        clean = _validated_row(row, index=index, dataset_id=dataset_id, collection_id=collection_id)
        if clean["feature_id"] not in feature_ids:
            raise ValueError(
                f"row {index}: feature '{clean['feature_id']}' is not in feature collection '{collection_id}'"
            )
        clean["period_start"] = start_of_period(clean["period_start"], period_type)
        if clean.get("unit") is not None:
            if unit is not None and clean["unit"] != unit:
                raise ValueError(
                    f"row {index}: table dataset '{dataset_id}' is in {unit}; it cannot take rows in {clean['unit']}"
                )
            unit = clean["unit"]
        clean["written_at"] = stamp
        key = tuple(clean.get(name) for name in KEY)
        if key in seen:
            raise ValueError(f"row {index}: another row in this write has the same key")
        seen.add(key)
        for name in SCHEMA.names:
            columns[name].append(clean.get(name))
    if not seen:
        raise ValueError(f"no rows to write to table dataset '{dataset_id}'")
    table = pa.table(columns, schema=SCHEMA)
    return table.replace_schema_metadata({PERIOD_TYPE_KEY: period_type.encode()}) if period_type else table


def read_rows(
    record: ArtifactRecord,
    *,
    collection_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
    reducer: str | None = None,
    weights: str | None = None,
) -> pa.Table:
    """Return a table dataset's rows, filtered, ordered by collection, feature and period.

    *start* and *end* bound `period_start`, inclusive.
    """
    if not record.asset_paths:
        return SCHEMA.empty_table()
    conditions = []
    if collection_id is not None:
        conditions.append(pc.field("collection_id") == collection_id)
    if start is not None:
        conditions.append(pc.field("period_start") >= start)
    if end is not None:
        conditions.append(pc.field("period_start") <= end)
    if reducer is not None:
        conditions.append(pc.field("reducer") == reducer)
    if weights is not None:
        conditions.append(pc.field("weights") == weights)
    flt = None
    for condition in conditions:
        flt = condition if flt is None else flt & condition
    return pds.dataset(record.asset_paths, schema=SCHEMA, format="parquet").to_table(filter=flt).sort_by(_ORDER)


_ORDER = [("collection_id", "ascending"), ("feature_id", "ascending"), ("period_start", "ascending")]
"""How rows are stored and returned, so a reader of the file finds a feature's periods together."""


def _read_file(record: ArtifactRecord | None) -> pa.Table | None:
    """The rows the dataset's file holds now, or None for a new dataset."""
    if record is None or not record.asset_paths:
        return None
    return pds.dataset(record.asset_paths, schema=SCHEMA, format="parquet").to_table()


def _merged(existing: pa.Table | None, new: pa.Table, collection_id: str, period_type: str | None) -> pa.Table:
    """The dataset's rows with *new* merged in: one row per `KEY`, the written one where they meet.

    A written row equal to the stored one in everything but `written_at` keeps the stored row, so
    sending the same values again changes nothing. When every written row names one version of
    the collection, the collection's rows of any other version are dropped.
    """
    stored = existing.to_pylist() if existing is not None else []
    versions = set(new.column("collection_version").to_pylist())
    if len(versions) == 1 and (version := next(iter(versions))) is not None:
        stored = [
            row for row in stored if row["collection_id"] != collection_id or row["collection_version"] == version
        ]
    rows = {tuple(row[name] for name in KEY): row for row in stored}
    compared = [name for name in SCHEMA.names if name != "written_at"]
    for row in new.to_pylist():
        key = tuple(row[name] for name in KEY)
        current = rows.get(key)
        if current is None or any(current[name] != row[name] for name in compared):
            rows[key] = row
    table = pa.Table.from_pylist(list(rows.values()), schema=SCHEMA).sort_by(_ORDER)
    return table.replace_schema_metadata({PERIOD_TYPE_KEY: period_type.encode()}) if period_type else table


def _write_file(folder: Path, table: pa.Table, stamp: datetime) -> Path:
    """Write *table* as a new file in *folder*, under a temporary name first, then renamed."""
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp:%Y%m%dT%H%M%S%fZ}-{uuid4().hex}.parquet"
    # A dot name no listing of `*.parquet` matches, so a half-written file is never taken for the dataset.
    staging = folder / f".{path.name}.writing"
    try:
        pq.write_table(table, staging)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def _dataset_root(dataset_id: str) -> Path:
    return api_config.get_tables_root() / dataset_id


def _prune_superseded(folder: Path, *, keep: Path) -> None:
    """Remove the files a newer file replaced more than `SUPERSEDED_FILE_GRACE_SECONDS` ago.

    A reader resolves the record, then opens the file it names, so a file just replaced must stay
    for a reader caught between the two. A file was replaced when the next newer one was written,
    so the next file's mtime is when it stopped being the dataset; *keep* is never removed. A file
    that cannot be removed is logged and tried again on the next write.
    """
    from open_climate_service.features.store import SUPERSEDED_FILE_GRACE_SECONDS

    files = sorted(folder.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    now = time.time()
    for older, newer in zip(files, files[1:], strict=False):
        if older == keep or now - newer.stat().st_mtime < SUPERSEDED_FILE_GRACE_SECONDS:
            continue
        try:
            older.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove superseded table file '%s'", older, exc_info=True)


def _remove_records(dataset_id: str) -> None:
    """Drop the dataset's records, for a first write whose publication failed."""
    from open_climate_service.ingestions import services as ingestion_services

    def remove(records: list[ArtifactRecord]) -> None:
        records[:] = [
            record for record in records if record.dataset_id != dataset_id or record.format != ArtifactFormat.PARQUET
        ]

    ingestion_services._mutate_records(remove)


def _existing_record(dataset_id: str) -> ArtifactRecord | None:
    from open_climate_service.ingestions import services as ingestion_services

    records = [
        record
        for record in ingestion_services.list_artifacts().items
        if record.dataset_id == dataset_id and record.format == ArtifactFormat.PARQUET
    ]
    return max(records, key=lambda record: record.created_at) if records else None


def _coverage(table: pa.Table, period_type: str | None) -> ArtifactCoverage:
    """The period range of *table*'s rows, as period labels, and the extent of the collections they report on."""
    from open_climate_service.features.services import registered_collections

    dates = table.column("period_start").unique().to_pylist()
    first, last = min(dates), max(dates)
    collections = registered_collections()
    boxes = [
        (record.coverage.spatial_wgs84 or record.coverage.spatial)
        for collection_id in sorted(set(table.column("collection_id").to_pylist()))
        if (record := collections.get(collection_id)) is not None
    ]
    spatial = CoverageSpatial(
        xmin=min(box.xmin for box in boxes),
        ymin=min(box.ymin for box in boxes),
        xmax=max(box.xmax for box in boxes),
        ymax=max(box.ymax for box in boxes),
    )
    return ArtifactCoverage(
        spatial=spatial,
        temporal=CoverageTemporal(start=period_label(first, period_type), end=period_label(last, period_type)),
    )


def _collection_feature_ids(collection_id: str) -> set[str]:
    """The feature ids of a registered collection, its `id_property` values."""
    from open_climate_service.features import store as feature_store
    from open_climate_service.features.services import registered_collections

    record = registered_collections().get(collection_id)
    if record is None or record.features is None:
        raise ValueError(f"feature collection '{collection_id}' is not registered")
    id_property = record.features.id_property
    return {
        str(row[id_property])
        for row in feature_store.read_feature_columns(record, [id_property])
        if row.get(id_property) is not None
    }


def _validated_row(row: object, *, index: int, dataset_id: str, collection_id: str) -> dict[str, Any]:
    """Return one row checked and normalised to `SCHEMA`, or raise ValueError naming the row."""
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index}: expected a mapping, got {type(row).__name__}")
    unknown = sorted(set(row) - _WRITABLE)
    if unknown:
        raise ValueError(f"row {index}: unknown columns {unknown}")
    missing = [name for name in _REQUIRED if name not in row]
    if missing:
        raise ValueError(f"row {index}: missing required columns {missing}")
    for name, expected in (("dataset_id", dataset_id), ("collection_id", collection_id)):
        if row.get(name, expected) != expected:
            raise ValueError(f"row {index}: {name} '{row[name]}' does not match '{expected}' this write is for")
    clean = dict(row)
    clean["dataset_id"] = dataset_id
    clean["collection_id"] = collection_id
    feature_id = row["feature_id"]
    if not isinstance(feature_id, str) or not feature_id:
        raise ValueError(f"row {index}: feature_id must be a non-empty string")
    clean["period_start"] = _period(row["period_start"], index=index)
    clean["value"] = _number(row["value"], "value", index=index)
    coverage = _number(row.get("coverage"), "coverage", index=index)
    if coverage is not None and not 0 <= coverage <= 1:
        raise ValueError(f"row {index}: coverage must be between 0 and 1, got {coverage}")
    if coverage == 0 and clean["value"] is not None:
        raise ValueError(
            f"row {index}: value {clean['value']} has coverage 0; a unit with no valid cells has no value, "
            "so store None rather than a number"
        )
    clean["coverage"] = coverage
    if row.get("reference_time") is not None:
        clean["reference_time"] = _utc(row["reference_time"], f"row {index}: reference_time")
    if row.get("lead_time") is not None and not isinstance(row["lead_time"], timedelta):
        raise ValueError(f"row {index}: lead_time must be a timedelta")
    for name in ("feature_name", "class", "reducer", "weights", "unit", "collection_version"):
        if row.get(name) is not None and not isinstance(row[name], str):
            raise ValueError(f"row {index}: {name} must be a string")
    return clean


def start_of_period(day: date, period_type: str | None) -> date:
    """The first day of the *period_type* period that contains *day*; *day* itself for any other.

    Producers label a period by different days: pandas resampling dates a month by its last day
    and a week by its Sunday, while OCS stores date them by their first. Storing the start keeps
    one key per period, so a revision replaces a value rather than standing beside it.
    """
    from open_climate_service.shared.time import dekad_start

    if period_type == "weekly":
        return day - timedelta(days=day.weekday())
    if period_type == "dekadal":
        return dekad_start(day)
    if period_type == "monthly":
        return day.replace(day=1)
    if period_type == "quarterly":
        return day.replace(month=3 * ((day.month - 1) // 3) + 1, day=1)
    if period_type == "yearly":
        return day.replace(month=1, day=1)
    return day


def end_of_period(start: date, period_type: str | None) -> date | None:
    """The last day, inclusive, of the *period_type* period starting on *start*; None for a type without one."""
    import calendar

    from open_climate_service.shared.time import dekad_bounds

    if period_type == "daily":
        return start
    if period_type == "weekly":
        return start + timedelta(days=6)
    if period_type == "dekadal":
        return dekad_bounds(start)[1]
    if period_type in ("monthly", "quarterly"):
        month = start.month + (2 if period_type == "quarterly" else 0)
        return start.replace(month=month, day=calendar.monthrange(start.year, month)[1])
    if period_type == "yearly":
        return start.replace(month=12, day=31)
    return None


def period_label(day: date, period_type: str | None) -> str:
    """How OCS names the period starting on *day*, as raster datasets name theirs: `2026-01`, `2026-W01`, `2026`.

    For reading only: storage, filters and the key use the date. A dekad is named by its first
    day, as in raster datasets; a period type OCS has no name for keeps the ISO date.
    """
    from open_climate_service.shared.time import datetime_to_period_string

    if period_type == "quarterly":
        return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"
    if period_type in ("daily", "dekadal", "weekly", "monthly", "yearly"):
        return datetime_to_period_string(datetime(day.year, day.month, day.day), period_type)
    return day.isoformat()


def _period(value: object, *, index: int) -> date:
    """A period start as a date, from a date or an ISO date string. A datetime is refused: periods are days."""
    if isinstance(value, datetime):
        raise ValueError(f"row {index}: period_start must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"row {index}: period_start must be a date or an ISO date string, got {value!r}")


def _number(value: object, name: str, *, index: int) -> float | None:
    """A finite float, or None for a missing one. NaN means missing; infinity and booleans are refused."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {index}: {name} must be a number or None, got {value!r}")
    number = float(value)
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ValueError(f"row {index}: {name} must be finite")
    return number


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _dataset_period_type(dataset_id: str, period_type: str | None) -> str | None:
    """The period type a write uses: the dataset's own, which *period_type* must not contradict."""
    existing = _existing_record(dataset_id)
    current = existing.period_type if existing else None
    if existing is not None and current is None and period_type is not None:
        # Its stored periods were never normalized to a period's start, so typed rows would not
        # share their keys.
        raise ValueError(f"table dataset '{dataset_id}' has no period type; it cannot take {period_type} rows")
    if period_type is not None and current is not None and period_type != current:
        raise ValueError(
            f"table dataset '{dataset_id}' is {current}; it cannot take {period_type} rows. "
            "Write them to another table dataset"
        )
    return period_type or current


def require_table_id(dataset_id: str) -> None:
    """Refuse a dataset id that a raster or feature collection already holds."""
    from open_climate_service.ingestions import services as ingestion_services

    for record in ingestion_services.group_datasets().get(dataset_id, []):
        if record.format != ArtifactFormat.PARQUET:
            raise ValueError(f"dataset id '{dataset_id}' is held by a {record.format} dataset; choose another id")


def _require_id(value: str, kind: str) -> None:
    if not is_segment_safe_id(value):
        raise ValueError(
            f"invalid {kind} id {value!r}; it names a folder and a URL segment, so it must start with a "
            "letter or digit and carry only letters, digits, '.', '_' or '-'"
        )
