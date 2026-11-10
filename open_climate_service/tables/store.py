"""The table store: values per feature and period, appended as Parquet files (CLIM-1553).

One folder per table dataset under `config.get_tables_root`, and one folder per feature
collection inside it. Every write adds a new immutable file, written under a temporary name and
renamed. The dataset's record lists its committed files in `asset_paths`, and readers read only
those: a file becomes part of the dataset when the record naming it is registered, so a reader
sees the files before a write or after it, and a write whose record fails leaves nothing behind.
Rows are never updated or deleted: a revised value is a new row with a later `written_at`, and
the current value is the latest row per `KEY`.

`write_rows` is the one writer. It validates rows against the feature collection they report
on, appends them, and registers the dataset's record, which is what lists it under `/datasets`
as `itemType: table`. `check_rows` gives the same verdict without writing, for a dry run.
`read_rows` reads them back with pyarrow.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable, Mapping
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
from open_climate_service.shared.urls import is_segment_safe_id

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
"""The columns that identify one value. The current value is the latest `written_at` per key."""

PERIOD_TYPE_KEY = b"ocs:period_type"
"""Parquet metadata key holding the dataset's period type, so a reader of the files alone knows
whether `period_start` starts a day, a month or a year."""

_WRITABLE = frozenset(SCHEMA.names) - {"written_at"}
_REQUIRED = ("feature_id", "period_start", "value")

_dataset_locks: dict[str, threading.Lock] = {}
_dataset_locks_mutex = threading.Lock()


def _dataset_lock(dataset_id: str) -> threading.Lock:
    """Serialize writes to one dataset, so its record always describes the files that are there."""
    with _dataset_locks_mutex:
        return _dataset_locks.setdefault(dataset_id, threading.Lock())


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
    """Check *rows* with `check_rows`, append them as one file and register the dataset.

    Raises ValueError before anything is written when any row is refused. *provenance* is how
    the rows were produced, for computed rows; the record keeps the latest write's. *publish*
    applies to a new dataset; a later write replaces the dataset's record and keeps its publication.
    """
    stamp = _utc(written_at or datetime.now(UTC), "written_at")

    from open_climate_service.ingestions import services as ingestion_services

    # Checked under the lock, so two first writes cannot give one dataset two period types.
    with _dataset_lock(dataset_id):
        table = check_rows(
            dataset_id=dataset_id, collection_id=collection_id, rows=rows, period_type=period_type, written_at=stamp
        )
        written = _append(_dataset_root(dataset_id) / collection_id, table, stamp)
        try:
            existing = _existing_record(dataset_id)
            period_type = period_type or (existing.period_type if existing else None)
            files = [*(existing.asset_paths if existing else []), str(written)]
            units = set(table.column("unit").drop_null().to_pylist())
            record = ArtifactRecord(
                artifact_id=str(uuid4()),
                dataset_id=dataset_id,
                dataset_name=dataset_name or (existing.dataset_name if existing else dataset_id),
                period_type=period_type,
                format=ArtifactFormat.PARQUET,
                path=str(_dataset_root(dataset_id)),
                asset_paths=files,
                units=units.pop() if len(units) == 1 else (existing.units if existing else None),
                size_bytes=sum(Path(file).stat().st_size for file in files),
                request_scope=ArtifactRequestScope(),
                coverage=_coverage(files, period_type),
                created_at=datetime.now(UTC),
                provenance=provenance or (existing.provenance if existing else None),
            )
            return ingestion_services.register_artifact_record(record, publish=publish)
        except BaseException:
            # No record names the file, so no reader sees it; remove it rather than leave it.
            written.unlink(missing_ok=True)
            raise


def check_rows(
    *,
    dataset_id: str,
    collection_id: str,
    rows: Iterable[Mapping[str, Any]],
    period_type: str | None = None,
    written_at: datetime | None = None,
) -> pa.Table:
    """Return *rows* as the table `write_rows` would append, or raise ValueError; nothing is written.

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
    """
    _require_id(dataset_id, "table dataset")
    _require_id(collection_id, "feature collection")
    require_table_id(dataset_id)
    period_type = _dataset_period_type(dataset_id, period_type)
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
    history: bool = False,
) -> pa.Table:
    """Return a table dataset's rows, filtered, ordered by collection, feature and period.

    Current values only, unless *history* asks for every row ever written. The filters are all
    on key columns, so filtering before picking the latest row per key gives the same answer as
    filtering after it. *start* and *end* bound `period_start`, inclusive.
    """
    files = _files(record, collection_id)
    if not files:
        return SCHEMA.empty_table()
    conditions = []
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
    table = pds.dataset([str(path) for path in files], schema=SCHEMA, format="parquet").to_table(filter=flt)
    # A key written twice at the same `written_at` is one write delivered again; history lists it once.
    table = _latest(table, KEY if not history else (*KEY, "written_at"))
    return table.sort_by(
        [
            ("collection_id", "ascending"),
            ("feature_id", "ascending"),
            ("period_start", "ascending"),
            ("written_at", "ascending"),
        ]
    )


def _latest(table: pa.Table, keys: tuple[str, ...]) -> pa.Table:
    """Keep the latest row per *keys*. pandas, because its duplicate test treats NULL keys as equal."""
    frame = table.select([*KEY, "written_at"]).to_pandas()
    frame["row"] = range(len(frame))
    latest = frame.sort_values("written_at", kind="stable").drop_duplicates(subset=list(keys), keep="last")
    return table.take(pa.array(sorted(latest["row"]), type=pa.int64()))


def _append(folder: Path, table: pa.Table, stamp: datetime) -> Path:
    """Write *table* as a new file in *folder*, under a temporary name first, then renamed."""
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp:%Y%m%dT%H%M%S%fZ}-{uuid4().hex}.parquet"
    # A dot name the reader's `*.parquet` listing never matches, so a half-written file is invisible.
    staging = folder / f".{path.name}.writing"
    try:
        pq.write_table(table, staging)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def _files(record: ArtifactRecord, collection_id: str | None) -> list[Path]:
    """The files *record* commits, for one collection or all of them; never a file it does not name."""
    files = [Path(file) for file in record.asset_paths]
    if collection_id is not None:
        if not is_segment_safe_id(collection_id):
            return []
        files = [file for file in files if file.parent.name == collection_id]
    return sorted(files)


def _dataset_root(dataset_id: str) -> Path:
    return api_config.get_tables_root() / dataset_id


def _existing_record(dataset_id: str) -> ArtifactRecord | None:
    from open_climate_service.ingestions import services as ingestion_services

    records = [
        record
        for record in ingestion_services.list_artifacts().items
        if record.dataset_id == dataset_id and record.format == ArtifactFormat.PARQUET
    ]
    return max(records, key=lambda record: record.created_at) if records else None


def _coverage(files: list[str], period_type: str | None) -> ArtifactCoverage:
    """The period range of the rows in *files*, as period labels, and the extent of the collections they report on."""
    from open_climate_service.features.services import registered_collections

    periods = pds.dataset(files, schema=SCHEMA, format="parquet").to_table(columns=["period_start"])
    dates = periods.column("period_start").unique().to_pylist()
    first, last = min(dates), max(dates)
    collections = registered_collections()
    boxes = [
        (record.coverage.spatial_wgs84 or record.coverage.spatial)
        for folder in sorted({Path(file).parent.name for file in files})
        if (record := collections.get(folder)) is not None
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
