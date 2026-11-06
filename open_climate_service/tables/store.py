"""The table store: values per feature and period, appended as Parquet files (CLIM-1553).

One folder per table dataset under `config.get_tables_root`, and one folder per feature
collection inside it. Every write adds a new immutable file, written under a temporary name and
renamed, so a reader lists either the files before a write or the files after it, never a
partial one. Rows are never updated or deleted: a revised value is a new row with a later
`written_at`, and the current value is the latest row per `KEY`.

`write_rows` is the one writer. It validates rows against the feature collection they report
on, appends them, and registers the dataset's record, which is what lists it under `/datasets`
as `itemType: table`. `read_rows` reads them back with pyarrow.
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
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
)
from open_climate_service.shared.storage_size import stored_bytes
from open_climate_service.shared.urls import is_segment_safe_id

SCHEMA = pa.schema(
    [
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("feature_id", pa.string(), nullable=False),
        pa.field("feature_name", pa.string()),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("period", pa.date32(), nullable=False),
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
    "period",
    "class",
    "reducer",
    "weights",
    "reference_time",
    "lead_time",
    "member",
)
"""The columns that identify one value. The current value is the latest `written_at` per key."""

_WRITABLE = frozenset(SCHEMA.names) - {"written_at"}
_REQUIRED = ("feature_id", "period", "value")

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
    publish: bool = True,
) -> ArtifactRecord:
    """Validate *rows* for one feature collection, append them as one file and register the dataset.

    Every row names a `feature_id` of the registered collection, a `period` and a `value`; the
    other `SCHEMA` columns are optional. `dataset_id` and `collection_id` come from the
    arguments, and a row that names others is refused. A missing value is None (NaN is read as
    None); a value with `coverage` 0 is refused, because a unit without valid cells has no value,
    and 0 would be exported as a measurement.

    All rows of a write share one `written_at`, now unless given. Raises ValueError before
    anything is written when any row is refused.
    """
    _require_id(dataset_id, "table dataset")
    _require_id(collection_id, "feature collection")
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

    from open_climate_service.ingestions import services as ingestion_services

    with _dataset_lock(dataset_id):
        _append(_dataset_root(dataset_id) / collection_id, table, stamp)
        existing = _existing_record(dataset_id)
        record = ArtifactRecord(
            artifact_id=str(uuid4()),
            dataset_id=dataset_id,
            dataset_name=dataset_name or (existing.dataset_name if existing else dataset_id),
            period_type=period_type or (existing.period_type if existing else None),
            format=ArtifactFormat.PARQUET,
            path=str(_dataset_root(dataset_id)),
            size_bytes=stored_bytes(_dataset_root(dataset_id)),
            request_scope=ArtifactRequestScope(),
            coverage=_coverage(dataset_id),
            created_at=datetime.now(UTC),
        )
        return ingestion_services.register_artifact_record(record, publish=publish)


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
    filtering after it. *start* and *end* are inclusive period starts.
    """
    files = _files(Path(str(record.path)), collection_id)
    if not files:
        return SCHEMA.empty_table()
    conditions = []
    if start is not None:
        conditions.append(pc.field("period") >= start)
    if end is not None:
        conditions.append(pc.field("period") <= end)
    if reducer is not None:
        conditions.append(pc.field("reducer") == reducer)
    if weights is not None:
        conditions.append(pc.field("weights") == weights)
    flt = None
    for condition in conditions:
        flt = condition if flt is None else flt & condition
    table = pds.dataset([str(path) for path in files], schema=SCHEMA, format="parquet").to_table(filter=flt)
    if not history:
        table = _current(table)
    return table.sort_by(
        [
            ("collection_id", "ascending"),
            ("feature_id", "ascending"),
            ("period", "ascending"),
            ("written_at", "ascending"),
        ]
    )


def _current(table: pa.Table) -> pa.Table:
    """Keep the latest row per `KEY`. pandas, because its duplicate test treats NULL keys as equal."""
    keys = table.select(list(KEY)).to_pandas()
    keys["written_at"] = table.column("written_at").to_pandas()
    keys["row"] = range(len(keys))
    latest = keys.sort_values("written_at", kind="stable").drop_duplicates(subset=list(KEY), keep="last")
    return table.take(pa.array(sorted(latest["row"])))


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


def _files(root: Path, collection_id: str | None) -> list[Path]:
    """The dataset's committed files, for one collection or all of them."""
    if collection_id is not None:
        if not is_segment_safe_id(collection_id):
            return []
        return sorted((root / collection_id).glob("*.parquet"))
    return sorted(root.glob("*/*.parquet"))


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


def _coverage(dataset_id: str) -> ArtifactCoverage:
    """The period range of every row stored, and the extent of the collections they report on."""
    from open_climate_service.features.services import registered_collections

    root = _dataset_root(dataset_id)
    files = _files(root, None)
    periods = pds.dataset([str(path) for path in files], schema=SCHEMA, format="parquet").to_table(columns=["period"])
    dates = periods.column("period").unique().to_pylist()
    first, last = min(dates), max(dates)
    collections = registered_collections()
    boxes = [
        (record.coverage.spatial_wgs84 or record.coverage.spatial)
        for folder in sorted(path.name for path in root.iterdir() if path.is_dir())
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
        temporal=CoverageTemporal(start=first.isoformat(), end=last.isoformat()),
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
    clean["period"] = _period(row["period"], index=index)
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


def _period(value: object, *, index: int) -> date:
    """A period start as a date, from a date or an ISO date string. A datetime is refused: periods are days."""
    if isinstance(value, datetime):
        raise ValueError(f"row {index}: period must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"row {index}: period must be a date or an ISO date string, got {value!r}")


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


def _require_id(value: str, kind: str) -> None:
    if not is_segment_safe_id(value):
        raise ValueError(
            f"invalid {kind} id {value!r}; it names a folder and a URL segment, so it must start with a "
            "letter or digit and carry only letters, digits, '.', '_' or '-'"
        )
