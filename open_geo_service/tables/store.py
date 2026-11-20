"""The table store: a table dataset's current rows, one Parquet file per dataset (CLIM-1553).

A table's rows follow its schema (`tables.schemas`, CLIM-1971): the columns it declares, the key
that says which stored row a write replaces, and what columns represent, which adds rules: feature ids checked
against their collection and periods stored as their first day. The dataset's record names its schema.

One folder per table dataset under `config.get_tables_root`, holding the dataset's one file of
current values. No history is kept: a revised value replaces its row, and a value written again
unchanged keeps the row and its `written_at`, so `written_at` says when each value last changed.

Every write produces a new file, written under a temporary name and renamed, holding the merged
rows. The dataset's record names it in `asset_paths`, and readers read only that: the file
becomes the dataset when the record naming it is registered. The previous file stays for a grace
period, for a reader that resolved the previous record and has not opened its file yet.
A reader sees the dataset before a write or after it, and a write whose record fails leaves the
dataset as it was. The one file is what a static catalog serves as the dataset's data.

`write_rows` is the one writer. It validates rows against the dataset's schema, merges them, and
registers the dataset's record, which is what lists it under `/datasets` as `itemType: table`.
`check_rows` gives the same verdict without writing, for a dry run. `read_rows` reads them back
with pyarrow.
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
from typing import Any, NamedTuple
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

from open_geo_service import config as api_config
from open_geo_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
)
from open_geo_service.shared.persistence import index_lock
from open_geo_service.shared.urls import is_segment_safe_id
from open_geo_service.tables.schemas import DEFAULT_SCHEMA, WRITTEN_AT, TableSchema, get_schema

logger = logging.getLogger(__name__)

PERIOD_TYPE_KEY = b"ogs:period_type"
"""Parquet metadata key holding the dataset's period type, so a reader of the files alone knows
whether a period column holds the start of a day, a month or a year."""

SCHEMA_KEY = b"ogs:table_schema"
"""Parquet metadata key holding the id of the dataset's schema (`tables.schemas`)."""


def schema_of(record: ArtifactRecord | None) -> TableSchema:
    """The schema a table dataset's rows follow: its record's, or the default for a dataset without one."""
    return get_schema(record.table_schema if record is not None and record.table_schema else DEFAULT_SCHEMA)


def _dataset_lock(dataset_id: str) -> AbstractContextManager[None]:
    """Serialize writes to one dataset across processes, so no write's record drops another's file.

    The lock file sits beside the records index, so a refused write leaves the table store as it was.
    """
    from open_geo_service.ingestions import services as ingestion_services

    _require_id(dataset_id, "table dataset")
    return index_lock(Path(ingestion_services.ARTIFACTS_DIR) / "locks" / f"table-{dataset_id}")


def write_rows(
    *,
    dataset_id: str,
    rows: Iterable[Mapping[str, Any]],
    collection_id: str | None = None,
    schema: str | None = None,
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

    from open_geo_service.ingestions import services as ingestion_services

    # Checked under the lock, so two first writes cannot give one dataset two schemas, period types or units.
    with _dataset_lock(dataset_id):
        table = check_rows(
            dataset_id=dataset_id,
            collection_id=collection_id,
            rows=rows,
            schema=schema,
            period_type=period_type,
            written_at=stamp,
            provenance=provenance,
        )
        existing = latest_record(dataset_id)
        declared = get_schema(table.schema.metadata[SCHEMA_KEY].decode())
        period_type = period_type or (existing.period_type if existing else None)
        merged = _merged(_read_file(existing, declared), table, declared, collection_id, period_type)
        written = _write_file(_dataset_root(dataset_id), merged, stamp)
        try:
            unit_column = declared.represented_by("unit")
            units = table.column(unit_column).drop_null().unique().to_pylist() if unit_column else []
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
                table_schema=declared.id,
                request_scope=ArtifactRequestScope(),
                coverage=_coverage(merged, declared, period_type),
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


def _require_same_sources(
    dataset_id: str, existing: ArtifactRecord | None, provenance: ArtifactProvenance | None
) -> None:
    """Refuse rows computed from other datasets than the table's stored rows were.

    A write keeps the stored rows it does not replace, while the record keeps only the latest
    write's provenance, from which the table's licence and providers are derived
    (`tables.sources`). Rows from other sources would leave the stored ones attributed to, and
    licensed as, datasets they were not computed from.
    """
    from open_geo_service.ingestions import services as ingestion_services

    if existing is None or existing.provenance is None or provenance is None:
        return
    before = {str(source["collection_id"]) for source in ingestion_services.recorded_inputs(existing)}
    written = existing.model_copy(update={"provenance": provenance})
    after = {str(source["collection_id"]) for source in ingestion_services.recorded_inputs(written)}
    if before and after != before:
        raise ValueError(
            f"table dataset '{dataset_id}' was computed from {sorted(before)}, not {sorted(after)}; "
            "its licence and providers come from the datasets it was computed from. "
            "Write them to another table dataset"
        )


def check_rows(
    *,
    dataset_id: str,
    rows: Iterable[Mapping[str, Any]],
    collection_id: str | None = None,
    schema: str | None = None,
    period_type: str | None = None,
    written_at: datetime | None = None,
    provenance: ArtifactProvenance | None = None,
) -> pa.Table:
    """Return *rows* as the table `write_rows` would merge in, or raise ValueError; nothing is written.

    For a dry run, which needs the writer's verdict without the write. Rows follow the dataset's
    schema: *schema* names it for a new dataset, the default one when None, and a dataset keeps
    the one it was first written with. A row names every `required` column, no undeclared one,
    and a value of each column's type; a missing value is None (NaN is read as None). Every row
    shares one `written_at`, now unless given. Rows that repeat a key are refused, and so is a
    dataset id a raster or feature collection already holds, so the two never share a listing.

    What the schema's columns represent adds rules (docs/managed_data_api_guide.md, Table schemas):

    - `dataset` and `collection` are filled from *dataset_id* and *collection_id*; a row naming others is refused.
    - `feature` must be a feature of the registered *collection_id*, which a schema with a feature column requires.
    - `period` is stored as the first day of the period that contains it (`start_of_period`). A
      dataset has one period type, the first written; a *period_type* that differs is refused,
      None keeps it. The table carries it as Parquet metadata (`PERIOD_TYPE_KEY`).
    - `unit`: a dataset has one, the first written; rows in another are refused.
    - `coverage` is from 0 to 1, and a value with coverage 0 is refused: a feature without valid
      data has no value, and 0 would be exported as a measurement.

    Rows computed from other datasets than the stored ones, as *provenance* names them, are
    refused (`_require_same_sources`).
    """
    _require_id(dataset_id, "table dataset")
    require_table_id(dataset_id)
    existing = latest_record(dataset_id)
    declared = _dataset_schema(dataset_id, existing, schema)
    _require_same_sources(dataset_id, existing, provenance)
    fixed: dict[str, str] = {}
    if dataset_column := declared.represented_by("dataset_id"):
        fixed[dataset_column] = dataset_id
    feature_ids: set[str] | None = None
    if collection_column := declared.represented_by("collection_id"):
        if collection_id is None:
            raise ValueError(f"table schema '{declared.id}' reports on features; name their feature collection")
        _require_id(collection_id, "feature collection")
        fixed[collection_column] = collection_id
        feature_ids = _collection_feature_ids(collection_id)
    elif collection_id is not None:
        raise ValueError(f"table schema '{declared.id}' has no feature column, so rows report on no collection")
    period_column = declared.represented_by("period_start")
    if period_column:
        period_type = _dataset_period_type(dataset_id, period_type)
    elif period_type is not None:
        raise ValueError(f"table schema '{declared.id}' has no period column, so it has no period type")
    feature_column, unit_column = declared.represented_by("feature_id"), declared.represented_by("unit")
    unit = existing.units if existing else None
    stamp = _utc(written_at or datetime.now(UTC), "written_at")
    columns: dict[str, list[Any]] = {name: [] for name in declared.names}
    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        clean = _validated_row(row, index=index, schema=declared, fixed=fixed)
        if feature_ids is not None and feature_column and clean[feature_column] not in feature_ids:
            raise ValueError(
                f"row {index}: feature '{clean[feature_column]}' is not in feature collection '{collection_id}'"
            )
        if period_column:
            clean[period_column] = start_of_period(clean[period_column], period_type)
        if unit_column and clean.get(unit_column) is not None:
            if unit is not None and clean[unit_column] != unit:
                raise ValueError(
                    f"row {index}: table dataset '{dataset_id}' is in {unit}; "
                    f"it cannot take rows in {clean[unit_column]}"
                )
            unit = clean[unit_column]
        clean[WRITTEN_AT] = stamp
        key = tuple(clean.get(name) for name in declared.key)
        if key in seen:
            raise ValueError(f"row {index}: another row in this write has the same key")
        seen.add(key)
        for name in declared.names:
            columns[name].append(clean.get(name))
    if not seen:
        raise ValueError(f"no rows to write to table dataset '{dataset_id}'")
    return pa.table(columns, schema=declared.arrow).replace_schema_metadata(_metadata(declared, period_type))


def _metadata(declared: TableSchema, period_type: str | None) -> dict[bytes, bytes]:
    """The Parquet metadata of a table's file: its schema id and, when it has one, its period type."""
    metadata = {SCHEMA_KEY: declared.id.encode()}
    if period_type:
        metadata[PERIOD_TYPE_KEY] = period_type.encode()
    return metadata


def read_rows(
    record: ArtifactRecord,
    *,
    collection_id: str | None = None,
    feature_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
    equals: Mapping[str, str] | None = None,
) -> pa.Table:
    """Return a table dataset's rows, filtered, in the order the file keeps them (`_order`).

    *collection_id*, *feature_id*, *start* and *end* filter on the columns representing them;
    *start* and *end* bound the period, inclusive. *equals* keeps rows whose named columns hold
    the given values. A filter on a column the table's schema lacks raises ValueError.
    """
    declared = schema_of(record)
    conditions = []
    for meaning, value in (("collection_id", collection_id), ("feature_id", feature_id)):
        if value is not None:
            conditions.append(pc.field(_column_representing(declared, meaning)) == value)
    if start is not None:
        conditions.append(pc.field(_column_representing(declared, "period_start")) >= start)
    if end is not None:
        conditions.append(pc.field(_column_representing(declared, "period_start")) <= end)
    for name, value in (equals or {}).items():
        if declared.column(name) is None:
            raise ValueError(f"table schema '{declared.id}' has no column '{name}'")
        conditions.append(pc.field(name) == value)
    if not record.asset_paths:
        return declared.arrow.empty_table()
    flt = None
    for condition in conditions:
        flt = condition if flt is None else flt & condition
    dataset = pds.dataset(record.asset_paths, schema=declared.arrow, format="parquet")
    return dataset.to_table(filter=flt).sort_by(_order(declared))


def _column_representing(declared: TableSchema, meaning: str) -> str:
    column = declared.represented_by(meaning)
    if column is None:
        raise ValueError(f"table schema '{declared.id}' has no column representing {meaning}")
    return column


class FileSummary(NamedTuple):
    """What a table dataset's file holds, for its STAC collection and README."""

    row_count: int
    collection_ids: list[str]
    first_period: date | None
    last_period: date | None


def summarize(record: ArtifactRecord) -> FileSummary:
    """Count the dataset's rows and find the collections and periods they cover, where it has those columns.

    The row count and period range come from the Parquet footer, so no values are read; the
    collection ids from the dictionary of their column, one small index per row.
    """
    import pyarrow.parquet as pq

    declared = schema_of(record)
    period_column, collection_column = declared.represented_by("period_start"), declared.represented_by("collection_id")
    row_count = 0
    collection_ids: set[str] = set()
    periods: list[date] = []
    for path in record.asset_paths:
        metadata = pq.ParquetFile(path).metadata
        row_count += metadata.num_rows
        if period_column:
            column = metadata.schema.names.index(period_column)
            for group in range(metadata.num_row_groups):
                statistics = metadata.row_group(group).column(column).statistics
                if statistics is not None and statistics.has_min_max:
                    periods += [statistics.min, statistics.max]
        if collection_column:
            ids = pq.read_table(path, columns=[collection_column], read_dictionary=[collection_column])
            collection_ids.update(
                str(value) for chunk in ids.column(collection_column).chunks for value in chunk.dictionary.to_pylist()
            )
    if row_count == 0 or not periods:
        return FileSummary(row_count, sorted(collection_ids), None, None)
    return FileSummary(row_count, sorted(collection_ids), min(periods), max(periods))


class LatestValues(NamedTuple):
    """Each feature's rows for the latest period it has a row for, for a table dataset's page."""

    rows: pa.Table
    """The first rows by feature name, at most the limit asked for."""
    row_count: int
    """The latest rows of every feature."""
    periods: list[date]
    """Every feature's latest period, each once."""


def latest_values(record: ArtifactRecord, limit: int) -> LatestValues | None:
    """Each feature's rows for the latest period it has a row for, ordered by feature name.

    None for a table without a feature or a period column. Per feature, not the
    dataset's last period: a feature whose newest value is older still has one. A feature with
    several values per period (per class, reducer or forecast) keeps them all. Whole rows are
    read for the first *limit* only; the latest periods are found from a few columns.
    """
    declared = schema_of(record)
    collection, feature, period = (
        declared.represented_by(meaning) for meaning in ("collection_id", "feature_id", "period_start")
    )
    if collection is None or feature is None or period is None:
        return None
    if not record.asset_paths:
        return LatestValues(declared.arrow.empty_table(), 0, [])
    name = declared.represented_by("feature_name")
    by_name = [*([(name, "ascending")] if name else []), (feature, "ascending")]
    dataset = pds.dataset(record.asset_paths, schema=declared.arrow, format="parquet")
    keys = [collection, feature]
    newest_column = f"{period}_max"
    narrow = dataset.to_table(columns=[*keys, *([name] if name else []), period])
    newest = narrow.group_by(keys).aggregate([(period, "max")])
    latest = narrow.join(newest, keys=keys).filter(pc.field(period) == pc.field(newest_column)).sort_by(by_name)
    shown_keys = [*keys, newest_column]
    shown = latest.slice(0, limit).select(shown_keys).group_by(shown_keys).aggregate([])
    candidates = dataset.to_table(
        filter=pc.field(feature).isin(shown.column(feature)) & pc.field(period).isin(shown.column(newest_column))
    )
    # A feature's rows of another shown feature's period are not its latest.
    rows = (
        candidates.join(shown, keys=keys)
        .filter(pc.field(period) == pc.field(newest_column))
        .drop_columns([newest_column])
        .sort_by(by_name)
        .slice(0, limit)
    )
    return LatestValues(rows, latest.num_rows, newest.column(newest_column).unique().to_pylist())


def _order(declared: TableSchema) -> list[tuple[str, str]]:
    """How rows are stored and returned: by collection, feature and period, as the schema has them.

    A reader of the file then finds a feature's periods together. A schema with none of those
    columns orders by its key.
    """
    located = [
        column
        for meaning in ("collection_id", "feature_id", "period_start")
        if (column := declared.represented_by(meaning))
    ]
    return [(name, "ascending") for name in located or declared.key]


def _read_file(record: ArtifactRecord | None, declared: TableSchema) -> pa.Table | None:
    """The rows the dataset's file holds now, or None for a new dataset."""
    if record is None or not record.asset_paths:
        return None
    return pds.dataset(record.asset_paths, schema=declared.arrow, format="parquet").to_table()


def _merged(
    existing: pa.Table | None, new: pa.Table, declared: TableSchema, collection_id: str | None, period_type: str | None
) -> pa.Table:
    """The dataset's rows with *new* merged in: one row per key, the written one where they meet.

    A written row equal to the stored one in everything but `written_at` keeps the stored row, so
    sending the same values again changes nothing. With a column representing `collection_version`, when every
    written row names one version of the collection, the collection's rows of any other version
    are dropped, so values for two boundary versions never mix.
    """
    stored = existing.to_pylist() if existing is not None else []
    version_column, collection_column = (
        declared.represented_by("collection_version"),
        declared.represented_by("collection_id"),
    )
    if version_column and collection_column:
        versions = set(new.column(version_column).to_pylist())
        if len(versions) == 1 and (version := next(iter(versions))) is not None:
            stored = [
                row for row in stored if row[collection_column] != collection_id or row[version_column] == version
            ]
    rows = {tuple(row[name] for name in declared.key): row for row in stored}
    compared = [name for name in declared.names if name != WRITTEN_AT]
    for row in new.to_pylist():
        key = tuple(row[name] for name in declared.key)
        current = rows.get(key)
        if current is None or any(current[name] != row[name] for name in compared):
            rows[key] = row
    table = pa.Table.from_pylist(list(rows.values()), schema=declared.arrow).sort_by(_order(declared))
    return table.replace_schema_metadata(_metadata(declared, period_type))


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
    from open_geo_service.features.store import SUPERSEDED_FILE_GRACE_SECONDS

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
    from open_geo_service.ingestions import services as ingestion_services

    def remove(records: list[ArtifactRecord]) -> None:
        records[:] = [
            record for record in records if record.dataset_id != dataset_id or record.format != ArtifactFormat.PARQUET
        ]

    ingestion_services._mutate_records(remove)


def latest_record(dataset_id: str) -> ArtifactRecord | None:
    """The table dataset's newest record, published or not, or None for a dataset never written."""
    from open_geo_service.ingestions import services as ingestion_services

    records = [
        record
        for record in ingestion_services.list_artifacts().items
        if record.dataset_id == dataset_id and record.format == ArtifactFormat.PARQUET
    ]
    return max(records, key=lambda record: record.created_at) if records else None


_WHOLE_WORLD = CoverageSpatial(xmin=-180, ymin=-90, xmax=180, ymax=90)
"""The extent of a table whose rows report on no feature collection: it says nothing narrower."""


def _coverage(table: pa.Table, declared: TableSchema, period_type: str | None) -> ArtifactCoverage:
    """The period range of *table*'s rows, as period labels, and the extent of the collections they report on.

    Each from the column representing it: a table without a period column has no temporal extent,
    and one without a collection column covers the whole world.
    """
    from open_geo_service.features.services import registered_collections

    temporal = CoverageTemporal()
    if period_column := declared.represented_by("period_start"):
        dates = table.column(period_column).unique().to_pylist()
        temporal = CoverageTemporal(
            start=period_label(min(dates), period_type), end=period_label(max(dates), period_type)
        )
    spatial = _WHOLE_WORLD
    if collection_column := declared.represented_by("collection_id"):
        collections = registered_collections()
        boxes = [
            (record.coverage.spatial_wgs84 or record.coverage.spatial)
            for collection_id in sorted(set(table.column(collection_column).to_pylist()))
            if (record := collections.get(collection_id)) is not None
        ]
        spatial = CoverageSpatial(
            xmin=min(box.xmin for box in boxes),
            ymin=min(box.ymin for box in boxes),
            xmax=max(box.xmax for box in boxes),
            ymax=max(box.ymax for box in boxes),
        )
    return ArtifactCoverage(spatial=spatial, temporal=temporal)


def _collection_feature_ids(collection_id: str) -> set[str]:
    """The feature ids of a registered collection, its `id_property` values."""
    from open_geo_service.features import store as feature_store
    from open_geo_service.features.services import registered_collections

    record = registered_collections().get(collection_id)
    if record is None or record.features is None:
        raise ValueError(f"feature collection '{collection_id}' is not registered")
    id_property = record.features.id_property
    return {
        str(row[id_property])
        for row in feature_store.read_feature_columns(record, [id_property])
        if row.get(id_property) is not None
    }


def _validated_row(row: object, *, index: int, schema: TableSchema, fixed: Mapping[str, str]) -> dict[str, Any]:
    """Return one row checked against *schema* and normalised to its types, or raise ValueError naming the row.

    *fixed* holds the columns the write fills itself, the dataset and collection ids: a row
    may name them only with the same value.
    """
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index}: expected a mapping, got {type(row).__name__}")
    unknown = sorted(set(row) - {column.name for column in schema.columns})
    if unknown:
        raise ValueError(f"row {index}: unknown columns {unknown}")
    missing = [column.name for column in schema.columns if column.required and column.name not in row]
    if missing:
        raise ValueError(f"row {index}: missing required columns {missing}")
    for name, expected in fixed.items():
        if row.get(name, expected) != expected:
            raise ValueError(f"row {index}: {name} '{row[name]}' does not match '{expected}' this write is for")
    clean: dict[str, Any] = {}
    for column in schema.columns:
        value = _typed(fixed.get(column.name, row.get(column.name)), column.name, column.type, index=index)
        if value is None and not column.nullable:
            raise ValueError(f"row {index}: {column.name} must have a value")
        clean[column.name] = value
    feature = schema.represented_by("feature_id")
    if feature and not clean[feature]:
        raise ValueError(f"row {index}: {feature} must be a non-empty string")
    if coverage_column := schema.represented_by("coverage"):
        coverage = clean[coverage_column]
        if coverage is not None and not 0 <= coverage <= 1:
            raise ValueError(f"row {index}: {coverage_column} must be between 0 and 1, got {coverage}")
        for name in schema.values:
            if coverage == 0 and clean[name] is not None:
                raise ValueError(
                    f"row {index}: {name} {clean[name]} has coverage 0; a feature with no valid data has no value, "
                    "so store None rather than a number"
                )
    return clean


def _typed(value: object, name: str, kind: str, *, index: int) -> Any:
    """*value* as a column of type *kind* stores it, None for a missing one, or ValueError naming the row."""
    if value is None:
        return None
    if kind == "string" and isinstance(value, str):
        return value
    if kind == "double":
        return _number(value, name, index=index)
    if kind in ("int32", "int64") and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "date":
        return _date(value, name, index=index)
    if kind == "timestamp":
        return _utc(value, f"row {index}: {name}")
    if kind == "duration" and isinstance(value, timedelta):
        return value
    article = "an" if kind[0] in "aeiou" else "a"
    raise ValueError(f"row {index}: {name} must be {article} {kind}, got {value!r}")


def start_of_period(day: date, period_type: str | None) -> date:
    """The first day of the *period_type* period that contains *day*; *day* itself for any other.

    Producers label a period by different days: pandas resampling dates a month by its last day
    and a week by its Sunday, while OGS stores date them by their first. Storing the start keeps
    one key per period, so a revision replaces a value rather than standing beside it.
    """
    from open_geo_service.shared.time import dekad_start

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

    from open_geo_service.shared.time import dekad_bounds

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
    """How OGS names the period starting on *day*, as raster datasets name theirs: `2026-01`, `2026-W01`, `2026`.

    For reading only: storage, filters and the key use the date. A dekad is named by its first
    day, as in raster datasets; a period type OGS has no name for keeps the ISO date.
    """
    from open_geo_service.shared.time import datetime_to_period_string

    if period_type == "quarterly":
        return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"
    if period_type in ("daily", "dekadal", "weekly", "monthly", "yearly"):
        return datetime_to_period_string(datetime(day.year, day.month, day.day), period_type)
    return day.isoformat()


def _date(value: object, name: str, *, index: int) -> date:
    """A date, from a date or an ISO date string. A datetime is refused: periods are days."""
    if isinstance(value, datetime):
        raise ValueError(f"row {index}: {name} must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"row {index}: {name} must be a date or an ISO date string, got {value!r}")


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
    existing = latest_record(dataset_id)
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


def _dataset_schema(dataset_id: str, existing: ArtifactRecord | None, schema: str | None) -> TableSchema:
    """The schema a write uses: the dataset's own, which *schema* must not contradict, or *schema* for a new one."""
    current = schema_of(existing).id if existing is not None else None
    if schema is not None and current is not None and schema != current:
        raise ValueError(
            f"table dataset '{dataset_id}' has schema '{current}'; it cannot take rows of schema '{schema}'. "
            "Write them to another table dataset"
        )
    return get_schema(schema or current or DEFAULT_SCHEMA)


def require_table_id(dataset_id: str) -> None:
    """Refuse a dataset id that a raster or feature collection already holds."""
    from open_geo_service.ingestions import services as ingestion_services

    for record in ingestion_services.group_datasets().get(dataset_id, []):
        if record.format != ArtifactFormat.PARQUET:
            raise ValueError(f"dataset id '{dataset_id}' is held by a {record.format} dataset; choose another id")


def _require_id(value: str, kind: str) -> None:
    if not is_segment_safe_id(value):
        raise ValueError(
            f"invalid {kind} id {value!r}; it names a folder and a URL segment, so it must start with a "
            "letter or digit and carry only letters, digits, '.', '_' or '-'"
        )
