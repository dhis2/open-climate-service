"""Table schemas: the columns and key a table dataset declares, and what columns represent (CLIM-1971).

A table has no fixed schema. A schema file in a `tables/schemas/` folder declares one: its
columns, the key columns that say which stored row a write replaces, and what a column
represents, where the store and its readers need to know: the feature, the period, a value and
so on. Files are read from core (`open_geo_service/plugins/tables/schemas/`), then installed plugin
packages, then the instance `plugins_dir`; a later schema with the same id replaces an earlier one.

The built-in `aggregation` schema, values aggregated per feature and period, is what the
pipeline destination writes and what a table that names no schema gets. Core holds no schema
of its own in code: its rules apply to what a column represents, whatever the column is called.
"""

from __future__ import annotations

import functools
import importlib.resources
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from open_geo_service import config as api_config
from open_geo_service.shared.urls import is_segment_safe_id

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA = "aggregation"
"""The schema of a table dataset that names none."""

WRITTEN_AT = "written_at"
"""The column the store adds to every table: when the row's values last changed."""

WRITTEN_AT_DESCRIPTION = "When the value last changed."

TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "double": pa.float64(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "date": pa.date32(),
    "timestamp": pa.timestamp("us", tz="UTC"),
    "duration": pa.duration("s"),
}
"""The column types a schema can declare, and how each is stored."""

_REPRESENTED_ONCE = {
    "dataset": ("string",),
    "collection": ("string",),
    "collection_version": ("string",),
    "feature": ("string",),
    "feature_name": ("string",),
    "period": ("date",),
    "unit": ("string",),
    "coverage": ("double",),
}
"""What at most one column represents, with the types that column may have. Any number represent a value."""

_NUMBER_TYPES = ("double", "int32", "int64")


@dataclass(frozen=True)
class Column:
    """One column a schema declares."""

    name: str
    type: str
    description: str = ""
    unit: str | None = None
    required: bool = False
    """A row must name the column."""
    nullable: bool = True
    """The column may hold no value."""
    represents: str | None = None
    """What the column holds, for the rules and readers that need it: `feature`, `period`, `value`, …"""


@dataclass(frozen=True)
class TableSchema:
    """A table's columns, key and what its columns represent, as one schema file declares them."""

    id: str
    name: str
    description: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]
    represented: dict[str, str]
    """What a column represents, other than a value, and that column."""
    values: tuple[str, ...]
    """The columns that represent a value."""

    @property
    def names(self) -> list[str]:
        """The stored columns, in order: the declared ones, then `written_at`."""
        return [*(column.name for column in self.columns), WRITTEN_AT]

    @property
    def arrow(self) -> pa.Schema:
        fields = [pa.field(column.name, TYPES[column.type], nullable=column.nullable) for column in self.columns]
        return pa.schema([*fields, pa.field(WRITTEN_AT, TYPES["timestamp"], nullable=False)])

    def column(self, name: str) -> Column | None:
        return next((column for column in self.columns if column.name == name), None)

    def represented_by(self, meaning: str) -> str | None:
        """The column that represents *meaning*, such as `period`, or None when no column does."""
        return self.represented.get(meaning)

    def holds(self, period_type: str | None) -> str:
        """What a row holds, from its feature and period columns: 'values per feature and monthly period'."""
        period = f"{period_type} period" if period_type else "period"
        if self.represented_by("feature") and self.represented_by("period"):
            return f"values per feature and {period}"
        if self.represented_by("feature"):
            return "values per feature"
        if self.represented_by("period"):
            return f"values per {period}"
        return "rows of values"


def get_schema(schema_id: str) -> TableSchema:
    """The schema with *schema_id*, or ValueError naming the ones there are."""
    schemas = _schemas_cached(*_cache_key())
    if schema_id not in schemas:
        raise ValueError(f"no table schema '{schema_id}'; the schemas are {sorted(schemas)}")
    return schemas[schema_id]


def reset_schema_caches() -> None:
    """Forget the parsed schemas, for a test that changes `plugins_dir` or installs a plugin."""
    _builtin.cache_clear()
    _schemas_cached.cache_clear()


def _cache_key() -> tuple[str | None, str | None]:
    config = api_config.get_config() or {}
    path = api_config.get_config_path()
    plugins_dir = config.get("plugins_dir")
    return (str(path) if path else None, str(plugins_dir) if plugins_dir is not None else None)


@functools.lru_cache(maxsize=8)
def _schemas_cached(config_path: str | None, plugins_dir: str | None) -> dict[str, TableSchema]:
    from open_geo_service.plugin_discovery import iter_plugin_subdirs

    merged = {schema.id: schema for schema in _builtin()}
    for plugin_name, _package, tables in iter_plugin_subdirs("tables"):
        folder = tables / "schemas"
        if folder.is_dir():
            for schema in _parse_files(folder.iterdir(), source=f"plugin '{plugin_name}'"):
                merged[schema.id] = schema
    if plugins_dir:
        base = Path(config_path).parent if config_path else Path()
        folder = (base / plugins_dir).resolve() / "tables" / "schemas"
        if folder.is_dir():
            for schema in _parse_files(sorted(folder.iterdir()), source=str(folder)):
                merged[schema.id] = schema
    return merged


@functools.lru_cache(maxsize=1)
def _builtin() -> tuple[TableSchema, ...]:
    folder = importlib.resources.files("open_geo_service") / "plugins" / "tables" / "schemas"
    return tuple(_parse_files(folder.iterdir(), source="core"))


def _parse_files(files: Iterable[Any], *, source: str) -> list[TableSchema]:
    schemas: list[TableSchema] = []
    for file in files:
        if not file.name.endswith((".yaml", ".yml")):
            continue
        declared = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(declared, list):
            raise ValueError(f"{source} ({file.name}) must contain a list of table schemas")
        schemas += [parse_schema(item, source=f"{source} ({file.name})") for item in declared]
    return schemas


def parse_schema(declared: object, *, source: str) -> TableSchema:
    """A schema from its declaration, or ValueError naming *source* and what is wrong."""
    if not isinstance(declared, dict):
        raise ValueError(f"{source} contains a table schema that is not a mapping")
    schema_id = declared.get("id")
    if not isinstance(schema_id, str) or not is_segment_safe_id(schema_id):
        raise ValueError(f"{source} contains a table schema with a missing or invalid id {schema_id!r}")
    where = f"table schema '{schema_id}' in {source}"
    columns = tuple(_column(item, where) for item in _list(declared, "columns", where))
    names = [column.name for column in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"{where} declares a column more than once")
    if WRITTEN_AT in names:
        raise ValueError(f"{where} declares '{WRITTEN_AT}', which the store adds to every table")
    key = tuple(str(name) for name in _list(declared, "key", where))
    if unknown := sorted(set(key) - set(names)):
        raise ValueError(f"{where} has key columns it does not declare: {unknown}")

    if unknown := sorted(set(declared) - {"id", "name", "description", "columns", "key"}):
        raise ValueError(f"{where} has unknown fields {unknown}")
    represented: dict[str, str] = {}
    values: list[str] = []
    for column in columns:
        meaning = column.represents
        if meaning is None:
            continue
        if meaning == "value":
            if column.type not in _NUMBER_TYPES:
                raise ValueError(f"{where}: column '{column.name}' represents a value, so it must be a number")
            values.append(column.name)
            continue
        if meaning not in _REPRESENTED_ONCE:
            known = sorted([*_REPRESENTED_ONCE, "value"])
            raise ValueError(f"{where}: column '{column.name}' represents unknown '{meaning}'; known are {known}")
        if column.type not in _REPRESENTED_ONCE[meaning]:
            kinds = " or ".join(_REPRESENTED_ONCE[meaning])
            raise ValueError(f"{where}: column '{column.name}' represents {meaning}, so it must be a {kinds}")
        if meaning in represented:
            raise ValueError(f"{where}: columns '{represented[meaning]}' and '{column.name}' both represent {meaning}")
        represented[meaning] = column.name
    # A feature is checked against its collection, and a collection version is one of a collection's.
    if ("feature" in represented) != ("collection" in represented):
        raise ValueError(f"{where}: a column representing feature needs one representing collection, and back")
    if "collection_version" in represented and "collection" not in represented:
        raise ValueError(f"{where}: a column representing collection_version needs one representing collection")
    if ("coverage" in represented or "unit" in represented) and not values:
        raise ValueError(f"{where}: coverage and unit describe values; mark the value columns 'represents: value'")

    name = declared.get("name", schema_id)
    description = declared.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError(f"{where} has a 'name' or 'description' that is not text")
    return TableSchema(schema_id, name, description, columns, key, represented, tuple(values))


def _list(declared: dict[str, Any], field: str, where: str) -> list[Any]:
    items = declared.get(field)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{where} must list its '{field}'")
    return items


def _column(declared: object, where: str) -> Column:
    if not isinstance(declared, dict):
        raise ValueError(f"{where} has a column that is not a mapping")
    name, kind = declared.get("name"), declared.get("type")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where} has a column without a name")
    if kind not in TYPES:
        raise ValueError(f"{where}: column '{name}' has type {kind!r}; types are {sorted(TYPES)}")
    unknown = sorted(set(declared) - {"name", "type", "description", "unit", "required", "nullable", "represents"})
    if unknown:
        raise ValueError(f"{where}: column '{name}' has unknown fields {unknown}")
    return Column(
        name=name,
        type=kind,
        description=str(declared.get("description", "")),
        unit=declared.get("unit"),
        required=bool(declared.get("required", False)),
        nullable=bool(declared.get("nullable", True)),
        represents=str(declared["represents"]) if declared.get("represents") is not None else None,
    )
