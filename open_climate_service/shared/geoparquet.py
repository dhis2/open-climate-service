"""Reading what a GeoParquet file says about itself, from its footer.

The GeoParquet spec puts a `geo` JSON document in the parquet key-value metadata, naming the
primary geometry column and, per column, its CRS and geometry types. Everything here reads that
document and nothing else: no rows are decoded, so the cost does not grow with the collection.

Lives in `shared` rather than beside the store because two layers need it and neither should
import the other — `features.store` reports what it holds, and `ingestions` checks a declared
CRS against what was actually written before it records one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from open_climate_service.shared.crs import canonical_crs_code

logger = logging.getLogger(__name__)

DEFAULT_GEOMETRY_COLUMN = "geometry"


def read_geo_metadata(path: Path | str) -> dict[str, Any] | None:
    """Return a GeoParquet file's `geo` metadata document, or None when it has none.

    None rather than an exception for a file that is not GeoParquet, or is unreadable: every
    caller here is describing or checking a collection, and each has a more useful thing to say
    about the absence than this does.
    """
    import pyarrow.parquet as pq

    try:
        metadata = pq.read_schema(str(path)).metadata or {}
        raw = metadata.get(b"geo")
        if raw is None:
            return None
        document = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read GeoParquet metadata from '%s': %s", path, exc)
        return None
    return document if isinstance(document, dict) else None


def primary_geometry_column(path: Path | str, *, default: str = DEFAULT_GEOMETRY_COLUMN) -> str:
    """Return the name the file gives its primary geometry column."""
    document = read_geo_metadata(path) or {}
    column = document.get("primary_column")
    return column if isinstance(column, str) and column else default


def stored_crs(path: Path | str, *, column: str | None = None) -> str | None:
    """Return the CRS a GeoParquet file records for its geometry column, canonicalized.

    GeoParquet writes the CRS as PROJJSON. Reading it back through pyproj and asking for an
    authority code is what makes it comparable to a declared `EPSG:xxxx`: the same CRS has many
    spellings, and comparing PROJJSON documents textually would report a mismatch between two
    descriptions of the same thing.

    An omitted or null CRS in valid GeoParquet metadata means OGC:CRS84. None means the
    metadata is missing or cannot establish a valid geometry CRS.
    """
    document = read_geo_metadata(path)
    if document is None:
        return None
    columns = document.get("columns")
    if not isinstance(columns, dict):
        return None
    name = column or document.get("primary_column")
    entry = columns.get(name)
    if not isinstance(entry, dict):
        return None
    declared = entry.get("crs")
    if declared is None:
        # Both omission and explicit null mean OGC:CRS84.
        return "EPSG:4326"
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs = CRS.from_user_input(declared)
    except (CRSError, TypeError, ValueError) as exc:
        logger.warning("Could not interpret the CRS stored in '%s': %s", path, exc)
        return None
    code = crs.to_string() if crs.to_epsg() is None else f"EPSG:{crs.to_epsg()}"
    return canonical_crs_code(code)


def stored_geometry_types(path: Path | str, *, column: str | None = None) -> list[str]:
    """Return the geometry types a file declares for its geometry column.

    An empty list means the file declares none, which a pre-1.0 writer may do. Reported as
    "not stated" rather than guessed at by scanning geometries.
    """
    document = read_geo_metadata(path)
    if document is None:
        return []
    columns = document.get("columns")
    if not isinstance(columns, dict):
        return []
    entry = columns.get(column or primary_geometry_column(path))
    declared = entry.get("geometry_types") if isinstance(entry, dict) else None
    if not isinstance(declared, list):
        return []
    return sorted(str(value) for value in declared)
