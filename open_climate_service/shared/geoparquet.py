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

PARQUET_MEDIA_TYPE = "application/x-parquet"
"""The one media type OCS advertises for Parquet, wherever it advertises one.

Two spellings were in use — `application/vnd.apache.parquet` on the openEO job-result path and
pystac's own `application/x-parquet` — and neither is registered with IANA, so this is a choice
rather than a correction. `x-parquet` wins because it is what the STAC ecosystem reads: pystac
ships it as a constant, and a feature collection's data asset is the thing a STAC client has to
recognise. Kept here, beside the reader, so the catalogue and the job-result path cannot drift.
"""


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

    An *omitted* crs means OGC:CRS84, which GeoParquet defines as the default. An *explicit*
    null means the CRS is undefined, which is a different fact and comes back as None — along
    with metadata that is missing or unreadable.
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
    if "crs" not in entry:
        # GeoParquet defines an *omitted* crs as OGC:CRS84, so this is a statement, not a gap.
        return "EPSG:4326"
    declared = entry["crs"]
    if declared is None:
        # An *explicit* null is the opposite statement: the CRS is undefined. Reading it as
        # WGS 84 would let a file of unknown coordinates register as degrees and then be
        # windowed as degrees — wrong extents, wrong bbox reads, and no error anywhere. None
        # sends it down the "the file does not say" path, where the caller's declaration stands
        # on its own rather than being confirmed by something the file never claimed.
        return None
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs = CRS.from_user_input(declared)
    except (CRSError, TypeError, ValueError) as exc:
        logger.warning("Could not interpret the CRS stored in '%s': %s", path, exc)
        return None
    code = crs.to_string() if crs.to_epsg() is None else f"EPSG:{crs.to_epsg()}"
    return canonical_crs_code(code)


def covering_bbox_column(path: Path | str, *, column: str | None = None) -> str | None:
    """Return the name of the GeoParquet covering-bbox column, when the file has one.

    GeoParquet 1.1 lets a writer add a struct column of per-row bounds so a reader can skip row
    groups, and names it under `covering.bbox`. It is a real column in the file and an
    implementation detail of the pushdown, so anything describing the collection's *data* needs
    to be able to tell it apart from a column a provider actually supplied.

    Read from the metadata rather than matched by name: "bbox" is a name a provider could
    legitimately use for its own column, and excluding that by string would hide real data.
    """
    document = read_geo_metadata(path)
    if document is None:
        return None
    columns = document.get("columns")
    if not isinstance(columns, dict):
        return None
    entry = columns.get(column or primary_geometry_column(path))
    covering = entry.get("covering") if isinstance(entry, dict) else None
    bbox = covering.get("bbox") if isinstance(covering, dict) else None
    if not isinstance(bbox, dict):
        return None
    # Each corner is a path into the file, e.g. ["bbox", "xmin"]; the first element names the
    # struct column they all share.
    for corner in ("xmin", "ymin", "xmax", "ymax"):
        reference = bbox.get(corner)
        if isinstance(reference, list) and reference and isinstance(reference[0], str):
            return reference[0]
    return None


def table_columns(path: Path | str, *, geometry_column: str | None = None) -> list[dict[str, str]]:
    """Describe a stored collection's columns, for the STAC table extension's `table:columns`.

    Read from the parquet footer, so the cost is the schema rather than the rows — a
    country-scale collection is described without decoding one feature.

    The covering-bbox column is left out. It is bookkeeping this service wrote to make windowed
    reads cheap, not something a provider supplied, and advertising it would invite a client to
    treat it as an attribute of the features.

    Types are reported as Arrow spells them (`string`, `int64`, `double`, `binary`). The table
    extension leaves `type` free-form, and the file's own vocabulary is the one a caller reading
    that file will meet.
    """
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(str(path))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the parquet schema of '%s': %s", path, exc)
        return []
    excluded = {name for name in (covering_bbox_column(path, column=geometry_column),) if name}
    return [{"name": field.name, "type": str(field.type)} for field in schema if field.name not in excluded]


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
