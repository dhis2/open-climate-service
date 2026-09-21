"""The feature collection store: where a collection is written, and how it is read back.

One location and one file per collection (`config.get_features_root`), written as GeoParquet
and read back through `read_feature_collection`. `ingestions.services.create_feature_artifact`
turns a written file into a registered collection; nothing here writes a record, and nothing
here discovers a file on disk. That split is deliberate: a record is what makes a collection
exist, so the store directory is not an inbox.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from open_climate_service import config as api_config
from open_climate_service.ingestions.schemas import ArtifactRecord
from open_climate_service.shared.crs import canonical_crs_code, transform_bbox

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
"""The CRS a bbox is expressed in unless a caller says otherwise.

GeoJSON is WGS 84 by RFC 7946, every OGC API bbox defaults to it, and the configured instance
extent is stored in it — so it is what a caller means when it passes four numbers and says
nothing else.
"""

UNQUALIFIED_READ_LIMIT = 5000
"""Feature count above which a read must narrow itself with a bbox or say `limit=None`.

A DHIS2 hierarchy runs to the thousands, and pulling all of it is a real operation — just
never an accidental one. The guard catches the caller that forgot a bbox, not the caller that
means it, which is why `limit=None` is an ordinary argument rather than a setting an operator
has to find. The number is a backstop rather than a measured capacity: it sits above a
country's divisions at every level and below a national facility register.
"""


def feature_store_path(dataset_id: str) -> Path:
    """Return the path this collection's GeoParquet occupies, whether or not it exists yet.

    Derived from the dataset id rather than stored, so the writer and any future reader agree
    on the location without consulting a record. `dataset_id` reaches here from a template id
    or a provider, so it is checked rather than trusted: a name that escapes the store
    directory would let a caller write through it.
    """
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id.startswith("."):
        raise ValueError(
            f"invalid feature collection id {dataset_id!r}; it names one file in the feature store, "
            "so it cannot be empty, contain a path separator, or start with a dot"
        )
    return api_config.get_features_root() / f"{dataset_id}.parquet"


def write_feature_collection(
    *,
    dataset_id: str,
    features: object,
    id_property: str,
    store_crs: str = WGS84,
) -> tuple[Path, int, str]:
    """Write one GeoJSON FeatureCollection to the store, returning (path, count, geometry column).

    Returns exactly what `create_feature_artifact` needs to build a record, because those three
    facts are the store's to report: it is the thing that knows how many rows it wrote and what
    it called the geometry column. Nothing is registered here — a caller that writes and then
    fails to register has left a file the listing ignores, which is the intended failure rather
    than a half-registered collection.

    Identity is validated before anything is written. A duplicate identifier is not a dropped
    feature: two features map onto one org unit, DHIS2 keeps whichever value arrives last, and
    the result is silently wrong — so the write does not happen at all.

    `write_covering_bbox` is what makes `read_feature_collection`'s bbox argument a pushdown
    rather than a filter over everything: it writes the GeoParquet 1.1 bbox covering column, so
    a windowed read skips whole row groups instead of decoding them.

    The input is GeoJSON, so its coordinates are WGS 84 by RFC 7946 whatever `store_crs` says.
    A different `store_crs` therefore *reprojects* rather than relabels: tagging WGS 84
    coordinates with a projected CRS would leave every later read windowing the wrong numbers,
    and nothing downstream could detect it.
    """
    import geopandas as gpd

    from open_climate_service.shared.features import validate_feature_ids

    validate_feature_ids(features, id_property=id_property)
    target = canonical_crs_code(store_crs)
    frame = gpd.GeoDataFrame.from_features(_members(features), crs=WGS84)
    if frame.empty:
        raise ValueError(f"feature collection '{dataset_id}' has no features to write")
    if id_property not in frame.columns:
        # Checked against the frame, not only against the input. `validate_feature_ids` reads
        # the GeoJSON while `from_features` decides what becomes a column, and a file stored
        # without its identifier column is the one failure that raises nothing later: the record
        # would name an `id_property` no read could find.
        raise ValueError(
            f"feature collection '{dataset_id}' declares id_property '{id_property}', but no such "
            f"column survived conversion; got {sorted(frame.columns)}"
        )
    if target != WGS84:
        frame = frame.to_crs(target)

    path = feature_store_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved into place, so a reader never sees a partial file and
    # a failed write leaves the previous collection intact. The staging name is unique per write
    # rather than per collection: two refreshes of one collection run under the same record lock
    # only once they reach registration, so until the replace they are genuinely concurrent —
    # sharing one name, each would write into the other's file and the loser's cleanup would
    # delete the winner's. Same directory, so the replace stays atomic.
    staging = path.with_name(f"{path.name}.{uuid4().hex}.writing")
    try:
        frame.to_parquet(staging, write_covering_bbox=True, schema_version="1.1.0")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    return path, len(frame), str(frame.geometry.name)


def read_feature_collection(
    record: ArtifactRecord,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: str = WGS84,
    limit: int | None = UNQUALIFIED_READ_LIMIT,
) -> gpd.GeoDataFrame:
    """Read a registered collection, optionally windowed to *bbox*.

    `bbox` is interpreted in `bbox_crs` and reprojected into the collection's own CRS before it
    reaches the file, so a WGS 84 window works against a collection stored in a projected CRS.
    Reprojecting the window rather than the rows is what keeps the read a pushdown: the rows
    stay in the CRS they were written in, and only four numbers cross CRSs.

    `limit` guards an *unqualified* read — one with no bbox — against pulling a whole hierarchy
    by accident. A windowed read is already narrowed by the window, so the guard does not apply
    to it; `limit=None` turns it off for a caller that means to read everything.
    """
    import geopandas as gpd

    detail = record.features
    if detail is None:
        raise ValueError(f"artifact '{record.artifact_id}' is not a feature collection, so it has no features to read")
    path = _stored_path(record)
    if bbox is None and limit is not None and detail.feature_count > limit:
        raise ValueError(
            f"feature collection '{record.dataset_id}' holds {detail.feature_count} features, above the "
            f"unqualified-read limit of {limit}; pass a bbox to narrow the read, or limit=None to "
            "read all of it deliberately"
        )
    window = transform_bbox(bbox, source=bbox_crs, target=detail.crs) if bbox is not None else None
    frame: gpd.GeoDataFrame = gpd.read_parquet(path, bbox=window)
    return frame


def stored_geometry_types(record: ArtifactRecord) -> list[str]:
    """Return the geometry types a stored collection declares, read from its file footer.

    From the GeoParquet `geo` metadata rather than from the rows: the spec has the writer record
    `geometry_types` per geometry column, so this is a footer read whose cost does not grow with
    the collection. An empty list means the file declares none, which a pre-1.0 writer may do —
    reported as "unknown" rather than guessed at by scanning geometries.
    """
    import pyarrow.parquet as pq

    try:
        metadata = pq.read_schema(_stored_path(record)).metadata or {}
        geo = json.loads(metadata[b"geo"].decode("utf-8"))
        column = geo["columns"][record.features.primary_geometry if record.features else "geometry"]
        declared = column.get("geometry_types", [])
    except (OSError, KeyError, ValueError, TypeError):
        logger.warning("Could not read geometry types for '%s' from its GeoParquet metadata", record.dataset_id)
        return []
    return sorted(str(value) for value in declared) if isinstance(declared, list) else []


def _stored_path(record: ArtifactRecord) -> Path:
    """Return the file a record points at, or say plainly that it is gone."""
    raw = record.path or (record.asset_paths[0] if record.asset_paths else None)
    if raw is None:
        raise ValueError(f"feature collection '{record.dataset_id}' has no stored path on its record")
    path = Path(raw)
    if not path.is_file():
        raise ValueError(
            f"feature collection '{record.dataset_id}' is registered at {path}, but no file is there; "
            "the record and the store disagree"
        )
    return path


def _members(features: object) -> list[Any]:
    """Return the member features of an already-validated FeatureCollection.

    A non-string Sequence, matching `validate_feature_ids` and the ingestion path: a tuple is
    what a provider that built its features with a comprehension hands over, and the three
    checks are on one path, so they have to agree about what a collection is.
    """
    if isinstance(features, Mapping) and features.get("type") == "FeatureCollection":
        members = features.get("features")
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
            return list(members)
    raise ValueError("expected a GeoJSON FeatureCollection to write")
