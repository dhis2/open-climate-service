"""The feature collection store: where a collection is written, and how it is read back.

One location and one file per collection (`config.get_features_root`), written as GeoParquet
and read back through `read_feature_collection`. `ingestions.services.create_feature_artifact`
turns a written file into a registered collection; nothing here writes a record, and nothing
here discovers a file on disk. That split is deliberate: a record is what makes a collection
exist, so the store directory is not an inbox.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from open_climate_service import config as api_config
from open_climate_service.ingestions.schemas import ArtifactRecord
from open_climate_service.shared import geoparquet
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
"""Feature count above which a read must narrow itself with a bbox or say `max_unqualified_read=None`.

A DHIS2 hierarchy runs to the thousands, and pulling all of it is a real operation — just
never an accidental one. The guard catches the caller that forgot a bbox, not the caller that
means it, which is why `max_unqualified_read=None` is an ordinary argument rather than a
setting an operator has to find. The number is a backstop rather than a measured capacity: it
sits above a country's divisions at every level and below a national facility register.
"""


_collection_locks: dict[str, threading.Lock] = {}
_collection_locks_mutex = threading.Lock()


def collection_lock(dataset_id: str) -> threading.Lock:
    """Return the exclusive lock for one collection, creating it on first use.

    Serializes a refresh end to end — write, replace, register — rather than only the write.
    Those are separate operations on the same collection, so without this two refreshes can
    interleave: both replace the file, and whichever registers last stamps its count, extent and
    CRS onto whatever bytes happen to be on disk.

    In-process, like `ingestions.services._acquire_store_lock` for raster stores. The record
    index itself is guarded across processes by portalocker; a store is guarded within one, which
    is the deployment this serves — one instance owning its data directory.
    """
    with _collection_locks_mutex:
        return _collection_locks.setdefault(dataset_id, threading.Lock())


def validate_collection_id(dataset_id: str) -> str:
    """Return *dataset_id* unchanged, or refuse an id that cannot name a file or a URL segment.

    `dataset_id` reaches here from a template id or a provider, so it is checked rather than
    trusted: a path separator would escape the store directory, and a `?` or `#` would silently
    change what every link to the collection means.
    """
    if (
        not dataset_id
        or dataset_id != dataset_id.strip()
        or any(not char.isprintable() for char in dataset_id)
        or "/" in dataset_id
        or "\\" in dataset_id
        or dataset_id.startswith(".")
    ):
        raise ValueError(
            f"invalid feature collection id {dataset_id!r}; it names files in the feature store, "
            "so it cannot be blank, have surrounding whitespace, contain a path separator or a "
            "non-printing character, or start with a dot"
        )
    return dataset_id


def new_collection_file(dataset_id: str) -> Path:
    """Return the path a *new* version of this collection will be written to.

    Each write lands on its own immutable path rather than replacing one stable file, and the
    record is what says which of them is current. That is what closes the gap between the bytes
    and the record describing them: replacing a stable path makes the new bytes visible to every
    reader the instant they land, while the record still says how many features the *old* file
    held — and a crash in that window leaves the mismatch for good, because both the record and
    the file are individually valid.

    With a version per write, a reader resolves record then path and gets bytes that record
    actually describes, whichever record it read. The cost is that old files outlive their
    records until `prune_superseded_files` removes them, which is the right way round: an
    unreferenced file wastes space, a misdescribed one gives wrong answers.
    """
    return _collection_root(dataset_id) / f"{dataset_id}.{uuid4().hex}.parquet"


def superseded_collection_files(dataset_id: str, *, keep: Path | None) -> list[Path]:
    """Return this collection's stored files other than *keep*, current and not-yet-pruned alike.

    The uuid is matched exactly rather than globbed loosely, so a collection named `a` cannot
    claim the files of one named `a.b` — ids may contain dots, and `a.*.parquet` would match
    `a.b.<uuid>.parquet`. Grace markers (see `prune_superseded_files`) are a different suffix
    and are not returned here — this lists the Parquet files themselves.
    """
    root = _collection_root(dataset_id)
    if not root.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(dataset_id)}\.[0-9a-f]{{32}}\.parquet$")
    kept = keep.resolve() if keep is not None else None
    return [
        candidate
        for candidate in sorted(root.iterdir())
        if pattern.fullmatch(candidate.name) and (kept is None or candidate.resolve() != kept)
    ]


SUPERSEDED_FILE_GRACE_SECONDS = 60
"""How long a superseded file stays reachable at its original path before it may be removed.

A reader resolves a record, then opens the file at `record.path` — two steps, not one, and
nothing serializes them against a refresh running in between. Deleting a file the instant its
record stops being current breaks a reader caught in exactly that gap: it read a valid record a
moment ago, and the path that record named is now gone. This store has no reference count and no
reader registry to know when the last such reader has finished, so the practical alternative is
time: keep a just-superseded file untouched for comfortably longer than any real read takes, and
remove it only once that window has passed.

Sixty seconds is chosen to be far longer than opening one Parquet file ever needs — even a
national hierarchy is a single file, read in a fraction of a second — while still bounding how
long a superseded version lingers. It is not a hard guarantee against a reader that pauses for
minutes with several refreshes in between; nothing short of reference counting is, and this
module does not have one.

Two further limits worth naming, both accepted rather than closed:

The clock behind this is wall-clock (`time.time()`, via a marker file's mtime), not monotonic —
deliberately, since the "first seen as superseded" moment has to survive a process restart, and
a monotonic reading cannot. A backward clock step only delays deletion, which is safe; a large
forward step (an NTP correction, a paused-and-resumed VM or container, a manual clock fix)
landing between two prune calls can make a marker look older than it is and shorten the
practical window below `SUPERSEDED_FILE_GRACE_SECONDS`. Closing this fully would need a
process-local monotonic checkpoint alongside the on-disk marker; this module does not have one.

The bound only holds across a dataset's *own* subsequent refreshes, because the only caller of
`prune_superseded_files` is `refresh_feature_collection` — nothing sweeps the store
independently. A collection refreshed once more and then never again (its template retired, its
provider discontinued) has its superseded file marked exactly once and never revisited: the file
and its marker persist indefinitely, one of each per abandoned collection, capped in size but
not in time. An independent sweep (a scheduled job, a startup pass over every registered
collection) would close this; none exists yet.
"""

_SUPERSEDED_MARKER_SUFFIX = ".superseded"


def prune_superseded_files(dataset_id: str, *, keep: Path | None) -> None:
    """Remove this collection's older files, but only once each has aged past the grace period.

    `keep` is the file the current record (after any rollback) points at, or None when this
    dataset has no current record at all -- a first-ever registration whose publish step failed
    leaves nothing to keep, and every file `write_feature_collection` produced for it is stale.

    Two-phase, not immediate. The first time a prune call finds a file no longer named by any
    record, it leaves the file exactly as it is and only drops a marker beside it recording that
    moment — so a reader that resolved the file's record just before this refresh still finds
    the file untouched. Only a *later* prune call, once that marker is older than
    `SUPERSEDED_FILE_GRACE_SECONDS`, actually deletes the file. A superseded file therefore
    survives at least one full refresh cycle after it stops being current, and normally the
    grace period on top of that.

    The marker's own mtime is the clock, stamped fresh the moment a file is first seen as
    superseded — not the file's mtime, which records when it was *written* and would make a
    long-lived file that was *just* superseded look old enough to delete immediately.

    Called after either outcome of a refresh, not only success: once the new record is durable
    and superseded a prior one, or once a failed refresh has rolled its record back and `keep`
    names whatever is current again. Either way, a file this drops the marker for is one no
    *current* record names, never the file `keep` itself. A failure to delete, or to write a
    marker, is logged rather than raised: the record state is already correct by the time this
    runs, and a lingering file or marker is not a reason to fail the operation that triggered it.
    """
    now = time.time()
    for stale in superseded_collection_files(dataset_id, keep=keep):
        marker = stale.with_name(stale.name + _SUPERSEDED_MARKER_SUFFIX)
        try:
            marker_age = now - marker.stat().st_mtime
        except FileNotFoundError:
            # First time this file has been seen as superseded: mark it and leave it alone.
            # It gets at least one more full refresh cycle before deletion is even considered.
            try:
                marker.touch()
            except OSError:
                logger.warning("Could not mark '%s' as superseded", stale, exc_info=True)
            continue
        if marker_age < SUPERSEDED_FILE_GRACE_SECONDS:
            continue
        try:
            stale.unlink()
            marker.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove the superseded feature collection file '%s'", stale, exc_info=True)
    _prune_orphaned_markers(dataset_id)


def _prune_orphaned_markers(dataset_id: str) -> None:
    """Remove a marker whose Parquet file is already gone.

    Reachable only if a previous prune deleted the file but then failed to delete its marker —
    the two unlinks above are not atomic with each other. Harmless to leave (nothing reads a
    marker for a file that is not also returned by `superseded_collection_files`), but there is
    no reason to let them accumulate forever either.
    """
    root = _collection_root(dataset_id)
    if not root.is_dir():
        return
    pattern = re.compile(rf"^{re.escape(dataset_id)}\.[0-9a-f]{{32}}\.parquet{re.escape(_SUPERSEDED_MARKER_SUFFIX)}$")
    for marker in root.iterdir():
        if not pattern.fullmatch(marker.name):
            continue
        if not marker.with_name(marker.name.removesuffix(_SUPERSEDED_MARKER_SUFFIX)).exists():
            marker.unlink(missing_ok=True)


def _collection_root(dataset_id: str) -> Path:
    """Return the store directory, having checked the id that will name files inside it."""
    validate_collection_id(dataset_id)
    return api_config.get_features_root()


def validate_features_for_write(*, dataset_id: str, features: object, id_property: str) -> gpd.GeoDataFrame:
    """Return a frame that can be stored and found by bbox reads, before touching any file.

    Three rules, one place, so `write_feature_collection` refuses on the same terms as the
    refresh path that calls this ahead of backing the previous collection up:

    * identity — each feature's id is present and unique, from `validate_feature_ids`;
    * geometry — missing and empty shapes are invisible to bbox reads, so check the converted
      frame rather than enumerating GeoJSON spellings for them;
    * names — `bbox` conflicts with the covering column, while `geometry` conflicts with the
      geometry column; both are refused with a rename the provider can act on.
    """
    import geopandas as gpd

    from open_climate_service.shared.features import validate_feature_ids

    members = _members(features)
    validate_feature_ids(features, id_property=id_property)
    for index, feature in enumerate(members):
        properties = feature.get("properties")
        if isinstance(properties, dict):
            for reserved in ("bbox", "geometry"):
                if reserved in properties:
                    raise ValueError(
                        f"feature collection '{dataset_id}' has a property named '{reserved}' at index {index}; "
                        "rename the property before writing"
                    )
    try:
        frame = gpd.GeoDataFrame.from_features(members, crs=WGS84)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"feature collection '{dataset_id}' has malformed GeoJSON geometry: {exc}") from exc
    if frame.empty:
        raise ValueError(f"feature collection '{dataset_id}' has no features to write")
    invalid = frame.geometry.isna() | frame.geometry.is_empty
    for index, bad in enumerate(invalid):
        if bad:
            raise ValueError(
                f"feature collection '{dataset_id}' has a feature with no geometry at index {index}; "
                "missing or empty geometry cannot be found by any bbox read"
            )
    if id_property not in frame.columns:
        raise ValueError(
            f"feature collection '{dataset_id}' declares id_property '{id_property}', but no such "
            f"column survived conversion; got {sorted(frame.columns)}"
        )
    return frame


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
    it called the geometry column. Nothing is registered here, and nothing existing is touched —
    each write lands on its own path (see `new_collection_file`), so a caller that writes and
    then fails to register has left an unreferenced file the listing ignores, with the previous
    collection still in place and still correctly described by its record.

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
    frame = validate_features_for_write(dataset_id=dataset_id, features=features, id_property=id_property)
    target = canonical_crs_code(store_crs)
    if target != WGS84:
        frame = frame.to_crs(target)

    path = new_collection_file(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved into place, so a reader never sees a partial file.
    # The target is new on every write, so this replaces nothing a record points at: an
    # interrupted write leaves an unreferenced file, never a truncated collection.
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
    max_unqualified_read: int | None = UNQUALIFIED_READ_LIMIT,
) -> gpd.GeoDataFrame:
    """Read a registered collection, optionally windowed to *bbox*.

    `bbox` is interpreted in `bbox_crs` and reprojected into the collection's own CRS before it
    reaches the file, so a WGS 84 window works against a collection stored in a projected CRS.
    Reprojecting the window rather than the rows is what keeps the read a pushdown: the rows
    stay in the CRS they were written in, and only four numbers cross CRSs.

    `max_unqualified_read` guards an *unqualified* read — one with no bbox — against pulling a
    whole hierarchy by accident. A windowed read is already narrowed by the window, so the guard
    does not apply to it; `max_unqualified_read=None` turns it off for a caller that means to
    read everything.
    """
    import geopandas as gpd

    detail = record.features
    if detail is None:
        raise ValueError(f"artifact '{record.artifact_id}' is not a feature collection, so it has no features to read")
    path = _stored_path(record)
    if bbox is None and max_unqualified_read is not None and detail.feature_count > max_unqualified_read:
        raise ValueError(
            f"feature collection '{record.dataset_id}' holds {detail.feature_count} features, above the "
            f"unqualified-read limit of {max_unqualified_read}; pass a bbox to narrow the read, or "
            "max_unqualified_read=None to read all of it deliberately"
        )
    window = transform_bbox(bbox, source=bbox_crs, target=detail.crs) if bbox is not None else None
    frame: gpd.GeoDataFrame = gpd.read_parquet(path, bbox=window)
    return frame


def stored_geometry_types(record: ArtifactRecord) -> list[str]:
    """Return the geometry types a stored collection declares, read from its file footer."""
    return geoparquet.stored_geometry_types(
        _stored_path(record), column=record.features.primary_geometry if record.features else None
    )


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
