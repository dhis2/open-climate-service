"""Reading the registered feature collections, for `GET /features`."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.features import store
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord, PublicationStatus
from open_climate_service.publications.services import managed_dataset_id_for
from open_climate_service.shared.licences import parse_licence


def refresh_feature_collection(
    *,
    template: dict[str, Any],
    features: Mapping[str, Any],
    store_crs: str = store.WGS84,
    bbox: Sequence[float] | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """Write a collection and register it as one operation, or leave the previous one in place.

    The single door a provider run comes through (CLIM-926), and the reason it exists is that
    writing and registering are two operations on one collection. Apart, they fail badly:

    * a refresh that writes and then fails to register leaves the *old* record describing the
      *new* file, so `feature_count`, `extent` and `crs` all describe bytes that are gone — and
      nothing raises, because the record is still valid and the file is still there;
    * two refreshes interleave, and whichever registers last stamps its numbers onto whichever
      file was replaced last.

    So the whole sequence runs under one per-collection lock, and the previous file is kept
    aside until the record is durable. On failure it is put back with the same atomic replace
    that installed the new one, which means a reader at any instant sees one complete file:
    the old collection or the new one, never a mixture and never a gap.
    """
    dataset_id = str(template.get("id", ""))
    if not dataset_id.strip():
        raise ValueError("feature collection template must declare a non-empty 'id'")
    store.feature_store_path(dataset_id)
    id_property = str(template.get("id_property", "")).strip()
    if not id_property:
        raise ValueError(f"feature collection template '{dataset_id}' must declare a non-empty 'id_property'")

    with store.collection_lock(dataset_id):
        path = store.feature_store_path(dataset_id)
        prior_records = [
            record
            for record in ingestion_services._load_records()
            if record.dataset_id == dataset_id and record.format == ArtifactFormat.GEOPARQUET
        ]
        # Refuse before touching anything: the checks that can fail a write run before the
        # previous collection is copied aside, so a doomed refresh leaves file and record alone.
        store.validate_features_for_write(dataset_id=dataset_id, features=features, id_property=id_property)
        previous = _keep_previous(path)
        try:
            written, _count, geometry = store.write_feature_collection(
                dataset_id=dataset_id,
                features=features,
                id_property=id_property,
                store_crs=store_crs,
            )
            return ingestion_services.create_feature_artifact(
                template=template,
                features=features,
                store_path=written,
                crs=store_crs,
                primary_geometry=geometry,
                bbox=bbox,
                publish=publish,
            )
        except BaseException:
            _restore_previous(path, previous)
            _restore_records(dataset_id, prior_records)
            raise
        finally:
            if previous is not None:
                previous.unlink(missing_ok=True)


def _keep_previous(path: Path) -> Path | None:
    """Copy the current collection aside so a failed refresh can be undone, or None if new.

    Copied rather than renamed: a rename would leave `path` absent for as long as the write
    takes, and a concurrent read would see a collection that briefly does not exist.
    """
    if not path.is_file():
        return None
    kept = path.with_name(f"{path.name}.{uuid4().hex}.previous")
    shutil.copy2(path, kept)
    return kept


def _restore_previous(path: Path, previous: Path | None) -> None:
    """Put the previous collection back, or remove a first write that was never registered."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    # The backup is already a complete file on the same filesystem.
    previous.replace(path)


def _restore_records(dataset_id: str, prior_records: list[ArtifactRecord]) -> None:
    """Undo a registration that succeeded before a later publication step failed."""

    def restore(records: list[ArtifactRecord]) -> None:
        records[:] = [
            record
            for record in records
            if record.dataset_id != dataset_id or record.format != ArtifactFormat.GEOPARQUET
        ]
        records.extend(prior_records)

    ingestion_services._mutate_records(restore)


def registered_collections() -> dict[str, ArtifactRecord]:
    """Return the latest record for each registered feature collection, by collection id.

    Reads records, never the filesystem. A GeoParquet file sitting in the store directory that
    nothing registered is not a collection and does not appear here — there is no discovery step
    and so no state in which disk and index disagree. `_materialized_records` upstream already
    drops a record whose file has gone, so what is listed here is registered *and* present.

    Publication is not a filter. `/features` is the operator-facing inventory of what this
    instance holds, and an unpublished collection is still held; publication decides what the
    catalogues advertise, which is a different question asked in `stac_eligible_artifacts_by_dataset`.
    """
    collections: dict[str, ArtifactRecord] = {}
    for record in ingestion_services.list_artifacts().items:
        if record.format != ArtifactFormat.GEOPARQUET or record.features is None:
            continue
        collection_id = managed_dataset_id_for(record)
        current = collections.get(collection_id)
        if current is None or record.created_at > current.created_at:
            collections[collection_id] = record
    return dict(sorted(collections.items()))


def list_feature_collections() -> FeatureCollectionListResponse:
    """Return every registered feature collection.

    Templates are loaded once for the whole listing. `registry_datasets.get_dataset` rebuilds
    its lookup from `list_datasets()` on every call, which reloads every built-in and plugin
    template, so asking it per collection turned one listing into N full registry scans.
    """
    collections = registered_collections()
    templates = _templates_by_id() if collections else {}
    return FeatureCollectionListResponse(
        items=[
            _build_record(collection_id, record, templates.get(record.dataset_id, {}))
            for collection_id, record in collections.items()
        ]
    )


def get_feature_collection_or_404(collection_id: str) -> FeatureCollectionRecord:
    """Return one registered feature collection, or raise 404."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return _build_record(collection_id, record, registry_datasets.get_dataset(record.dataset_id) or {})


def _templates_by_id() -> dict[str, dict[str, Any]]:
    """Return every declared template keyed by id, in one registry scan."""
    return {str(template["id"]): template for template in registry_datasets.list_datasets() if "id" in template}


def get_collection_record_or_404(collection_id: str) -> ArtifactRecord:
    """Return the artifact record behind a registered collection, for callers that read it."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return record


def published_collection_file_or_404(collection_id: str) -> Path:
    """Return the GeoParquet file a *published* collection is registered at, or raise 404.

    Resolved through the record, never by looking in the store directory. That is the same rule
    the listing follows, and it is what stops this route becoming a way to read any file that
    happens to be under the store root: a caller can only reach bytes some record already points
    at, and the record is the only thing that puts a file there.

    Publication is required here, unlike `/features`. The listing is the operator's inventory of
    what this instance holds; this is the asset a STAC collection advertises, and STAC only
    advertises published collections — so serving an unpublished one would hand out data the
    catalogue deliberately withholds.
    """
    record = registered_collections().get(collection_id)
    if record is None or record.publication.status != PublicationStatus.PUBLISHED:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    raw = record.path or (record.asset_paths[0] if record.asset_paths else None)
    if raw is None:
        raise HTTPException(status_code=409, detail=f"Feature collection '{collection_id}' has no stored path")
    path = Path(raw)
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Feature collection '{collection_id}' is registered but its file is missing"
        )
    return path


def _build_record(collection_id: str, record: ArtifactRecord, template: dict[str, Any]) -> FeatureCollectionRecord:
    """Build one response row from a record and its already-resolved template.

    The template is passed in rather than looked up, so a listing resolves them once. A feature
    template (plugins/features/) is CLIM-926's work, so today it is empty and the descriptive
    fields come back null — read through the same registry the raster path uses rather than a
    second one, so they populate when templates exist without this needing to change.
    """
    detail = record.features
    if detail is None:  # pragma: no cover - registered_collections filters these out
        raise HTTPException(status_code=500, detail=f"Feature collection '{collection_id}' has no feature detail")
    licence = parse_licence(template.get("license"))
    return FeatureCollectionRecord(
        id=collection_id,
        name=record.dataset_name,
        description=_as_text(template.get("description")),
        license=licence.stac_license,
        license_url=licence.url,
        attribution=_as_text(template.get("attribution")),
        id_property=detail.id_property,
        feature_count=detail.feature_count,
        geometry_types=store.stored_geometry_types(record),
        primary_geometry=detail.primary_geometry,
        crs=detail.crs,
        version=record.version,
        extent=record.coverage,
        last_updated=record.created_at,
    )


def _as_text(value: Any) -> str | None:
    """Return a non-blank prose field, normalised the way the dataset path normalises one."""
    if not isinstance(value, str):
        return None
    return value.strip() or None
