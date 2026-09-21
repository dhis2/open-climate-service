"""Reading the registered feature collections, for `GET /features`."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.features import store
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord
from open_climate_service.publications.services import managed_dataset_id_for
from open_climate_service.shared.licences import parse_licence


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
    """Return every registered feature collection."""
    return FeatureCollectionListResponse(
        items=[_build_record(collection_id, record) for collection_id, record in registered_collections().items()]
    )


def get_feature_collection_or_404(collection_id: str) -> FeatureCollectionRecord:
    """Return one registered feature collection, or raise 404."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return _build_record(collection_id, record)


def get_collection_record_or_404(collection_id: str) -> ArtifactRecord:
    """Return the artifact record behind a registered collection, for callers that read it."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return record


def _build_record(collection_id: str, record: ArtifactRecord) -> FeatureCollectionRecord:
    detail = record.features
    if detail is None:  # pragma: no cover - registered_collections filters these out
        raise HTTPException(status_code=500, detail=f"Feature collection '{collection_id}' has no feature detail")
    # A feature template (plugins/features/) is CLIM-926's work, so today this lookup finds
    # nothing and the descriptive fields come back null. Read through the same registry the
    # raster path uses rather than inventing a second one, so those fields start being populated
    # when templates exist without this needing to change.
    template = registry_datasets.get_dataset(record.dataset_id) or {}
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
