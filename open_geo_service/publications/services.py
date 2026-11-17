"""Artifact publication helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from open_geo_service.ingestions.schemas import ArtifactRecord, PublicationStatus


def publish_artifact(record: ArtifactRecord) -> ArtifactRecord:
    """Mark an artifact as published."""
    collection_id = managed_dataset_id_for(record)
    return record.model_copy(
        update={
            "publication": record.publication.model_copy(
                update={
                    "status": PublicationStatus.PUBLISHED,
                    "collection_id": collection_id,
                    "published_at": datetime.now(UTC),
                }
            )
        }
    )


def managed_dataset_id_for_scope(dataset_id: str) -> str:
    """Return a stable managed dataset identifier for the single configured extent."""
    return dataset_id


def _collection_id_for(record: ArtifactRecord) -> str:
    return managed_dataset_id_for_scope(record.dataset_id)


def managed_dataset_id_for(record: ArtifactRecord) -> str:
    """Return the stable managed dataset id for a stored record."""
    return _collection_id_for(record)
