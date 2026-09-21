"""FastAPI routes for registered feature collections."""

from fastapi import APIRouter

from open_climate_service.features import services
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord

router = APIRouter()


@router.get("", response_model=FeatureCollectionListResponse)
def list_feature_collections() -> FeatureCollectionListResponse:
    """List the feature collections this instance holds.

    Registered collections only. A GeoParquet file placed in the store directory by hand does
    not appear: a record is what brings a collection into existence, so the store directory is
    not an inbox.
    """
    return services.list_feature_collections()


@router.get("/{collection_id}", response_model=FeatureCollectionRecord)
def get_feature_collection(collection_id: str) -> FeatureCollectionRecord:
    """Return one registered feature collection."""
    return services.get_feature_collection_or_404(collection_id)
