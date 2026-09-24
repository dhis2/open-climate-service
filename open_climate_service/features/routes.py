"""FastAPI routes for registered feature collections."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from open_climate_service.features import services
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord
from open_climate_service.shared.geoparquet import PARQUET_MEDIA_TYPE

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


@router.get("/{collection_id}/data.parquet", response_class=FileResponse)
def download_feature_collection(collection_id: str) -> FileResponse:
    """Serve the GeoParquet a published collection is stored as.

    The href its STAC collection advertises as the `data` asset, so a client that reads the
    catalogue can fetch the bytes it describes. The whole file: windowed reads are the reader's
    job inside a workflow, and a query surface over collections is not this ticket's.

    The path comes from the registered record — see `published_collection_file_or_404`.
    """
    path = services.published_collection_file_or_404(collection_id)
    return FileResponse(path, media_type=PARQUET_MEDIA_TYPE, filename=f"{collection_id}.parquet")
