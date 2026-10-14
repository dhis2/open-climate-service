"""FastAPI routes for registered feature collections."""

from fastapi import APIRouter, Header, Request
from fastapi.responses import FileResponse, JSONResponse

from open_climate_service.features import services
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord
from open_climate_service.features.tiles import PMTILES_MEDIA_TYPE, STYLE_MEDIA_TYPE
from open_climate_service.ingestions.job_submission import INGESTION_JOB_HREF_BASE
from open_climate_service.shared.geoparquet import PARQUET_MEDIA_TYPE
from open_climate_service.shared.urls import absolute_url, path_segment

router = APIRouter()


def _prefers_async(prefer: str | None) -> bool:
    directives = [item.strip().split(";", 1)[0].strip().lower() for item in (prefer or "").split(",")]
    return "respond-async" in directives


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


@router.post(
    "/{collection_id}/refresh",
    response_model=FeatureCollectionRecord,
    responses={202: {"description": "Queued; follow the Location header to the job."}},
)
def refresh_feature_collection(
    collection_id: str,
    publish: bool = True,
    prefer: str | None = Header(default=None),
) -> FeatureCollectionRecord | JSONResponse:
    """Fetch a feature collection from its template's provider, creating or replacing it.

    Pass ``Prefer: respond-async`` to queue it as a background job and get 202 with
    ``Location: /ingestions/jobs/{id}``, as for ``POST /ingestions``. Refused with 404 for an
    unknown template and 400 for one naming no provider this instance has, before anything is
    queued; refused on a read-only instance like every other write.
    """
    services.refreshable_feature_template_or_error(collection_id)
    if _prefers_async(prefer):
        from open_climate_service.jobs.service import get_job_service

        job = get_job_service().submit_callable_job(
            func=services.execute_feature_refresh,
            label="feature refresh",
            request={"collection_id": collection_id, "publish": publish},
            job_href_base=INGESTION_JOB_HREF_BASE,
        )
        location = f"{INGESTION_JOB_HREF_BASE}/{job.job_id}"
        return JSONResponse(
            status_code=202,
            headers={"Location": location},
            content={"job_id": job.job_id, "status": str(getattr(job.status, "value", job.status)), "href": location},
        )
    return services.execute_feature_refresh(collection_id=collection_id, publish=publish)


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


@router.get("/{collection_id}/data.pmtiles", response_class=FileResponse)
# HEAD as well, for a client that sizes the archive before reading ranges of it. Kept out of the
# OpenAPI schema so the operation is listed once.
@router.head("/{collection_id}/data.pmtiles", response_class=FileResponse, include_in_schema=False)
def download_feature_tiles(collection_id: str) -> FileResponse:
    """Serve the PMTiles archive the map viewer draws a published collection from.

    Range requests are answered by `FileResponse`, which is all a PMTiles reader needs: it
    fetches the header and the tiles in view, never the whole archive. Advertised as the
    collection's `tiles` STAC asset whenever it exists.
    """
    path = services.published_collection_tiles_or_404(collection_id)
    return FileResponse(path, media_type=PMTILES_MEDIA_TYPE, filename=f"{collection_id}.pmtiles")


@router.get("/{collection_id}/styles/default.json")
def feature_style(collection_id: str, request: Request) -> JSONResponse:
    """Serve the MapLibre style a published collection is drawn with, over its vector tiles.

    Advertised as the collection's `style` STAC asset whenever it has tiles. The map viewer draws
    from it, so any other MapLibre client draws the collection the same way.
    """
    tiles_url = absolute_url(request, f"/features/{path_segment(collection_id)}/data.pmtiles")
    style = services.published_collection_style_or_404(collection_id, tiles_url)
    return JSONResponse(style, media_type=STYLE_MEDIA_TYPE)
