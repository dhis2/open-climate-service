"""Routes for EO ingestion, datasets, and sync operations."""

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.responses import Response

from open_climate_service.data_registry.routes import _get_dataset_or_404
from open_climate_service.extents.services import get_extent_or_404
from open_climate_service.ingestions import services
from open_climate_service.ingestions.job_submission import INGESTION_JOB_HREF_BASE, submit_sync_job
from open_climate_service.ingestions.schemas import (
    CreateIngestionRequest,
    DatasetDetailRecord,
    DatasetListResponse,
    IngestionListResponse,
    IngestionResponse,
    SyncDatasetRequest,
    SyncDetail,
    SyncResponse,
)
from open_climate_service.jobs.models import JobLink, JobRecord
from open_climate_service.jobs.service import get_job_service
from open_climate_service.shared.thumbnails import thumbnail_path

ingestions_router = APIRouter()
datasets_router = APIRouter()
zarr_router = APIRouter()
icechunk_router = APIRouter()
sync_router = APIRouter()


def _prefer_respond_async(prefer: str | None) -> bool:
    if prefer is None:
        return False
    directives = [item.strip().split(";", 1)[0].strip().lower() for item in prefer.split(",")]
    return "respond-async" in directives


def _with_ingestion_job_self_link(record: JobRecord) -> JobRecord:
    self_link = JobLink(href=f"{INGESTION_JOB_HREF_BASE}/{record.job_id}", rel="self", title="Job detail")
    other_links = [link for link in record.links if link.rel != "self"]
    return record.model_copy(update={"links": [self_link, *other_links]})


@ingestions_router.get("/jobs/{job_id}", response_model=JobRecord)
def get_ingestion_job(job_id: str) -> JobRecord:
    """Return the status of an async ingestion or sync job."""
    from open_climate_service.jobs import store

    record = store.get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _with_ingestion_job_self_link(record)


@ingestions_router.delete("/jobs/{job_id}", response_model=JobRecord, status_code=202)
def cancel_ingestion_job(job_id: str) -> JobRecord:
    """Request cooperative cancellation for an async ingestion or sync job."""
    record = get_job_service().request_cancellation(job_id)
    return _with_ingestion_job_self_link(record)


@ingestions_router.post("")
def create_ingestion(
    request: CreateIngestionRequest,
    response: Response,
    prefer: str | None = Header(default=None),
) -> IngestionResponse:
    """Create or update a managed dataset from a dataset template and configured extent.

    Pass ``Prefer: respond-async`` to queue the ingestion as a background job and
    return immediately with 202 + ``Location: /ingestions/jobs/{id}``.
    """
    if _prefer_respond_async(prefer):
        _get_dataset_or_404(request.dataset_id)
        get_extent_or_404()

        from open_climate_service.ingestions.processes import execute_ingestion
        from open_climate_service.jobs.service import get_job_service

        job = get_job_service().submit_callable_job(
            func=execute_ingestion,
            label="ingestion",
            request=request.model_dump(),
            job_href_base=INGESTION_JOB_HREF_BASE,
        )
        response.status_code = 202
        response.headers["Location"] = f"{INGESTION_JOB_HREF_BASE}/{job.job_id}"
        return IngestionResponse(ingestion_id=job.job_id, status=job.status, dataset=None)
    dataset = _get_dataset_or_404(request.dataset_id)
    extent = get_extent_or_404()
    resolved_bbox = list(extent["bbox"])
    resolved_country_code = extent.get("country_code")
    artifact = services.create_artifact(
        dataset=dataset,
        start=request.start,
        end=request.end,
        bbox=resolved_bbox,
        country_code=resolved_country_code,
        overwrite=request.overwrite,
        publish=request.publish,
    )
    return IngestionResponse(
        ingestion_id=artifact.artifact_id,
        status="completed",
        dataset=services.get_dataset_summary_for_artifact_or_404(artifact.artifact_id),
    )


@ingestions_router.get("", response_model=IngestionListResponse)
def list_ingestions() -> IngestionListResponse:
    """List ingestion run records for operational and admin use."""
    return services.list_ingestions()


@ingestions_router.get("/{ingestion_id}", response_model=IngestionResponse)
def get_ingestion(ingestion_id: str) -> IngestionResponse:
    """Return the managed dataset view created for a given ingestion."""
    return services.get_ingestion_or_404(ingestion_id)


@datasets_router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List managed datasets."""
    return services.list_datasets()


@datasets_router.get("/{dataset_id}", response_model=DatasetDetailRecord)
def get_dataset(dataset_id: str) -> DatasetDetailRecord:
    """Get managed dataset metadata and available versions."""
    return services.get_dataset_or_404(dataset_id)


@datasets_router.get("/{dataset_id}/thumbnail.png", response_class=FileResponse)
def get_dataset_thumbnail(dataset_id: str) -> FileResponse:
    """Serve a dataset's thumbnail, the image its STAC collection points at.

    404 when the dataset has none. A thumbnail is written at the end of each ingest and sync
    run, so a published dataset normally has one and the miss is a dataset not yet ingested, or
    a render that failed or found nothing to draw. It is a "no preview" answer rather than a
    server fault. The STAC collection only advertises the asset when the file is there, so a
    client following a published href does not meet this.
    """
    path = thumbnail_path(dataset_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No thumbnail for dataset '{dataset_id}'")
    return FileResponse(path, media_type="image/png")


@datasets_router.get("/{dataset_id}/download")
def download_artifact_file(dataset_id: str) -> FileResponse:
    """Download the primary saved file for a dataset when available."""
    artifact = services.get_latest_artifact_for_dataset_or_404(dataset_id)
    if artifact.path is None or artifact.format.value == "zarr":
        raise HTTPException(
            status_code=409,
            detail="Dataset is not a single downloadable file; use metadata and dataset assets instead",
        )

    media_type = "application/x-netcdf"
    filename = f"{dataset_id}.nc"
    return FileResponse(artifact.path, media_type=media_type, filename=filename)


@zarr_router.api_route("/{dataset_id}/{relative_path:path}", methods=["GET", "HEAD"], response_model=None)
def get_canonical_zarr_store_file(
    request: Request, dataset_id: str, relative_path: str
) -> FileResponse | Response | dict[str, object]:
    """Serve canonical Zarr store content for a managed dataset."""
    range_header = request.headers.get("range")
    return services.get_dataset_zarr_store_file_or_404(dataset_id, relative_path, range_header=range_header)


@icechunk_router.api_route("/{dataset_id}/{file_path:path}", methods=["GET", "HEAD"], response_model=None)
def serve_icechunk_store_file(dataset_id: str, file_path: str) -> FileResponse:
    """Serve a raw Icechunk store file for native SDK access.

    Enables clients to open the store directly with::

        import icechunk, zarr

        repo = icechunk.Repository.open(icechunk.http_storage("http://<host>/icechunk/<dataset_id>/"))
        ds = zarr.open(repo.readonly_session("main").store, zarr_format=3)
    """
    return services.serve_icechunk_file(dataset_id, file_path)


@sync_router.post("/{dataset_id}")
def sync_dataset(
    dataset_id: str,
    request: SyncDatasetRequest,
    response: Response,
    prefer: str | None = Header(default=None),
) -> SyncResponse:
    """Sync a managed dataset forward from its latest available time step.

    Pass ``Prefer: respond-async`` to queue the sync as a background job and
    return immediately with 202 + ``Location: /ingestions/jobs/{id}``.
    """
    if _prefer_respond_async(prefer):
        services.plan_sync_dataset(dataset_id=dataset_id, end=request.end)

        job = submit_sync_job(
            dataset_id=dataset_id,
            end=request.end,
            publish=request.publish,
        )
        response.status_code = 202
        response.headers["Location"] = f"{INGESTION_JOB_HREF_BASE}/{job.job_id}"
        return SyncResponse(sync_id=None, status=job.status, message="Sync queued", dataset=None, sync_detail=None)

    return services.sync_dataset(
        dataset_id=dataset_id,
        end=request.end,
        publish=request.publish,
    )


@sync_router.get("/{dataset_id}/plan", response_model=SyncDetail)
def plan_sync_dataset(dataset_id: str, end: str | None = None) -> SyncDetail:
    """Return the sync plan for a managed dataset without starting a download."""
    return services.plan_sync_dataset(dataset_id=dataset_id, end=end)
