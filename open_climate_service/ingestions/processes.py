"""Process execution wrappers for ingestion and sync operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents.services import get_extent_or_404
from open_climate_service.ingestions import services
from open_climate_service.ingestions.schemas import IngestionResponse, SyncAction
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEventDraft, JobExecutionResult


def execute_ingestion(
    *,
    dataset_id: str,
    start: str | None = None,
    end: str | None = None,
    overwrite: bool = False,
    publish: bool = True,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, Any]], None] | None = None,
) -> IngestionResponse:
    """Execute one managed-dataset ingestion using the configured extent."""
    dataset = registry_datasets.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    extent = get_extent_or_404()
    artifact = services.create_artifact(
        dataset=dataset,
        start=start,
        end=end,
        bbox=list(extent["bbox"]),
        country_code=extent.get("country_code"),
        overwrite=overwrite,
        publish=publish,
        on_progress=on_progress,
        is_cancel_requested=is_cancel_requested,
        save_cursor=save_cursor,
    )
    return services.get_ingestion_or_404(artifact.artifact_id)


def execute_sync(
    *,
    dataset_id: str,
    end: str | None = None,
    publish: bool = True,
) -> JobExecutionResult:
    """Execute one sync and describe any resulting dataset update."""
    response = services.sync_dataset(
        dataset_id=dataset_id,
        end=end,
        publish=publish,
    )
    events: list[JobEventDraft] = []
    detail = response.sync_detail
    if (
        response.status == "completed"
        and detail is not None
        and detail.action
        in {
            SyncAction.APPEND,
            SyncAction.REMATERIALIZE,
        }
    ):
        events.append(
            JobEventDraft(
                type=DATASET_UPDATED_EVENT_TYPE,
                source=f"/datasets/{dataset_id}",
                data={
                    "dataset_id": dataset_id,
                    "artifact_id": response.sync_id,
                    "action": detail.action.value,
                    "previous_end": detail.current_end,
                    "current_end": detail.target_end,
                },
            )
        )
    return JobExecutionResult(result=response, events=events)
