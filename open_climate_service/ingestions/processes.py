"""Process execution wrappers for ingestion and sync operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents import services as extent_services
from open_climate_service.ingestions import services
from open_climate_service.ingestions.schemas import ArtifactRecord, SyncAction, SyncResponse
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEventDraft, JobExecutionResult

logger = logging.getLogger(__name__)

INGEST_EVENT_ACTION = "ingest"
"""`action` of the update event an ingestion emits; syncs report their planner action instead."""

_UPDATE_MARKER = "dataset_update_planned"
"""Cursor key recording that an attempt of this job planned a change to stored data.

Written before anything is fetched. A recovered job whose earlier attempt committed its data
and died before completing finds the dataset already current, so without this marker it would
succeed without an event and the configured workflows would never run for that ingestion.
"""


def dataset_update_event(
    *,
    dataset_id: str,
    artifact_id: str | None,
    action: str,
    previous_end: str | None,
    current_start: str | None,
    current_end: str | None,
) -> JobEventDraft:
    """Describe one change to a managed dataset for workflow automation.

    `previous_end` is the coverage end before the change; None means every period up to
    `current_end` is new or rewritten. `current_start` and `current_end` describe the
    coverage after it.
    """
    return JobEventDraft(
        type=DATASET_UPDATED_EVENT_TYPE,
        source=f"/datasets/{dataset_id}",
        data={
            "dataset_id": dataset_id,
            "artifact_id": artifact_id,
            "action": action,
            "previous_end": previous_end,
            "current_start": current_start,
            "current_end": current_end,
        },
    )


def ingest_dataset(
    *,
    dataset: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
    overwrite: bool = False,
    publish: bool = True,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    save_cursor: Callable[[dict[str, Any]], None] | None = None,
    load_cursor: Callable[[], dict[str, Any] | None] | None = None,
) -> tuple[ArtifactRecord, list[JobEventDraft]]:
    """Ingest a dataset over the configured extent and describe any change it made.

    Returns the update event only when stored data changed, so re-ingesting a dataset that
    is already current triggers nothing. Inside a job, the planned change is also recorded
    in the job cursor, so a recovered attempt still emits the event its earlier attempt owed.
    """
    extent = extent_services.get_extent_or_404()
    cursor = load_cursor() if load_cursor is not None else None
    earlier = cursor.get(_UPDATE_MARKER) if isinstance(cursor, dict) else None
    # An earlier attempt's marker wins: it saw the coverage before any of this job's commits.
    marker = earlier if isinstance(earlier, dict) else None

    def planned(previous_end: str | None) -> None:
        nonlocal marker
        if marker is None:
            marker = {"previous_end": previous_end}
            if save_cursor is not None:
                save_cursor({**(cursor or {}), _UPDATE_MARKER: marker})

    def save(checkpoint: dict[str, Any]) -> None:
        # The streaming ingest replaces the whole cursor on each commit; keep the marker.
        if save_cursor is not None:
            save_cursor({**checkpoint, _UPDATE_MARKER: marker} if marker is not None else checkpoint)

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
        save_cursor=save if save_cursor is not None else None,
        on_update_planned=planned,
    )
    if marker is None:
        return artifact, []
    temporal = artifact.coverage.temporal
    event = dataset_update_event(
        dataset_id=str(dataset["id"]),
        artifact_id=artifact.artifact_id,
        action=INGEST_EVENT_ACTION,
        previous_end=marker.get("previous_end"),
        current_start=temporal.start,
        current_end=temporal.end,
    )
    return artifact, [event]


def sync_update_events(dataset_id: str, response: SyncResponse) -> list[JobEventDraft]:
    """Describe the change a completed sync made, if it wrote data."""
    detail = response.sync_detail
    if (
        response.status != "completed"
        or detail is None
        or detail.action
        not in {
            SyncAction.APPEND,
            SyncAction.REMATERIALIZE,
        }
    ):
        return []
    return [
        dataset_update_event(
            dataset_id=dataset_id,
            artifact_id=response.sync_id,
            action=detail.action.value,
            # A rematerialization may rewrite every historical value, so workflows
            # must process the complete post-update coverage rather than only the
            # periods after the old boundary.
            previous_end=detail.current_end if detail.action == SyncAction.APPEND else None,
            current_start=detail.current_start,
            current_end=detail.target_end,
        )
    ]


def record_inline_update(*, label: str, request: dict[str, Any], result: Any, events: list[JobEventDraft]) -> None:
    """Make the events of work done inline durable, as a completed job, for automation.

    Logged rather than raised on failure: the data is already committed, so failing the
    request would misreport it. The update simply does not reach automation.
    """
    if not events:
        return
    from open_climate_service.ingestions.job_submission import INGESTION_JOB_HREF_BASE
    from open_climate_service.jobs.service import get_job_service

    try:
        get_job_service().record_completed_job(
            label=label,
            request=request,
            result=result,
            events=events,
            job_href_base=INGESTION_JOB_HREF_BASE,
        )
    except Exception:
        logger.exception("Could not record the dataset update from %s; automation will not see it", label)


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
    load_cursor: Callable[[], dict[str, Any] | None] | None = None,
) -> JobExecutionResult:
    """Execute one managed-dataset ingestion as a job, emitting `dataset.updated` on change."""
    dataset = registry_datasets.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    artifact, events = ingest_dataset(
        dataset=dataset,
        start=start,
        end=end,
        overwrite=overwrite,
        publish=publish,
        on_progress=on_progress,
        is_cancel_requested=is_cancel_requested,
        save_cursor=save_cursor,
        load_cursor=load_cursor,
    )
    return JobExecutionResult(result=services.get_ingestion_or_404(artifact.artifact_id), events=events)


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
    return JobExecutionResult(result=response, events=sync_update_events(dataset_id, response))
