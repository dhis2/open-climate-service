"""Workflow triggers, compiled from the workflow and deliver tasks in the tasks store (CLIM-1378).

A trigger is the runtime shape of a workflow task: what it waits for (a dataset update, a
collection refresh, or nothing when it runs on a cron or by hand), the workflow and arguments,
and the delivery a deliver task after it asks for. The automation service consumes triggers, so
the event, replay, retry and delivery machinery is the same whichever way a task was written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_climate_service import config as api_config
from open_climate_service.jobs.models import COLLECTION_UPDATED_EVENT_TYPE
from open_climate_service.openeo.jobs import MAX_TRIGGERED_ATTEMPTS

if TYPE_CHECKING:
    from open_climate_service.tasks.models import Task

TriggerEvent = Literal["dataset.updated", "collection.updated"]


class TriggerDelivery(BaseModel):
    """Deliver a triggered job's named export once the job finishes."""

    model_config = ConfigDict(extra="forbid")

    export: str = Field(min_length=1)
    # Writing to a production DHIS2 is an explicit opt-in.
    dry_run: bool = True
    # The deliver task this delivery comes from, for its run records.
    task_id: str | None = None


class WorkflowTrigger(BaseModel):
    """Bind one change, or a cron or a hand, to an existing openEO workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    # The dataset (or, for collection.updated, the feature collection) whose change starts the
    # workflow. None for a task that runs on a cron or by hand.
    on_update_of: str | None = Field(default=None, min_length=1)
    event: TriggerEvent = "dataset.updated"
    workflow_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    replay_existing: bool = False
    deliver: TriggerDelivery | None = None
    # Attempts per triggered job, the first included. A transient failure or a restart during
    # execution is retried up to this bound; a permanent error is not retried at all.
    max_attempts: int = Field(default=3, ge=1, le=MAX_TRIGGERED_ATTEMPTS)

    def matches(self, event_type: str, data: dict[str, Any]) -> bool:
        """Whether a persisted event starts this trigger."""
        if self.on_update_of is None or event_type != self.event:
            return False
        key = "collection_id" if self.event == COLLECTION_UPDATED_EVENT_TYPE else "dataset_id"
        return data.get(key) == self.on_update_of


class AutomationConfig(BaseModel):
    """Workflow automation owned by this OCS instance."""

    model_config = ConfigDict(extra="forbid")

    workflow_triggers: list[WorkflowTrigger] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "AutomationConfig":
        ids = [trigger.id for trigger in self.workflow_triggers]
        duplicates = sorted({trigger_id for trigger_id in ids if ids.count(trigger_id) > 1})
        if duplicates:
            raise ValueError(f"workflow trigger ids must be unique: {duplicates}")
        return self


def compile_tasks(tasks: list[Task]) -> AutomationConfig:
    """Turn the enabled workflow tasks, and the deliver tasks after them, into triggers.

    A deliver task after a workflow task becomes that trigger's delivery; at most one deliver
    task may follow a workflow task, and it must follow one that exists. A paused workflow task
    is left out, and so is a paused deliver task (the workflow then runs without delivering).
    """
    workflows = {task.id: task for task in tasks if task.kind == "workflow"}
    deliveries: dict[str, Task] = {}
    for task in tasks:
        if task.kind != "deliver" or task.after is None or task.after.task is None:
            continue
        upstream = task.after.task
        if upstream not in workflows:
            raise ValueError(f"Deliver task {task.id!r} runs after {upstream!r}, which is not a workflow task")
        if upstream in deliveries:
            raise ValueError(
                f"Workflow task {upstream!r} is followed by two deliver tasks, "
                f"{deliveries[upstream].id!r} and {task.id!r}; one workflow result is delivered once"
            )
        deliveries[upstream] = task
    triggers: list[WorkflowTrigger] = []
    for task in workflows.values():
        if not task.enabled:
            continue
        on_update_of: str | None = None
        event: TriggerEvent = "dataset.updated"
        if task.after is not None and task.after.dataset is not None:
            on_update_of = task.after.dataset
        elif task.after is not None and task.after.collection is not None:
            on_update_of = task.after.collection
            event = "collection.updated"
        delivery = deliveries.get(task.id)
        triggers.append(
            WorkflowTrigger(
                id=task.id,
                on_update_of=on_update_of,
                event=event,
                workflow_id=task.target,
                arguments=task.arguments,
                max_attempts=task.max_attempts,
                deliver=(
                    TriggerDelivery(export=delivery.target, dry_run=delivery.dry_run, task_id=delivery.id)
                    if delivery is not None and delivery.enabled
                    else None
                ),
            )
        )
    return AutomationConfig(workflow_triggers=triggers)


def get_automation_config() -> AutomationConfig:
    """Compile workflow automation from the tasks store.

    The ``automation`` block of ``climate-service.yaml`` is refused rather than merged: the store
    is the one source, so there is no precedence rule to explain.
    """
    if "automation" in api_config.get_config():
        raise ValueError(
            "automation in climate-service.yaml is no longer supported; workflow triggers and their "
            "deliveries are workflow and deliver tasks, managed through /tasks (CLIM-1378)"
        )
    from open_climate_service.tasks.store import list_tasks

    return compile_tasks(list_tasks())
