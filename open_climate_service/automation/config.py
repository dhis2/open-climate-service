"""Workflow triggers, compiled from the workflow and deliver steps in the steps store (CLIM-1378).

A trigger is the runtime shape of a workflow step: what it waits for (a dataset update, a
collection refresh, or nothing when it runs on a cron or by hand), the workflow and arguments,
and the delivery a deliver step after it asks for. The automation service consumes triggers, so
the event, replay, retry and delivery machinery is the same whichever way a step was written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_climate_service import config as api_config
from open_climate_service.jobs.models import COLLECTION_UPDATED_EVENT_TYPE
from open_climate_service.openeo.jobs import MAX_TRIGGERED_ATTEMPTS

if TYPE_CHECKING:
    from open_climate_service.steps.models import Step

TriggerEvent = Literal["dataset.updated", "collection.updated"]


class TriggerDelivery(BaseModel):
    """Deliver a triggered job's named export once the job finishes."""

    model_config = ConfigDict(extra="forbid")

    export: str = Field(min_length=1)
    # Writing to a production DHIS2 is an explicit opt-in.
    dry_run: bool = True


class WorkflowTrigger(BaseModel):
    """Bind one change, or a cron or a hand, to an existing openEO workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    # The dataset (or, for collection.updated, the feature collection) whose change starts the
    # workflow. None for a step that runs on a cron or by hand.
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


def compile_steps(steps: list[Step]) -> AutomationConfig:
    """Turn the enabled workflow steps, and the deliver steps after them, into triggers.

    A deliver step after a workflow step becomes that trigger's delivery; at most one deliver
    step may follow a workflow step, and it must follow one that exists. A paused workflow step
    is left out, and so is a paused deliver step (the workflow then runs without delivering).
    """
    workflows = {step.id: step for step in steps if step.kind == "workflow"}
    deliveries: dict[str, Step] = {}
    for step in steps:
        if step.kind != "deliver" or step.after is None or step.after.step is None:
            continue
        upstream = step.after.step
        if upstream not in workflows:
            raise ValueError(f"Deliver step {step.id!r} runs after {upstream!r}, which is not a workflow step")
        if upstream in deliveries:
            raise ValueError(
                f"Workflow step {upstream!r} is followed by two deliver steps, "
                f"{deliveries[upstream].id!r} and {step.id!r}; one workflow result is delivered once"
            )
        deliveries[upstream] = step
    triggers: list[WorkflowTrigger] = []
    for step in workflows.values():
        if not step.enabled:
            continue
        on_update_of: str | None = None
        event: TriggerEvent = "dataset.updated"
        if step.after is not None and step.after.dataset is not None:
            on_update_of = step.after.dataset
        elif step.after is not None and step.after.collection is not None:
            on_update_of = step.after.collection
            event = "collection.updated"
        delivery = deliveries.get(step.id)
        triggers.append(
            WorkflowTrigger(
                id=step.id,
                on_update_of=on_update_of,
                event=event,
                workflow_id=step.target,
                arguments=step.arguments,
                max_attempts=step.max_attempts,
                deliver=(
                    TriggerDelivery(export=delivery.target, dry_run=delivery.dry_run)
                    if delivery is not None and delivery.enabled
                    else None
                ),
            )
        )
    return AutomationConfig(workflow_triggers=triggers)


def get_automation_config() -> AutomationConfig:
    """Compile workflow automation from the steps store.

    The ``automation`` block of ``climate-service.yaml`` is refused rather than merged: the store
    is the one source, so there is no precedence rule to explain.
    """
    if "automation" in api_config.get_config():
        raise ValueError(
            "automation in climate-service.yaml is no longer supported; workflow triggers and their "
            "deliveries are workflow and deliver steps, managed through /steps (CLIM-1378)"
        )
    from open_climate_service.steps.store import list_steps

    return compile_steps(list_steps())
