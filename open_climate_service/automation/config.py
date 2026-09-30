"""Validated instance configuration for event-driven workflow triggers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_climate_service import config as api_config


class TriggerDelivery(BaseModel):
    """Deliver a triggered job's named export once the job finishes."""

    model_config = ConfigDict(extra="forbid")

    export: str = Field(min_length=1)
    # Writing to a production DHIS2 is an explicit opt-in.
    dry_run: bool = True


class WorkflowTrigger(BaseModel):
    """Bind one dataset update to an existing openEO workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    on_update_of: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    replay_existing: bool = False
    deliver: TriggerDelivery | None = None


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


def get_automation_config() -> AutomationConfig:
    """Load workflow automation from the instance configuration."""
    raw = api_config.get_config().get("automation", {})
    if not isinstance(raw, dict):
        raise ValueError("automation in CLIMATE_SERVICE_CONFIG must be a mapping")
    return AutomationConfig.model_validate(raw)
