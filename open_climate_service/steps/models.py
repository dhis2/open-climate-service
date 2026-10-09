"""The step model: what runs, and when (CLIM-1378).

A step is one unit of work an instance does without someone pressing a button: sync a dataset,
refresh a feature collection, run a workflow, or deliver an export. Every kind says *when* in the
same two ways, on a cron or after something it depends on changed, and every kind can also be
run by hand. A step without a cron and without ``after`` runs only by hand.

| kind       | target                   | cron | after                         |
| ---------- | ------------------------ | ---- | ----------------------------- |
| `sync`     | a managed dataset        | yes  | no                            |
| `refresh`  | a feature collection     | yes  | no                            |
| `workflow` | an openEO workflow       | yes  | a dataset or a collection     |
| `deliver`  | a named export           | no   | a workflow step               |

``after`` is what used to be a workflow trigger in ``automation.workflow_triggers``; a deliver
step after a workflow step is what used to be that trigger's ``deliver`` block.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from open_climate_service.shared.time import utc_now

StepKind = Literal["sync", "refresh", "workflow", "deliver"]

STEP_KINDS: tuple[StepKind, ...] = ("sync", "refresh", "workflow", "deliver")

_STEP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

MAX_STEP_ATTEMPTS = 10
"""Upper bound on attempts per run, the same as a triggered openEO job's."""


class After(BaseModel):
    """What a step waits for: exactly one of a dataset, a feature collection or another step."""

    model_config = ConfigDict(extra="forbid")

    dataset: str | None = Field(default=None, min_length=1, description="Run after this dataset changed.")
    collection: str | None = Field(
        default=None, min_length=1, description="Run after this feature collection was refreshed."
    )
    step: str | None = Field(default=None, min_length=1, description="Run after this step's job finished.")

    @model_validator(mode="after")
    def exactly_one(self) -> "After":
        given = [name for name in ("dataset", "collection", "step") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError("after must name exactly one of dataset, collection or step")
        return self

    @property
    def describe(self) -> str:
        if self.dataset is not None:
            return f"dataset {self.dataset}"
        if self.collection is not None:
            return f"collection {self.collection}"
        return f"step {self.step}"


class Step(BaseModel):
    """One unit of automated work, stored and edited live."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128, description="Stable id, unique on the instance.")
    kind: StepKind
    target: str = Field(min_length=1, description="The dataset, collection, workflow or export the step acts on.")
    cron: str | None = Field(default=None, description="Five-field cron, in the scheduler's timezone.")
    after: After | None = None
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow arguments. `$event.*` references resolve against the event that started the run.",
    )
    publish: bool = Field(default=True, description="sync and refresh: publish the result.")
    dry_run: bool = Field(default=True, description="deliver: validate in the target, store nothing.")
    max_attempts: int = Field(default=3, ge=1, le=MAX_STEP_ATTEMPTS)
    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _STEP_ID.match(value):
            raise ValueError("id must start with a letter or digit and hold only letters, digits, '.', '_' or '-'")
        return value

    @field_validator("cron")
    @classmethod
    def valid_cron(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            CronTrigger.from_crontab(value)
        except ValueError as exc:
            raise ValueError(f"invalid five-field cron expression {value!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def valid_for_kind(self) -> "Step":
        if self.cron is not None and self.after is not None:
            raise ValueError("a step runs on a cron or after something changed, not both")
        if self.kind in ("sync", "refresh") and self.after is not None:
            raise ValueError(f"a {self.kind} step runs on a cron or by hand; it cannot wait for another change")
        if self.kind == "workflow" and self.after is not None and self.after.step is not None:
            raise ValueError("a workflow step runs after a dataset or a collection changed, not after another step")
        if self.kind == "deliver":
            if self.after is None or self.after.step is None:
                raise ValueError("a deliver step runs after the workflow step whose result it delivers")
            if self.cron is not None:
                raise ValueError("a deliver step runs after its workflow step, not on a cron")
        if self.kind != "workflow" and self.arguments:
            raise ValueError("only a workflow step takes arguments")
        return self

    @property
    def when(self) -> str:
        """How the step is started, in words: the cron, what it waits for, or by hand."""
        if self.cron is not None:
            return f"cron {self.cron}"
        if self.after is not None:
            return f"after {self.after.describe}"
        return "by hand"
