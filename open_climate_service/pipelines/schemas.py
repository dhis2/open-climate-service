"""The pipeline object and the records of what was checked and run for it.

A workflow is a reusable, parameterized sequence of processes that defines how data is
transformed or analyzed. A pipeline is a managed configuration that connects a workflow to
its inputs and destination, governing when it runs, how results are delivered, and how
execution is monitored. The spec therefore holds a reference to one workflow and its
argument bindings, a destination plugin and its opaque settings, and a delivery policy;
never a process graph or destination-specific fields.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A pipeline id is also the id of the named export its results are saved through.
_PIPELINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class WorkflowReference(BaseModel):
    """Exactly one registered workflow and the arguments the pipeline binds to it.

    Arguments are checked against the workflow's declared parameters. A feature collection is
    bound as ``{"from_features": "<collection id>"}``, resolved at submission to the
    collection's current version. The workflow's one unbound ``temporal-interval`` parameter
    receives each run's interval, so it is never bound here.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Id of a registered workflow, as listed under GET /process_graphs.")
    arguments: dict[str, Any] = Field(default_factory=dict)


class PipelineDestination(BaseModel):
    """Which export plugin renders and delivers the results, with that plugin's own settings.

    The core never reads ``settings``: the plugin validates them, compiles them into the
    pipeline's named export, and runs its destination-side checks.
    """

    model_config = ConfigDict(extra="forbid")

    plugin: str = Field(min_length=1, description="Id of an export plugin that is a pipeline destination.")
    settings: dict[str, Any] = Field(default_factory=dict)


DeliveryMode = Literal["dry_run", "live", "paused"]
DeliveryPolicy = Literal["on_update", "manual"]


class PipelineDelivery(BaseModel):
    """Whether and when results go out.

    ``mode`` says whether anything goes out: validated by the destination only (dry run),
    written (live), or held (paused). ``policy`` says when a run starts by itself: after each
    update of a source dataset, or never (manual). A release schedule of its own is not
    implemented, so it is not accepted.
    """

    model_config = ConfigDict(extra="forbid")

    mode: DeliveryMode = "dry_run"
    policy: DeliveryPolicy = "on_update"

    @model_validator(mode="before")
    @classmethod
    def _refuse_unimplemented(cls, value: Any) -> Any:
        if isinstance(value, dict) and (value.get("policy") == "scheduled" or value.get("release_cron")):
            raise ValueError(
                "a scheduled release is not implemented yet; use on_update or manual, and keep the source dataset "
                "current with its sync schedule"
            )
        return value


class PipelineSpec(BaseModel):
    """What an operator declares: one workflow with its bindings, a destination and a delivery."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    workflow: WorkflowReference
    destination: PipelineDestination
    delivery: PipelineDelivery = Field(default_factory=PipelineDelivery)

    @field_validator("id")
    @classmethod
    def _export_safe(cls, value: str) -> str:
        if not _PIPELINE_ID.fullmatch(value):
            raise ValueError("id must start with a letter or digit and carry only letters, digits, '_' or '-'")
        return value


class Check(BaseModel):
    """One validation step and its verdict."""

    id: str
    status: Literal["pass", "fail", "skip"]
    message: str


class ValidationResult(BaseModel):
    """The outcome of validating a pipeline, kept on its record."""

    model_config = ConfigDict(extra="ignore")

    valid: bool
    checked_at: str
    bindings: str | None = Field(default=None, description="Digest of the bindings this result judged.")
    revision: int | None = Field(
        default=None, description="The pipeline revision this result was stored for; set when it is stored."
    )
    export: dict[str, Any] | None = Field(
        default=None, description="The named-export definition compiled from the destination, when it compiled."
    )
    checks: list[Check]


class DryRunResult(BaseModel):
    """The outcome of a dry run: what was rendered and the destination's own verdict on it."""

    model_config = ConfigDict(extra="ignore")

    ran_at: str
    start: str
    end: str
    bindings: str | None = Field(default=None, description="Digest of the bindings the dry run was judged on.")
    values: int = 0
    periods: list[str] = Field(default_factory=list)
    sample: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    rendered_only: bool = Field(
        default=False, description="True when the destination cannot validate a dry run, so it was only rendered."
    )
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Rendered without error and, where the destination validates, accepted with no conflicts."""
        if self.error is not None:
            return False
        if self.rendered_only:
            return self.values > 0
        report = self.report or {}
        return bool(report) and report.get("outcome") == "dry_run" and not report.get("conflicts")


RunTrigger = Literal["manual", "dataset_update", "retry"]
RunSubmission = Literal["pending", "submitted", "failed"]
RunJobOutcome = Literal["error", "canceled", "missing"]
RunDelivery = Literal["awaiting", "held", "submitted", "failed"]
DeliveryResult = Literal[
    "delivered", "validated", "acknowledged", "partial", "rejected", "unknown", "failed", "cancelled"
]


class PipelineRun(BaseModel):
    """One invocation of the pipeline's workflow over an explicit interval, and its delivery.

    The run is stored with ``submission: pending`` and its deterministic job id before the
    job exists, so an interrupted submission is found and finished later.

    States, and who moves a run out of them:

    - ``submission``: ``pending`` is retried automatically, a bounded number of times;
      ``failed`` is terminal (a permanent refusal, or the attempts ran out): run it again.
    - ``job_outcome`` is recorded once the job ended without a result: ``error``,
      ``canceled`` or ``missing`` (deleted). Terminal: run it again.
    - ``delivery``: ``awaiting`` is handed off automatically when the job finishes, and a
      transient hand-off failure is retried a bounded number of times under the same
      idempotency key; ``held`` waits for the pipeline to be resumed or live again, and is
      released then; ``failed`` is terminal until an operator delivers it by hand;
      ``submitted`` means handed to the delivery framework, not delivered.
    - ``delivery_result`` is the delivery job's confirmed outcome, recorded once it ends:
      ``delivered`` or ``validated`` (a dry run) confirm it; ``partial`` (conflicts),
      ``rejected``, ``unknown`` (no answer after sending, or the delivery job's record is gone),
      ``failed`` and ``cancelled`` need an operator, who sends it again where that is safe, or
      checks the destination and records ``acknowledged``.
    """

    model_config = ConfigDict(extra="ignore")

    run_id: str
    trigger: RunTrigger = "manual"
    event_id: str | None = None
    retry_of: str | None = None
    revision: int = 1
    bindings: str | None = None
    workflow_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict, description="The resolved arguments, as submitted.")
    process: dict[str, Any] | None = Field(default=None, description="The submitted process graph.")
    source: dict[str, Any] = Field(default_factory=dict, description="Source dataset and artifact, where known.")
    job_id: str
    mode: Literal["dry_run", "live"]
    start: str
    end: str
    submitted_at: str
    idempotency_key: str
    submission: RunSubmission = "submitted"
    submission_attempts: int = 0
    job_outcome: RunJobOutcome | None = None
    delivery: RunDelivery = "awaiting"
    delivery_attempts: int = 0
    delivery_round: int = Field(default=0, description="Bumped by a redelivery after the delivery job failed.")
    delivery_result: DeliveryResult | None = None
    delivery_job_id: str | None = None
    delivery_status_url: str | None = None
    delivered_at: str | None = None
    error: str | None = None


class SkippedUpdate(BaseModel):
    """A dataset update the pipeline decided not to run, kept so an operator can run it by hand."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    time: str
    dataset_id: str | None = None
    start: str | None = None
    end: str | None = None
    reason: str


class EventCheckpoint(BaseModel):
    """Which dataset updates this pipeline has decided on, independent of its run history.

    Every update with a time before ``through`` is decided; past it, ``handled`` holds the ids
    of the decided ones with their times. ``through`` only advances to a time no update still
    to be persisted can precede, so a delayed or out-of-order update is never skipped, and
    pruning the run history cannot cause a replay.
    """

    model_config = ConfigDict(extra="ignore")

    through: str | None = None
    handled: dict[str, str] = Field(default_factory=dict)
    retries: dict[str, int] = Field(
        default_factory=dict, description="Failed attempts to decide an update that may succeed later."
    )
    skipped: list[SkippedUpdate] = Field(
        default_factory=list,
        description=(
            "Updates not run, newest first; kept across pauses and resumes. A bounded list for an operator, "
            "not an audit trail: past its size the oldest entries are discarded and counted below."
        ),
    )
    skipped_discarded: int = Field(default=0, description="Entries discarded from `skipped` to keep it bounded.")
    skipped_discarded_through: str | None = Field(
        default=None, description="Time of the newest update whose entry was discarded."
    )


class PipelineRecord(BaseModel):
    """A stored pipeline with its revision, latest validation, dry run and runs."""

    model_config = ConfigDict(extra="ignore")

    spec: PipelineSpec
    revision: int = Field(
        default=1, description="Bumped on every change to the spec; edits must name the one they read."
    )
    created_at: str
    updated_at: str | None = None
    automation_since: str | None = Field(
        default=None, description="When dataset updates started to run this pipeline; none while it does not."
    )
    validation: ValidationResult | None = None
    dry_run: DryRunResult | None = None
    runs: list[PipelineRun] = Field(
        default_factory=list, description="Newest first; settled runs beyond the retained number are dropped."
    )
    events: EventCheckpoint = Field(default_factory=EventCheckpoint)
