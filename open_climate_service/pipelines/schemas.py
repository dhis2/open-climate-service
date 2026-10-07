"""The pipeline object and the records of what was checked and run for it."""

from __future__ import annotations

from typing import Any, Literal

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from open_climate_service.shared.features import is_dhis2_uid
from open_climate_service.shared.urls import is_segment_safe_id

SpatialReducer = Literal["mean", "sum", "min", "max", "median"]
SPATIAL_REDUCERS: tuple[str, ...] = ("mean", "sum", "min", "max", "median")


class PipelineSource(BaseModel):
    """Which managed dataset the pipeline reads. Its cadence sets the export period.

    The pipeline never fetches data itself. Keeping the dataset current is the dataset's own
    sync schedule, managed with the instance's schedules; the pipeline reacts to its updates.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="Id of a published managed dataset, as listed under GET /datasets.")


class OrganisationUnits(BaseModel):
    """Where the organisation unit geometry comes from: a registered feature collection."""

    feature_collection: str = Field(description="Id of a registered feature collection, as listed under GET /features.")


class PipelineSeries(BaseModel):
    """One dataset variable going to one DHIS2 data element."""

    data_element: str = Field(description="DHIS2 data element UID the values are imported into.")
    variable: str | None = Field(default=None, description="Dataset variable to export; defaults to the dataset's one.")

    @field_validator("data_element")
    @classmethod
    def _uid(cls, value: str) -> str:
        if not is_dhis2_uid(value):
            raise ValueError("data_element must be a DHIS2 UID: 11 characters, a letter first")
        return value


class PipelineDestination(BaseModel):
    """The DHIS2 side: connection, data set, organisation units and the series mapping."""

    type: Literal["dhis2"] = "dhis2"
    connection: str = Field(description="Id of a named DHIS2 connection from the instance configuration.")
    data_set: str | None = Field(
        default=None, description="DHIS2 data set UID, checked for period type and assignment."
    )
    organisation_units: OrganisationUnits
    series: list[PipelineSeries] = Field(min_length=1)

    @field_validator("data_set")
    @classmethod
    def _data_set_uid(cls, value: str | None) -> str | None:
        if value is not None and not is_dhis2_uid(value):
            raise ValueError("data_set must be a DHIS2 UID: 11 characters, a letter first")
        return value


class SpatialAggregation(BaseModel):
    """How pixel values are combined within each organisation unit."""

    reducer: SpatialReducer = "mean"


class PipelineAggregation(BaseModel):
    """Spatial only in this slice: the export period is the dataset's own cadence."""

    spatial: SpatialAggregation = Field(default_factory=SpatialAggregation)


DeliveryMode = Literal["dry_run", "live", "paused"]
DeliveryPolicy = Literal["on_update", "manual", "scheduled"]


class PipelineDelivery(BaseModel):
    """Whether, which and when: how a finished export is sent to DHIS2.

    ``mode`` says whether anything goes out: validated only (dry run), written (live), or held
    (paused). ``range`` says which values. ``policy`` says when: after each dataset update, only
    when an operator runs or delivers by hand, or at a release time of its own. A scheduled
    release is accepted so the shape is final, but it is not applied until stored pipelines are
    read by the automation; until then it behaves as manual and the validation says so.
    """

    model_config = ConfigDict(extra="forbid")

    mode: DeliveryMode = "dry_run"
    policy: DeliveryPolicy = "on_update"
    release_cron: str | None = Field(
        default=None, description="Five-field cron for a scheduled release; required when policy is scheduled."
    )
    range: Literal["updated", "all"] = Field(
        default="updated",
        description="Process the updated interval (default) or the complete stored history after every update.",
    )

    @model_validator(mode="after")
    def _release_matches_policy(self) -> "PipelineDelivery":
        if self.policy == "scheduled":
            if not self.release_cron:
                raise ValueError("a scheduled release needs release_cron")
            try:
                CronTrigger.from_crontab(self.release_cron)
            except ValueError as exc:
                raise ValueError(f"invalid five-field cron expression {self.release_cron!r}: {exc}") from exc
        elif self.release_cron:
            raise ValueError("release_cron only applies when policy is scheduled")
        return self


class PipelineSpec(BaseModel):
    """What an operator declares: a source, a destination, an aggregation and a delivery."""

    id: str
    name: str | None = None
    source: PipelineSource
    destination: PipelineDestination
    aggregation: PipelineAggregation = Field(default_factory=PipelineAggregation)
    delivery: PipelineDelivery = Field(default_factory=PipelineDelivery)

    @field_validator("id")
    @classmethod
    def _segment_safe(cls, value: str) -> str:
        if not is_segment_safe_id(value):
            raise ValueError("id must start with a letter or digit and carry only letters, digits, '.', '_' or '-'")
        return value


class Check(BaseModel):
    """One validation step and its verdict."""

    id: str
    status: Literal["pass", "fail", "skip"]
    message: str


class ValidationResult(BaseModel):
    """The outcome of validating a pipeline, kept on its record."""

    valid: bool
    checked_at: str
    period_type: str | None = Field(
        default=None, description="The DHIS2 period the pipeline exports, from the dataset cadence."
    )
    checks: list[Check]


class DryRunResult(BaseModel):
    """The outcome of a dry run: the payload summary and DHIS2's own validation of it."""

    ran_at: str
    start: str
    end: str
    values: int = 0
    org_units: int = 0
    periods: list[str] = Field(default_factory=list)
    sample: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether DHIS2 validated every value: a dry-run outcome and no conflicts."""
        report = self.report or {}
        return (
            self.error is None and bool(report) and report.get("outcome") == "dry_run" and not report.get("conflicts")
        )


class PipelineRun(BaseModel):
    """One deliberate run: a batch job through the pipeline's export, then its delivery."""

    job_id: str
    mode: Literal["dry_run", "live"]
    start: str
    end: str
    submitted_at: str
    idempotency_key: str
    delivery_job_id: str | None = None
    delivery_status_url: str | None = None
    delivered_at: str | None = None
    error: str | None = None


class PipelineRecord(BaseModel):
    """A stored pipeline with its latest validation, dry run and runs."""

    spec: PipelineSpec
    created_at: str
    validation: ValidationResult | None = None
    dry_run: DryRunResult | None = None
    runs: list[PipelineRun] = Field(default_factory=list, description="Newest first.")
    model_config = ConfigDict(extra="forbid")
