"""Pydantic schemas for ingestion, dataset, and sync APIs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError, field_validator

from open_climate_service.shared.licences import STAC_LICENSE_OTHER


class ArtifactFormat(StrEnum):
    """Supported stored artifact formats."""

    ZARR = "zarr"
    NETCDF = "netcdf"
    ICECHUNK = "icechunk"


class PublicationStatus(StrEnum):
    """Publication lifecycle states."""

    UNPUBLISHED = "unpublished"
    PUBLISHED = "published"


class SyncKind(StrEnum):
    """Supported sync planning modes declared by dataset templates."""

    TEMPORAL = "temporal"
    RELEASE = "release"
    STATIC = "static"


class SyncAction(StrEnum):
    """Planner-selected sync action.

    APPEND means Open Climate Service updates an existing managed dataset by writing only
    the missing periods needed to reach the planned target end by extending the
    committed artifact store.
    """

    REMATERIALIZE = "rematerialize"
    APPEND = "append"
    NO_OP = "no_op"
    NOT_SYNCABLE = "not_syncable"


class CoverageSpatial(BaseModel):
    """Spatial extent summary."""

    xmin: float = Field(description="Minimum longitude of the covered spatial extent.")
    ymin: float = Field(description="Minimum latitude of the covered spatial extent.")
    xmax: float = Field(description="Maximum longitude of the covered spatial extent.")
    ymax: float = Field(description="Maximum latitude of the covered spatial extent.")


class CoverageTemporal(BaseModel):
    """Temporal extent summary.

    ``start``/``end`` are ``None`` for a non-temporal (ordinal) dataset such as a
    day-of-year climatology, which has no datetime extent.
    """

    start: str | None = Field(default=None, description="First covered time period in dataset-native string form.")
    end: str | None = Field(default=None, description="Last covered time period in dataset-native string form.")


class ArtifactCoverage(BaseModel):
    """Artifact coverage metadata."""

    spatial: CoverageSpatial = Field(description="Covered spatial extent of the managed dataset.")
    spatial_wgs84: CoverageSpatial | None = Field(
        default=None,
        description="Spatial extent in WGS84. None for WGS84 instances or legacy records.",
    )
    temporal: CoverageTemporal = Field(description="Covered temporal extent of the managed dataset.")


class ArtifactRequestScope(BaseModel):
    """Original request parameters used to create an artifact."""

    start: str | None = Field(
        default=None,
        description="Requested start period (None for a non-temporal dataset, e.g. a climatology).",
    )
    end: str | None = Field(default=None, description="Requested end period for the ingestion or sync operation.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Requested bounding box when the artifact was created.",
    )


class ArtifactPublication(BaseModel):
    """Publication metadata for an artifact."""

    status: PublicationStatus = PublicationStatus.UNPUBLISHED
    collection_id: str | None = None
    published_at: datetime | None = None


OCS_AUTHORITY = "ocs"
"""Version authority for a release Open Climate Service defines itself.

A constant rather than a literal at each site, so the path that stamps a derived dataset's
release and the planner that compares one cannot disagree about its spelling.
"""

AUTHORITY_PATTERN = r"^[a-z0-9]+([._-][a-z0-9]+)*$"
"""Shape of a version authority: a stable machine identifier, not a display label.

Lowercase so two templates cannot name the same authority differently and have the planner
read that as a release change; separators so a narrower authority (`dhis2.national-hmis`)
stays expressible without a schema change. Display names live on `source` and `providers`.
"""

AUTHORITY_MAX_LENGTH = 64
"""Length bound for both halves of a release identity, applied wherever one is built."""


class ArtifactVersion(BaseModel):
    """A logical release identity: what the release is, and whose scheme names it.

    The meaningful identity is the pair. A bare "1.0" or "R2025A" says nothing on its own —
    it is `worldpop:R2025A` or `ocs:1.0` that identifies a release — so one namespaced
    concept is carried here rather than parallel `source_version` / `dataset_version` /
    `release_version` fields that would each mean something slightly different.

    Distinct from the other two identities on a record. `artifact_id` answers "which exact
    materialization is this?" and every artifact has one; this answers "which logical
    release is this?" and only a versioned dataset has one. How an artifact was produced is
    provenance, and a derived artifact does not inherit a version from its inputs — an
    openEO result carries `version=None` unless OCS deliberately releases it, with its
    inputs recorded as provenance rather than folded into this field.
    """

    value: str = Field(
        min_length=1,
        max_length=AUTHORITY_MAX_LENGTH,
        description=(
            "The release identifier itself, verbatim as the authority publishes it "
            "(for example 'R2025A' or '2026-08-19.0'). Opaque to OCS: not parsed, ordered, "
            "or normalized, because its syntax belongs to the authority. Surrounding "
            "whitespace is rejected rather than trimmed, so 'verbatim' stays true and a "
            "padded declaration cannot compare unequal to the same release declared cleanly."
        ),
    )
    authority: str = Field(
        pattern=AUTHORITY_PATTERN,
        max_length=AUTHORITY_MAX_LENGTH,
        description=(
            "Whose versioning scheme gives `value` its meaning, as a stable machine "
            "identifier ('worldpop', 'overture', 'ocs') — never a display label. Compared "
            "exactly, and part of the identity, so changing it renames every release under "
            "it; choose it once."
        ),
    )

    @field_validator("value")
    @classmethod
    def _value_is_present_and_unpadded(cls, value: str) -> str:
        # Enforced on the model rather than only at template registration, so every
        # construction path — a loaded record, an API payload, a future provider — gets the
        # same invariant. Rejecting rather than stripping keeps the field honest about being
        # verbatim, and stops " R2025A " from reading as a different release than "R2025A".
        if not value.strip():
            raise ValueError("release version value must not be blank")
        if value != value.strip():
            raise ValueError(
                f"release version value {value!r} has leading or trailing whitespace; "
                "it is stored verbatim, so declare it without padding"
            )
        return value


def parse_declared_artifact_version(declared: object) -> ArtifactVersion | None:
    """Read a template's declared `sync.version` into a release identity.

    The single interpreter of that declaration. Template registration, materialization and
    sync planning all come through here, so the version stamped on an artifact and the
    version the planner compares against cannot drift apart — and registration's promise to
    reject a malformed declaration is the same check the other two would have made.

    Returns None when nothing is declared. Raises ValueError when something is declared but
    is not a usable identity, because silently reading a malformed declaration as "no
    version" would disable release-change detection for a dataset that asked for it.
    """
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError(
            f"invalid sync.version {declared!r}; it must declare a mapping of 'value' "
            "(the upstream identifier) and 'authority' (whose scheme names it)"
        )
    missing = [key for key in ("value", "authority") if key not in declared]
    if missing:
        raise ValueError(f"sync.version is missing {', '.join(missing)}; both halves are required")
    try:
        return ArtifactVersion(value=declared["value"], authority=declared["authority"])
    except ValidationError as exc:
        raise ValueError(_describe_version_error(exc)) from exc


def _describe_version_error(exc: ValidationError) -> str:
    """Turn a pydantic failure on ArtifactVersion into one template-author-facing line."""
    problems = []
    for error in exc.errors():
        field = str(error["loc"][0]) if error["loc"] else "value"
        if field == "authority":
            problems.append(
                f"invalid sync.version.authority {error.get('input')!r}. It is a stable machine "
                "identifier, not a display label: lowercase letters and digits, separated by "
                f"'.', '-' or '_' (for example 'worldpop'), at most {AUTHORITY_MAX_LENGTH} "
                "characters. Display names belong on 'source' and 'providers'. It is part of "
                "the release identity, so changing it later renames every release under it."
            )
        else:
            problems.append(f"invalid sync.version.{field}: {error['msg'].removeprefix('Value error, ')}")
    return "; ".join(problems)


class ArtifactRecord(BaseModel):
    """Stored artifact metadata."""

    artifact_id: str
    dataset_id: str
    source_dataset_id: str | None = None
    dataset_name: str
    variable: str
    period_type: str | None = None
    version: ArtifactVersion | None = Field(
        default=None,
        description=(
            "Logical release identity, independent of period_type. Distinct from "
            "coverage.temporal.end: a period is a point on this dataset's own temporal "
            "axis, while a version is a release identifier that is not always expressible "
            "as a period (for example a build-dated '2026-08-19.0'). None for a dataset "
            "with no release identity, where sync planning still uses coverage.temporal.end."
        ),
    )
    format: ArtifactFormat
    path: str | None = None
    asset_paths: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    request_scope: ArtifactRequestScope
    coverage: ArtifactCoverage
    created_at: datetime
    publication: ArtifactPublication = Field(default_factory=ArtifactPublication)


class CreateIngestionRequest(BaseModel):
    """Request payload for creating or updating a managed dataset."""

    dataset_id: str = Field(description="Source dataset template id from the Open Climate Service registry.")
    start: str | None = Field(
        default=None,
        description=(
            "Start period to ingest. Required for historical datasets. May be omitted for a "
            "dataset declaring 'temporal_direction: future' (e.g. a forecast), where it means "
            "'from now' — a fixed date would be stale by the next day."
        ),
    )
    end: str | None = Field(default=None, description="Optional end period to ingest.")
    overwrite: bool = Field(
        default=False,
        description="Whether to force regeneration of an existing matching artifact.",
    )
    publish: bool = Field(
        default=True,
        description="Whether to publish the resulting dataset.",
    )


class ArtifactListResponse(BaseModel):
    """Envelope response for internal artifact records."""

    kind: str = Field(
        default="ArtifactList",
        description="Self-describing envelope type for this collection response.",
        examples=["ArtifactList"],
    )
    items: list[ArtifactRecord] = Field(
        default_factory=list,
        description="Internal artifact records managed by this Open Climate Service instance.",
    )


class DatasetAccessLink(BaseModel):
    """Access link for a managed dataset."""

    href: str = Field(description="Relative API path for this dataset access mode.")
    rel: str = Field(description="Relationship type of the link.")
    title: str = Field(description="Human-readable label for the link target.")


class DatasetPublication(BaseModel):
    """Public publication summary for a managed dataset."""

    status: PublicationStatus = Field(description="Publication state of the dataset in the OGC-facing layer.")
    published_at: datetime | None = Field(default=None, description="Timestamp when the dataset was last published.")


class DatasetRecord(BaseModel):
    """Native FastAPI view of a managed dataset."""

    dataset_id: str = Field(description="Stable public identifier for the managed dataset.")
    source_dataset_id: str = Field(description="Dataset template id from which this managed dataset was created.")
    dataset_name: str = Field(description="Full display name of the dataset.")
    short_name: str | None = Field(default=None, description="Short display name of the dataset.")
    description: str | None = Field(
        default=None,
        description=(
            "Longer prose description from the dataset template, where the caveats about what the values mean belong."
        ),
    )
    variable: str = Field(description="Primary raster variable stored in the dataset.")
    period_type: str = Field(description="Temporal period type of the dataset, for example daily or yearly.")
    units: str | None = Field(default=None, description="Units of the primary variable.")
    resolution: str | None = Field(default=None, description="Native spatial resolution summary.")
    source: str | None = Field(default=None, description="Upstream source name.")
    source_url: str | None = Field(default=None, description="Upstream source documentation URL.")
    license: str = Field(
        default=STAC_LICENSE_OTHER,
        description=(
            "SPDX identifier for the dataset's licence, or 'other' when the licence has no "
            "SPDX identifier. Never absent: an undeclared licence reports 'other' rather than "
            "something that reads as permissive."
        ),
    )
    license_url: str | None = Field(default=None, description="URL of the licence text, when known.")
    extent: ArtifactCoverage = Field(description="Current covered spatial and temporal extent of the dataset.")
    last_updated: datetime = Field(
        description="Timestamp when Open Climate Service last materialized or updated the dataset."
    )
    links: list[DatasetAccessLink] = Field(
        default_factory=list,
        description="Available API access links for this managed dataset.",
    )
    publication: DatasetPublication = Field(description="Publication summary for this managed dataset.")


class DatasetVersionRecord(BaseModel):
    """Version summary as exposed from a dataset detail view."""

    created_at: datetime = Field(description="Timestamp when this dataset version was created.")
    format: ArtifactFormat = Field(description="Stored format of this dataset version.")
    coverage: ArtifactCoverage = Field(description="Covered spatial and temporal extent for this dataset version.")
    request_scope: ArtifactRequestScope | None = Field(
        default=None,
        description="Original request scope that produced this version, when available.",
    )


class DatasetDetailRecord(DatasetRecord):
    """Detailed native FastAPI view of a managed dataset."""

    versions: list[DatasetVersionRecord] = Field(
        description="Slim version history derived from internal artifact records."
    )


class IngestionResponse(BaseModel):
    """Response returned after creating or looking up a managed dataset via ingestion."""

    ingestion_id: str = Field(description="Identifier of the ingestion event.")
    status: str = Field(description="Execution status of the ingestion request.")
    dataset: DatasetRecord | None = Field(default=None, description="Managed dataset summary. None for async requests.")


class IngestionListResponse(BaseModel):
    """Envelope response for ingestion run records."""

    kind: str = Field(
        default="IngestionList",
        description="Self-describing envelope type for this collection response.",
        examples=["IngestionList"],
    )
    items: list[IngestionResponse] = Field(
        default_factory=list,
        description="Ingestion run records available in this Open Climate Service instance.",
    )


class DatasetListResponse(BaseModel):
    """Envelope response for managed datasets."""

    kind: str = Field(
        default="DatasetList",
        description="Self-describing envelope type for this collection response.",
        examples=["DatasetList"],
    )
    items: list[DatasetRecord] = Field(
        default_factory=list,
        description="Managed datasets available in this Open Climate Service instance.",
        examples=[
            [
                {
                    "dataset_id": "chirps3_precipitation_daily_sle",
                    "source_dataset_id": "chirps3_precipitation_daily",
                    "dataset_name": "Total precipitation (CHIRPS3)",
                    "short_name": "Total precipitation",
                    "variable": "precip",
                    "period_type": "daily",
                    "units": "mm",
                    "resolution": "5 km x 5 km",
                    "source": "CHIRPS v3",
                    "source_url": "https://www.chc.ucsb.edu/data/chirps3",
                    "extent": {
                        "spatial": {"xmin": -13.5, "ymin": 6.9, "xmax": -10.1, "ymax": 10.0},
                        "temporal": {"start": "2024-01-01", "end": "2024-01-31"},
                    },
                    "last_updated": "2026-03-27T08:40:24.344473Z",
                    "links": [
                        {
                            "href": "/datasets/chirps3_precipitation_daily_sle",
                            "rel": "self",
                            "title": "Dataset detail",
                        },
                        {
                            "href": "/zarr/chirps3_precipitation_daily_sle",
                            "rel": "zarr",
                            "title": "Zarr store",
                        },
                    ],
                    "publication": {"status": "published", "published_at": "2026-03-27T08:40:24.346357Z"},
                }
            ]
        ],
    )


class SyncDatasetRequest(BaseModel):
    """Request payload for syncing a managed dataset forward."""

    end: str | None = Field(default=None, description="Optional end period to sync through.")
    publish: bool = Field(default=True, description="Whether to publish the resulting dataset version.")


class SyncDetail(BaseModel):
    """Structured planner output for one managed dataset sync decision.

    This record exists so callers can see both the operational outcome and the
    reasoning that led to it without needing to infer that logic from status
    strings alone.
    """

    source_dataset_id: str = Field(description="Source dataset template id used to plan the sync.")
    sync_kind: SyncKind = Field(description="Sync planning mode declared by the dataset template.")
    action: SyncAction = Field(description="Planner-selected sync action.")
    reason: str = Field(description="Stable machine-readable reason for the selected action.")
    message: str = Field(description="Human-readable summary of the planned sync outcome.")
    current_start: str | None = Field(
        default=None,
        description="First period currently covered by the managed dataset before sync.",
    )
    current_end: str | None = Field(
        default=None,
        description="Last period currently covered by the managed dataset before sync.",
    )
    target_end: str | None = Field(
        default=None,
        description="Resolved target period after applying request defaults and availability constraints.",
    )
    target_end_source: str = Field(
        description=(
            "Where target_end came from, for example request, default_today, "
            "request_clamped_by_availability, default_today_clamped_by_availability, or current_coverage."
        ),
    )
    current_version: ArtifactVersion | None = Field(
        default=None,
        description=(
            "Release identity of the currently materialized artifact, read from "
            "ArtifactRecord.version. Populated only for sync_kind=release when the "
            "artifact carries one; current_end remains the period-domain value used for "
            "availability queries even when this is set."
        ),
    )
    target_version: ArtifactVersion | None = Field(
        default=None,
        description="Release identity the template declares as current, for sync_kind=release.",
    )
    delta_start: str | None = Field(
        default=None,
        description="First missing period planned for append execution, when applicable.",
    )
    delta_end: str | None = Field(
        default=None,
        description="Last missing period planned for append execution, when applicable.",
    )
    periods: list[str] | None = Field(
        default=None,
        description=(
            "Period list already fetched from the plugin during planning. "
            "Passed to the orchestrator to avoid a second periods() call at execution time."
        ),
        exclude=True,
    )


class SyncResponse(BaseModel):
    """Public response returned after planning and optionally running a sync."""

    sync_id: str | None = Field(
        default=None,
        description="Identifier of the sync-created version when a new version was written.",
    )
    status: str = Field(description="Execution status, for example completed or up_to_date.")
    message: str | None = Field(default=None, description="Human-readable explanation of the sync outcome.")
    dataset: DatasetDetailRecord | None = Field(
        default=None, description="Current dataset detail. None for async requests."
    )
    sync_detail: SyncDetail | None = Field(
        default=None,
        description="Planner output describing how Open Climate Service interpreted the sync request.",
    )
