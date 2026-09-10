"""Pydantic schemas for ingestion, dataset, and sync APIs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

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


class ArtifactRecord(BaseModel):
    """Stored artifact metadata."""

    artifact_id: str
    dataset_id: str
    source_dataset_id: str | None = None
    dataset_name: str
    variable: str
    period_type: str | None = None
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
