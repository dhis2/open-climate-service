"""Pydantic schemas for ingestion, dataset, and sync APIs."""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from open_climate_service.shared.crs import canonical_crs_code
from open_climate_service.shared.licences import STAC_LICENSE_OTHER


class ArtifactFormat(StrEnum):
    """Supported stored artifact formats."""

    ZARR = "zarr"
    NETCDF = "netcdf"
    ICECHUNK = "icechunk"
    GEOPARQUET = "geoparquet"
    """A feature collection: rows of geometry and properties, not a datacube.

    The first format here that no raster reader opens, which is why the catalogue gates split
    in CLIM-1066 before this value existed. Every branch that dispatches on format has to
    answer for it rather than fall through to a Zarr path that would fail obscurely.

    A record carrying it is registered, published and listed under `/datasets`, and is in
    neither catalogue yet: openEO `/collections` refuses it permanently, since it is not a
    datacube, while STAC admits it in CLIM-1069 — together with the collection document and
    the `table` extension, so the catalogue never advertises a child it cannot serve.
    """


class DatasetItemType(StrEnum):
    """What a managed dataset holds, as the public `itemType` discriminator.

    The field name and the `feature` value are OGC API - Features Part 1, which defines
    `itemType` on the collection object as an "indicator about the type of the items in the
    collection (the default value is 'feature')". `coverage` is convention rather than
    conformance: OGC API - Coverages is a candidate draft and silent on the field. `/datasets`
    is OCS's own API, so borrowing the name buys consistency, not a promise that the rest of
    an OGC collection object is there.
    """

    FEATURE = "feature"
    COVERAGE = "coverage"


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


_CRS_CODE_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]*):([A-Za-z0-9_.+-]+)$")
"""Shape of a coordinate reference system code: an authority, then its code.

Structural only. It separates 'EPSG:4326' and 'ESRI:102008' from blank space, a bare label or
a sentence; whether the authority publishes that code, and whether the stored file agrees, is
the reader's question (CLIM-1068).
"""


def canonical_feature_crs(declared: object) -> str | None:
    """Return one spelling of a declared CRS, or None when it is not a CRS code at all.

    The single normalizer for a stored collection's CRS, shared by `FeatureDetail` and by the
    vector entry point that builds one. Two callers each doing "most of" this is how a record
    ends up holding `epsg:4326` while another holds `EPSG:4326`, and those compare unequal.

    Three steps. `canonical_crs_code` collapses the CRS84 aliases GeoJSON and GeoParquet both
    use, and prefixes a bare EPSG number. The authority is then uppercased, since authorities
    are conventionally uppercase and case is not part of their identity — `epsg` and `EPSG`
    name the same register. The code half is left exactly as declared: it is a token the
    authority defines, and this has no standing to recase it.
    """
    if not isinstance(declared, str) or not declared.strip():
        return None
    match = _CRS_CODE_PATTERN.match(canonical_crs_code(declared.strip()))
    if match is None:
        return None
    authority, code = match.groups()
    return f"{authority.upper()}:{code}"


class FeatureDetail(BaseModel):
    """What a feature collection must remember that has no home on the record.

    A nested submodel rather than three fields flattened onto `ArtifactRecord`, following
    `request_scope`, `coverage` and `publication`. Flattening would make every raster record
    carry `id_property: null` forever, and a free-form dict would leave these untyped and
    absent from the API contract.

    The shape is deliberate about `id_property` being required *inside* here: a feature record
    cannot exist without one, which neither alternative can guarantee. It is the one value
    whose loss does not raise — a missing or duplicated identifier pushes values against the
    wrong org unit silently, because DHIS2 keeps whichever value arrives last.

    Nothing temporal, which is correct for the static geometry in scope: org unit boundaries
    and facility points are versioned when they change (see `ArtifactRecord.version`) rather
    than timestamped per observation. CLIM-1095 decides what an observed, time-varying
    collection looks like; if it lands on a time column rather than a collection per
    acquisition, that shape arrives here and existing records need migrating.
    """

    model_config = ConfigDict(populate_by_name=True)

    id_property: str = Field(
        min_length=1,
        description=(
            "Name of the property each feature is identified by. Read from a feature's "
            "`properties`, which is what the openEO specification guarantees survives "
            "aggregation, rather than a top-level GeoJSON `id`, which the usual "
            "GeoJSON-to-frame conversion drops. Becomes the geometry-dimension label that the "
            "DHIS2 and CHAP exports use as their location column, so it must identify exactly "
            "one feature."
        ),
    )
    feature_count: int = Field(
        ge=0,
        description=(
            "Number of features in the stored collection. Recorded rather than counted on "
            "read, so 'why did yesterday cover 47 districts and today 48' is answerable from "
            "the record alone."
        ),
    )
    primary_geometry: str = Field(
        min_length=1,
        description=(
            "Name of the geometry *column* in the stored GeoParquet, not a geometry type — one "
            "column may hold points and polygons together. Named rather than assumed, because a "
            "collection may carry more than one geometry column, and this is what the STAC table "
            "extension publishes as `table:primary_geometry` (CLIM-1069)."
        ),
    )
    crs: str = Field(
        min_length=1,
        description=(
            "Coordinate reference system of the stored geometry, as a canonical authority code "
            "such as 'EPSG:4326'. Required, never defaulted at read time: ADR 0002 decision 9 "
            "makes an explicit CRS a property of every stored collection, because the "
            "alternative is a reader assuming WGS 84 and a process silently sampling a raster "
            "at the wrong places. A spatial read declares the CRS of its own bbox against this, "
            "and a process combining raster and vector reprojects to the raster's CRS before "
            "sampling. Canonicalized on the way in, so a CRS84 alias and a bare EPSG number "
            "are stored in the one spelling every consumer compares against."
        ),
    )

    @field_validator("id_property", "primary_geometry")
    @classmethod
    def _names_a_real_column(cls, value: str, info: ValidationInfo) -> str:
        """Reject a blank or padded column name rather than storing one that cannot match.

        Both fields name something in the stored file — a property key and a geometry column —
        and are compared exactly by whatever reads it. A padded ' orgUnitCode ' matches no
        property, and the failure is the quiet kind: every lookup misses, so the identifier is
        absent rather than wrong, and CLIM-1068's identity check reports a collection with no
        usable ids instead of a template with a stray space.

        Rejected rather than stripped, following `ArtifactVersion.value`. Stripping would
        silently accept a template whose declaration does not say what its author meant, and a
        record loaded from disk would then disagree with the template it came from.
        """
        if not value.strip():
            raise ValueError(f"{info.field_name} must name a column, not blank space")
        if value != value.strip():
            raise ValueError(
                f"{info.field_name} {value!r} has leading or trailing whitespace; it is compared "
                "exactly against the stored file, so declare it without padding"
            )
        return value

    @field_validator("crs")
    @classmethod
    def _is_a_canonical_authority_code(cls, value: str) -> str:
        """Canonicalize a declared CRS and reject one no consumer could resolve.

        Enforced on the model rather than only in `create_feature_artifact`, so a record loaded
        from `records.json` or built by a future provider gets the same guarantee: reading
        `features.crs` never yields blank space, a padded string, or a bare number.

        Canonicalized rather than rejected for padding, unlike the column names above: this
        field's value is an identifier of a known thing rather than a name that must match
        bytes in a file. Padding, a CRS84 alias, a bare EPSG number and a lowercase authority
        all resolve to the one spelling, because two records naming WGS 84 differently would
        otherwise compare unequal — which is the whole reason this normalizes at all.

        The check is structural — an authority and a code — not a registry lookup. Whether
        EPSG:1234567 exists, and whether it is the CRS the GeoParquet actually persists, is a
        question for the reader that opens the file (CLIM-1068); this rules out the inputs that
        are not a CRS at all.
        """
        canonical = canonical_feature_crs(value)
        if canonical is None:
            raise ValueError(
                f"invalid crs {value!r}; declare an authority code such as 'EPSG:4326' "
                "(a CRS84 alias, a bare EPSG number, or a lowercase authority is accepted "
                "and canonicalized)"
            )
        return canonical


class ArtifactRecord(BaseModel):
    """Stored artifact metadata."""

    artifact_id: str
    dataset_id: str
    source_dataset_id: str | None = None
    dataset_name: str
    variable: str | None = Field(
        default=None,
        description=(
            "Primary raster variable stored in the artifact. None for an artifact that is not "
            "a raster: a boundary set has properties, not a measured variable. Defaulted rather "
            "than required so a feature record does not have to name a variable it does not have."
        ),
    )
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
    features: FeatureDetail | None = Field(
        default=None,
        description=(
            "Feature collection detail. Present exactly when `format` is geoparquet, and None "
            "for every raster record, which is what keeps a raster from carrying three null "
            "vector fields forever."
        ),
    )

    @model_validator(mode="after")
    def _shape_matches_the_format(self) -> "ArtifactRecord":
        """Hold each format to its own required shape.

        `format` is the discriminator every branch in the codebase dispatches on, so the fields
        that only make sense for one kind of artifact are tied to it here rather than left to
        each construction site. Without this the model accepts three records that cannot be
        served: a GeoParquet with no `FeatureDetail` — which is a feature collection with no
        `id_property`, the exact loss `FeatureDetail` exists to prevent — a Zarr carrying
        feature detail, and a raster with no `variable`.

        Relaxing `variable` to optional (CLIM-1067) is what made the third of those reachable.
        It was relaxed for feature collections alone, so rasters keep the requirement they
        have always had, and this is where that stays true.
        """
        if self.format == ArtifactFormat.GEOPARQUET:
            if self.features is None:
                raise ValueError("a geoparquet artifact must carry a 'features' detail, including its id_property")
            # Not merely unused. Each of these three says "raster" to something that reads it:
            # `variable` and `variables` name measured data variables, and `period_type` is the
            # period axis the sync planner does arithmetic on. A feature record carrying any of
            # them would read as a coverage to anything scanning for one, and `itemType` would
            # then disagree with the record it is derived from.
            if self.variable is not None:
                raise ValueError("a geoparquet artifact has properties rather than a measured variable")
            if self.variables:
                raise ValueError("a geoparquet artifact has properties rather than data variables")
            if self.period_type is not None:
                # Not a statement about time-varying collections: CLIM-1095 gives an observed
                # collection a time column or a version, neither of which is a raster period.
                raise ValueError("a geoparquet artifact has no period axis, so it declares no period_type")
            return self
        if self.features is not None:
            raise ValueError(f"a {self.format} artifact is a raster and must not carry feature detail")
        if self.variable is None or not self.variable.strip():
            raise ValueError(f"a {self.format} artifact must name the raster variable it stores")
        return self


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

    model_config = ConfigDict(populate_by_name=True)

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
    item_type: DatasetItemType = Field(
        validation_alias="itemType",
        serialization_alias="itemType",
        description=(
            "What this dataset holds: 'feature' for a feature collection, 'coverage' for a "
            "raster. The one field that lets a client filter a listing without a request per "
            "row — `format` sits on the nested version record, which only the detail endpoint "
            "returns. Spelled in OGC API - Features' camelCase because the name is borrowed "
            "from that specification rather than invented here."
        ),
    )
    variable: str | None = Field(
        default=None,
        description=(
            "Primary raster variable stored in the dataset. None for a feature collection, "
            "which has properties rather than a measured variable."
        ),
    )
    period_type: str | None = Field(
        default=None,
        description=(
            "Temporal period type of the dataset, for example daily or yearly. None for a "
            "dataset with no temporal axis, such as a boundary set."
        ),
    )
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
                    "itemType": "coverage",
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
