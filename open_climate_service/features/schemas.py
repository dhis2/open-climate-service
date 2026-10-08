"""Response schemas for the feature collection API."""

from datetime import datetime

from pydantic import BaseModel, Field

from open_climate_service.ingestions.schemas import ArtifactCoverage, ArtifactVersion
from open_climate_service.shared.licences import STAC_LICENSE_OTHER


class FeatureCollectionRecord(BaseModel):
    """One registered feature collection, as `GET /features` reports it.

    Two kinds of fact meet here. Record-derived ones — count, geometry types, CRS, coverage,
    version — describe what was actually stored, and are answerable for every collection.
    Template-derived ones — licence, attribution, description — describe what the collection
    *is*, and are only as good as the template that declared it.
    """

    id: str = Field(description="Stable identifier of the collection, and the name it is loaded by.")
    name: str = Field(description="Display name of the collection.")
    description: str | None = Field(
        default=None,
        description="Prose from the collection's template, where caveats about what the features mean belong.",
    )
    license: str = Field(
        default=STAC_LICENSE_OTHER,
        description=(
            "SPDX identifier for the collection's licence, or 'other' when it has none. Never "
            "absent: an undeclared licence reports 'other' rather than something that reads as "
            "permissive."
        ),
    )
    license_url: str | None = Field(default=None, description="URL of the licence text, when known.")
    attribution: str | None = Field(
        default=None,
        description=(
            "Required attribution for the collection. Not decoration: Overture divisions carry an "
            "obligation from the OpenStreetMap data they incorporate."
        ),
    )
    id_property: str = Field(
        description=(
            "Property each feature is identified by. The value that becomes the location column "
            "of a DHIS2 or CHAP export, which is why it can never be absent."
        )
    )
    feature_count: int = Field(description="Number of features stored in the collection.")
    geometry_types: list[str] = Field(
        default_factory=list,
        description=(
            "Geometry types the stored file declares, for example ['Polygon']. Empty when the "
            "file declares none, which is 'not stated' rather than 'no geometry'."
        ),
    )
    primary_geometry: str = Field(description="Name of the geometry column in the stored file.")
    crs: str = Field(description="Coordinate reference system the geometry is stored in.")
    version: ArtifactVersion | None = Field(
        default=None,
        description="Release identity of the stored collection, when its source publishes one.",
    )
    extent: ArtifactCoverage = Field(description="Spatial extent of the stored collection.")
    last_updated: datetime = Field(description="When this collection was last written.")


class FeatureCollectionListResponse(BaseModel):
    """Envelope response for registered feature collections."""

    kind: str = Field(
        default="FeatureCollectionList",
        description="Self-describing envelope type for this collection response.",
        examples=["FeatureCollectionList"],
    )
    items: list[FeatureCollectionRecord] = Field(
        default_factory=list,
        description=(
            "Feature collections registered in this Open Climate Service instance. A GeoParquet "
            "file in the store directory that nothing registered is not one of these: the listing "
            "reads records, never the filesystem."
        ),
    )
