"""Pydantic schemas for extent discovery APIs."""

from pydantic import BaseModel, Field


class ExtentRecord(BaseModel):
    """Configured spatial extent for this Open Geo Service instance."""

    name: str | None = Field(default=None, description="Human-readable name of the extent.")
    description: str | None = Field(default=None, description="Optional descriptive text for the extent.")
    bbox: tuple[float, float, float, float] = Field(
        description="Configured bounding box for the extent as xmin, ymin, xmax, ymax."
    )
