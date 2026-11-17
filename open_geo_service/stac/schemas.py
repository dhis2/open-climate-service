"""Pydantic response models for the STAC surface."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StacLink(BaseModel):
    """Minimal STAC link representation."""

    rel: str
    href: str
    type: str | None = None
    title: str | None = None


class StacCatalogResponse(BaseModel):
    """Catalog response with flexible extension fields."""

    model_config = ConfigDict(extra="allow")

    stac_version: str
    type: Literal["Catalog"]
    id: str
    title: str
    description: str
    links: list[StacLink]


class StacCollectionResponse(BaseModel):
    """Collection response with flexible extension fields."""

    model_config = ConfigDict(extra="allow")

    stac_version: str
    type: Literal["Collection"]
    id: str
    title: str
    description: str
    links: list[StacLink]
    assets: dict[str, dict[str, Any]] = Field(default_factory=dict)
