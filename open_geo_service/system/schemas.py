"""Pydantic response models."""

from enum import StrEnum

from pydantic import BaseModel


class StatusMessage(BaseModel):
    """Simple status message response."""

    message: str


class Status(StrEnum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthStatus(BaseModel):
    """Health check response."""

    status: Status


class Link(BaseModel):
    """Hypermedia link."""

    href: str
    rel: str
    title: str


class RootResponse(BaseModel):
    """Root endpoint response with navigation links."""

    message: str
    links: list[Link]


class AppInfo(BaseModel):
    """Application version and environment info."""

    app_version: str
    python_version: str
    uvicorn_version: str
    # Advertised so a client can tell whether writes are possible before attempting one.
    read_only: bool = False
