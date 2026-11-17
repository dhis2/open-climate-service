"""Extent registry backed by GEOSERVICE_CONFIG."""

from typing import Any

from fastapi import HTTPException

from open_geo_service import config as api_config


def get_extent() -> dict[str, Any] | None:
    """Return the configured extent for this Open Geo Service instance, or None if not set."""
    extent = api_config.get_config().get("extent")
    if extent is None:
        return None
    if not isinstance(extent, dict):
        raise ValueError("extent in GEOSERVICE_CONFIG must be a mapping")
    if not isinstance(extent.get("bbox"), list) or len(extent["bbox"]) != 4:
        raise ValueError("extent.bbox in GEOSERVICE_CONFIG must be a list of four numbers [xmin, ymin, xmax, ymax]")
    return extent


def get_extent_or_404() -> dict[str, Any]:
    """Return the configured extent or raise 404 if none is configured."""
    extent = get_extent()
    if extent is not None:
        return extent
    raise HTTPException(status_code=404, detail="No extent is configured for this Open Geo Service instance")
