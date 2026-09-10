"""openEO /collections endpoint — unified openEO + STAC response."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request

from open_climate_service.shared.urls import absolute_base
from open_climate_service.stac import services as stac_services

logger = logging.getLogger(__name__)


def _normalize_cube_dimensions(collection: dict[str, Any]) -> dict[str, Any]:
    """Normalize cube:dimensions to openEO conventions.

    - Renames the temporal dimension key to "t" (openEO standard)
    - Adds a "bands" dimension listing each published variable
    """
    dimensions = collection.get("cube:dimensions")
    if not isinstance(dimensions, dict):
        return collection

    new_dims: dict[str, Any] = {}
    for key, value in dimensions.items():
        if isinstance(value, dict) and value.get("type") == "temporal":
            new_dims["t"] = value
        else:
            new_dims[key] = value

    variables = collection.get("cube:variables")
    if isinstance(variables, dict) and variables:
        band_names = [k for k, v in variables.items() if isinstance(v, dict) and v.get("type") in ("data", None)]
        if band_names:
            new_dims["bands"] = {"type": "bands", "values": band_names}

    return {**collection, "cube:dimensions": new_dims}


def _rewrite_collection_links(collection: dict[str, Any], request: Request) -> dict[str, Any]:
    """Replace /stac/collections links with /collections links."""
    base_url = absolute_base(request)
    links = collection.get("links", [])
    rewritten: list[dict[str, Any]] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        href = link.get("href", "")
        if isinstance(href, str):
            href = href.replace(f"{base_url}/stac/collections", f"{base_url}/collections")
            href = href.replace("/stac/catalog.json", "/stac")
        rewritten.append({**link, "href": href})
    return {**collection, "links": rewritten}


def list_collections(request: Request) -> dict[str, Any]:
    """Return the openEO /collections response (openEO + STAC compatible)."""
    eligible = stac_services._eligible_artifacts_by_dataset()
    collections = []
    for dataset_id in eligible:
        try:
            col = stac_services.build_collection(dataset_id, request)
            col = _rewrite_collection_links(col, request)
            col = _normalize_cube_dimensions(col)
            collections.append(col)
        except HTTPException as exc:
            logger.warning(
                "Skipping collection '%s' from openEO listing: %s",
                dataset_id,
                exc.detail,
            )
            continue

    base_url = absolute_base(request)
    return {
        "collections": collections,
        "links": [
            {"rel": "self", "href": f"{base_url}/collections", "type": "application/json"},
            {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
        ],
    }


def get_collection(dataset_id: str, request: Request) -> dict[str, Any]:
    """Return one openEO/STAC collection."""
    collection = stac_services.build_collection(dataset_id, request)
    collection = _rewrite_collection_links(collection, request)
    return _normalize_cube_dimensions(collection)
