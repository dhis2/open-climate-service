"""load_features -- load a registered feature collection as a GeoJSON FeatureCollection (CLIM-926)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from open_climate_service.features import services as feature_services
from open_climate_service.features import store
from open_climate_service.features.templates import get_feature_template
from open_climate_service.process import process
from open_climate_service.shared.crs import canonical_crs_code, validate_crs_code
from open_climate_service.shared.provenance import record_source

if TYPE_CHECKING:
    import geopandas as gpd

_BBOX_KEYS = ("west", "south", "east", "north")


@process(
    summary="Load a registered feature collection",
    parameters={
        "id": {"description": "Feature collection id, as registered under GET /features."},
        "spatial_extent": {"description": "Bounding box filter: {west, south, east, north, crs}."},
        "version": {
            "description": "Optional record timestamp that pins this read to the submitted collection version.",
            "optional": True,
        },
    },
)
def load_features(id: str, spatial_extent: Any = None, version: str | None = None) -> dict[str, Any]:
    """Load a registered feature collection as a GeoJSON FeatureCollection, reprojected to WGS 84.

    A fast local read of what is already registered, never a fetch -- materializing a collection
    from its provider is the separate, deliberate action `refresh_feature_collection_from_provider`
    performs. Because this never fetches or writes, it behaves identically on a read-only instance
    and a writable one, exactly like `load_collection`: it serves a registered collection, and
    refuses outright, with no other state, the one nothing has ever been registered for.

    GeoJSON has no CRS of its own -- RFC 7946 fixes it to WGS 84 -- so a collection stored in a
    projected CRS is reprojected here before being handed to any downstream process such as
    `aggregate_spatial`. This is the permanent output contract of `load_features`, not a
    temporary shim: every caller gets WGS 84 coordinates regardless of the collection's native
    storage CRS.

    Each feature's `id_property` value is re-stamped onto the feature's top-level `id`, because
    `aggregate_spatial` reads its geometry labels from there rather than from `properties`.

    The parameter remains named `id` because that is the public openEO process parameter used
    by process graphs.
    """
    if get_feature_template(id) is None:
        raise ValueError(f"load_features: '{id}' does not name a known feature collection")

    record = feature_services.registered_collections().get(id)
    if record is None:
        raise ValueError(
            f"load_features: feature collection '{id}' is declared but has never been ingested; "
            "refresh it before loading it"
        )
    actual_version = record.created_at.isoformat()
    if version is not None and version != actual_version:
        raise ValueError(
            f"load_features: feature collection {id!r} changed after this job was submitted "
            f"(expected {version}, current {actual_version})"
        )
    detail = record.features
    if detail is None:  # pragma: no cover -- registered_collections() already filters on this
        raise ValueError(f"load_features: '{id}' is not a feature collection")

    bbox, bbox_crs = _parse_spatial_extent(spatial_extent)
    frame = store.read_feature_collection(
        record,
        bbox=bbox,
        bbox_crs=bbox_crs,
        max_unqualified_read=None,
    )
    record_source(id, record)

    if canonical_crs_code(detail.crs) != store.WGS84:
        frame = frame.to_crs(store.WGS84)

    return _to_labeled_geojson(frame, id_property=detail.id_property)


def _parse_spatial_extent(spatial_extent: Any) -> tuple[tuple[float, float, float, float] | None, str]:
    """Return (bbox, bbox_crs) from an openEO spatial_extent object, or (None, WGS84) for none."""
    if spatial_extent is None:
        return None, store.WGS84
    if not isinstance(spatial_extent, dict):
        raise ValueError(f"load_features: spatial_extent must be an object, got {type(spatial_extent).__name__}")
    try:
        west, south, east, north = (float(spatial_extent[key]) for key in _BBOX_KEYS)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"load_features: spatial_extent must declare west/south/east/north: {exc}") from exc
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("load_features: spatial_extent coordinates must be finite numbers")
    if west >= east or south >= north:
        raise ValueError("load_features: spatial_extent must satisfy west < east and south < north")
    crs = spatial_extent.get("crs") or store.WGS84
    try:
        canonical_crs = validate_crs_code(crs)
    except ValueError as exc:
        raise ValueError(f"load_features: spatial_extent has an invalid CRS: {crs!r}") from exc
    return (west, south, east, north), canonical_crs


def _to_labeled_geojson(frame: gpd.GeoDataFrame, *, id_property: str) -> dict[str, Any]:
    """Convert a GeoDataFrame to a GeoJSON FeatureCollection, promoting id_property to top-level id.

    `aggregate_spatial._parse_geometries` reads its geometry labels from each feature's top-level
    `id`, not from `properties[id_property]` -- the opposite of the convention the feature store
    itself uses for identity (`shared.features.validate_feature_ids`). Re-stamping here is what
    lets a loaded collection feed straight into `aggregate_spatial` with meaningful labels instead
    of sequential integers.
    """
    collection: dict[str, Any] = json.loads(frame.to_json())
    for feature in collection.get("features", []):
        properties = feature.get("properties")
        if isinstance(properties, dict) and id_property in properties:
            feature["id"] = properties[id_property]
    return collection
