"""openEO /collections endpoint — unified openEO + STAC response."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request

from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord
from open_climate_service.shared.urls import absolute_base
from open_climate_service.stac import services as stac_services

logger = logging.getLogger(__name__)

DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.3.0/schema.json"
_GEOJSON_GEOMETRY_TYPES = (
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
)
GEOMETRY_DIMENSION = "geometry"
"""The vector dimension of a feature collection, named as `aggregate_spatial` names its output's."""


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


def _add_geometry_dimension(collection: dict[str, Any], artifact: ArtifactRecord) -> dict[str, Any]:
    """Describe a feature collection as an openEO vector cube: one `geometry` dimension.

    The STAC document already carries the feature properties as `table:columns`; this adds the
    `cube:dimensions` an openEO client reads, using the vector dimension of the openEO API and
    the STAC datacube extension. The bbox and reference system describe what `load_collection`
    returns, which is WGS 84 whatever CRS the file is stored in. Feature identifiers are not
    listed as `values`: a collection can hold far more features than a document should carry.
    """
    from open_climate_service.features import store as feature_store

    spatial = artifact.coverage.spatial_wgs84 or artifact.coverage.spatial
    dimension: dict[str, Any] = {
        "type": "geometry",
        "axes": ["x", "y"],
        "bbox": [spatial.xmin, spatial.ymin, spatial.xmax, spatial.ymax],
        "reference_system": 4326,
    }
    try:
        geometry_types = feature_store.stored_geometry_types(artifact)
    except Exception:  # an unreadable footer leaves the types out, which means "mixed"
        logger.warning("Could not read the geometry types of '%s'", artifact.dataset_id, exc_info=True)
        geometry_types = []
    # GeoParquet may say "Polygon Z"; the openEO API lists only the GeoJSON type names.
    declared = {str(name).split(" ")[0] for name in geometry_types}
    if declared and declared <= set(_GEOJSON_GEOMETRY_TYPES):
        dimension["geometry_types"] = [name for name in _GEOJSON_GEOMETRY_TYPES if name in declared]
    extensions = [*collection.get("stac_extensions", [])]
    if DATACUBE_EXTENSION not in extensions:
        extensions.append(DATACUBE_EXTENSION)
    return {**collection, "stac_extensions": extensions, "cube:dimensions": {GEOMETRY_DIMENSION: dimension}}


def _openeo_collection(dataset_id: str, artifact: ArtifactRecord, request: Request) -> dict[str, Any]:
    """The STAC collection with openEO links and the cube dimensions `load_collection` returns."""
    # Built from `artifact`, not looked up again, so one record describes the whole document.
    collection = _rewrite_collection_links(
        stac_services.build_collection_for_artifact(dataset_id, artifact, request), request
    )
    if artifact.format == ArtifactFormat.GEOPARQUET:
        return _add_geometry_dimension(collection, artifact)
    return _normalize_cube_dimensions(collection)


def _eligible_artifacts_by_dataset() -> dict[str, ArtifactRecord]:
    """Return the datasets openEO advertises: published rasters and feature collections.

    Both are loadable: a raster as a datacube, a feature collection as a vector cube
    (CLIM-1326). openEO asks its own gate rather than reading STAC's set, so a format STAC can
    describe but `load_collection` cannot load stays out of here.
    """
    return ingestion_services.openeo_collection_artifacts_by_dataset()


def list_collections(request: Request) -> dict[str, Any]:
    """Return the openEO /collections response (openEO + STAC compatible)."""
    eligible = _eligible_artifacts_by_dataset()
    collections = []
    for dataset_id, artifact in eligible.items():
        try:
            collections.append(_openeo_collection(dataset_id, artifact, request))
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
    # Gate the detail route on openEO's own set too. Without this it inherits STAC's gate
    # through build_collection, so /collections/{id} would keep serving anything STAC
    # serves even once the listing above stops advertising it.
    artifact = _eligible_artifacts_by_dataset().get(dataset_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Collection '{dataset_id}' not found")
    return _openeo_collection(dataset_id, artifact, request)
