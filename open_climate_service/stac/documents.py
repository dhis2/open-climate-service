"""README.md and AGENTS.md for the STAC catalog and each feature collection.

Linked from the STAC documents as `describedby` and `agents`, so a person or an agent can find
out what the data is and how to read it without knowing the API. Generated from the records and
templates the catalogue is already built from, so they never say anything the STAC does not.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from open_climate_service.features import store
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord
from open_climate_service.shared.licences import parse_licence
from open_climate_service.shared.urls import absolute_url, path_segment
from open_climate_service.stac import services


def catalog_readme(request: Request) -> str:
    """What the catalog holds: every published collection, with the licence each declares."""
    lines = [
        f"# {_instance_name()}",
        "",
        f"The published datasets of this instance, at {absolute_url(request, '/')}: rasters as "
        "GeoZarr stores and vector datasets as GeoParquet.",
        "",
        "Each collection declares its own licence in its STAC document, and its providers where they are known.",
        "",
        "## Collections",
        "",
    ]
    for dataset_id, artifact in services._eligible_artifacts_by_dataset().items():
        href = absolute_url(request, f"/stac/collections/{path_segment(dataset_id)}")
        kind = "vector" if artifact.format == ArtifactFormat.GEOPARQUET else "raster"
        lines.append(f"- [{artifact.dataset_name}]({href}) ({kind})")
    return "\n".join(lines) + "\n"


def catalog_agents(request: Request) -> str:
    """How an agent reads this catalog."""
    catalog = absolute_url(request, "/stac/catalog.json")
    return "\n".join(
        [
            f"# Reading the {_instance_name()} catalog",
            "",
            f"Start at {catalog}. Each `child` link is one collection.",
            "",
            "- A **raster** collection has a `zarr` asset: a Zarr v3 store with GeoZarr metadata. Open it "
            "with the asset's own `xarray:open_kwargs`, which name the full-resolution group of a "
            'multiscale store: `xr.open_zarr(asset["href"], **asset["xarray:open_kwargs"])`.',
            "- A **vector** collection has a `data` asset: one GeoParquet 1.1 file with a `bbox` covering "
            "column, readable with DuckDB, GeoPandas or any Parquet reader. Its `AGENTS.md` link says how.",
            "",
            "Every asset is served with range requests, so a reader fetches only what it needs.",
            "",
        ]
    )


def collection_readme(dataset_id: str, request: Request) -> str:
    """Title, description, licence and provenance of one feature collection."""
    artifact, template = _feature_collection(dataset_id)
    licence = parse_licence(template.get("license"))
    licence_text = licence.name or licence.identifier or licence.stac_license
    if licence.url:
        licence_text = f"[{licence_text}]({licence.url})"
    lines = [
        f"# {artifact.dataset_name}",
        "",
        services._feature_collection_description(dataset_id, artifact, template),
        "",
        "## Licence",
        "",
        licence_text,
    ]
    attribution = template.get("attribution")
    if isinstance(attribution, str) and attribution.strip():
        lines += ["", f"Attribution: {attribution.strip()}"]
    lines += ["", "## Provenance", ""]
    # The providers the STAC document publishes, this instance last as the one host, so the two
    # documents agree.
    for provider in services._feature_providers(template, request):
        name = f"[{provider['name']}]({provider['url']})" if provider.get("url") else provider["name"]
        roles = f" ({', '.join(provider['roles'])})" if provider.get("roles") else ""
        lines.append(f"- {name}{roles}")
    source_url = template.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        lines.append(f"- Source: {source_url.strip()}")
    if artifact.version is not None:
        lines.append(f"- Version: {artifact.version.value} ({artifact.version.authority})")
    lines.append(f"- Last refreshed: {artifact.created_at.isoformat()}")
    detail = artifact.features
    if detail is not None:
        types = ", ".join(store.stored_geometry_types(artifact)) or "unknown"
        lines += [
            "",
            "## Data",
            "",
            f"- {detail.feature_count} features, geometry types: {types}.",
            f"- Identifier property: `{detail.id_property}`.",
            f"- Stored in {detail.crs}.",
        ]
    return "\n".join(lines) + "\n"


def collection_agents(dataset_id: str, request: Request) -> str:
    """How an agent reads one feature collection: the file, its id column, a windowed query."""
    artifact, _template = _feature_collection(dataset_id)
    detail = artifact.features
    id_property = detail.id_property if detail is not None else "id"
    geometry = detail.primary_geometry if detail is not None else "geometry"
    crs = detail.crs if detail is not None else "EPSG:4326"
    # The covering bbox is in the stored CRS, which a collection can set to a projected one.
    units = "longitude and latitude" if crs == "EPSG:4326" else f"{crs} coordinates"
    data = absolute_url(request, f"/features/{path_segment(dataset_id)}/data.parquet")
    url_literal = data.replace("'", "''")
    return "\n".join(
        [
            f"# Reading {artifact.dataset_name}",
            "",
            f"The data is one GeoParquet 1.1 file: {data}",
            "",
            f"- One row per feature, identified by `{id_property}`; the geometry is in `{geometry}`.",
            "- A `bbox` covering column holds each feature's bounds, for filtering by area.",
            "",
            f"With DuckDB, reading only one area (replace the four bounds, in {units}):",
            "",
            "```sql",
            "INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;",
            f"SELECT {_sql_identifier(id_property)}, ST_AsText({_sql_identifier(geometry)}) FROM '{url_literal}'",
            "WHERE bbox.xmin < <east> AND bbox.xmax > <west> AND bbox.ymin < <north> AND bbox.ymax > <south>;",
            "```",
            "",
            f"With GeoPandas: `geopandas.read_parquet(path, bbox=(west, south, east, north))` on a downloaded "
            f"copy, with the bounds in {units}.",
            "",
            f"The licence, providers and extent are in the STAC collection: "
            f"{absolute_url(request, f'/stac/collections/{path_segment(dataset_id)}')}",
            "",
        ]
    )


def _feature_collection(dataset_id: str) -> tuple[ArtifactRecord, dict[str, Any]]:
    """The published feature collection's record and template, or 404 as the STAC route does."""
    artifact = services._eligible_artifacts_by_dataset().get(dataset_id)
    if artifact is None or artifact.format != ArtifactFormat.GEOPARQUET:
        raise HTTPException(status_code=404, detail=f"No published feature collection '{dataset_id}'")
    return artifact, feature_templates.get_feature_template(artifact.dataset_id) or {}


def _sql_identifier(name: str) -> str:
    """A column name quoted for SQL, so spaces, punctuation or a reserved word stay one name."""
    return '"' + name.replace('"', '""') + '"'


def _instance_name() -> str:
    from open_climate_service import config as api_config

    return api_config.get_name()
