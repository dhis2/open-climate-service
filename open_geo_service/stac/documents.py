"""README.md and AGENTS.md for the STAC catalog and each collection: raster, feature or table.

Linked from the STAC documents as `describedby` and `agents`, so a person or an agent can find
out what the data is and how to read it without knowing the API. Generated from the records and
templates the catalogue is already built from, so they never say anything the STAC does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request

from open_geo_service.data_registry.services import datasets as registry_datasets
from open_geo_service.features import store
from open_geo_service.features import templates as feature_templates
from open_geo_service.ingestions import services as ingestion_services
from open_geo_service.ingestions.schemas import ArtifactFormat, ArtifactRecord
from open_geo_service.shared.licences import parse_licence
from open_geo_service.shared.urls import absolute_url, path_segment
from open_geo_service.stac import services
from open_geo_service.stac.media_types import data_group_open_kwargs

if TYPE_CHECKING:
    from open_geo_service.tables.schemas import TableSchema


def catalog_readme(request: Request) -> str:
    """What the catalog holds: every published collection, with the licence each declares."""
    lines = [
        f"# {_instance_name()}",
        "",
        f"The published datasets of this instance, at {absolute_url(request, '/')}: rasters as "
        "GeoZarr stores, vector datasets as GeoParquet, and tables as Parquet.",
        "",
        "Each collection declares its own licence in its STAC document, and its providers where they are known.",
        "",
        "## Collections",
        "",
    ]
    for dataset_id, artifact in services._eligible_artifacts_by_dataset().items():
        href = absolute_url(request, services.collection_path(dataset_id))
        kind = _KINDS.get(artifact.format, "raster")
        lines.append(f"- [{artifact.dataset_name}]({href}) ({kind})")
    return "\n".join(lines) + "\n"


def catalog_agents(request: Request) -> str:
    """How an agent reads this catalog."""
    catalog = absolute_url(request, services.CATALOG_PATH)
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
            "- A **table** collection has a `data` asset: one Parquet file, whose columns `table:columns` "
            "describes. Its `README.md` link says how to read it and what it joins to.",
            "",
            "Every asset is served with range requests, so a reader fetches only what it needs.",
            "",
        ]
    )


def collection_readme(dataset_id: str, request: Request) -> str:
    """Title, description, licence, provenance and shape of one collection."""
    artifact, template = _published_collection(dataset_id)
    if artifact.format == ArtifactFormat.PARQUET:
        return _table_readme(dataset_id, artifact, template, request)
    raster = artifact.format != ArtifactFormat.GEOPARQUET
    licence = parse_licence(template.get("license"))
    licence_text = licence.name or licence.identifier or licence.stac_license
    if licence.url:
        licence_text = f"[{licence_text}]({licence.url})"
    description = (
        services._collection_description(dataset_id, artifact, template)
        if raster
        else services._feature_collection_description(dataset_id, artifact, template)
    )
    lines = [f"# {artifact.dataset_name}", "", description, "", "## Licence", "", licence_text]
    attribution = template.get("attribution")
    if isinstance(attribution, str) and attribution.strip():
        lines += ["", f"Attribution: {attribution.strip()}"]
    lines += ["", "## Provenance", ""]
    # The providers the STAC document publishes, a workflow's processor before this instance as
    # the one host, so the two documents agree.
    processors = [services._processor_provider(artifact, request)] if artifact.provenance is not None else []
    for provider in services._published_providers(template, request, processors):
        name = f"[{provider['name']}]({provider['url']})" if provider.get("url") else provider["name"]
        roles = f" ({', '.join(provider['roles'])})" if provider.get("roles") else ""
        lines.append(f"- {name}{roles}")
    source_url = template.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        lines.append(f"- Source: {source_url.strip()}")
    if artifact.version is not None:
        lines.append(f"- Version: {artifact.version.value} ({artifact.version.authority})")
    lines.append(f"- Last refreshed: {artifact.created_at.isoformat()}")
    lines += ["", "## Data", ""]
    if raster:
        units = template.get("units")
        unit_text = f" in {units}" if isinstance(units, str) and units.strip() else ""
        temporal = artifact.coverage.temporal
        lines.append(f"- Variable `{artifact.variable}`{unit_text}.")
        if temporal.start and temporal.end:
            period = f"{artifact.period_type.capitalize()} data" if artifact.period_type else "Data"
            lines.append(f"- {period} from {temporal.start} to {temporal.end}.")
        lines.append("- A Zarr v3 store with GeoZarr metadata; its `AGENTS.md` link says how to open it.")
    elif artifact.features is not None:
        detail = artifact.features
        types = ", ".join(store.stored_geometry_types(artifact)) or "unknown"
        lines += [
            f"- {detail.feature_count} features, geometry types: {types}.",
            f"- Identifier property: `{detail.id_property}`.",
            f"- Stored in {detail.crs}.",
        ]
    return "\n".join(lines) + "\n"


def collection_agents(dataset_id: str, request: Request) -> str:
    """How an agent reads one collection's data."""
    artifact, _template = _published_collection(dataset_id)
    if artifact.format == ArtifactFormat.PARQUET:
        return _table_agents(dataset_id, artifact, request)
    if artifact.format != ArtifactFormat.GEOPARQUET:
        return _raster_agents(dataset_id, artifact, request)
    return _feature_agents(dataset_id, artifact, request)


def _raster_agents(dataset_id: str, artifact: ArtifactRecord, request: Request) -> str:
    """How an agent reads one raster collection: the store, how to open it, the variable."""
    store_href = absolute_url(request, f"/zarr/{path_segment(dataset_id)}")
    # The same arguments the STAC `zarr` asset carries as `xarray:open_kwargs`.
    open_kwargs = {
        **services._zarr_open_kwargs(artifact),
        **data_group_open_kwargs(services._zarr_media_type(artifact)),
    }
    arguments = "".join(f", {name}={value!r}" for name, value in open_kwargs.items() if value is not None)
    return "\n".join(
        [
            f"# Reading {artifact.dataset_name}",
            "",
            f"The data is a Zarr v3 store with GeoZarr metadata: {store_href}",
            "",
            f"- The values are in the variable `{artifact.variable}`, over time, latitude and longitude "
            "or the store's own projected axes; the store's `spatial:` and `proj:` attributes give the grid.",
            "- Any Zarr v3 reader can open it, over HTTP, reading only the chunks it needs.",
            "",
            "With xarray:",
            "",
            "```python",
            "import xarray as xr",
            f'ds = xr.open_zarr("{store_href}"{arguments})',
            f'values = ds["{artifact.variable}"]',
            "```",
            "",
            f"The licence, providers and extent are in the STAC collection: "
            f"{absolute_url(request, services.collection_path(dataset_id))}",
            "",
        ]
    )


def _feature_agents(dataset_id: str, artifact: ArtifactRecord, request: Request) -> str:
    """How an agent reads one feature collection: the file, its id column, a windowed query."""
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
            f"{absolute_url(request, services.collection_path(dataset_id))}",
            "",
        ]
    )


def _table_readme(dataset_id: str, artifact: ArtifactRecord, template: dict[str, Any], request: Request) -> str:
    """A table's README: what it holds, its sources' licences and providers, its columns, and how to join it."""
    from open_geo_service.tables.sources import table_sources
    from open_geo_service.tables.store import period_label, schema_of, summarize

    sources = table_sources(artifact)
    summary = summarize(artifact)
    declared = schema_of(artifact)
    lines = [f"# {artifact.dataset_name}", "", sources.description(artifact)]
    lines += ["", "## Licence", ""]
    licences = [licence for licence in sources.licences if licence.identifier or licence.name]
    if not licences:
        lines.append("Not declared by the datasets it was computed from.")
    else:
        if len(sources.licences) > 1:
            lines += ["Computed from several datasets, so the licence of each applies:", ""]
        for licence in sources.licences:
            text = licence.name or licence.identifier or "Not declared"
            lines.append(f"- [{text}]({licence.url})" if licence.url else f"- {text}")
    lines += ["", "## Provenance", ""]
    for provider in services._published_providers(template, request, [services._processor_provider(artifact, request)]):
        name = f"[{provider['name']}]({provider['url']})" if provider.get("url") else provider["name"]
        roles = f" ({', '.join(provider['roles'])})" if provider.get("roles") else ""
        lines.append(f"- {name}{roles}")
    for recorded in ingestion_services.recorded_inputs(artifact):
        input_id = str(recorded["collection_id"])
        lines.append(f"- Computed from: [{input_id}]({absolute_url(request, services.collection_path(input_id))})")
    for source_url in services._source_pages(dataset_id, {}):
        lines.append(f"- Source: {source_url}")
    lines.append(f"- Last updated: {artifact.created_at.isoformat()}")

    data = absolute_url(request, f"/tables/{path_segment(dataset_id)}/data.parquet")
    lines += ["", "## Data", "", f"One Parquet file, {data}, of {declared.holds(artifact.period_type)}."]
    lines.append("")
    if summary.first_period is not None and summary.last_period is not None:
        period = f"{artifact.period_type.capitalize()} periods" if artifact.period_type else "Periods"
        first = period_label(summary.first_period, artifact.period_type)
        last = period_label(summary.last_period, artifact.period_type)
        lines.append(f"- {summary.row_count} rows. {period} from {first} to {last}.")
    if artifact.units:
        lines.append(f"- Values in {artifact.units}.")
    lines += ["", "| Column | Type | Description |", "| --- | --- | --- |"]
    for column in services._table_dataset_columns(artifact):
        lines.append(f"| `{column['name']}` | {column['type']} | {column.get('description', '')} |")

    join = _join_example(data, summary.collection_ids, declared, request)
    if join:
        collection, feature = declared.represented_by("collection"), declared.represented_by("feature")
        lines += [
            "",
            "## Joining to the features",
            "",
            f"`{collection}` names the feature collection a row reports on, and `{feature}` the feature, "
            "by the collection's identifier property. With DuckDB, the values with their features' geometry:",
            "",
            "```sql",
            *join,
            "```",
        ]
    return "\n".join(lines) + "\n"


def _join_example(data: str, collection_ids: list[str], declared: TableSchema, request: Request) -> list[str]:
    """A DuckDB query joining the table to the first published feature collection it reports on.

    None for a table without a column representing the feature: its rows report on no collection.
    """
    from open_geo_service.features.services import registered_collections

    collection, feature = declared.represented_by("collection"), declared.represented_by("feature")
    if collection is None or feature is None:
        return []
    shown = [
        f"t.{_sql_identifier(name)}"
        for name in (
            feature,
            declared.represented_by("feature_name"),
            declared.represented_by("period"),
            *declared.values,
        )
        if name
    ]
    order = ", ".join(f"t.{_sql_identifier(name)}" for name in (feature, declared.represented_by("period")) if name)
    published = services._eligible_artifacts_by_dataset()
    collections = registered_collections()
    for collection_id in collection_ids:
        record = collections.get(collection_id)
        if collection_id not in published or record is None or record.features is None:
            continue
        features = absolute_url(request, f"/features/{path_segment(collection_id)}/data.parquet")
        id_column = _sql_identifier(record.features.id_property)
        geometry = _sql_identifier(record.features.primary_geometry)
        return [
            "INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;",
            "SELECT " + ", ".join(shown) + ", ST_AsText(f." + geometry + ") AS geometry",
            "FROM '" + data.replace("'", "''") + "' AS t",
            "JOIN '" + features.replace("'", "''") + "' AS f",
            "  ON t." + _sql_identifier(collection) + " = '" + collection_id.replace("'", "''") + "'",
            "  AND CAST(f." + id_column + " AS VARCHAR) = t." + _sql_identifier(feature),
            "ORDER BY " + order + ";",
        ]
    return []


def _table_agents(dataset_id: str, artifact: ArtifactRecord, request: Request) -> str:
    """How an agent reads one table dataset: the file, its key columns, and the rows API."""
    from open_geo_service.tables.store import schema_of

    declared = schema_of(artifact)
    data = absolute_url(request, f"/tables/{path_segment(dataset_id)}/data.parquet")
    rows = absolute_url(request, f"/tables/{path_segment(dataset_id)}/rows")
    key = ", ".join(f"`{name}`" for name in declared.key)
    about = [f"- The rows are {declared.holds(artifact.period_type)}, one per key: {key}."]
    if declared.values:
        about.append(f"- The values are in {', '.join(f'`{name}`' for name in declared.values)}.")
    if (collection := declared.represented_by("collection")) and (feature := declared.represented_by("feature")):
        about.append(
            f"- Join it to the feature collection's GeoParquet on `{collection}` and `{feature}`; the README "
            "shows a DuckDB query."
        )
    return "\n".join(
        [
            f"# Reading {artifact.dataset_name}",
            "",
            f"The data is one Parquet file: {data}",
            "",
            *about,
            "",
            "With pandas:",
            "",
            "```python",
            "import pandas as pd",
            f'rows = pd.read_parquet("{data}")',
            "```",
            "",
            f"Or as JSON or CSV, filtered by feature, collection or period: {rows}?feature=<feature_id>"
            if declared.represented_by("feature")
            else f"Or as JSON or CSV: {rows}",
            "",
            f"The licence, providers and columns are in the STAC collection: "
            f"{absolute_url(request, services.collection_path(dataset_id))}",
            "",
        ]
    )


def _published_collection(dataset_id: str) -> tuple[ArtifactRecord, dict[str, Any]]:
    """The published collection's record and template, or 404 as the STAC route does."""
    artifact = services._eligible_artifacts_by_dataset().get(dataset_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"No published collection '{dataset_id}'")
    if artifact.format == ArtifactFormat.GEOPARQUET:
        return artifact, feature_templates.get_feature_template(artifact.dataset_id) or {}
    if artifact.format == ArtifactFormat.PARQUET:
        from open_geo_service.tables.sources import table_sources

        return artifact, table_sources(artifact).template()
    return artifact, registry_datasets.get_dataset(artifact.dataset_id) or {}


_KINDS = {ArtifactFormat.GEOPARQUET: "vector", ArtifactFormat.PARQUET: "table"}
"""How the catalog README names each collection's kind; any other format is a raster."""


def _sql_identifier(name: str) -> str:
    """A column name quoted for SQL, so spaces, punctuation or a reserved word stay one name."""
    return '"' + name.replace('"', '""') + '"'


def _instance_name() -> str:
    from open_geo_service import config as api_config

    return api_config.get_name()
