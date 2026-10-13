"""MCP tools and resources over the OCS API: the first, read-only slice of CLIM-851.

Every tool calls OCS's own HTTP routes in-process, never its services, so a call is subject to
the read-only middleware and the same rules as any outside client. The numbers come from a
process graph OCS executed, and each result carries that graph and the dataset's links, so an
answer can be reproduced without the assistant.
"""

import csv
import io
import json
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq
import pyarrow.types as pa_types
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.requests import Request

from open_climate_service.mcp_server.endpoint import current_scope

# A result is read by a model, so its size is bounded by what fits usefully in a context window
# rather than by what the server can compute. A daily series for a few months over a hundred
# districts is about this size.
MAX_ROWS = 2000

# Half the side of the square a point query aggregates over, in degrees: far smaller than any
# grid cell, so the square falls inside one cell and its weighted mean is that cell's value.
# A square rather than a point because the weighted aggregation takes polygons only.
_POINT_HALF_SIDE = 1e-5

Statistic = Literal["mean", "sum", "min", "max", "median"]

INSTRUCTIONS = """\
Climate and earth observation data held by this Open Climate Service instance, for one country.

Start with list_datasets to see what the instance holds: raster datasets (climate variables over
time) and area collections (districts or other administrative units). To answer a question about
a place, call list_areas on an area collection, pick the area by name, then call
get_zonal_statistics with its id. For a single location, use get_timeseries.

Every value is computed by the instance, not estimated. Each result includes the process graph
that produced it and links to the dataset; cite the dataset link in answers.
"""


class DatasetSummary(BaseModel):
    """One published dataset, as list_datasets reports it."""

    id: str
    title: str
    kind: Literal["raster", "areas"] = Field(
        description="raster: values on a grid, queried with get_timeseries or get_zonal_statistics. "
        "areas: a collection of areas, listed with list_areas."
    )
    description: str | None = None
    variable: str | None = Field(default=None, description="Name of the raster variable.")
    unit: str | None = None
    dimensions: list[str] = Field(
        default_factory=list,
        description="Non-spatial dimensions of a raster: 't' for time, or e.g. 'month' for a climate normal.",
    )
    time_start: str | None = None
    time_end: str | None = None
    time_step: str | None = Field(default=None, description="ISO 8601 duration between time steps, e.g. P1M.")
    area_count: int | None = Field(default=None, description="Number of areas in an area collection.")
    last_updated: str | None = Field(default=None, description="When the instance last updated this dataset.")
    dataset_url: str
    stac_url: str
    map_url: str = Field(description="The dataset in the instance's map viewer, to show the user.")


class Area(BaseModel):
    """One area of a collection."""

    id: str
    name: str
    properties: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, description="Other attributes, such as the parent region or level."
    )


class AreaList(BaseModel):
    """The areas of one collection, sorted by name."""

    collection_id: str
    areas: list[Area]


class TimeseriesRow(BaseModel):
    """One period of a point time series."""

    period: str
    value: float | None


class ZonalRow(BaseModel):
    """One period of one area."""

    period: str
    area_id: str
    area_name: str
    value: float | None


class _Provenance(BaseModel):
    dataset_id: str
    variable: str
    unit: str | None
    period_dimension: str | None = Field(
        description="The dimension `period` comes from: 't' for time, or e.g. 'month' for a climate normal."
    )
    process_graph: dict[str, Any] = Field(description="The openEO process graph OCS executed, for reproduction.")
    dataset_url: str
    stac_url: str
    map_url: str = Field(description="The dataset in the instance's map viewer, to show the user.")


class TimeseriesResult(_Provenance):
    """A point time series, with the process graph that produced it."""

    lon: float
    lat: float
    rows: list[TimeseriesRow]


class ZonalResult(_Provenance):
    """Statistics per area and period, with the process graph that produced them."""

    collection_id: str
    statistic: Statistic
    rows: list[ZonalRow]


@asynccontextmanager
async def _ocs() -> AsyncGenerator[httpx.AsyncClient]:
    """A client for this instance's API, reached in-process at the address the call came in on."""
    scope = current_scope.get()
    prefix = str(scope.get("root_path", "")).rstrip("/")
    url = Request(scope).url
    transport = httpx.ASGITransport(app=scope["app"], root_path=prefix)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"{url.scheme}://{url.netloc}{prefix}", timeout=None
    ) as client:
        yield client


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else response.text


async def _get_json(client: httpx.AsyncClient, path: str) -> Any:
    response = await client.get(path)
    if response.status_code == 404:
        raise ToolError(f"Not found: {path}. Use list_datasets to see what this instance holds.")
    if not response.is_success:
        raise ToolError(f"OCS refused {path} ({response.status_code}): {_error_detail(response)}")
    return response.json()


def _links(client: httpx.AsyncClient, dataset_id: str) -> dict[str, str]:
    base = str(client.base_url).rstrip("/")
    return {
        "dataset_url": f"{base}/datasets/{dataset_id}",
        "stac_url": f"{base}/stac/collections/{dataset_id}",
        "map_url": f"{base}/map?dataset={dataset_id}",
    }


def _raster_variable(collection: dict[str, Any]) -> tuple[str, str | None]:
    """The variable a raster collection renders, and its unit."""
    variables: dict[str, Any] = collection.get("cube:variables") or {}
    render = (collection.get("renders") or {}).get("default") or {}
    variable = render.get("open_climate_service:variable") or next(iter(variables), "")
    return variable, (variables.get(variable) or {}).get("unit")


def _summarise(collection: dict[str, Any], links: dict[str, str]) -> DatasetSummary:
    summary = DatasetSummary(
        id=collection["id"],
        title=collection.get("title") or collection["id"],
        kind="raster" if "zarr" in (collection.get("assets") or {}) else "areas",
        description=collection.get("description"),
        dataset_url=links["dataset_url"],
        stac_url=links["stac_url"],
        map_url=links["map_url"],
    )
    if summary.kind == "areas":
        summary.area_count = collection.get("table:row_count")
        return summary
    summary.variable, summary.unit = _raster_variable(collection)
    dimensions = {
        key: dim
        for key, dim in (collection.get("cube:dimensions") or {}).items()
        if dim.get("type") not in ("spatial", "bands", "geometry")
    }
    summary.dimensions = list(dimensions)
    time = dimensions.get("t") or {}
    summary.time_start, summary.time_end = (time.get("extent") or [None, None])[:2]
    step = time.get("step")
    summary.time_step = step if isinstance(step, str) else None
    return summary


async def _catalogue(client: httpx.AsyncClient) -> list[DatasetSummary]:
    catalog = await _get_json(client, "/stac")
    ids = [
        str(link["href"]).rstrip("/").rsplit("/", 1)[-1]
        for link in catalog.get("links", [])
        if link.get("rel") == "child"
    ]
    # The records, not the STAC documents, know when each dataset was last updated.
    records = await _get_json(client, "/datasets?f=json")
    last_updated = {item["dataset_id"]: item.get("last_updated") for item in records.get("items", [])}
    summaries = []
    for dataset_id in ids:
        collection = await _get_json(client, f"/stac/collections/{dataset_id}")
        summary = _summarise(collection, _links(client, dataset_id))
        summary.last_updated = last_updated.get(dataset_id)
        summaries.append(summary)
    return summaries


async def _areas(client: httpx.AsyncClient, collection_id: str) -> list[Area]:
    record = await _get_json(client, f"/features/{collection_id}")
    response = await client.get(f"/features/{collection_id}/data.parquet")
    if not response.is_success:
        raise ToolError(f"OCS refused the areas of {collection_id} ({response.status_code}): {_error_detail(response)}")
    table = pq.read_table(io.BytesIO(response.content))
    id_property = record["id_property"]
    names = table.column_names
    name_property = next((name for name in ("name", "displayName") if name in names), id_property)
    scalar = [
        name
        for name, field_type in zip(names, table.schema.types, strict=True)
        if name not in (id_property, name_property, record.get("primary_geometry"))
        and not pa_types.is_nested(field_type)  # structs such as Overture's bbox
        and not pa_types.is_binary(field_type)
    ]
    rows = table.select([id_property, name_property, *scalar]).to_pylist()
    areas = [
        Area(
            id=str(row[id_property]),
            name=str(row[name_property]),
            properties={key: row[key] for key in scalar},
        )
        for row in rows
    ]
    return sorted(areas, key=lambda area: area.name)


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else value


async def _execute_table(
    client: httpx.AsyncClient, graph: dict[str, Any], variable: str
) -> tuple[str | None, list[dict[str, str]]]:
    """Run a process graph that saves CSV; return the period column's name and the rows."""
    response = await client.post("/result", json={"process": {"process_graph": graph}})
    if not response.is_success:
        raise ToolError(f"OCS could not compute this ({response.status_code}): {_error_detail(response)}")
    reader = csv.DictReader(io.StringIO(response.text))
    columns = reader.fieldnames or []
    value_column = variable if variable in columns else columns[-1]
    period_column = next((c for c in columns if c not in ("feature_id", value_column, "x", "y")), None)
    rows = [
        {
            "period": row[period_column] if period_column else "",
            "feature_id": row.get("feature_id", ""),
            "value": row[value_column],
        }
        for row in reader
    ]
    return period_column, rows


def _check_size(rows: int) -> None:
    if rows > MAX_ROWS:
        raise ToolError(
            f"That is {rows} rows, more than the {MAX_ROWS} a result may hold. "
            "Narrow the period with start and end, or ask for fewer areas."
        )


def _load(dataset_id: str, start: str | None, end: str | None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"id": dataset_id}
    if start or end:
        arguments["temporal_extent"] = [start, end]
    return {"process_id": "load_collection", "arguments": arguments}


def build_server() -> MCPServer:
    """The MCP server: catalogue, areas, and point and zonal statistics."""
    server = MCPServer(name="Open Climate Service", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

    @server.resource(
        "ocs://catalogue",
        name="catalogue",
        description="Every published dataset and area collection on this instance.",
        mime_type="application/json",
    )
    async def catalogue() -> str:
        async with _ocs() as client:
            return json.dumps([summary.model_dump() for summary in await _catalogue(client)])

    @server.tool(annotations=read_only)
    async def list_datasets() -> list[DatasetSummary]:
        """List the datasets this instance publishes.

        Rasters are climate variables over time, with their unit, variable and time range. Area
        collections are districts or other administrative units.
        """
        async with _ocs() as client:
            return await _catalogue(client)

    @server.tool(annotations=read_only)
    async def list_areas(collection_id: str) -> AreaList:
        """List the areas in an area collection, by name and id.

        Each comes with attributes such as its parent region. Pick an area by name here, then pass
        its id to get_zonal_statistics.
        """
        async with _ocs() as client:
            return AreaList(collection_id=collection_id, areas=await _areas(client, collection_id))

    @server.tool(annotations=read_only)
    async def get_timeseries(
        dataset_id: str,
        lon: float = Field(ge=-180, le=180, description="Longitude, WGS 84."),
        lat: float = Field(ge=-90, le=90, description="Latitude, WGS 84."),
        start: str | None = Field(default=None, description="First period, ISO 8601 date, e.g. 2024-01-01."),
        end: str | None = Field(default=None, description="Last period, ISO 8601 date."),
    ) -> TimeseriesResult:
        """Values of a raster dataset at one location over time.

        The value of the grid cell the point falls in, for each period between start and end.
        """
        async with _ocs() as client:
            collection = await _get_json(client, f"/stac/collections/{dataset_id}")
            variable, unit = _raster_variable(collection)
            e = _POINT_HALF_SIDE
            ring = [[lon - e, lat - e], [lon + e, lat - e], [lon + e, lat + e], [lon - e, lat + e], [lon - e, lat - e]]
            point = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
            graph = {
                "load": _load(dataset_id, start, end),
                "point": {
                    "process_id": "aggregate_spatial_weighted",
                    "arguments": {
                        "data": {"from_node": "load"},
                        "geometries": {"type": "FeatureCollection", "features": [point]},
                        "reducer": "mean",
                    },
                },
                "save": {
                    "process_id": "save_result",
                    "arguments": {"data": {"from_node": "point"}, "format": "CSV"},
                    "result": True,
                },
            }
            period_column, rows = await _execute_table(client, graph, variable)
            _check_size(len(rows))
            return TimeseriesResult(
                dataset_id=dataset_id,
                variable=variable,
                unit=unit,
                period_dimension=period_column,
                lon=lon,
                lat=lat,
                rows=[TimeseriesRow(period=row["period"], value=_number(row["value"])) for row in rows],
                process_graph=graph,
                **_links(client, dataset_id),
            )

    @server.tool(annotations=read_only)
    async def get_zonal_statistics(
        dataset_id: str,
        collection_id: str = Field(description="The area collection, as list_datasets names it."),
        area_ids: list[str] | None = Field(
            default=None, description="Ids from list_areas. Omit to get every area, e.g. to rank them."
        ),
        start: str | None = Field(default=None, description="First period, ISO 8601 date, e.g. 2024-01-01."),
        end: str | None = Field(default=None, description="Last period, ISO 8601 date."),
        statistic: Statistic = "mean",
    ) -> ZonalResult:
        """A statistic of a raster dataset over each area of a collection, per period.

        Covers each period between start and end. Area-weighted: a grid cell counts in proportion
        to how much of it lies in the area.
        """
        async with _ocs() as client:
            collection = await _get_json(client, f"/stac/collections/{dataset_id}")
            variable, unit = _raster_variable(collection)
            names = {area.id: area.name for area in await _areas(client, collection_id)}
            unknown = sorted(set(area_ids or []) - names.keys())
            if unknown:
                raise ToolError(f"No area {', '.join(unknown)} in {collection_id}. Use list_areas for the ids.")
            graph = {
                "load": _load(dataset_id, start, end),
                "areas": {"process_id": "load_collection", "arguments": {"id": collection_id}},
                "zonal": {
                    "process_id": "aggregate_spatial_weighted",
                    "arguments": {
                        "data": {"from_node": "load"},
                        "geometries": {"from_node": "areas"},
                        "reducer": statistic,
                    },
                },
                "save": {
                    "process_id": "save_result",
                    "arguments": {"data": {"from_node": "zonal"}, "format": "CSV"},
                    "result": True,
                },
            }
            period_column, rows = await _execute_table(client, graph, variable)
            if area_ids:
                wanted = set(area_ids)
                rows = [row for row in rows if row["feature_id"] in wanted]
            _check_size(len(rows))
            return ZonalResult(
                dataset_id=dataset_id,
                collection_id=collection_id,
                statistic=statistic,
                variable=variable,
                unit=unit,
                period_dimension=period_column,
                rows=[
                    ZonalRow(
                        period=row["period"],
                        area_id=row["feature_id"],
                        area_name=names.get(row["feature_id"], row["feature_id"]),
                        value=_number(row["value"]),
                    )
                    for row in rows
                ],
                process_graph=graph,
                **_links(client, dataset_id),
            )

    return server
