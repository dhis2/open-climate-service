"""The `/mcp` endpoint and its tools, driven over HTTP as an MCP client would (CLIM-851).

The raster is the vector-collection tests' synthetic one: two months on a 1-degree grid, 1.0 in
the west half and 3.0 in the east, next to two published regions covering those halves. So every
value a tool returns is known exactly.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactRecord
from open_climate_service.main import app
from open_climate_service.mcp_server import tools
from open_climate_service.openeo import execution
from open_climate_service.stac import services as stac_services
from tests.test_openeo_vector_collections import (
    _rain_cube,  # pyright: ignore[reportPrivateUsage]
    _raster_record,  # pyright: ignore[reportPrivateUsage]
    isolated_store,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    regions,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

# What the STAC builder would derive from the synthetic store, which it cannot open.
_RAIN_COLLECTION = {
    "id": "rain",
    "title": "Rain",
    "description": "Synthetic rain.",
    "assets": {"zarr": {"href": "http://testserver/zarr/rain"}},
    "renders": {"default": {"open_climate_service:variable": "tp"}},
    "cube:variables": {"tp": {"unit": "mm/d"}},
    "cube:dimensions": {
        "t": {"type": "temporal", "extent": ["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"], "step": "P1M"},
        "x": {"type": "spatial"},
        "y": {"type": "spatial"},
    },
}


def _with_config(**overrides: Any) -> Iterator[None]:
    original = api_config.get_config
    api_config.get_config = lambda: {**original(), **overrides}  # type: ignore[assignment]
    try:
        yield
    finally:
        api_config.get_config = original  # type: ignore[assignment]


@pytest.fixture
def mcp_enabled() -> Iterator[None]:
    yield from _with_config(mcp=True)


@pytest.fixture
def rain(monkeypatch: pytest.MonkeyPatch, regions: ArtifactRecord) -> None:  # noqa: F811
    """A published raster next to the published regions; only its store and STAC document are faked."""
    record = _raster_record()
    published = {**ingestion_services.openeo_collection_artifacts_by_dataset(), "rain": record}
    monkeypatch.setattr(ingestion_services, "openeo_collection_artifacts_by_dataset", lambda: published)
    monkeypatch.setattr(execution, "_open_artifact", _rain_cube)

    eligible = stac_services._eligible_artifacts_by_dataset  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(stac_services, "_eligible_artifacts_by_dataset", lambda: {**eligible(), "rain": record})
    build_collection = stac_services.build_collection
    monkeypatch.setattr(
        stac_services,
        "build_collection",
        lambda dataset_id, request: (
            dict(_RAIN_COLLECTION) if dataset_id == "rain" else build_collection(dataset_id, request)
        ),
    )


@pytest.fixture
def mcp(mcp_enabled: None) -> TestClient:
    # No lifespan: the endpoint needs nothing from it.
    return TestClient(app)


Rpc = Callable[..., dict[str, Any]]


def _rpc(client: TestClient, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post(
        "/mcp", headers=_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _call(client: TestClient, tool: str, **arguments: Any) -> dict[str, Any]:
    """A tool's result: its structured content, or the error text when the tool failed."""
    result = _rpc(client, "tools/call", {"name": tool, "arguments": arguments})["result"]
    if result.get("isError"):
        return {"error": result["content"][0]["text"]}
    return result["structuredContent"]


def _initialize(client: TestClient) -> dict[str, Any]:
    return _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    )


# --- the endpoint ---------------------------------------------------------------------------


def test_mcp_is_off_unless_the_instance_enables_it() -> None:
    client = TestClient(app)
    response = client.post("/mcp", headers=_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert response.status_code == 404
    assert "mcp: true" in response.json()["detail"]


def test_an_enabled_instance_answers_as_an_mcp_server(mcp: TestClient) -> None:
    assert _initialize(mcp)["result"]["serverInfo"]["name"] == "Open Climate Service"

    listed = _rpc(mcp, "tools/list")["result"]["tools"]

    assert sorted(tool["name"] for tool in listed) == [
        "get_timeseries",
        "get_zonal_statistics",
        "list_areas",
        "list_datasets",
    ]
    assert all(tool["annotations"]["readOnlyHint"] for tool in listed)
    assert all("outputSchema" in tool for tool in listed)


@pytest.fixture
def read_only_mcp() -> Iterator[None]:
    yield from _with_config(mcp=True, read_only=True)


def test_a_read_only_instance_still_serves_mcp(read_only_mcp: None, rain: None) -> None:
    """MCP sends reads as POSTs; the tools' own calls are what the read-only rule governs."""
    client = TestClient(app)
    areas = _call(client, "list_areas", collection_id="regions")

    assert [area["id"] for area in areas["areas"]] == ["EAST", "WEST"]


# --- the tools ------------------------------------------------------------------------------


def test_list_datasets_reports_rasters_and_area_collections(mcp: TestClient, rain: None) -> None:
    datasets = {dataset["id"]: dataset for dataset in _call(mcp, "list_datasets")["result"]}

    assert datasets["rain"]["kind"] == "raster"
    assert datasets["rain"]["unit"] == "mm/d"
    assert datasets["rain"]["time_step"] == "P1M"
    assert datasets["rain"]["dataset_url"] == "http://testserver/datasets/rain"
    assert datasets["rain"]["map_url"] == "http://testserver/map?dataset=rain"
    assert datasets["regions"]["kind"] == "areas"
    assert datasets["regions"]["area_count"] == 2
    # From the dataset records, which the STAC documents do not carry.
    assert datasets["regions"]["last_updated"]


def test_zonal_statistics_are_computed_per_area_and_period(mcp: TestClient, rain: None) -> None:
    result = _call(mcp, "get_zonal_statistics", dataset_id="rain", collection_id="regions")

    values = {(row["period"], row["area_id"]): row["value"] for row in result["rows"]}
    assert values == {
        ("2025-01-01", "WEST"): 1.0,
        ("2025-01-01", "EAST"): 3.0,
        ("2025-02-01", "WEST"): 1.0,
        ("2025-02-01", "EAST"): 3.0,
    }
    assert result["unit"] == "mm/d"
    assert result["period_dimension"] == "t"
    # The graph returned is the one that ran, so the answer can be reproduced without the tool.
    assert result["process_graph"]["zonal"]["process_id"] == "aggregate_spatial_weighted"
    assert result["process_graph"]["areas"]["arguments"] == {"id": "regions"}


def test_zonal_statistics_narrow_to_the_areas_asked_for(mcp: TestClient, rain: None) -> None:
    result = _call(
        mcp, "get_zonal_statistics", dataset_id="rain", collection_id="regions", area_ids=["EAST"], start="2025-02-01"
    )

    assert [(row["period"], row["area_id"], row["value"]) for row in result["rows"]] == [("2025-02-01", "EAST", 3.0)]
    assert result["process_graph"]["load"]["arguments"]["temporal_extent"] == ["2025-02-01", None]


def test_an_unknown_area_points_the_caller_at_list_areas(mcp: TestClient, rain: None) -> None:
    result = _call(mcp, "get_zonal_statistics", dataset_id="rain", collection_id="regions", area_ids=["NORTH"])

    assert "No area NORTH in regions" in result["error"]
    assert "list_areas" in result["error"]


def test_a_timeseries_is_the_value_of_the_cell_the_point_falls_in(mcp: TestClient, rain: None) -> None:
    west = _call(mcp, "get_timeseries", dataset_id="rain", lon=0.5, lat=1.5)
    east = _call(mcp, "get_timeseries", dataset_id="rain", lon=3.2, lat=2.7)

    assert [(row["period"], row["value"]) for row in west["rows"]] == [("2025-01-01", 1.0), ("2025-02-01", 1.0)]
    assert [row["value"] for row in east["rows"]] == [3.0, 3.0]


def test_a_result_too_large_to_read_is_refused_with_how_to_narrow_it(
    mcp: TestClient, rain: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "MAX_ROWS", 3)

    result = _call(mcp, "get_zonal_statistics", dataset_id="rain", collection_id="regions")

    assert "4 rows" in result["error"]
    assert "Narrow the period" in result["error"]


def test_an_unknown_dataset_is_a_tool_error_not_a_failure(mcp: TestClient, rain: None) -> None:
    result = _call(mcp, "get_timeseries", dataset_id="nope", lon=0.5, lat=0.5)

    assert "list_datasets" in result["error"]


def test_links_carry_the_mount_prefix(
    mcp_enabled: None, rain: None, mounted_client_factory: Callable[..., TestClient]
) -> None:
    """Tools reach the API at the address the call came in on, prefix included."""
    client = mounted_client_factory("/ocs")
    result = client.post(
        "/mcp",
        headers=_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_timeseries", "arguments": {"dataset_id": "rain", "lon": 0.5, "lat": 0.5}},
        },
    ).json()["result"]["structuredContent"]

    assert result["dataset_url"] == "http://testserver/ocs/datasets/rain"
    assert result["map_url"] == "http://testserver/ocs/map?dataset=rain"
    assert [row["value"] for row in result["rows"]] == [1.0, 1.0]
