from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.client import (
    _FALLBACK_BASE_URL,
    ClimateService,
    _id_from_href,
    list_datasets,
    open_dataset,
)


def _make_catalog(hrefs: list[str]) -> dict:
    return {
        "links": [{"rel": "child", "href": href, "title": href.split("/")[-1]} for href in hrefs]
        + [{"rel": "root", "href": "http://localhost/stac/catalog.json"}]
    }


def _make_collection(zarr_href: str) -> dict:
    return {
        "assets": {
            "zarr": {
                "href": zarr_href,
                "xarray:open_kwargs": {"consolidated": True},
            }
        }
    }


def _make_response(
    json_body: dict | list | None = None,
    *,
    status_code: int = 200,
    content: bytes | None = None,
    content_type: str = "application/json",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.content = content if content is not None else b""
    resp.headers = {"content-type": content_type}
    resp.raise_for_status.return_value = None
    return resp


# ── _id_from_href ──────────────────────────────────────────────────────────────


def test_id_from_href_extracts_last_path_segment() -> None:
    assert _id_from_href("http://localhost/stac/collections/chirps3_daily_rwa") == "chirps3_daily_rwa"


def test_id_from_href_ignores_query_and_fragment() -> None:
    assert _id_from_href("http://localhost/stac/collections/ds?foo=bar#section") == "ds"


def test_id_from_href_strips_trailing_slash() -> None:
    assert _id_from_href("http://localhost/stac/collections/ds/") == "ds"


# ── module-level list_datasets ─────────────────────────────────────────────────


def test_list_datasets_returns_child_links() -> None:
    catalog = _make_catalog(["http://localhost/stac/collections/chirps3_precipitation_daily_rwa"])
    with patch("open_climate_service.client.httpx.get", return_value=_make_response(catalog)) as mock_get:
        result = list_datasets("http://localhost")

    mock_get.assert_called_once_with("http://localhost/stac/catalog.json", timeout=30)
    assert len(result) == 1
    assert result[0]["rel"] == "child"
    assert "chirps3" in result[0]["href"]


def test_list_datasets_returns_empty_for_no_children() -> None:
    catalog = {"links": [{"rel": "root", "href": "http://localhost/stac/catalog.json"}]}
    with patch("open_climate_service.client.httpx.get", return_value=_make_response(catalog)):
        result = list_datasets("http://localhost")

    assert result == []


def test_list_datasets_raises_on_http_error() -> None:
    with patch("open_climate_service.client.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        with pytest.raises(httpx.HTTPStatusError):
            list_datasets("http://localhost")


# ── module-level open_dataset ──────────────────────────────────────────────────


def test_open_dataset_fetches_collection_and_opens_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "test.zarr"
    ds = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2024-01-01", periods=2, freq="D"),
            "latitude": [3.0, 2.0, 1.0],
            "longitude": [10.0, 11.0, 12.0],
        },
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    collection = _make_collection(str(zarr_path))
    with patch("open_climate_service.client.httpx.get", return_value=_make_response(collection)):
        result = open_dataset("chirps3_precipitation_daily_rwa", base_url="http://localhost")

    try:
        assert "precip" in result.data_vars
        assert result.sizes["time"] == 2
        assert "latitude" in result.coords
        assert "longitude" in result.coords
    finally:
        result.close()


def test_open_dataset_raises_on_http_error() -> None:
    with patch("open_climate_service.client.httpx.get") as mock_get:
        mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        with pytest.raises(httpx.HTTPStatusError):
            open_dataset("nonexistent", base_url="http://localhost")


def test_open_dataset_uses_default_base_url() -> None:
    collection = _make_collection("/dev/null")
    with patch("open_climate_service.client.httpx.get", return_value=_make_response(collection)) as mock_get:
        with patch("xarray.open_zarr", return_value=MagicMock()):
            open_dataset("any_dataset")

    mock_get.assert_called_once_with(f"{_FALLBACK_BASE_URL}/stac/collections/any_dataset", timeout=30)


def test_open_dataset_uses_env_var_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arbitrary host:port — the point is that the env var is honored verbatim,
    # independent of the default base URL or port.
    monkeypatch.setenv("CLIMATE_SERVICE_BASE_URL", "http://env-host:4321")
    collection = _make_collection("/dev/null")
    with patch("open_climate_service.client.httpx.get", return_value=_make_response(collection)) as mock_get:
        with patch("xarray.open_zarr", return_value=MagicMock()):
            open_dataset("any_dataset")

    mock_get.assert_called_once_with("http://env-host:4321/stac/collections/any_dataset", timeout=30)


# ── lazy xarray import ─────────────────────────────────────────────────────────


def test_open_zarr_without_xarray_raises_helpful_error() -> None:
    from open_climate_service.client import _open_zarr

    # Simulate xarray not being installed (the base client has no xarray).
    with patch("open_climate_service.client.importlib.util.find_spec", return_value=None):
        with pytest.raises(ModuleNotFoundError, match=r"open-climate-service\[xarray\]"):
            _open_zarr("s3://bucket/x.zarr", {})


# ── ClimateService class ───────────────────────────────────────────────────────


def _make_service(base_url: str = "http://localhost") -> tuple[ClimateService, MagicMock]:
    """Return a ClimateService with a mocked internal httpx.Client."""
    service = ClimateService(base_url)
    mock_http = MagicMock()
    service._http = mock_http
    return service, mock_http


def test_datasets_uses_persistent_http_session() -> None:
    catalog = _make_catalog(["http://localhost/stac/collections/chirps3_precipitation_daily_rwa"])
    service, mock_http = _make_service()
    mock_http.get.return_value = _make_response(catalog)

    result = service.datasets()

    mock_http.get.assert_called_once_with("http://localhost/stac/catalog.json")
    assert len(result) == 1
    assert result[0]["rel"] == "child"
    assert result[0]["id"] == "chirps3_precipitation_daily_rwa"


def test_open_dataset_uses_persistent_http_session(tmp_path: Path) -> None:
    zarr_path = tmp_path / "test.zarr"
    ds = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2024-01-01", periods=2, freq="D"),
            "latitude": [3.0, 2.0, 1.0],
            "longitude": [10.0, 11.0, 12.0],
        },
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    service, mock_http = _make_service()
    mock_http.get.return_value = _make_response(_make_collection(str(zarr_path)))

    result = service.open_dataset("chirps3_precipitation_daily_rwa")
    try:
        assert "precip" in result.data_vars
        assert result.sizes["time"] == 2
    finally:
        result.close()

    mock_http.get.assert_called_once_with("http://localhost/stac/collections/chirps3_precipitation_daily_rwa")


def test_processes_returns_process_list() -> None:
    service, mock_http = _make_service()
    mock_http.get.return_value = _make_response({"processes": [{"id": "mean"}, {"id": "sum"}], "links": []})

    result = service.processes()

    mock_http.get.assert_called_once_with("http://localhost/processes")
    assert [p["id"] for p in result] == ["mean", "sum"]


def test_workflows_returns_workflow_list() -> None:
    service, mock_http = _make_service()
    mock_http.get.return_value = _make_response({"processes": [{"id": "aggregate_to_dhis2_json"}], "links": []})

    result = service.workflows()

    mock_http.get.assert_called_once_with("http://localhost/process_graphs")
    assert result[0]["id"] == "aggregate_to_dhis2_json"


def test_execute_returns_json_for_json_result() -> None:
    service, mock_http = _make_service()
    payload = {"dataValues": [{"orgUnit": "OU_A", "period": "202501", "value": "1.0"}]}
    mock_http.post.return_value = _make_response(payload, content_type="application/json")

    graph = {"agg": {"process_id": "aggregate_to_dhis2_json", "arguments": {}, "result": True}}
    result = service.execute(graph)

    args, kwargs = mock_http.post.call_args
    assert args[0] == "http://localhost/result"
    assert kwargs["json"] == {"process": {"process_graph": graph}}
    assert result == payload


def test_execute_unwraps_process_graph_key() -> None:
    service, mock_http = _make_service()
    mock_http.post.return_value = _make_response({"ok": True})

    nodes = {"agg": {"process_id": "x", "arguments": {}, "result": True}}
    service.execute({"process_graph": nodes})

    _, kwargs = mock_http.post.call_args
    assert kwargs["json"] == {"process": {"process_graph": nodes}}


def test_execute_returns_bytes_for_file_result() -> None:
    service, mock_http = _make_service()
    csv = b"time_period,location,hotspot\n202501,OU_A,0.1\n"
    mock_http.post.return_value = _make_response(content=csv, content_type="text/csv")

    result = service.execute({"r": {"process_id": "x", "result": True}})
    assert result == csv


def test_execute_writes_file_result_to_path(tmp_path: Path) -> None:
    service, mock_http = _make_service()
    csv = b"a,b\n1,2\n"
    mock_http.post.return_value = _make_response(content=csv, content_type="text/csv")

    out = tmp_path / "result.csv"
    result = service.execute({"r": {"process_id": "x", "result": True}}, path=out)

    assert result == out
    assert out.read_bytes() == csv


def test_context_manager_closes_http_session() -> None:
    with ClimateService("http://localhost") as service:
        mock_http = MagicMock()
        service._http = mock_http

    mock_http.close.assert_called_once()


def test_accepts_custom_timeout() -> None:
    with patch("open_climate_service.client.httpx.Client") as mock_cls:
        ClimateService("http://localhost", timeout=60)
    mock_cls.assert_called_once_with(timeout=60)


def test_strips_trailing_slash() -> None:
    service, mock_http = _make_service("http://localhost/")
    mock_http.get.return_value = _make_response(_make_catalog(["http://localhost/stac/collections/ds"]))
    service.datasets()
    mock_http.get.assert_called_once_with("http://localhost/stac/catalog.json")


# ── error detail (CLIM-1140) ───────────────────────────────────────────────────


def _service_on(http: httpx.Client, base_url: str = "http://testserver") -> ClimateService:
    service = ClimateService(base_url)
    service._http.close()
    service._http = http
    return service


def _service_replying(status_code: int, **response: Any) -> ClimateService:
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code, **response))
    return _service_on(httpx.Client(transport=transport), "http://ocs.example")


def test_an_unknown_collection_names_what_was_missing(client: TestClient) -> None:
    """The case from the review: a wrong dataset id in a process graph."""
    service = _service_on(client)
    graph = {
        "load": {
            "process_id": "load_collection",
            "arguments": {"id": "missing", "spatial_extent": None, "temporal_extent": None},
            "result": True,
        }
    }

    with pytest.raises(httpx.HTTPStatusError, match="Collection 'missing' not found") as raised:
        service.execute(graph)

    assert raised.value.response.status_code == 404
    assert "POST http://testserver/result" in str(raised.value)


def test_open_dataset_names_the_missing_collection(client: TestClient) -> None:
    with pytest.raises(httpx.HTTPStatusError, match="STAC collection 'missing' not found"):
        _service_on(client).open_dataset("missing")


def test_an_openeo_error_message_reaches_the_caller() -> None:
    service = _service_replying(400, json={"code": "ProcessGraphInvalid", "message": "Node 'x' has no process_id"})

    with pytest.raises(httpx.HTTPStatusError, match="Node 'x' has no process_id"):
        service.processes()


def test_a_list_of_validation_errors_reaches_the_caller() -> None:
    service = _service_replying(422, json={"detail": [{"loc": ["body", "process"], "msg": "Field required"}]})

    with pytest.raises(httpx.HTTPStatusError, match="Field required"):
        service.workflows()


def test_a_plain_text_error_body_reaches_the_caller() -> None:
    service = _service_replying(502, text="Bad gateway: upstream timed out")

    with pytest.raises(httpx.HTTPStatusError, match="upstream timed out"):
        service.datasets()


def test_an_empty_error_body_still_raises_the_status() -> None:
    service = _service_replying(500)

    with pytest.raises(httpx.HTTPStatusError, match="500 Internal Server Error") as raised:
        service.datasets()

    assert raised.value.response.status_code == 500


def test_the_module_level_functions_keep_the_detail_too() -> None:
    request = httpx.Request("GET", "http://localhost/stac/catalog.json")
    response = httpx.Response(503, json={"detail": "Catalog is rebuilding"}, request=request)
    with patch("open_climate_service.client.httpx.get", return_value=response):
        with pytest.raises(httpx.HTTPStatusError, match="Catalog is rebuilding"):
            list_datasets("http://localhost")
