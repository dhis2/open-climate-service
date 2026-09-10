"""Named export rendering, plugin discovery, and openEO integration."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service import config
from open_climate_service.exports import RenderedExport
from open_climate_service.exports.registry import load_export_plugins
from open_climate_service.exports.service import render_named_export
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.jobs import OpenEOJobService, _build_dhis2_json_payload, _result_assets
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now


@pytest.fixture
def definition(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "id": "rainfall",
        "plugin": "dhis2",
        "period_type": "monthly",
        "series": [{"select": {}, "data_element": "BXgDHhPdFVU"}],
    }
    monkeypatch.setattr(config, "_cache", {"exports": [definition]})
    return definition


def _frame(**overrides: Any) -> pd.DataFrame:
    return pd.DataFrame({"geometry": ["DiszpKrYNg8"], "t": ["202501"], "precip": [0.0]} | overrides)


def _render(data: Any) -> RenderedExport:
    return render_named_export(data, "DHIS2JSON", {"export": "rainfall"})[1]


def test_single_series_matches_legacy_without_resolving_credentials(definition: dict[str, Any]):
    definition["connection"] = "national-hmis"
    frame = _frame(t=["202501", "202502"], geometry=["DiszpKrYNg8"] * 2, precip=[0, np.nan])
    rendered = _render(frame)
    assert json.loads(rendered.content) == _build_dhis2_json_payload(
        frame,
        {"data_element_id": "BXgDHhPdFVU", "org_unit_field": "geometry", "period_type": "monthly"},
    )
    assert rendered.record_count == 1
    assert rendered.skipped_count == 1
    assert "token" not in json.dumps(config.get_config())


@pytest.mark.parametrize("value", [None, "0", "OU_1", 7, ""])
def test_missing_or_fallback_org_units_fail(definition: dict[str, Any], value: Any):
    with pytest.raises(ValueError, match="feature.id"):
        _render(_frame(geometry=[value]))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), "23.4", [1, 2]])
def test_invalid_values_fail(definition: dict[str, Any], value: Any):
    with pytest.raises(ValueError):
        _render(_frame(precip=[value]))


@pytest.mark.parametrize("period", ["202413", "202400", None])
def test_invalid_periods_fail(definition: dict[str, Any], period: Any):
    with pytest.raises(ValueError):
        _render(_frame(t=[period]))


def test_dekads_cannot_be_collapsed_by_formatting(definition: dict[str, Any]):
    with pytest.raises(ValueError, match="Duplicate DHIS2 value"):
        _render(_frame(t=["2025-01-01", "2025-01-11"], geometry=["DiszpKrYNg8"] * 2, precip=[1, 2]))


@pytest.mark.parametrize(
    ("kind", "period"),
    [("daily", "20250228"), ("weekly", "2025W01"), ("quarterly", "2025Q1"), ("yearly", "2025")],
)
def test_supported_periods(definition: dict[str, Any], kind: str, period: str):
    definition["period_type"] = kind
    assert json.loads(_render(_frame(t=[period])).content)["dataValues"][0]["period"] == period


def test_variable_selection_and_option_combinations(definition: dict[str, Any]):
    definition["series"][0].update(
        select={"variable": "precip"}, category_option_combo="bRowv6yZOF2", attribute_option_combo="HllvX50cXC0"
    )
    frame = _frame(temperature=[25.0])
    ds = frame.set_index(["geometry", "t"]).to_xarray()
    rendered = _render(ds)
    value = json.loads(rendered.content)["dataValues"][0]
    assert value["value"] == "0"
    assert value["categoryOptionCombo"] == "bRowv6yZOF2"
    assert value["attributeOptionCombo"] == "HllvX50cXC0"
    assert len(ds.data_vars) == 2


def test_unresolved_dimensions_fail(definition: dict[str, Any]):
    ds = xr.Dataset({"precip": (("t", "geometry", "quantile"), np.zeros((1, 1, 2)))})
    with pytest.raises(ValueError, match="only organisation-unit and period dimensions"):
        _render(ds)


@pytest.mark.parametrize(
    "change",
    [
        {"series": []},
        {"series": [{"data_element": "BXgDHhPdFVU"}] * 2},
        {"series": [{"data_element": "bad"}]},
        {"series": [{"data_element": "BXgDHhPdFVU", "select": {"quantile": 0.1}}]},
        {"period_type": "dekadal"},
        {"period_type": None},
        {"token": "invalid-secret"},
        {"plugin": "missing"},
    ],
)
def test_bad_mappings_fail(definition: dict[str, Any], change: dict[str, Any]):
    definition.update(change)
    with pytest.raises(ValueError):
        _render(_frame())


def test_mapping_is_not_mutated_and_request_cannot_override_it(definition: dict[str, Any]):
    original = json.dumps(definition)
    _render(_frame())
    assert json.dumps(definition) == original
    with pytest.raises(ValueError, match="only"):
        render_named_export(_frame(), "DHIS2JSON", {"export": "rainfall", "data_element_id": "OtherElemen"})
    with pytest.raises(ValueError, match="requires format"):
        render_named_export(_frame(), "CHAPCSV", {"export": "rainfall"})


def test_duplicate_export_ids_fail(definition: dict[str, Any]):
    config.get_config()["exports"].append(definition.copy())
    with pytest.raises(ValueError, match="Duplicate export ID"):
        _render(_frame())


def _plugin_source(marker: str) -> str:
    return (
        "from open_climate_service.exports import BaseExportPlugin, RenderedExport\n"
        "class ExampleExport(BaseExportPlugin):\n"
        "    id = 'example'\n"
        "    format = 'EXAMPLE'\n"
        "    extension = '.example'\n"
        "    media_type = 'application/x-example'\n"
        "    def validate_mapping(self, mapping):\n"
        "        return mapping\n"
        "    def render(self, data, mapping):\n"
        f"        return RenderedExport({marker!r}.encode(), 1)\n"
        "plugin = ExampleExport()\n"
    )


def _install_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "example_export_" + uuid4().hex
    package = tmp_path / name
    (package / "exports").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "exports" / "__init__.py").write_text("")
    (package / "exports" / "example.py").write_text(_plugin_source("installed"))
    metadata = tmp_path / f"{name}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(f"[open_climate_service.plugins]\n{name} = {name}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_installed_plugin_discovery_and_instance_precedence(
    definition: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fixture(tmp_path, monkeypatch)
    plugin = load_export_plugins()["example"]
    assert plugin.render(None, {}).content == b"installed"
    local = tmp_path / "instance" / "exports"
    local.mkdir(parents=True)
    (local / "_helper.py").write_text("MARKER = 'local'\n")
    (local / "example.py").write_text(
        "from ._helper import MARKER\n" + _plugin_source("local").replace("'local'.encode()", "MARKER.encode()")
    )
    config.get_config()["plugins_dir"] = str(local.parent)
    assert load_export_plugins()["example"].render(None, {}).content == b"local"


def test_broken_plugin_is_skipped(definition: dict[str, Any], tmp_path: Path):
    local = tmp_path / "exports"
    local.mkdir()
    (local / "broken.py").write_text("plugin = object()\n")
    config.get_config()["plugins_dir"] = str(tmp_path)
    plugins = load_export_plugins()
    assert set(plugins) == {"dhis2"}


def test_sync_and_batch_external_export(
    definition: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
):
    _install_fixture(tmp_path, monkeypatch)
    definition.clear()
    definition.update(id="rainfall", plugin="example")
    envelope = SaveResultEnvelope(_frame(), "EXAMPLE", {"export": "rainfall"})
    monkeypatch.setattr("open_climate_service.openeo.execution.run_process_graph", lambda *args: envelope)
    formats = client.get("/file_formats").json()["output"]
    assert "export" in formats["EXAMPLE"]["parameters"]
    response = client.post("/result", json={"process_graph": {}})
    assert response.status_code == 200
    assert response.content == b"installed"
    assert response.headers["content-type"] == "application/x-example"

    monkeypatch.setattr("open_climate_service.openeo.jobs._JOBS_DIR", tmp_path / "jobs")
    service = OpenEOJobService()
    try:
        path = service._persist_result("job-example", envelope)
    finally:
        service.shutdown()
    assert path is not None
    record = OpenEOJobRecord(
        id="job-example", status=OpenEOJobStatus.FINISHED, created=utc_now(), usage={"output_path": path}
    )
    asset = _result_assets(record)["result"]
    assert asset["type"] == "application/x-example"
    response = client.get(asset["href"])
    assert response.content == b"installed"
    assert response.headers["content-type"] == "application/x-example"
    # Historical asset metadata does not depend on the current plugin definition.
    assets = _result_assets(record)
    assert "manifest" in assets
    config.get_config()["exports"] = []
    assert _result_assets(record) == assets


def test_sync_named_dhis2_in_read_only_mode(
    definition: dict[str, Any], monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    config.get_config()["read_only"] = True
    envelope = SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": "rainfall"})
    monkeypatch.setattr("open_climate_service.openeo.execution.run_process_graph", lambda *args: envelope)
    response = client.post("/result", json={"process_graph": {}})
    assert response.status_code == 200
    assert response.json()["dataValues"][0]["value"] == "0"
    definition["series"][0]["data_element"] = "invalid"
    assert client.post("/result", json={"process_graph": {}}).status_code == 400
