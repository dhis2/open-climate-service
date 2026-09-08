"""Slice 6: multi-series rendering to multiple DHIS2 data elements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_climate_service import config
from open_climate_service.exports.service import render_named_export


@pytest.fixture
def definition(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "id": "three",
        "plugin": "dhis2",
        "period_type": "monthly",
        "series": [
            {"select": {"variable": "temperature"}, "data_element": "TEMP0000001"},
            {"select": {"variable": "precipitation"}, "data_element": "PREC0000001"},
            {"select": {"variable": "population"}, "data_element": "POPU0000001"},
        ],
    }
    monkeypatch.setattr(config, "_cache", {"exports": [definition]})
    return definition


def _render(data: Any, definition: dict[str, Any]) -> Any:
    return render_named_export(data, "DHIS2JSON", {"export": definition["id"]})[1]


def test_multi_series_wide_frame(definition: dict[str, Any]):
    frame = pd.DataFrame(
        {
            "geometry": ["DiszpKrYNg8", "DiszpKrYNg8"],
            "t": ["202501", "202502"],
            "temperature": [26.4, 27.0],
            "precipitation": [112.0, 100.0],
            "population": [45000, 45100],
        }
    )
    rendered = _render(frame, definition)
    values = json.loads(rendered.content)["dataValues"]
    assert len(values) == 6
    assert rendered.record_count == 6
    assert rendered.skipped_count == 0
    assert rendered.periods == ("202501", "202502")

    by_key = {(value["dataElement"], value["orgUnit"], value["period"]): value["value"] for value in values}
    assert by_key[("TEMP0000001", "DiszpKrYNg8", "202501")] == "26.4"
    assert by_key[("PREC0000001", "DiszpKrYNg8", "202502")] == "100"
    assert by_key[("POPU0000001", "DiszpKrYNg8", "202501")] == "45000"


def test_multi_series_merged_cube_pivots_away_internal_dimension(definition: dict[str, Any]):
    # merge_cubes yields one value array stacked on a synthetic "__cubes__" dim.
    merged = xr.DataArray(
        [[[26.4, 112.0, 45000.0]]],
        dims=("t", "geometry", "__cubes__"),
        coords={
            "t": ["202501"],
            "geometry": ["DiszpKrYNg8"],
            "__cubes__": ["temperature", "precipitation", "population"],
        },
        name="result",
    )
    rendered = _render(merged, definition)
    values = json.loads(rendered.content)["dataValues"]
    assert rendered.record_count == 3
    assert {value["dataElement"] for value in values} == {"TEMP0000001", "PREC0000001", "POPU0000001"}
    by_element = {value["dataElement"]: value["value"] for value in values}
    assert by_element["TEMP0000001"] == "26.4"
    assert by_element["PREC0000001"] == "112"
    assert by_element["POPU0000001"] == "45000"


def test_multi_series_quantile_selection(definition: dict[str, Any]):
    definition["series"] = [
        {"select": {"variable": "precip", "quantile": 0.1}, "data_element": "PREC0000001"},
        {"select": {"variable": "precip", "quantile": 0.9}, "data_element": "PREC0000002"},
    ]
    ds = xr.Dataset(
        {"precip": (("t", "geometry", "quantile"), [[[1.0, 9.0]]])},
        coords={"t": ["202501"], "geometry": ["DiszpKrYNg8"], "quantile": [0.1, 0.9]},
    )
    rendered = _render(ds, definition)
    values = json.loads(rendered.content)["dataValues"]
    assert {(value["dataElement"], value["value"]) for value in values} == {
        ("PREC0000001", "1"),
        ("PREC0000002", "9"),
    }


def test_multi_series_skips_missing_values_per_series(definition: dict[str, Any]):
    definition["series"] = definition["series"][:2]
    frame = pd.DataFrame(
        {
            "geometry": ["DiszpKrYNg8", "DiszpKrYNg8"],
            "t": ["202501", "202502"],
            "temperature": [26.4, np.nan],
            "precipitation": [np.nan, 100.0],
        }
    )
    rendered = _render(frame, definition)
    assert rendered.record_count == 2
    assert rendered.skipped_count == 2


def test_multi_series_duplicate_destination_key_rejected(definition: dict[str, Any]):
    definition["series"] = [
        {"select": {"variable": "temperature"}, "data_element": "TEMP0000001"},
        {"select": {"variable": "temperature"}, "data_element": "TEMP0000001"},
    ]
    frame = pd.DataFrame({"geometry": ["DiszpKrYNg8"], "t": ["202501"], "temperature": [26.4]})
    with pytest.raises(ValueError, match="Duplicate DHIS2 value"):
        _render(frame, definition)


def test_multi_series_unresolved_variable_rejected(definition: dict[str, Any]):
    frame = pd.DataFrame({"geometry": ["DiszpKrYNg8"], "t": ["202501"], "temperature": [26.4]})
    with pytest.raises(ValueError, match="not present in the result"):
        _render(frame, definition)


def test_multi_series_ambiguous_empty_selector_rejected(definition: dict[str, Any]):
    definition["series"] = [{"select": {}, "data_element": "TEMP0000001"}]
    frame = pd.DataFrame(
        {"geometry": ["DiszpKrYNg8"], "t": ["202501"], "temperature": [26.4], "precipitation": [112.0]}
    )
    with pytest.raises(ValueError, match="exactly one value column"):
        _render(frame, definition)


def test_multi_series_captures_all_contributing_sources(definition: dict[str, Any], tmp_path: Path):
    import hashlib

    from open_climate_service.exports.manifest import ExportManifest
    from open_climate_service.exports.service import write_named_export

    definition["series"] = [
        {"select": {"variable": "temperature"}, "data_element": "TEMP0000001"},
        {"select": {"variable": "population"}, "data_element": "POPU0000001"},
    ]
    frame = pd.DataFrame({"geometry": ["DiszpKrYNg8"], "t": ["202501"], "temperature": [26.4], "population": [45000]})
    directory = tmp_path / "results"
    directory.mkdir()
    provenance = {
        "process_sha256": hashlib.sha256(b"{}").hexdigest(),
        "sources": [
            {"collection_id": "era5-temperature", "artifact_id": "a1", "source_dataset_id": None, "snapshot_id": "s1"},
            {"collection_id": "worldpop", "artifact_id": "a2", "source_dataset_id": None, "snapshot_id": "s2"},
        ],
        "features": [],
        "missing": [],
    }
    path = write_named_export(frame, directory, "DHIS2JSON", {"export": "three"}, job_id="j", provenance=provenance)
    metadata = json.loads((Path(path).parent / ".export.json").read_text())
    manifest = ExportManifest.model_validate_json((Path(path).parent / metadata["manifest"]).read_bytes())
    collections = {source["collection_id"] for source in manifest.provenance["sources"]}
    assert collections == {"era5-temperature", "worldpop"}
