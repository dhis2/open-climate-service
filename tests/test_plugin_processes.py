"""Tests for @process decorator and plugin process discovery."""

from pathlib import Path

import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.process import get_process_metadata, process


def test_process_decorator_attaches_metadata() -> None:
    @process
    def my_func(data: xr.DataArray, scale: float = 1.0) -> xr.DataArray:
        """Scale a DataArray."""
        return data * scale

    meta = get_process_metadata(my_func)
    assert meta is not None
    assert meta["id"] == "my_func"
    assert meta["summary"] == "Scale a DataArray."
    param_names = [p["name"] for p in meta["parameters"]]
    assert "data" in param_names
    assert "scale" in param_names
    scale_param = next(p for p in meta["parameters"] if p["name"] == "scale")
    assert scale_param["optional"] is True
    assert scale_param["default"] == 1.0
    assert scale_param["schema"] == {"type": "number"}


def test_process_decorator_with_explicit_metadata() -> None:
    @process(summary="Custom summary", parameters={"scale": {"description": "Scale factor"}})
    def my_func2(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
        return data * scale

    meta = get_process_metadata(my_func2)
    assert meta is not None
    assert meta["summary"] == "Custom summary"
    scale_param = next(p for p in meta["parameters"] if p["name"] == "scale")
    assert scale_param["description"] == "Scale factor"


def test_process_decorator_preserves_function_behaviour() -> None:
    @process(summary="Double")
    def double(data: xr.DataArray) -> xr.DataArray:
        return data * 2

    da = xr.DataArray([1.0, 2.0, 3.0])
    result = double(data=da)
    assert list(result.values) == [2.0, 4.0, 6.0]


def test_plugin_processes_appear_in_get_processes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A @process function in plugins_dir/processes/ appears in GET /processes."""
    import open_climate_service.openeo.execution as execution_mod

    processes_dir = tmp_path / "processes"
    processes_dir.mkdir()
    (processes_dir / "my_index.py").write_text(
        """
import xarray as xr
from open_climate_service.process import process

@process(summary="My custom climate index")
def my_index(data: xr.DataArray, thresh: float = 0.5) -> xr.DataArray:
    return (data > thresh).astype(float)
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {tmp_path}\n", encoding="utf-8")

    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    # Invalidate the execution registry singleton so it picks up new plugins
    monkeypatch.setattr(execution_mod, "_registry", None)

    response = client.get("/processes")
    assert response.status_code == 200
    ids = {p["id"] for p in response.json()["processes"]}
    assert "my_index" in ids


def test_plugin_process_summary_in_catalog(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import open_climate_service.openeo.execution as execution_mod

    processes_dir = tmp_path / "processes"
    processes_dir.mkdir()
    (processes_dir / "cdd.py").write_text(
        """
import xarray as xr
from open_climate_service.process import process

@process(summary="Consecutive dry days")
def cdd(pr: xr.DataArray, thresh: str = "1mm/day") -> xr.DataArray:
    return pr
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {tmp_path}\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.setattr(execution_mod, "_registry", None)

    response = client.get("/processes")
    procs = {p["id"]: p for p in response.json()["processes"]}
    assert "cdd" in procs
    assert procs["cdd"]["summary"] == "Consecutive dry days"


def test_string_annotations_are_resolved_to_schemas() -> None:
    """`from __future__ import annotations` stores annotations as strings.

    A raw `param.annotation` is then `"str"` rather than `str`, the type map never matches, and the
    parameter is published with an empty schema — a client building a graph gets an untyped field
    with nothing to validate against. Written here as explicit string annotations, which is exactly
    what that import produces.
    """

    @process
    def scaled(threshold: "str", count: "int" = 3, ratio: "float | None" = None) -> "str":
        """Do something with a threshold."""
        return threshold

    meta = get_process_metadata(scaled)
    assert meta is not None
    schemas = {p["name"]: p.get("schema") for p in meta["parameters"]}
    assert schemas == {
        "threshold": {"type": "string"},
        "count": {"type": "integer"},
        # `float | None` keeps its null, as the openEO specs do for a nullable parameter.
        "ratio": {"type": ["number", "null"]},
    }


def test_a_nullable_annotation_keeps_its_null_and_its_default() -> None:
    """`str | None` must publish `["string", "null"]`, not a bare `"string"`.

    This is how the openEO specs express an optional parameter defaulting to null —
    `aggregate_spatial`'s `target_dimension` is exactly this shape. Unwrapping to `"string"`
    publishes a schema that rejects the documented default, so a client validating the graph
    would refuse a valid call. That is worse than publishing no schema at all.
    """

    @process
    def nullable(target: str | None = None, count: int = 1) -> None:
        """Has a nullable parameter."""

    meta = get_process_metadata(nullable)
    assert meta is not None
    target, count = meta["parameters"]

    assert target["schema"] == {"type": ["string", "null"]}
    assert target["optional"] is True
    assert target["default"] is None

    # A non-nullable parameter is unaffected.
    assert count["schema"] == {"type": "integer"}
    assert count["default"] == 1


def test_an_unresolvable_annotation_does_not_lose_the_process() -> None:
    """A plugin may annotate a type it imports only under TYPE_CHECKING.

    Resolution needs the module namespace and fails for those, which must cost the schema for that
    process rather than the process itself.
    """

    @process
    def exotic(data: "SomeTypeThatIsNotImported", factor: "int" = 2) -> None:  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        """Takes something unresolvable."""

    meta = get_process_metadata(exotic)
    assert meta is not None
    assert [p["name"] for p in meta["parameters"]] == ["data", "factor"]
    assert meta["parameters"][0]["schema"] == {}
