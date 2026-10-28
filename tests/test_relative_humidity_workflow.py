"""Tests for the built-in relative_humidity_from_daily_dewpoint workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr
from openeo_pg_parser_networkx.process_registry import Process

from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.openeo import jobs
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.openeo.execution import (
    _augment_with_workflows,
    _build_process_registry,
)


def _cube(celsius: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 2, 3), celsius, dtype="float32"),
        dims=("t", "y", "x"),
        coords={
            "t": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "y": [1.0, 2.0],
            "x": [1.0, 2.0, 3.0],
        },
        attrs={"units": "degC"},
    ).rio.write_crs("EPSG:4326")


def _overlay(cubes: dict[str, xr.DataArray]) -> Any:
    reg = _build_process_registry()
    reg["load_collection"] = Process(spec={}, implementation=lambda id=None, temporal_extent=None, **k: cubes[str(id)])
    return _augment_with_workflows(reg)


def _run(overlay: Any) -> Any:
    return overlay["relative_humidity_from_daily_dewpoint"].implementation(
        temperature_dataset_id="era5land_temperature_daily",
        dewpoint_dataset_id="era5land_dewpoint_daily",
        output_dataset_id="era5land_relative_humidity_daily_estimated",
        variable="rh",
        temporal_extent=["2024-01-01", "2024-01-02"],
    )


def test_relative_humidity_workflow_registered_as_process() -> None:
    ids = {wf.id for wf in workflow_store.list_workflows().processes}
    assert "relative_humidity_from_daily_dewpoint" in ids


def test_relative_humidity_workflow_computes_percent_from_celsius_inputs() -> None:
    overlay = _overlay({"era5land_temperature_daily": _cube(25.0), "era5land_dewpoint_daily": _cube(15.0)})

    envelope = _run(overlay)

    assert envelope.format.upper() == "ZARR"
    assert envelope.options["dataset_id"] == "era5land_relative_humidity_daily_estimated"
    assert envelope.options["source_dataset_id"] == "era5land_dewpoint_daily"
    assert envelope.options["variable"] == "rh"
    assert envelope.options["publish"] is True
    da = envelope.data
    assert da.dims == ("t", "y", "x")
    assert da.attrs["units"] == "%"
    # 100 * e(15 degC) / e(25 degC) with the IFS saturation formula over water.
    assert np.allclose(da.values, 53.82, atol=0.01)


def test_relative_humidity_workflow_is_saturated_when_dewpoint_equals_temperature() -> None:
    overlay = _overlay({"era5land_temperature_daily": _cube(20.0), "era5land_dewpoint_daily": _cube(20.0)})

    assert np.allclose(_run(overlay).data.values, 100.0)


def test_relative_humidity_workflow_refuses_inputs_without_units() -> None:
    unitless = _cube(25.0)
    unitless.attrs = {}
    overlay = _overlay({"era5land_temperature_daily": unitless, "era5land_dewpoint_daily": _cube(15.0)})

    with pytest.raises(Exception, match="units"):
        _run(overlay)


@pytest.fixture
def managed_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary instance whose registry holds the dewpoint source the output derives from."""
    configs = tmp_path / "plugins" / "rasters"
    configs.mkdir(parents=True)
    (configs / "source.yaml").write_text(
        "- id: era5land_dewpoint_daily\n"
        "  name: 2m dewpoint temperature (ERA5-Land)\n"
        "  variable: d2m\n"
        "  period_type: daily\n"
        "  units: degC\n"
        "  license: CC-BY-4.0\n"
        "  sync:\n"
        "    kind: temporal\n"
        "  ingestion:\n"
        "    plugin: open_climate_service.plugins.rasters.era5_land_cds.ERA5LandCDSZarrDailyFromHourlyPlugin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", configs)
    registry.reset_template_caches()
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path / "rasters")
    (tmp_path / "rasters").mkdir()
    return tmp_path


def test_relative_humidity_workflow_publishes_as_percent(managed_instance: Path) -> None:
    """The dewpoint source declares degC; the auto-registered output must not inherit it."""
    overlay = _overlay({"era5land_temperature_daily": _cube(25.0), "era5land_dewpoint_daily": _cube(15.0)})
    envelope = _run(overlay)
    # As the batch job does before a managed write.
    ds = envelope.data.to_dataset(name=envelope.data.name or "result")

    jobs._write_managed_zarr(ds, envelope.options)

    template = registry.get_dataset("era5land_relative_humidity_daily_estimated")
    assert template is not None
    assert template["units"] == "%"
    assert template["variable"] == "rh"
    assert template["period_type"] == "daily"
    store_path = managed_instance / "rasters" / "era5land_relative_humidity_daily_estimated.icechunk"
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store_path)))
    stored = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    assert stored["rh"].attrs["units"] == "%"
    assert np.allclose(stored["rh"].values, 53.82, atol=0.01)
