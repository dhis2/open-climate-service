from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.rasters import esa_cci_lc


def test_fetch_period_reads_zipped_netcdf_and_masks_no_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    netcdf_path = tmp_path / "land_cover.nc"
    source = xr.Dataset(
        {
            "lccs_class": (
                ("time", "lat", "lon"),
                np.array([[[10, 0], [20, -9999]]], dtype="int16"),
            )
        },
        coords={
            "time": [np.datetime64("2020-01-01")],
            "lat": [1.0, 0.0],
            "lon": [2.0, 3.0],
        },
    )
    source["lccs_class"].encoding["_FillValue"] = -9999
    source.to_netcdf(netcdf_path)

    archive_path = tmp_path / "download.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(netcdf_path, "land_cover.nc")
    archive_bytes = archive_path.read_bytes()

    requests: list[tuple[str, dict[str, Any]]] = []

    class SubmittedRequest:
        def download(self, target: str) -> None:
            Path(target).write_bytes(archive_bytes)

    class Client:
        def check_authentication(self) -> None:
            pass

        def submit(self, dataset: str, request: dict[str, Any]) -> SubmittedRequest:
            requests.append((dataset, request))
            return SubmittedRequest()

    monkeypatch.setattr(esa_cci_lc, "_CdsClient", Client)

    result = esa_cci_lc.ESACCILandCoverClassPlugin().fetch_period("2020", [2.0, 0.0, 3.0, 1.0])

    assert requests == [
        (
            "satellite-land-cover",
            {
                "variable": "all",
                "year": ["2020"],
                "version": ["v2_1_1"],
                "area": [1.0, 2.0, 0.0, 3.0],
            },
        )
    ]
    classes = result["lccs_class"]
    assert classes.dtype == np.dtype("float32")
    assert classes.dims == ("t", "y", "x")
    np.testing.assert_equal(classes.values, [[[10.0, np.nan], [20.0, np.nan]]])


def test_periods_uses_years_allowed_by_cds(monkeypatch: Any) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def apply_constraints(self, dataset: str, request: dict[str, Any]) -> dict[str, Any]:
            requests.append((dataset, request))
            return {"year": ["2024", "1992", "2022", "2016"]}

    monkeypatch.setattr(esa_cci_lc, "_CdsClient", Client)

    periods = asyncio.run(esa_cci_lc.ESACCILandCoverClassPlugin().periods("2015", "2025"))

    assert periods == ["2016", "2022", "2024"]
    assert requests == [("satellite-land-cover", {})]


def test_periods_returns_empty_when_request_has_no_available_years(monkeypatch: Any) -> None:
    class Client:
        def apply_constraints(self, dataset: str, request: dict[str, Any]) -> dict[str, Any]:
            assert dataset == "satellite-land-cover"
            assert request == {}
            return {"year": ["1992", "2015"]}

    monkeypatch.setattr(esa_cci_lc, "_CdsClient", Client)

    periods = asyncio.run(esa_cci_lc.ESACCILandCoverClassPlugin().periods("2016", "2025"))

    assert periods == []


def test_periods_fails_if_cds_does_not_return_year_options(monkeypatch: Any) -> None:
    class Client:
        def apply_constraints(self, dataset: str, request: dict[str, Any]) -> dict[str, Any]:
            assert dataset == "satellite-land-cover"
            assert request == {}
            return {"year": None}

    monkeypatch.setattr(esa_cci_lc, "_CdsClient", Client)

    with pytest.raises(RuntimeError, match="no valid year options"):
        asyncio.run(esa_cci_lc.ESACCILandCoverClassPlugin().periods("1992", "2025"))
