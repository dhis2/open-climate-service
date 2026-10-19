from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.plugins.rasters import esa_cci_lc


def test_load_years_concatenates_files_with_their_requested_years(monkeypatch: Any) -> None:
    requests: list[dict[str, Any]] = []

    class RemoteRequest:
        def __init__(self, request: dict[str, Any]) -> None:
            self.request = request

        def download(self, target: str) -> None:
            with zipfile.ZipFile(target, "w") as archive:
                for year in self.request["year"]:
                    archive.writestr(f"land_cover_{year}.nc", b"")

    class Client:
        def check_authentication(self) -> None:
            pass

        def submit(self, dataset: str, request: dict[str, Any]) -> RemoteRequest:
            assert dataset == "satellite-land-cover"
            requests.append(request)
            return RemoteRequest(request)

    def open_mfdataset(paths: list[str | Path], **_: Any) -> xr.Dataset:
        years = [Path(path).stem.rsplit("_", maxsplit=1)[-1] for path in paths]
        return xr.Dataset(
            {
                "lccs_class": (
                    ("year", "lat", "lon"),
                    np.array([[[int(year) % 255]] for year in years], dtype="uint8"),
                )
            },
            coords={"year": range(len(years)), "lat": [0.5], "lon": [0.5]},
        )

    monkeypatch.setattr(esa_cci_lc, "_CdsClient", Client)
    monkeypatch.setattr(esa_cci_lc.xr, "open_mfdataset", open_mfdataset)

    classes = esa_cci_lc._load_years(["1992", "2015", "2016", "2022"], [0, 0, 1, 1])

    assert classes["year"].values.tolist() == ["1992", "2015", "2016", "2022"]
    assert classes.dims == ("year", "y", "x")
    assert len(requests) == 1
    assert requests[0]["year"] == ["1992", "2015", "2016", "2022"]
    assert set(requests[0]["version"]) == {"v2_0_7cds", "v2_1_1"}


def test_plugin_fetches_requested_years_once_and_returns_each_period(monkeypatch: Any) -> None:
    requests: list[list[str]] = []

    def load_years(years: list[str], bbox: list[float]) -> xr.DataArray:
        assert bbox == [0, 0, 1, 1]
        requests.append(years)
        return xr.DataArray(
            np.array([[[10]], [[20]]], dtype="uint8"),
            dims=("year", "y", "x"),
            coords={"year": years, "y": [0.5], "x": [0.5]},
        )

    monkeypatch.setattr(esa_cci_lc, "_load_years", load_years)
    plugin = esa_cci_lc.ESACCILandCoverClassPlugin()

    assert asyncio.run(plugin.periods("1992", "1993")) == ["1992", "1993"]
    first = plugin.fetch_period("1992", [0, 0, 1, 1])
    second = plugin.fetch_period("1993", [0, 0, 1, 1])

    assert requests == [["1992", "1993"]]
    assert first["lccs_class"].values.item() == 10
    assert second["lccs_class"].values.item() == 20
