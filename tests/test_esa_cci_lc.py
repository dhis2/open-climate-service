from __future__ import annotations

import asyncio
from typing import Any

import pytest

from open_climate_service.plugins.rasters import esa_cci_lc


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
