"""CHIRPS3 plugins, read from a store laid out like dynamical.org's: rates in kg m-2 s-1, north-up."""

import asyncio
import calendar
import importlib
from typing import Any

import numpy as np
import pytest
import xarray as xr
from xclim.core.units import units2pint

from open_geo_service.data_registry.services import datasets as registry
from open_geo_service.plugins.rasters.chirps3 import CHIRPS3DailyPlugin, CHIRPS3MonthlyPlugin

_BBOX = [29.0, -2.5, 30.0, -1.5]
_MM_PER_DAY = 1 / 86400  # in kg m-2 s-1


def _store(end: str, mm_per_day: np.ndarray | None = None) -> xr.Dataset:
    """Daily values from 2025-01-01 to ``end``; day i rains i mm unless ``mm_per_day`` says otherwise."""
    last = np.datetime64(end, "D") + np.timedelta64(1, "D")
    times = np.arange("2025-01-01", last, dtype="datetime64[D]").astype("datetime64[ns]")
    values = np.arange(len(times), dtype="float32") if mm_per_day is None else mm_per_day[: len(times)]
    data = np.broadcast_to(values.reshape(-1, 1, 1) * _MM_PER_DAY, (len(times), 3, 3)).astype("float32")
    attrs = {"units": "kg m-2 s-1", "standard_name": "precipitation_flux", "comment": "Units equivalent to mm/s."}
    return xr.Dataset(
        {"precipitation_surface": (("time", "latitude", "longitude"), data, attrs)},
        coords={"time": times, "latitude": [-1.5, -2.0, -2.5], "longitude": [29.0, 29.5, 30.0]},
    )


def _serve(monkeypatch: pytest.MonkeyPatch, store: xr.Dataset) -> list[str]:
    """Serve ``store`` in place of dynamical.org; returns the URLs opened."""
    opened: list[str] = []

    def open_source(self: CHIRPS3DailyPlugin) -> xr.Dataset:
        if self._source is None:
            opened.append(self._zarr.url)
            self._source = store
        return self._source

    monkeypatch.setattr(CHIRPS3DailyPlugin, "_open_source", open_source)
    return opened


# --- daily -------------------------------------------------------------------------------


def test_daily_periods_end_at_the_newest_day_in_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _store("2025-03-12"))

    periods = asyncio.run(CHIRPS3DailyPlugin().periods("2025-03-10", "2025-03-31"))

    assert periods == ["2025-03-10", "2025-03-11", "2025-03-12"]


def test_a_day_is_stored_in_mm_per_day_north_up(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _store("2025-01-31"))

    day = CHIRPS3DailyPlugin().fetch_period("2025-01-08", _BBOX)

    assert dict(day.sizes) == {"t": 1, "y": 3, "x": 3}
    assert day.t.values[0] == np.datetime64("2025-01-08", "ns")
    assert float(day.precip.mean()) == pytest.approx(7.0)
    # The source's attributes describe kg m-2 s-1, so none of them survive the conversion.
    assert day.precip.attrs == {"long_name": "Precipitation rate", "units": "mm/d"}
    assert day.y.values.tolist() == [-1.5, -2.0, -2.5]


@pytest.mark.parametrize(("stage", "store"), [("final", "analysis-final"), ("prelim", "analysis-preliminary")])
def test_each_stage_reads_its_own_store_once(monkeypatch: pytest.MonkeyPatch, stage: str, store: str) -> None:
    opened = _serve(monkeypatch, _store("2025-01-31"))
    plugin = CHIRPS3DailyPlugin(stage=stage, unknown_future_field="ignored")

    asyncio.run(plugin.periods("2025-01-01", "2025-01-31"))
    for day in ("2025-01-01", "2025-01-02"):
        plugin.fetch_period(day, _BBOX)

    assert len(opened) == 1
    assert f"ucsb-chc-chirps-{store}/" in opened[0]


def test_rejects_an_unknown_stage() -> None:
    with pytest.raises(ValueError, match="stage must be"):
        CHIRPS3DailyPlugin(stage="bad")


def test_a_day_outside_the_store_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _store("2025-01-31"))

    with pytest.raises(ValueError, match="has 0 of the 1 days from 2025-02-01"):
        CHIRPS3DailyPlugin().fetch_period("2025-02-01", _BBOX)


# --- monthly -----------------------------------------------------------------------------


def test_monthly_periods_stop_at_the_newest_complete_month(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _store("2025-03-30"))

    assert asyncio.run(CHIRPS3MonthlyPlugin().periods("2025-01", "2025-12")) == ["2025-01", "2025-02"]


@pytest.mark.parametrize("period", ["2025-01", "2025-02", "2025-04", "2025-04-17"])
def test_a_month_is_the_mean_daily_rate_on_its_first_day(monkeypatch: pytest.MonkeyPatch, period: str) -> None:
    """The month's total over its own number of days: CHC's monthly total divided as before."""
    rain = np.arange(400, dtype="float32") % 7
    _serve(monkeypatch, _store("2025-12-31", rain))

    month = CHIRPS3MonthlyPlugin().fetch_period(period, _BBOX)

    first = np.datetime64(period[:7], "D")
    days = calendar.monthrange(int(period[:4]), int(period[5:7]))[1]
    offset = int((first - np.datetime64("2025-01-01", "D")) // np.timedelta64(1, "D"))
    assert dict(month.sizes) == {"t": 1, "y": 3, "x": 3}
    assert month.t.values[0] == first.astype("datetime64[ns]")
    assert float(month.precip.mean()) == pytest.approx(rain[offset : offset + days].sum() / days, rel=1e-6)
    assert month.precip.attrs["units"] == "mm/d"


def test_the_monthly_plugin_always_reads_the_final_stage() -> None:
    assert CHIRPS3MonthlyPlugin(stage="prelim").stage == "final"


@pytest.mark.parametrize("value", ["not-a-month", "2025-13", "2025-1"])
def test_rejects_an_unparseable_month(value: str) -> None:
    with pytest.raises(ValueError, match="expected YYYY-MM"):
        CHIRPS3MonthlyPlugin().fetch_period(value, _BBOX)


# --- templates ---------------------------------------------------------------------------


def _chirps3_templates() -> list[dict[str, Any]]:
    return [
        t
        for t in registry.list_datasets()
        if "ingestion" in t and t["ingestion"]["plugin"].startswith("open_geo_service.plugins.rasters.chirps3.")
    ]


@pytest.mark.parametrize("template", _chirps3_templates(), ids=lambda t: t["id"])
def test_plugin_output_units_match_the_template(monkeypatch: pytest.MonkeyPatch, template: dict[str, Any]) -> None:
    _serve(monkeypatch, _store("2025-01-31"))
    module_name, class_name = template["ingestion"]["plugin"].rsplit(".", 1)
    plugin = getattr(importlib.import_module(module_name), class_name)(**(template["ingestion"].get("params") or {}))
    period = "2025-01" if template["period_type"] == "monthly" else "2025-01-02"

    produced = plugin.fetch_period(period, _BBOX)[template["variable"]].attrs["units"]

    assert units2pint(produced) == units2pint(template["units"])
