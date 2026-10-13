from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_climate_service.plugins.rasters import era5_land_daily_stats as mod
from open_climate_service.plugins.rasters.era5_land_daily_stats import (
    ERA5LandCDSDailyStatisticPlugin,
    ERA5LandCDSDailyWindSpeedPlugin,
    _cds_time_zone,
)


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(0, "utc+00:00"), (1, "utc+01:00"), (5.5, "utc+05:30"), (-3, "utc-03:00")],
)
def test_cds_time_zone_formats_offset(hours: float, expected: str) -> None:
    assert _cds_time_zone(timedelta(hours=hours)) == expected


def test_daily_statistic_rejects_unknown_variable_and_statistic() -> None:
    with pytest.raises(ValueError, match="unsupported variable"):
        ERA5LandCDSDailyStatisticPlugin(variable="tp")
    with pytest.raises(ValueError, match="unsupported statistic"):
        ERA5LandCDSDailyStatisticPlugin(variable="t2m", statistic="median")


def test_daily_statistic_request_uses_statistic_and_local_time_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service import config as api_config

    monkeypatch.setattr(api_config, "get_utc_offset", lambda: timedelta(hours=1))
    monkeypatch.setattr(mod, "_daily_availability_cutoff", lambda: datetime(2030, 1, 1).date())
    request = ERA5LandCDSDailyStatisticPlugin(variable="t2m", statistic="max")._request(2024, 2, (0.0, 10.0, 2.0, 12.0))
    assert request["daily_statistic"] == "daily_maximum"
    assert request["time_zone"] == "utc+01:00"
    assert request["variable"] == ["2m_temperature"]
    assert len(request["day"]) == 29


def test_daily_statistic_converts_kelvin_and_serves_one_day_from_the_month(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandCDSDailyStatisticPlugin(variable="d2m")
    calls = []
    times = pd.date_range("2024-08-01", periods=3, freq="D")

    def fake_month(self: object, year: int, month: int, bbox: tuple) -> xr.Dataset:
        calls.append((year, month))
        ds = xr.Dataset(
            {"d2m": (("t", "y", "x"), np.full((3, 1, 1), 293.15))},
            coords={"t": times, "y": [1.0], "x": [1.0]},
        )
        return mod.kelvin_to_celsius(ds, {"variable": "d2m"})

    monkeypatch.setattr(ERA5LandCDSDailyStatisticPlugin, "_fetch_month", fake_month)
    first = plugin.fetch_period("2024-08-01", [0, 0, 2, 2])
    second = plugin.fetch_period("2024-08-02", [0, 0, 2, 2])
    assert calls == [(2024, 8)]
    assert first.sizes["t"] == 1 and second.sizes["t"] == 1
    assert float(second["d2m"].squeeze()) == pytest.approx(20.0)


def test_wind_speed_is_mean_of_hourly_speeds_over_the_local_day(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service import config as api_config

    monkeypatch.setattr(api_config, "get_utc_offset", lambda: timedelta(hours=1))
    monkeypatch.setattr(mod, "_hourly_availability_cutoff", lambda: datetime(2030, 1, 1))
    # Wind blows east for 12 hours then west for 12: mean vector is zero, mean speed is 2.
    hours = pd.date_range("2024-07-31T23:00", periods=31 * 24, freq="h")
    u = np.where((np.arange(len(hours)) % 24) < 12, 2.0, -2.0)[:, None, None]

    def fake_hourly(self: object, first: datetime, last: datetime, bbox: tuple) -> xr.Dataset:
        assert first == datetime(2024, 7, 31, 23)
        return xr.Dataset(
            {"u10": (("t", "y", "x"), u), "v10": (("t", "y", "x"), np.zeros_like(u))},
            coords={"t": hours, "y": [1.0], "x": [1.0]},
        )

    monkeypatch.setattr(ERA5LandCDSDailyWindSpeedPlugin, "_fetch_hourly", fake_hourly)
    ds = ERA5LandCDSDailyWindSpeedPlugin().fetch_period("2024-08-01", [0, 0, 2, 2])
    assert ds["wind_speed"].sizes["t"] == 1
    assert float(ds["wind_speed"].squeeze()) == pytest.approx(2.0)
