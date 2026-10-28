import asyncio

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.rasters import era5_land_cds, era5_land_edh
from open_climate_service.plugins.rasters.era5_land_cds import (
    ERA5LandCDSZarrDailyFromHourlyPlugin,
    ERA5LandCDSZarrDailyRelHumPlugin,
)
from open_climate_service.plugins.rasters.era5_land_edh import ERA5LandEDHMonthlyFromDailyPlugin

_CDS_BBOX = [28.0, -3.0, 29.5, -1.5]
# Crosses the prime meridian, which the EDH store's 0-360 longitude axis wraps.
_EDH_BBOX = [-1.0, 9.0, 1.0, 10.5]


def _cds_store(variable: str, hourly_values: np.ndarray) -> xr.Dataset:
    """Two days of hourly values on the CDS ARCO layout: -180..180 longitude, south-up latitude."""
    times = np.arange("2020-01-01", "2020-01-03", dtype="datetime64[h]").astype("datetime64[ns]")
    data = np.broadcast_to(hourly_values.reshape(-1, 1, 1), (len(times), 2, 2)).copy()
    return xr.Dataset(
        {variable: (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": [-2.5, -2.0], "longitude": [28.5, 29.0]},
    )


def _edh_store(variable: str, end: str, daily_values: np.ndarray | None = None) -> xr.Dataset:
    """Daily values from 2020-01-01 to ``end`` on the EDH layout: 0-360 longitude, north-up latitude."""
    times = np.arange("2020-01-01", np.datetime64(end) + 1, dtype="datetime64[D]").astype("datetime64[ns]")
    values = np.arange(len(times), dtype=float) if daily_values is None else daily_values[: len(times)]
    data = np.broadcast_to(values.reshape(-1, 1, 1), (len(times), 2, 4)).copy()
    return xr.Dataset(
        {variable: (("valid_time", "latitude", "longitude"), data)},
        coords={"valid_time": times, "latitude": [10.0, 9.5], "longitude": [0.0, 0.5, 359.0, 359.5]},
    )


# --- CDS: hourly to daily ----------------------------------------------------------------


def test_cds_daily_temperature_is_the_mean_of_the_utc_day_in_celsius(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _cds_store("t2m", np.arange(48, dtype=float) + 273.15)
    monkeypatch.setattr(era5_land_cds, "_open_cds_zarr", lambda _url: store)
    plugin = ERA5LandCDSZarrDailyFromHourlyPlugin(
        variable="t2m", temporal_aggregation="mean", unit_transform="kelvin_to_celsius"
    )

    day = plugin.fetch_period("2020-01-02", _CDS_BBOX)

    assert dict(day.sizes) == {"t": 1, "y": 2, "x": 2}
    assert day.t.values[0] == np.datetime64("2020-01-02", "ns")
    assert float(day.t2m.mean()) == pytest.approx(35.5)  # hours 24..47
    assert day.t2m.attrs["units"] == "degC"
    assert day.y.values.tolist() == [-2.0, -2.5]  # north-up


def test_cds_daily_precipitation_sums_the_hours_in_mm(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _cds_store("tp", np.full(48, 0.001))  # 1 mm every hour
    monkeypatch.setattr(era5_land_cds, "_open_cds_zarr", lambda _url: store)
    plugin = ERA5LandCDSZarrDailyFromHourlyPlugin(
        variable="tp", temporal_aggregation="sum", unit_transform="metres_to_mm"
    )

    day = plugin.fetch_period("2020-01-01", _CDS_BBOX)

    assert float(day.tp.mean()) == pytest.approx(24.0)


def test_cds_daily_periods_stop_at_the_latest_hour_in_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _cds_store("t2m", np.zeros(48))
    monkeypatch.setattr(era5_land_cds, "_open_cds_zarr", lambda _url: store)
    plugin = ERA5LandCDSZarrDailyFromHourlyPlugin(variable="t2m", temporal_aggregation="mean")

    assert asyncio.run(plugin.periods("2019-12-31", "2020-01-10")) == ["2019-12-31", "2020-01-01", "2020-01-02"]


def test_cds_rejects_a_variable_without_a_known_store() -> None:
    with pytest.raises(ValueError, match="zarr_url"):
        ERA5LandCDSZarrDailyFromHourlyPlugin(variable="nope", temporal_aggregation="mean")


def test_cds_rejects_an_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="temporal_aggregation"):
        ERA5LandCDSZarrDailyFromHourlyPlugin(variable="t2m", temporal_aggregation="median")


def test_cds_rejects_an_unknown_unit_transform() -> None:
    with pytest.raises(ValueError, match="unit_transform"):
        ERA5LandCDSZarrDailyFromHourlyPlugin(variable="t2m", temporal_aggregation="mean", unit_transform="fahrenheit")


def _cds_humidity_store(temperature: np.ndarray, dewpoint: np.ndarray) -> xr.Dataset:
    """The ARCO ``sfc-2m-temperature`` group, which holds both 2 m temperature and dewpoint."""
    return xr.merge([_cds_store("t2m", temperature), _cds_store("d2m", dewpoint)])


def test_cds_daily_relative_humidity_averages_hourly_values_not_daily_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A constant 15 degC dewpoint under a diurnal swing of 15..35 degC: RH runs from 100 %
    # at night to about 30 % in the afternoon.
    hours = np.arange(48)
    temperature = 298.15 + 10.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    dewpoint = np.full(48, 288.15)
    store = _cds_humidity_store(temperature, dewpoint)
    monkeypatch.setattr(era5_land_cds, "_open_cds_zarr", lambda _url: store)
    mean_plugin = ERA5LandCDSZarrDailyRelHumPlugin(temporal_aggregation="mean")
    max_plugin = ERA5LandCDSZarrDailyRelHumPlugin(temporal_aggregation="max")

    day = mean_plugin.fetch_period("2020-01-02", _CDS_BBOX)
    day_max = max_plugin.fetch_period("2020-01-02", _CDS_BBOX)

    assert dict(day.sizes) == {"t": 1, "y": 2, "x": 2}
    assert day.t.values[0] == np.datetime64("2020-01-02", "ns")
    assert day.rh.attrs == {"units": "%"}
    assert float(day_max.rh.max()) == pytest.approx(100.0, abs=0.01)
    # RH of the daily means would be 53.8 %.
    assert float(day.rh.mean()) == pytest.approx(59.41, abs=0.01)


def test_cds_daily_relative_humidity_rejects_a_sum() -> None:
    with pytest.raises(ValueError, match="temporal_aggregation"):
        ERA5LandCDSZarrDailyRelHumPlugin(temporal_aggregation="sum")


# --- EDH: daily to monthly ---------------------------------------------------------------


def test_edh_monthly_temperature_is_the_mean_of_daily_means(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _edh_store("t2m", "2020-02-29", np.arange(60, dtype=float) + 273.15)
    monkeypatch.setattr(era5_land_edh, "_open_edh_zarr", lambda _url: store)
    plugin = ERA5LandEDHMonthlyFromDailyPlugin(
        variable="t2m", temporal_aggregation="mean", unit_transform="kelvin_to_celsius"
    )

    january = plugin.fetch_period("2020-01", _EDH_BBOX)
    february = plugin.fetch_period("2020-02", _EDH_BBOX)

    assert january.sizes["t"] == 1
    assert january.t.values[0] == np.datetime64("2020-01-01", "ns")
    assert float(january.t2m.mean()) == pytest.approx(15.0)  # days 0..30
    assert float(february.t2m.mean()) == pytest.approx(45.0)  # days 31..59
    assert january.t2m.attrs["units"] == "degC"


def test_edh_monthly_precipitation_is_a_mean_daily_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _edh_store("tp", "2020-01-31", np.full(31, 0.002))  # 2 mm every day
    monkeypatch.setattr(era5_land_edh, "_open_edh_zarr", lambda _url: store)
    plugin = ERA5LandEDHMonthlyFromDailyPlugin(
        variable="tp", temporal_aggregation="mean", unit_transform="metres_to_mm"
    )

    month = plugin.fetch_period("2020-01", _EDH_BBOX)

    assert float(month.tp.mean()) == pytest.approx(2.0)


def test_edh_longitude_is_remapped_so_a_bbox_across_greenwich_runs_west_to_east(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _edh_store("t2m", "2020-01-31")
    monkeypatch.setattr(era5_land_edh, "_open_edh_zarr", lambda _url: store)
    plugin = ERA5LandEDHMonthlyFromDailyPlugin(variable="t2m", temporal_aggregation="mean")

    month = plugin.fetch_period("2020-01", _EDH_BBOX)

    assert month.x.values.tolist() == [-1.0, -0.5, 0.0, 0.5]
    assert month.y.values.tolist() == [10.0, 9.5]


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        ("2020-03-15", ["2020-01", "2020-02"]),  # March is still incomplete
        ("2020-03-31", ["2020-01", "2020-02", "2020-03"]),
    ],
)
def test_edh_monthly_periods_offer_only_complete_months(
    monkeypatch: pytest.MonkeyPatch, latest: str, expected: list[str]
) -> None:
    store = _edh_store("t2m", latest)
    monkeypatch.setattr(era5_land_edh, "_open_edh_zarr", lambda _url: store)
    plugin = ERA5LandEDHMonthlyFromDailyPlugin(variable="t2m", temporal_aggregation="mean")

    assert asyncio.run(plugin.periods("2020-01", "2020-12")) == expected


def test_edh_rejects_an_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="temporal_aggregation"):
        ERA5LandEDHMonthlyFromDailyPlugin(variable="t2m", temporal_aggregation="median")
