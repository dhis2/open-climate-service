"""C3S seasonal anomaly plugin: the first forecast whose lead is counted in months.

The fixture stands in for what cfgrib returns from the CDS product — ``(number, forecastMonth,
latitude, longitude)`` with a scalar ``time`` issue coordinate — so no test needs credentials or
the network.
"""

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.datasets.c3s_seasonal import C3SSeasonalAnomalyPlugin
from open_climate_service.shared import forecast

BBOX = [32.0, -17.0, 36.0, -9.0]
ISSUED = np.datetime64("2026-07-01", "ns")


def upstream(*, variable: str = "t2a", members: int = 51, leads: int = 6, units: str = "K") -> xr.Dataset:
    """A miniature of one decoded GRIB run.

    ``forecastMonth`` starts at 1 upstream, where 1 is the month the run was issued in — the
    off-by-one the plugin has to normalise away.
    """
    number = np.arange(members)
    forecast_month = np.arange(1, leads + 1)
    lat = np.array([-9.0, -10.0])  # descending, as upstream
    lon = np.array([32.0, 33.0, 34.0])
    shape = (members, leads, len(lat), len(lon))
    # A ramp over members so quantiles are predictable: member m contributes m + 1.
    values = np.broadcast_to((number + 1).reshape(-1, 1, 1, 1), shape).astype("float32")
    ds = xr.Dataset(
        {variable: (("number", "forecastMonth", "latitude", "longitude"), values.copy())},
        coords={
            "number": number,
            "forecastMonth": forecast_month,
            "latitude": lat,
            "longitude": lon,
            "time": ISSUED,
            "surface": 0.0,
        },
    )
    ds[variable].attrs["units"] = units
    return ds


def plugin(
    monkeypatch: pytest.MonkeyPatch, *, upstream_ds: xr.Dataset | None = None, **kwargs: object
) -> tuple[C3SSeasonalAnomalyPlugin, dict[str, Any]]:
    kwargs.setdefault("variable", "2m_temperature_anomaly")
    made = C3SSeasonalAnomalyPlugin(**kwargs)  # type: ignore[arg-type]
    source = upstream_ds if upstream_ds is not None else upstream()
    captured: dict[str, Any] = {}

    def fake_fetch(self: C3SSeasonalAnomalyPlugin, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        captured["period_id"] = period_id
        issued = np.datetime64(np.asarray(source["time"].values).item(), "ns").astype("datetime64[M]")
        reduced = self._reduce_members(source)
        return self._as_forecast_cube(reduced, issued)

    monkeypatch.setattr(C3SSeasonalAnomalyPlugin, "fetch_period", fake_fetch)
    return made, captured


@pytest.fixture
def published_through(monkeypatch: pytest.MonkeyPatch) -> Callable[[date], None]:
    """Pin the CDS availability lookup so tests never depend on the catalogue or the clock."""

    def _set(cutoff: date) -> None:
        monkeypatch.setattr(
            "open_climate_service.plugins.datasets.c3s_seasonal._availability_cutoff",
            lambda: cutoff,
        )

    _set(date(2026, 8, 1))
    return _set


def test_periods_enumerates_issue_months(published_through: Callable[[date], None]) -> None:
    made = C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly")
    assert asyncio.run(made.periods("2026-05", "2026-08")) == ["2026-05", "2026-06", "2026-07", "2026-08"]


def test_periods_clamp_to_the_start_of_the_archive(published_through: Callable[[date], None]) -> None:
    """The anomalies product begins in 2017; asking earlier is a request for nothing."""
    made = C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly")
    assert asyncio.run(made.periods("2015-11", "2017-02")) == ["2017-01", "2017-02"]


def test_periods_cross_a_year_boundary(published_through: Callable[[date], None]) -> None:
    made = C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly")
    assert asyncio.run(made.periods("2025-11", "2026-02")) == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_periods_clip_to_the_latest_published_run(published_through: Callable[[date], None]) -> None:
    """`temporal_direction: future` hands the plugin a horizon a year ahead.

    The framework leaves clipping to the plugin, so enumerating the calendar to that horizon
    would ask for twelve runs that do not exist and fail twelve times over.
    """
    published_through(date(2026, 8, 1))
    made = C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly")
    assert asyncio.run(made.periods("2026-06", "2027-08")) == ["2026-06", "2026-07", "2026-08"]


def test_periods_are_empty_after_the_latest_run(published_through: Callable[[date], None]) -> None:
    published_through(date(2026, 8, 1))
    made = C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly")
    assert asyncio.run(made.periods("2026-09", "2027-01")) == []


def test_fetch_returns_a_forecast_cube_with_a_month_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)

    assert forecast.is_forecast_cube(ds)
    assert ds["t2a"].dims == (forecast.REFERENCE_DIM, forecast.LEAD_DIM, "y", "x")
    assert forecast.lead_unit(ds) == "month"


def test_the_upstream_lead_is_normalised_to_zero_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """C3S numbers forecastMonth from 1 for the issue month; 0-based keeps one valid-time rule."""
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    assert list(forecast.lead_values(ds)) == [0, 1, 2, 3, 4, 5]


def test_valid_times_are_the_six_months_from_the_issue_month(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    months = [str(value)[:7] for value in np.asarray(ds[forecast.VALID_COORD].values)[0]]
    assert months == ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]


def test_quantiles_replace_the_members_and_keep_the_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch, quantiles=[0.0, 0.5, 1.0], upstream_ds=upstream(members=51))
    ds = made.fetch_period("2026-07", BBOX)
    assert "number" not in ds.dims
    spread = ds["t2a"].isel(reference_time=0, lead_time=0, y=0, x=0)
    # Members carry 1..51.
    assert spread.values.tolist() == pytest.approx([1.0, 26.0, 51.0])


def test_a_hindcast_sized_ensemble_lands_on_the_same_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-time runs carry 51 members and hindcasts 25 — quantiles must not notice."""
    made, _ = plugin(monkeypatch, quantiles=[0.0, 0.5, 1.0], upstream_ds=upstream(members=25))
    ds = made.fetch_period("2026-07", BBOX)
    assert ds["quantile"].values.tolist() == [0.0, 0.5, 1.0]
    spread = ds["t2a"].isel(reference_time=0, lead_time=0, y=0, x=0)
    assert spread.values.tolist() == pytest.approx([1.0, 13.0, 25.0])


def test_members_average_when_no_quantiles_are_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch, upstream_ds=upstream(members=3))
    ds = made.fetch_period("2026-07", BBOX)
    assert "quantile" not in ds.dims
    assert float(ds["t2a"].mean()) == pytest.approx(2.0)


def test_a_rate_anomaly_is_converted_to_mm_per_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """A precipitation anomaly arrives as m s-1, which is unusable in a dashboard."""
    made, _ = plugin(
        monkeypatch,
        variable="total_precipitation_anomalous_rate_of_accumulation",
        unit_transform="metres_per_second_to_mm_per_day",
        upstream_ds=upstream(variable="tpara", members=1, units="m s**-1"),
    )
    ds = made.fetch_period("2026-07", BBOX)
    assert float(ds["tpara"].isel(reference_time=0, lead_time=0, y=0, x=0)) == pytest.approx(86400000.0)
    assert ds["tpara"].attrs["units"] == "mm/d"


def test_a_temperature_anomaly_is_relabelled_not_shifted(monkeypatch: pytest.MonkeyPatch) -> None:
    """One kelvin of difference is one degree Celsius of difference.

    Subtracting 273.15 — what `kelvin_to_celsius` does, correctly, for an absolute temperature —
    would report a +1.4 K anomaly as −271.75 °C.
    """
    made, _ = plugin(
        monkeypatch,
        unit_transform="kelvin_difference_to_celsius",
        upstream_ds=upstream(members=1, units="K"),
    )
    ds = made.fetch_period("2026-07", BBOX)
    assert float(ds["t2a"].isel(reference_time=0, lead_time=0, y=0, x=0)) == pytest.approx(1.0)
    assert ds["t2a"].attrs["units"] == "degC"


def test_the_store_holds_the_units_the_template_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Converted at ingest, as the ERA5-Land plugins do, so no consumer has to know the source's."""
    made, _ = plugin(monkeypatch, unit_transform="kelvin_difference_to_celsius")
    ds = made.fetch_period("2026-07", BBOX)
    assert ds["t2a"].attrs["units"] == "degC"


def test_an_unknown_unit_transform_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch, unit_transform="fahrenheit")
    with pytest.raises(ValueError, match="Unknown unit_transform"):
        made.fetch_period("2026-07", BBOX)


def test_no_unit_transform_leaves_the_upstream_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    assert ds["t2a"].attrs["units"] == "K"


def test_cfgrib_scalars_do_not_leak_into_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """`valid_time` is in get_time_dim's lookup list, so leaving it makes the cube ambiguous."""
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    for name in ("time", "surface", "step", "valid_time", "number"):
        assert name not in ds.coords


def test_cf_standard_names_are_published(monkeypatch: pytest.MonkeyPatch) -> None:
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    assert ds[forecast.REFERENCE_DIM].attrs["standard_name"] == "forecast_reference_time"
    assert ds[forecast.LEAD_DIM].attrs["standard_name"] == "forecast_period"


def test_max_lead_months_is_bounded() -> None:
    with pytest.raises(ValueError, match="max_lead_months"):
        C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly", max_lead_months=7)
    with pytest.raises(ValueError, match="Quantiles must lie"):
        C3SSeasonalAnomalyPlugin(variable="2m_temperature_anomaly", quantiles=[-0.1])


def test_the_append_axis_is_the_issue_time() -> None:
    assert C3SSeasonalAnomalyPlugin.time_dim == forecast.REFERENCE_DIM


def test_the_store_is_chunked_one_run_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ragged chunks are rejected outright by `to_zarr`."""
    made, _ = plugin(monkeypatch)
    ds = made.fetch_period("2026-07", BBOX)
    assert ds["t2a"].chunksizes[forecast.REFERENCE_DIM] == (1,)
