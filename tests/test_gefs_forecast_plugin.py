"""GEFS forecast plugin: the reduction from an ensemble store to a forecast cube.

Every test runs against a synthetic stand-in for dynamical.org's store rather than the network,
so the fixture encodes the two upstream traits that shape the code: leads that widen from
3-hourly to 6-hourly partway through a run, and a final step that lands one instant past the
last whole day.
"""

import asyncio

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.datasets.gefs_forecast import GefsForecastPlugin
from open_climate_service.shared import forecast

INITS = np.array(["2026-01-01T00", "2026-01-02T00"], dtype="datetime64[ns]")
# Day 0 3-hourly (8 steps), day 1 6-hourly (4 steps), then one step into day 2. The widening
# sits immediately before the tail on purpose: a rule that compares step counts against the
# previous day discards day 1 here, while a rule that tests coverage keeps it.
LEAD_HOURS = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48])


def upstream(units: str = "kg m-2 s-1", variable: str = "precipitation_surface") -> xr.Dataset:
    """A miniature of the upstream store: (init, member, lead, latitude, longitude)."""
    leads = LEAD_HOURS.astype("timedelta64[h]").astype("timedelta64[ns]")
    members = np.arange(3)
    lat = np.array([1.0, 0.75, 0.5])  # descending, as upstream
    lon = np.array([30.0, 30.25])
    shape = (len(INITS), len(members), len(leads), len(lat), len(lon))
    # A ramp over members so quantiles are predictable: member m contributes m + 1.
    values = np.broadcast_to((members + 1).reshape(1, -1, 1, 1, 1), shape).astype("float32")
    valid = INITS[:, None] + leads[None, :]
    ds = xr.Dataset(
        {variable: (("init_time", "ensemble_member", "lead_time", "latitude", "longitude"), values.copy())},
        coords={
            "init_time": INITS,
            "ensemble_member": members,
            "lead_time": leads,
            "latitude": lat,
            "longitude": lon,
            "valid_time": (("init_time", "lead_time"), valid),
        },
    )
    # Upstream publishes the units, and the rate guard reads them.
    ds[variable].attrs["units"] = units
    return ds


def plugin(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> GefsForecastPlugin:
    kwargs.setdefault("variable", "precipitation_surface")
    made = GefsForecastPlugin(**kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(GefsForecastPlugin, "_open", lambda self: upstream())
    return made


BBOX = [30.0, 0.5, 30.25, 1.0]


def test_periods_lists_issue_times_within_the_range(monkeypatch: pytest.MonkeyPatch) -> None:
    made = plugin(monkeypatch)
    assert asyncio.run(made.periods("2026-01-01", "2026-01-02")) == ["2026-01-01 00:00:00", "2026-01-02 00:00:00"]
    assert asyncio.run(made.periods("2026-01-02", "2026-01-02")) == ["2026-01-02 00:00:00"]


def test_fetch_returns_a_forecast_cube_on_the_published_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, resample="mean").fetch_period("2026-01-01 00:00:00", BBOX)

    assert forecast.is_forecast_cube(ds)
    assert ds["precipitation_surface"].dims == (forecast.REFERENCE_DIM, forecast.LEAD_DIM, "y", "x")
    assert ds[forecast.REFERENCE_DIM].values.tolist() == [INITS[0].astype("datetime64[ns]").astype(int)]
    # Day 2 holds a single step, so only the two whole days survive.
    assert ds[forecast.LEAD_DIM].values.tolist() == [0, 1]


def test_a_complete_day_survives_the_widening_of_the_upstream_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Day 1 has half the steps of day 0 yet covers just as much of the day."""
    ds = plugin(monkeypatch, resample="mean").fetch_period("2026-01-01 00:00:00", BBOX)
    assert 1 in ds[forecast.LEAD_DIM].values.tolist()


def test_the_trailing_partial_day_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, resample="mean").fetch_period("2026-01-01 00:00:00", BBOX)
    valid = np.asarray(ds[forecast.VALID_COORD].values)[0]
    assert np.datetime64("2026-01-03", "ns") not in valid


def test_valid_time_is_the_issue_time_plus_the_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, resample="mean").fetch_period("2026-01-02 00:00:00", BBOX)
    valid = np.asarray(ds[forecast.VALID_COORD].values)
    assert valid.shape == (1, 2)
    assert [str(value)[:10] for value in valid[0]] == ["2026-01-02", "2026-01-03"]


def test_members_collapse_to_the_mean_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch).fetch_period("2026-01-01 00:00:00", BBOX)
    assert "ensemble_member" not in ds.dims
    assert "quantile" not in ds.dims
    # Members carry 1, 2 and 3.
    assert float(ds["precipitation_surface"].mean()) == pytest.approx(2.0)


def test_quantiles_become_an_axis_and_keep_the_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, quantiles=[0.0, 0.5, 1.0]).fetch_period("2026-01-01 00:00:00", BBOX)
    assert ds["quantile"].values.tolist() == [0.0, 0.5, 1.0]
    spread = ds["precipitation_surface"].isel(reference_time=0, lead_time=0, y=0, x=0)
    assert spread.values.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_accumulate_integrates_a_rate_into_a_daily_total(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, accumulate=True).fetch_period("2026-01-01 00:00:00", BBOX)
    variable = ds["precipitation_surface"]
    # mm/d, matching CHIRPS: a daily total in mm is the same number as the mean daily rate, and
    # the rate spelling says which period the total is over.
    assert variable.attrs["units"] == "mm/d"
    assert variable.attrs["cell_methods"] == "time: sum"
    # A constant mean rate of 2 kg m-2 s-1 over a day is 2 * 86400 mm.
    assert float(variable.isel(reference_time=0, lead_time=0, y=0, x=0)) == pytest.approx(2.0 * 86400)


def test_summing_a_rate_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    made = plugin(monkeypatch, resample="sum")
    with pytest.raises(ValueError, match="is a rate"):
        made.fetch_period("2026-01-01 00:00:00", BBOX)


def test_summing_a_non_rate_variable_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is about per-second units, not about summing."""
    monkeypatch.setattr(GefsForecastPlugin, "_open", lambda self: upstream(units="mm"))
    made = GefsForecastPlugin(variable="precipitation_surface", resample="sum")
    ds = made.fetch_period("2026-01-01 00:00:00", BBOX)
    # Day 0 holds 8 steps, each a member mean of 2.
    assert float(ds["precipitation_surface"].isel(reference_time=0, lead_time=0, y=0, x=0)) == pytest.approx(16.0)


def test_accumulate_requires_a_mean() -> None:
    with pytest.raises(ValueError, match="resample must be 'mean'"):
        GefsForecastPlugin(variable="precipitation_surface", accumulate=True, resample="max")


def test_unknown_resample_and_out_of_range_quantiles_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown resample"):
        GefsForecastPlugin(variable="temperature_2m", resample="median")
    with pytest.raises(ValueError, match="Quantiles must lie"):
        GefsForecastPlugin(variable="temperature_2m", quantiles=[0.5, 1.5])


def test_the_append_axis_is_the_issue_time() -> None:
    """Without this the framework appends along `t` and the two axes collapse into one."""
    assert GefsForecastPlugin.time_dim == forecast.REFERENCE_DIM


def test_latitude_is_subset_despite_descending_upstream_order(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch).fetch_period("2026-01-01 00:00:00", [30.0, 0.5, 30.25, 0.75])
    assert ds["y"].values.tolist() == [0.75, 0.5]


def test_max_lead_days_caps_the_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch, max_lead_days=1).fetch_period("2026-01-01 00:00:00", BBOX)
    assert ds[forecast.LEAD_DIM].values.tolist() == [0]


def test_cf_standard_names_are_published(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = plugin(monkeypatch).fetch_period("2026-01-01 00:00:00", BBOX)
    assert ds[forecast.REFERENCE_DIM].attrs["standard_name"] == "forecast_reference_time"
    assert ds[forecast.LEAD_DIM].attrs["standard_name"] == "forecast_period"
    assert ds[forecast.LEAD_DIM].attrs["units"] == "days"


def test_the_upstream_valid_time_coordinate_does_not_leak_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """`valid_time` is in get_time_dim's lookup list, so leaving it in makes the cube ambiguous."""
    ds = plugin(monkeypatch).fetch_period("2026-01-01 00:00:00", BBOX)
    assert "valid_time" not in ds.coords
    assert "valid_day" not in ds.coords
