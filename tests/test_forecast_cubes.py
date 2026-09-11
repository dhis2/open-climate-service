"""Forecast cubes: two temporal axes, and what each surface must do with them.

The recurring failure these pin is a forecast being described by the wrong axis — coverage
reporting when runs were made rather than what they cover, or a request that selects runs being
compared against the dates those runs speak to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("icechunk")

from open_climate_service.data_manager.services.utils import get_time_dim  # noqa: E402
from open_climate_service.shared import forecast  # noqa: E402


def _cube(inits: list[str], leads: int = 5, *, with_valid_coord: bool = True):  # `xr` is an importorskip variable
    ny, nx = 4, 3
    reference = np.array(inits, dtype="datetime64[ns]")
    lead = np.arange(1, leads + 1, dtype="int32")
    data = np.arange(len(inits) * leads * ny * nx, dtype="float32").reshape(len(inits), leads, ny, nx)
    coords: dict = {
        forecast.REFERENCE_DIM: reference,
        forecast.LEAD_DIM: lead,
        "y": np.linspace(-9.0, -12.0, ny),
        "x": np.linspace(32.0, 34.0, nx),
    }
    if with_valid_coord:
        offsets = lead.astype("timedelta64[D]").astype("timedelta64[ns]")
        coords[forecast.VALID_COORD] = (
            (forecast.REFERENCE_DIM, forecast.LEAD_DIM),
            reference[:, None] + offsets[None, :],
        )
    return xr.Dataset({"tfc": ((forecast.REFERENCE_DIM, forecast.LEAD_DIM, "y", "x"), data)}, coords=coords)


def test_both_axes_are_required_to_be_a_forecast_cube() -> None:
    """One axis alone is a different thing, not a partial forecast.

    A cube with only `lead_time` is a single run that has lost its issue time; one with only
    `reference_time` is a time series whose axis is misnamed. Treating either as an archive
    would produce a coverage or a slider that means something other than it says.
    """
    assert forecast.is_forecast_cube(_cube(["2026-03-01"]))
    assert not forecast.is_forecast_cube(_cube(["2026-03-01"]).isel({forecast.LEAD_DIM: 0}, drop=True))
    assert not forecast.is_forecast_cube(_cube(["2026-03-01"]).isel({forecast.REFERENCE_DIM: 0}, drop=True))
    assert not forecast.is_forecast_cube(xr.Dataset({"v": ("t", [1.0])}, coords={"t": [np.datetime64("2026-01-01")]}))


def test_get_time_dim_refuses_a_forecast_cube() -> None:
    """The deliberate loud failure: neither axis means "the period this value describes".

    Every module that resolves a time dimension has to opt in, rather than silently keying on
    the issue times and reporting when forecasts were made.
    """
    with pytest.raises(ValueError, match="Unable to find time dimension"):
        get_time_dim(_cube(["2026-03-01"]))


def test_valid_time_is_recomputed_when_the_store_did_not_publish_it() -> None:
    published = forecast.valid_time(_cube(["2026-03-01"], leads=3))
    derived = forecast.valid_time(_cube(["2026-03-01"], leads=3, with_valid_coord=False))
    assert derived.dims == (forecast.REFERENCE_DIM, forecast.LEAD_DIM)
    np.testing.assert_array_equal(np.asarray(derived.values), np.asarray(published.values))
    # Lead 1 from a 1 March run is 2 March, not 1 March.
    assert str(np.asarray(derived.values)[0, 0])[:10] == "2026-03-02"


def test_coverage_bounds_span_the_forecast_horizon_not_the_issue_times() -> None:
    """Three daily runs reaching five days ahead cover past the last run, by design."""
    first, last = forecast.valid_time_bounds(_cube(["2026-03-01", "2026-03-02", "2026-03-03"], leads=5))
    assert str(first)[:10] == "2026-03-02"
    assert str(last)[:10] == "2026-03-08"


def test_latest_reference_view_is_an_ordinary_time_cube() -> None:
    """The escape hatch: existing consumers work on a slice with no forecast awareness."""
    view = forecast.latest_reference_view(_cube(["2026-03-01", "2026-03-02", "2026-03-03"], leads=4))

    assert "t" in view.dims and forecast.LEAD_DIM not in view.dims
    assert view.sizes["t"] == 4
    assert get_time_dim(view) == "t"  # resolvable again
    # The latest run, and its valid times run from the day after it.
    assert str(np.asarray(view[forecast.REFERENCE_DIM].values))[:10] == "2026-03-03"
    assert str(np.asarray(view["t"].values)[0])[:10] == "2026-03-04"
    # Lead survives as a coordinate, so "how far ahead was this" stays answerable.
    assert forecast.LEAD_DIM in view.coords


def test_latest_reference_view_can_select_an_earlier_run() -> None:
    """What verification needs: the forecast as it stood on a given day."""
    cube = _cube(["2026-03-01", "2026-03-02", "2026-03-03"], leads=4)
    view = forecast.latest_reference_view(cube, reference="2026-03-01")
    assert str(np.asarray(view["t"].values)[0])[:10] == "2026-03-02"


def test_latest_reference_view_passes_a_plain_cube_through() -> None:
    plain = xr.Dataset(
        {"v": ("t", [1.0, 2.0])}, coords={"t": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")}
    )
    assert forecast.latest_reference_view(plain) is plain


def test_period_bounds_are_dataset_native_period_strings() -> None:
    start, end = forecast.period_bounds_as_strings(_cube(["2026-03-01", "2026-03-02"], leads=3), "daily")
    assert (start, end) == ("2026-03-02", "2026-03-05")


def test_coverage_reports_the_horizon_and_the_issue_times_separately(tmp_path: Path) -> None:
    """Coverage answers "what does it cover"; the scope check needs "which runs are these".

    Conflating them rejects every forecast ingest, since coverage legitimately reaches past the
    requested window.
    """
    import icechunk

    from open_climate_service.data_accessor.services.accessor import _coverage_from_dataset

    cube = _cube(["2026-03-01", "2026-03-02", "2026-03-03"], leads=5)
    store = tmp_path / "fc.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(store)))
    session = repo.writable_session("main")
    cube.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("fc")

    result = _coverage_from_dataset(ds=cube, period_type="daily")

    assert result["coverage"]["temporal"] == {"start": "2026-03-02", "end": "2026-03-08"}
    assert result["forecast_reference"] == {"start": "2026-03-01", "end": "2026-03-03"}


def test_coverage_omits_the_issue_times_for_a_plain_cube() -> None:
    """`forecast_reference` present would make the scope check use the wrong axis."""
    from open_climate_service.data_accessor.services.accessor import _coverage_from_dataset

    plain = xr.Dataset(
        {"v": (("t", "y", "x"), np.zeros((2, 4, 3), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
            "y": np.linspace(-9.0, -12.0, 4),
            "x": np.linspace(32.0, 34.0, 3),
        },
    )
    result = _coverage_from_dataset(ds=plain, period_type="daily")
    assert result["forecast_reference"] is None
    assert result["coverage"]["temporal"] == {"start": "2026-01-01", "end": "2026-01-02"}


def test_stac_declares_both_axes_with_the_issue_times_as_the_temporal_one() -> None:
    from open_climate_service.stac.services import _build_forecast_dimensions

    dims = _build_forecast_dimensions(_cube(["2026-03-01", "2026-03-02"], leads=3))

    reference = dims[forecast.REFERENCE_DIM]
    assert reference["type"] == "temporal"
    assert reference["extent"] == ["2026-03-01T00:00:00Z", "2026-03-02T00:00:00Z"]
    assert len(reference["values"]) == 2

    lead = dims[forecast.LEAD_DIM]
    assert lead["values"] == [1, 2, 3]
    assert lead["unit"] == "day"
    # Without the hint a viewer offers a dropdown of horizons rather than stepping them.
    assert lead["open_climate_service:control"] == "slider"


def test_stac_declares_nothing_for_a_plain_cube() -> None:
    plain = xr.Dataset({"v": ("t", [1.0])}, coords={"t": np.array(["2026-01-01"], dtype="datetime64[ns]")})
    from open_climate_service.stac.services import _build_forecast_dimensions

    assert _build_forecast_dimensions(plain) == {}


def _timedelta_lead_cube(inits: list[str], leads: int = 3):
    """A cube whose lead axis is ``timedelta64`` — what a reader gets back from a store.

    A lead is written as integer days with ``units: days``, the CF encoding for
    ``forecast_period``, and xarray decodes that back to ``timedelta64``. So both spellings of
    the same axis occur in normal operation and every consumer meets both.
    """
    cube = _cube(inits, leads=leads, with_valid_coord=False)
    offsets = np.asarray(cube[forecast.LEAD_DIM].values, dtype="int64").astype("timedelta64[D]")
    return cube.assign_coords({forecast.LEAD_DIM: offsets.astype("timedelta64[ns]")})


def test_lead_values_reads_both_spellings_of_the_lead_axis() -> None:
    """The axis is integer days in a plugin and timedelta64 in a reader."""
    assert list(forecast.lead_values(_cube(["2026-03-01"], leads=3))) == [1, 2, 3]
    assert list(forecast.lead_values(_timedelta_lead_cube(["2026-03-01"], leads=3))) == [1, 2, 3]


def test_stac_publishes_days_not_nanoseconds() -> None:
    """`int()` on a decoded lead axis publishes 86400000000000 against `"unit": "day"`."""
    from open_climate_service.stac.services import _build_forecast_dimensions

    dims = _build_forecast_dimensions(_timedelta_lead_cube(["2026-03-01"], leads=3))
    assert dims[forecast.LEAD_DIM]["values"] == [1, 2, 3]


def test_valid_time_recompute_handles_a_decoded_lead_axis() -> None:
    """Taking timedelta nanoseconds as a count of days lands ~274,000 years out."""
    cube = _timedelta_lead_cube(["2026-03-01"], leads=3)
    valid = forecast.valid_time(cube)
    assert [str(value)[:10] for value in np.asarray(valid.values).ravel()] == [
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
    ]


def test_the_layout_contract_accepts_a_forecast_cube() -> None:
    """A forecast store has no `t` by design, so the contract must not read that as a defect."""
    from open_climate_service.shared.raster_contract import ContractViolation, published_contract_violations

    kinds = [v.kind for v in published_contract_violations(_cube(["2026-03-01"]))]
    assert ContractViolation.TEMPORAL_DIM_NAME not in kinds


def test_the_contract_still_requires_t_on_a_plain_cube() -> None:
    from open_climate_service.shared.raster_contract import ContractViolation, published_contract_violations

    plain = xr.Dataset(
        {"v": (("time", "y", "x"), np.zeros((1, 4, 3), dtype="float32"))},
        coords={
            "time": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": np.linspace(-9.0, -12.0, 4),
            "x": np.linspace(32.0, 34.0, 3),
        },
    )
    kinds = [v.kind for v in published_contract_violations(plain)]
    assert ContractViolation.TEMPORAL_DIM_NAME in kinds


def test_committed_periods_of_a_forecast_store_are_its_issue_times(tmp_path: Path) -> None:
    """Read against `t`, a forecast store looks empty — and sync then takes the coverage
    horizon as "already ingested", which is 35 days ahead, so no new run is ever due."""
    import icechunk

    from open_climate_service.streaming.store import read_committed_period_ids

    cube = _cube(["2026-03-01", "2026-03-02"], leads=5)
    store = tmp_path / "fc.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(store)))
    session = repo.writable_session("main")
    cube.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("fc")

    assert read_committed_period_ids(store, "daily") == {"2026-03-01", "2026-03-02"}


def _month_lead_cube(inits: list[str], leads: int = 6):
    """A cube whose lead axis counts months — a seasonal forecast rather than a daily one."""
    cube = _cube(inits, leads=leads, with_valid_coord=False)
    zero_based = np.arange(leads, dtype="int32")
    cube = cube.assign_coords({forecast.LEAD_DIM: zero_based})
    cube[forecast.LEAD_DIM].attrs["units"] = "months"
    return cube


def test_the_lead_unit_comes_from_the_axis_not_from_the_code() -> None:
    assert forecast.lead_unit(_cube(["2026-03-01"])) == "day"  # no units attr: days
    assert forecast.lead_unit(_month_lead_cube(["2026-03-01"])) == "month"


def test_an_unknown_lead_unit_raises_rather_than_defaulting() -> None:
    """Treating an unrecognised unit as days would put a six-month outlook inside one week."""
    cube = _month_lead_cube(["2026-03-01"])
    cube[forecast.LEAD_DIM].attrs["units"] = "fortnights"
    with pytest.raises(ValueError, match="Unsupported lead_time unit"):
        forecast.lead_unit(cube)


def test_month_leads_use_calendar_arithmetic() -> None:
    """A month is not a fixed duration, so a timedelta cannot express it.

    From 31 January, five successive month-steps land on the 31st of each month that has one and
    on the last of those that do not — never in the following month, which is what adding 30 or
    31 days would do.
    """
    grid = forecast.valid_times_for(np.array(["2026-01-31"], dtype="datetime64[ns]"), np.arange(6), "month")
    assert [str(value)[:7] for value in grid[0]] == [
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    ]


def test_valid_time_of_a_seasonal_cube_is_the_six_months_from_the_issue_month() -> None:
    cube = _month_lead_cube(["2026-07-01"], leads=6)
    months = [str(value)[:7] for value in np.asarray(forecast.valid_time(cube).values).ravel()]
    assert months == ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]


def test_coverage_of_a_seasonal_cube_spans_the_months_it_describes() -> None:
    lo, hi = forecast.valid_time_bounds(_month_lead_cube(["2026-07-01"], leads=6))
    assert str(lo)[:7] == "2026-07"
    assert str(hi)[:7] == "2026-12"


def test_stac_declares_the_lead_unit_it_finds() -> None:
    from open_climate_service.stac.services import _build_forecast_dimensions

    daily = _build_forecast_dimensions(_cube(["2026-03-01"], leads=3))[forecast.LEAD_DIM]
    seasonal = _build_forecast_dimensions(_month_lead_cube(["2026-07-01"], leads=6))[forecast.LEAD_DIM]
    assert daily["unit"] == "day"
    assert seasonal["unit"] == "month"
    assert seasonal["values"] == [0, 1, 2, 3, 4, 5]
    assert "Months ahead" in seasonal["description"]
