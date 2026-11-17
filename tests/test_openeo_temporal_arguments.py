"""Scalar openEO date arguments must reach processes timezone-naive (CLIM-826).

The pg-parser validates any date-typed process argument into a pydantic RootModel wrapping
a *timezone-aware* pendulum DateTime. Our stores carry timezone-naive time coordinates, so
a process slicing with such a value fails inside pandas. `temporal_extent` was already
normalised at the load_collection boundary; scalar arguments such as spi's
cal_start/cal_end were not.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import Date, ParameterReference, TemporalInterval
from pydantic import ValidationError

from open_geo_service.openeo import execution
from open_geo_service.openeo.execution import _naive_temporal_scalar


@pytest.fixture
def registry() -> object:
    """A freshly built process registry (the module keeps a lazy singleton)."""
    execution._registry = None
    try:
        yield execution._build_process_registry()
    finally:
        execution._registry = None


def _monthly_precip(periods: int = 60) -> xr.DataArray:
    """Monthly precipitation on a timezone-naive time axis, as published stores are."""
    return xr.DataArray(
        np.random.default_rng(0).gamma(2.0, 20.0, size=(periods, 2, 2)).astype("float32"),
        dims=("t", "y", "x"),
        coords={
            "t": xr.date_range("1991-01-01", periods=periods, freq="MS"),
            "y": [1.0, 2.0],
            "x": [3.0, 4.0],
        },
        name="pr",
        attrs={"units": "mm/d", "standard_name": "precipitation_flux"},
    )


# --- the normalisation itself ---------------------------------------------------------


def test_date_becomes_a_plain_day() -> None:
    """A Date is a day, and as a slice bound "2020-12-31" covers the whole of it."""
    assert _naive_temporal_scalar(Date.model_validate("2020-12-31")) == "2020-12-31"


def test_strips_the_timezone_not_the_time() -> None:
    """Non-Date wrappers keep their time component, minus the zone."""
    value = TemporalInterval.model_validate(["1991-01-01T06:30:00Z", None]).root[0]
    assert _naive_temporal_scalar(value) == "1991-01-01T06:30:00"


def test_temporal_interval_is_left_alone() -> None:
    """Its root is a list; load_collection normalises it separately."""
    interval = TemporalInterval.model_validate(["1991-01-01", "2020-12-31"])
    assert _naive_temporal_scalar(interval) is interval


@pytest.mark.parametrize("value", [None, 3, "1991-01-01", 2.5, True, [1, 2], {"a": 1}])
def test_non_temporal_values_pass_through(value: object) -> None:
    assert _naive_temporal_scalar(value) is value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1991-01-01T00:00:00+00:00", "1991-01-01T00:00:00"),
        ("1991-01-01T00:00:00Z", "1991-01-01T00:00:00"),
        ("2020-06-01T12:30:00+02:00", "2020-06-01T12:30:00"),
    ],
)
def test_tz_qualified_iso_strings_lose_the_offset(value: str, expected: str) -> None:
    """These reach a process as raw strings, because the pg-parser will not coerce them.

    Only the plain forms ("1991-01-01") validate as a Date. A fully-qualified timestamp
    fails that validation, stays a string, and then fails inside xclim with
    "Both dates must have the same UTC offset" — so it has to be handled here too.
    """
    assert _naive_temporal_scalar(value) == expected


@pytest.mark.parametrize("value", ["MS", "YS", "1991", "07-01", "mm/d", "degC", "1 mm/day", "dayofyear"])
def test_ordinary_string_arguments_are_untouched(value: str) -> None:
    """The ISO-with-offset rule must not catch frequencies, units or thresholds."""
    assert _naive_temporal_scalar(value) is value


@pytest.mark.parametrize("value", ["2020-12-31", "2020-06-01T12:00:00"])
def test_naive_iso_strings_are_returned_unchanged(value: str) -> None:
    """Nothing to strip, so don't reformat — "2020-12-31" must not become a timestamp."""
    assert _naive_temporal_scalar(value) is value


def test_cube_arguments_pass_through() -> None:
    """A datacube has no `root`; it must not be touched."""
    cube = _monthly_precip(3)
    assert _naive_temporal_scalar(cube) is cube


@pytest.mark.parametrize("value", ["07-01", "0701", "12-25"])
def test_day_of_year_strings_are_left_intact(value: str) -> None:
    """Twenty of the date-argument indicators take MM-DD, not a full date.

    `after_date`, `mid_date`, `start_date` and friends expect a day-of-year string. Those
    never validate as a pg-parser Date, so they arrive as plain strings and must reach
    xclim unchanged — normalising them into a full date would break those indicators.
    """
    with pytest.raises(ValidationError):
        Date.model_validate(value)
    assert _naive_temporal_scalar(value) is value


# --- the crash this fixes -------------------------------------------------------------


def test_spi_accepts_tz_aware_calibration_bounds(registry: object) -> None:
    """The reported failure: TypeError from pandas on a tz-naive DatetimeIndex."""
    result = registry["spi"].implementation(  # type: ignore[index]
        pr=_monthly_precip(),
        window=3,
        cal_start=Date.model_validate("1991-01-01"),
        cal_end=Date.model_validate("2020-12-31"),
        freq="MS",
    )
    assert result.dims == ("t", "y", "x")


def test_spi_still_accepts_plain_string_bounds(registry: object) -> None:
    """Strings were never broken; make sure normalisation did not change that."""
    result = registry["spi"].implementation(  # type: ignore[index]
        pr=_monthly_precip(), window=3, cal_start="1991-01-01", cal_end="2020-12-31", freq="MS"
    )
    assert result.dims == ("t", "y", "x")


def test_tz_aware_and_string_bounds_agree(registry: object) -> None:
    """Both spellings must produce the same answer, not merely both succeed."""
    kwargs = {"pr": _monthly_precip(), "window": 3, "freq": "MS"}
    from_dates = registry["spi"].implementation(  # type: ignore[index]
        cal_start=Date.model_validate("1991-01-01"), cal_end=Date.model_validate("2020-12-31"), **kwargs
    )
    from_strings = registry["spi"].implementation(  # type: ignore[index]
        cal_start="1991-01-01", cal_end="2020-12-31", **kwargs
    )
    xr.testing.assert_allclose(from_dates, from_strings)


def test_date_arriving_via_from_parameter_is_normalised(registry: object) -> None:
    """The case that decides where the wrapper sits.

    A workflow passing a date through `{"from_parameter": ...}` is still a
    ParameterReference when the outer wrapper is entered, and only becomes a Date once that
    wrapper resolves it. Normalising outside openeo-processes-dask's wrapper would miss it.
    """
    result = registry["spi"].implementation(  # type: ignore[index]
        pr=_monthly_precip(),
        window=3,
        cal_start=ParameterReference(from_parameter="cal_start"),
        cal_end=ParameterReference(from_parameter="cal_end"),
        freq="MS",
        named_parameters={
            "cal_start": Date.model_validate("1991-01-01"),
            "cal_end": Date.model_validate("2020-12-31"),
        },
    )
    assert result.dims == ("t", "y", "x")


# --- the wrapper must stay transparent ------------------------------------------------


def test_wrapper_preserves_the_wrapped_signature() -> None:
    """openeo-processes-dask decides whether to forward axis/keepdims/context from
    inspect.signature of what it wraps. An opaque wrapper would make every process look
    like it accepts them, and they would be forwarded to implementations that reject them.
    """
    import inspect

    from open_geo_service.openeo.execution import _normalise_temporal_arguments

    def implementation(data: object, axis: object = None) -> None: ...

    def without_axis(data: object) -> None: ...

    assert "axis" in inspect.signature(_normalise_temporal_arguments(implementation)).parameters
    assert "axis" not in inspect.signature(_normalise_temporal_arguments(without_axis)).parameters


def test_standard_processes_still_execute(registry: object) -> None:
    """A reducer that relies on the special-argument filtering above."""
    cube = _monthly_precip(12)
    result = registry["mean"].implementation(data=cube, axis=0)  # type: ignore[index]
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    "cal_start",
    [
        "1991-01-01",
        "1991-01-01T00:00:00+00:00",
        "1991-01-01T00:00:00Z",
        "1991",
        None,
    ],
)
def test_spi_accepts_every_bound_form_the_instance_override_handled(registry: object, cal_start: object) -> None:
    """Parity check against the Uganda instance override this replaces.

    Its `_to_date_str` accepted None, a plain string, a tz-suffixed string and a
    pendulum/datetime object. All of those must work through the core process before that
    override can be deleted — the tz-suffixed strings in particular, which the pg-parser
    does not coerce into a Date and which the first version of this fix let through.
    """
    result = registry["spi"].implementation(  # type: ignore[index]
        pr=_monthly_precip(), window=3, cal_start=cal_start, cal_end="2020-12-31", freq="MS"
    )
    assert result.dims == ("t", "y", "x")


def test_temporal_extent_normalisation_is_unaffected() -> None:
    """load_collection's own path must keep working — this fix is additive to it."""
    from open_geo_service.openeo.execution import _temporal_to_list

    assert _temporal_to_list(TemporalInterval.model_validate(["1991-01-01", "2020-12-31"])) == [
        "1991-01-01T00:00:00",
        "2020-12-31T00:00:00",
    ]
    assert _temporal_to_list(None) is None
