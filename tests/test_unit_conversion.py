import logging

import pytest
import xarray as xr

from open_climate_service.transforms.unit_conversion import amount_as_rate, is_accumulated


@pytest.mark.parametrize(("step_type", "expected"), [("accum", True), ("instant", False)])
def test_is_accumulated_reads_the_grib_step_type(step_type: str, expected: bool) -> None:
    da = xr.DataArray(1.0, name="v", attrs={"GRIB_stepType": step_type})

    assert is_accumulated(da) is expected


def test_a_missing_step_type_is_instantaneous_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert is_accumulated(xr.DataArray(1.0, name="v")) is False

    assert [r.getMessage() for r in caplog.records] == ["v declares no GRIB_stepType; treating it as instantaneous"]


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("mm", "mm d-1"),
        ("J m**-2", "J m**-2 d-1"),  # an amount, though a joule carries time in its base dimensions
        ("kg m-2", "kg m-2 d-1"),
        ("kg m-2 s-1", "kg m-2 s-1"),  # already a rate
        ("mm/d", "mm/d"),
        ("W m-2", "W m-2"),  # a watt is J/s, so already a rate
        ("m of water equivalent", "m of water equivalent"),  # not parseable, so left as declared
    ],
)
def test_amount_as_rate_labels_only_amounts(units: str, expected: str) -> None:
    ds = xr.Dataset({"v": ((), 1.0, {"units": units})})

    out = amount_as_rate(ds, "v", "d")

    assert out["v"].attrs["units"] == expected
    assert float(out["v"]) == 1.0


def test_amount_as_rate_leaves_a_variable_without_units() -> None:
    ds = xr.Dataset({"v": ((), 1.0)})

    assert "units" not in amount_as_rate(ds, "v", "d")["v"].attrs
