"""Tests for CF-metadata stamping and units validation (issue #280)."""

from __future__ import annotations

import re

import numpy as np
import pytest
import xarray as xr

from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.shared.cf import apply_cf_metadata, cf_attrs_from_template, validate_units


def test_cf_attrs_from_template_extracts_cf_fields() -> None:
    template = {
        "units": "mm",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "cell_methods": "time: sum",
        "name": "x",
    }
    assert cf_attrs_from_template(template) == {
        "units": "mm",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "cell_methods": "time: sum",
        "long_name": "x",
    }
    assert cf_attrs_from_template(None) == {}
    assert cf_attrs_from_template({"units": ""}) == {"units": ""}  # dimensionless kept


def test_the_template_names_and_describes_the_variable() -> None:
    template = {"name": "Heat-stress days", "description": "Days above\n  30 degrees C.\n"}
    assert cf_attrs_from_template(template) == {
        "long_name": "Heat-stress days",
        "description": "Days above 30 degrees C.",
    }


def test_a_null_standard_name_removes_one_left_on_the_variable() -> None:
    attrs = cf_attrs_from_template({"standard_name": None})
    da = xr.DataArray([1.0], dims=["t"], attrs={"standard_name": "air_temperature", "units": "d"})

    apply_cf_metadata(da, attrs)
    assert da.attrs["standard_name"] == "air_temperature"  # only an overwrite removes it

    apply_cf_metadata(da, attrs, overwrite=True)
    assert "standard_name" not in da.attrs
    assert da.attrs["units"] == "d"


def test_apply_cf_metadata_sets_and_preserves() -> None:
    da = xr.DataArray(np.zeros((2, 2), dtype="float32"), dims=("y", "x"))
    apply_cf_metadata(da, {"units": "mm", "standard_name": "lwe_thickness_of_precipitation_amount"})
    assert da.attrs["units"] == "mm"
    assert da.attrs["standard_name"] == "lwe_thickness_of_precipitation_amount"

    # existing attrs win unless overwrite=True
    da2 = xr.DataArray(np.zeros(2), dims=("x",), attrs={"units": "kg m-2 s-1"})
    apply_cf_metadata(da2, {"units": "mm"})
    assert da2.attrs["units"] == "kg m-2 s-1"
    apply_cf_metadata(da2, {"units": "mm"}, overwrite=True)
    assert da2.attrs["units"] == "mm"


def test_apply_cf_metadata_targets_dataset_variable() -> None:
    ds = xr.Dataset({"precip": (("y", "x"), np.zeros((1, 1), dtype="float32"))}, coords={"y": [0.0], "x": [0.0]})
    apply_cf_metadata(ds, {"units": "mm"}, variable="precip")
    assert ds["precip"].attrs["units"] == "mm"


@pytest.mark.parametrize("units", ["mm", "mm/d", "degC", "kg m-2 s-1", ""])
def test_validate_units_accepts_valid(units: str) -> None:
    assert validate_units(units) is None


@pytest.mark.parametrize("units", ["people", "not-a-unit"])
def test_validate_units_rejects_invalid(units: str) -> None:
    msg = validate_units(units)
    assert msg is not None and "not a recognised" in msg


# The methods CF defines for cell_methods (CF 1.11, Appendix E).
CF_CELL_METHODS = {
    "point",
    "sum",
    "maximum",
    "maximum_absolute_value",
    "median",
    "mid_range",
    "minimum",
    "minimum_absolute_value",
    "mean",
    "mean_absolute_value",
    "mean_of_upper_decile",
    "mode",
    "range",
    "root_mean_square",
    "standard_deviation",
    "sum_of_squares",
    "variance",
}


def test_builtin_templates_use_cf_method_names_in_cell_methods() -> None:
    """`time: max` is not CF; a CF-aware reader does not recognise it (CLIM-1484)."""
    invalid = {}
    for template in registry_datasets._load_builtin_datasets():  # pyright: ignore[reportPrivateUsage]
        cell_methods = template.get("cell_methods")
        if not cell_methods:
            continue
        # Each "name: method" pair; parenthesised interval and comment text is not a method.
        methods = re.findall(r":\s*([a-z_]+)", re.sub(r"\([^)]*\)", "", cell_methods))
        if not methods or not set(methods) <= CF_CELL_METHODS:
            invalid[template["id"]] = cell_methods
    assert not invalid
