"""Tests for earthkit-meteo auto-registration and its unit enforcement."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from open_geo_service.openeo import earthkit_processes
from open_geo_service.openeo.plugin_processes import load_plugin_processes
from open_geo_service.process import get_process_metadata


def _cube(value: float, units: str | None, name: str = "var") -> xr.DataArray:
    attrs = {} if units is None else {"units": units}
    return xr.DataArray(
        np.full((2, 3), value, dtype="float64"),
        dims=("y", "x"),
        coords={"y": [1.0, 2.0], "x": [10.0, 11.0, 12.0]},
        name=name,
        attrs=attrs,
    )


def _process(process_id: str) -> object:
    for func in earthkit_processes.scan():
        meta = get_process_metadata(func)
        if meta and meta["id"] == process_id:
            return func
    raise AssertionError(f"{process_id} is not registered")


# --- registration -------------------------------------------------------------------


def test_scan_registers_thermo_functions() -> None:
    ids = {get_process_metadata(f)["id"] for f in earthkit_processes.scan()}  # type: ignore[index]
    assert "relative_humidity_from_dewpoint" in ids
    assert "dewpoint_from_relative_humidity" in ids
    assert "saturation_vapour_pressure" in ids


def test_scan_excludes_typing_and_dispatch_reexports() -> None:
    """The thermo namespace re-exports typing helpers and earthkit-utils' `dispatch`."""
    ids = {get_process_metadata(f)["id"] for f in earthkit_processes.scan()}  # type: ignore[index]
    assert not ids & {"Any", "ArrayLike", "TypeAlias", "dispatch", "overload"}


def test_scan_excludes_multi_output_functions() -> None:
    """`lcl` returns a (temperature, pressure) tuple, which the return schema cannot describe."""
    ids = {get_process_metadata(f)["id"] for f in earthkit_processes.scan()}  # type: ignore[index]
    assert "lcl" not in ids
    assert "lcl_temperature" in ids  # the single-output sibling is kept


def test_scan_is_cached() -> None:
    assert earthkit_processes.scan() is earthkit_processes.scan()


def test_metadata_describes_parameters_and_return() -> None:
    meta = get_process_metadata(_process("relative_humidity_from_dewpoint"))
    assert meta is not None
    assert meta["summary"].startswith("Compute the relative humidity")
    assert meta["returns"] == {"schema": {"type": "object", "subtype": "datacube"}}
    params = {p["name"]: p for p in meta["parameters"]}
    assert set(params) == {"t", "td"}
    for param in params.values():
        assert param["schema"] == {"type": "object", "subtype": "datacube"}
        assert "Expected unit: K" in param["description"]


def test_metadata_types_non_physical_parameters_from_defaults() -> None:
    """`phase` is prose-documented, not a unit — it must not be typed as a datacube."""
    meta = get_process_metadata(_process("saturation_vapour_pressure"))
    assert meta is not None
    params = {p["name"]: p for p in meta["parameters"]}
    assert params["t"]["schema"] == {"type": "object", "subtype": "datacube"}
    assert params["phase"]["schema"] == {"type": "string"}
    assert params["phase"]["default"] == "mixed"
    assert params["phase"]["optional"] is True


def test_metadata_drops_sphinx_implementations_section() -> None:
    meta = get_process_metadata(_process("relative_humidity_from_dewpoint"))
    assert meta is not None
    assert "Implementations" not in meta["description"]
    assert "Parameters" in meta["description"]


def test_registered_in_the_process_catalogue() -> None:
    ids = {process_id for process_id, _ in load_plugin_processes()}
    assert "relative_humidity_from_dewpoint" in ids


def test_resolvable_and_callable_through_the_openeo_execution_registry() -> None:
    """The catalogue and the execution registry are built separately; check the latter too.

    A process graph reaches the implementation via ``registry[process_id]``, so this is the
    path that decides whether the process is actually usable rather than merely advertised.
    """
    from open_geo_service.openeo import execution

    execution._registry = None  # the registry is a module-level singleton
    try:
        registry = execution._build_process_registry()
        entry = registry["relative_humidity_from_dewpoint"]
        assert [p["name"] for p in entry.spec["parameters"]] == ["t", "td"]
        result = entry.implementation(t=_cube(20.0, "degC"), td=_cube(10.0, "degC"))
        assert float(result[0, 0]) == pytest.approx(52.5198, abs=1e-3)
        assert result.attrs["units"] == "%"
    finally:
        execution._registry = None


def test_ids_do_not_collide_with_xclim_or_standard_processes() -> None:
    from open_geo_service.openeo import xclim_processes

    earthkit_ids = {get_process_metadata(f)["id"] for f in earthkit_processes.scan()}  # type: ignore[index]
    xclim_ids = {get_process_metadata(f)["id"] for f in xclim_processes.scan()}  # type: ignore[index]
    assert not earthkit_ids & xclim_ids


# --- unit enforcement ---------------------------------------------------------------


def test_kelvin_input_passes_through() -> None:
    func = _process("relative_humidity_from_dewpoint")
    result = func(t=_cube(293.15, "K"), td=_cube(283.15, "K"))  # type: ignore[operator]
    assert float(result[0, 0]) == pytest.approx(52.5198, abs=1e-3)


def test_celsius_input_is_converted_not_refused() -> None:
    """Our stores are degC (ERA5-Land applies kelvin_to_celsius at ingest); earthkit wants K."""
    func = _process("relative_humidity_from_dewpoint")
    kelvin = func(t=_cube(293.15, "K"), td=_cube(283.15, "K"))  # type: ignore[operator]
    celsius = func(t=_cube(20.0, "degC"), td=_cube(10.0, "degC"))  # type: ignore[operator]
    xr.testing.assert_allclose(kelvin, celsius)


@pytest.mark.parametrize("units", ["degC", "°C", "celsius", "degree_Celsius", "C"])
def test_celsius_spellings_are_all_recognised(units: str) -> None:
    func = _process("relative_humidity_from_dewpoint")
    result = func(t=_cube(20.0, units), td=_cube(10.0, units))  # type: ignore[operator]
    assert float(result[0, 0]) == pytest.approx(52.5198, abs=1e-3)


def test_celsius_would_be_silently_wrong_without_conversion() -> None:
    """Guard the reason this adapter exists: raw degC yields a plausible, wrong answer."""
    from earthkit.meteo import thermo

    unconverted = thermo.relative_humidity_from_dewpoint(_cube(20.0, "degC"), _cube(10.0, "degC"))
    correct = _process("relative_humidity_from_dewpoint")(t=_cube(20.0, "degC"), td=_cube(10.0, "degC"))  # type: ignore[operator]
    # No exception, no NaN — just a different number. Hence the enforcement.
    assert np.isfinite(float(unconverted[0, 0]))
    assert float(unconverted[0, 0]) != pytest.approx(float(correct[0, 0]), abs=1.0)


def test_hectopascal_pressure_is_converted() -> None:
    func = _process("specific_humidity_from_dewpoint")
    pascal = func(td=_cube(283.15, "K"), p=_cube(101325.0, "Pa"))  # type: ignore[operator]
    hpa = func(td=_cube(283.15, "K"), p=_cube(1013.25, "hPa"))  # type: ignore[operator]
    xr.testing.assert_allclose(pascal, hpa)


def test_missing_units_is_refused_with_actionable_message() -> None:
    func = _process("relative_humidity_from_dewpoint")
    with pytest.raises(ValueError, match="needs CF units"):
        func(t=_cube(293.15, None), td=_cube(283.15, "K"))  # type: ignore[operator]


def test_incompatible_units_are_refused() -> None:
    func = _process("relative_humidity_from_dewpoint")
    with pytest.raises(ValueError, match="cannot be converted"):
        func(t=_cube(293.15, "mm/d"), td=_cube(283.15, "K"))  # type: ignore[operator]


def test_converted_input_reports_the_expected_unit() -> None:
    converted = earthkit_processes._coerce_units("p", "t", _cube(20.0, "degC"), "K")
    assert converted.attrs["units"] == "K"
    assert float(converted[0, 0]) == pytest.approx(293.15)


def test_coercion_preserves_other_attributes() -> None:
    cube = _cube(20.0, "degC")
    cube.attrs["standard_name"] = "air_temperature"
    converted = earthkit_processes._coerce_units("p", "t", cube, "K")
    assert converted.attrs["standard_name"] == "air_temperature"


def test_non_xarray_values_pass_through() -> None:
    assert earthkit_processes._coerce_units("p", "eps", 0.5, "K") == 0.5


# --- output shape -------------------------------------------------------------------


def test_output_preserves_grid_and_cf_attributes() -> None:
    func = _process("relative_humidity_from_dewpoint")
    result = func(t=_cube(293.15, "K"), td=_cube(283.15, "K"))  # type: ignore[operator]
    assert result.dims == ("y", "x")
    assert list(result.coords) == ["y", "x"]
    assert result.attrs["units"] == "%"
    assert result.attrs["standard_name"] == "relative_humidity"


def test_output_is_named_for_the_process_not_the_input() -> None:
    func = _process("relative_humidity_from_dewpoint")
    result = func(t=_cube(293.15, "K", name="t2m"), td=_cube(283.15, "K", name="d2m"))  # type: ignore[operator]
    assert result.name == "relative_humidity_from_dewpoint"


def test_dask_backed_input_stays_lazy() -> None:
    func = _process("relative_humidity_from_dewpoint")
    t = _cube(293.15, "K").chunk({"x": 2})
    td = _cube(283.15, "K").chunk({"x": 2})
    result = func(t=t, td=td)  # type: ignore[operator]
    assert result.chunks is not None


# --- docstring parsing --------------------------------------------------------------


def test_documented_units_reads_physical_units_only() -> None:
    from earthkit.meteo import thermo

    units = earthkit_processes._documented_units(thermo.saturation_vapour_pressure)
    assert units == {"t": "K"}  # `phase` is prose, not a unit


def test_every_registered_cube_parameter_has_a_known_unit() -> None:
    """A parameter typed as a datacube must have a unit we can enforce."""
    for func in earthkit_processes.scan():
        meta = get_process_metadata(func)
        assert meta is not None
        for param in meta["parameters"]:
            if param["schema"].get("subtype") == "datacube":
                assert "Expected unit:" in param["description"], (meta["id"], param["name"])


# --- unit enforcement must not depend on upstream docstrings being well-formed -----------


def test_virtual_temperature_enforces_kelvin_despite_the_upstream_typo() -> None:
    """earthkit-meteo 1.0.0 documents `virtual_temperature.t` as "Temperature (K)s".

    The stray trailing "s" defeats the end-of-line unit match, so the docstring scan alone
    yields no unit for `t`. Without an override the parameter would be advertised as a plain
    number and handed to earthkit unconverted — a degC cube read as Kelvin, wrong by 273.15
    and labelled Kelvin. See _UNIT_OVERRIDES.
    """
    from earthkit.meteo import thermo

    # The upstream defect this guards against, asserted so the override is removed only when
    # upstream actually fixes it.
    assert "t" not in earthkit_processes._documented_units(thermo.virtual_temperature)

    meta = get_process_metadata(_process("virtual_temperature"))
    assert meta is not None
    params = {p["name"]: p for p in meta["parameters"]}
    assert params["t"]["schema"] == {"type": "object", "subtype": "datacube"}
    assert "Expected unit: K" in params["t"]["description"]

    func = _process("virtual_temperature")
    celsius = func(t=_cube(20.0, "degC"), q=_cube(0.01, "kg/kg"))  # type: ignore[operator]
    kelvin = func(t=_cube(293.15, "K"), q=_cube(0.01, "kg/kg"))  # type: ignore[operator]
    assert float(celsius[0, 0]) == pytest.approx(float(kelvin[0, 0]), abs=1e-9)
    # And it is the Kelvin-scale answer, not the ~20 the bug produced.
    assert float(celsius[0, 0]) > 290.0


def test_every_cube_typed_parameter_is_advertised_as_a_datacube() -> None:
    """The converse of test_every_registered_cube_parameter_has_a_known_unit.

    That test checks datacube => has a unit. This checks array-like => datacube, which is the
    direction that catches a *missed* unit: an unparsed physical input silently falls into the
    scalar branch and is advertised as a number, so no conversion is applied.
    """
    from earthkit.meteo import thermo

    for func in earthkit_processes.scan():
        meta = get_process_metadata(func)
        assert meta is not None
        cube_params = earthkit_processes._cube_parameters(getattr(thermo, meta["id"]))
        advertised = {p["name"]: p for p in meta["parameters"]}
        for name in cube_params:
            assert advertised[name]["schema"] == {"type": "object", "subtype": "datacube"}, (
                meta["id"],
                name,
            )


def test_a_cube_parameter_with_no_enforceable_unit_is_refused_registration() -> None:
    """Fail closed rather than advertise a cube whose units cannot be checked."""

    def bogus(x: object) -> object:
        """Compute something.

        Parameters
        ----------
        x: array-like | xarray.DataArray
            Some quantity (furlongs)

        Returns
        -------
        array-like
        """
        return x

    assert earthkit_processes._cube_parameters(bogus) == {"x"}
    assert earthkit_processes._documented_units(bogus) == {}


def test_the_slope_precompute_parameter_keeps_its_unit_enforced() -> None:
    """`es_slope` is documented in Pa/K — a real unit, so the function stays registered."""
    meta = get_process_metadata(_process("saturation_mixing_ratio_slope"))
    assert meta is not None
    params = {p["name"]: p for p in meta["parameters"]}
    assert params["es_slope"]["schema"] == {"type": "object", "subtype": "datacube"}
    assert "Expected unit: Pa/K" in params["es_slope"]["description"]


def test_an_unnamed_input_still_yields_a_named_result() -> None:
    """earthkit inherits the input's name, so an unnamed input left the result anonymous."""
    func = _process("relative_humidity_from_dewpoint")
    t = _cube(20.0, "degC").rename(None)
    td = _cube(10.0, "degC").rename(None)
    assert t.name is None

    result = func(t=t, td=td)  # type: ignore[operator]
    assert result.name == "relative_humidity_from_dewpoint"
