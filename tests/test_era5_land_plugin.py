import asyncio
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from open_climate_service.plugins.datasets.era5_land import (
    ERA5LandCDSHourlyPlugin,
    ERA5LandDailyTemperaturePlugin,
    ERA5LandEDHDailyPlugin,
    ERA5LandMonthlyPlugin,
    ERA5LandMonthlyPrecipitationPlugin,
    ERA5LandPrecipDailyPlugin,
    ERA5LandPrecipitationPlugin,
    _collapse_expver,
    _daily_availability_cutoff,
    _edh_open_zarr,
    _era5land_monthly_product_type,
    _hourly_availability_cutoff,
    _monthly_availability_cutoff,
)


def test_era5_land_hourly_plugin_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError, match="unsupported variable"):
        ERA5LandCDSHourlyPlugin(variable="unknown")


def test_era5_land_hourly_periods_enumerates_hours() -> None:
    from datetime import timezone

    plugin = ERA5LandCDSHourlyPlugin(variable="t2m")
    cutoff = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    with patch(
        "open_climate_service.plugins.datasets.era5_land._hourly_availability_cutoff",
        return_value=cutoff,
    ):
        periods = asyncio.run(plugin.periods("2026-01-01T00", "2026-01-01T05"))
    assert periods == ["2026-01-01T00", "2026-01-01T01", "2026-01-01T02"]


def test_era5_land_hourly_fetch_uses_cached_monthly_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandCDSHourlyPlugin(variable="t2m")
    fetch_calls: list[tuple[int, int]] = []

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        fetch_calls.append((year, month))
        return xr.Dataset(
            {"t2m": (("t", "y", "x"), np.array([[[20.0]], [[21.0]]], dtype=np.float32))},
            coords={
                "t": np.array(["2026-01-01T00", "2026-01-01T01"], dtype="datetime64[h]").astype("datetime64[ns]"),
                "y": [9.0],
                "x": [30.0],
            },
        )

    monkeypatch.setattr(ERA5LandCDSHourlyPlugin, "_fetch_month", fake_fetch_month)

    plugin.fetch_period("2026-01-01T00", [-1.0, 8.0, 31.0, 10.0])
    plugin.fetch_period("2026-01-01T01", [-1.0, 8.0, 31.0, 10.0])

    assert fetch_calls == [(2026, 1)]


def test_era5_land_hourly_fetch_converts_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandCDSHourlyPlugin(variable="t2m")

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        ds = xr.Dataset(
            {"t2m": (("t", "y", "x"), np.array([[[300.0]]], dtype=np.float32))},
            coords={
                "t": np.array(["2026-01-01T00"], dtype="datetime64[h]").astype("datetime64[ns]"),
                "y": [9.0],
                "x": [30.0],
            },
        )
        from open_climate_service.transforms.unit_conversion import kelvin_to_celsius

        return kelvin_to_celsius(ds, {"variable": "t2m"})

    monkeypatch.setattr(ERA5LandCDSHourlyPlugin, "_fetch_month", fake_fetch_month)
    ds = plugin.fetch_period("2026-01-01T00", [-1.0, 8.0, 31.0, 10.0])
    np.testing.assert_allclose(ds["t2m"].values, [[300.0 - 273.15]], atol=1e-3)


def test_era5_land_precipitation_plugin_uses_tp_and_converts_to_mm(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandPrecipitationPlugin()

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        return xr.Dataset(
            {"tp": (("t", "y", "x"), np.array([[[0.002]]], dtype=np.float32))},
            coords={
                "t": np.array(["2026-01-01T00"], dtype="datetime64[h]").astype("datetime64[ns]"),
                "y": [9.0],
                "x": [30.0],
            },
        )

    monkeypatch.setattr(ERA5LandCDSHourlyPlugin, "_fetch_month", fake_fetch_month)
    ds = plugin.fetch_period("2026-01-01T00", [-1.0, 8.0, 31.0, 10.0])
    assert list(ds.data_vars) == ["tp"]
    np.testing.assert_allclose(ds["tp"].values, [[2.0]], atol=1e-3)


def test_era5_land_hourly_availability_cutoff_uses_cds() -> None:
    from datetime import timezone

    fake_col = MagicMock()
    fake_col.end_datetime = datetime(2026, 5, 29, 23, tzinfo=timezone.utc)
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_col

    with patch("open_climate_service.plugins.datasets.era5_land._CdsClient", return_value=fake_client):
        cutoff = _hourly_availability_cutoff()

    assert cutoff == datetime(2026, 5, 29, 23, tzinfo=timezone.utc)


def test_era5_land_daily_periods_enumerates_days() -> None:
    plugin = ERA5LandDailyTemperaturePlugin()
    with patch(
        "open_climate_service.plugins.datasets.era5_land._daily_availability_cutoff",
        return_value=date(2024, 1, 3),
    ):
        periods = asyncio.run(plugin.periods("2024-01-01", "2024-01-10"))
    assert periods == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_era5_land_daily_fetch_uses_cached_monthly_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandDailyTemperaturePlugin()
    fetch_calls: list[tuple[int, int]] = []

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        fetch_calls.append((year, month))
        return xr.Dataset(
            {"t2m": (("t", "y", "x"), np.zeros((3, 1, 1), dtype=np.float32))},
            coords={
                "t": np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]").astype(
                    "datetime64[ns]"
                ),
                "y": [9.0],
                "x": [30.0],
            },
        )

    monkeypatch.setattr(ERA5LandDailyTemperaturePlugin, "_fetch_month", fake_fetch_month)

    plugin.fetch_period("2024-01-01", [-1.0, 8.0, 31.0, 10.0])
    plugin.fetch_period("2024-01-02", [-1.0, 8.0, 31.0, 10.0])
    plugin.fetch_period("2024-01-03", [-1.0, 8.0, 31.0, 10.0])

    # Three days in the same month → only one CDS API call
    assert fetch_calls == [(2024, 1)]


def test_era5_land_daily_availability_cutoff_uses_cds() -> None:
    fake_col = MagicMock()
    fake_col.end_datetime = datetime(2026, 5, 29, tzinfo=None)
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_col

    with patch("open_climate_service.plugins.datasets.era5_land._CdsClient", return_value=fake_client):
        cutoff = _daily_availability_cutoff()

    assert cutoff == date(2026, 5, 29)


def test_era5_land_monthly_plugin_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError, match="unsupported variable"):
        ERA5LandMonthlyPlugin(variable="unknown")


def test_era5_land_monthly_periods_enumerates_months() -> None:
    plugin = ERA5LandMonthlyPlugin(variable="t2m")
    with patch(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        return_value=date(2024, 3, 1),
    ):
        periods = asyncio.run(plugin.periods("2024-01", "2024-06"))
    assert periods == ["2024-01", "2024-02", "2024-03"]


def test_era5_land_monthly_periods_empty_when_start_after_cutoff() -> None:
    plugin = ERA5LandMonthlyPlugin(variable="t2m")
    with patch(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        return_value=date(2023, 12, 1),
    ):
        periods = asyncio.run(plugin.periods("2024-01", "2024-06"))
    assert periods == []


def _make_monthly_nc(variable: str, value: float, kelvin: bool = False) -> xr.Dataset:
    data = np.array([[[[value]]]], dtype=np.float32)
    return xr.Dataset(
        {variable: (("valid_time", "latitude", "longitude"), data[0])},
        coords={
            "valid_time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "latitude": [9.0],
            "longitude": [30.0],
        },
    )


def test_era5_land_monthly_fetch_renames_coords_and_converts_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = ERA5LandMonthlyPlugin(variable="t2m")
    raw_ds = _make_monthly_nc("t2m", 300.0)

    # fetch_period now does the CDS download inline, then renames + converts. Mock the
    # client + open_dataset so the real fetch_period runs against the raw netCDF.
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land._CdsClient", lambda: MagicMock())
    # The year batch clamps to what CDS publishes; pin it so the test does not reach the
    # catalogue (CLIM-956).
    monkeypatch.setattr(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        lambda: date(2026, 7, 1),
    )
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land.xr.open_dataset", lambda *a, **k: raw_ds)
    ds = plugin.fetch_period("2024-01", [-1.0, 8.0, 31.0, 10.0])

    assert "t" in ds.dims
    assert "x" in ds.dims
    assert "y" in ds.dims
    np.testing.assert_allclose(ds["t2m"].values, [[[300.0 - 273.15]]], atol=1e-3)


def test_era5_land_monthly_precipitation_converts_metres_to_mm_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = ERA5LandMonthlyPrecipitationPlugin()
    raw_ds = xr.Dataset(
        {"tp": (("t", "y", "x"), np.array([[[0.05]]], dtype=np.float32))},
        coords={"t": np.array(["2024-01-01"], dtype="datetime64[ns]"), "y": [9.0], "x": [30.0]},
    )

    def fake_parent_fetch(self: object, period_id: str, bbox: list[float]) -> xr.Dataset:
        return raw_ds

    monkeypatch.setattr(ERA5LandMonthlyPlugin, "fetch_period", fake_parent_fetch)
    ds = plugin.fetch_period("2024-01", [-1.0, 8.0, 31.0, 10.0])

    # CDS gives the mean daily total (m/day). We store the mean daily *rate* in mm/day,
    # NOT the calendar-month total: 0.05 m/day × 1000 mm/m = 50 mm/day. (A flux is the
    # dimensionally-correct input for xclim indices; the monthly total is rate × days.)
    np.testing.assert_allclose(ds["tp"].values, [[[50.0]]], atol=1e-1)


def test_monthly_availability_cutoff_uses_cds_end_datetime() -> None:
    from datetime import timezone

    fake_col = MagicMock()
    fake_col.end_datetime = datetime(2026, 4, 1, tzinfo=timezone.utc)
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_col

    with patch("open_climate_service.plugins.datasets.era5_land._CdsClient", return_value=fake_client):
        cutoff = _monthly_availability_cutoff()

    assert cutoff == date(2026, 4, 1)


def test_monthly_availability_cutoff_raises_when_cds_unavailable() -> None:
    with (
        patch("open_climate_service.plugins.datasets.era5_land._CdsClient", side_effect=Exception("network error")),
        pytest.raises(Exception, match="network error"),
    ):
        _monthly_availability_cutoff()


def test_era5_land_hourly_refetches_when_month_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandCDSHourlyPlugin(variable="t2m")
    fetch_calls: list[tuple[int, int]] = []

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        fetch_calls.append((year, month))
        ts = f"{year:04d}-{month:02d}-01T00"
        return xr.Dataset(
            {"t2m": (("t", "y", "x"), np.zeros((1, 1, 1), dtype=np.float32))},
            coords={
                "t": np.array([ts], dtype="datetime64[h]").astype("datetime64[ns]"),
                "y": [9.0],
                "x": [30.0],
            },
        )

    monkeypatch.setattr(ERA5LandCDSHourlyPlugin, "_fetch_month", fake_fetch_month)

    plugin.fetch_period("2026-01-01T00", [-1.0, 8.0, 31.0, 10.0])
    plugin.fetch_period("2026-02-01T00", [-1.0, 8.0, 31.0, 10.0])

    assert fetch_calls == [(2026, 1), (2026, 2)]


# ---------------------------------------------------------------------------
# EDH plugin tests
# ---------------------------------------------------------------------------


def _fake_edh_hourly_region() -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.array([[[280.0]], [[281.0]]], dtype=np.float32))},
        coords={
            "valid_time": np.array(["2026-01-01T00", "2026-01-01T01"], dtype="datetime64[h]").astype("datetime64[ns]"),
            "latitude": [9.0],
            "longitude": [30.0],
        },
    )


def _fake_edh_daily_region() -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.array([[[285.0]], [[286.0]]], dtype=np.float32))},
        coords={
            "valid_time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]").astype("datetime64[ns]"),
            "latitude": [9.0],
            "longitude": [30.0],
        },
    )


def test_edh_daily_plugin_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError, match="unsupported variable"):
        ERA5LandEDHDailyPlugin(variable="unknown")


def test_edh_daily_fetch_renames_coords_and_converts_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandEDHDailyPlugin(variable="t2m")
    monkeypatch.setattr(plugin, "_region_for_bbox", lambda bbox: _fake_edh_daily_region())
    ds = plugin.fetch_period("2026-01-01", [-1.0, 8.0, 31.0, 10.0])
    assert set(ds.dims) == {"t", "y", "x"}
    np.testing.assert_allclose(ds["t2m"].values, [[[285.0 - 273.15]]], atol=1e-3)


def test_edh_daily_periods_capped_to_latest_available(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ERA5LandEDHDailyPlugin(variable="t2m")
    monkeypatch.setattr(plugin, "_latest_available", lambda: "2026-01-03")
    periods = asyncio.run(plugin.periods("2026-01-01", "2026-01-10"))
    assert periods == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_edh_open_zarr_injects_api_key_in_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_open_zarr(url: str, **kwargs: object) -> xr.Dataset:
        captured.append(url)
        return MagicMock()

    monkeypatch.setenv("EDH_API_KEY", "mytoken")
    # `_edh_open_zarr` also probes the store's Zarr version on open. Stubbed, or this unit
    # test reaches the real EDH host: the probe swallows its own failures, so it would still
    # pass while depending on DNS and spending up to two 15-second timeouts.
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land._edh_is_zarr_v3", lambda _url: True)
    with patch("open_climate_service.plugins.datasets.era5_land.xr.open_zarr", fake_open_zarr):
        _edh_open_zarr("https://api.earthdatahub.destine.eu/era5/test.zarr")

    assert captured[0] == "https://edh:mytoken@api.earthdatahub.destine.eu/era5/test.zarr"


def test_era5land_monthly_product_type_uses_workaround_for_affected_period() -> None:
    # Workaround only applies to accumulated variables (tp), not instantaneous (t2m)
    # Affected period: September 2022 – February 2024 (CDS data bug, forum.ecmwf.int/t/2370)
    assert _era5land_monthly_product_type(2022, 9, "tp") == "monthly_averaged_reanalysis_by_hour_of_day"
    assert _era5land_monthly_product_type(2023, 6, "tp") == "monthly_averaged_reanalysis_by_hour_of_day"
    assert _era5land_monthly_product_type(2024, 2, "tp") == "monthly_averaged_reanalysis_by_hour_of_day"
    # t2m always uses standard product (switching type would change from monthly mean to 00:00 mean)
    assert _era5land_monthly_product_type(2022, 9, "t2m") == "monthly_averaged_reanalysis"
    assert _era5land_monthly_product_type(2023, 6, "t2m") == "monthly_averaged_reanalysis"
    # Outside affected period: standard product for all variables
    assert _era5land_monthly_product_type(2022, 8, "tp") == "monthly_averaged_reanalysis"
    assert _era5land_monthly_product_type(2024, 3, "tp") == "monthly_averaged_reanalysis"
    assert _era5land_monthly_product_type(2026, 1, "tp") == "monthly_averaged_reanalysis"


# --- auxiliary-variable cleanup ------------------------------------------------------------


def _era5_cube_with_extras() -> xr.Dataset:
    """An ERA5-shaped cube carrying the scalar coords the source attaches, plus a CRS."""
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

    ds = xr.Dataset(
        {"t2m": (("t", "y", "x"), np.ones((1, 2, 2), dtype="float32"))},
        coords={"t": [np.datetime64("2026-01-01")], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    return ds.assign_coords(number=0, expver="0001").rio.write_crs(4326)


def test_auxiliary_cleanup_drops_the_phantom_ensemble_coords() -> None:
    """`number`/`expver` otherwise surface as a stray `bands` dimension in an anomaly."""
    from open_climate_service.plugins.datasets.era5_land import _drop_auxiliary_variables

    out = _drop_auxiliary_variables(_era5_cube_with_extras(), "t2m")

    assert "number" not in out.variables
    assert "expver" not in out.variables
    assert set(out.data_vars) == {"t2m"}


def test_auxiliary_cleanup_keeps_the_crs() -> None:
    """The CRS lives in a non-dimension coordinate, so a blanket drop would take it.

    Grid inference falls back to EPSG:4326 for a cube that carries none, so ERA5-Land would
    still be recorded correctly — by luck. A projected source cleaned the same way would be
    silently mislabelled, which is what this pins.
    """
    from open_climate_service.plugins.datasets.era5_land import _drop_auxiliary_variables

    before = _era5_cube_with_extras()

    out = _drop_auxiliary_variables(before, "t2m")

    assert "spatial_ref" in out.variables
    # Compared against the input rather than via to_epsg(), which needs a working PROJ database
    # and is what makes the tests in test_openeo_execution.py fail on a broken local install.
    assert out.rio.crs is not None and out.rio.crs == before.rio.crs


def test_auxiliary_cleanup_honours_a_declared_grid_mapping() -> None:
    """A source naming its grid mapping something other than `spatial_ref` must survive too."""
    from open_climate_service.plugins.datasets.era5_land import _drop_auxiliary_variables

    ds = _era5_cube_with_extras().rename({"spatial_ref": "my_crs"})
    ds["t2m"].attrs["grid_mapping"] = "my_crs"

    out = _drop_auxiliary_variables(ds, "t2m")

    assert "my_crs" in out.variables


def _expver_nc(variable: str) -> xr.Dataset:
    """A CDS download straddling the ERA5/ERA5T boundary.

    Each ``expver`` slice is NaN where the other supplies data, which is how CDS returns a
    request spanning the boundary: 1 is final, 5 is preliminary ERA5T.
    """
    final = np.array([[[1.0]], [[np.nan]]], dtype=np.float32)
    preliminary = np.array([[[np.nan]], [[5.0]]], dtype=np.float32)
    return xr.Dataset(
        {variable: (("valid_time", "expver", "latitude", "longitude"), np.stack([final, preliminary], axis=1))},
        coords={
            "valid_time": np.array(["2026-05-01", "2026-06-01"], dtype="datetime64[ns]"),
            "expver": [1, 5],
            "latitude": [9.0],
            "longitude": [30.0],
        },
    )


def test_collapse_expver_prefers_final_and_fills_from_preliminary() -> None:
    """Final ERA5 wins where it has data; ERA5T fills the recent months it does not cover."""
    collapsed = _collapse_expver(_expver_nc("t2m"))

    assert "expver" not in collapsed.dims
    assert "expver" not in collapsed.coords
    np.testing.assert_allclose(collapsed["t2m"].values.ravel(), [1.0, 5.0])
    assert collapsed["t2m"].dtype == np.float32


def test_collapse_expver_passes_through_a_cube_without_expver() -> None:
    plain = _expver_nc("t2m").sel(expver=1, drop=True)

    assert _collapse_expver(plain).identical(plain)


def test_collapse_expver_drops_a_scalar_expver_coordinate() -> None:
    """A single-version download carries expver as a coordinate, not a dimension."""
    scalar = _expver_nc("t2m").sel(expver=5)
    assert "expver" in scalar.coords

    assert "expver" not in _collapse_expver(scalar).coords


def test_monthly_fetch_collapses_expver_so_the_cube_stays_three_dimensional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synced month straddling the ERA5T boundary must not double the array (CLIM-923).

    Left in place, the extra dimension reached the store and broke aggregation with a mask
    half the size of the data. ``_drop_auxiliary_variables`` does not catch it: that keeps
    every dimension by design.
    """
    plugin = ERA5LandMonthlyPlugin(variable="t2m")
    raw_ds = _expver_nc("t2m")
    raw_ds["t2m"] = raw_ds["t2m"] + 273.15  # kelvin, as CDS delivers it

    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land._CdsClient", lambda: MagicMock())
    # The year batch clamps to what CDS publishes; pin it so the test does not reach the
    # catalogue (CLIM-956).
    monkeypatch.setattr(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        lambda: date(2026, 7, 1),
    )
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land.xr.open_dataset", lambda *a, **k: raw_ds)
    ds = plugin.fetch_period("2026-06", [-1.0, 8.0, 31.0, 10.0])

    assert set(ds.dims) == {"t", "y", "x"}
    assert list(ds.data_vars) == ["t2m"]
    # One month, not the whole batch. The fixture holds May (final, 1.0) and June
    # (preliminary, 5.0); asking for June must yield June. The collapse itself is asserted
    # across both slices by test_collapse_expver_prefers_final_and_fills_from_preliminary.
    np.testing.assert_allclose(ds["t2m"].values.ravel(), [5.0], atol=1e-3)


def test_monthly_fetch_drops_auxiliary_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """``number`` is a scalar ensemble coordinate that otherwise persists as a phantom band."""
    plugin = ERA5LandMonthlyPlugin(variable="t2m")
    raw_ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.array([[[300.0]]], dtype=np.float32))},
        coords={
            "valid_time": np.array(["2026-06-01"], dtype="datetime64[ns]"),
            "latitude": [9.0],
            "longitude": [30.0],
            "number": 0,
        },
    )

    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land._CdsClient", lambda: MagicMock())
    # The year batch clamps to what CDS publishes; pin it so the test does not reach the
    # catalogue (CLIM-956).
    monkeypatch.setattr(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        lambda: date(2026, 7, 1),
    )
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land.xr.open_dataset", lambda *a, **k: raw_ds)
    ds = plugin.fetch_period("2026-06", [-1.0, 8.0, 31.0, 10.0])

    assert "number" not in ds.coords
    assert "number" not in ds.variables


def test_precip_daily_edh_branch_delegates_so_the_parent_cleaning_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override used to skip its parent's cleaning entirely (CLIM-923)."""
    plugin = ERA5LandPrecipDailyPlugin()
    dirty = xr.Dataset(
        {"tp": (("t", "y", "x"), np.array([[[2.0]]], dtype=np.float32))},
        coords={
            "t": np.array(["2026-06-01"], dtype="datetime64[ns]"),
            "y": [9.0],
            "x": [30.0],
            "number": 0,
        },
    )

    monkeypatch.setattr(type(plugin), "_latest_available", lambda self: "2026-12-31")
    monkeypatch.setattr(type(plugin), "_fetch_daily_sync", lambda self, period_id, bbox: dirty)
    ds = plugin.fetch_period("2026-06-01", [-1.0, 8.0, 31.0, 10.0])

    assert "number" not in ds.variables
    assert list(ds.data_vars) == ["tp"]


def test_hourly_fetch_keeps_its_time_coordinate_after_cleaning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hourly plugin selects a scalar timestamp, which demotes `t` to a coordinate.

    Cleaning after that selection would drop `t` along with the auxiliary coordinates, since
    it is no longer a dimension — losing the period's own timestamp. The cleaning therefore
    runs on the cached monthly cube instead.
    """
    plugin = ERA5LandCDSHourlyPlugin(variable="t2m")

    def fake_fetch_month(self: object, year: int, month: int, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        from open_climate_service.plugins.datasets.era5_land import _drop_auxiliary_variables

        ds = xr.Dataset(
            {"t2m": (("t", "y", "x"), np.array([[[20.0]]], dtype=np.float32))},
            coords={
                "t": np.array(["2026-01-01T00"], dtype="datetime64[h]").astype("datetime64[ns]"),
                "y": [9.0],
                "x": [30.0],
                "number": 0,
            },
        )
        return _drop_auxiliary_variables(ds, "t2m")

    monkeypatch.setattr(ERA5LandCDSHourlyPlugin, "_fetch_month", fake_fetch_month)
    ds = plugin.fetch_period("2026-01-01T00", [-1.0, 8.0, 31.0, 10.0])

    assert "t" in ds.coords
    assert "number" not in ds.variables


# --- year batching (CLIM-956) -----------------------------------------------------------


def _yearly_nc(variable: str, months: list[int], year: int = 2003) -> xr.Dataset:
    """A CDS response covering several months, valued so each month is identifiable."""
    times = np.array([f"{year:04d}-{m:02d}-01" for m in months], dtype="datetime64[ns]")
    values = np.array([[[273.15 + m]] for m in months], dtype=np.float32)
    return xr.Dataset(
        {variable: (("valid_time", "latitude", "longitude"), values)},
        coords={"valid_time": times, "latitude": [9.0], "longitude": [30.0]},
    )


def _capture_cds(monkeypatch: pytest.MonkeyPatch, year: int = 2003) -> list[dict]:
    """Record every CDS request and answer it with exactly the months it asked for."""
    submitted: list[dict] = []

    def fake_client() -> MagicMock:
        client = MagicMock()

        def submit(_collection: str, params: dict) -> MagicMock:
            submitted.append(params)
            return MagicMock()

        client.submit.side_effect = submit
        return client

    def fake_open(*_a: object, **_k: object) -> xr.Dataset:
        months = [int(m) for m in submitted[-1]["month"]]
        return _yearly_nc("t2m", months, year)

    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land._CdsClient", fake_client)
    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land.xr.open_dataset", fake_open)
    monkeypatch.setattr(
        "open_climate_service.plugins.datasets.era5_land._monthly_availability_cutoff",
        lambda: date(2026, 7, 1),
    )
    return submitted


def test_a_year_of_months_costs_one_cds_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of CLIM-956: request cost is queue and dispatch, not data.

    Measured live, a 12-month request took 34.4 s against 28.7 s for a single month — 2.9 s
    per month rather than 28.7. Twelve requests per year was paying that overhead twelve times.
    """
    submitted = _capture_cds(monkeypatch)
    plugin = ERA5LandMonthlyPlugin(variable="t2m")

    for month in range(1, 13):
        plugin.fetch_period(f"2003-{month:02d}", [-1.0, 8.0, 31.0, 10.0])

    assert len(submitted) == 1, f"{len(submitted)} CDS requests for one year"
    assert submitted[0]["month"] == [f"{m:02d}" for m in range(1, 13)]


def test_each_month_gets_its_own_values_from_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batching must not smear months together — the whole risk of serving from a cache."""
    _capture_cds(monkeypatch)
    plugin = ERA5LandMonthlyPlugin(variable="t2m")

    for month in (1, 6, 12):
        ds = plugin.fetch_period(f"2003-{month:02d}", [-1.0, 8.0, 31.0, 10.0])
        assert ds.sizes["t"] == 1
        assert np.datetime_as_string(ds["t"].values[0], unit="M") == f"2003-{month:02d}"
        np.testing.assert_allclose(ds["t2m"].values.ravel(), [float(month)], atol=1e-3)


def test_a_changed_extent_refetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache is keyed on bbox too, or a second country would silently reuse the first."""
    submitted = _capture_cds(monkeypatch)
    plugin = ERA5LandMonthlyPlugin(variable="t2m")

    plugin.fetch_period("2003-01", [-1.0, 8.0, 31.0, 10.0])
    plugin.fetch_period("2003-01", [80.0, 26.0, 88.0, 30.0])

    assert len(submitted) == 2


def test_a_partial_year_asks_only_for_published_months(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting an unpublished month is an error at CDS, not an empty result."""
    submitted = _capture_cds(monkeypatch, year=2026)
    plugin = ERA5LandMonthlyPlugin(variable="t2m")

    plugin.fetch_period("2026-03", [-1.0, 8.0, 31.0, 10.0])

    assert submitted[0]["month"] == [f"{m:02d}" for m in range(1, 8)], "asked past the cutoff"


def test_a_year_spanning_the_product_type_boundary_is_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accumulated variables switch product mid-year, so a whole-year request would be wrong.

    `_era5land_monthly_product_type` works around a CDS defect in `tp` for September 2022 to
    February 2024. Batching a whole year across that boundary would silently return the buggy
    product for part of it — the very thing the workaround exists to prevent.
    """
    submitted = _capture_cds(monkeypatch, year=2022)

    def fake_open(*_a: object, **_k: object) -> xr.Dataset:
        return _yearly_nc("tp", [int(m) for m in submitted[-1]["month"]], 2022)

    monkeypatch.setattr("open_climate_service.plugins.datasets.era5_land.xr.open_dataset", fake_open)
    plugin = ERA5LandMonthlyPlugin(variable="tp")

    plugin.fetch_period("2022-10", [-1.0, 8.0, 31.0, 10.0])

    assert len(submitted) == 2, "the year was not split at the product-type boundary"
    by_product = {params["product_type"][0]: [int(m) for m in params["month"]] for params in submitted}
    assert by_product["monthly_averaged_reanalysis"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert by_product["monthly_averaged_reanalysis_by_hour_of_day"] == [9, 10, 11, 12]
