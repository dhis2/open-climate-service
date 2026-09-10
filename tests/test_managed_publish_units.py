"""Units carried through managed openEO publication.

A published store's units come from a dataset template, and for a managed publish that
template may be auto-registered from the *source* dataset. That inheritance is a guess which
only holds while the process preserves units — and a relative anomaly does not: it returns
percent of normal. These tests pin the precedence so percentages cannot be published as the
observed variable's unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr

from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.openeo import jobs
from open_climate_service.plugins.processes.compute_anomaly import compute_anomaly

_OBSERVED_TEMPLATE = {
    "id": "chirps3_precipitation_daily",
    "name": "Precipitation (CHIRPS3, daily)",
    "short_name": "Precipitation",
    "variable": "tp",
    "units": "mm/d",
    "period_type": "daily",
    "sync": {"kind": "temporal"},
}


def _cube(units: str, value: float = 20.0) -> xr.Dataset:
    """A minimal published-shaped cube carrying `units` on its single variable."""
    times = pd.date_range("2024-01-01", periods=3, freq="D")
    data = np.full((len(times), 2, 2), value, dtype="float32")
    ds = xr.Dataset(
        {"tp": (("t", "y", "x"), data, {"units": units})},
        coords={"t": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    return ds.rio.write_crs("EPSG:4326")


# --- template derivation: which units win ------------------------------------------------


def test_process_units_beat_units_inherited_from_the_source_dataset() -> None:
    """A relative anomaly is a percentage, whatever the observed dataset measured."""
    template = jobs._derive_managed_dataset_template(
        _cube("%"),
        {"dataset_id": "chirps3_precipitation_daily_anomaly", "variable": "tp"},
        _OBSERVED_TEMPLATE,
        "t",
    )
    assert template["units"] == "%"


def test_absolute_anomaly_still_inherits_the_observed_unit() -> None:
    """The inheritance is still right when the process preserved units."""
    template = jobs._derive_managed_dataset_template(
        _cube("mm/d"),
        {"dataset_id": "chirps3_precipitation_daily_anomaly", "variable": "tp"},
        _OBSERVED_TEMPLATE,
        "t",
    )
    assert template["units"] == "mm/d"


@pytest.mark.parametrize("placeholder", ["unknown", "-", "none", "n/a", "na"])
def test_uninformative_units_on_the_cube_still_inherit(placeholder: str) -> None:
    """A cube that asserts nothing about its units is what inheritance exists for."""
    template = jobs._derive_managed_dataset_template(
        _cube(placeholder),
        {"dataset_id": "chirps3_precipitation_daily_anomaly", "variable": "tp"},
        _OBSERVED_TEMPLATE,
        "t",
    )
    assert template["units"] == "mm/d"


@pytest.mark.parametrize("dimensionless", ["", "1", "unitless", "   "])
def test_declared_dimensionless_units_are_not_inherited_over(dimensionless: str) -> None:
    """Dimensionless is a claim about the data, not a gap to be filled from the source.

    Inheriting here registered a ratio or an index — SPI carries `units: ""` — as the source's
    `mm/d`, which is the silent relabelling the unit checks exist to prevent.
    """
    template = jobs._derive_managed_dataset_template(
        _cube(dimensionless),
        {"dataset_id": "chirps3_precipitation_ratio", "variable": "tp"},
        _OBSERVED_TEMPLATE,
        "t",
    )
    assert template["units"] != "mm/d"


def test_explicit_units_option_still_wins() -> None:
    template = jobs._derive_managed_dataset_template(
        _cube("%"),
        {"dataset_id": "x", "variable": "tp", "units": "mm/month"},
        _OBSERVED_TEMPLATE,
        "t",
    )
    assert template["units"] == "mm/month"


# --- the pre-registered template case ----------------------------------------------------


def test_template_units_of_a_different_dimension_are_refused() -> None:
    """The shipped anomaly templates declare mm/d; a relative anomaly must not be relabelled.

    20 percent-of-normal published as 20 mm/d is plausible on its face and inside the
    template's diverging display range, so nothing later could catch it.
    """
    with pytest.raises(ValueError, match="measures a different quantity"):
        jobs._reject_incompatible_template_units(_cube("%"), "tp", {"units": "mm/d"})


def test_a_different_spelling_of_the_same_unit_is_allowed() -> None:
    """`mm/d` and `mm/day` are one unit, so a tidier spelling must not be an error."""
    jobs._reject_incompatible_template_units(_cube("mm/d"), "tp", {"units": "mm/day"})
    jobs._reject_incompatible_template_units(_cube("degC"), "tp", {"units": "celsius"})


@pytest.mark.parametrize(
    ("produced", "declared"),
    [("K", "degC"), ("m", "mm"), ("mm/d", "m/d")],
)
def test_the_same_quantity_on_a_different_scale_is_refused(produced: str, declared: str) -> None:
    """Matching dimensionality is not enough to make a relabel safe.

    `K` and `degC` differ by 273.15 and `m` and `mm` by a factor of 1000, so stamping one over
    the other leaves the numbers meaning something else entirely — the same failure as the
    percent case below, and just as invisible afterwards.
    """
    with pytest.raises(ValueError, match="same quantity on different scales"):
        jobs._reject_incompatible_template_units(_cube(produced), "tp", {"units": declared})


def test_a_blank_template_unit_asserts_nothing_and_is_not_checked() -> None:
    """`""` is listed in _PLACEHOLDER_UNITS as carrying no information, on either side.

    No shipped template declares an empty unit, so reading it as a positive claim of
    dimensionlessness would invent intent rather than enforce it.
    """
    jobs._reject_incompatible_template_units(_cube("mm/d"), "tp", {"units": ""})
    jobs._reject_incompatible_template_units(_cube("mm/d"), "tp", {"units": "   "})


def test_an_absent_template_declaration_is_not_an_error() -> None:
    jobs._reject_incompatible_template_units(_cube("mm/d"), "tp", {})


def test_a_dimensionless_result_is_refused_by_a_dimensional_template() -> None:
    """The produced side declaring `""` is a claim, so relabelling it `mm/d` is a mismatch."""
    with pytest.raises(ValueError, match="different quantity"):
        jobs._reject_incompatible_template_units(_cube(""), "tp", {"units": "mm/d"})


# --- end to end --------------------------------------------------------------------------


@pytest.fixture
def managed_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the template registry and the store directory at a temporary instance."""
    configs = tmp_path / "datasets"
    configs.mkdir()
    # The source these tests declare as `source_dataset_id`. Registered because lineage that
    # does not resolve is now an error rather than silently "not derived from anything"
    # (CLIM-946) — a typo there would otherwise disable licence propagation. Writing it also
    # makes the fixture match production, where a source you derive from necessarily exists.
    (configs / "source.yaml").write_text(
        "- id: chirps3_precipitation_daily\n"
        "  name: Total precipitation (CHIRPS3)\n"
        "  variable: precip\n"
        "  period_type: daily\n"
        "  units: mm/d\n"
        "  license: CC-BY-4.0\n"
        "  sync:\n"
        "    kind: temporal\n"
        "  ingestion:\n"
        "    plugin: open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", configs)
    registry.reset_template_caches()
    monkeypatch.setattr("open_climate_service.data_manager.services.downloader.DOWNLOAD_DIR", tmp_path / "downloads")
    (tmp_path / "downloads").mkdir()
    return configs


def _published_units(dataset_id: str) -> Any:
    template = registry.get_dataset(dataset_id)
    assert template is not None
    return template.get("units")


def test_relative_anomaly_publishes_as_percent_end_to_end(managed_instance: Path) -> None:
    """The case abyot's review asked for: publish a relative anomaly, check what lands.

    Runs the real process and the real managed-publish path, then reads the units back off
    both the registered template and the written store.
    """
    times = pd.date_range("2024-01-01", periods=5, freq="D")
    observed = xr.DataArray(
        np.full((len(times), 2, 2), 12.0, dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="tp",
        attrs={"units": "mm/d", "standard_name": "precipitation_flux"},
    )
    normal = xr.DataArray(
        np.full((366, 2, 2), 10.0, dtype="float32"),
        dims=("dayofyear", "y", "x"),
        coords={"dayofyear": np.arange(1, 367), "y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="tp",
        attrs={"units": "mm/d"},
    )

    anomaly = compute_anomaly(observed, normal, method="relative")
    assert anomaly.attrs["units"] == "%"
    assert float(anomaly.isel(t=0, y=0, x=0)) == pytest.approx(20.0)  # 12 vs 10 → +20%

    ds = anomaly.to_dataset(name="tp").rio.write_crs("EPSG:4326")
    jobs._write_managed_zarr(
        ds,
        {
            "dataset_id": "chirps3_precipitation_daily_relative_anomaly",
            "variable": "tp",
            "source_dataset_id": "chirps3_precipitation_daily",
        },
    )

    # The auto-registered template must not have inherited mm/d from the observed dataset.
    assert _published_units("chirps3_precipitation_daily_relative_anomaly") == "%"
    # ...and the value written must still be the percentage, not reinterpreted.
    assert ds["tp"].attrs["units"] == "%"
    assert float(ds["tp"].isel(t=0, y=0, x=0)) == pytest.approx(20.0)


def test_publish_against_a_preregistered_absolute_template_is_refused(managed_instance: Path) -> None:
    """The guard must be reachable from the publish path, not merely exist.

    This is the shipped-template case: `chirps3_precipitation_daily_anomaly_1991_2020`
    declares `units: mm/d`, so publishing a relative anomaly into it would stamp mm/d over
    the percentages. Pre-register such a template and check the publish stops.
    """
    registry.write_dataset_template(
        {
            "id": "preregistered_anomaly",
            "name": "Precipitation anomaly",
            "short_name": "Anomaly",
            "variable": "tp",
            "units": "mm/d",
            "period_type": "daily",
            "sync": {"kind": "static"},
            "display": {"colormap": "rdbu_r", "range": [-20.0, 20.0]},
        }
    )

    with pytest.raises(ValueError, match="measures a different quantity"):
        jobs._write_managed_zarr(
            _cube("%"),
            {
                "dataset_id": "preregistered_anomaly",
                "variable": "tp",
                "source_dataset_id": "chirps3_precipitation_daily",
            },
        )


def test_absolute_anomaly_publishes_with_the_observed_unit_end_to_end(managed_instance: Path) -> None:
    """The counterpart, so the fix cannot pass by dropping inheritance altogether."""
    times = pd.date_range("2024-01-01", periods=5, freq="D")
    observed = xr.DataArray(
        np.full((len(times), 2, 2), 12.0, dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="tp",
        attrs={"units": "mm/d", "standard_name": "precipitation_flux"},
    )
    normal = xr.DataArray(
        np.full((366, 2, 2), 10.0, dtype="float32"),
        dims=("dayofyear", "y", "x"),
        coords={"dayofyear": np.arange(1, 367), "y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="tp",
        attrs={"units": "mm/d"},
    )

    ds = compute_anomaly(observed, normal, method="absolute").to_dataset(name="tp").rio.write_crs("EPSG:4326")
    jobs._write_managed_zarr(
        ds,
        {
            "dataset_id": "chirps3_precipitation_daily_anomaly",
            "variable": "tp",
            "source_dataset_id": "chirps3_precipitation_daily",
        },
    )

    assert _published_units("chirps3_precipitation_daily_anomaly") == "mm/d"
    assert float(ds["tp"].isel(t=0, y=0, x=0)) == pytest.approx(2.0)
