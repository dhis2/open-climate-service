"""Artifact statistics: what is measured from a stored raster (CLIM-1439)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from open_climate_service.data_manager.services import downloader
from open_climate_service.ingestions.schemas import ArtifactStatistics
from open_climate_service.shared.statistics import measure_statistics


def _store(tmp_path: Path, ds: xr.Dataset, *, t_dim: str | None) -> Path:
    path = tmp_path / "stats.icechunk"
    downloader.write_to_icechunk_store(ds, path, t_dim=t_dim, commit_message="test")
    return path


def _daily() -> xr.Dataset:
    """The extremes in different time steps, beside missing cells that must not count."""
    data = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, -4.0], [5.0, 0.5]], [[6.5, 0.0], [np.nan, 1.0]]])
    return xr.Dataset(
        {"precip": (("t", "y", "x"), data.astype("float32"))},
        coords={"t": pd.date_range("2026-01-01", periods=3, freq="D"), "y": [1.5, 0.5], "x": [10.5, 11.5]},
    )


def _pyramid() -> xr.Dataset:
    """Over the 1024-pixel threshold, so a pyramid; a one-pixel spike the averaged levels lose."""
    data = np.zeros((1100, 1100), dtype="float32")
    data[3, 7] = 100.0
    return xr.Dataset(
        {"elevation": (("y", "x"), data)},
        coords={"y": np.linspace(10.0, 0.0, 1100), "x": np.linspace(0.0, 10.0, 1100)},
    )


@pytest.mark.parametrize(
    ("ds", "variable", "t_dim", "expected"),
    [
        pytest.param(_daily(), "precip", "t", ArtifactStatistics(min=-4.0, max=6.5), id="across-periods"),
        pytest.param(_daily().isel(t=0, drop=True), "precip", None, ArtifactStatistics(min=1.0, max=3.0), id="static"),
        pytest.param(_pyramid(), "elevation", None, ArtifactStatistics(min=0.0, max=100.0), id="pyramid"),
        pytest.param(_daily() * np.nan, "precip", "t", None, id="all-missing"),
        pytest.param(_daily(), "temperature", "t", None, id="absent-variable"),
    ],
)
def test_statistics_are_measured_over_the_whole_stored_variable(
    tmp_path: Path, ds: xr.Dataset, variable: str, t_dim: str | None, expected: ArtifactStatistics | None
) -> None:
    assert measure_statistics(_store(tmp_path, ds, t_dim=t_dim), variable) == expected


def test_an_unreadable_store_does_not_fail_the_ingest(tmp_path: Path) -> None:
    assert measure_statistics(tmp_path / "not-a-store.icechunk", "precip") is None
