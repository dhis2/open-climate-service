"""Tests for non-temporal (ordinal) stepping-dimension support.

Covers the STAC ``cube:dimensions`` declaration and the ingest orchestrator's CF-encoding
handling for an integer day-of-year axis (vs a datetime time axis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from open_climate_service.stac.services import _build_ordinal_dimensions
from open_climate_service.streaming.orchestrator import _strip_cf_encoding
from open_climate_service.streaming.store import open_or_create_repo, read_committed_period_ids


def _dayofyear_cube() -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("dayofyear", "y", "x"), np.zeros((366, 2, 2), dtype="float32"))},
        coords={"dayofyear": list(range(1, 367)), "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )


def _daily_cube() -> xr.Dataset:
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    return xr.Dataset(
        {"v": (("t", "y", "x"), np.zeros((3, 2, 2), dtype="float32"))},
        coords={"t": times, "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )


def test_build_ordinal_dimensions_declares_dayofyear() -> None:
    dims = _build_ordinal_dimensions(_dayofyear_cube(), "x", "y", None)

    assert "dayofyear" in dims
    doy = dims["dayofyear"]
    assert doy["type"] == "other"
    assert doy["values"][0] == 1 and doy["values"][-1] == 366
    assert doy["step"] == 1


def test_build_ordinal_dimensions_excludes_spatial_and_temporal() -> None:
    # Only x/y (spatial) and t (temporal) present — nothing ordinal to declare.
    assert _build_ordinal_dimensions(_daily_cube(), "x", "y", "t") == {}


def test_build_ordinal_dimensions_omits_step_for_irregular_axis() -> None:
    # Unevenly spaced integer axis: a step derived from the first delta alone
    # would be wrong, so no step should be published.
    ds = xr.Dataset(
        {"v": (("band", "y", "x"), np.zeros((4, 2, 2), dtype="float32"))},
        coords={"band": [1, 2, 5, 8], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )
    dims = _build_ordinal_dimensions(ds, "x", "y", None)

    assert dims["band"]["values"] == [1, 2, 5, 8]
    assert "step" not in dims["band"]


def test_build_ordinal_dimensions_publishes_string_labels() -> None:
    """A string-labelled axis must publish its labels, because clients cannot read them.

    Load-bearing rather than cosmetic. A string coordinate is written as the Zarr v3
    extension type ``fixed_length_utf32``, which is not in the core specification and which
    the JavaScript reader rejects outright. STAC is therefore the only place a browser
    client can learn what the labels are, and it resolves them to indices before selecting.
    Drop these values and a true-colour composite has nothing to map its bands onto.

    ``worldpop_agesex_*`` has carried a string ``sex`` axis for the same reason, so this
    covers existing datasets and not only the composite case.
    """
    ds = xr.Dataset(
        {"reflectance": (("band", "y", "x"), np.zeros((3, 2, 2), dtype="uint8"))},
        coords={"band": ["red", "green", "blue"], "y": [1.0, 2.0], "x": [1.0, 2.0]},
    )

    dims = _build_ordinal_dimensions(ds, "x", "y", None)

    assert dims["band"]["values"] == ["red", "green", "blue"]
    assert dims["band"]["type"] == "other"
    # No step: a step on a non-numeric axis would invite a client to synthesise labels.
    assert "step" not in dims["band"]


def test_read_committed_period_ids_ordinal_coord(tmp_path: Path) -> None:
    # Round-trip an integer day-of-year axis through the store and confirm
    # committed ids come back as plain strings ("1".."366"), not datetime-parsed
    # garbage — otherwise resume would re-append everything.
    store_path = tmp_path / "ordinal.icechunk"
    repo = open_or_create_repo(store_path)
    session = repo.writable_session("main")
    _dayofyear_cube().to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("ingest: ordinal")

    committed = read_committed_period_ids(store_path, "daily", time_dim="dayofyear")

    assert committed == {str(i) for i in range(1, 367)}


def test_strip_cf_encoding_skips_datetime_for_ordinal_coord() -> None:
    ds = _dayofyear_cube()
    _strip_cf_encoding(ds, "daily", time_dim="dayofyear")
    # An integer day-of-year axis must not get datetime CF encoding.
    assert "units" not in ds["dayofyear"].encoding


def test_strip_cf_encoding_applies_datetime_for_time_coord() -> None:
    ds = _daily_cube()
    _strip_cf_encoding(ds, "daily", time_dim="t")
    assert ds["t"].encoding.get("units") == "days since 1970-01-01"
