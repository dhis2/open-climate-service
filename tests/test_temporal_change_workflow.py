"""Tests for the built-in temporal_change workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from openeo_pg_parser_networkx.process_registry import Process

from open_geo_service.openeo import workflows as workflow_store
from open_geo_service.openeo.execution import (
    _augment_with_workflows,
    _new_process_registry,
)


def _overlay_with_mock_dataset() -> Any:
    # t=0 -> values 0..19, t=1 -> values 20..39, so last - first == 20 for every pixel.
    ds = xr.Dataset(
        {"pop": (("t", "y", "x"), np.arange(2 * 4 * 5, dtype="float32").reshape(2, 4, 5))},
        coords={
            "t": np.array(["2015-01-01", "2030-01-01"], dtype="datetime64[ns]"),
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    # A registry of its own: the cached one is shared by every later test in this process.
    reg = _new_process_registry()
    reg["load_collection"] = Process(spec={}, implementation=lambda id=None, temporal_extent=None, **k: ds["pop"])
    return _augment_with_workflows(reg)


def test_temporal_change_registered_as_process() -> None:
    ids = {wf.id for wf in workflow_store.list_workflows().processes}
    assert "temporal_change" in ids


def test_temporal_change_computes_last_minus_first() -> None:
    overlay = _overlay_with_mock_dataset()
    envelope = overlay["temporal_change"].implementation(
        dataset_id="worldpop_population_yearly",
        output_dataset_id="worldpop_population_change",
        variable="pop_change",
        temporal_extent=["2015-01-01", "2030-12-31"],
    )

    # save_result envelope carries the managed-publish options through unchanged.
    assert envelope.format.upper() == "ZARR"
    assert envelope.options["dataset_id"] == "worldpop_population_change"
    assert envelope.options["variable"] == "pop_change"
    assert envelope.options["publish"] is True

    # The time dimension is reduced away, leaving a 2-D change raster ...
    da = envelope.data
    assert "t" not in da.dims
    # ... whose value is last (2030) - first (2015) == 20 everywhere.
    assert np.allclose(da.values, 20.0)
