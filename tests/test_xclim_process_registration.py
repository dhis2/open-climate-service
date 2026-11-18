"""What the auto-registered xclim indicator processes promise, and whether they can keep it.

`_collect()` declares the same `returns` schema — a single datacube — for every indicator it
registers. Five of xclim 1.0's indicators return several cubes instead, so advertising them
published a contract the process could not honour: a client composing a graph from
`GET /processes` would feed a tuple into save_result or a reducer and get a confusing failure
rather than a clear "not supported" (CLIM-860).

The general assertion matters more than the three names. A future xclim release can add
multi-output indicators, and this catches that at test time instead of in someone's process graph.
"""

from __future__ import annotations

import pytest

from open_geo_service.openeo import xclim_processes
from open_geo_service.process import get_process_metadata

pytest.importorskip("xclim")

# Output counts per xclim 1.0's `Indicator.outputs`.
_MULTI_OUTPUT_IN_XCLIM_1 = {
    "cffwis": 6,
    "rain_season": 3,
    "jetstream_metric_woollings": 2,
    "base_flow_index_seasonal_ratio": 2,
    "sen_slope": 2,
}


def _registered() -> dict[str, dict]:
    metas = {}
    for func in xclim_processes.scan():
        meta = get_process_metadata(func)
        assert meta is not None
        metas[meta["id"]] = meta
    return metas


def test_every_registered_indicator_returns_exactly_one_cube() -> None:
    """The declared `returns` must match what the indicator actually produces."""
    import xclim.indicators.atmos as atmos
    import xclim.indicators.land as land
    import xclim.indicators.seaIce as seaice
    from xclim.core.indicator import Indicator

    by_id = {}
    for mod in (atmos, land, seaice):
        for obj in vars(mod).values():
            if isinstance(obj, Indicator):
                by_id.setdefault(obj.identifier, len(obj.outputs))

    offenders = {name: by_id[name] for name in _registered() if by_id.get(name, 1) > 1}
    assert offenders == {}, f"registered indicators that return multiple cubes: {offenders}"


@pytest.mark.parametrize("indicator_id", sorted(_MULTI_OUTPUT_IN_XCLIM_1))
def test_known_multi_output_indicators_are_not_advertised(indicator_id: str) -> None:
    assert indicator_id not in _registered()


def test_multi_output_counts_are_read_from_xclim() -> None:
    """The skip depends on reading each indicator's output count; a renamed attribute must fail here."""
    import xclim.indicators.atmos as atmos
    import xclim.indicators.land as land
    import xclim.indicators.seaIce as seaice

    counts = {
        obj.identifier: len(obj.outputs)
        for mod in (atmos, land, seaice)
        for obj in vars(mod).values()
        if getattr(obj, "identifier", None) in _MULTI_OUTPUT_IN_XCLIM_1
    }
    assert counts == _MULTI_OUTPUT_IN_XCLIM_1


def test_registered_indicator_returns_a_data_array() -> None:
    """xclim 1.0 indicators return a Dataset by default; the process must hand back the cube."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    t = pd.date_range("2000-01-01", periods=731, freq="D")
    tas = xr.DataArray(
        290 + np.arange(731) % 7, coords={"t": t}, dims="t", attrs={"units": "K", "standard_name": "air_temperature"}
    )
    tx_max = next(f for f in xclim_processes.scan() if (get_process_metadata(f) or {}).get("id") == "tx_max")

    result = tx_max(tasmax=tas)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("t",)
    assert result.values.tolist() == [296.0, 296.0]


def test_single_output_indicators_are_still_registered() -> None:
    """The skip must not thin the catalogue beyond the three that cannot be honoured."""
    registered = _registered()

    assert len(registered) >= 170
    assert "tx_max" in registered
    assert registered["tx_max"]["returns"]["schema"]["subtype"] == "datacube"
