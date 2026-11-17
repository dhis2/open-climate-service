"""What the auto-registered xclim indicator processes promise, and whether they can keep it.

`_collect()` declares the same `returns` schema — a single datacube — for every indicator it
registers. Three of xclim 0.61's indicators return a tuple of cubes instead, so advertising them
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

_MULTI_OUTPUT_IN_XCLIM_061 = {"cffwis", "rain_season", "jetstream_metric_woollings"}


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

    by_id = {}
    for mod in (atmos, land, seaice):
        for obj in vars(mod).values():
            if hasattr(obj, "identifier") and hasattr(obj, "cf_attrs"):
                by_id.setdefault(obj.identifier, len(obj.cf_attrs))

    offenders = {name: by_id[name] for name in _registered() if by_id.get(name, 1) > 1}
    assert offenders == {}, f"registered indicators that return multiple cubes: {offenders}"


@pytest.mark.parametrize("indicator_id", sorted(_MULTI_OUTPUT_IN_XCLIM_061))
def test_known_multi_output_indicators_are_not_advertised(indicator_id: str) -> None:
    assert indicator_id not in _registered()


def test_single_output_indicators_are_still_registered() -> None:
    """The skip must not thin the catalogue beyond the three that cannot be honoured."""
    registered = _registered()

    assert len(registered) >= 170
    assert "tx_max" in registered
    assert registered["tx_max"]["returns"]["schema"]["subtype"] == "datacube"
