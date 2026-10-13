import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from open_climate_service.plugins.rasters.modis_vi import (
    MODISVegetationIndexMonthlyPlugin,
    _latest_items,
    _midpoint,
)


def _item(item_id: str, start: str = "2024-08-28T00:00:00Z", end: str = "2024-09-12T23:59:59Z") -> SimpleNamespace:
    return SimpleNamespace(id=item_id, properties={"start_datetime": start, "end_datetime": end})


def test_rejects_unknown_index_and_product() -> None:
    with pytest.raises(ValueError, match="unsupported index"):
        MODISVegetationIndexMonthlyPlugin(index="savi")
    with pytest.raises(ValueError, match="unsupported products"):
        MODISVegetationIndexMonthlyPlugin(index="ndvi", products=["MCD13Q1"])


def test_composite_belongs_to_the_month_of_its_midpoint() -> None:
    # 2024-08-28..2024-09-12 is centred on 2024-09-04, so it counts towards September.
    assert _midpoint(_item("MOD13Q1.A2024241.h19v07.061.2024258125635")) == date(2024, 9, 4)


def test_latest_items_keeps_the_newest_production_per_tile() -> None:
    old = _item("MOD13Q1.A2024129.h18v07.061.2024146000000")
    new = _item("MOD13Q1.A2024129.h18v07.061.2024200000000")
    other_tile = _item("MOD13Q1.A2024129.h19v07.061.2024146000000")
    aqua = _item("MYD13Q1.A2024129.h18v07.061.2024146000000")
    kept = {i.id for i in _latest_items([old, new, other_tile, aqua])}
    assert kept == {new.id, other_tile.id, aqua.id}


def test_periods_stop_at_the_last_complete_month() -> None:
    plugin = MODISVegetationIndexMonthlyPlugin(index="ndvi")
    periods = asyncio.run(plugin.periods("2024-01", "2099-12"))
    assert periods[0] == "2024-01"
    last = date.fromisoformat(periods[-1] + "-01")
    assert last < date.today().replace(day=1)
