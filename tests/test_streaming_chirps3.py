import asyncio
from datetime import date

import pytest

from open_climate_service.plugins.rasters.chirps3 import CHIRPS3DailyPlugin


def test_chirps3_plugin_periods_are_clamped_to_complete_month(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = CHIRPS3DailyPlugin()
    monkeypatch.setattr(plugin, "_availability_cutoff", lambda: date(2026, 1, 31))

    periods = asyncio.run(plugin.periods("2026-01-30", "2026-02-03"))

    assert periods == ["2026-01-30", "2026-01-31"]


def test_chirps3_plugin_accepts_extra_kwargs() -> None:
    plugin = CHIRPS3DailyPlugin(stage="final", flavor="rnl", unknown_future_field="ignored")
    assert plugin.stage == "final"
    assert plugin.flavor == "rnl"


def test_chirps3_plugin_still_validates_stage_with_extra_kwargs() -> None:
    with pytest.raises(ValueError, match="stage must be"):
        CHIRPS3DailyPlugin(stage="bad", unknown_future_field="ignored")
