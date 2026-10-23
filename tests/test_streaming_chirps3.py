import asyncio
from datetime import date
from typing import Any

import pytest

from open_climate_service.plugins.rasters import chirps3
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


class _Today:
    """Pin "now" to 29 August 2026."""

    @staticmethod
    def now(_tz: Any = None) -> Any:
        class _Now:
            @staticmethod
            def date() -> date:
                return date(2026, 8, 29)

        return _Now()


def _publish(monkeypatch: pytest.MonkeyPatch, *days: date) -> list[str]:
    """The CDN publishing the given days (any stage); returns the URLs probed."""
    import httpx

    published = {f"{day.year}.{day.month:02d}.{day.day:02d}." for day in days}
    probed: list[str] = []

    class _Response:
        def __init__(self, url: str) -> None:
            self.status_code = 200 if any(stamp in url for stamp in published) else 404

    def head(url: str, **_: Any) -> _Response:
        probed.append(url)
        return _Response(url)

    monkeypatch.setattr(httpx, "head", head)
    monkeypatch.setattr(chirps3, "datetime", _Today)
    return probed


def test_prelim_reaches_the_last_published_day_of_the_current_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """August is only partly published on the 29th; the cutoff is its newest day, not July."""
    _publish(monkeypatch, date(2026, 8, 25), date(2026, 7, 31))

    plugin = CHIRPS3DailyPlugin(stage="prelim", flavor="sat")

    assert plugin._availability_cutoff() == date(2026, 8, 25)
    assert asyncio.run(plugin.periods("2026-08-24", "2026-08-31")) == ["2026-08-24", "2026-08-25"]


def test_final_still_refuses_the_current_month(monkeypatch: pytest.MonkeyPatch) -> None:
    probed = _publish(monkeypatch, date(2026, 8, 25), date(2026, 7, 31))

    assert CHIRPS3DailyPlugin()._availability_cutoff() == date(2026, 7, 31)
    assert all("2026.08." not in url for url in probed)


def test_prelim_raises_when_no_recent_day_is_published(monkeypatch: pytest.MonkeyPatch) -> None:
    _publish(monkeypatch)

    with pytest.raises(RuntimeError, match="No published CHIRPS3 preliminary day"):
        CHIRPS3DailyPlugin(stage="prelim", flavor="sat")._availability_cutoff()
