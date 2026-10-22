"""Loading the process implementations and a request's first xclim import never run at once (CLIM-1549).

Startup loads the process registry in a worker thread. A request that imports xclim while that
is in progress could deadlock with it on Python's module import locks, failing both. Here the
warm-up's load is held in progress, and the request's units check has to wait for it.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from open_climate_service.openeo import execution, plugin_processes, xclim_processes
from open_climate_service.shared import cf


def _hold_the_plugin_load(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Make `load_plugin_processes` stop in its first scan until released."""
    entered, release = threading.Event(), threading.Event()

    def held_scan() -> list[Any]:
        entered.set()
        assert release.wait(10)
        return []

    monkeypatch.setattr(plugin_processes.xclim_processes, "scan", held_scan)
    for name in ("_scan_builtin_processes", "_scan_plugin_package_processes", "_scan_instance_processes"):
        monkeypatch.setattr(plugin_processes, name, lambda: [])
    monkeypatch.setattr(plugin_processes.earthkit_processes, "scan", lambda: [])
    return entered, release


def test_a_units_check_waits_for_a_plugin_load_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    # Loaded beforehand, so the check is quick once it may run: what is measured is the wait,
    # not how long a cold import of xclim takes.
    pytest.importorskip("xclim.core.units")
    entered, release = _hold_the_plugin_load(monkeypatch)
    warm_up = threading.Thread(target=plugin_processes.load_plugin_processes)
    warm_up.start()
    assert entered.wait(10)

    result: list[str | None] = []
    request = threading.Thread(target=lambda: result.append(cf.validate_units("m")))
    request.start()
    request.join(0.5)
    assert request.is_alive(), "the units check imported xclim alongside the plugin load"

    release.set()
    warm_up.join(10)
    request.join(10)
    assert result == [None]


def test_two_threads_build_the_registry_once(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[object] = []
    building = threading.Event()

    def build() -> object:
        building.set()
        threading.Event().wait(0.2)  # long enough for the other thread to arrive
        registry = object()
        builds.append(registry)
        return registry

    monkeypatch.setattr(execution, "_registry", None)
    monkeypatch.setattr(execution, "_new_process_registry", build)

    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(execution._build_process_registry())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(builds) == 1
    assert results == [builds[0], builds[0]]


def test_xclim_indicators_are_collected_once_without_the_caller_holding_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A caller of scan() that does not go through load_plugin_processes still reaches the xclim
    # imports one thread at a time.
    collects: list[list[object]] = []

    def collect() -> list[object]:
        threading.Event().wait(0.2)  # long enough for the other thread to arrive
        indicators: list[object] = [object()]
        collects.append(indicators)
        return indicators

    monkeypatch.setattr(xclim_processes, "_cache", None)
    monkeypatch.setattr(xclim_processes, "_collect", collect)

    results: list[list[object]] = []
    threads = [threading.Thread(target=lambda: results.append(xclim_processes.scan())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(collects) == 1
    assert results == [collects[0], collects[0]]
