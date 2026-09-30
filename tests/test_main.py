from __future__ import annotations

import pytest
from fastapi import FastAPI

import open_climate_service.main as main


@pytest.mark.anyio
async def test_lifespan_recovers_jobs_and_shuts_down(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeJobService:
        def recover_pending_jobs(self) -> None:
            calls.append("recover")

        def shutdown(self) -> None:
            calls.append("shutdown")

        def set_event_consumer(self, consumer: object | None) -> None:
            calls.append("consumer-set" if consumer is not None else "consumer-unset")

    class FakeOpenEOJobService:
        def recover_pending_jobs(self) -> None:
            calls.append("openeo-recover")

        def set_finished_listener(self, listener: object) -> None:
            calls.append("finished-listener-set" if listener is not None else "finished-listener-cleared")

        def set_delivery_due_provider(self, provider: object) -> None:
            calls.append("delivery-due-set" if provider is not None else "delivery-due-cleared")

        def shutdown(self) -> None:
            calls.append("openeo-shutdown")

    class FakeAutomationService:
        def consume(self, events: object) -> None:
            pass

        def start(self) -> None:
            calls.append("automation-start")

        def on_job_finished(self, record: object) -> None:
            pass

        def delivery_due_for(self, record: object) -> None:
            return None

        def reconcile_deliveries(self) -> None:
            calls.append("automation-reconcile")

        def replay(self) -> None:
            calls.append("automation-replay")

    class FakeSchedulerService:
        def start(self) -> None:
            calls.append("scheduler-start")

        def shutdown(self) -> None:
            calls.append("scheduler-shutdown")

    monkeypatch.setattr(main, "get_job_service", lambda: FakeJobService())
    monkeypatch.setattr(main, "get_openeo_job_service", lambda: FakeOpenEOJobService())
    monkeypatch.setattr(main, "get_workflow_automation_service", lambda: FakeAutomationService())
    monkeypatch.setattr(main, "get_scheduler_service", lambda: FakeSchedulerService())

    async with main._lifespan(FastAPI()):
        assert calls == [
            "recover",
            "automation-start",
            "consumer-set",
            "delivery-due-set",
            "finished-listener-set",
            "openeo-recover",
            "automation-reconcile",
            "automation-replay",
            "scheduler-start",
        ]

    assert calls == [
        "recover",
        "automation-start",
        "consumer-set",
        "delivery-due-set",
        "finished-listener-set",
        "openeo-recover",
        "automation-reconcile",
        "automation-replay",
        "scheduler-start",
        "consumer-unset",
        "finished-listener-cleared",
        "scheduler-shutdown",
        "shutdown",
        "openeo-shutdown",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("read_only", [False, True])
async def test_lifespan_runs_pending_store_collections_at_startup_and_periodically(
    monkeypatch: pytest.MonkeyPatch, read_only: bool
) -> None:
    """What guarantees a failed attempt's data is freed even if the store is never synced again."""
    import asyncio

    from open_climate_service.ingestions import services

    sweeps: list[int] = []
    monkeypatch.setattr(services, "collect_pending_garbage_everywhere", lambda: sweeps.append(1) or 0)
    monkeypatch.setattr(services, "remove_leftover_rebuilds", lambda: 0)
    monkeypatch.setattr(main.api_config, "is_read_only", lambda: read_only)
    monkeypatch.setattr(main, "_MAINTENANCE_INTERVAL_S", 0.01)

    class _Quiet:
        def __getattr__(self, name: str) -> object:
            return lambda *args, **kwargs: None

    for factory in (
        "get_job_service",
        "get_openeo_job_service",
        "get_workflow_automation_service",
        "get_scheduler_service",
    ):
        monkeypatch.setattr(main, factory, lambda: _Quiet())

    async with main._lifespan(FastAPI()):
        await asyncio.sleep(0.1)
        seen = len(sweeps)
    await asyncio.sleep(0.05)

    if read_only:
        assert sweeps == []  # may share its data directory with a writing instance
    else:
        assert seen >= 2  # once at startup, then on the interval
        assert len(sweeps) == seen, "the loop stops with the app"
