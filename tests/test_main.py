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

        def shutdown(self) -> None:
            calls.append("openeo-shutdown")

    class FakeAutomationService:
        def consume(self, events: object) -> None:
            pass

        def start(self) -> None:
            calls.append("automation-start")

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
            "openeo-recover",
            "automation-start",
            "consumer-set",
            "automation-replay",
            "scheduler-start",
        ]

    assert calls == [
        "recover",
        "openeo-recover",
        "automation-start",
        "consumer-set",
        "automation-replay",
        "scheduler-start",
        "consumer-unset",
        "scheduler-shutdown",
        "shutdown",
        "openeo-shutdown",
    ]
