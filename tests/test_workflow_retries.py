"""CLIM-1218: bounded retry and restart recovery for workflow jobs started by triggers."""

from __future__ import annotations

import time
from datetime import timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from open_climate_service.automation.config import TriggerDelivery, WorkflowTrigger
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.execution import SaveResultEnvelope
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now

# `instance_fixture` and `sent_fixture` register the `instance` and `sent` fixtures.
from tests.test_trigger_delivery import (
    _EXPORT,
    _await_deliveries,
    _await_openeo,
    _deliveries,
    _event,
    _frame,
    _service,
    _triggered_job_id,
    instance_fixture,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    sent_fixture,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)


def _failing_then(monkeypatch: pytest.MonkeyPatch, failures: list[BaseException]) -> list[int]:
    """Make each workflow run raise the next failure, then succeed once they run out."""
    calls: list[int] = []

    def run(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if failures:
            raise failures.pop(0)
        return SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": _EXPORT})

    monkeypatch.setattr(execution, "run_process_graph", run)
    return calls


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openeo_jobs, "_retry_delay_seconds", lambda attempt: 0)


def _await_done(job_id: str) -> OpenEOJobRecord:
    """Wait for a terminal state, allowing for a cold start (imports, first graph build)."""
    return _await_openeo(job_id, timeout=30.0)


def _await_backoff(job_id: str, timeout: float = 30.0) -> OpenEOJobRecord:
    """Wait for the QUEUED state a failed attempt leaves, not the QUEUED a job starts in."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = openeo_jobs.store_get_job(job_id)
        if record is not None and record.status == OpenEOJobStatus.QUEUED and record.retry_at is not None:
            return record
        time.sleep(0.02)
    pytest.fail(f"openEO job {job_id} did not start a retry backoff within {timeout}s")


# --- configuration -------------------------------------------------------------------------


def test_attempt_limit_defaults_to_three_and_is_bounded() -> None:
    trigger = WorkflowTrigger(id="t", on_update_of="d", workflow_id="w")
    assert trigger.max_attempts == 3
    for invalid in (0, 11):
        with pytest.raises(ValidationError):
            WorkflowTrigger(id="t", on_update_of="d", workflow_id="w", max_attempts=invalid)


# --- transient and permanent failures ------------------------------------------------------


def test_transient_failure_is_retried_and_only_the_successful_attempt_delivers(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    calls = _failing_then(monkeypatch, [OSError("remote store unreachable")])
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert record.status == OpenEOJobStatus.FINISHED
    assert (record.attempt, record.max_attempts, len(calls)) == (2, 3, 2)
    logs = (record.logs or "").splitlines()
    assert "attempt 1 of 3 failed: OSError: remote store unreachable" in logs[0]
    assert logs[-1].endswith("attempt 2 of 3 finished")  # the history ends with the outcome
    # The delivery runs as its own job after the finish: wait for it rather than assume it.
    [delivery] = _await_deliveries(1, timeout=30.0)
    assert delivery.request["job_id"] == job_id
    assert len(sent) == 1


def test_a_retry_waits_out_its_backoff_without_a_worker(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [OSError("remote store unreachable")])
    openeo = instance["openeo"]
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_backoff(job_id)
    time.sleep(0.2)

    assert record.retry_at is not None and record.updated is not None
    # Both are stamped when the failure is recorded, so their gap is the backoff itself,
    # the first being one minute, however slowly this test reaches the assertion.
    assert 59 <= (record.retry_at - record.updated).total_seconds() <= 60
    assert job_id in openeo._retry_timers
    assert job_id not in openeo._futures
    assert _deliveries() == []


def _invalid_graph() -> BaseException:
    from open_climate_service.openeo.execution import InvalidProcessGraph

    try:
        raise InvalidProcessGraph(status_code=400, detail="Invalid process graph: unknown process") from KeyError("x")
    except InvalidProcessGraph as exc:
        return exc


@pytest.mark.parametrize(
    "failure",
    [
        _invalid_graph(),
        HTTPException(status_code=404, detail="Collection 'rain_monthly' not found"),
    ],
    ids=["invalid-graph", "request-error"],
)
def test_permanent_error_is_not_retried(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    calls = _failing_then(monkeypatch, [failure, failure, failure])
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert record.status == OpenEOJobStatus.ERROR
    assert len(calls) == 1
    assert (record.error_message or "").endswith("(attempt 1 of 3)")
    assert "failed with a permanent error, not retried" in (record.logs or "")
    assert _deliveries() == []


def test_an_invalid_export_found_while_saving_is_not_retried(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[int] = []

    def run(*args: Any, **kwargs: Any) -> Any:
        runs.append(1)
        return SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": "no-such-export"})

    monkeypatch.setattr(execution, "run_process_graph", run)
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert record.status == OpenEOJobStatus.ERROR
    assert "Unknown export 'no-such-export'" in (record.error_message or "")
    assert len(runs) == 1


def test_a_validation_type_error_while_running_is_retried(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    """A truncated remote response raises JSONDecodeError, a ValueError; it is worth a retry."""
    import json

    calls = _failing_then(monkeypatch, [json.JSONDecodeError("Expecting value", "", 0)])
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert (record.status, record.attempt, len(calls)) == (OpenEOJobStatus.FINISHED, 2, 2)


@pytest.mark.parametrize("phase", ["graph", "save"])
def test_cooperative_cancellation_is_not_retried_or_delivered(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """The cancellation scope must survive the distinction between graph and save failures."""
    from open_climate_service.shared.cancellation import raise_if_cancelled

    openeo = instance["openeo"]
    calls: list[str] = []

    def cancel() -> None:
        [job] = openeo_jobs.store_list_jobs()
        openeo.cancel_job(job.id)
        raise_if_cancelled(force=True)
        pytest.fail("cancellation did not escape the active execution scope")

    def run(*args: Any, **kwargs: Any) -> Any:
        calls.append("graph")
        if phase == "graph":
            cancel()
        return SaveResultEnvelope(_frame(), "DHIS2JSON", {"export": _EXPORT})

    def save(*args: Any, **kwargs: Any) -> None:
        calls.append("save")
        cancel()

    monkeypatch.setattr(execution, "run_process_graph", run)
    monkeypatch.setattr(openeo, "_persist_result", save)
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))
    record = _await_done(job_id)
    deadline = time.monotonic() + 5
    while job_id in openeo._futures and time.monotonic() < deadline:
        time.sleep(0.01)

    assert (record.status, record.attempt, record.max_attempts) == (OpenEOJobStatus.CANCELED, 1, 3)
    assert calls == (["graph"] if phase == "graph" else ["graph", "save"])
    assert job_id not in openeo._futures
    assert job_id not in openeo._retry_timers
    assert record.publishing is False
    assert _deliveries() == []
    assert sent == []


def test_exhausted_attempts_leave_the_job_visibly_failed(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    calls = _failing_then(monkeypatch, [OSError(f"outage {n}") for n in range(1, 4)])
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert record.status == OpenEOJobStatus.ERROR
    assert len(calls) == 3
    assert record.error_message == "OSError: outage 3 (attempt 3 of 3)"
    logs = record.logs or ""
    assert "attempt 1 of 3 failed: OSError: outage 1" in logs
    assert "attempt 3 of 3 failed: OSError: outage 3" in logs
    # The error is what GET /jobs/{id}/results answers with.
    with pytest.raises(HTTPException) as error:
        instance["openeo"].get_results(job_id)
    assert (error.value.status_code, error.value.detail) == (424, record.error_message)
    assert _deliveries() == []


def test_a_replayed_event_does_not_create_a_second_job_for_a_retrying_workflow(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [OSError("remote store unreachable")])
    service = _service(instance, TriggerDelivery(export=_EXPORT))
    job_id = _triggered_job_id(service)
    _await_backoff(job_id)

    service.consume([_event()])  # the same durable event, replayed

    assert [record.id for record in openeo_jobs.store_list_jobs()] == [job_id]


# --- cancellation and shutdown during a backoff --------------------------------------------


def _wait_for_timer(openeo: Any, job_id: str, timeout: float = 30.0) -> None:
    """The backoff is recorded first and its timer registered once the worker lets go."""
    deadline = time.monotonic() + timeout
    while job_id not in openeo._retry_timers:
        if time.monotonic() > deadline:
            pytest.fail(f"no retry timer for {job_id}")
        time.sleep(0.01)


def test_a_cancel_landing_before_the_retry_timer_exists_takes_effect_at_once(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [OSError("remote store unreachable")])
    openeo = instance["openeo"]
    original = openeo_jobs.OpenEOJobService._schedule_retry

    def cancel_first(self: Any, job_id: str, seconds: float) -> None:
        # The failure is recorded as QUEUED with retry_at; the cancel arrives before the timer.
        self.cancel_job(job_id)
        original(self, job_id, seconds)

    monkeypatch.setattr(openeo_jobs.OpenEOJobService, "_schedule_retry", cancel_first)
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))

    record = _await_done(job_id)

    assert record.status == OpenEOJobStatus.CANCELED  # not after the one-minute backoff
    assert job_id not in openeo._retry_timers
    assert len(calls) == 1
    assert "cancelled during the retry backoff after attempt 1" in (record.logs or "")


def test_a_cancelled_job_is_logged_when_dequeued(instance: dict[str, Any]) -> None:
    """A cancellation observed before execution is visible in the attempt history."""
    job_id = "cancelled-before-attempt"
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id=job_id,
            status=OpenEOJobStatus.QUEUED,
            created=utc_now(),
            process={"process_graph": {}},
            cancel_requested=True,
        )
    )

    assert instance["openeo"]._execute(job_id) is None

    record = openeo_jobs.store_get_job(job_id)
    assert record is not None
    assert record.status == OpenEOJobStatus.CANCELED
    assert "cancelled before the next attempt started" in (record.logs or "")


def test_cancelling_during_a_backoff_takes_effect_at_once(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [OSError("remote store unreachable")])
    openeo = instance["openeo"]
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))
    _await_backoff(job_id)
    _wait_for_timer(openeo, job_id)

    openeo.cancel_job(job_id)

    assert openeo_jobs.store_get_job(job_id).status == OpenEOJobStatus.CANCELED  # type: ignore[union-attr]
    assert job_id not in openeo._retry_timers
    assert len(calls) == 1


def test_a_backoff_survives_shutdown_and_resumes_at_the_next_start(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [OSError("remote store unreachable")])
    openeo = instance["openeo"]
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))
    queued = _await_backoff(job_id)
    time.sleep(0.1)
    openeo.shutdown()
    assert openeo._retry_timers == {}

    restarted = openeo_jobs.OpenEOJobService()
    try:
        restarted.recover_pending_jobs()
        # Not yet due: waits the remainder of its backoff rather than running at once.
        assert job_id in restarted._retry_timers
        assert openeo_jobs.store_get_job(job_id).status == OpenEOJobStatus.QUEUED  # type: ignore[union-attr]
        assert openeo_jobs.store_get_job(job_id).retry_at == queued.retry_at  # type: ignore[union-attr]
    finally:
        restarted.shutdown()


def test_a_job_whose_backoff_passed_while_down_runs_at_the_next_start(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [])
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="overdue",
            status=OpenEOJobStatus.QUEUED,
            created=utc_now(),
            process={"process_graph": {}},
            trigger_id="rain-to-districts",
            attempt=1,
            max_attempts=3,
            retry_at=utc_now() - timedelta(minutes=5),
        )
    )
    instance["openeo"].recover_pending_jobs()
    record = _await_done("overdue")
    assert (record.status, record.attempt) == (OpenEOJobStatus.FINISHED, 2)


# --- restart recovery ----------------------------------------------------------------------


def _interrupted(job_id: str, *, attempt: int, trigger_id: str | None = "rain-to-districts") -> None:
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id=job_id,
            status=OpenEOJobStatus.RUNNING,
            created=utc_now(),
            process={"process_graph": {}},
            trigger_id=trigger_id,
            attempt=attempt,
            max_attempts=3 if trigger_id else 1,
        )
    )


def test_a_triggered_job_interrupted_by_a_restart_is_requeued(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [])
    _interrupted("interrupted", attempt=1)

    instance["openeo"].recover_pending_jobs()
    record = _await_done("interrupted")

    assert (record.status, record.attempt, len(calls)) == (OpenEOJobStatus.FINISHED, 2, 1)
    assert "attempt 1 of 3 interrupted by a server restart; requeued" in (record.logs or "")


def test_an_interrupted_job_with_no_attempts_left_is_marked_failed(instance: dict[str, Any]) -> None:
    _interrupted("exhausted", attempt=3)
    instance["openeo"].recover_pending_jobs()
    record = openeo_jobs.store_get_job("exhausted")
    assert record is not None
    assert (record.status, record.error_message) == (
        OpenEOJobStatus.ERROR,
        "Interrupted by server restart (attempt 3 of 3)",
    )


def test_an_interrupted_manual_job_is_still_marked_failed(instance: dict[str, Any]) -> None:
    _interrupted("manual", attempt=1, trigger_id=None)
    instance["openeo"].recover_pending_jobs()
    record = openeo_jobs.store_get_job("manual")
    assert record is not None
    assert (record.status, record.error_message) == (OpenEOJobStatus.ERROR, "Interrupted by server restart")


def test_a_job_running_in_another_process_is_not_marked_failed_and_is_taken_over(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overlapping restart (CLIM-1252): the old process still holds the job's lease."""
    import portalocker

    calls = _failing_then(monkeypatch, [])
    openeo = instance["openeo"]
    openeo.lease_poll_seconds = 0.05
    _interrupted("elsewhere", attempt=1)
    lease = openeo_jobs._execution_lease_path("elsewhere")
    lock_path: Path = lease.with_suffix(lease.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
    portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
    try:
        openeo.recover_pending_jobs()
        time.sleep(0.2)
        assert openeo_jobs.store_get_job("elsewhere").status == OpenEOJobStatus.RUNNING  # type: ignore[union-attr]
        assert calls == []
    finally:
        # The old process exits without finishing the job.
        portalocker.unlock(handle)
        handle.close()

    record = _await_done("elsewhere")
    assert (record.status, record.attempt) == (OpenEOJobStatus.FINISHED, 2)


def test_a_manual_rerun_starts_a_fresh_attempt_budget(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    _failing_then(monkeypatch, [OSError(f"outage {n}") for n in range(1, 4)])
    openeo = instance["openeo"]
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT)))
    assert _await_done(job_id).status == OpenEOJobStatus.ERROR

    openeo.start_job(job_id)

    rerun = _await_done(job_id)
    assert (rerun.status, rerun.attempt) == (OpenEOJobStatus.FINISHED, 1)


# --- classification ------------------------------------------------------------------------


def _wrapped(status: int, cause: BaseException) -> HTTPException:
    try:
        raise HTTPException(status_code=status, detail=str(cause)) from cause
    except HTTPException as exc:
        return exc


def _raised_while_handling(exc: BaseException, handled: BaseException) -> BaseException:
    """``exc`` with ``handled`` as its implicit ``__context__``, not its cause."""
    try:
        try:
            raise handled
        except type(handled):
            raise exc  # noqa: B904 -- the implicit context is the point
    except type(exc) as raised:
        return raised


@pytest.mark.parametrize(
    ("exc", "while_saving", "permanent"),
    [
        (_invalid_graph(), False, True),
        # The executor's 400 wrapping a runtime error says nothing about the request.
        (_wrapped(400, TypeError("bad argument")), False, False),
        (_wrapped(400, JSONDecodeError("Expecting value", "", 0)), False, False),
        (_wrapped(500, ValueError("partly readable store")), False, False),
        (_wrapped(500, OSError("connection reset")), False, False),
        (_wrapped(500, RuntimeError("store is being written by an ingestion")), False, False),
        (_wrapped(500, TimeoutError("read timed out")), False, False),
        (HTTPException(status_code=409, detail="in use"), False, False),
        (HTTPException(status_code=429, detail="rate limited"), False, False),
        (HTTPException(status_code=404, detail="no such collection"), False, True),
        (FileNotFoundError("gone"), False, False),
        (KeyError("variable"), False, False),
        # While saving, a validation error means invalid configuration; I/O does not.
        (ValueError("Unknown export 'x'"), True, True),
        (KeyError("series"), True, True),
        (OSError("disk full"), True, False),
        # Only explicit causes count: a ConnectionError raised while handling a cache miss.
        (_raised_while_handling(ConnectionError("reset"), KeyError("cache miss")), True, False),
    ],
)
def test_failures_are_classified(exc: BaseException, while_saving: bool, permanent: bool) -> None:
    assert openeo_jobs._is_permanent_error(exc, while_saving=while_saving) is permanent


@pytest.mark.parametrize("stage", ["build", "compile", "run"])
def test_executor_distinguishes_invalid_graphs_from_runtime_value_errors(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from unittest.mock import MagicMock

    from open_climate_service.openeo import workflows

    failure = JSONDecodeError("Expecting value", "", 0)
    graph_factory = MagicMock()
    graph = graph_factory.return_value
    if stage == "build":
        graph_factory.side_effect = failure
    elif stage == "compile":
        graph.to_callable.side_effect = failure
    else:
        graph.to_callable.return_value.side_effect = failure
    monkeypatch.setattr("openeo_pg_parser_networkx.OpenEOProcessGraph", graph_factory)
    monkeypatch.setattr(execution, "_build_process_registry", lambda: {})
    monkeypatch.setattr(workflows, "list_workflows", lambda: MagicMock(processes=[]))

    with pytest.raises(HTTPException) as raised:
        execution.run_process_graph({"process_graph": {}})

    assert raised.value.__cause__ is failure
    assert isinstance(raised.value, execution.InvalidProcessGraph) is (stage != "run")
    assert openeo_jobs._is_permanent_error(raised.value) is (stage != "run")


# --- service API invariants ----------------------------------------------------------------


@pytest.mark.parametrize("invalid", [0, -1, 11, True, 2.5])
def test_create_triggered_job_refuses_an_attempt_limit_outside_the_bounds(
    instance: dict[str, Any], invalid: Any
) -> None:
    from open_climate_service.openeo.schemas import OpenEOJobCreate

    body = OpenEOJobCreate(process={"process_graph": {"result": {"process_id": "constant"}}})
    with pytest.raises(ValueError, match="max_attempts must be an integer from 1 to 10"):
        instance["openeo"].create_triggered_job(body, source_event_id="e:0", trigger_id="t", max_attempts=invalid)
    assert openeo_jobs.store_list_jobs() == []


def test_a_retry_timer_firing_during_shutdown_enqueues_nothing(instance: dict[str, Any]) -> None:
    openeo = instance["openeo"]
    enqueued: list[str] = []
    openeo._enqueue = lambda job_id: enqueued.append(job_id)  # type: ignore[method-assign]
    openeo._schedule_retry("late", 3600)
    timer = openeo._retry_timers["late"]
    # Shutdown has begun but has not yet cancelled this timer when it fires.
    openeo._stopping.set()
    openeo._retry_due("late")
    timer.cancel()
    assert enqueued == []


def test_scheduling_a_retry_again_replaces_the_pending_timer(instance: dict[str, Any]) -> None:
    openeo = instance["openeo"]
    openeo._schedule_retry("twice", 3600)
    first = openeo._retry_timers["twice"]
    openeo._schedule_retry("twice", 3600)
    second = openeo._retry_timers["twice"]
    try:
        assert second is not first
        first.join(timeout=1)
        assert not first.is_alive()  # the replaced timer was cancelled, not left to fire
    finally:
        second.cancel()


def test_a_replaced_timer_that_fires_anyway_does_not_cut_the_new_backoff_short(instance: dict[str, Any]) -> None:
    openeo = instance["openeo"]
    enqueued: list[str] = []
    openeo._enqueue = lambda job_id: enqueued.append(job_id)  # type: ignore[method-assign]
    openeo._schedule_retry("rescheduled", 3600)
    old = openeo._retry_timers["rescheduled"]
    openeo._schedule_retry("rescheduled", 3600)
    new = openeo._retry_timers["rescheduled"]
    try:
        # The old timer had already entered its callback when it was replaced.
        openeo._retry_due("rescheduled", old)
        assert enqueued == []
        assert openeo._retry_timers["rescheduled"] is new
        # The registered timer still requeues the job when it fires.
        openeo._retry_due("rescheduled", new)
        assert enqueued == ["rescheduled"]
    finally:
        new.cancel()


def test_a_single_attempt_job_keeps_no_attempt_history(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [])
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT), max_attempts=1))
    record = _await_done(job_id)
    assert (record.status, record.logs) == (OpenEOJobStatus.FINISHED, None)


# --- review follow-ups -----------------------------------------------------------------------


def test_a_worker_does_not_run_a_job_before_its_backoff_has_passed(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """E.g. a second process requeued it after a takeover while the backoff was still running."""
    calls = _failing_then(monkeypatch, [])
    openeo = instance["openeo"]
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="early",
            status=OpenEOJobStatus.QUEUED,
            created=utc_now(),
            process={"process_graph": {}},
            trigger_id="rain-to-districts",
            attempt=1,
            max_attempts=3,
            retry_at=utc_now() + timedelta(minutes=2),
        )
    )

    openeo._run_job("early")

    assert calls == []
    assert "early" in openeo._retry_timers  # waits out the remainder instead
    assert openeo_jobs.store_get_job("early").attempt == 1  # type: ignore[union-attr]


def test_one_unrecoverable_job_does_not_stop_recovery_of_the_others(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [])
    openeo = instance["openeo"]
    for job_id in ("broken", "fine"):
        openeo_jobs.store_create_job(
            OpenEOJobRecord(id=job_id, status=OpenEOJobStatus.QUEUED, created=utc_now(), process={"process_graph": {}})
        )
    original = openeo._recover

    def recover(job_id: str) -> bool:
        if job_id == "broken":
            raise RuntimeError("corrupt record")
        return original(job_id)

    monkeypatch.setattr(openeo, "_recover", recover)
    openeo.recover_pending_jobs()

    assert _await_done("fine").status == OpenEOJobStatus.FINISHED
    assert len(calls) == 1


def test_deleting_a_job_removes_its_execution_lease_file(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_then(monkeypatch, [])
    openeo = instance["openeo"]
    job_id = _triggered_job_id(_service(instance, TriggerDelivery(export=_EXPORT, dry_run=True)))
    _await_done(job_id)
    lease = openeo_jobs._execution_lease_path(job_id)
    lease_file = lease.with_suffix(lease.suffix + ".lock")
    assert lease_file.exists()
    _wait_for_no_worker(openeo, job_id)

    openeo.delete_job(job_id)

    assert not lease_file.exists()


def _wait_for_no_worker(openeo: Any, job_id: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while job_id in openeo._futures:
        if time.monotonic() > deadline:
            pytest.fail(f"worker for {job_id} did not finish")
        time.sleep(0.01)


def test_re_running_a_cancelled_job_is_not_cancelled_again(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [])
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="stopped",
            status=OpenEOJobStatus.CANCELED,
            created=utc_now(),
            process={"process_graph": {}},
            cancel_requested=True,
        )
    )

    instance["openeo"].start_job("stopped")

    record = _await_done("stopped")
    assert (record.status, record.cancel_requested, len(calls)) == (OpenEOJobStatus.FINISHED, False, 1)


# --- CLIM-1252: a job held by another process ------------------------------------------------


def _hold_lease(job_id: str) -> Any:
    import portalocker

    lease = openeo_jobs._execution_lease_path(job_id)
    lock_path = lease.with_suffix(lease.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
    portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
    return handle


def _release_lease(handle: Any) -> None:
    import portalocker

    portalocker.unlock(handle)
    handle.close()


def test_a_job_the_other_process_finishes_is_left_as_it_wrote_it(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [])
    openeo = instance["openeo"]
    openeo.lease_poll_seconds = 0.05
    _interrupted("finished-elsewhere", attempt=1)
    handle = _hold_lease("finished-elsewhere")
    openeo.recover_pending_jobs()
    finished_at = utc_now()
    # The other process finishes the job, then exits.
    openeo_jobs.store_update_job(
        "finished-elsewhere",
        lambda r: r.model_copy(update={"status": OpenEOJobStatus.FINISHED, "finished_at": finished_at}),
    )
    written = openeo_jobs.store_get_job("finished-elsewhere")
    _release_lease(handle)
    time.sleep(0.5)  # several takeover polls

    assert openeo_jobs.store_get_job("finished-elsewhere") == written
    assert calls == []


def test_a_worker_that_wins_the_lease_never_reruns_a_finished_job(
    instance: dict[str, Any], sent: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _failing_then(monkeypatch, [])
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="done", status=OpenEOJobStatus.FINISHED, created=utc_now(), process={"process_graph": {}}, attempt=1
        )
    )
    before = openeo_jobs.store_get_job("done")

    instance["openeo"]._run_job("done")  # e.g. queued here before another process finished it

    assert openeo_jobs.store_get_job("done") == before
    assert calls == []
