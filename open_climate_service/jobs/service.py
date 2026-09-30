"""Runtime service for native asynchronous process jobs."""

from __future__ import annotations

import inspect
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Protocol
from uuid import uuid4

from fastapi import HTTPException

from open_climate_service.jobs import store
from open_climate_service.jobs.models import (
    JobCancelledError,
    JobError,
    JobEvent,
    JobEventDraft,
    JobExecutionResult,
    JobLink,
    JobListResponse,
    JobProgress,
    JobRecord,
    JobStatus,
)
from open_climate_service.shared.compute import get_job_slots
from open_climate_service.shared.dynamic_import import get_dynamic_function
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)


def _retry_delay_seconds(attempt: int) -> int:
    """Return the retry delay in seconds for a given failed attempt count."""
    exponent = attempt - 1 if attempt > 1 else 0
    return int(min(240, 60 * (2**exponent)))


def _job_links(job_id: str, href_base: str = "/jobs") -> list[JobLink]:
    base = href_base.rstrip("/")
    if not base:
        base = "/jobs"
    return [JobLink(href=f"{base}/{job_id}", rel="self", title="Job detail")]


def _persisted_events(job_id: str, drafts: list[JobEventDraft], time: Any) -> list[JobEvent]:
    """Assign each event its durable identity: the job id and its position in the job."""
    return [
        JobEvent(event_id=f"{job_id}:{index}", time=time, **draft.model_dump()) for index, draft in enumerate(drafts)
    ]


def _catalog_links() -> list[JobLink]:
    return [JobLink(href="/jobs", rel="self", title="Jobs")]


def _supports_argument(func: Any, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _is_pre_execution_cancellation(record: JobRecord) -> bool:
    return record.cancel_requested and record.status in {JobStatus.ACCEPTED, JobStatus.RETRYING}


class JobExecutionContext:
    """Callbacks and state helpers exposed to one running job."""

    def __init__(self, service: "JobService", job_id: str) -> None:
        self._service = service
        self.job_id = job_id

    def report_progress(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        self._service.update_progress(self.job_id, done=done, total=total, message=message)

    def is_cancel_requested(self) -> bool:
        record = store.get_job_record(self.job_id)
        return bool(record and record.cancel_requested)

    def save_cursor(self, cursor: dict[str, Any]) -> None:
        self._service.save_cursor(self.job_id, cursor)

    def load_cursor(self) -> dict[str, Any] | None:
        record = store.get_job_record(self.job_id)
        return None if record is None else record.cursor


class ProcessExecutor(Protocol):
    """Execution backend contract for native jobs."""

    kind: str

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[None]:
        """Submit one callable for asynchronous execution."""
        ...

    def shutdown(self) -> None:
        """Release executor resources."""
        ...


class ThreadProcessExecutor:
    """Default in-process thread-backed job executor."""

    kind = "thread"

    def __init__(self, *, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="climate-service-job")

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[None]:
        """Submit one callable to the thread pool."""
        return self._pool.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        """Stop the thread pool without waiting for queued work to finish."""
        self._pool.shutdown(wait=False, cancel_futures=True)


class JobService:
    """Persisted job store plus in-process executor runtime."""

    def __init__(self, *, executor: ProcessExecutor | None = None, max_workers: int = 4) -> None:
        self._executor = executor or ThreadProcessExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._event_consumer: Callable[[list[JobEvent]], None] | None = None
        self._stopping = threading.Event()
        # Jobs waiting out a retry backoff. A timer, not a sleeping worker, so a backoff
        # holds neither a job slot nor one of the executor's threads.
        self._retry_timers: dict[str, threading.Timer] = {}

    def set_event_consumer(self, consumer: Callable[[list[JobEvent]], None] | None) -> None:
        """Register the process-local consumer for newly persisted domain events."""
        self._event_consumer = consumer

    def shutdown(self) -> None:
        """Stop the executor without waiting for outstanding work.

        A job waiting out a retry backoff stays RETRYING, and the next start requeues it.
        """
        self._stopping.set()
        with self._lock:
            timers = list(self._retry_timers.values())
            self._retry_timers.clear()
        for timer in timers:
            timer.cancel()
        self._executor.shutdown()

    def list_jobs(self) -> JobListResponse:
        """Return all persisted jobs ordered by creation time descending."""
        records = sorted(store.list_job_records(), key=lambda record: record.created_at, reverse=True)
        return JobListResponse(jobs=records, links=_catalog_links())

    def get_job_or_404(self, job_id: str) -> JobRecord:
        """Return one persisted job or raise 404."""
        record = store.get_job_record(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return record

    def submit_callable_job(
        self,
        *,
        func: Any,
        label: str,
        request: dict[str, Any],
        max_attempts: int = 1,
        job_href_base: str = "/jobs",
        job_id: str | None = None,
    ) -> JobRecord:
        """Submit a callable directly as a background job — no YAML registry needed.

        ``func`` must be a module-level function so its dotted path can be stored
        in the job record and re-imported on restart.  ``label`` is the human-readable
        process name shown in GET /jobs/{id} as ``processID``.
        """
        qualname = getattr(func, "__qualname__", "")
        if "<locals>" in qualname:
            raise ValueError(f"submit_callable_job requires a module-level function; got {qualname!r}")
        fn_path = f"{func.__module__}.{qualname}"
        # Strip any caller-supplied __fn_path__ to prevent function-path injection,
        # then set the reserved key so it cannot be overridden.
        safe_request = {k: v for k, v in request.items() if k != "__fn_path__"}
        safe_request["__fn_path__"] = fn_path
        return self._create_and_enqueue(
            process_id=label,
            request=safe_request,
            max_attempts=max_attempts,
            job_href_base=job_href_base,
            job_id=job_id,
        )

    def record_completed_job(
        self,
        *,
        label: str,
        request: dict[str, Any],
        result: Any,
        events: list[JobEventDraft],
        job_href_base: str = "/jobs",
    ) -> JobRecord:
        """Persist work that already ran outside the queue as a successful job, with its events.

        Events are durable only on a job record: that is what startup replay reads. A caller
        that did its work synchronously, such as an HTTP request that ingested inline, records
        it here so its events reach automation exactly like a queued job's. The record is
        created in its terminal state and is never executed or recovered.
        """
        job_id = str(uuid4())
        now = utc_now()
        record = JobRecord(
            job_id=job_id,
            process_id=label,
            status=JobStatus.SUCCESSFUL,
            created_at=now,
            started_at=now,
            finished_at=now,
            attempt=1,
            executor_kind="inline",
            request={key: value for key, value in request.items() if key != "__fn_path__"},
            progress=JobProgress(message="Completed"),
            result=result,
            events=_persisted_events(job_id, events, now),
            links=_job_links(job_id, href_base=job_href_base),
        )
        created = store.create_job_record(record)
        self._consume_events(created)
        return created

    def _consume_events(self, record: JobRecord) -> None:
        if record.events and self._event_consumer is not None:
            try:
                self._event_consumer(record.events)
            except Exception:
                logger.exception("Failed to consume events for completed job %s", record.job_id)

    def _create_and_enqueue(
        self,
        *,
        process_id: str,
        request: dict[str, Any],
        max_attempts: int,
        job_href_base: str,
        job_id: str | None = None,
    ) -> JobRecord:
        job_id = job_id or str(uuid4())
        record = JobRecord(
            job_id=job_id,
            process_id=process_id,
            status=JobStatus.ACCEPTED,
            created_at=utc_now(),
            max_attempts=max_attempts,
            executor_kind=self._executor.kind,
            request=request,
            links=_job_links(job_id, href_base=job_href_base),
        )
        store.create_job_record(record)
        self._enqueue_job(record.job_id)
        return self.get_job_or_404(record.job_id)

    def request_cancellation(self, job_id: str) -> JobRecord:
        """Request cooperative cancellation for a job."""
        self.get_job_or_404(job_id)
        record = store.mutate_job_record(
            job_id,
            lambda current: (
                current
                if current.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}
                else current.model_copy(update={"cancel_requested": True})
            ),
        )

        if record.status == JobStatus.RETRYING:
            with self._lock:
                timer = self._retry_timers.pop(job_id, None)
            if timer is not None:
                # Popped before it fired, so no attempt will start: record it now rather than
                # when the backoff ends.
                timer.cancel()
                record = store.mutate_job_record(
                    job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.CANCELLED,
                            "finished_at": utc_now(),
                            "progress": JobProgress(message="Cancelled before retry execution resumed"),
                        }
                    ),
                )
        elif record.status == JobStatus.ACCEPTED:
            with self._lock:
                future = self._futures.get(job_id)
            if future is not None and future.cancel():
                record = store.mutate_job_record(
                    job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.CANCELLED,
                            "finished_at": utc_now(),
                            "progress": JobProgress(message="Cancellation accepted before execution started"),
                        }
                    ),
                )
        return record

    def recover_pending_jobs(self) -> None:
        """Requeue interrupted jobs on startup."""
        for record in store.list_job_records():
            if record.status not in {JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.RETRYING}:
                continue
            if record.cancel_requested:
                store.mutate_job_record(
                    record.job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.CANCELLED,
                            "finished_at": utc_now(),
                            "progress": JobProgress(message="Cancelled before recovery requeue"),
                        }
                    ),
                )
                continue
            if record.status == JobStatus.RUNNING:
                store.mutate_job_record(
                    record.job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.ACCEPTED,
                            "attempt": max(0, current.attempt - 1),
                            "finished_at": None,
                            "retry_after": None,
                            "error": None,
                            "progress": JobProgress(message="Requeued after restart during execution"),
                        }
                    ),
                )
            self._enqueue_job(record.job_id)

    def update_progress(
        self,
        job_id: str,
        *,
        done: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> JobRecord:
        """Persist progress details for one job."""

        def _mutation(current: JobRecord) -> JobRecord:
            current_done = done if done is not None else current.progress.done
            current_total = total if total is not None else current.progress.total
            percent: float | None = current.progress.percent
            if current_done is not None and current_total is not None and current_total != 0:
                percent = round((current_done / current_total) * 100.0, 2)
            progress = JobProgress(
                done=current_done,
                total=current_total,
                percent=percent,
                message=message if message is not None else current.progress.message,
            )
            return current.model_copy(update={"progress": progress})

        return store.mutate_job_record(job_id, _mutation)

    def save_cursor(self, job_id: str, cursor: dict[str, Any]) -> JobRecord:
        """Persist a lightweight checkpoint cursor for one job."""
        return store.mutate_job_record(job_id, lambda current: current.model_copy(update={"cursor": dict(cursor)}))

    def _enqueue_job(self, job_id: str) -> None:
        with self._lock:
            existing = self._futures.get(job_id)
            if existing is not None and not existing.done():
                return
            future = self._executor.submit(self._run_job, job_id)
            self._futures[job_id] = future

    def _schedule_retry(self, job_id: str, seconds: int) -> None:
        """Requeue a job once its retry backoff has passed, without holding a worker meanwhile."""
        with self._lock:
            if self._stopping.is_set():
                return  # stays RETRYING, so the next start requeues it
            timer = threading.Timer(seconds, self._retry_due, args=(job_id,))
            timer.daemon = True
            self._retry_timers[job_id] = timer
        timer.start()
        # A cancellation that arrived after the attempt failed but before the timer existed
        # found nothing to stop; requeue now so the pre-execution check records it.
        if self._cancel_requested(job_id):
            self._retry_due(job_id, cancel_timer=True)

    def _retry_due(self, job_id: str, *, cancel_timer: bool = False) -> None:
        with self._lock:
            timer = self._retry_timers.pop(job_id, None)
        if timer is None:
            return  # cancelled, or the service stopped
        if cancel_timer:
            timer.cancel()
        self._enqueue_job(job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            self._execute_job(job_id)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def _execute_job(self, job_id: str) -> None:
        while True:
            record = self.get_job_or_404(job_id)
            if _is_pre_execution_cancellation(record):
                message = (
                    "Cancellation accepted before execution started"
                    if record.status == JobStatus.ACCEPTED
                    else "Cancelled before retry execution resumed"
                )
                store.mutate_job_record(
                    job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.CANCELLED,
                            "finished_at": utc_now(),
                            "progress": JobProgress(message=message),
                        }
                    ),
                )
                return

            slots = get_job_slots()
            if not slots.acquire(
                should_stop=lambda: self._stopping.is_set() or self._cancel_requested(job_id),
                on_wait=lambda: self._report_waiting_for_slot(job_id),
            ):
                if self._stopping.is_set():
                    return  # still accepted, so the next start recovers it
                continue  # cancelled while waiting; the check above records it
            try:
                started = store.mutate_job_record(
                    job_id,
                    lambda current: current.model_copy(
                        update={
                            "status": JobStatus.RUNNING,
                            "started_at": current.started_at or utc_now(),
                            "finished_at": None,
                            "attempt": current.attempt + 1,
                            "retry_after": None,
                            "error": None,
                        }
                    ),
                )
                retry_after = self._run_attempt(job_id, started)
            finally:
                slots.release()
            if retry_after is not None:
                self._schedule_retry(job_id, retry_after)
            return

    def _cancel_requested(self, job_id: str) -> bool:
        record = store.get_job_record(job_id)
        return bool(record and record.cancel_requested)

    def _report_waiting_for_slot(self, job_id: str) -> None:
        store.mutate_job_record(
            job_id,
            lambda current: current.model_copy(update={"progress": JobProgress(message="Waiting for a free job slot")}),
        )

    def _run_attempt(self, job_id: str, started: JobRecord) -> int | None:
        """Run one attempt of a started job: the retry delay if it should run again, else None."""
        try:
            execution_result = self._invoke_process(started)
            completed_at = utc_now()
            if isinstance(execution_result, JobExecutionResult):
                result = execution_result.result
                events = _persisted_events(job_id, execution_result.events, completed_at)
            else:
                result = execution_result
                events = []
            completed = store.mutate_job_record(
                job_id,
                lambda current: current.model_copy(
                    update={
                        "status": JobStatus.SUCCESSFUL,
                        "finished_at": completed_at,
                        "result": result,
                        "events": events,
                        "progress": JobProgress(
                            done=current.progress.done,
                            total=current.progress.total,
                            percent=current.progress.percent,
                            message="Completed",
                        ),
                    }
                ),
            )
            self._consume_events(completed)
            return None
        except JobCancelledError as exc:
            cancelled_result = exc.result
            store.mutate_job_record(
                job_id,
                lambda current: current.model_copy(
                    update={
                        "status": JobStatus.CANCELLED,
                        "finished_at": utc_now(),
                        "result": cancelled_result,
                        "progress": JobProgress(
                            done=current.progress.done,
                            total=current.progress.total,
                            percent=current.progress.percent,
                            message="Cancelled",
                        ),
                    }
                ),
            )
            return None
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            error = JobError(type=type(exc).__name__, message=str(exc))
            if started.attempt < started.max_attempts:
                retry_after = _retry_delay_seconds(started.attempt)
                store.mutate_job_record(
                    job_id,
                    lambda latest: latest.model_copy(
                        update={
                            "status": JobStatus.RETRYING,
                            "retry_after": retry_after,
                            "error": error,
                            "progress": JobProgress(message="Retry scheduled"),
                        }
                    ),
                )
                return retry_after

            store.mutate_job_record(
                job_id,
                lambda latest: latest.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "finished_at": utc_now(),
                        "error": error,
                        "progress": JobProgress(
                            done=latest.progress.done,
                            total=latest.progress.total,
                            percent=latest.progress.percent,
                            message="Failed",
                        ),
                    }
                ),
            )
            return None

    def _invoke_process(self, record: JobRecord) -> Any:
        fn_path = record.request.get("__fn_path__")
        if not fn_path or not isinstance(fn_path, str):
            logger.warning(
                "Job '%s' has no __fn_path__ (may be a pre-migration job) — marking failed",
                record.job_id,
            )
            raise ValueError(
                f"Job '{record.job_id}' has no execution path recorded"
                " — this job may have been created before the current server version"
            )
        func = get_dynamic_function(fn_path)
        context = JobExecutionContext(self, record.job_id)
        kwargs = {k: v for k, v in record.request.items() if k != "__fn_path__"}
        if _supports_argument(func, "on_progress"):
            kwargs["on_progress"] = context.report_progress
        if _supports_argument(func, "is_cancel_requested"):
            kwargs["is_cancel_requested"] = context.is_cancel_requested
        if _supports_argument(func, "load_cursor"):
            kwargs["load_cursor"] = context.load_cursor
        if _supports_argument(func, "save_cursor"):
            kwargs["save_cursor"] = context.save_cursor
        return func(**kwargs)


_job_service: JobService | None = None


def get_job_service() -> JobService:
    """Return the singleton native job runtime."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service


def reset_job_service() -> None:
    """Reset the singleton runtime for tests."""
    global _job_service
    if _job_service is not None:
        _job_service.shutdown()
    _job_service = None
