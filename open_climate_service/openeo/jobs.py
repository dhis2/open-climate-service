"""openEO job persistence and execution service."""

from __future__ import annotations

import json
import logging
import numbers
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import NAMESPACE_URL, uuid4, uuid5

import portalocker
from fastapi import HTTPException

from open_climate_service import config as api_config
from open_climate_service.exports.retention import result_lease
from open_climate_service.exports.tabular import (
    _NON_VALUE_FIELDS as _NON_VALUE_FIELDS,
)
from open_climate_service.exports.tabular import (
    _build_dhis2_json_payload as _build_dhis2_json_payload,
)
from open_climate_service.exports.tabular import (
    _format_dhis2_timestamp as _format_dhis2_timestamp,
)
from open_climate_service.exports.tabular import (
    _is_nullish as _is_nullish,
)
from open_climate_service.exports.tabular import (
    _optional_str_option as _optional_str_option,
)
from open_climate_service.exports.tabular import (
    _select_dhis2_value_field as _select_dhis2_value_field,
)
from open_climate_service.exports.tabular import (
    _to_dhis2_period_string as _to_dhis2_period_string,
)
from open_climate_service.exports.tabular import (
    _to_dhis2_value_string as _to_dhis2_value_string,
)
from open_climate_service.exports.tabular import (
    non_value_fields as _non_value_fields,
)
from open_climate_service.openeo.schemas import (
    OpenEOJobCreate,
    OpenEOJobListResponse,
    OpenEOJobRecord,
    OpenEOJobResults,
    OpenEOJobStatus,
    OpenEOJobUpdate,
)
from open_climate_service.shared.cancellation import (
    ExecutionCancelled,
    cancellation_scope,
    enter_publication,
    raise_if_cancelled,
)
from open_climate_service.shared.cf import is_temperature_like
from open_climate_service.shared.compute import get_job_slots
from open_climate_service.shared.geoparquet import PARQUET_MEDIA_TYPE
from open_climate_service.shared.persistence import execution_lease
from open_climate_service.shared.storage_size import stored_bytes
from open_climate_service.shared.thumbnails import write_dataset_thumbnail
from open_climate_service.shared.time import utc_now
from open_climate_service.shared.vectors import encode_vector_cube, feature_id_field, holds_shapes, vector_dim
from open_climate_service.stac.media_types import ZARR_V3_MEDIA_TYPE, data_group_open_kwargs, zarr_media_type

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _resolve_openeo_jobs_dir() -> Path:
    return api_config.get_data_root() / "openeo_jobs"


_JOBS_DIR = _resolve_openeo_jobs_dir()


def _jobs_index() -> Path:
    """Return the openEO jobs index path, derived from ``_JOBS_DIR`` at call time.

    Derived at call time rather than import time so a test that monkeypatches ``_JOBS_DIR``
    isolates the whole store. The previous module-level constant froze ``jobs.json`` at import;
    ``_ensure_store`` then mkdir'd the patched ``tmp_path`` while ``write_text`` still targeted
    the real XDG path, whose parent does not exist on a fresh runner (CLIM-849 CI failure).
    """
    return _JOBS_DIR / "jobs.json"


def _ensure_store() -> None:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if not _jobs_index().exists():
        _jobs_index().write_text("[]\n", encoding="utf-8")


def _load_raw_records() -> list[dict[str, object]]:
    _ensure_store()
    with open(_jobs_index(), encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_SH)
        try:
            payload = json.load(fh)
        finally:
            portalocker.unlock(fh)
    if not isinstance(payload, list):
        raise ValueError("openeo jobs.json must contain a list")
    return payload


def _mutate_store(mutation: Callable[[list[dict[str, object]]], _T]) -> _T:
    _ensure_store()
    with open(_jobs_index(), "r+", encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        try:
            payload = json.load(fh)
            records: list[dict[str, object]] = payload if isinstance(payload, list) else []
            result = mutation(records)
            fh.seek(0)
            json.dump(records, fh, indent=2, default=str)
            fh.write("\n")
            fh.truncate()
            return result
        finally:
            portalocker.unlock(fh)


def store_list_jobs() -> list[OpenEOJobRecord]:
    """Return all persisted openEO job records."""
    return [OpenEOJobRecord.model_validate(r) for r in _load_raw_records()]


def store_get_job(job_id: str) -> OpenEOJobRecord | None:
    """Return one job record, or None if not found."""
    for raw in _load_raw_records():
        if raw.get("id") == job_id:
            return OpenEOJobRecord.model_validate(raw)
    return None


def store_create_job(record: OpenEOJobRecord) -> OpenEOJobRecord:
    """Persist a newly created job; raises ValueError if id already exists."""

    def _mutation(records: list[dict[str, object]]) -> OpenEOJobRecord:
        if any(r.get("id") == record.id for r in records):
            raise ValueError(f"Job '{record.id}' already exists")
        records.append(_serialize(record))
        return record

    return _mutate_store(_mutation)


def store_update_job(job_id: str, mutation: Callable[[OpenEOJobRecord], OpenEOJobRecord]) -> OpenEOJobRecord:
    """Load, mutate, and persist one existing job record."""

    def _apply(records: list[dict[str, object]]) -> OpenEOJobRecord:
        for idx, raw in enumerate(records):
            if raw.get("id") != job_id:
                continue
            updated = mutation(OpenEOJobRecord.model_validate(raw))
            records[idx] = _serialize(updated)
            return updated
        raise KeyError(job_id)

    return _mutate_store(_apply)


def store_delete_job(job_id: str) -> bool:
    """Delete a job; returns True if it existed."""

    def _mutation(records: list[dict[str, object]]) -> bool:
        for idx, raw in enumerate(records):
            if raw.get("id") == job_id:
                records.pop(idx)
                return True
        return False

    return _mutate_store(_mutation)


def _serialize(record: OpenEOJobRecord) -> dict[str, object]:
    # model_dump() respects Field(exclude=True) on error_message and cancel_requested,
    # which is correct for HTTP responses but wrong for disk persistence.
    # Explicitly re-add those fields so they survive a server restart.
    data: dict[str, object] = record.model_dump(mode="json", exclude_none=False)
    data["error_message"] = record.error_message
    data["cancel_requested"] = record.cancel_requested
    data["trigger_id"] = record.trigger_id
    data["source_event_id"] = record.source_event_id
    data["finished_at"] = record.finished_at.isoformat() if record.finished_at is not None else None
    data["delivery_due"] = record.delivery_due
    data["attempt"] = record.attempt
    data["max_attempts"] = record.max_attempts
    data["retry_at"] = record.retry_at.isoformat() if record.retry_at is not None else None
    data["publishing"] = record.publishing
    return data


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _cancel_requested(job_id: str) -> bool:
    record = store_get_job(job_id)
    return bool(record and record.cancel_requested)


def _seconds_until(moment: datetime | None) -> float:
    """Seconds from now until ``moment``; zero when it is absent or past."""
    if moment is None:
        return 0.0
    return max(0.0, (_as_utc(moment) - utc_now()).total_seconds())


def _execution_lease_path(job_id: str) -> Path:
    return _JOBS_DIR / ".execution-leases" / job_id


MAX_TRIGGERED_ATTEMPTS = 10
"""Upper bound on attempts per triggered job, shared with the trigger configuration."""


def _retry_delay_seconds(attempt: int) -> int:
    """Backoff before the next attempt: 1, 2, then 4 minutes."""
    return int(min(240, 60 * (2 ** max(0, attempt - 1))))


_VALIDATION_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    LookupError,
    NotImplementedError,
)
"""Validation-type failures. Permanent where they mean invalid configuration, not elsewhere."""

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 423, 429})


def _is_permanent_error(exc: BaseException, *, while_saving: bool = False) -> bool:
    """True when retrying cannot help.

    Decided by what failed and where, following only explicit causes (``raise ... from``):
    an implicit ``__context__`` is whatever happened to be handled at the time, so a
    ``ConnectionError`` raised while handling a cache-miss ``KeyError`` is not a ``KeyError``.

    * An invalid process graph is permanent: it fails the same way on every attempt.
    * An ``HTTPException`` a process raised itself with a 4xx, other than a conflict, lock,
      timeout or rate limit, is permanent: an unknown collection, a refused request.
    * While saving the result, a validation-type error is permanent: an unknown export, a
      mapping or unit mismatch. Its I/O errors are not.
    * Anything else raised while the graph runs is retried within the attempt budget. That
      includes ``ValueError``: a truncated remote response or a partly readable store raises
      one too, and type alone cannot tell it from a bad argument.
    """
    from open_climate_service.openeo.execution import InvalidProcessGraph

    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__
    for item in chain:
        if isinstance(item, InvalidProcessGraph):
            return True
        # A 4xx wrapping a cause is the executor translating a failure for HTTP callers;
        # only one raised directly states a request error.
        if isinstance(item, HTTPException) and item.__cause__ is None and 400 <= item.status_code < 500:
            return item.status_code not in _TRANSIENT_HTTP_STATUSES
    root = chain[-1]
    return while_saving and isinstance(root, _VALIDATION_ERROR_TYPES) and not isinstance(root, OSError)


def _with_attempts(message: str, record: OpenEOJobRecord) -> str:
    """Name the attempts used, for a job that could retry; a single-attempt job is unchanged."""
    if record.max_attempts <= 1:
        return message
    return f"{message} (attempt {record.attempt} of {record.max_attempts})"


def _append_log(record: OpenEOJobRecord, line: str) -> str:
    """Add one timestamped line to the job's visible attempt history."""
    entry = f"{utc_now().isoformat()} {line}"
    return f"{record.logs}\n{entry}" if record.logs else entry


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class OpenEOJobService:
    """Manages openEO job lifecycle and asynchronous execution."""

    def __init__(self, *, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openeo-job")
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._finished_listener: Callable[[OpenEOJobRecord], None] | None = None
        self._delivery_due: Callable[[OpenEOJobRecord], dict[str, str] | None] | None = None
        self._stopping = threading.Event()
        # Jobs waiting out a retry backoff: a timer requeues each, so no worker sleeps.
        self._retry_timers: dict[str, threading.Timer] = {}
        self._watched: set[str] = set()
        # How often a job leased by another process is checked for takeover.
        self.lease_poll_seconds = 5.0

    def set_delivery_due_provider(self, provider: Callable[[OpenEOJobRecord], dict[str, str] | None] | None) -> None:
        """Register the callback that says which delivery a job owes as it finishes.

        It is called inside the store mutation that marks the job FINISHED, so its answer is
        persisted atomically with that state. It must be a fast, in-memory lookup.
        """
        self._delivery_due = provider

    def set_finished_listener(self, listener: Callable[[OpenEOJobRecord], None] | None) -> None:
        """Register the process-local callback run after a job is persisted as FINISHED.

        The callback runs on the job's worker thread. Its failures are logged and never
        change the finished job's status.
        """
        self._finished_listener = listener

    def shutdown(self) -> None:
        """Stop executing. A job waiting out a retry backoff stays QUEUED for the next start."""
        self._stopping.set()
        with self._lock:
            timers = list(self._retry_timers.values())
            self._retry_timers.clear()
        for timer in timers:
            timer.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def recover_pending_jobs(self) -> None:
        """Recover jobs left in a non-terminal state from a previous server run.

        QUEUED jobs are re-enqueued, or wait out the rest of a retry backoff. A RUNNING
        triggered job with attempts left is requeued, since its interruption says nothing
        about the workflow; any other RUNNING job is marked ERROR. A job still executing in
        another live process, typically one that is shutting down, is left to it and watched
        instead, and taken over if that process exits without finishing it.
        """
        for record in store_list_jobs():
            if record.status not in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
                continue
            # One job that cannot be recovered must not stop the recovery of the others.
            try:
                requeue = False
                with execution_lease(_execution_lease_path(record.id)) as won:
                    if won:
                        requeue = self._recover(record.id)
                if not won:
                    logger.warning("openEO job %s is still executing in another process; watching it", record.id)
                    self._watch_for_takeover(record.id)
                elif requeue:
                    self._enqueue(record.id)
            except Exception:
                logger.exception("Could not recover openEO job %s; continuing with the others", record.id)

    def _recover(self, job_id: str) -> bool:
        """Prepare one interrupted job while holding its lease; return whether to run it now."""
        record = store_get_job(job_id)
        if record is None or record.status not in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
            return False
        if record.cancel_requested:
            store_update_job(
                job_id, lambda r: r.model_copy(update={"status": OpenEOJobStatus.CANCELED, "updated": utc_now()})
            )
            return False
        if record.status == OpenEOJobStatus.QUEUED:
            if record.retry_at is not None:
                remaining = (_as_utc(record.retry_at) - utc_now()).total_seconds()
                if remaining > 0:
                    logger.info("openEO job %s resumes its retry backoff (%.0fs left)", job_id, remaining)
                    self._schedule_retry(job_id, remaining)
                    return False
            logger.info("openEO job %s was QUEUED at restart — re-enqueueing", job_id)
            return True
        if record.trigger_id is not None and record.attempt < record.max_attempts:
            logger.warning("Triggered openEO job %s was interrupted by a restart; requeueing it", job_id)
            store_update_job(
                job_id,
                lambda r: r.model_copy(
                    update={
                        "status": OpenEOJobStatus.QUEUED,
                        "updated": utc_now(),
                        "publishing": False,
                        "logs": _append_log(
                            r, f"attempt {r.attempt} of {r.max_attempts} interrupted by a server restart; requeued"
                        ),
                    }
                ),
            )
            return True
        logger.warning("openEO job %s was RUNNING at restart — marking as error", job_id)
        store_update_job(
            job_id,
            lambda r: r.model_copy(
                update={
                    "status": OpenEOJobStatus.ERROR,
                    "error_message": _with_attempts("Interrupted by server restart", r),
                    "updated": utc_now(),
                    "publishing": False,
                    "logs": _append_log(r, f"attempt {r.attempt} of {r.max_attempts} interrupted by a server restart")
                    if r.max_attempts > 1
                    else r.logs,
                }
            ),
        )
        return False

    def _watch_for_takeover(self, job_id: str) -> None:
        """Take over a job from another process once its execution lease is released.

        The record is re-read under the lease, so a job the other process finished is left
        alone and one it abandoned is recovered as at startup. One watcher per job at most.
        """
        with self._lock:
            if self._stopping.is_set() or job_id in self._watched:
                return
            self._watched.add(job_id)

        def watch() -> None:
            try:
                while not self._stopping.wait(self.lease_poll_seconds):
                    with execution_lease(_execution_lease_path(job_id)) as won:
                        if not won:
                            continue
                        requeue = self._recover(job_id)
                    if requeue and not self._stopping.is_set():
                        logger.info("Took over openEO job %s after its previous process released it", job_id)
                        with self._lock:
                            self._watched.discard(job_id)
                        self._enqueue(job_id)
                    return
            finally:
                with self._lock:
                    self._watched.discard(job_id)

        threading.Thread(target=watch, name=f"openeo-takeover-{job_id}", daemon=True).start()

    # ------------------------------------------------------------------
    # HTTP-layer helpers
    # ------------------------------------------------------------------

    def list_jobs(self) -> OpenEOJobListResponse:
        records = sorted(store_list_jobs(), key=lambda r: r.created, reverse=True)
        return OpenEOJobListResponse(
            jobs=records,
            links=[{"rel": "self", "href": "/jobs", "type": "application/json"}],
        )

    def create_job(self, body: OpenEOJobCreate) -> OpenEOJobRecord:
        if not isinstance(body.process.get("process_graph"), dict):
            raise HTTPException(
                status_code=422,
                detail="process.process_graph must be an object",
            )
        job_id = str(uuid4())
        now = utc_now()
        record = OpenEOJobRecord(
            id=job_id,
            title=body.title if body.title is not None else _derive_job_title(body.process),
            description=body.description,
            process=body.process,
            status=OpenEOJobStatus.CREATED,
            created=now,
            updated=now,
            plan=body.plan,
            budget=body.budget,
            links=[
                {"rel": "self", "href": f"/jobs/{job_id}", "type": "application/json"},
                {"rel": "results", "href": f"/jobs/{job_id}/results", "type": "application/json"},
            ],
        )
        return store_create_job(record)

    def create_triggered_job(
        self,
        body: OpenEOJobCreate,
        *,
        source_event_id: str,
        trigger_id: str,
        max_attempts: int = 1,
    ) -> tuple[OpenEOJobRecord, bool]:
        """Create at most one job for a durable event and automation trigger.

        ``max_attempts`` bounds how often the job runs under its one deterministic ID: a
        transient failure or an interrupting restart is retried, a permanent error is not.
        It must lie within the same bounds as a trigger's configuration.
        """
        # `type(...) is int`, not isinstance: a bool is an int, and a caller outside the
        # configuration path is not held to the annotation at runtime.
        if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_TRIGGERED_ATTEMPTS:
            raise ValueError(
                f"max_attempts must be an integer from 1 to {MAX_TRIGGERED_ATTEMPTS}, got {max_attempts!r}"
            )
        if not isinstance(body.process.get("process_graph"), dict):
            raise ValueError("process.process_graph must be an object")
        job_id = str(uuid5(NAMESPACE_URL, f"ocs:{source_event_id}:{trigger_id}"))
        now = utc_now()
        candidate = OpenEOJobRecord(
            id=job_id,
            title=body.title if body.title is not None else _derive_job_title(body.process),
            description=body.description,
            process=body.process,
            status=OpenEOJobStatus.CREATED,
            created=now,
            updated=now,
            plan=body.plan,
            budget=body.budget,
            links=[
                {"rel": "self", "href": f"/jobs/{job_id}", "type": "application/json"},
                {"rel": "results", "href": f"/jobs/{job_id}/results", "type": "application/json"},
            ],
            trigger_id=trigger_id,
            source_event_id=source_event_id,
            max_attempts=max_attempts,
        )

        def _create_once(records: list[dict[str, object]]) -> tuple[OpenEOJobRecord, bool]:
            for raw in records:
                if raw.get("id") == job_id:
                    return OpenEOJobRecord.model_validate(raw), False
            records.append(_serialize(candidate))
            return candidate, True

        return _mutate_store(_create_once)

    def start_triggered_job(self, job_id: str) -> bool:
        """Atomically claim and enqueue a created automation job.

        The conditional transition prevents two OCS processes replaying the same
        durable event from both executing its deterministic job.
        """

        def _claim(records: list[dict[str, object]]) -> bool:
            for index, raw in enumerate(records):
                if raw.get("id") != job_id:
                    continue
                current = OpenEOJobRecord.model_validate(raw)
                if current.status != OpenEOJobStatus.CREATED:
                    return False
                records[index] = _serialize(
                    current.model_copy(update={"status": OpenEOJobStatus.QUEUED, "updated": utc_now()})
                )
                return True
            raise KeyError(job_id)

        claimed = _mutate_store(_claim)
        if claimed:
            self._enqueue(job_id)
        return claimed

    def get_job_or_404(self, job_id: str) -> OpenEOJobRecord:
        record = store_get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return record

    def update_job(self, job_id: str, body: OpenEOJobUpdate) -> OpenEOJobRecord:
        # 404 first so an arbitrary or malformed ID never creates a lease file.
        self.get_job_or_404(job_id)
        with result_lease(job_id):
            record = self.get_job_or_404(job_id)
            if record.status in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
                raise HTTPException(status_code=400, detail="Cannot update a job that is queued or running")
            updates: dict[str, Any] = {}
            if body.title is not None:
                updates["title"] = body.title
            if body.description is not None:
                updates["description"] = body.description
            if body.process is not None:
                if not isinstance(body.process.get("process_graph"), dict):
                    raise HTTPException(status_code=422, detail="process.process_graph must be an object")
                updates["process"] = body.process
            if body.plan is not None:
                updates["plan"] = body.plan
            if body.budget is not None:
                updates["budget"] = body.budget
            if updates:
                updates["updated"] = utc_now()
                return store_update_job(job_id, lambda r: r.model_copy(update=updates))
            return record

    def delete_job(self, job_id: str) -> None:
        # 404 first so an arbitrary or malformed ID never creates a lease file.
        record = self.get_job_or_404(job_id)
        if record.status in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
            raise HTTPException(status_code=400, detail="Cannot delete a running job; cancel it first")
        with result_lease(job_id):
            record = self.get_job_or_404(job_id)
            if record.status in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
                raise HTTPException(status_code=400, detail="Cannot delete a running job; cancel it first")
            store_delete_job(job_id)
            import shutil

            job_dir = _JOBS_DIR / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        # The lease files live outside the job directory; remove them now the job is gone.
        (_JOBS_DIR / ".export-locks" / f"{job_id}.lock").unlink(missing_ok=True)
        lease = _execution_lease_path(job_id)
        lease.with_suffix(lease.suffix + ".lock").unlink(missing_ok=True)

    def start_job(self, job_id: str) -> None:
        """Queue a job for processing (POST /jobs/{id}/results)."""
        # 404 first so an arbitrary or malformed ID never creates a lease file.
        self.get_job_or_404(job_id)
        with result_lease(job_id):
            record = self.get_job_or_404(job_id)
            if record.status == OpenEOJobStatus.RUNNING:
                raise HTTPException(status_code=400, detail="Job is already running")
            if record.status == OpenEOJobStatus.QUEUED:
                return
            # A deliberate re-run starts a fresh attempt budget.
            store_update_job(
                job_id,
                lambda r: r.model_copy(
                    update={
                        "status": OpenEOJobStatus.QUEUED,
                        "updated": utc_now(),
                        "attempt": 0,
                        "retry_at": None,
                        # A re-run is a new request: an earlier cancellation must not cancel it.
                        "cancel_requested": False,
                    }
                ),
            )
            self._enqueue(job_id)

    def cancel_job(self, job_id: str) -> None:
        """Request cancellation (DELETE /jobs/{id}/results)."""
        record = self.get_job_or_404(job_id)
        if record.status not in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
            raise HTTPException(status_code=400, detail="Job is not running or queued")

        # Read the future and attempt cancel while holding the lock so we don't race
        # with the executor thread transitioning QUEUED→RUNNING between our status
        # read and the future.cancel() call.
        with self._lock:
            future = self._futures.get(job_id)
            cancelled_before_start = future is not None and future.cancel()
            # A job waiting out a retry backoff has no worker: stopping its timer means no
            # attempt will start, so it is cancelled now rather than when the backoff ends.
            timer = self._retry_timers.pop(job_id, None)
        if timer is not None:
            timer.cancel()
            cancelled_before_start = True

        def _cancel(r: OpenEOJobRecord) -> OpenEOJobRecord:
            # Re-checked inside the store mutation: the worker may have finished the job since
            # the read above. Whichever mutation lands first decides. If cancellation does,
            # the worker's own finishing mutation sees the flag and records CANCELED; if
            # completion does, this refuses, so a finished job never gains a late flag after
            # its delivery may already have been submitted.
            if r.status not in {OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING}:
                raise HTTPException(status_code=400, detail="Job is not running or queued")
            if r.publishing:
                raise HTTPException(
                    status_code=409,
                    detail="Job is publishing its result and can no longer be cancelled",
                )
            if cancelled_before_start and r.status == OpenEOJobStatus.QUEUED:
                # future.cancel() returned True: the job was still queued in the thread pool
                # and will never start, or its retry timer was stopped.
                backing_off = r.retry_at is not None and r.attempt > 0
                return r.model_copy(
                    update={
                        "status": OpenEOJobStatus.CANCELED,
                        "updated": utc_now(),
                        "logs": _append_log(r, f"cancelled during the retry backoff after attempt {r.attempt}")
                        if backing_off
                        else r.logs,
                    }
                )
            # Running (or the worker already claimed it): cooperative cancellation; the worker
            # checks this flag in the same mutation that would mark the job FINISHED.
            return r.model_copy(update={"cancel_requested": True, "updated": utc_now()})

        store_update_job(job_id, _cancel)

    def get_results(self, job_id: str) -> OpenEOJobResults:
        """Return result asset links for a finished job."""
        record = self.get_job_or_404(job_id)
        if record.status == OpenEOJobStatus.ERROR:
            raise HTTPException(
                status_code=424,
                detail=record.error_message or "Job finished with an error",
            )
        if record.status != OpenEOJobStatus.FINISHED:
            raise HTTPException(
                status_code=400,
                detail=f"Results not available yet; job status is '{record.status}'",
            )
        assets = _result_assets(record)
        return OpenEOJobResults(
            stac_version="1.1.0",
            id=job_id,
            assets=assets,
            links=[{"rel": "self", "href": f"/jobs/{job_id}/results", "type": "application/json"}],
        )

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _enqueue(self, job_id: str) -> None:
        with self._lock:
            existing = self._futures.get(job_id)
            if existing is not None and not existing.done():
                return
            future = self._pool.submit(self._run_job, job_id)
            self._futures[job_id] = future

    def _run_job(self, job_id: str) -> None:
        retry_after: float | None = None
        watch = False
        try:
            with execution_lease(_execution_lease_path(job_id)) as won:
                if not won:
                    # Another process is executing this job. Leave its record alone, but watch
                    # it: if that process exits without finishing, nothing else would.
                    logger.warning("openEO job %s is executing in another process; watching it", job_id)
                    watch = True
                else:
                    # Re-read under the lease: another process may have finished or cancelled
                    # the job since it was queued here.
                    current = store_get_job(job_id)
                    if current is not None and current.status == OpenEOJobStatus.QUEUED:
                        wait = _seconds_until(current.retry_at)
                        if wait > 0:
                            # Queued here before its backoff passed, e.g. by a second process
                            # after a takeover: wait out the rest instead of retrying early.
                            retry_after = wait
                        else:
                            retry_after = self._execute_in_slot(job_id)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
        # Only once this worker is neither leased nor registered: otherwise a requeue from a
        # watcher or a retry timer could be refused as a duplicate and leave the job idle.
        if watch:
            self._watch_for_takeover(job_id)
        elif retry_after is not None:
            self._schedule_retry(job_id, retry_after)

    def _execute_in_slot(self, job_id: str) -> float | None:
        """Run one attempt holding a shared job slot, so ingests and openEO jobs share one limit.

        The job waits QUEUED for the slot. One cancelled meanwhile is still handed to `_execute`,
        which records the cancellation without computing; one still waiting at shutdown stays
        QUEUED for the next start to re-enqueue.
        """
        slots = get_job_slots()
        if not slots.acquire(should_stop=lambda: self._stopping.is_set() or _cancel_requested(job_id)):
            return None if self._stopping.is_set() else self._execute(job_id)
        try:
            return self._execute(job_id)
        finally:
            slots.release()

    def _schedule_retry(self, job_id: str, seconds: float) -> None:
        """Requeue a job once its retry backoff has passed, holding no worker meanwhile.

        A job has at most one pending retry: scheduling again replaces the earlier timer.
        """
        with self._lock:
            if self._stopping.is_set():
                return  # stays QUEUED with its retry_at, so the next start resumes the wait
            previous = self._retry_timers.pop(job_id, None)
            timer = threading.Timer(max(0.0, seconds), self._retry_due)
            # The timer passes itself, so a replaced timer that fires anyway can tell it is stale.
            timer.args = (job_id, timer)
            timer.daemon = True
            self._retry_timers[job_id] = timer
        if previous is not None:
            previous.cancel()
        timer.start()
        # A cancel that landed after the failure was recorded but before this timer existed
        # found no timer to stop and only set the flag. Apply it now, not when the backoff ends.
        if _cancel_requested(job_id):
            self._cancel_pending_retry(job_id)

    def _cancel_pending_retry(self, job_id: str) -> None:
        """Cancel a job waiting out its backoff: stop its timer and record it as cancelled."""
        with self._lock:
            timer = self._retry_timers.pop(job_id, None)
        if timer is None:
            return  # already fired, so the worker's pre-execution check records the cancel
        timer.cancel()

        def _cancelled(r: OpenEOJobRecord) -> OpenEOJobRecord:
            if r.status != OpenEOJobStatus.QUEUED:
                return r
            return r.model_copy(
                update={
                    "status": OpenEOJobStatus.CANCELED,
                    "updated": utc_now(),
                    "logs": _append_log(r, f"cancelled during the retry backoff after attempt {r.attempt}"),
                }
            )

        store_update_job(job_id, _cancelled)

    def _retry_due(self, job_id: str, timer: threading.Timer | None = None) -> None:
        """Requeue a job whose backoff has passed, if ``timer`` is still the one registered for it.

        A timer replaced by a later `_schedule_retry` may already be firing when it is
        cancelled. Without the identity check it would pop its replacement and requeue the job
        early, cutting the new backoff short. ``None`` acts on whichever timer is registered.
        """
        with self._lock:
            current = self._retry_timers.get(job_id)
            if current is None or (timer is not None and current is not timer):
                return  # cancelled, or superseded by a newer timer that will requeue the job
            del self._retry_timers[job_id]
            # A timer can fire while shutdown is in progress. The job then stays QUEUED with
            # its retry_at, and the next start requeues it.
            if self._stopping.is_set():
                return
        self._enqueue(job_id)

    def _execute(self, job_id: str) -> float | None:
        """Run one attempt; return the retry delay in seconds if the job should run again."""
        from open_climate_service.openeo.execution import run_process_graph

        record = store_get_job(job_id)
        if record is None:
            return None
        if record.cancel_requested:
            store_update_job(
                job_id,
                lambda r: r.model_copy(
                    update={
                        "status": OpenEOJobStatus.CANCELED,
                        "updated": utc_now(),
                        "logs": _append_log(r, "cancelled before the next attempt started"),
                    }
                ),
            )
            return None

        started = store_update_job(
            job_id,
            lambda r: r.model_copy(
                update={
                    "status": OpenEOJobStatus.RUNNING,
                    "updated": utc_now(),
                    "attempt": r.attempt + 1,
                    "retry_at": None,
                    "publishing": False,
                }
            ),
        )

        saving = False
        try:
            # Each attempt owns a fresh result directory. Without this, a successful rerun in
            # another format made files left by an earlier or cancelled attempt downloadable.
            import shutil

            results_dir = _JOBS_DIR / job_id / "results"
            shutil.rmtree(results_dir, ignore_errors=True)
            results_dir.mkdir(parents=True, exist_ok=True)
            # Cancellation is checked before every process and dask task, and once more,
            # atomically, at the point of no return of any publication (CLIM-1221).
            with cancellation_scope(
                lambda: _cancel_requested(job_id),
                enter_publication=lambda: self._enter_publication(job_id),
            ):
                result = run_process_graph(record.process)
                raise_if_cancelled(force=True)
                saving = True
                output_path = self._persist_result(job_id, result)
            finished = store_update_job(job_id, lambda r: self._finish(r, output_path))
        except ExecutionCancelled:
            logger.info("openEO job %s was cancelled while running; nothing was published", job_id)
            store_update_job(
                job_id,
                lambda r: r.model_copy(
                    update={"status": OpenEOJobStatus.CANCELED, "updated": utc_now(), "publishing": False}
                ),
            )
            return None
        except Exception as job_exc:
            logger.exception("openEO job %s failed", job_id)
            return self._record_failure(job_id, started, job_exc, while_saving=saving)
        # Outside the try: a listener failure must not turn a finished job into an error.
        # Only a finished attempt reaches this, so only a successful attempt can deliver.
        if finished.status == OpenEOJobStatus.FINISHED:
            self._notify_finished(finished)
        return None

    def _enter_publication(self, job_id: str) -> None:
        """Pass the point of no return, or raise if the job was cancelled first.

        One store mutation, so it is atomic with `cancel_job`: whichever lands first decides.
        If the cancellation did, nothing is published. If this did, the attempt finishes,
        and a later cancel request is refused rather than leaving a half-published result.
        """

        def _gate(r: OpenEOJobRecord) -> OpenEOJobRecord:
            if r.cancel_requested:
                raise ExecutionCancelled("The job was cancelled before it could publish")
            return r.model_copy(update={"publishing": True, "updated": utc_now()})

        store_update_job(job_id, _gate)

    def _record_failure(
        self, job_id: str, started: OpenEOJobRecord, exc: Exception, *, while_saving: bool = False
    ) -> float | None:
        """Record a failed attempt: requeue it for a retry, or mark the job ERROR."""
        error = f"{type(exc).__name__}: {exc}"
        permanent = _is_permanent_error(exc, while_saving=while_saving)
        if not permanent and started.attempt < started.max_attempts:
            delay = float(_retry_delay_seconds(started.attempt))
            retry_at = utc_now() + timedelta(seconds=delay)

            def _retry(r: OpenEOJobRecord) -> OpenEOJobRecord:
                if r.cancel_requested:  # cancelled while the attempt ran: nothing to retry
                    return r.model_copy(
                        update={
                            "status": OpenEOJobStatus.CANCELED,
                            "updated": utc_now(),
                            "logs": _append_log(
                                r, f"attempt {r.attempt} of {r.max_attempts} failed: {error}; cancelled"
                            ),
                        }
                    )
                return r.model_copy(
                    update={
                        "status": OpenEOJobStatus.QUEUED,
                        "error_message": error,
                        "retry_at": retry_at,
                        "publishing": False,
                        "updated": utc_now(),
                        "logs": _append_log(
                            r,
                            f"attempt {r.attempt} of {r.max_attempts} failed: {error}; "
                            f"retrying at {retry_at.isoformat()}",
                        ),
                    }
                )

            updated = store_update_job(job_id, _retry)
            return delay if updated.status == OpenEOJobStatus.QUEUED else None

        def _fail(r: OpenEOJobRecord) -> OpenEOJobRecord:
            outcome = "failed with a permanent error, not retried" if permanent else "failed"
            return r.model_copy(
                update={
                    "status": OpenEOJobStatus.ERROR,
                    "error_message": _with_attempts(error, r),
                    "updated": utc_now(),
                    "publishing": False,
                    "logs": _append_log(r, f"attempt {r.attempt} of {r.max_attempts} {outcome}: {error}")
                    if r.max_attempts > 1
                    else r.logs,
                }
            )

        store_update_job(job_id, _fail)
        return None

    def _finish(self, record: OpenEOJobRecord, output_path: str | None) -> OpenEOJobRecord:
        """Mark a job FINISHED, or CANCELED if cancellation arrived while its result was saved.

        Runs inside the store mutation, so a cancel request cannot land between this check
        and the write.
        """
        now = utc_now()
        if record.cancel_requested:
            return record.model_copy(update={"status": OpenEOJobStatus.CANCELED, "updated": now})
        usage: dict[str, Any] = {"output_path": output_path} if output_path else {}
        deliveries = (record.usage or {}).get("deliveries")
        if isinstance(deliveries, list) and deliveries:
            # A re-run keeps its delivery links, so an automated delivery is not repeated.
            usage["deliveries"] = deliveries
        finished = record.model_copy(
            update={
                "status": OpenEOJobStatus.FINISHED,
                "updated": now,
                "finished_at": now,
                "usage": usage,
                "publishing": False,
                # The attempt history ends with the outcome that counts, in the same write.
                "logs": _append_log(record, f"attempt {record.attempt} of {record.max_attempts} finished")
                if record.max_attempts > 1
                else record.logs,
            }
        )
        due: dict[str, str] | None = None
        if self._delivery_due is not None:
            try:
                due = self._delivery_due(finished)
            except Exception:
                logger.exception("Delivery lookup failed for openEO job %s; it will not be delivered", record.id)
        return finished.model_copy(update={"delivery_due": due})

    def _notify_finished(self, record: OpenEOJobRecord) -> None:
        listener = self._finished_listener
        if listener is None:
            return
        try:
            listener(record)
        except Exception:
            logger.exception("Finished-job listener failed for openEO job %s", record.id)

    def _persist_result(self, job_id: str, result: Any) -> str | None:
        import xarray as xr

        from open_climate_service.openeo.execution import SaveResultEnvelope

        results_dir = _JOBS_DIR / job_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap format envelope from save_result
        fmt = "ZARR"
        options: dict[str, Any] = {}
        provenance: dict[str, Any] | None = None
        if isinstance(result, SaveResultEnvelope):
            fmt = result.format
            options = result.options
            provenance = result.provenance
            result = result.data

        # Resolve a lazy dask_geopandas GeoDataFrame before any tabular path,
        # including named exports, so a lazy frame never reaches a renderer.
        try:
            import dask_geopandas

            if isinstance(result, dask_geopandas.GeoDataFrame):
                result = result.compute()
        except ImportError:
            pass

        if "export" in options:
            from open_climate_service.exports.service import write_named_export

            return write_named_export(result, results_dir, fmt, options, job_id=job_id, provenance=provenance)

        # Resolve DataArray → Dataset for raster formats
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=result.name or "result")

        if isinstance(result, xr.Dataset):
            # Zarr format with dataset_id → write directly to managed Icechunk/Zarr store
            if fmt == "ZARR" and options.get("dataset_id"):
                _write_managed_zarr(result, options)
                # Managed datasets are not served as job-local files; advertise the
                # managed dataset via a marker that _result_assets expands into
                # /datasets, /zarr (and /stac when published) result links.
                return f"managed://{options['dataset_id']}"
            if fmt in _TABULAR_EXPORT_FORMATS:
                return _write_dataset_tabular_export(result, results_dir, fmt, options)
            return _write_raster(result, results_dir, fmt)

        try:
            import geopandas as gpd

            if isinstance(result, gpd.GeoDataFrame):
                if fmt in _TABULAR_EXPORT_FORMATS:
                    return _write_tabular_export(
                        result.drop(columns="geometry", errors="ignore"),
                        results_dir,
                        fmt,
                        options,
                    )
                return _write_vector(result, results_dir, fmt)
        except ImportError:
            pass

        # Unrecognised result type — raise so the job is marked ERROR rather than
        # silently finishing with an empty assets dict and no indication of failure.
        raise TypeError(
            f"Unsupported result type '{type(result).__name__}': expected xr.Dataset, xr.DataArray, or GeoDataFrame"
        )


def _strip_non_serializable_attrs(ds: Any) -> Any:
    """Return a copy of ds with any non-JSON-serializable attrs removed.

    openeo-processes-dask injects numpy scalars and datetime64 values into
    variable attrs (e.g. reduced_dimensions_min_values) after reduce_dimension.
    Zarr requires all attrs to be JSON-serializable; strip the offenders so the
    write succeeds while keeping the data and coordinates intact.
    """
    import json

    def _safe(attrs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in attrs.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                pass
        return out

    ds = ds.copy()
    ds.attrs = _safe(ds.attrs)
    for name in list(ds.data_vars) + list(ds.coords):
        if ds[name].attrs:
            ds[name].attrs = _safe(ds[name].attrs)
    return ds


def _netcdf_safe_attrs(ds: Any) -> Any:
    """Return a copy of ds keeping only attrs netCDF can encode.

    JSON-serializability is Zarr's contract, not netCDF's, and the two disagree in both
    directions — measured against ``to_netcdf`` rather than inferred:

    | attr value            | JSON | netCDF |
    |-----------------------|------|--------|
    | ``{'t': '2025-01-01'}`` | ok   | fails  |
    | ``[{'a': 1}]``          | ok   | fails  |
    | ``None``                | ok   | fails  |
    | ``True``                | ok   | fails  |
    | ``np.array([1., 2.])``  | fails| ok     |
    | ``np.float32(0.5)``     | fails| ok     |

    So a JSON scrub both misses dict attrs whose contents happen to be JSON-safe and throws
    away arrays and numpy scalars that netCDF writes happily. This keeps str, bytes, numbers
    (excluding bool, which netCDF has no type for), numpy arrays and scalars, and sequences of
    those.
    """

    def _encodable(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (str, bytes, numbers.Number)):
            return True
        import numpy as np

        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (list, tuple)):
            return all(not isinstance(item, bool) and isinstance(item, (str, numbers.Number)) for item in value)
        return False

    def _safe(attrs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in attrs.items() if _encodable(value)}

    ds = ds.copy()
    ds.attrs = _safe(ds.attrs)
    for name in list(ds.data_vars) + list(ds.coords):
        if ds[name].attrs:
            ds[name].attrs = _safe(ds[name].attrs)
    return ds


def _write_managed_zarr(ds: Any, options: dict[str, Any]) -> None:
    """Write a computed xr.Dataset to the managed Icechunk/Zarr store and register it."""
    import uuid
    from datetime import UTC, datetime

    import xarray as xr

    from open_climate_service.data_manager.services import downloader
    from open_climate_service.data_manager.services.utils import get_time_dim, get_x_y_dims
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.ingestions.schemas import (
        ArtifactFormat,
        ArtifactPublication,
        ArtifactRecord,
        ArtifactRequestScope,
    )

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Managed Zarr write requires an xr.Dataset, got {type(ds).__name__}")

    dataset_id = options["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"'dataset_id' option must be a non-empty string, got {type(dataset_id).__name__}")
    if Path(dataset_id).name != dataset_id:
        raise ValueError(
            f"Invalid dataset_id '{dataset_id}': must be a plain name with no path separators or traversal segments"
        )

    from open_climate_service.data_registry.services import datasets as _reg
    from open_climate_service.shared.cf import apply_cf_metadata, cf_attrs_from_template

    # Rename the variable in the store to match the user-specified variable name,
    # so the on-disk name matches what is advertised in the STAC collection.
    if options.get("variable") and len(ds.data_vars) == 1:
        current_name = next(iter(ds.data_vars))
        desired_name = str(options["variable"])
        if current_name != desired_name:
            ds = ds.rename({current_name: desired_name})

    try:
        x_dim, y_dim = get_x_y_dims(ds)
    except ValueError as exc:
        raise ValueError(f"Cannot write managed dataset '{dataset_id}': {exc}") from exc

    try:
        t_dim: str | None = get_time_dim(ds)
    except ValueError:
        t_dim = None

    variable = _derive_variable(ds, options)
    source_template = _resolve_source_template(options)
    _publish_raw = options.get("publish", True)
    if not isinstance(_publish_raw, bool):
        raise ValueError(f"'publish' option must be a boolean, got {type(_publish_raw).__name__!r}: {_publish_raw!r}")
    template = _reg.get_dataset(dataset_id)
    # A template this call registers, so a cancellation before publication can remove it again.
    created_template: Path | None = None
    if template is None:
        # Validate the candidate before it reaches disk. Persisting first left an incompatible
        # template behind when publication then failed, and the corrected retry reloaded that
        # template and failed again — the operator had to delete a YAML to get unstuck.
        candidate = _derive_managed_dataset_template(ds, options, source_template, t_dim)
        _reject_incompatible_template_units(ds, variable, cf_attrs_from_template(candidate))
        # Nothing reaches disk for a job that is already cancelled.
        raise_if_cancelled(force=True)
        try:
            created_template = _reg.write_dataset_template(candidate)
        except FileExistsError:
            pass
        template = _reg.get_dataset(dataset_id)
    if template is None:
        raise ValueError(f"Auto-registered dataset template for '{dataset_id}' could not be reloaded")

    # Prefer an explicit option, then the registered template's display name, and
    # only fall back to the raw id so published collections read as e.g.
    # "Mosquito hotspots (Rwanda 2018 Q1)" rather than "mosquito_hotspots".
    dataset_name: str = options.get("dataset_name") or template.get("name") or dataset_id

    # The cube's own CRS — the store's `proj:code` when it has one, else whatever rioxarray
    # detects from its grid mapping. Deliberately NOT the instance config CRS: falling back to
    # that stamped e.g. EPSG:32633 onto an untagged WGS84 cube, which puts the published store
    # at the projection's origin instead of on the map (CLIM-821).
    from open_climate_service.shared.crs import dataset_crs

    crs: str = dataset_crs(ds)

    # Stamp CF attributes (units / standard_name / cell_methods) from the template so the
    # published store is CF-compliant on disk (#280). The template is authoritative for the
    # fields it declares, so overwrite any placeholder/generic value left on the variable.
    cf_attrs = cf_attrs_from_template(template)
    _reject_incompatible_template_units(ds, variable, cf_attrs)
    apply_cf_metadata(ds, cf_attrs, overwrite=True)

    coverage = _derive_coverage(ds, x_dim, y_dim, t_dim)
    period_type: str | None = _derive_period_type(ds, options, source_template, t_dim)
    store_path = downloader.DOWNLOAD_DIR / f"{dataset_id}.icechunk"
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate the complete record before crossing the store commit's point of no return.
    # Only its measured byte size depends on the committed store and is filled in afterwards.
    record = ArtifactRecord(
        artifact_id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        variable=variable,
        period_type=period_type,
        format=ArtifactFormat.ICECHUNK,
        path=str(store_path),
        asset_paths=[str(store_path)],
        size_bytes=0,
        variables=[str(v) for v in ds.data_vars],
        request_scope=ArtifactRequestScope(
            start=coverage.temporal.start,
            end=coverage.temporal.end,
        ),
        coverage=coverage,
        created_at=datetime.now(UTC),
        publication=ArtifactPublication(),
    )

    # The same writer lock as ingestion and sync. Publishing into a store while one of those
    # writes it would make one of them fail with an Icechunk commit conflict.
    store_lock = ingestion_services._acquire_store_lock(store_path)
    if not store_lock.acquire(blocking=False):
        raise RuntimeError(
            f"Managed dataset '{dataset_id}' is being written by an ingestion, sync, or another job; "
            "run this job again once that finishes"
        )
    try:
        try:
            downloader.write_to_icechunk_store(
                _strip_non_serializable_attrs(ds),
                store_path,
                x_dim,
                y_dim,
                t_dim,
                crs=crs,
                pyramid_method=downloader.resampling_method_from_template(template),
                commit_message=f"Published from openEO job: {dataset_id}",
                # The point of no return: a cancelled job stops here, before the commit, so
                # the store keeps its previous state and nothing is published.
                before_commit=enter_publication,
            )
        except ExecutionCancelled:
            if created_template is not None:
                # Registered by this attempt for a dataset that now will not exist.
                created_template.unlink(missing_ok=True)
                _reg.reset_template_caches()
            raise

        # A derived product is a published dataset and appears in the same lists, so it gets a
        # thumbnail on the same terms. One write, so this is already the once-per-run render the
        # streaming path has to arrange deliberately. Never raises.
        write_dataset_thumbnail(
            store_path,
            {**template, "id": dataset_id, "variable": variable},
        )

        record = record.model_copy(update={"size_bytes": stored_bytes(store_path)})
        ingestion_services.register_artifact_record(record, publish=_publish_raw)
    finally:
        store_lock.release()


def _recover_temporal_from_attrs(ds: Any) -> tuple[str | None, str | None]:
    """Extract temporal extent from reduce_dimension min/max attrs.

    openeo-processes-dask stores the reduced dimension's value range in
    ``reduced_dimensions_min_values`` / ``reduced_dimensions_max_values`` on each
    variable's attrs after ``reduce_dimension``.  Fall back to ``(None, None)`` when not
    found — a non-temporal output (e.g. a day-of-year/month climatology) genuinely has no
    temporal extent, matching the ``None`` convention used elsewhere for coverage.
    """
    import numpy as np

    _TIME_NAMES = ("t", "time", "valid_time")
    sources: list[dict[str, Any]] = [ds.attrs]
    for name in list(ds.data_vars) + list(ds.coords):
        attrs = getattr(ds[name], "attrs", {})
        if attrs:
            sources.append(attrs)
    for attrs in sources:
        min_vals = attrs.get("reduced_dimensions_min_values", {})
        max_vals = attrs.get("reduced_dimensions_max_values", {})
        if not isinstance(min_vals, dict) or not isinstance(max_vals, dict):
            continue
        for tname in _TIME_NAMES:
            if tname in min_vals and tname in max_vals:
                try:
                    t_start = str(np.datetime_as_string(np.datetime64(min_vals[tname]), unit="D"))
                    t_end = str(np.datetime_as_string(np.datetime64(max_vals[tname]), unit="D"))
                    return t_start, t_end
                except Exception:
                    pass
    return None, None


def _resolve_source_template(options: dict[str, Any]) -> dict[str, Any] | None:
    """Return the source dataset template referenced in save_result options, if any."""
    source_dataset_id = options.get("source_dataset_id")
    if not isinstance(source_dataset_id, str) or not source_dataset_id:
        return None
    from open_climate_service.data_registry.services import datasets as _reg

    return _reg.get_dataset(source_dataset_id)


def _derive_period_type(
    ds: Any, options: dict[str, Any], source_template: dict[str, Any] | None, t_dim: str | None
) -> str | None:
    """Return period_type from explicit options, inference, or source template fallback."""
    explicit = options.get("period_type")
    if isinstance(explicit, str) and explicit:
        return explicit
    inferred = _infer_period_type(ds, t_dim) if t_dim is not None else None
    if inferred is not None:
        return inferred
    inherited = source_template.get("period_type") if isinstance(source_template, dict) else None
    if isinstance(inherited, str) and inherited:
        return inherited
    return None


def _derive_managed_dataset_template(
    ds: Any, options: dict[str, Any], source_template: dict[str, Any] | None, t_dim: str | None
) -> dict[str, Any]:
    """Synthesize a static dataset template for a managed openEO publish output."""
    dataset_id = str(options["dataset_id"])
    variable = _derive_variable(ds, options)
    output_kind = _derived_output_kind(dataset_id, variable)
    name = _derive_template_name(dataset_id, options, source_template, output_kind)
    short_name = _derive_template_short_name(name, options, source_template, output_kind)
    period_type = _derive_period_type(ds, options, source_template, t_dim)
    display = _derive_display_config(ds, variable, dataset_id, options, source_template, output_kind)

    template: dict[str, Any] = {
        "id": dataset_id,
        "name": name,
        "short_name": short_name,
        "variable": variable,
        "sync": {"kind": "static"},
        "display": display,
    }
    if period_type is not None:
        template["period_type"] = period_type

    for field in ("units", "resolution", "source", "source_url"):
        explicit = options.get(field)
        if isinstance(explicit, str) and explicit:
            template[field] = explicit
            continue
        # Units the process actually produced beat units inherited from the source dataset.
        # A source template's units describe its *own* variable, so inheriting them is a
        # guess that only holds while the process preserves units — and processes that change
        # them say so on the result. `compute_anomaly(method="relative")` returns percent of
        # normal with `units: "%"`; inheriting `mm/d` from the observed precipitation dataset
        # published percentages as a precipitation depth, and 20 "mm/d" looks entirely
        # plausible, so nothing downstream could catch it.
        if field == "units":
            produced = _variable_units(ds, variable)
            if produced is not None:
                template[field] = produced
                continue
        inherited = source_template.get(field) if isinstance(source_template, dict) else None
        if isinstance(inherited, str) and inherited:
            template[field] = inherited

    return template


_UNKNOWN_UNITS = frozenset({"-", "none", "unknown", "n/a", "na"})
"""Unit strings that assert nothing, so inheriting the source's units over them is an improvement.

`""`, `"1"` and `"unitless"` are deliberately *not* here: they declare a dimensionless quantity,
which is a claim, not a gap. Treating them as gaps let a dimensionless result — an SPI value, a
ratio — inherit `mm/d` from its precipitation source, which is the silent relabelling the unit
checks exist to prevent. `shared/cf.py` already treats `""` as a declared dimensionless unit.
"""


def _variable_units(ds: Any, variable: str) -> str | None:
    """The units the result variable declares, or None when it declares nothing meaningful."""
    try:
        units = ds[variable].attrs.get("units")
    except Exception:
        return None
    if not isinstance(units, str):
        return None
    text = units.strip()
    # A declared dimensionless unit ("" / "1" / "unitless") is returned as-is: it is an assertion
    # about the data, and the caller must not paper over it with the source's units.
    return None if text.lower() in _UNKNOWN_UNITS else text


def _reject_incompatible_template_units(ds: Any, variable: str, cf_attrs: dict[str, str]) -> None:
    """Refuse to relabel a result with template units of a different physical dimension.

    A pre-registered template's units are authoritative over a *placeholder* left on the
    variable — that is what the overwrite at the call site is for. They are not a licence to
    relabel a quantity as something it is not: publishing `compute_anomaly(method="relative")`
    against the shipped `..._anomaly_1991_2020` templates would stamp `mm/d` over the `%`
    earthkit produced, turning 20 percent-of-normal into 20 mm of rain per day. Both values
    are plausible and the template's diverging range covers both, so no later check could
    notice.

    The comparison is on the *parsed unit*, so a tidier spelling of the same unit passes
    (`mm/d` and `mm/day` are one unit to pint) while anything that would change the meaning of
    the numbers does not. Dimensionality alone is too weak a test: `K` and `degC` share a
    dimensionality but differ by 273.15, as do `m` and `mm` by a factor of 1000, so a
    dimensional check would wave through exactly the relabelling this exists to stop.

    An absent or uninformative *declaration* on the template side asserts nothing and so is not
    checked. A `""` on the *produced* side is different: it is a claim of dimensionlessness, so
    publishing it into a template declaring `mm/d` is refused like any other mismatch.

    Overwriting a *placeholder* unit remains the point of the call site — a placeholder is not
    a parseable unit, so `_variable_units` reports it as absent and this never fires. What is
    refused is overwriting a unit the result genuinely carries.

    `units` in the save_result options is **not** an escape hatch from that refusal, and the
    messages below deliberately do not offer it as one (CLIM-918). It is a declaration of what
    the result already is, never a conversion or an override:

    * With a **pre-registered** template, `cf_attrs` comes from the template and the option is
      never read, so passing it re-raises this same error.
    * With an **auto-derived** template, the option becomes the declared unit, so it is checked
      here like any other declaration — it can only cause this refusal, never avoid it. Omitting
      it is what lets the produced unit through.

    The two real recoveries are converting the result in the process graph, or targeting a
    template that declares the units the process produces.
    """
    declared = cf_attrs.get("units")
    produced = _variable_units(ds, variable)
    if not isinstance(declared, str) or not declared.strip() or produced is None:
        return
    declared = declared.strip()
    if declared == produced:
        return
    try:
        from xclim.core.units import units2pint
    except ImportError:
        return  # client-only install; validate_units() makes the same allowance
    try:
        declared_unit = units2pint(declared)
        produced_unit = units2pint(produced)
    except Exception:  # noqa: BLE001 — an unparseable unit is validate_units()' problem, not ours
        return
    if declared_unit == produced_unit:
        return
    if declared_unit.dimensionality != produced_unit.dimensionality:
        raise ValueError(
            f"dataset template declares units '{declared or 'dimensionless'}' but the result carries "
            f"'{produced}', which measures a different quantity "
            f"({declared_unit.dimensionality or 'dimensionless'} vs "
            f"{produced_unit.dimensionality or 'dimensionless'}). Publishing would relabel the values "
            "rather than convert them. Use a template whose units match the process output (a relative "
            "anomaly is a percentage, not the observed variable's unit)."
        )
    raise ValueError(
        f"dataset template declares units '{declared or 'dimensionless'}' but the result carries "
        f"'{produced}'. They measure the same quantity on different scales, so publishing would "
        "relabel the values without converting them. Convert the result in the process graph, or use "
        "a template declaring the units the process produces."
    )


def _derive_template_name(
    dataset_id: str, options: dict[str, Any], source_template: dict[str, Any] | None, output_kind: str | None
) -> str:
    """Return a friendly display name for an auto-derived template."""
    explicit = options.get("dataset_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    inherited = None
    if isinstance(source_template, dict):
        inherited = source_template.get("short_name") or source_template.get("name")
    if isinstance(inherited, str) and inherited.strip() and output_kind is not None:
        return f"{inherited.strip()} {output_kind.lower()}"
    return _humanize_identifier(dataset_id)


def _derive_template_short_name(
    name: str, options: dict[str, Any], source_template: dict[str, Any] | None, output_kind: str | None
) -> str:
    """Return a short_name for an auto-derived template."""
    explicit = options.get("short_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    inherited = source_template.get("short_name") if isinstance(source_template, dict) else None
    if isinstance(inherited, str) and inherited.strip() and output_kind is not None:
        return f"{inherited.strip()} {output_kind.lower()}"
    return name


def _derive_display_config(
    ds: Any,
    variable: str,
    dataset_id: str,
    options: dict[str, Any],
    source_template: dict[str, Any] | None,
    output_kind: str | None,
) -> dict[str, Any]:
    """Return display metadata for an auto-derived template."""
    explicit = _explicit_display_overrides(options)
    source_display = source_template.get("display") if isinstance(source_template, dict) else None
    display: dict[str, Any] = {}

    if isinstance(explicit.get("colormap"), str):
        display["colormap"] = explicit["colormap"]
    if isinstance(explicit.get("range"), list) and len(explicit["range"]) == 2:
        display["range"] = [float(explicit["range"][0]), float(explicit["range"][1])]
    if explicit.get("nodata") is not None:
        display["nodata"] = float(explicit["nodata"])

    signed_output = output_kind in {"Change", "Anomaly", "Difference", "Delta"}
    if signed_output:
        # Diverging either way, but the ends swap by variable: warm is red for temperature,
        # while wet is blue for precipitation. `RdBu` runs low→red/high→blue; `rdbu_r` reverses it.
        variable_da = ds[variable] if variable in getattr(ds, "data_vars", {}) else None
        display.setdefault("colormap", "rdbu_r" if is_temperature_like(variable_da, ds) else "RdBu")
        if "range" not in display:
            data_min, data_max = _data_range(ds, variable)
            bound = max(abs(data_min), abs(data_max))
            if bound == 0:
                bound = 1.0
            display["range"] = [-bound, bound]
    else:
        if isinstance(source_display, dict):
            colormap = source_display.get("colormap")
            value_range = source_display.get("range")
            nodata = source_display.get("nodata")
            if "colormap" not in display and isinstance(colormap, str):
                display["colormap"] = colormap
            if "range" not in display and isinstance(value_range, list) and len(value_range) == 2:
                display["range"] = [float(value_range[0]), float(value_range[1])]
            if "nodata" not in display and nodata is not None:
                display["nodata"] = float(nodata)
        display.setdefault("colormap", "viridis")
        if "range" not in display:
            data_min, data_max = _data_range(ds, variable)
            display["range"] = _normalize_range(data_min, data_max)

    return display


def _explicit_display_overrides(options: dict[str, Any]) -> dict[str, Any]:
    """Extract display overrides from save_result options."""
    display: dict[str, Any] = {}
    nested = options.get("display")
    if isinstance(nested, dict):
        display.update(nested)
    for key in ("colormap", "range", "nodata"):
        if key in options:
            display[key] = options[key]
    return display


def _data_range(ds: Any, variable: str) -> tuple[float, float]:
    """Return finite min/max for one variable, falling back to (0, 1)."""
    import numpy as np

    array = ds[variable].astype("float64")
    min_value = array.min(skipna=True)
    max_value = array.max(skipna=True)
    if hasattr(min_value, "compute"):
        min_value = min_value.compute()
    if hasattr(max_value, "compute"):
        max_value = max_value.compute()
    low = float(np.asarray(min_value.values))
    high = float(np.asarray(max_value.values))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    return low, high


def _normalize_range(data_min: float, data_max: float) -> list[float]:
    """Return a non-degenerate display range."""
    if data_min == data_max:
        if data_min == 0:
            return [0.0, 1.0]
        pad = abs(data_min) * 0.1 or 1.0
        return [data_min - pad, data_max + pad]
    return [data_min, data_max]


def _derived_output_kind(dataset_id: str, variable: str) -> str | None:
    """Classify a derived output from its id/variable name for display/name defaults."""
    haystack = f"{dataset_id} {variable}".lower()
    for token, label in (
        ("anomaly", "Anomaly"),
        ("difference", "Difference"),
        ("change", "Change"),
        ("delta", "Delta"),
    ):
        if token in haystack:
            return label
    return None


def _humanize_identifier(value: str) -> str:
    """Convert an identifier like `worldpop_population_change` into title case."""
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value)).strip().title()


def _derive_coverage(ds: Any, x_dim: str, y_dim: str, t_dim: str | None) -> Any:
    """Derive ArtifactCoverage from an xr.Dataset's coordinates."""
    import numpy as np
    import pyproj

    from open_climate_service.ingestions.schemas import (
        ArtifactCoverage,
        CoverageSpatial,
        CoverageTemporal,
    )

    xmin = float(ds[x_dim].min())
    xmax = float(ds[x_dim].max())
    ymin = float(ds[y_dim].min())
    ymax = float(ds[y_dim].max())
    spatial = CoverageSpatial(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    native_crs: str = ds.attrs.get("proj:code", "EPSG:4326")
    spatial_wgs84: Any = None
    if native_crs not in ("EPSG:4326", "CRS84", "OGC:CRS84"):
        try:
            transformer = pyproj.Transformer.from_crs(native_crs, "EPSG:4326", always_xy=True)
            # transform_bounds is more accurate than transforming individual corners —
            # it densifies the edges, which matters for non-rectilinear projections.
            wx_min, wy_min, wx_max, wy_max = transformer.transform_bounds(xmin, ymin, xmax, ymax)
            spatial_wgs84 = CoverageSpatial(xmin=wx_min, ymin=wy_min, xmax=wx_max, ymax=wy_max)
        except Exception:
            pass

    t_start: str | None
    t_end: str | None
    if t_dim is not None and t_dim in ds.coords and ds.sizes.get(t_dim, 0) > 0:
        # min/max rather than first/last so coverage is correct for a non-monotonic time axis.
        t_values = ds[t_dim].values
        t_start = str(np.datetime_as_string(t_values.min(), unit="D"))
        t_end = str(np.datetime_as_string(t_values.max(), unit="D"))
    else:
        # Dataset has no time dimension (e.g. after reduce_dimension); recover the
        # original temporal range from attrs that openeo-processes-dask attaches.
        t_start, t_end = _recover_temporal_from_attrs(ds)

    return ArtifactCoverage(
        spatial=spatial,
        spatial_wgs84=spatial_wgs84,
        temporal=CoverageTemporal(start=t_start, end=t_end),
    )


def _is_dekadal_axis(t_values: Any) -> bool:
    """Whether every timestamp starts a dekad, at dekadal spacing; see `shared.time.is_dekadal_axis`."""
    from open_climate_service.shared.time import is_dekadal_axis

    return is_dekadal_axis(t_values)


def _infer_period_type(ds: Any, t_dim: str) -> str | None:
    """Infer period type from the median time step of a dataset; see `shared.time.infer_cadence`.

    Kept here by name because the managed-output and export paths call it with a dataset and
    a dimension name; the cadence rule itself lives beside the period reachability table it
    now serves (CLIM-1302).
    """
    from open_climate_service.shared.time import infer_cadence

    if t_dim not in ds.coords or ds.sizes.get(t_dim, 0) < 2:
        return None
    return infer_cadence(ds[t_dim].values)


def _derive_variable(ds: Any, options: dict[str, Any]) -> str:
    """Return the primary variable name from options or the sole data variable."""
    if options.get("variable"):
        name = str(options["variable"])
        if name not in ds.data_vars:
            raise ValueError(
                f"Variable {name!r} specified in options not found in dataset; available: {list(ds.data_vars)!r}"
            )
        return name
    vars_list = list(ds.data_vars)
    if len(vars_list) == 1:
        return str(vars_list[0])
    raise ValueError(f"Dataset has multiple variables {vars_list!r}; specify 'variable' in save_result options")


def _result_assets(record: OpenEOJobRecord) -> dict[str, Any]:
    usage = record.usage or {}
    output_path = usage.get("output_path")
    if not output_path or not isinstance(output_path, str):
        return {}
    if not output_path.startswith("managed://"):
        from open_climate_service.exports.service import read_export_metadata

        metadata = read_export_metadata(Path(output_path))
        if metadata is not None:
            export_assets = {
                "result": {
                    "href": f"/jobs/{record.id}/results/{metadata['filename']}",
                    "type": metadata["media_type"],
                    "title": f"{metadata['format']} export",
                    "roles": ["data"],
                }
            }
            if "manifest" in metadata:
                export_assets["manifest"] = {
                    "href": f"/jobs/{record.id}/results/{metadata['manifest']}",
                    "type": "application/json",
                    "title": "Export manifest",
                    "roles": ["metadata"],
                }
            return export_assets
    if output_path.startswith("managed://"):
        dataset_id = output_path[len("managed://") :]
        assets: dict[str, Any] = {
            "dataset": {
                "href": f"/datasets/{dataset_id}",
                "type": "application/json",
                "title": "Managed dataset",
                "roles": ["metadata"],
            },
            "zarr": {
                "href": f"/zarr/{dataset_id}",
                "type": ZARR_V3_MEDIA_TYPE,
                "title": "Zarr store",
                "roles": ["data"],
                "xarray:open_kwargs": {"consolidated": True},
            },
        }
        # Advertise the STAC collection only when the dataset is actually published.
        # The raster gate rather than the STAC one, even though the asset points at STAC:
        # the record is dereferenced as a Zarr store just below, to keep the media type in
        # step with the collection's own asset.
        try:
            from open_climate_service.ingestions import services as _ingestion_services
            from open_climate_service.ingestions.schemas import ArtifactFormat

            artifact = _ingestion_services.latest_published_raster_artifacts_by_dataset().get(dataset_id)
            if artifact is not None:
                assets["stac"] = {
                    "href": f"/stac/collections/{dataset_id}",
                    "type": "application/json",
                    "title": "STAC collection",
                    "roles": ["metadata"],
                }
                # Keep the claim in step with the STAC collection's zarr asset, so a client
                # sees the same media type and open arguments from either surface. Uncached,
                # unlike the STAC side — a job-result read is rare enough not to warrant one.
                store_path = artifact.path or (artifact.asset_paths[0] if artifact.asset_paths else None)
                if store_path:
                    media_type = zarr_media_type(store_path, icechunk=artifact.format == ArtifactFormat.ICECHUNK)
                    assets["zarr"]["type"] = media_type
                    assets["zarr"]["xarray:open_kwargs"] = {
                        **assets["zarr"]["xarray:open_kwargs"],
                        **data_group_open_kwargs(media_type),
                    }
        except Exception:
            logger.debug("Could not resolve STAC publication for managed dataset '%s'", dataset_id, exc_info=True)
        return assets
    if output_path.endswith(".zarr"):
        return {
            "result": {
                # Trailing slash signals a directory root; Zarr HTTP clients
                # append chunk paths (e.g. .zmetadata, t/0.0) to this href.
                "href": f"/jobs/{record.id}/results/result.zarr/",
                "type": "application/x-zarr",
                "title": "Zarr result store",
                "roles": ["data"],
            }
        }
    if output_path.endswith(".geojson"):
        return {
            "result": {
                "href": f"/jobs/{record.id}/results/result.geojson",
                "type": "application/geo+json",
                "title": "GeoJSON result",
                "roles": ["data"],
            }
        }
    ext_map = {
        ".nc": ("application/netcdf", "NetCDF result"),
        ".tif": ("image/tiff; subtype=geotiff", "GeoTIFF result"),
        ".png": ("image/png", "PNG result"),
        ".csv": ("text/csv", "CSV result"),
        ".json": ("application/json", "JSON result"),
        ".parquet": (PARQUET_MEDIA_TYPE, "GeoParquet result"),
    }
    for ext, (mime, title) in ext_map.items():
        if output_path.endswith(ext):
            fname = output_path.rsplit("/", 1)[-1]
            return {
                "result": {
                    "href": f"/jobs/{record.id}/results/{fname}",
                    "type": mime,
                    "title": title,
                    "roles": ["data"],
                }
            }
    return {}


# ---------------------------------------------------------------------------
# Format writers
# ---------------------------------------------------------------------------

_RASTER_FORMATS: dict[str, tuple[str, str]] = {
    "ZARR": (".zarr", "application/x-zarr"),
    "NETCDF": (".nc", "application/netcdf"),
    "NC": (".nc", "application/netcdf"),
    "NETCDF4": (".nc", "application/netcdf"),
    "GTIFF": (".tif", "image/tiff; subtype=geotiff"),
    "GEOTIFF": (".tif", "image/tiff; subtype=geotiff"),
    "TIFF": (".tif", "image/tiff; subtype=geotiff"),  # common alias
    "TIF": (".tif", "image/tiff; subtype=geotiff"),
    "PNG": (".png", "image/png"),
    "CSV": (".csv", "text/csv"),
}

_VECTOR_FORMATS: dict[str, tuple[str, str]] = {
    "GEOJSON": (".geojson", "application/geo+json"),
    "CSV": (".csv", "text/csv"),
    "PARQUET": (".parquet", PARQUET_MEDIA_TYPE),
}

_TABULAR_EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "CHAPCSV": (".csv", "text/csv"),
    "DHIS2JSON": (".json", "application/json"),
}


def _write_raster(ds: Any, results_dir: Any, fmt: str) -> str | None:
    """Write an xr.Dataset to disk in the requested format. Returns the output path."""
    # A format that carries geometry gets the real shapes written out, rather than a table
    # that has to be joined back to a boundary file. E.g. `aggregate_spatial_weighted`` returns
    # a vector datacube.
    geom_dim = vector_dim(ds)
    if geom_dim is not None:
        # CSV is listed as a vector format but carries no shapes, so it must not demand them: a
        # cube with feature ids and no geometry is still a perfectly good table.
        if fmt in _VECTOR_FORMATS and fmt != "CSV":
            try:
                frame = _vector_frame(ds, geom_dim)
            except Exception as exc:
                # Only the geometry conversion is described this way. A failure writing the file --
                # a full disk, a driver problem -- is a different thing and keeps its own error.
                # Re-raised as ValueError: that is what the sync route turns into a 400, and a
                # cube without shapes is the caller's problem, not the server's.
                raise ValueError(f"Cannot write {fmt}: the vector datacube has no usable geometry ({exc})") from exc
            # Outside the try, so a write failure still cannot fall through to a raster writer: a
            # request for GeoParquet coming back as a Zarr directory is worse than an error.
            return _write_vector(frame, results_dir, fmt)
        # A raster or tabular format was asked for, so honour it. Its shapes are not numbers or
        # strings: Zarr and NetCDF get them encoded as CF geometry, a table goes without them
        # and keeps each feature's id.
        if _RASTER_FORMATS.get(fmt, ("",))[0] in (".zarr", ".nc"):
            ds = encode_vector_cube(ds)
        elif holds_shapes(ds, geom_dim):
            ds = ds.drop_vars(geom_dim)

    if fmt not in _RASTER_FORMATS:
        # Defaulting an unwritable format to Zarr wrote a `result.zarr` directory and called it
        # the requested format. Synchronously that surfaced as a 500 — `IsADirectoryError` when
        # the route read the "file" back — and in a batch job as a job that succeeded while
        # advertising output it had not produced (CLIM-909).
        if fmt in _VECTOR_FORMATS:
            raise ValueError(
                f"Format '{fmt}' describes vector features, but this result is a raster datacube "
                "with no geometry dimension. Aggregate to geometries first (e.g. aggregate_spatial_weighted), "
                "or request a raster format: " + ", ".join(sorted(_RASTER_FORMATS))
            )
        raise ValueError(f"Unsupported output format '{fmt}'. Supported: " + ", ".join(sorted(_RASTER_FORMATS)))

    ext, _ = _RASTER_FORMATS[fmt]

    # `reduce_dimension` (openeo-processes-dask) stamps dict-valued bookkeeping attrs such as
    # `reduced_dimensions_min_values={'t': numpy.datetime64(...)}`, which neither writer can
    # encode. The managed-publish path already scrubbed these; the file export paths did not,
    # so a graph ending in reduce_dimension failed at write time (CLIM-825). Temporal extent is
    # recovered from these attrs earlier, so dropping them here loses nothing the output needed.
    #
    # Each format is filtered against its own contract: JSON for Zarr, netCDF's attr types for
    # netCDF. They disagree in both directions, so using one rule for both would still fail on
    # a JSON-safe dict and would discard arrays netCDF can write.
    if ext == ".zarr":
        path = str(results_dir / "result.zarr")
        _strip_non_serializable_attrs(ds).to_zarr(path, mode="w")
        return path

    if ext == ".nc":
        path = str(results_dir / "result.nc")
        _netcdf_safe_attrs(ds).to_netcdf(path)
        return path

    if ext == ".tif":
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]

        path = str(results_dir / "result.tif")
        # GeoTIFF requires a 2-D or 3-D array; use the first variable
        var = list(ds.data_vars)[0]
        da = ds[var]
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        da.rio.to_raster(path)
        return path

    if ext == ".png":
        return _write_png(ds, results_dir)

    if ext == ".csv":
        path = str(results_dir / "result.csv")
        df = ds.to_dataframe().reset_index()
        # Drop internal Zarr artefacts (spatial_ref, index) that add noise for consumers, and a
        # dimension without labels: a vector cube's once its shapes are dropped, which pandas
        # would otherwise write as row numbers beside the feature ids.
        unlabelled = {str(dim) for dim in ds.dims if dim not in ds.coords}
        drop = [c for c in df.columns if c in ("spatial_ref", "index") or c in unlabelled or c.startswith("level_")]
        df.drop(columns=drop, errors="ignore").to_csv(path, index=False)
        return path

    # Unknown format — raise so the caller can surface a clear 400/500 rather than
    # silently writing a .zarr directory that read_bytes() would crash on.
    known = ", ".join(sorted(_RASTER_FORMATS))
    raise ValueError(f"Unsupported raster format '{fmt}'. Known formats: {known}")


def _vector_crs(ds: Any, geom_dim: str) -> Any:
    """Return the CRS of the cube's geometries.

    Uses the CRS declared on the geometry coordinate's GeometryIndex, which the
    aggregations set; a cube without one is taken as WGS 84, as GeoJSON is.
    """
    index = getattr(ds, "xindexes", {}).get(geom_dim)
    crs = getattr(index, "crs", None)
    return crs if crs is not None else "EPSG:4326"


def _vector_frame(ds: Any, geom_dim: str) -> Any:
    """Build a GeoDataFrame from a vector datacube, preserving feature IDs.

    The shapes come from the geometry dimension, as Shapely geometries or WKT; each
    feature's id stays a column. Repeated geometries are parsed once and reused
    across rows. Raises if any row has no geometry.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt as shapely_wkt

    crs = _vector_crs(ds, geom_dim)
    frame = ds.to_dataframe().reset_index()

    def _as_geometry(value: Any) -> Any:
        if hasattr(value, "geom_type"):
            return value
        return shapely_wkt.loads(str(value))

    source = geom_dim
    # A flattened vector cube has one row per (feature, timestep), so the same handful of polygons
    # repeat once per step: a daily year over 500 districts is 182,500 rows carrying 500 distinct
    # shapes. Parse each distinct value once and fan it back out, rather than paying WKT parsing per
    # row — for large boundaries that is the dominant cost of writing the file.
    codes, uniques = pd.factorize(frame[source])
    # factorize codes a null as -1, and `parsed[-1]` is the last polygon, not a missing one: a
    # feature without geometry would silently be written with its neighbour's shape.
    if (codes < 0).any():
        raise ValueError(f"{int((codes < 0).sum())} rows have no geometry in '{source}'")
    parsed = [_as_geometry(value) for value in uniques]
    geoms = [parsed[code] for code in codes]
    # `feature_id` is an ordinary column here, the id every consumer joins on.
    attributes = frame.drop(columns=[geom_dim])
    return gpd.GeoDataFrame(attributes, geometry=geoms, crs=crs)


def _as_wgs84(gdf: Any) -> Any:
    """The frame reprojected to WGS 84, for GeoJSON only.

    RFC 7946 fixes GeoJSON coordinates to WGS 84, and the format carries no CRS of its own to
    say otherwise, so a projected frame written straight out reads as degrees and lands off the
    coast of Africa. GeoParquet is the opposite case -- it records the CRS in its metadata, so a
    projected cube keeps its native coordinates there and loses no precision to a round trip.
    """
    crs = getattr(gdf, "crs", None)
    if crs is None or crs.to_epsg() == 4326:
        return gdf
    return gdf.to_crs("EPSG:4326")


def _write_vector(gdf: Any, results_dir: Any, fmt: str) -> str | None:
    """Write a GeoDataFrame to disk in the requested format. Returns the output path."""
    ext, _ = _VECTOR_FORMATS.get(fmt, (".geojson", "application/geo+json"))

    if ext == ".geojson":
        path = str(results_dir / "result.geojson")
        _as_wgs84(gdf).to_file(path, driver="GeoJSON")
        return path

    if ext == ".parquet":
        path = str(results_dir / "result.parquet")
        gdf.to_parquet(path)
        return path

    if ext == ".csv":
        path = str(results_dir / "result.csv")
        # CSV drops the shapes; each feature's id stays as `feature_id`.
        gdf.drop(columns="geometry", errors="ignore").to_csv(path, index=False)
        return path

    # Fallback to GeoJSON
    path = str(results_dir / "result.geojson")
    _as_wgs84(gdf).to_file(path, driver="GeoJSON")
    return path


def check_ad_hoc_period_reachability(ds: Any, options: dict[str, Any]) -> None:
    """Refuse a `period_type` the result's own cadence cannot honestly be labelled with (CLIM-1139).

    A coarser label would put several timestamps on one DHIS2 key, which DHIS2 resolves by
    keeping whichever arrives last; a finer one cannot be derived at all. Both are named,
    with the step to add for the first. The cadence the result carries, stamped at load and
    rewritten by each temporal aggregation, is authoritative; the axis spacing is inferred
    only for a result that carries none, and a single timestamp is accepted as is.
    """
    from open_climate_service.shared.time import (
        Reachability,
        cadence_to_openeo_period,
        normalise_export_period,
        period_reachability,
    )

    period_field = _optional_str_option(options, "period_field") or "t"
    destination = normalise_export_period(_optional_str_option(options, "period_type"))
    if destination is None or period_field not in getattr(ds, "coords", {}):
        return
    from open_climate_service.shared.time import cadence_of

    source = cadence_of(ds) or _infer_period_type(ds, period_field)
    if source is None:
        return
    outcome, reason = period_reachability(source, destination)
    if outcome is Reachability.PASS_THROUGH:
        return
    if outcome is Reachability.AGGREGATE:
        period = cadence_to_openeo_period(destination)
        step = f"aggregate_temporal_period(period='{period}')" if period else "a temporal aggregation"
        raise ValueError(
            f"period_type '{destination}' is coarser than the result's {source} spacing, so each "
            f"{destination} period would receive several values. Add {step} with the reducer you mean "
            f"before save_result, or set period_type to '{source}'"
        )
    raise ValueError(f"period_type '{destination}' cannot be derived from {source} data: {reason}")


def _write_dataset_tabular_export(ds: Any, results_dir: Any, fmt: str, options: dict[str, Any]) -> str | None:
    import pandas as pd

    check_ad_hoc_period_reachability(ds, options)
    inferred_options = dict(options)
    period_field = _optional_str_option(inferred_options, "period_field") or "t"
    period_type = _optional_str_option(inferred_options, "period_type")
    if period_type is None and period_field in getattr(ds, "coords", {}):
        inferred = _infer_period_type(ds, period_field)
        if inferred in {"daily", "weekly", "monthly", "quarterly", "yearly"}:
            inferred_options["period_type"] = inferred
    if hasattr(ds, "to_dataframe"):
        df = ds.to_dataframe().reset_index()
    elif isinstance(ds, pd.DataFrame):
        df = ds
    else:
        raise TypeError(f"Unsupported data type for tabular export: {type(ds).__name__}")
    return _write_tabular_export(df, results_dir, fmt, inferred_options)


def _write_tabular_export(df: Any, results_dir: Any, fmt: str, options: dict[str, Any]) -> str | None:
    if fmt == "CHAPCSV":
        return _write_chap_csv(df, results_dir, options)
    if fmt == "DHIS2JSON":
        return _write_dhis2_json(df, results_dir, options)
    known = ", ".join(sorted(_TABULAR_EXPORT_FORMATS))
    raise ValueError(f"Unsupported tabular export format '{fmt}'. Known formats: {known}")


def _write_chap_csv(df: Any, results_dir: Any, options: dict[str, Any]) -> str:
    frame = _build_chap_csv_frame(df, options)
    path = str(results_dir / "result.csv")
    frame.to_csv(path, index=False)
    return path


def _build_chap_csv_frame(df: Any, options: dict[str, Any]) -> Any:
    import pandas as pd

    period_field = _optional_str_option(options, "period_field") or "t"
    location_field = _optional_str_option(options, "location_field") or "geometry"
    period_type = _optional_str_option(options, "period_type")
    cube_labels_raw = options.get("cube_labels")

    frame = pd.DataFrame(df).copy()
    location_field = feature_id_field(frame, location_field)
    if location_field not in frame.columns:
        if location_field == "geometry":
            raise ValueError(
                "Missing location field 'geometry' in aggregated result; "
                "for GeoDataFrame inputs set save_result option 'location_field' explicitly"
            )
        raise ValueError(f"Missing location field '{location_field}' in aggregated result")
    if period_field not in frame.columns:
        raise ValueError(f"Missing period field '{period_field}' in aggregated result")

    # merge_cubes produces a single value column plus a synthetic "__cubes__"
    # label dimension. Pivot that long form to one CHAP value column per cube.
    if "__cubes__" in frame.columns:
        cube_field = "__cubes__"
        non_value_fields = {location_field, period_field, cube_field, *_non_value_fields(frame)}
        candidate_value_fields = [
            str(c) for c in frame.columns if c not in non_value_fields and not str(c).startswith("level_")
        ]
        if len(candidate_value_fields) != 1:
            raise ValueError(
                "CHAPCSV export with merged cubes requires exactly one value column before pivoting; "
                f"found {candidate_value_fields}"
            )
        value_field = candidate_value_fields[0]
        frame = (
            frame[[period_field, location_field, cube_field, value_field]]
            .pivot(index=[period_field, location_field], columns=cube_field, values=value_field)
            .reset_index()
        )
        frame.columns.name = None
        if cube_labels_raw is not None:
            if not isinstance(cube_labels_raw, dict):
                raise ValueError("CHAPCSV option 'cube_labels' must be an object mapping cube ids to output columns")
            rename_map: dict[str, str] = {}
            for raw_key, label_value in cube_labels_raw.items():
                key = str(raw_key).strip()
                value = str(label_value).strip()
                if not key or not value:
                    raise ValueError("CHAPCSV option 'cube_labels' must map non-empty cube ids to non-empty labels")
                rename_map[key] = value
            frame = frame.rename(columns=rename_map)

    value_fields = _select_chap_value_fields(frame, location_field, period_field)
    rows: list[dict[str, str]] = []
    for row_index, record in enumerate(frame.to_dict(orient="records")):
        period_value = record.get(period_field)
        if _is_nullish(period_value):
            raise ValueError(f"Null period value in field '{period_field}' at row {row_index}")
        location = record.get(location_field)
        if _is_nullish(location):
            raise ValueError(f"Null location value in field '{location_field}' at row {row_index}")

        row: dict[str, str] = {
            "time_period": _to_dhis2_period_string(period_value, period_type),
            "location": str(location),
        }
        for value_field in value_fields:
            raw_value: Any | None = record.get(value_field)
            row[value_field] = "" if _is_nullish(raw_value) else _to_dhis2_value_string(raw_value)
        rows.append(row)

    return pd.DataFrame(rows, columns=["time_period", "location", *value_fields])


def _select_chap_value_fields(frame: Any, location_field: str, period_field: str) -> list[str]:
    excluded = {location_field, period_field, "__cubes__", *_non_value_fields(frame)}
    candidates = [str(c) for c in frame.columns if c not in excluded and not str(c).startswith("level_")]
    if not candidates:
        raise ValueError("CHAPCSV export requires at least one value column")
    return candidates


def _write_dhis2_json(df: Any, results_dir: Any, options: dict[str, Any]) -> str:
    payload = _build_dhis2_json_payload(df, options)
    path = str(results_dir / "result.json")
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_png(ds: Any, results_dir: Any) -> str | None:
    """Render an xr.Dataset as a styled PNG using the collection's render settings.

    Applies the same colormap, rescale range and NaN transparency as the /map viewer,
    through the shared renderer in ``shared/thumbnails.py``. Squeezes to a 2-D slice by
    taking the first step of each leading dimension: this is a *result* the caller asked
    for, so it shows the front of the cube they computed, where a catalogue thumbnail of a
    published store instead shows the step nearest today.
    """
    from open_climate_service.shared.thumbnails import render_png

    var = list(ds.data_vars)[0]
    arr = ds[var]
    while arr.ndim > 2:
        arr = arr.isel({arr.dims[0]: 0})

    # Render settings from the published collection, via the dataset registry.
    colormap: str | None = None
    clim: tuple[float, float] | None = None
    try:
        from open_climate_service.data_registry.services import datasets as reg

        for _ds_meta in reg.list_datasets():
            display = _ds_meta.get("display", {})
            ds_var = _ds_meta.get("variable", "")
            if ds_var == var or _ds_meta.get("id", "").endswith(var):
                colormap = display.get("colormap", colormap)
                rng = display.get("range")
                if isinstance(rng, list) and len(rng) == 2:
                    clim = (float(rng[0]), float(rng[1]))
                break
    except Exception:
        pass

    return str(render_png(arr, results_dir / "result.png", colormap=colormap, clim=clim))


def _derive_job_title(process: dict[str, Any]) -> str | None:
    """Generate a human-readable job title from a process graph when none is provided.

    Looks for a load_collection node and uses the collection id (plus temporal
    extent if present) to build a short label, e.g. "chirps3_precipitation_daily
    2023-01-01–2023-12-31". Returns None if no load_collection is found.
    """
    graph = process.get("process_graph")
    if not isinstance(graph, dict):
        return None
    for node in graph.values():
        if not isinstance(node, dict) or node.get("process_id") != "load_collection":
            continue
        args = node.get("arguments", {})
        collection_id = args.get("id")
        if not isinstance(collection_id, str):
            continue
        temporal = args.get("temporal_extent")
        if isinstance(temporal, (list, tuple)) and len(temporal) == 2:
            start, end = temporal[0], temporal[1]
            if start and end:
                return f"{collection_id} {start}–{end}"
            if start:
                return f"{collection_id} from {start}"
            if end:
                return f"{collection_id} until {end}"
        return collection_id
    return None


_service: OpenEOJobService | None = None


def get_openeo_job_service() -> OpenEOJobService:
    """Return the singleton openEO job service."""
    global _service
    if _service is None:
        _service = OpenEOJobService()
    return _service


def reset_openeo_job_service() -> None:
    """Reset singleton for tests."""
    global _service
    if _service is not None:
        _service.shutdown()
    _service = None
