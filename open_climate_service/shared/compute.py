"""Process-wide limits on background computation.

Two limits, because two things multiplied without bound (CLIM-1229):

* **One dask thread pool for the whole process.** Dask's threaded scheduler gives every
  calling thread other than the main one a pool of its own, sized to the core count. Each
  job thread and each request thread that computed therefore started a full pool, so one
  ingest and two openEO jobs on a 12-core server ran well over a hundred compute threads,
  and the storage layer's I/O threads grew with them.
* **A shared number of job slots** across native jobs (ingestion, sync, feature refresh)
  and openEO jobs, which otherwise each ran up to four at a time from separate pools.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import dask.config
from dask.threaded import ContextAwareThreadPoolExecutor

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS_ENV = "CLIMATE_SERVICE_MAX_CONCURRENT_JOBS"
DEFAULT_MAX_CONCURRENT_JOBS = 3
"""One ingest and two openEO jobs, the load CLIM-1229 was measured under, all make progress."""

_SLOT_POLL_SECONDS = 2.0


def dask_thread_budget() -> int:
    """Threads for all dask computation in this process.

    `DASK_NUM_WORKERS` (dask's own `num_workers` setting) wins when set. Otherwise one less
    than the core count, so the web server keeps a core while jobs compute.
    """
    configured = dask.config.get("num_workers", None)
    if configured:
        return max(1, int(configured))
    return max(1, (os.cpu_count() or 1) - 1)


_in_shared_pool = threading.local()


def _mark_pool_thread() -> None:
    _in_shared_pool.active = True


class SharedDaskPool(ContextAwareThreadPoolExecutor):
    """The one dask pool; a computation started inside one of its tasks runs inline.

    Every computation shares these threads, so a task that itself calls `.compute()`
    would wait for a free thread of the pool it is occupying, and with every thread doing
    the same, none would come. Running such a nested computation on the task's own thread
    keeps it within the budget and cannot deadlock.
    """

    def __init__(self, max_workers: int) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="dask-shared", initializer=_mark_pool_thread)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any]
        if not getattr(_in_shared_pool, "active", False):
            future = super().submit(fn, *args, **kwargs)
            return future
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


_pool: SharedDaskPool | None = None
_pool_mutex = threading.Lock()


def install_shared_dask_pool() -> SharedDaskPool:
    """Route every threaded dask computation in this process through one bounded pool.

    Idempotent. The pool lives for the process, like dask's own default pool.
    """
    global _pool
    with _pool_mutex:
        if _pool is None:
            _pool = SharedDaskPool(dask_thread_budget())
            dask.config.set(pool=_pool)
            logger.info("Dask computations share %d threads", _pool._max_workers)
        return _pool


def max_concurrent_jobs() -> int:
    """How many native and openEO jobs may run at once, from the environment."""
    raw = os.environ.get(MAX_CONCURRENT_JOBS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{MAX_CONCURRENT_JOBS_ENV} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{MAX_CONCURRENT_JOBS_ENV} must be a positive integer, got {raw!r}")
    return value


class JobSlots:
    """A counted limit on running jobs, shared by every job runtime in the process."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._semaphore = threading.BoundedSemaphore(size)

    def acquire(self, should_stop: Callable[[], bool], on_wait: Callable[[], None] | None = None) -> bool:
        """Take a slot, waiting as long as needed; False if `should_stop` turned true first.

        Waits in short intervals so a cancelled job or a stopping service gives up its
        place instead of holding a pool thread forever.
        """
        if self._semaphore.acquire(blocking=False):
            return True
        if on_wait is not None:
            on_wait()
        while not should_stop():
            if self._semaphore.acquire(timeout=_SLOT_POLL_SECONDS):
                return True
        return False

    def release(self) -> None:
        self._semaphore.release()


_slots: JobSlots | None = None
_slots_mutex = threading.Lock()


def get_job_slots() -> JobSlots:
    """The process's job slots, sized from `CLIMATE_SERVICE_MAX_CONCURRENT_JOBS`."""
    global _slots
    with _slots_mutex:
        if _slots is None:
            _slots = JobSlots(max_concurrent_jobs())
        return _slots


def reset_job_slots() -> None:
    """Forget the slots so the next job reads the limit again; for tests."""
    global _slots
    with _slots_mutex:
        _slots = None
