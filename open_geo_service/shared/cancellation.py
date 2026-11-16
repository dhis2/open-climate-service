"""Cooperative cancellation for computation running inside a job (CLIM-1221).

A running openEO job used to notice a cancellation only after its whole process graph had run,
by which point it could already have published a managed dataset. Cancellation is now checked
throughout execution, at two granularities, and once more at the point of no return:

* **Before every process** in a graph, through the process registry.
* **Before every dask task**, through a dask callback. A long `.compute()` is a stream of small
  tasks, so a cancelled job stops within about a second of its next task rather than after the
  whole computation. Raising from the callback aborts the computation in the calling thread.
* **At publication.** :func:`enter_publication` is called just before an irreversible side
  effect, such as committing a managed store. It is the job's to define atomically: either the
  cancellation is honoured and nothing is published, or publication starts and the job can no
  longer be cancelled.

Every check reads state through a callable the job supplies, and is a no-op outside a
:func:`cancellation_scope`, so synchronous requests and code outside jobs are unaffected.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from dask.callbacks import Callback

CHECK_INTERVAL_SECONDS = 1.0
"""How long a cached answer to "is this job cancelled?" is trusted.

The answer comes from the job store, so it is not read on every dask task. This bounds how long
a running job keeps computing after cancellation, given tasks shorter than the interval.
"""


class ExecutionCancelled(Exception):  # noqa: N818 -- reads as a state, like CancelledError
    """Raised inside a cancellation scope once its job has been cancelled."""


@dataclass
class _Scope:
    is_cancelled: Callable[[], bool]
    enter_publication: Callable[[], None] | None
    interval: float
    last_checked: float = float("-inf")
    cancelled: bool = False


_current: ContextVar[_Scope | None] = ContextVar("ocs_cancellation_scope", default=None)


@contextmanager
def cancellation_scope(
    is_cancelled: Callable[[], bool],
    *,
    enter_publication: Callable[[], None] | None = None,
    interval: float = CHECK_INTERVAL_SECONDS,
) -> Generator[None]:
    """Make cancellation checks in this context consult ``is_cancelled``.

    ``enter_publication`` is called by :func:`enter_publication`; it must raise
    :class:`ExecutionCancelled` if the job is cancelled, and otherwise make the job
    uncancellable, atomically with that check.
    """
    install_dask_cancellation()
    token = _current.set(_Scope(is_cancelled, enter_publication, interval))
    try:
        yield
    finally:
        _current.reset(token)


def raise_if_cancelled(*, force: bool = False) -> None:
    """Raise :class:`ExecutionCancelled` if the current job has been cancelled.

    Throttled to one store read per interval; ``force`` reads now, for a check that guards a
    side effect. A no-op outside a cancellation scope.
    """
    scope = _current.get()
    if scope is None:
        return
    if not scope.cancelled:
        now = time.monotonic()
        if not force and now - scope.last_checked < scope.interval:
            return
        scope.last_checked = now
        scope.cancelled = bool(scope.is_cancelled())
    if scope.cancelled:
        raise ExecutionCancelled("The job was cancelled")


def enter_publication() -> None:
    """Pass the point of no return, or raise :class:`ExecutionCancelled`.

    Call immediately before an irreversible side effect. Outside a scope, or in one that
    defines no gate, it only performs a forced cancellation check.
    """
    raise_if_cancelled(force=True)
    scope = _current.get()
    if scope is not None and scope.enter_publication is not None:
        scope.enter_publication()


class _DaskCancellation(Callback):
    """Check for cancellation before each dask task; runs in the computing thread."""

    def _pretask(self, key: object, dsk: object, state: object) -> None:
        raise_if_cancelled()


_dask_callback = _DaskCancellation()
_install_lock = threading.Lock()


def install_dask_cancellation() -> None:
    """Make sure the dask cancellation callback is active in this process. Idempotent.

    Checks dask's own set of active callbacks rather than remembering that it registered
    once: anything that resets that process-wide set would otherwise silently turn
    cancellation of long computations off.
    """
    with _install_lock:
        if _dask_callback._callback not in Callback.active:
            _dask_callback.register()
