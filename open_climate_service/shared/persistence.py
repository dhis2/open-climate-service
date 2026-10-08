"""Atomic JSON replacement protected by a stable sibling lock file."""

import json
import os
import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import portalocker


@contextmanager
def index_lock(path: Path) -> Generator[None]:
    """Lock before opening the index; replacement must not replace the lock inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "a", encoding="utf-8") as handle:
        portalocker.lock(handle, portalocker.LOCK_EX)
        try:
            yield
        finally:
            portalocker.unlock(handle)


class AlreadyLocked(Exception):
    """Raised by :func:`try_index_lock` when the lock is already held."""


@contextmanager
def try_index_lock(path: Path) -> Generator[None]:
    """Acquire the index lock without blocking; raise ``AlreadyLocked`` if held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(
        path.with_suffix(path.suffix + ".lock"),
        timeout=0,
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    )
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as exc:
        raise AlreadyLocked(str(path)) from exc
    try:
        yield
    finally:
        lock.release()


@contextmanager
def execution_lease(path: Path) -> Generator[bool]:
    """Hold an exclusive execution lease for the block; yield whether it was won.

    A job may execute in at most one process at a time. The lease is a file lock, so the
    operating system releases it when its process exits, and a crash never leaves a job
    leased. A lost lease is reported by yielding False rather than raising, so the caller
    decides what not running the job means.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(try_index_lock(path))
        except AlreadyLocked:
            yield False
            return
        yield True


def atomic_json(path: Path, value: Any) -> None:
    """Flush contents and directory entries before returning to the caller."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".index-", mode="w", encoding="utf-8", delete=False
        ) as handle:
            temporary = handle.name
            # Keep parity with the previous store: a completed job whose result
            # carries NaN must still persist rather than failing the write.
            json.dump(value, handle, allow_nan=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
