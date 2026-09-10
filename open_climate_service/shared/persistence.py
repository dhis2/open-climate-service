"""Atomic JSON replacement protected by a stable sibling lock file."""

import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
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
