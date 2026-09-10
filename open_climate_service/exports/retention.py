"""Cross-process leases protecting job results during consumption."""

import re
from collections.abc import Generator
from contextlib import contextmanager

import portalocker
from fastapi import HTTPException


@contextmanager
def result_lease(job_id: str) -> Generator[None]:
    """Exclude deletion, updates, reruns, and other consumers until release.

    The lock file lives outside the job directory so deletion cannot unlink the
    lock inode while another process holds it. Process exit releases the OS lock.
    """
    from open_climate_service.openeo.jobs import _JOBS_DIR

    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise HTTPException(status_code=400, detail="Invalid source job ID")
    directory = _JOBS_DIR / ".export-locks"
    directory.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(directory / f"{job_id}.lock", timeout=0, flags=portalocker.LOCK_EX | portalocker.LOCK_NB)
    try:
        lock.acquire()
    except portalocker.exceptions.LockException:
        raise HTTPException(status_code=409, detail="Source job results are currently in use") from None
    try:
        yield
    finally:
        lock.release()
