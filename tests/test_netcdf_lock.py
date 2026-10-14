import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.shared import netcdf
from open_climate_service.shared.netcdf import NETCDF_LOCK, load_netcdf, load_netcdfs, open_netcdfs, write_netcdf


def _held_by_this_thread() -> bool:
    """Whether this thread holds the re-entrant lock (an RLock has no `locked()` before 3.14)."""
    return NETCDF_LOCK._is_owned()  # type: ignore[attr-defined]


def _held_by_any_thread() -> bool:
    """Whether some thread holds the lock: another thread cannot take it without waiting."""
    taken: list[bool] = []

    def probe() -> None:
        acquired = NETCDF_LOCK.acquire(blocking=False)
        taken.append(acquired)
        if acquired:
            NETCDF_LOCK.release()  # by the thread that took it: an RLock is released by its owner

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not taken[0]


def _write(path: Path, day: str) -> Path:
    values = np.ones((1, 3), dtype="float32")
    coords = {"t": [np.datetime64(day)], "y": [0.0, 1.0, 2.0]}
    xr.Dataset({"precip": (("t", "y"), values)}, coords=coords).to_netcdf(path)
    return path


def test_load_netcdf_reads_into_memory_and_closes(tmp_path: Path) -> None:
    ds = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))
    assert float(ds["precip"].sum()) == 3.0
    (tmp_path / "a.nc").unlink()  # nothing is left open
    assert not _held_by_any_thread()


def test_load_netcdfs_combines_files(tmp_path: Path) -> None:
    files = [_write(tmp_path / f"{d}.nc", f"2024-01-0{d}") for d in (1, 2)]
    ds = load_netcdfs(files, combine="by_coords")
    assert ds.sizes["t"] == 2


def test_reads_hold_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "a.nc", "2024-01-01")
    real_open = xr.open_dataset

    def checked_open(*args: Any, **kwargs: Any) -> xr.Dataset:
        assert _held_by_this_thread(), "netCDF opened without the lock"
        return real_open(*args, **kwargs)

    monkeypatch.setattr(netcdf.xr, "open_dataset", checked_open)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: load_netcdf(path), range(8)))
    assert all(float(ds["precip"].sum()) == 3.0 for ds in results)


def test_a_write_holds_the_lock_and_the_file_reads_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))
    real_to_netcdf = xr.Dataset.to_netcdf

    def checked(self: xr.Dataset, *args: Any, **kwargs: Any) -> Any:
        assert _held_by_this_thread(), "netCDF written without the lock"
        return real_to_netcdf(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", checked)
    write_netcdf(source, tmp_path / "out.nc")

    assert float(load_netcdf(tmp_path / "out.nc")["precip"].sum()) == 3.0


def test_writing_a_lazily_opened_netcdf_does_not_deadlock(tmp_path: Path) -> None:
    """The write holds the lock while the lazy source reads take it again: re-entrant, same thread."""
    files = [_write(tmp_path / f"{d}.nc", f"2024-01-0{d}") for d in (1, 2)]
    lazy = open_netcdfs(files, combine="by_coords")
    done = threading.Event()

    def write() -> None:
        write_netcdf(lazy, tmp_path / "out.nc")
        done.set()

    worker = threading.Thread(target=write, daemon=True)
    worker.start()
    worker.join(timeout=30)

    assert done.is_set(), "writing a lazily opened netCDF dataset deadlocked on NETCDF_LOCK"
    assert load_netcdf(tmp_path / "out.nc").sizes["t"] == 2


def _waits_for_the_lock(action: Any) -> bool:
    """Whether *action*, run in a thread, waits while another thread holds the lock."""
    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with NETCDF_LOCK:
            holding.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(timeout=10)
    worker = threading.Thread(target=action)
    worker.start()
    worker.join(timeout=0.5)
    waited = worker.is_alive()
    release.set()
    holder.join()
    worker.join(timeout=10)
    return waited


def test_opening_and_closing_a_lazy_dataset_take_the_lock(tmp_path: Path) -> None:
    """Opening reads every file's metadata and closing releases HDF5 handles: both under the lock."""
    files = [_write(tmp_path / "a.nc", "2024-01-01")]
    opened: list[xr.Dataset] = []

    assert _waits_for_the_lock(lambda: opened.append(open_netcdfs(files, combine="by_coords")))
    assert _waits_for_the_lock(opened[0].close)


def test_lazy_reads_take_the_shared_lock(tmp_path: Path) -> None:
    """A lazily opened file's reads wait while another thread holds the lock for a whole read."""
    lazy = open_netcdfs([_write(tmp_path / "a.nc", "2024-01-01")], combine="by_coords")
    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with NETCDF_LOCK:
            holding.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(timeout=10)
    reader = threading.Thread(target=lambda: lazy["precip"].values)
    reader.start()
    reader.join(timeout=0.5)
    blocked = reader.is_alive()
    release.set()
    holder.join()
    reader.join(timeout=10)

    assert blocked, "a lazy read ran while another thread held NETCDF_LOCK"


def test_the_download_endpoint_writes_under_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.data_accessor.services.accessor import xarray_to_temporary_netcdf

    source = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))  # written before the check is in place
    seen: list[bool] = []
    real_to_netcdf = xr.Dataset.to_netcdf

    def checked(self: xr.Dataset, *args: Any, **kwargs: Any) -> Any:
        seen.append(_held_by_this_thread())
        return real_to_netcdf(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", checked)
    path = xarray_to_temporary_netcdf(source)
    Path(path).unlink()

    assert seen == [True]
