import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from open_climate_service.shared import netcdf
from open_climate_service.shared.netcdf import NETCDF_LOCK, load_netcdf, load_netcdfs, write_netcdf


def _write(path: Path, day: str) -> Path:
    values = np.ones((1, 3), dtype="float32")
    coords = {"t": [np.datetime64(day)], "y": [0.0, 1.0, 2.0]}
    xr.Dataset({"precip": (("t", "y"), values)}, coords=coords).to_netcdf(path)
    return path


def test_load_netcdf_reads_into_memory_and_closes(tmp_path: Path) -> None:
    ds = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))
    assert float(ds["precip"].sum()) == 3.0
    (tmp_path / "a.nc").unlink()  # nothing is left open
    assert not NETCDF_LOCK.locked()


def test_load_netcdfs_combines_files(tmp_path: Path) -> None:
    files = [_write(tmp_path / f"{d}.nc", f"2024-01-0{d}") for d in (1, 2)]
    ds = load_netcdfs(files, combine="by_coords")
    assert ds.sizes["t"] == 2


def test_reads_from_threads_run_one_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "a.nc", "2024-01-01")
    real_open = xr.open_dataset
    active: list[int] = []
    counter = threading.Lock()

    def checked_open(*args: Any, **kwargs: Any) -> xr.Dataset:
        with counter:
            active.append(1)
            assert len(active) == 1, "two netCDF reads ran at once"
        time.sleep(0.01)  # long enough for another thread to arrive
        with counter:
            active.pop()
        return real_open(*args, **kwargs)

    monkeypatch.setattr(netcdf.xr, "open_dataset", checked_open)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: load_netcdf(path), range(8)))
    assert all(float(ds["precip"].sum()) == 3.0 for ds in results)


def test_a_write_holds_the_lock_and_the_file_reads_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))
    real_to_netcdf = xr.Dataset.to_netcdf

    def checked(self: xr.Dataset, *args: Any, **kwargs: Any) -> Any:
        assert NETCDF_LOCK.locked(), "netCDF written without the lock"
        return real_to_netcdf(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", checked)
    write_netcdf(source, tmp_path / "out.nc")

    assert float(load_netcdf(tmp_path / "out.nc")["precip"].sum()) == 3.0


def test_the_download_endpoint_writes_under_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.data_accessor.services.accessor import xarray_to_temporary_netcdf

    source = load_netcdf(_write(tmp_path / "a.nc", "2024-01-01"))  # written before the check is in place
    seen: list[bool] = []
    real_to_netcdf = xr.Dataset.to_netcdf

    def checked(self: xr.Dataset, *args: Any, **kwargs: Any) -> Any:
        seen.append(NETCDF_LOCK.locked())
        return real_to_netcdf(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", checked)
    path = xarray_to_temporary_netcdf(source)
    Path(path).unlink()

    assert seen == [True]
