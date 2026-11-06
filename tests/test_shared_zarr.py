import numpy as np
import pytest
import xarray as xr

from open_climate_service.shared.zarr import CachedTimeSeriesZarr

_STEPS = 30
_CHUNK = 5


def _archive() -> xr.Dataset:
    """Thirty daily steps of one variable stored in five-step time chunks, plus a second variable."""
    times = np.arange("2020-01-01", _STEPS, dtype="datetime64[D]").astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "v": (("t", "y", "x"), np.zeros((_STEPS, 1, 1))),
            "other": (("t", "y", "x"), np.zeros((_STEPS, 1, 1))),
        },
        coords={"t": times},
    )
    ds["v"].encoding["chunks"] = (_CHUNK, 1, 1)
    return ds


def _cached_steps(chunks_per_read: int, first_step: int) -> int:
    ds = _archive()
    zarr = CachedTimeSeriesZarr("unused", "v", time_dim="t", opener=lambda _: ds, chunks_per_read=chunks_per_read)
    start = ds.t.values[first_step]
    zarr.slice_time(start, start + np.timedelta64(1, "D"))
    assert zarr._cache is not None
    return zarr._cache.sizes["t"]


def test_a_cache_miss_reads_through_the_next_chunk_boundary_by_default() -> None:
    assert _cached_steps(chunks_per_read=1, first_step=0) == _CHUNK


def test_chunks_per_read_extends_the_window_by_whole_chunks() -> None:
    assert _cached_steps(chunks_per_read=3, first_step=0) == 3 * _CHUNK


def test_chunks_per_read_stops_at_the_end_of_the_archive() -> None:
    assert _cached_steps(chunks_per_read=3, first_step=_STEPS - 2) == 2


def test_chunks_per_read_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="chunks_per_read"):
        CachedTimeSeriesZarr("unused", "v", time_dim="t", opener=lambda _: _archive(), chunks_per_read=0)


def test_prepare_sees_only_the_selected_variable() -> None:
    seen: list[list[str]] = []

    def prepare(ds: xr.Dataset) -> xr.Dataset:
        seen.append([str(name) for name in ds.data_vars])
        return ds

    ds = _archive()
    zarr = CachedTimeSeriesZarr("unused", "v", time_dim="t", opener=lambda _: ds)
    zarr.slice_time(ds.t.values[0], ds.t.values[1], prepare=prepare)

    assert seen == [["v"]]
