"""Helpers for reading time-chunked Zarr datasets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024


class CachedTimeSeriesZarr:
    """Read small time slices from a Zarr archive with a bounded in-memory cache.

    ``time_dim`` is the archive's time dimension, and its coordinate must be sorted
    ascending.
    The archive is opened lazily on the first call to :meth:`slice_time`, then
    ``prepare`` (if supplied) is applied once.

    ``max_cache_bytes`` bounds the estimated selected-variable data retained by the
    cache and accepted per request. The estimate uses the prepared variable's
    in-memory byte size divided by its number of time steps. Requests whose
    estimated result exceeds the limit raise ``ValueError``; use xarray directly
    for large-range or lazy Dask workflows. Otherwise, the helper loads a window
    of at most the number of time steps that fit in the limit, choosing a window
    that contains the requested range. The cache is reused when it already
    contains the full request, and replaced otherwise.

    ``slice_time(start, end)`` selects the half-open timestamp range ``[start, end)``.
    Like ``xarray.sel``, it selects by coordinate values; unlike a plain ``sel`` or
    positional ``isel``, it also loads and retains a bounded time window so nearby
    requests can reuse it. The returned dataset contains only the requested range.

    This is an estimate of array data size, not a strict process-memory limit.
    Decoding, coordinates, transformations, and physical Zarr chunk reads can use
    additional memory or I/O even when the retained time window is small.
    """

    def __init__(
        self,
        url: str,
        variable: str,
        time_dim: str,
        opener: Callable[[str], xr.Dataset],
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be greater than zero")
        self.url = url
        self.variable = variable
        self._time_dim = time_dim
        self._opener = opener
        self._max_cache_bytes = max_cache_bytes
        self._dataset: xr.Dataset | None = None
        self._bytes_per_time_step: int | None = None
        self._cache_steps: int | None = None
        self._cache_start: int | None = None
        self._cache: xr.Dataset | None = None

    def slice_time(
        self,
        start: np.datetime64,
        end: np.datetime64,
        prepare: Callable[[xr.Dataset], xr.Dataset] | None = None,
    ) -> xr.Dataset:
        """Return timestamps in ``[start, end)`` from the bounded cache.

        The time coordinate must be sorted in ascending order. ``prepare`` runs
        once after opening and before data-size estimation and cache loading.
        Requests larger than ``max_cache_bytes`` raise ``ValueError`` rather than
        unexpectedly materializing a large range.
        """
        if self._dataset is None:
            source = self._opener(self.url)
            dataset = prepare(source) if prepare else source
            self._dataset = dataset[[self.variable]]
            time_steps = self._dataset.sizes[self._time_dim]
            self._bytes_per_time_step = max(1, self._dataset[self.variable].nbytes // time_steps)
            self._cache_steps = max(1, self._max_cache_bytes // self._bytes_per_time_step)

        time_values = self._dataset[self._time_dim].values
        first = int(np.searchsorted(time_values, start))
        last = int(np.searchsorted(time_values, end))
        if last - first == 0:
            return self._dataset.isel({self._time_dim: slice(first, last)})

        assert self._bytes_per_time_step is not None
        assert self._cache_steps is not None
        requested_bytes = (last - first) * self._bytes_per_time_step
        if requested_bytes > self._max_cache_bytes:
            raise ValueError(
                f"Requested time range needs about {requested_bytes} bytes, exceeding "
                f"max_cache_bytes={self._max_cache_bytes}; use xarray directly for large ranges"
            )

        cache_end = (
            self._cache_start + self._cache.sizes[self._time_dim]
            if self._cache is not None and self._cache_start is not None
            else -1
        )
        if self._cache is None or self._cache_start is None or first < self._cache_start or last > cache_end:
            cache_start = min(first, max(0, last - self._cache_steps))
            cache_end = min(self._dataset.sizes[self._time_dim], cache_start + self._cache_steps)
            self._cache = self._dataset.isel({self._time_dim: slice(cache_start, cache_end)}).load()
            self._cache_start = cache_start

        assert self._cache_start is not None
        return self._cache.isel({self._time_dim: slice(first - self._cache_start, last - self._cache_start)})
