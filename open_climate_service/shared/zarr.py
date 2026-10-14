"""Helpers for reading time-chunked Zarr datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024


class CachedTimeSeriesZarr:
    """Read small time slices from a Zarr archive with a bounded in-memory cache.

    ``time_dim`` is the archive's time dimension, and its coordinate must be sorted
    ascending.
    The archive is opened lazily on the first call to :meth:`slice_time`, then
    ``prepare`` (if supplied) is applied once.

    ``max_cache_bytes`` bounds the estimated selected-variable data retained by
    the cache and accepted per request. The estimate uses the prepared
    variable's in-memory byte size divided by its number of time steps.
    Requests whose estimated result exceeds the limit raise ``ValueError``;
    use xarray directly for large-range or lazy Dask workflows.

    On a cache miss, the helper starts the window at the request's first time
    step and extends it through the next source time-chunk boundary, unless
    limited sooner by ``max_cache_bytes``. If source chunk metadata is
    unavailable, it warns and uses the byte limit as the only prefetch bound.
    The cache is reused when it already contains the full request, and
    replaced otherwise.

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
        self._max_cache_steps: int | None = None
        self._time_chunk_steps: int | None = None
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
        # Initial opening and preparation of the source dataset, if not already done
        if self._dataset is None:
            source = self._opener(self.url)
            source_variable = source[self.variable]

            # Determine the source's time chunk size, if available
            chunks = source_variable.encoding.get("chunks")
            if chunks is not None and len(chunks) == source_variable.ndim:
                chunk_steps = chunks[source_variable.get_axis_num(self._time_dim)]
                if chunk_steps is not None and chunk_steps > 0:
                    self._time_chunk_steps = chunk_steps

            # Warn if the source's time chunk size is unavailable
            if self._time_chunk_steps is None:
                logger.warning(
                    f"Time chunk metadata is unavailable for {self.variable} from {self.url}; "
                    "cache prefetch will be limited only by max_cache_bytes"
                )

            # Prepare the dataset and compute various cache size estimates
            dataset = prepare(source) if prepare else source
            self._dataset = dataset[[self.variable]]
            time_steps = self._dataset.sizes[self._time_dim]
            self._bytes_per_time_step = max(1, self._dataset[self.variable].nbytes // time_steps)
            self._max_cache_steps = max(1, self._max_cache_bytes // self._bytes_per_time_step)

        # Return early if the requested range contains no timestamps
        time_values = self._dataset[self._time_dim].values
        first = int(np.searchsorted(time_values, start))
        last = int(np.searchsorted(time_values, end))
        if last - first == 0:
            return self._dataset.isel({self._time_dim: slice(first, last)})

        # Check if the requested range exceeds the max cache size
        assert self._bytes_per_time_step is not None
        assert self._max_cache_steps is not None
        requested_bytes = (last - first) * self._bytes_per_time_step
        if requested_bytes > self._max_cache_bytes:
            raise ValueError(
                f"Requested time range needs about {requested_bytes} bytes, exceeding "
                f"max_cache_bytes={self._max_cache_bytes}; either increase the plugin's "
                f"max_cache_bytes or use xarray directly for large ranges"
            )

        # Trigger a cache refresh if the requested range is outside the current cache
        cache_end = (
            self._cache_start + self._cache.sizes[self._time_dim]
            if self._cache is not None and self._cache_start is not None
            else -1
        )
        if self._cache is None or self._cache_start is None or first < self._cache_start or last > cache_end:
            cache_start = first

            # Determine cache end
            max_cache_end = min(self._dataset.sizes[self._time_dim], cache_start + self._max_cache_steps)
            if self._time_chunk_steps is None:
                # Time chunk metadata is unavailable, so limit the cache to the max cache size
                cache_end = max_cache_end
            else:
                # Set the cache end to the next source time-chunk boundary, but not beyond the max cache size
                next_chunk_end = (
                    (last + self._time_chunk_steps - 1) // self._time_chunk_steps
                ) * self._time_chunk_steps
                cache_end = min(max_cache_end, max(last, next_chunk_end))
            cache_steps = cache_end - cache_start
            cache_bytes = self._bytes_per_time_step * cache_steps
            time_values = self._dataset[self._time_dim].values
            logger.info(
                f"Requested time range {time_values[first]} to {time_values[last - 1]} "
                f"has triggered fetching new zarr cache for variable {self.variable} from {self.url}; "
                f"cache contains {cache_steps} time steps "
                f"({time_values[cache_start]} through {time_values[cache_end - 1]}); "
                f"estimated cache size is {cache_bytes / 1024 / 1024:.1f} MiB"
            )

            # Fetch the new cache and retain it in memory for future requests
            self._cache = self._dataset.isel({self._time_dim: slice(cache_start, cache_end)}).load()
            self._cache_start = cache_start
            logger.info(f"Actual cache size: {self._cache.nbytes / 1024 / 1024:.1f} MiB")

        # Return the requested slice from the cache
        assert self._cache_start is not None
        return self._cache.isel({self._time_dim: slice(first - self._cache_start, last - self._cache_start)})
