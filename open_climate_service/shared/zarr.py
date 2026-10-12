"""Helpers for reading time-chunked Zarr datasets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr


class TimeChunkedZarr:
    """Read time ranges from a Zarr archive while reusing loaded time chunks.

    ``time_dim`` is the name of the archive's time dimension, and must be supplied
    explicitly. The selected variable must have Zarr chunk metadata for that
    dimension. ``slice_time`` accepts timestamp boundaries, unlike ``isel``, which
    accepts positional indexes. Like ``sel``, it selects by time labels, but unlike
    a plain xarray selection, it also loads and retains the aligned source chunk(s)
    so later calls can reuse them. It returns only the requested half-open range
    ``[start, end)``.

    The archive is opened on the first call, not during construction. ``_dataset``
    holds the lazily opened and prepared archive; ``_chunk_size`` is the variable's
    chunk length along ``time_dim``; ``_chunk_start`` is the index of the currently
    cached range's first time value, and ``_chunk`` holds that loaded range.

    Each request loads from the chunk-aligned start through the chunk-aligned end
    needed to cover the requested range. A short request usually loads one source
    chunk; a request spanning many chunks loads and retains that whole span, so
    large range requests can use substantial memory. The cache is replaced if a
    request starts at a different chunk boundary, no chunk is cached yet, or the
    cached range does not extend far enough to cover the new request.
    """

    def __init__(
        self,
        url: str,
        variable: str,
        time_dim: str,
        opener: Callable[[str], xr.Dataset],
    ) -> None:
        self.url = url
        self.variable = variable
        self._time_dim = time_dim
        self._opener = opener
        self._dataset: xr.Dataset | None = None
        self._chunk_size: int | None = None
        self._chunk_start: int | None = None
        self._chunk: xr.Dataset | None = None

    def slice_time(
        self,
        start: np.datetime64,
        end: np.datetime64,
        prepare: Callable[[xr.Dataset], xr.Dataset] | None = None,
    ) -> xr.Dataset:
        """Return timestamps in ``[start, end)`` from cached time chunk(s).

        The time coordinate must be sorted in ascending order. ``prepare`` runs
        once after opening and before chunk loading, so spatial clipping and other
        lazy transformations are applied before data is read into memory. The
        requested timestamps are converted to positional bounds, the required
        chunk-aligned range is loaded if it is not already cached, and only the
        requested values are returned.
        """
        if self._dataset is None:
            source = self._opener(self.url)
            time_axis = source[self.variable].dims.index(self._time_dim)
            self._chunk_size = source[self.variable].encoding["chunks"][time_axis]
            self._dataset = prepare(source) if prepare else source

        assert self._chunk_size is not None
        time_values = self._dataset[self._time_dim].values
        first = int(np.searchsorted(time_values, start))
        last = int(np.searchsorted(time_values, end))
        chunk_start = first // self._chunk_size * self._chunk_size
        chunk_size = last - chunk_start

        if self._chunk_start != chunk_start or self._chunk is None or self._chunk.sizes[self._time_dim] < chunk_size:
            chunk_end = chunk_start + ((chunk_size + self._chunk_size - 1) // self._chunk_size) * self._chunk_size
            self._chunk = self._dataset.isel({self._time_dim: slice(chunk_start, chunk_end)}).load()
            self._chunk_start = chunk_start

        return self._chunk.isel({self._time_dim: slice(first - chunk_start, last - chunk_start)})
