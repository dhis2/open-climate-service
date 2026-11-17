"""Climatology helpers that fill gaps in ``earthkit.transforms``.

Everything here exists because earthkit does not (yet) cover it. Each helper delegates as
much as possible to upstream and keeps only the missing piece, so it shrinks or disappears
as earthkit gains the feature — see the issue references on each function.
"""

from __future__ import annotations

from typing import cast

import xarray as xr
from earthkit.transforms import temporal as ek_temporal

__all__ = ["circular_rolling_mean"]


def circular_rolling_mean(da: xr.DataArray, window: int, dim: str = "dayofyear") -> xr.DataArray:
    """Centred rolling mean over a *cyclic* ordinal axis, wrapping end→start.

    WMO day-of-year normals are smoothed with a centred 31-day window, and the axis is
    cyclic: 1 January must average across the year boundary rather than being truncated.

    The reduction itself is earthkit's — ``temporal.rolling_reduce`` — so we are not
    reimplementing a rolling mean. What earthkit is missing is the *wrap*: it leaves NaN at
    both ends of the axis, and its rolling support lives in ``temporal`` (oriented at a
    datetime axis) rather than in ``climatology``, whose output axis is an ordinal
    ``dayofyear`` (1..366) or ``month`` (1..12). Tracked upstream as
    `ecmwf/earthkit-transforms#103 <https://github.com/ecmwf/earthkit-transforms/issues/103>`_;
    delete this wrapper and call ``rolling_reduce`` directly once that lands.

    So the only local logic is: pad ``window // 2`` steps from each end of the circular
    axis, let earthkit reduce, then drop the padding. The padded region absorbs the NaN
    edges, leaving every ordinal step with a full window.

    Laziness is preserved throughout — ``concat`` and ``rolling_reduce`` are both dask-aware,
    so a ``(dim, y, x)`` cube is never materialised. The input's chunking along ``dim`` is
    restored at the end: padding and slicing leave the axis fragmented (a 13-chunk input came
    back as ``(15, 30, …, 36, 15)``), and Zarr accepts uniform chunks with a smaller final one
    and nothing else, so writing that result would fail.

    Args:
        da: cube with ``dim`` as a cyclic ordinal axis.
        window: centred window length in steps. Must be odd (a centred window has a true
            middle) and no longer than the axis; ``1`` is the identity.
        dim: name of the cyclic axis.

    Returns:
        The smoothed cube, same shape and coordinates as ``da``.
    """
    n = da.sizes[dim]
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window % 2 == 0:
        raise ValueError(f"window must be odd (a centred window), got {window}")
    if window > n:
        raise ValueError(f"window ({window}) must be <= the length of {dim!r} ({n})")

    half = window // 2
    if half == 0:  # window == 1
        return da

    head = da.isel({dim: slice(0, half)})
    tail = da.isel({dim: slice(n - half, n)})
    padded = xr.concat([tail, da, head], dim=dim)

    smoothed = cast(
        xr.DataArray,
        ek_temporal.rolling_reduce(
            padded,
            window_length=window,
            time_dim=dim,
            center=True,
            how_reduce="mean",
        ),
    )

    # Drop the wrap padding and restore the original ordinal coordinate: concat left the
    # coordinate values duplicated across the padded region.
    result = smoothed.isel({dim: slice(half, half + n)}).assign_coords({dim: da[dim]})

    # Restore the caller's chunking along `dim`; see the note on Zarr's chunk rules above.
    if da.chunks is not None:
        result = result.chunk({dim: da.chunksizes[dim]})
    return result
