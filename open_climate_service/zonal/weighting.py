"""Area-weighted statistics over polygons, read and reduced a block at a time."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.zonal.categorical import (
    FRACTIONS_DIM,
    fraction_classes,
    weighted_fractions,
    weighted_majority,
)
from open_climate_service.zonal.grid import Grid, VarResult, Zone, blocks, crop


def weighted_statistic(values: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    """Reduce the last axis of *values* (…, cells) by *method*, weighting each cell.

    NaN cells are left out, as the unweighted path drops them. A zone with no valid cell is NaN.
    """
    valid = ~np.isnan(values)
    w = np.where(valid, weights, 0.0)
    total = w.sum(axis=-1)
    # Whether the zone has any value at all; an infinite min or max is still a value.
    has_value = (w > 0).any(axis=-1)
    filled = np.where(valid, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "mean":
            out = (filled * w).sum(axis=-1) / total
        elif method == "sum":
            out = (filled * w).sum(axis=-1)
            out = np.where(total > 0, out, np.nan)
        elif method == "min":
            out = np.where(valid, values, np.inf).min(axis=-1, initial=np.inf)
            out = np.where(has_value, out, np.nan)
        elif method == "max":
            out = np.where(valid, values, -np.inf).max(axis=-1, initial=-np.inf)
            out = np.where(has_value, out, np.nan)
        elif method == "median":
            out = _weighted_quantile(values, w, 0.5)
        elif method == "majority":
            out = weighted_majority(values, w)
        else:
            raise ValueError(f"no weighted implementation for '{method}'")
    return np.asarray(out, dtype="float64")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """The value at which cumulative weight reaches *q* of the zone's total.

    When the cumulative weight lands exactly on *q*, the midpoint of that value and the next is
    taken, so equal weights give what ``np.median`` gives: two whole cells of 0 and 10 are 5.
    Exactly means within rounding error; covered areas of 0.500001 and 0.499999 are not a tie.

    Computed for every row at once: one sort along the cell axis, so 30 daily years over a
    zone cost about what the mean does rather than a Python step per day.
    """
    flat_v = values.reshape(-1, values.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    keep = flat_w > 0
    # Cells left out sort last and weigh nothing, so each row's kept cells lead in order.
    order = np.argsort(np.where(keep, flat_v, np.inf), axis=-1, kind="stable")
    ordered = np.take_along_axis(flat_v, order, axis=-1)
    cumulative = np.cumsum(np.take_along_axis(np.where(keep, flat_w, 0.0), order, axis=-1), axis=-1)
    n_kept = keep.sum(axis=-1)
    target = q * cumulative[:, -1:] if cumulative.shape[-1] else np.zeros((flat_v.shape[0], 1))
    # The first cell whose cumulative weight reaches the target, as searchsorted finds it.
    idx = np.minimum((cumulative < target).sum(axis=-1), np.maximum(n_kept - 1, 0))
    rows = np.arange(flat_v.shape[0])
    out = np.full(flat_v.shape[0], np.nan)
    if not cumulative.shape[-1]:
        return out.reshape(values.shape[:-1])
    at = ordered[rows, idx]
    following = ordered[rows, np.minimum(idx + 1, ordered.shape[-1] - 1)]
    tie = (idx + 1 < n_kept) & np.isclose(cumulative[rows, idx], target[:, 0], rtol=1e-9, atol=0.0)
    out = np.where(tie, (at + following) / 2, at)
    out = np.where(n_kept > 0, out, np.nan)
    return out.reshape(values.shape[:-1])


def weighted_polygons(data: xr.Dataset, grid: Grid, zones: list[Zone], methods: dict[str, str]) -> dict[str, VarResult]:
    """Per variable, reduced by its own method: an array (zone, …other dims) and its dims and coords.

    Read and reduced a block at a time (``READ_BLOCK_BYTES``), so memory does not grow with
    the length of the series.
    """
    window, r0, c0 = crop(data, grid, zones)
    covered = [z for z in zones if z.rows.size]

    def other_dims(vname: str) -> list[Hashable]:
        return [d for d in window[vname].dims if d not in {grid.y_dim, grid.x_dim}]

    # One class axis for every variable reduced to fractions, found only in covered cells. It
    # must be known before any block is reduced, so fractions take one extra pass over the data.
    classes = fraction_classes(
        block[..., z.rows - r0, z.cols - c0]
        for v, m in methods.items()
        if m == "fractions"
        for _axis, block in blocks(window[v], other_dims(v), grid)
        for z in covered
    )
    out: dict[str, VarResult] = {}
    for name in data.data_vars:
        vname = str(name)
        method = methods[vname]
        da = window[vname]
        other = other_dims(vname)
        pieces: list[list[np.ndarray]] = [[] for _ in zones]
        concat_axis = 0
        for axis, block in blocks(da, other, grid):
            concat_axis = axis or 0
            for zone, zone_pieces in zip(zones, pieces, strict=True):
                if not zone.rows.size:
                    extra = (classes.size,) if method == "fractions" else ()
                    zone_pieces.append(np.full(block.shape[:-2] + extra, np.nan))
                    continue
                values = block[..., zone.rows - r0, zone.cols - c0]
                weights = np.broadcast_to(zone.weights, values.shape)
                if method == "fractions":
                    zone_pieces.append(weighted_fractions(values, weights, classes))
                else:
                    zone_pieces.append(weighted_statistic(values, weights, method))
        rows = [np.concatenate(p, axis=concat_axis) if other else p[0] for p in pieces]
        coords: dict[Hashable, Any] = {d: da.coords[d].values for d in other if d in da.coords}
        dims: list[Hashable] = list(other)
        if method == "fractions":
            dims.append(FRACTIONS_DIM)
            coords[FRACTIONS_DIM] = classes
        out[vname] = (np.stack(rows), dims, coords)
    return out
