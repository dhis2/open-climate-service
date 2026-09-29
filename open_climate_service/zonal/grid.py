"""The cube's cell layout, each polygon's coverage of it, and reading a cube in blocks."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

# Coverage below this share of a zone's largest is floating-point residue at its edge, not a
# covered cell.
_MIN_COVERAGE = 1e-9

VarResult = tuple[np.ndarray, list[Hashable], dict[Hashable, Any]]
"""One variable's result: an array led by the geometry axis, and its other dims and coords."""


def find_dim(data: xr.Dataset | xr.DataArray, candidates: list[str]) -> str | None:
    """The first of *candidates* that is a dimension of *data*."""
    dims = data.dims if isinstance(data, xr.DataArray) else set(data.dims)
    for c in candidates:
        if c in dims:
            return c
    return None


def _declared_resolution(data: xr.Dataset) -> tuple[float, float]:
    """The cell size the cube's geotransform declares, for an axis one cell long.

    Coordinates cannot give a cell size from a single value, so a cube sliced to one cell
    falls back on the transform it carries; without one, a cell of 1 unit is assumed.
    """
    try:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers .rio

        x_res, y_res = data.rio.resolution()
        return float(abs(x_res)), float(abs(y_res))
    except Exception:
        return 1.0, 1.0


@dataclass(frozen=True)
class Grid:
    """The cube's cell layout, as exactextract sees it (row 0 north, column 0 west)."""

    x_dim: str
    y_dim: str
    width: int
    height: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    x_ascending: bool
    y_ascending: bool

    @classmethod
    def of(cls, data: xr.Dataset, x_dim: str, y_dim: str) -> Grid:
        x = data[x_dim].values.astype(float)
        y = data[y_dim].values.astype(float)
        declared = _declared_resolution(data)
        dx = float(abs(x[1] - x[0])) if x.size > 1 else declared[0]
        dy = float(abs(y[1] - y[0])) if y.size > 1 else declared[1]
        return cls(
            x_dim=x_dim,
            y_dim=y_dim,
            width=int(x.size),
            height=int(y.size),
            xmin=float(x.min()) - dx / 2,
            xmax=float(x.max()) + dx / 2,
            ymin=float(y.min()) - dy / 2,
            ymax=float(y.max()) + dy / 2,
            x_ascending=not (x.size > 1 and x[1] < x[0]),
            y_ascending=bool(y.size > 1 and y[1] > y[0]),
        )

    def array_indices(self, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map exactextract cell ids to (row, column) indices into the cube's own arrays."""
        rows, cols = np.divmod(cell_ids.astype(np.int64), self.width)
        if self.y_ascending:
            rows = self.height - 1 - rows
        if not self.x_ascending:
            cols = self.width - 1 - cols
        return rows, cols


@dataclass
class Zone:
    """The cells a polygon covers, as indices into the cube, with the fraction of each covered."""

    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray


def polygon_zones(grid: Grid, polygons: Sequence[Any]) -> list[Zone]:
    """Coverage fractions for each polygon, in one exactextract call."""
    from exactextract import exact_extract
    from exactextract.raster import NumPyRasterSource
    from shapely.geometry import mapping

    if not polygons:
        return []
    # A zero-stride array: exactextract needs the grid, not the values, so nothing is allocated.
    source = NumPyRasterSource(
        np.broadcast_to(np.float32(0), (grid.height, grid.width)),
        xmin=grid.xmin,
        ymin=grid.ymin,
        xmax=grid.xmax,
        ymax=grid.ymax,
    )
    features = [{"type": "Feature", "properties": {}, "geometry": mapping(geom)} for geom in polygons]
    # A DataFrame with output="pandas"; the untyped return is annotated so pyright accepts it.
    table: Any = exact_extract(source, features, ["cell_id", "coverage"], output="pandas")
    zones: list[Zone] = []
    for cell_ids, coverage in zip(table["cell_id"], table["coverage"], strict=True):
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        weights = np.asarray(coverage, dtype="float64")
        # Relative to the zone's own largest coverage, so a zone far smaller than its one cell
        # keeps that cell, while rounding residue beside fully covered cells still goes.
        keep = weights > _MIN_COVERAGE * (weights.max() if weights.size else 0.0)
        rows, cols = grid.array_indices(cell_ids[keep])
        zones.append(Zone(rows=rows, cols=cols, weights=weights[keep]))
    return zones


def crop(data: xr.Dataset, grid: Grid, zones: list[Zone]) -> tuple[xr.Dataset, int, int]:
    """Load only the window the zones cover, returning it with its row and column offsets."""
    covered = [z for z in zones if z.rows.size]
    if not covered:
        return data.isel({grid.y_dim: slice(0, 0), grid.x_dim: slice(0, 0)}), 0, 0
    r0 = int(min(z.rows.min() for z in covered))
    r1 = int(max(z.rows.max() for z in covered))
    c0 = int(min(z.cols.min() for z in covered))
    c1 = int(max(z.cols.max() for z in covered))
    return data.isel({grid.y_dim: slice(r0, r1 + 1), grid.x_dim: slice(c0, c1 + 1)}), r0, c0


READ_BLOCK_BYTES = 512 * 2**20
"""Most bytes of one variable read into memory at once, as float64.

A cube is reduced block by block along its largest non-spatial dimension, so memory stays
bounded however long the series is: 30 years of seNorge daily temperature over Norway is about
160 GB as float64, and was measured at 39 GiB peak for 5 years before blocking.
"""


def blocks(da: xr.DataArray, other: list[Hashable], grid: Grid) -> Iterator[tuple[int | None, np.ndarray]]:
    """Yield *da* as float64 (…other, y, x) pieces along its largest non-spatial dimension.

    Each piece is paired with the position of that dimension in *other*, or None when there is
    no non-spatial dimension and the whole array is one piece.
    """
    ordered = da.transpose(*other, grid.y_dim, grid.x_dim)
    if not other:
        yield None, np.asarray(ordered.values, dtype="float64")
        return
    axis = max(range(len(other)), key=lambda i: int(ordered.sizes[other[i]]))
    dim = other[axis]
    per_step = 8 * int(np.prod([int(ordered.sizes[d]) for d in ordered.dims if d != dim]))
    step = max(1, READ_BLOCK_BYTES // max(per_step, 1))
    for start in range(0, int(ordered.sizes[dim]), step):
        yield axis, np.asarray(ordered.isel({dim: slice(start, start + step)}).values, dtype="float64")
