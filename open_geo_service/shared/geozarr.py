"""GeoZarr ``spatial:`` convention attributes derived from a store's own grid.

Write-time only, and deliberately dependency-light: everything here is arithmetic over
coordinate arrays plus one attribute write. What a *reader* is told about a finished store —
its media type — is a serving concern and lives in ``stac/media_types.py``.

Both write paths (the streaming per-period appender and the batch downloader) declare the
`spatial:` convention in ``zarr_conventions``, so both must describe the grid the same way.
This module is the single source of that description, so the two can't drift.

Two things matter to direct-Zarr clients, and both were wrong or missing before:

* **Axis order is array order, ``(y, x)``.** ``spatial:dimensions`` and ``spatial:shape``
  are read positionally — a client takes the second-to-last entry as the row (y) axis and
  the last as the column (x) axis. Writing ``["x", "y"]`` / ``[width, height]`` transposes
  the grid for anything that trusts the convention.
* **The affine matters more than the bbox.** ``spatial:transform`` is what a client needs to
  place a raster; without it, viewers fall back to guessing from the coordinate arrays, and
  that guess is usually hardcoded to EPSG:4326 — which silently mislocates every projected
  store. See CLIM-852.

Everything here is derived from the store's *actual* cell-centre coordinates rather than the
requested bbox: a source may deliver a smaller grid than was asked for (CHIRPS stops at 60°N,
so a request up to 72.5°N yields a grid ending at 60°N), and describing the request instead
of the data stretches the raster over ground it does not cover.
"""

from __future__ import annotations

from typing import Any, Sequence

# ``spatial:registration: "pixel"`` (what both write paths declare) puts the affine origin on
# the outer *edge* of the first cell, not its centre — so a half-step is subtracted below.
# This matches what topozarr writes for pyramid levels, keeping root and level-0 consistent.
_PIXEL_REGISTRATION_HALF_STEP = 0.5


def _as_floats(values: Any) -> list[float]:
    """Coerce a coordinate array (numpy, xarray, list) to a plain list of floats."""
    data = getattr(values, "values", values)
    return [float(v) for v in data]


def grid_geometry(
    x_centres: Sequence[float] | Any,
    y_centres: Sequence[float] | Any,
) -> dict[str, Any] | None:
    """Affine transform, shape and edge bbox for a regular grid, or None if undetermined.

    ``x_centres`` / ``y_centres`` are the store's 1-D coordinate arrays, holding cell
    *centres*. The step is taken from the first two cells and may be negative — a north-up
    raster stores y descending, and the sign has to survive into the transform or the image
    lands mirrored. At least two cells per axis are needed; a single-cell axis leaves the
    cell size unknowable from coordinates alone, so the whole geometry is reported as
    undetermined rather than guessed.
    """
    x = _as_floats(x_centres)
    y = _as_floats(y_centres)
    if len(x) < 2 or len(y) < 2:
        return None

    step_x = x[1] - x[0]
    step_y = y[1] - y[0]
    if step_x == 0.0 or step_y == 0.0:
        return None

    origin_x = x[0] - step_x * _PIXEL_REGISTRATION_HALF_STEP
    origin_y = y[0] - step_y * _PIXEL_REGISTRATION_HALF_STEP
    far_x = x[-1] + step_x * _PIXEL_REGISTRATION_HALF_STEP
    far_y = y[-1] + step_y * _PIXEL_REGISTRATION_HALF_STEP

    return {
        # GeoZarr `spatial:transform`: [stepX, rotX, originX, rotY, stepY, originY].
        "transform": [step_x, 0.0, origin_x, 0.0, step_y, origin_y],
        # Array order, (rows, columns) == (y, x).
        "shape": [len(y), len(x)],
        # Outer edges, ordered [xmin, ymin, xmax, ymax] regardless of which way the axes run.
        "bbox": [
            min(origin_x, far_x),
            min(origin_y, far_y),
            max(origin_x, far_x),
            max(origin_y, far_y),
        ],
    }


def check_grid_description(
    attrs: dict[str, Any],
    *,
    y_dim: str,
    x_dim: str,
    y_size: int,
    x_size: int,
    store: str = "",
) -> None:
    """Raise when the attributes about to be written contradict the grid they describe.

    ``spatial:dimensions`` and ``spatial:shape`` are read positionally, so getting their
    order wrong does not fail loudly: the store publishes successfully and every client that
    trusts the convention silently renders a transposed grid. That is what happened before
    CLIM-852 — 39 of 68 stores on one machine declared ``["x", "y"]`` with a reversed shape,
    and it went unnoticed until a viewer started reading the attributes (CLIM-1008).

    Both are derived from the same dataset moments earlier, so a mismatch here is a bug in
    our own writer rather than bad input. Refusing the write is therefore the right response:
    a store that never claims the wrong geometry cannot mislead a reader later, and no amount
    of downstream care recovers a wrong claim once it is published.

    Square grids are the case worth having a checker for at all — ``[n, n]`` matches either
    way round, so the shape check passes and only the dimension names give the transposition
    away.
    """
    where = f" for {store}" if store else ""

    declared_dims = attrs.get("spatial:dimensions")
    if list(declared_dims or []) != [y_dim, x_dim]:
        raise ValueError(
            f"spatial:dimensions{where} must be array order (y, x) — expected "
            f"{[y_dim, x_dim]}, got {declared_dims!r}. Written x-first, every client that "
            "reads the convention positionally transposes the grid."
        )

    declared_shape = attrs.get("spatial:shape")
    if declared_shape is not None and list(declared_shape) != [y_size, x_size]:
        raise ValueError(f"spatial:shape{where} must be (y, x) == {[y_size, x_size]}, got {list(declared_shape)!r}.")


def gdal_geotransform(transform: Sequence[float]) -> str:
    """A GeoZarr ``spatial:transform`` as GDAL's ``GeoTransform`` string.

    GDAL orders the same six coefficients differently — ``originX stepX rotX originY rotY
    stepY`` — and stores them space-separated on the CF grid-mapping variable. Writing it
    makes the store self-describing to GDAL/QGIS, and is what tells a viewer that a store is
    a *projected* grid rather than degrees (zarr-viewer requires the GeoTransform before it
    will route a store to its projected-grid renderer).
    """
    step_x, rot_x, origin_x, rot_y, step_y, origin_y = transform
    return " ".join(repr(float(v)) for v in (origin_x, step_x, rot_x, origin_y, rot_y, step_y))


def write_gdal_geotransform(root: Any, transform: Sequence[float]) -> None:
    """Stamp the GDAL ``GeoTransform`` onto a store's CF ``spatial_ref`` grid-mapping array.

    ``rio.write_crs`` writes ``crs_wkt`` and ``grid_mapping_name`` but not the transform —
    ``rio.write_transform`` does that, and neither write path calls it, so whether a store
    carries a GeoTransform currently depends on whether its *source* happened to have one
    (GeoTIFF-backed CHIRPS does; NetCDF-backed seNorge does not). That makes the attribute
    unreliable exactly where it matters most: it is the signal zarr-viewer uses to recognise
    a projected grid, and its absence sends a UTM store down the degrees-assuming path.

    A no-op when the store has no ``spatial_ref`` array (nothing to attach it to).
    """
    try:
        spatial_ref = root["spatial_ref"]
    except (KeyError, TypeError):
        return
    spatial_ref.attrs.update({"GeoTransform": gdal_geotransform(transform)})
