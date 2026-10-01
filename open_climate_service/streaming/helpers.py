"""Shared helpers for streaming dataset plugins.

`normalize_period` turns a freshly read raster / dataset into the canonical
``(t, y, x)`` single-variable shape the store expects: drop curvilinear 2-D
lon/lat helpers, rename dims (incl. projected ``X``/``Y``), reproject-clip to the
requested bbox, drop the band axis, mask the nodata sentinel, and stamp the period
onto a time dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

# Source dim/coord names mapped onto the canonical x / y axes. Includes geographic
# (lon/lat) and projected (X/Y) spellings so projected-grid plugins need no bespoke
# rename. First match wins.
_X_NAMES = ("lon", "longitude", "x", "X")
_Y_NAMES = ("lat", "latitude", "y", "Y")
_TIME_NAMES = ("time", "valid_time")


def cell_pad(coord: xr.DataArray | np.ndarray) -> float:
    """Return half the widest cell spacing on ``coord``.

    Coordinates name cell *centres*, so a bbox edge falling inside a cell still needs that
    whole cell. Padding a label slice by this much before selecting is what turns
    centre-based selection into footprint-based selection.

    The widest spacing is used rather than the local one so the result is provably sufficient
    on an irregular axis; at worst it includes one extra cell, which `normalize_period` then
    trims. Returns 0.0 for an axis with fewer than two values, where there is no spacing to
    infer and nothing to pad.
    """
    values = np.asarray(getattr(coord, "values", coord), dtype="float64").ravel()
    if values.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values)))) / 2.0


def bbox_slice(coord: xr.DataArray | np.ndarray, low: float, high: float) -> slice:
    """Return a label slice covering every cell on ``coord`` whose footprint meets ``[low, high]``.

    Use this instead of ``slice(low, high)`` when selecting a bbox from a source grid. Label
    selection keeps only cells whose centre lies inside the bounds, so the cells straddling
    each edge are dropped and the result covers *less* than the bbox — up to half a cell short
    on every side. On a coarse grid that is kilometres of missing coverage at the edge of the
    instance extent, which shows up as an uncovered strip on the map and as border districts
    aggregated from partial data.

    The returned slice follows the coordinate's own direction, so it works unchanged on a
    descending latitude axis (as most geographic sources have).

    Plugins that read a remote store and cannot afford to fetch it whole should slice with
    this: over-selecting by a cell is free once `normalize_period` clips exactly, whereas
    under-selecting cannot be recovered downstream.
    """
    pad = cell_pad(coord)
    values = np.asarray(getattr(coord, "values", coord), dtype="float64").ravel()
    descending = values.size > 1 and values[0] > values[-1]
    if descending:
        return slice(high + pad, low - pad)
    return slice(low - pad, high + pad)


def bbox_slices(
    obj: xr.Dataset | xr.DataArray,
    bbox: list[float] | tuple[float, float, float, float],
    *,
    x_dim: str,
    y_dim: str,
) -> dict[str, slice]:
    """Return ``{x_dim: slice, y_dim: slice}`` covering every cell that meets ``bbox``.

    A convenience over :func:`bbox_slice` for the common case, so a plugin can write
    ``ds.sel(**bbox_slices(ds, bbox, x_dim="longitude", y_dim="latitude"))``. Takes the bbox in
    the source's own coordinate values — reproject first if the source is not in the bbox CRS.
    """
    xmin, ymin, xmax, ymax = (float(value) for value in bbox)
    return {
        x_dim: bbox_slice(obj[x_dim], xmin, xmax),
        y_dim: bbox_slice(obj[y_dim], ymin, ymax),
    }


def normalize_period(
    obj: "xr.DataArray | xr.Dataset",
    *,
    variable: str,
    source_variable: str | None = None,
    period: str | None = None,
    nodata: float | None = None,
    bbox: list[float] | None = None,
    bbox_crs: int | str = "EPSG:4326",
    squeeze_band: bool = True,
    time_dim: str = "t",
    x_dim: str = "x",
    y_dim: str = "y",
) -> "xr.Dataset":
    """Normalize a fetched raster/dataset to the canonical single-period shape.

    Steps (each applied only when relevant):
    1. drop curvilinear 2-D ``lon``/``lat`` helper coordinates (they are not the
       store's spatial dims and confuse both the rename below and rioxarray)
    2. rename the source x axis (``lon``/``longitude``/``X``) → `x_dim`, the y axis
       (``lat``/``latitude``/``Y``) → `y_dim`, and ``time``/``valid_time`` → `time_dim`
    3. clip to `bbox`, reprojecting the bbox from `bbox_crs` to the source CRS — so
       a projected-grid source (e.g. UTM) clips correctly against a WGS84 bbox
       without any bespoke coordinate transform (requires the source to carry a CRS)
    4. drop a singleton ``band`` dimension
    5. wrap a `DataArray` into a `Dataset` named `variable` (or rename
       `source_variable` → `variable` for a `Dataset`)
    6. mask the `nodata` sentinel to NaN
    7. stamp `period` onto `time_dim` when the data has no time dimension yet
    """
    import xarray as xr

    aux_2d = [c for c in getattr(obj, "coords", {}) if obj[c].ndim > 1 and c in (_X_NAMES + _Y_NAMES)]
    if aux_2d:
        obj = obj.drop_vars(aux_2d)

    names = set(getattr(obj, "dims", ())) | set(getattr(obj, "coords", {}))
    rename: dict[str, str] = {}
    for src in _X_NAMES:
        if src in names and x_dim not in names:
            rename[src] = x_dim
            break
    for src in _Y_NAMES:
        if src in names and y_dim not in names:
            rename[src] = y_dim
            break
    for src in _TIME_NAMES:
        if src in names and time_dim not in names:
            rename[src] = time_dim
            break
    if rename:
        obj = obj.rename(rename)

    if bbox is not None:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

        xmin, ymin, xmax, ymax = map(float, bbox)
        obj = obj.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax, crs=bbox_crs)

    if squeeze_band and "band" in getattr(obj, "dims", ()):
        obj = obj.squeeze("band", drop=True)

    if isinstance(obj, xr.DataArray):
        ds = obj.to_dataset(name=variable)
    else:
        ds = obj
        if source_variable and source_variable != variable and source_variable in ds:
            ds = ds.rename({source_variable: variable})
        if variable in ds:
            ds = ds[[variable]]

    if nodata is not None and variable in ds:
        ds[variable] = ds[variable].where(ds[variable] != nodata)

    if period is not None and time_dim not in ds.dims:
        ds = ds.expand_dims({time_dim: [np.datetime64(period)]})

    return ds
