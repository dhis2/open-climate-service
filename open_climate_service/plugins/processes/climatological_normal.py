"""Built-in openEO process: climatological normal (day-of-year or month-of-year).

Reduces a cube's temporal dimension to a WMO-style climatology, so normals can be
computed from an already-ingested managed GeoZarr collection (load the reference period
with ``load_collection``'s ``temporal_extent``, then ``save_result``/publish). The output
replaces the temporal axis with a non-temporal ordinal dimension — ``dayofyear`` (1..366)
or ``month`` (1..12) depending on ``frequency`` — which the STAC layer declares and the
map viewer's generic slider/dropdown renders.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr
from earthkit.transforms import climatology as ek_climatology

from open_climate_service.process import process
from open_climate_service.transforms.climatology import circular_rolling_mean

_FREQUENCIES = ("dayofyear", "month")


@process(
    summary="Climatological normal (day-of-year or month-of-year)",
    parameters={
        "data": {"description": "A raster data cube with a temporal dimension."},
        "frequency": {"description": "Climatology resolution: 'dayofyear' (1..366, default) or 'month' (1..12)."},
        "smoothing_window": {
            "description": (
                "Circular rolling-mean window in days for WMO day-of-year smoothing "
                "(0 disables, must be odd and <= the number of days, default 31). "
                "Ignored when frequency='month'."
            )
        },
    },
)
def climatological_normal(data: xr.DataArray, frequency: str = "dayofyear", smoothing_window: int = 31) -> xr.DataArray:
    """Compute a climatology from a cube's temporal dimension.

    The mean per ``frequency`` bin over the loaded time range; the temporal dimension is
    reduced to an ordinal ``dayofyear`` (1..366) or ``month`` (1..12) dimension. For
    ``dayofyear`` the result is optionally circular-smoothed (WMO 31-day window); smoothing
    is not applied to ``month`` (a 12-value axis). Select the reference period via
    ``load_collection``'s ``temporal_extent`` (e.g. ``["1991-01-01", "2020-12-31"]``).
    """
    if frequency not in _FREQUENCIES:
        raise ValueError(f"frequency must be one of {_FREQUENCIES}, got {frequency!r}")
    window = int(smoothing_window)
    if window < 0:
        raise ValueError(f"smoothing_window must be >= 0 (0 disables), got {window}")

    # Require a datetime-coordinate axis to group on: falling back to a non-datetime
    # dim named "t" would only surface as a confusing error inside groupby(".dayofyear").
    t_dim = next(
        (d for d in data.dims if d in data.coords and np.issubdtype(data[d].dtype, np.datetime64)),
        None,
    )
    if t_dim is None:
        raise ValueError("climatological_normal requires a datetime temporal dimension on the input cube")
    t_dim = str(t_dim)

    # earthkit reduces the temporal axis to the day-of-year (1..366) or month (1..12) mean,
    # handling the calendar binning and preserving dask laziness; the output ordinal dim is
    # named exactly 'dayofyear'/'month'.
    reducer = ek_climatology.daily_mean if frequency == "dayofyear" else ek_climatology.monthly_mean
    clim = cast(xr.DataArray, reducer(data, time_dim=t_dim))
    clim = clim.transpose(frequency, ...)  # ensure the ordinal axis leads (for smoothing)
    # Circular smoothing is only meaningful for the dense day-of-year axis, not 12 months.
    # The window checks are repeated here rather than left to the helper so the error names
    # `smoothing_window` — the process parameter the caller actually passed.
    if frequency == "dayofyear" and window > 0:
        n = clim.sizes["dayofyear"]
        if window % 2 == 0:
            raise ValueError(f"smoothing_window must be odd (a centred window), got {window}")
        if window > n:
            raise ValueError(f"smoothing_window ({window}) must be <= the number of days ({n})")
        clim = circular_rolling_mean(clim, window, "dayofyear")
    # The groupby/smoothing leaves uneven dask chunks along the ordinal axis whose final
    # chunk can exceed the first (e.g. 30,30,…,36), which Zarr rejects on write. Re-chunk to
    # one ordinal step per chunk: zarr-safe (uniform), and — mirroring the observed daily
    # store's per-timestep chunking — it keeps map stepping and compute_anomaly's per-step
    # `.sel` reads cheap (one chunk per day-of-year/month).
    return clim.chunk({frequency: 1})
