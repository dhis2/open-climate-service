"""Measure the statistics of a stored raster (CLIM-1439).

Measured once, when an ingest, sync or openEO publish has finished writing, and recorded on the
artifact as `statistics`, like `size_bytes`. Never measured on a request: a full reduction reads
every chunk, and a store holds one time step per chunk, so a few years of daily data takes
seconds however small the grid.

Measured over the whole store on every run rather than combined with the previous run's figures,
because a sync can replace periods (CHIRPS preliminary by final) as well as append them, and a
replaced extreme would otherwise never leave the range. Measured on the full-resolution data, a
pyramid's level 0: coarser levels are averaged and understate the extremes.

Percentiles come from `dask.array.percentile`, in the same single pass as the exact minimum,
maximum and mean. It summarises each chunk by its own percentiles and merges the summaries, so it
is approximate, more so the more chunks differ (urban against rural tiles, wet against dry
seasons). Asked for the seven percentiles alone it put WorldPop's p98 at 62 people against an
exact 36.5; asked for 1001 evenly spaced percentiles and interpolated, it was within 0.15% of the
range on the stores measured, well under one step of a 256-colour map.
"""

from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from open_climate_service.ingestions.schemas import ArtifactStatistics

logger = logging.getLogger(__name__)

_PERCENTILE_STATS = {name: float(name[1:]) for name in ArtifactStatistics.model_fields if name[0] == "p"}
_PERCENTILE_SAMPLE_AT = np.linspace(0.0, 100.0, 1001)


def measure_statistics(store_path: str | Path, variable: str) -> ArtifactStatistics | None:
    """The statistics of *variable* in the Icechunk store at *store_path*, or None.

    Missing cells are skipped; the store's fill value is already decoded to NaN on open. None
    when the variable is absent or has no finite value. **Never raises**: statistics feed the
    display, and an ingest that otherwise succeeded must not fail over them, so a failure is
    logged with its traceback and recorded as None.
    """
    try:
        from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

        with closing(open_icechunk_dataset(store_path)) as ds:
            if variable not in ds.data_vars:
                logger.warning("No variable '%s' in %s to measure statistics from", variable, store_path)
                return None
            return statistics_of(ds[variable].data)
    except Exception:
        logger.exception("Could not measure statistics of '%s' in %s", variable, store_path)
        return None


def statistics_of(values: Any) -> ArtifactStatistics | None:
    """The statistics of a dask or numpy array's finite values, or None if it has none."""
    import dask.array as da
    from dask.base import compute

    array = da.asarray(values)
    flat = array.ravel()
    try:
        sampled, low, high, mean = compute(
            da.percentile(flat[~da.isnan(flat)], _PERCENTILE_SAMPLE_AT),
            da.nanmin(array),
            da.nanmax(array),
            da.nanmean(array),
        )
    except ValueError:
        # Every value is missing, so `da.percentile` has no chunk summary to merge.
        return None
    return ArtifactStatistics(
        min=float(low),
        max=float(high),
        mean=float(mean),
        **{name: float(np.interp(q, _PERCENTILE_SAMPLE_AT, sampled)) for name, q in _PERCENTILE_STATS.items()},
    )
