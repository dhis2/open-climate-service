"""Measure the statistics of a stored raster (CLIM-1439).

Measured once, when an ingest, sync or openEO publish has finished writing, and recorded on the
artifact as `statistics`, like `size_bytes`. Never measured on a request: a full reduction reads
every chunk, and a store holds one time step per chunk, so a few years of daily data takes
seconds however small the grid.

Measured over the whole store on every run rather than combined with the previous run's figures,
because a sync can replace periods (CHIRPS preliminary by final) as well as append them, and a
replaced extreme would otherwise never leave the range. Measured on the full-resolution data, a
pyramid's level 0: coarser levels are averaged and understate the extremes.
"""

from __future__ import annotations

import logging
import math
from contextlib import closing
from pathlib import Path

from open_climate_service.ingestions.schemas import ArtifactStatistics

logger = logging.getLogger(__name__)


def measure_statistics(store_path: str | Path, variable: str) -> ArtifactStatistics | None:
    """The minimum and maximum of *variable* in the Icechunk store at *store_path*, or None.

    Missing cells are skipped; the store's fill value is already decoded to NaN on open. None
    when the variable is absent or has no finite value. **Never raises**: statistics feed the
    display, and an ingest that otherwise succeeded must not fail over them, so a failure is
    logged with its traceback and recorded as None.
    """
    import dask

    try:
        from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

        with closing(open_icechunk_dataset(store_path)) as ds:
            if variable not in ds.data_vars:
                logger.warning("No variable '%s' in %s to measure statistics from", variable, store_path)
                return None
            array = ds[variable]
            # One pass over the chunks for both reductions, not one each.
            low, high = (float(value) for value in dask.compute(array.min(skipna=True), array.max(skipna=True)))
    except Exception:
        logger.exception("Could not measure statistics of '%s' in %s", variable, store_path)
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return ArtifactStatistics(minimum=low, maximum=high)
