"""MODIS near-real-time global flood product (MCDWD) from NASA LANCE.

MCDWD maps water seen by Terra and Aqua at 250 m, in 10x10 degree latitude/longitude GeoTIFF tiles, as
1-, 2- and 3-day composites (``composite_days``; 2 by default). Each pixel is classified as:

    0   no water
    1   surface water that is normally there (reference water)
    2   recurring flood (water seen here in most years at this time)
    3   flood (unusual water)
    255 insufficient data (cloud, no observation)

This plugin stores a flood mask: 1 where the class is ``flood_classes`` (unusual flood only,
by default), 0 for any other observed class and NaN where there was not enough data. The
mean over a district is then the observed fraction of its area under flood water.

LANCE keeps only the last week or so of near-real-time granules; there is no historical
archive to backfill, so the record starts when ingestion first runs and has to be kept up by
a daily sync. Downloads use the Earthdata Login credentials in ``~/.netrc``.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from open_climate_service.plugins.rasters.earthdata import cmr_granules, data_links, download, earthdata_session
from open_climate_service.streaming import BaseDatasetPlugin

_SHORT_NAMES = {1: "MCDWD_L3_F1_NRT", 2: "MCDWD_L3_F2_NRT", 3: "MCDWD_L3_F3_NRT"}
_NO_DATA = 255
_RESOLUTION = 0.0025
# LANCE holds about a week; look back a little further so periods() lists all of it.
_LOOKBACK = timedelta(days=14)


class MCDWDFloodDailyPlugin(BaseDatasetPlugin):
    """Daily flood mask from the MODIS near-real-time global flood product.

    Template params:
        composite_days: 1, 2 (default) or 3.
        flood_classes: classes counted as flooded; ``[3]`` (unusual flood) by default, add 2 to
            include recurring floods.
    """

    max_concurrency = 1

    def __init__(self, composite_days: int = 2, flood_classes: list[int] | None = None, **_: Any) -> None:
        if int(composite_days) not in _SHORT_NAMES:
            raise ValueError(f"{type(self).__name__}: composite_days must be 1, 2 or 3, got {composite_days!r}")
        self.short_name = _SHORT_NAMES[int(composite_days)]
        self.flood_classes = [int(c) for c in (flood_classes or [3])]

    def _granules(self, first: date, last: date, bbox: list[float]) -> list[dict]:
        return cmr_granules(self.short_name, (f"{first}T00:00:00Z", f"{last}T23:59:59Z"), bbox=bbox)

    async def periods(self, start: str, end: str) -> list[str]:
        today = datetime.now(UTC).date()
        first = max(date.fromisoformat(start[:10]), today - _LOOKBACK)
        last = min(date.fromisoformat(end[:10]), today)
        if first > last:
            return []
        from open_climate_service.extents.services import get_extent

        extent = get_extent()
        bbox = list(extent["bbox"]) if extent else [-180.0, -90.0, 180.0, 90.0]
        granules = await asyncio.to_thread(self._granules, first, last, bbox)
        return sorted({g["time_start"][:10] for g in granules})

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rioxarray
        from odc.geo.geobox import GeoBox
        from odc.geo.xr import xr_reproject

        day = date.fromisoformat(period_id)
        granules = [g for g in self._granules(day, day, bbox) if g["time_start"][:10] == period_id]
        urls = [url for g in granules for url in data_links(g, ".tif")]
        if not urls:
            raise RuntimeError(f"No {self.short_name} tiles for {period_id} over {bbox}")
        xmin, ymin, xmax, ymax = map(float, bbox)
        geobox = GeoBox.from_bbox((xmin, ymin, xmax, ymax), crs="EPSG:4326", resolution=_RESOLUTION)
        mosaic = np.full(geobox.shape, _NO_DATA, dtype=np.uint8)
        session = earthdata_session()
        with tempfile.TemporaryDirectory() as tmp:
            for url in urls:
                path = download(url, Path(tmp) / Path(url).name, session=session)
                tile = cast(xr.DataArray, rioxarray.open_rasterio(path)).squeeze("band", drop=True)
                warped = xr_reproject(tile, geobox, resampling="nearest", dst_nodata=_NO_DATA).values
                # Tiles do not overlap; fill only where the mosaic has no observation yet.
                mosaic = np.where((mosaic == _NO_DATA) & (warped != _NO_DATA), warped, mosaic)
        flooded = np.where(mosaic == _NO_DATA, np.nan, np.isin(mosaic, self.flood_classes)).astype("float32")
        coords = geobox.coordinates
        data = xr.DataArray(
            flooded[np.newaxis],
            dims=("t", "y", "x"),
            coords={
                "t": [np.datetime64(period_id, "ns")],
                "y": coords["latitude"].values,
                "x": coords["longitude"].values,
            },
            attrs={"units": "1", "long_name": "flooded"},
        )
        return cast(xr.Dataset, data.to_dataset(name="flooded").rio.write_crs("EPSG:4326"))
