"""Copernicus DEM 30m streaming plugin."""

import logging
from typing import Any

import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

logger = logging.getLogger(__name__)


_DEM30_ZARR_URL = "https://api.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr"


class CopDEM30Plugin(BaseDatasetPlugin):
    """Streaming plugin for Copernicus DEM 30m elevation data from the DestinE Earth Data Hub.

    This is a static dataset but currently represented as a temporal dataset with the year
    2010 as the only valid period (the year the measurements were made).
    """

    async def periods(self, start: str, end: str) -> list[str]:
        if int(end[:4]) < 2010 or int(start[:4]) > 2010:
            # start-end range does not include 2010, we therefore return no valid periods
            return []
        else:
            # start-end range includes 2010, we can then constrain that range to only 2010
            return ['2010']

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        # read the source raster for this period
        ds = xr.open_dataset(
            _DEM30_ZARR_URL,
            storage_options={"client_kwargs": {"trust_env": True}},
            chunks={},
            engine="zarr",
        )
        # make rio compatible by adding crs
        ds = ds.rio.write_crs("EPSG:4326")
        # normalize ds
        ds = normalize_period(ds, source_variable="dsm", variable="elevation", period=period_id, bbox=bbox)
        # load into memory before returning
        ds = ds.load()

        return ds
