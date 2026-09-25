import logging

import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period


logger = logging.getLogger(__name__)


_DEM30_ZARR_URL = "https://api.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr"


class CopDEM30Plugin(BaseDatasetPlugin):
    async def periods(self, start: str, end: str) -> list[str]:
        """Return the ordered list of period ids available between start and end."""
        if end < '2010' or start > '2010':
            # start-end range does not include 2010, we therefore return no valid periods
            return []
        else:
            # start-end range includes 2010, we can then constrain that range to only 2010
            return ['2010']

    def fetch_period(self, period_id: str, bbox: list[float], **params) -> xr.Dataset:
        """Fetch one period and return it as an xarray Dataset."""
        import traceback
        try:
            # period id should be empty...
            logger.info(f'period id {period_id}')
            # read the source raster for this period
            ds = xr.open_dataset(
                _DEM30_ZARR_URL,
                storage_options={"client_kwargs":{"trust_env":True}},
                chunks={},
                engine="zarr",
            )
            logger.info(f'original ds {ds}')
            # make rio compatible by adding crs
            ds = ds.rio.write_crs("EPSG:4326")
            # normalize ds
            ds = normalize_period(ds, source_variable="dsm", variable="elevation", period=period_id, bbox=bbox)
            logger.info(f'normalized ds {ds}')
            # load into memory before returning
            ds = ds.load()
            #logger.info(f'ds range {ds.elevation.min()} {ds.elevation.max()}')
        except: 
            raise Exception(traceback.format_exc())
        return ds
