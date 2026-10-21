"""Copernicus DEM 30m streaming plugin."""

from typing import Any

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_DEM30_ZARR_URL = "https://api.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr"


class CopDEM30Plugin(BaseDatasetPlugin):
    """Copernicus DEM 30m elevation data from the DestinE Earth Data Hub.

    A static dataset without a time axis (CLIM-1448): elevation does not refer to a period, so
    the plugin implements `fetch` and the store is ``(y, x)``.
    """

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = xr.open_dataset(
            _DEM30_ZARR_URL,
            storage_options={"client_kwargs": {"trust_env": True}},
            chunks={},
            engine="zarr",
        )
        # The store declares no CRS; it is geographic.
        ds = ds.rio.write_crs("EPSG:4326")
        # No period, so no time axis is added.
        ds = normalize_period(ds, source_variable="dsm", variable="elevation", bbox=bbox)
        return ds.load()
