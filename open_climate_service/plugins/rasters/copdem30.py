"""Copernicus DEM 30m streaming plugin."""

from typing import Any

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

# v1 is the v0 data rewritten as Zarr v3: north-up, one column per pixel (v0 repeated 180
# degrees), its CRS declared, and smaller chunks.
_DEM30_ZARR_URL = "https://data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v1.zarr"


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
        # The store's CRS is compound (EPSG:9518, WGS 84 with EGM2008 heights); the grid is its
        # horizontal part, and the heights stay in metres above the geoid.
        ds = ds.rio.write_crs("EPSG:4326")
        # No period, so no time axis is added.
        ds = normalize_period(ds, source_variable="dsm", variable="elevation", bbox=bbox)
        return ds.load()
