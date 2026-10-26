"""Copernicus DEM streaming plugin, for the 30 m (GLO-30) and 90 m (GLO-90) products."""

from typing import Any

import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

# v1 is the v0 data rewritten as Zarr v3: north-up, one column per pixel (v0 repeated 180
# degrees), its CRS declared, and smaller chunks. Both products are published this way.
_ZARR_URL = "https://data.earthdatahub.destine.eu/copernicus-dem/{product}-v1.zarr"
_PRODUCTS = {"GLO-30", "GLO-90"}


class CopDEMPlugin(BaseDatasetPlugin):
    """Copernicus DEM elevation data from the DestinE Earth Data Hub.

    A static dataset without a time axis (CLIM-1448): elevation does not refer to a period, so
    the plugin implements `fetch` and the store is ``(y, x)``.

    Args:
        product: ``GLO-30`` (1 arc-second, about 30 m) or ``GLO-90`` (3 arc-seconds, about 90 m).
    """

    def __init__(self, product: str = "GLO-30", **_: Any) -> None:
        if product not in _PRODUCTS:
            raise ValueError(f"product must be one of {sorted(_PRODUCTS)}, got {product!r}")
        self.product = product

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = xr.open_dataset(
            _ZARR_URL.format(product=self.product),
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
