"""Copernicus DEM GLO-90 from Microsoft Planetary Computer (no account needed).

The 1x1 degree COG tiles are mosaicked and clipped to the extent on their native
3 arc-second (about 90 m) EPSG:4326 grid. Like the GLO-30 plugin, the static dataset is
represented as a single yearly period; 2021 is the year the 2021 release was published.
"""

from typing import Any, cast

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "cop-dem-glo-90"
_RESOLUTION = 1 / 1200
_PERIOD = "2021"


class CopDEM90Plugin(BaseDatasetPlugin):
    """Streaming plugin for Copernicus DEM GLO-90 elevation."""

    async def periods(self, start: str, end: str) -> list[str]:
        return [_PERIOD] if int(start[:4]) <= int(_PERIOD) <= int(end[:4]) else []

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import odc.stac
        import planetary_computer
        import pystac_client
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

        catalog = pystac_client.Client.open(_STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(catalog.search(collections=[_COLLECTION], bbox=bbox).items())
        if not items:
            raise RuntimeError(f"No {_COLLECTION} tiles intersect {bbox}")
        loaded = odc.stac.load(
            items,
            bands=["data"],
            crs="EPSG:4326",
            resolution=_RESOLUTION,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            chunks={"x": 4096, "y": 4096},
            dtype="float32",
            nodata=np.nan,
        )
        elevation = loaded["data"].isel(time=0, drop=True).compute()
        elevation = elevation.rename({"latitude": "y", "longitude": "x"}).drop_vars("spatial_ref", errors="ignore")
        # Replace odc's attrs: its `nodata=nan` is not valid JSON in the zarr metadata.
        elevation.attrs = {"units": "m"}
        ds = elevation.to_dataset(name="elevation").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(period_id, "ns")]}))
