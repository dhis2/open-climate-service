"""NDVI from the Copernicus Land Monitoring Service (CLMS), global, 300 m, 10-daily (V3).

One composite per dekad (days 1-10, 11-20 and 21 to the end of the month) from 2014, from
PROBA-V until 2018 and Sentinel-3 OLCI since, published once a few days after the dekad ends.
Items are found through the Copernicus Data Space Ecosystem (CDSE) STAC API, which needs no
account. Each is a global cloud-optimised GeoTIFF on a 1/336 degree EPSG:4326 grid in the CDSE
S3 store, of which only the window inside the extent is read. Reading needs free CDSE S3 keys in
the ``cdse`` profile of ``~/.aws/credentials``; GDAL reads the profile itself.

Raw values 0-250 are NDVI, scaled with the file's own scale and offset (``0.004 * value - 0.08``);
252 (unknown), 253 (snow), 254 (water) and 255 (missing) become NaN.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import xarray as xr

from open_climate_service.shared.time import dekad_period_ids
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
_COLLECTION = "clms_ndvi_global_300m_10daily_v3_cog"
_SEARCH_RETRIES = 5
_MAX_VALID = 250
_S3_ENV = {
    "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
    "AWS_PROFILE": "cdse",
    "AWS_VIRTUAL_HOSTING": "FALSE",
    "AWS_HTTPS": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}
_NO_KEYS = (
    "Create free CDSE S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu and add them "
    "to a [cdse] profile in ~/.aws/credentials."
)


def _search(**params: Any) -> list[dict[str, Any]]:
    # The catalogue answers a burst of more than about four searches with a 429, which
    # parallel period fetches reach; it says how long to wait.
    query = {"collections": _COLLECTION, **params}
    response = httpx.get(_STAC_SEARCH, params=query, timeout=60)
    for _ in range(_SEARCH_RETRIES):
        if response.status_code != 429:
            break
        time.sleep(float(response.headers.get("Retry-After", 2)))
        response = httpx.get(_STAC_SEARCH, params=query, timeout=60)
    response.raise_for_status()
    return response.json()["features"]


class CLMSNDVIDekadalPlugin(BaseDatasetPlugin):
    """Streaming plugin for dekadal NDVI from the CLMS global 300 m V3 product."""

    max_concurrency = 4
    commit_batch_size = 36

    async def periods(self, start: str, end: str) -> list[str]:
        """Return the dekads in ``[start, end]``, clamped to the newest published one."""
        latest = await asyncio.to_thread(_search, sortby="-datetime", limit=1)
        if not latest:
            return []
        return dekad_period_ids(start, end, cutoff=latest[0]["properties"]["start_datetime"][:10])

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Read one dekad inside ``bbox`` and return it as NDVI."""
        import rasterio
        import rioxarray
        from rasterio.errors import RasterioIOError

        items = _search(datetime=f"{period_id}T00:00:00Z/{period_id}T23:59:59Z", limit=1)
        if not items:
            raise RuntimeError(f"No CLMS NDVI composite found for the dekad starting {period_id}")
        path = "/vsis3/" + items[0]["assets"]["ndvi"]["href"].removeprefix("s3://")

        # The read is lazy until load, so the whole of it has to happen inside the S3 environment.
        try:
            with rasterio.Env(**_S3_ENV):
                da = rioxarray.open_rasterio(path, chunks=None, mask_and_scale=False)
                if not isinstance(da, xr.DataArray):
                    raise TypeError(f"Expected DataArray from CLMS NDVI raster read, got {type(da).__name__}")
                ds = normalize_period(da, variable="ndvi", period=period_id, bbox=bbox).load()
        except RasterioIOError as exc:
            message = str(exc)
            if any(hint in message for hint in ("403", "AccessDenied", "InvalidAccessKeyId", "AWS_SECRET_ACCESS_KEY")):
                raise PermissionError(f"CDSE S3 refused {path}. {_NO_KEYS}") from exc
            raise

        raw = ds["ndvi"]
        ndvi = raw.where(raw <= _MAX_VALID) * raw.attrs["scale_factor"] + raw.attrs["add_offset"]
        # The file's attributes (fill value, scale, valid range) describe the raw bytes, and its
        # grid mapping coordinate stops carrying the CRS once the arithmetic drops the encoding.
        ndvi = ndvi.astype("float32").reset_coords(drop=True)
        ndvi.attrs = {"units": "1", "long_name": "NDVI"}
        return ndvi.to_dataset(name="ndvi").rio.write_crs(raw.rio.crs)
