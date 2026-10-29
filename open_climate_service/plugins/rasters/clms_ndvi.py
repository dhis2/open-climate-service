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
from threading import Lock
from typing import Any

import httpx
import xarray as xr

from open_climate_service.shared.time import dekad_period_ids
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
_COLLECTION = "clms_ndvi_global_300m_10daily_v3_cog"
_MAX_VALID = 250
_S3_ENV: dict[str, Any] = {
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


def _search_stac_items(**params: Any) -> list[dict[str, Any]]:
    """Return the collection's STAC items matching ``params``, from every page."""
    items: list[dict[str, Any]] = []
    url: str | None = _STAC_SEARCH
    query: dict[str, Any] | None = {"collections": _COLLECTION, **params}
    while url:
        response = httpx.get(url, params=query, timeout=60)
        response.raise_for_status()
        page = response.json()
        items.extend(page["features"])
        # The next link carries the query itself.
        url = next((link["href"] for link in page.get("links", []) if link["rel"] == "next"), None)
        query = None
    return items


class CLMSNDVIDekadalPlugin(BaseDatasetPlugin):
    """Streaming plugin for dekadal NDVI from the CLMS global 300 m V3 product."""

    max_concurrency = 4
    commit_batch_size = 36

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._stac_items_lock = Lock()
        self._stac_items_bbox: tuple[float, ...] | None = None
        self._stac_items: list[dict[str, Any]] = []

    async def periods(self, start: str, end: str) -> list[str]:
        """Return the dekads in ``[start, end]``, clamped to the newest published one."""
        cutoff = await asyncio.to_thread(self._fetch_cutoff)
        return dekad_period_ids(start, end, cutoff=cutoff)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Read one dekad inside ``bbox`` and return it as NDVI."""
        import rasterio
        import rioxarray
        from rasterio.errors import RasterioIOError

        items = [item for item in self._fetch_stac_items(bbox) if item["properties"]["start_datetime"][:10] == period_id]
        if not items:
            raise RuntimeError(f"No CLMS NDVI composite published for the dekad starting {period_id}")
        if len(items) > 1:
            raise RuntimeError(
                f"{len(items)} CLMS NDVI items cover the extent for the dekad starting {period_id}; "
                "this plugin reads one global raster per dekad and does not mosaic tiles"
            )
        path = items[0]["assets"]["ndvi"]["href"]

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

    def _fetch_cutoff(self) -> str:
        """Return the first day of the newest published dekad."""
        query = {"collections": _COLLECTION, "sortby": "-datetime", "limit": 1}
        response = httpx.get(_STAC_SEARCH, params=query, timeout=60)
        response.raise_for_status()
        return response.json()["features"][0]["properties"]["start_datetime"][:10]

    def _fetch_stac_items(self, bbox: list[float]) -> list[dict[str, Any]]:
        """Return every STAC item intersecting ``bbox``, from one search cached for the plugin's lifetime.

        Trimmed to the fields read here: about 460 items and 430 kB for the whole collection,
        against 5 MB for full items. A search per dekad would instead have
        parallel fetches reach the catalogue's limit of four concurrent connections. Parallel
        fetches arrive here together on the first period, and one of them searches.
        """
        with self._stac_items_lock:
            # fetch and cache the stac items only if the bbox is new or has changed
            if self._stac_items_bbox != tuple(bbox):
                self._stac_items = _search_stac_items(
                    bbox=",".join(map(str, bbox)),
                    limit=1000,
                    fields="id,properties.start_datetime,assets.ndvi.href",
                )
                self._stac_items_bbox = tuple(bbox)
            # return the cached stac items for the current bbox
            return self._stac_items
