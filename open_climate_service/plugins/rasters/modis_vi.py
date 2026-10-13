"""MODIS vegetation indices (NDVI, EVI) from MOD13Q1/MYD13Q1 v6.1 on Microsoft Planetary Computer.

MOD13Q1 (Terra) and MYD13Q1 (Aqua) are 16-day composites at 250 m on the MODIS sinusoidal
tile grid. OCS has no 16-day period type, so this plugin publishes monthly values: the mean
of every good-quality composite whose compositing window has its midpoint in the month.

Terra and Aqua are combined by default. Their composites are staggered by eight days, so a
month gets about four composites instead of two, which matters when one is missing: the
Planetary Computer catalogue has no Terra composite for 2024-08-12, for example.

Pixels are kept when ``pixel_reliability`` is 0 (good) or 1 (marginal); snow, cloud and fill
are dropped before averaging. Tiles are reprojected to EPSG:4326 at ``resolution`` degrees
(default 0.0025, about 250 m). The Planetary Computer STAC API needs no account.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin, monthly_period_ids

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "modis-13Q1-061"
_BANDS = {"ndvi": "250m_16_days_NDVI", "evi": "250m_16_days_EVI"}
_RELIABILITY_BAND = "250m_16_days_pixel_reliability"
_SCALE = 1e-4
# Valid VI range is -2000..10000 (scaled); the fill value is -3000.
_VALID_MIN = -2000
_LOAD_NODATA = -32768
_PRODUCTS = ("MOD13Q1", "MYD13Q1")
# A month is offered once the last composite whose midpoint falls in it has had time to be
# published: compositing ends up to ~8 days after the month, plus production latency.
_PUBLICATION_LAG = timedelta(days=30)


def _midpoint(item: Any) -> date:
    start = datetime.fromisoformat(item.properties["start_datetime"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(item.properties["end_datetime"].replace("Z", "+00:00"))
    return (start + (end - start) / 2).date()


def _latest_items(items: list[Any]) -> list[Any]:
    """Keep one item per (product, composite date, tile): the most recently produced.

    The catalogue sometimes holds a reprocessed item next to the original. Item ids look
    like ``MOD13Q1.A2024241.h19v07.061.2024258125635``; the last part is the production time.
    """
    newest: dict[tuple[str, str, str], Any] = {}
    for item in items:
        product, composite, tile, _, produced = item.id.split(".")
        key = (product, composite, tile)
        if key not in newest or produced > newest[key].id.split(".")[4]:
            newest[key] = item
    return list(newest.values())


class MODISVegetationIndexMonthlyPlugin(BaseDatasetPlugin):
    """Monthly mean NDVI or EVI from MODIS 16-day composites.

    Template params:
        index: ``ndvi`` or ``evi``.
        products: which of ``MOD13Q1`` (Terra) and ``MYD13Q1`` (Aqua) to use; both by default.
        resolution: output grid spacing in degrees (default 0.0025).
    """

    max_concurrency = 1

    def __init__(
        self,
        index: str,
        products: list[str] | None = None,
        resolution: float = 0.0025,
        **_: Any,
    ) -> None:
        if index not in _BANDS:
            raise ValueError(f"{type(self).__name__}: unsupported index {index!r}; expected one of {list(_BANDS)}")
        products = list(products or _PRODUCTS)
        unknown = set(products) - set(_PRODUCTS)
        if unknown:
            raise ValueError(f"{type(self).__name__}: unsupported products {sorted(unknown)}; expected {_PRODUCTS}")
        self.index = index
        self.products = products
        self.resolution = float(resolution)

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = datetime.now(UTC).date() - _PUBLICATION_LAG
        # The last complete month is the latest one ending on or before the cutoff.
        last = cutoff if (cutoff + timedelta(days=1)).day == 1 else cutoff.replace(day=1) - timedelta(days=1)
        return monthly_period_ids(start, end, cutoff=last)

    def _search(self, year: int, month: int, bbox: list[float]) -> list[Any]:
        import planetary_computer
        import pystac_client

        _, last_day = calendar.monthrange(year, month)
        # Composites start up to 15 days before their midpoint.
        window_start = date(year, month, 1) - timedelta(days=16)
        window_end = date(year, month, last_day)
        catalog = pystac_client.Client.open(_STAC_URL, modifier=planetary_computer.sign_inplace)
        found = catalog.search(collections=[_COLLECTION], bbox=bbox, datetime=f"{window_start}/{window_end}").items()
        items = [
            item
            for item in found
            if item.id.split(".")[0] in self.products and (_midpoint(item).year, _midpoint(item).month) == (year, month)
        ]
        return _latest_items(items)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import odc.stac
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

        year, month = int(period_id[:4]), int(period_id[5:7])
        items = self._search(year, month, bbox)
        if not items:
            raise RuntimeError(f"No {'/'.join(self.products)} composites found for {period_id} over {bbox}")
        band = _BANDS[self.index]
        loaded = odc.stac.load(
            items,
            bands=[band, _RELIABILITY_BAND],
            crs="EPSG:4326",
            resolution=self.resolution,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            groupby="solar_day",
            chunks={"x": 2048, "y": 2048},
            resampling="nearest",
            # The catalogue declares pixel_reliability as int8 with nodata 255, which odc
            # rejects; load every band as int16 with an explicit sentinel instead.
            dtype="int16",
            nodata=_LOAD_NODATA,
        )
        values = loaded[band]
        reliability = loaded[_RELIABILITY_BAND]
        good = (values >= _VALID_MIN) & (reliability >= 0) & (reliability <= 1)
        monthly = (values.where(good).astype("float32") * np.float32(_SCALE)).mean("time").compute()
        monthly = monthly.rename({"latitude": "y", "longitude": "x"}).drop_vars("spatial_ref", errors="ignore")
        monthly = monthly.assign_attrs(units="1", long_name=self.index.upper(), composites=len(items))
        ds = monthly.to_dataset(name=self.index).rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
