"""ESA CCI / C3S land cover (300 m, yearly) from Microsoft Planetary Computer.

Two plugins over the same source:

- ``ESACCILandCoverClassPlugin`` stores the class codes as published (the 37 LCCS classes,
  ``lccs_class``), for display. Its pyramid must use ``mode`` resampling so coarse zoom levels
  show a real class rather than an average of codes.
- ``ESACCILandCoverFractionPlugin`` stores, for each of ten class groups, the fraction of each
  output cell the group covers, along a ``land_cover_class`` dimension. Cells are blocks of
  ``block`` x ``block`` native pixels (3 by default, about 1 km). Averaging a fraction layer
  over a district with the existing ``mean`` aggregation gives the share of the district under
  that group, so land cover goes through the same aggregation and export path as any other
  continuous variable.

The groups follow the IPCC land categories the CCI user guide maps LCCS classes onto, with
shrubland, sparse vegetation and bare areas kept apart because they matter in drylands.

Planetary Computer serves 1992-2020 and needs no account. The C3S continuation (2016-2022)
is on the CDS as ``satellite-land-cover`` if more recent years are needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "esa-cci-lc"
_FIRST_YEAR, _LAST_YEAR = 1992, 2020
_RESOLUTION = 1 / 360  # 10 arc-seconds, the native grid
_NO_DATA = 0

# Class group -> LCCS codes. Every code except 0 (no data) belongs to exactly one group.
CLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "cropland": (10, 11, 12, 20, 30, 40),
    "tree_cover": (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100),
    "shrubland": (120, 121, 122),
    "grassland": (110, 130),
    "sparse_vegetation": (140, 150, 151, 152, 153),
    "wetland": (160, 170, 180),
    "urban": (190,),
    "bare": (200, 201, 202),
    "water": (210,),
    "snow_ice": (220,),
}


def _years(start: str, end: str) -> list[str]:
    first = max(int(str(start)[:4]), _FIRST_YEAR)
    last = min(int(str(end)[:4]), _LAST_YEAR)
    return [str(year) for year in range(first, last + 1)]


def _load_classes(year: str, bbox: list[float], pad: float = 0.0) -> xr.DataArray:
    """The ``lccs_class`` raster for ``year`` on the native grid over ``bbox`` (+ ``pad`` degrees)."""
    import odc.stac
    import planetary_computer
    import pystac_client

    xmin, ymin, xmax, ymax = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    catalog = pystac_client.Client.open(_STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=[_COLLECTION], bbox=[xmin, ymin, xmax, ymax], datetime=year).items())
    if not items:
        raise RuntimeError(f"No {_COLLECTION} tiles for {year} over {bbox}")
    loaded = odc.stac.load(
        items,
        bands=["lccs_class"],
        crs="EPSG:4326",
        resolution=_RESOLUTION,
        bbox=(xmin, ymin, xmax, ymax),
        chunks={"x": 4096, "y": 4096},
        resampling="nearest",
    )
    classes = loaded["lccs_class"].isel(time=0, drop=True).compute()
    classes = classes.rename({"latitude": "y", "longitude": "x"}).drop_vars("spatial_ref", errors="ignore")
    classes.attrs = {}
    return classes


def _with_time(ds: xr.Dataset, year: str) -> xr.Dataset:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

    ds = ds.rio.write_crs("EPSG:4326")
    return ds.expand_dims({"t": [np.datetime64(f"{year}-01-01", "ns")]})


class ESACCILandCoverClassPlugin(BaseDatasetPlugin):
    """Yearly LCCS land cover class codes (uint8; 0 is no data)."""

    async def periods(self, start: str, end: str) -> list[str]:
        return _years(start, end)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        classes = _load_classes(period_id, bbox)
        classes = classes.where(classes != _NO_DATA).astype("float32")
        classes.attrs = {"units": "1", "flag_meanings": "LCCS class codes, see the template description"}
        return _with_time(classes.to_dataset(name="lccs_class"), period_id)


class ESACCILandCoverFractionPlugin(BaseDatasetPlugin):
    """Yearly fraction of each cell covered by each land cover class group.

    Template params:
        block: native pixels per output cell along each axis (default 3, about 1 km).
    """

    def __init__(self, block: int = 3, **_: Any) -> None:
        if int(block) < 1:
            raise ValueError(f"{type(self).__name__}: block must be a positive integer, got {block!r}")
        self.block = int(block)

    async def periods(self, start: str, end: str) -> list[str]:
        return _years(start, end)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        classes = _load_classes(period_id, bbox)
        fractions = class_fractions(classes, self.block)
        return _with_time(fractions.to_dataset(name="land_cover_fraction"), period_id)


def class_fractions(classes: xr.DataArray, block: int) -> xr.DataArray:
    """Fraction of each ``block`` x ``block`` cell covered by each class group.

    No-data pixels are left out of the denominator, so a cell that is half no data and half
    cropland is 1.0 cropland; a cell that is all no data is NaN in every group.
    """
    valid = classes != _NO_DATA
    layers = [classes.isin(codes) for codes in CLASS_GROUPS.values()]
    stacked = xr.concat(layers, dim="land_cover_class").assign_coords(land_cover_class=list(CLASS_GROUPS))
    coarsen = {"y": block, "x": block}
    counts = stacked.astype("float32").coarsen(coarsen, boundary="trim").reduce(np.sum)
    observed = valid.astype("float32").coarsen(coarsen, boundary="trim").reduce(np.sum)
    fractions = (counts / observed.where(observed > 0)).astype("float32")
    fractions.attrs = {"units": "1", "long_name": "fraction of cell covered by the land cover class group"}
    return fractions.transpose("land_cover_class", "y", "x")
