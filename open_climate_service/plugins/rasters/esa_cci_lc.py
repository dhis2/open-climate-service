"""ESA CCI / C3S land cover (300 m, yearly) from the Copernicus Climate Data Store (CDS).

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

The CDS serves the maps as ``satellite-land-cover``: version 2.0.7 for 1992-2015 and the C3S
continuation, version 2.1.1, from 2016. Each year is requested for the extent only. Requests need
a CDS key (see the README) and the dataset's licence accepted once in the browser; they queue
on the CDS, so one year can take several minutes.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _CdsClient
from open_climate_service.streaming import BaseDatasetPlugin

_DATASET = "satellite-land-cover"
_FIRST_YEAR, _LAST_YEAR = 1992, 2022
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


def _cds_version(year: int) -> str:
    return "v2_0_7cds" if year <= 2015 else "v2_1_1"


def _load_year(year: str, bbox: list[float], pad: float = 0.0) -> xr.DataArray:
    """The ``lccs_class`` raster for ``year`` on the native grid over ``bbox`` (+ ``pad`` degrees)."""
    xmin, ymin, xmax, ymax = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    request = {
        "variable": "all",
        "year": [year],
        "version": [_cds_version(int(year))],
        "area": [ymax, xmin, ymin, xmax],
    }
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "land_cover"
        client = _CdsClient()
        client.check_authentication()
        client.submit(_DATASET, request).download(str(target))
        # The CDS delivers a zip holding one netCDF file, or the netCDF file itself.
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as archive:
                names = [name for name in archive.namelist() if name.endswith(".nc")]
                if not names:
                    raise RuntimeError(
                        f"No netCDF file in the CDS land cover download for {year}: {archive.namelist()}"
                    )
                path = Path(archive.extract(names[0], tmp))
        else:
            path = target
        ds = xr.open_dataset(path)
    classes = ds["lccs_class"]
    # Drop only the yearly time axis: an extent one pixel wide must keep both spatial axes.
    if "time" in classes.dims:
        classes = classes.isel(time=0, drop=True)
    # A declared fill value is decoded to NaN; make it the class code for no data, so the fractions
    # leave it out of the denominator instead of counting it as observed and unclassified.
    classes = classes.fillna(_NO_DATA).astype("uint8")
    classes = classes.rename({"lat": "y", "lon": "x"}).sortby("y", ascending=False)
    classes = classes.drop_vars([name for name in classes.coords if name not in ("x", "y")])
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
        classes = _load_year(period_id, bbox)
        classes = classes.where(classes != _NO_DATA).astype("float32")
        classes.attrs = {"units": "1", "flag_meanings": "LCCS class codes, see the template description"}
        return _with_time(classes.to_dataset(name="lccs_class"), period_id)
