"""ESA CCI / C3S land cover (300 m, yearly) from the Copernicus Climate Data Store (CDS).

Like all categorical datasets, the pyramid used to visualize this dataset uses ``mode``
resampling so coarse zoom levels show a real class rather than an average of codes.

The CDS serves the maps as ``satellite-land-cover``: version 2.0.7 for 1992-2015 and the C3S
continuation, version 2.1.1, from 2016. Each year is requested for the extent only. Requests need
a CDS key (see the README) and the dataset's licence accepted once in the browser.
The plugin submits each year as a separate request to avoid CDS request size limit, and supports
multiple concurrent requests to CDS. The CDS either processes these in parallel or queues them;
either way the downloads can take up to several minutes per year.
"""

from __future__ import annotations

import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from ecmwf.datastores import Client as _CdsClient

from open_climate_service.streaming import BaseDatasetPlugin

_DATASET = "satellite-land-cover"
_NO_DATA = 0


def _available_years() -> list[str]:
    constraints = _CdsClient().apply_constraints(_DATASET, {})
    years = constraints.get("year")
    if not isinstance(years, list) or not all(isinstance(year, str) for year in years):
        raise RuntimeError(f"CDS returned no valid year options for collection {_DATASET!r}: {years!r}")
    return sorted(set(years), key=int)


def _years(start: str, end: str, available_years: list[str]) -> list[str]:
    first = int(str(start)[:4])
    last = int(str(end)[:4])
    return [year for year in available_years if first <= int(year) <= last]


def _cds_version(year: int) -> str:
    return "v2_0_7cds" if year <= 2015 else "v2_1_1"


def _load_year(year: str, bbox: list[float]) -> xr.DataArray:
    """The ``lccs_class`` raster for ``year`` on the native grid over ``bbox``."""
    xmin, ymin, xmax, ymax = bbox
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

        # The CDS delivers a zip holding one netCDF file
        with zipfile.ZipFile(target) as archive:
            names = [name for name in archive.namelist() if name.endswith(".nc")]
            if not names:
                raise RuntimeError(f"No netCDF file in the CDS land cover download for {year}: {archive.namelist()}")
            path = Path(archive.extract(names[0], tmp))

        # Open, load, and close the ds before the temporary directory is removed
        with xr.open_dataset(path) as ds:
            classes = ds.load()["lccs_class"]

    # Drop only the yearly time axis: an extent one pixel wide must keep both spatial axes.
    if "time" in classes.dims:
        classes = classes.isel(time=0, drop=True)

    # Convert both NaN values and _NO_DATA to NaN
    classes = classes.where(classes != _NO_DATA).astype("float32")

    # Misc cleaning
    classes = classes.rename({"lat": "y", "lon": "x"}).sortby("y", ascending=False)
    classes = classes.drop_vars([name for name in classes.coords if name not in ("x", "y")])
    classes.attrs = {}

    return classes


def _with_time(ds: xr.Dataset, year: str) -> xr.Dataset:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

    ds = ds.rio.write_crs("EPSG:4326")
    return ds.expand_dims({"t": [np.datetime64(f"{year}-01-01", "ns")]})


class ESACCILandCoverClassPlugin(BaseDatasetPlugin):
    """Yearly LCCS land cover class codes as float32, with 0/no-data cells set to NaN."""

    max_concurrency = 5

    async def periods(self, start: str, end: str) -> list[str]:
        available_years = await asyncio.to_thread(_available_years)
        return _years(start, end, available_years)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        # Fetch the land cover class codes for a single year and bounding box
        classes = _load_year(period_id, bbox)

        # Set ds attributes
        # NOTE: flag_meanings attr are not set here since they are later set by the template definition during ingestion
        classes.attrs = {"units": "1"}

        # Return with time dimension added
        return _with_time(classes.to_dataset(name="lccs_class"), period_id)
