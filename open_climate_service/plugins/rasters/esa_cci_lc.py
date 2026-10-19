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
continuation, version 2.1.1, from 2016. Requested years and applicable versions are sent in one
request for the extent only. Requests need a CDS key (see the README) and the dataset's licence
accepted once in the browser; they queue on the CDS.
"""

from __future__ import annotations

import re
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


def _year_from_filename(path: Path, requested_years: set[str]) -> str:
    matches: set[str] = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", path.name)) & requested_years
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot determine the year for CDS land cover file {path.name!r}; "
            f"expected one of {sorted(requested_years)} in its name."
        )
    return matches.pop()


def _load_years(years: list[str], bbox: list[float], pad: float = 0.0) -> xr.DataArray:
    """Load ``lccs_class`` for each requested year over ``bbox`` (+ ``pad`` degrees)."""
    xmin, ymin, xmax, ymax = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    with tempfile.TemporaryDirectory() as tmp:
        client = _CdsClient()
        client.check_authentication()
        target = Path(tmp) / "land_cover"
        client.submit(
            _DATASET,
            {
                "variable": "all",
                "year": years,
                "version": list(set(_cds_version(int(year)) for year in years)),
                "area": [ymax, xmin, ymin, xmax],
            },
        ).download(str(target))

        paths_by_year: dict[str, Path] = {}
        with zipfile.ZipFile(target) as archive:
            names = [name for name in archive.namelist() if name.endswith(".nc")]
            if not names:
                raise RuntimeError(f"No netCDF file in the CDS land cover download: {archive.namelist()}")
            for name in names:
                year = _year_from_filename(Path(name), set(years))
                if year in paths_by_year:
                    raise RuntimeError(f"The CDS land cover download contains more than one file for {year}.")
                paths_by_year[year] = Path(archive.extract(name, tmp))

        missing_years = sorted(set(years) - paths_by_year.keys())
        if missing_years:
            raise RuntimeError(f"The CDS land cover download is missing years: {', '.join(missing_years)}.")

        with xr.open_mfdataset(
            [paths_by_year[year] for year in years],
            combine="nested",
            concat_dim="year",
        ) as ds:
            classes = ds["lccs_class"]
            if "time" in classes.dims:
                classes = classes.isel(time=0, drop=True)
            classes = classes.assign_coords(year=years).load()

    # A declared fill value is decoded to NaN; make it the class code for no data, so the fractions
    # leave it out of the denominator instead of counting it as observed and unclassified.
    classes = classes.fillna(_NO_DATA).astype("uint8")
    classes = classes.rename({"lat": "y", "lon": "x"}).sortby("y", ascending=False)
    classes = classes.drop_vars([name for name in classes.coords if name not in ("x", "y", "year")])
    classes.attrs = {}
    return classes


class _ESACCILandCoverPlugin(BaseDatasetPlugin):
    max_concurrency = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._requested_years: list[str] = []
        self._classes_cache: xr.DataArray | None = None
        self._cache_key: tuple[tuple[str, ...], tuple[float, ...]] | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        years = _years(start, end)
        if years != self._requested_years:
            self._requested_years = years
            self._classes_cache = None
            self._cache_key = None
        return years

    def _classes_for_period(self, period_id: str, bbox: list[float]) -> xr.DataArray:
        years = self._requested_years or [period_id]
        cache_key = (tuple(years), tuple(bbox))
        if self._classes_cache is None or self._cache_key != cache_key:
            self._classes_cache = _load_years(years, bbox)
            self._cache_key = cache_key
        return self._classes_cache.sel(year=period_id, drop=True).copy(deep=True)


def _with_time(ds: xr.Dataset, year: str) -> xr.Dataset:
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

    ds = ds.rio.write_crs("EPSG:4326")
    return ds.expand_dims({"t": [np.datetime64(f"{year}-01-01", "ns")]})


class ESACCILandCoverClassPlugin(_ESACCILandCoverPlugin):
    """Yearly LCCS land cover class codes (uint8; 0 is no data)."""

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        classes = self._classes_for_period(period_id, bbox)
        classes = classes.where(classes != _NO_DATA).astype("float32")
        classes.attrs = {"units": "1", "flag_meanings": "LCCS class codes, see the template description"}
        return _with_time(classes.to_dataset(name="lccs_class"), period_id)


class ESACCILandCoverFractionPlugin(_ESACCILandCoverPlugin):
    """Yearly fraction of each cell covered by each land cover class group.

    Template params:
        block: native pixels per output cell along each axis (default 3, about 1 km).
    """

    def __init__(self, block: int = 3, **_: Any) -> None:
        if int(block) < 1:
            raise ValueError(f"{type(self).__name__}: block must be a positive integer, got {block!r}")
        super().__init__(**_)
        self.block = int(block)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        classes = self._classes_for_period(period_id, bbox)
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
