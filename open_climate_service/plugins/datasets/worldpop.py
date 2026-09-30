"""WorldPop plugin for per-period streaming ingest.

Fetches WorldPop's yearly population GeoTIFFs from the public ``data.worldpop.org``
hub (no auth). The hub advertises but does not honour HTTP Range requests, so each
GeoTIFF is downloaded to a local cache and opened from disk rather than read lazily
over ``/vsicurl`` (see ``_open_worldpop_raster``); the grid is inferred from the first
fetched year.

Two plugins, both Global2 under stable dataset ids with the hub revision tracked as
release metadata (currently ``R2025A``):

- ``WorldPopYearlyPlugin`` — population counts (hub category id=3), shaped to
  ``(t, y, x)`` via ``normalize_period``. The URL builder also knows the 1 km
  (``1km_ua``) and unconstrained layouts, but only the 100 m constrained template
  ships today.
- ``WorldPopAgeSexYearlyPlugin`` — age/sex structures (hub category id=8): ~40
  per-(sex, age) rasters per country-year, combined into one ``population`` cube over
  ``sex`` and ``age_group`` dimensions (built directly, not via ``normalize_period``),
  100 m per-country only.

Global 1 km mosaics are not yet supported (they are not COGs — see
https://github.com/dhis2/open-climate-service/issues/269).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service import config as api_config
from open_climate_service.streaming import BaseDatasetPlugin, normalize_period

_WORLDPOP_NODATA = -99999.0
_HUB = "https://data.worldpop.org/GIS"
# Global2 covers 2015–2030; clamp requested periods so we never build URLs for
# years the hub doesn't publish (out-of-range requests would 404 at ingest).
_GLOBAL2_FIRST_YEAR, _GLOBAL2_LAST_YEAR = 2015, 2030


def _global2_years(start: str, end: str) -> list[str]:
    start_year = max(int(str(start)[:4]), _GLOBAL2_FIRST_YEAR)
    end_year = min(int(str(end)[:4]), _GLOBAL2_LAST_YEAR)
    if start_year > end_year:
        return []
    return [str(year) for year in range(start_year, end_year + 1)]


def _worldpop_cache_dir() -> Path:
    """Directory for cached WorldPop GeoTIFF downloads.

    Lives under the instance ``data_dir`` (``<data_dir>/cache/worldpop``) so the cache
    shares the instance's lifecycle, mirroring how downloads/jobs are resolved. Falls
    back to an XDG cache path when no config/``data_dir`` is set (e.g. CI).
    """
    data_dir = api_config.get_data_dir()
    if data_dir is not None:
        base = data_dir
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "climate-service"
    return base / "cache" / "worldpop"


def _open_worldpop_raster(url: str, **open_kwargs: Any) -> xr.DataArray:
    """Open a WorldPop GeoTIFF, downloading it to a local cache first.

    WorldPop's hub (``data.worldpop.org``) advertises ``Accept-Ranges: bytes`` but
    does not honour Range requests — a ranged GET returns the full ``200`` response —
    so GDAL's ``/vsicurl`` lazy reads abort with "Range downloading not supported by
    this server!". The per-country GeoTIFFs are small (a few MB), so fetch each whole
    file once to a local cache and open it from disk, where seeking works normally.
    """
    import rioxarray

    cache = _worldpop_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / url.rsplit("/", 1)[-1]
    if not target.exists():
        import httpx
        import portalocker

        # Hold an exclusive per-entry lock across the download so concurrent
        # ingestions (or worker processes) sharing this cache can't race on the
        # same file and corrupt it. Re-check existence inside the lock: a holder
        # that finished while we waited means there's nothing left to do. The .lock
        # sentinel is left on disk on purpose — unlinking it would reopen the race.
        lock_path = target.with_name(target.name + ".lock")
        with open(lock_path, "w") as lock_fh:
            portalocker.lock(lock_fh, portalocker.LOCK_EX)
            try:
                if not target.exists():
                    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
                        response.raise_for_status()
                        tmp = target.with_name(target.name + ".part")
                        with tmp.open("wb") as handle:
                            for chunk in response.iter_bytes():
                                handle.write(chunk)
                        tmp.rename(target)
            finally:
                portalocker.unlock(lock_fh)
    da = rioxarray.open_rasterio(target, masked=True, **open_kwargs)
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected DataArray from WorldPop raster read, got {type(da).__name__}")
    return da


# WorldPop age/sex 5-year bands (lower bound of each band), as the filename tokens.
_AGE_BANDS = (
    "00",
    "01",
    "05",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
    "80",
    "85",
    "90",
)
# Sex -> WorldPop filename token. Sex becomes a cube dimension on the single
# ``population`` variable; the t_* / T_F / T_M totals are redundant sums of these,
# so we skip them.
_SEX_TOKENS = {"female": "f", "male": "m"}


@dataclass(frozen=True)
class _WorldPopVariant:
    product: str
    output_variable: str


class WorldPopYearlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for yearly WorldPop country population rasters.

    WorldPop reissues the Global2 product under a new revision (currently
    ``R2025A``). The revision is a release identity, not part of the stable dataset
    id: changing it rematerializes the existing managed dataset and records which
    logical release it contains. Resolution and constrained/unconstrained flavour
    remain dataset identity concerns.

    Args:
        version: WorldPop release variant. Only ``global2`` (2015–2030) is supported.
        revision: Hub revision tag, e.g. ``R2025A``.
        resolution: ``100m`` (per-country constrained) or ``1km`` (per-country
            ``1km_ua``).
        constrained: Use the constrained (settlement-masked) product. Defaults to
            ``True``; ``False`` selects the unconstrained product.
        product: WorldPop product selector. Currently only ``total`` (population
            counts, id=3) is supported; the age/sex structures product (id=8) is
            handled separately.
        variable: Output variable name written into the managed dataset.

    The default concurrency (1) and commit cadence (1) suit the one-request-per-year
    fetch, so no overrides are needed. The grid is inferred from the first fetched year.
    """

    def __init__(
        self,
        version: str = "global2",
        revision: str = "R2025A",
        resolution: str = "100m",
        constrained: bool = True,
        product: str = "total",
        variable: str = "pop_total",
        **_: object,
    ) -> None:
        if version != "global2":
            raise ValueError(f"Unsupported WorldPop version '{version}'; only 'global2' is supported")
        self.version = version
        self.revision = revision
        self.resolution = resolution
        self.constrained = constrained
        self.variant = _resolve_variant(product=product, variable=variable)

    async def periods(self, start: str, end: str) -> list[str]:
        return _global2_years(start, end)

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        """Read one year's country GeoTIFF straight from the WorldPop hub.

        A regular (blocking) method — the framework runs it in a worker thread.
        """
        url = _population_url(
            int(period_id), _required_country_code(params), self.revision, self.resolution, self.constrained
        )
        da = _open_worldpop_raster(url, chunks="auto")
        # Mask the WorldPop sentinel (-99999) so the Zarr store uses NaN as the fill
        # value for unpopulated / ocean cells. No bbox clip: WorldPop publishes one
        # GeoTIFF per country, so we ingest the whole country rather than crop it to
        # the instance extent.
        return normalize_period(da, variable=self.variant.output_variable, period=period_id, nodata=_WORLDPOP_NODATA)


class WorldPopAgeSexYearlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for WorldPop Global2 age/sex population structures (hub id=8).

    For each country-year WorldPop publishes one GeoTIFF per (sex, 5-year age band).
    This plugin combines them into a single ``population`` variable over a ``sex``
    dimension (``female`` / ``male``) and an ordinal ``age_group`` dimension (the lower
    bound of the 5-year band: 0, 1, 5, 10, … 90) — so the cube is ``(t, sex, age_group,
    y, x)``. Population is the quantity; sex and age are both disaggregation dimensions.
    Only 100 m per-country is supported.

    Memory: each band is opened **lazily** (dask-chunked) from a local cache, so the
    ~40 rasters never all sit in memory — the orchestrator streams the combined cube to
    Zarr one year at a time. No bbox clip: the per-country rasters are ingested whole.
    """

    def __init__(
        self, revision: str = "R2025A", resolution: str = "100m", constrained: bool = True, **_: object
    ) -> None:
        if resolution != "100m":
            raise ValueError(f"WorldPop age/sex: only resolution '100m' is supported, got {resolution!r}")
        self.revision = revision
        self.resolution = resolution
        self.constrained = constrained

    async def periods(self, start: str, end: str) -> list[str]:
        return _global2_years(start, end)

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        """Combine the per-(sex, age) GeoTIFFs into one lazy ``(t, sex, age_group, y, x)`` cube.

        A regular (blocking) method — the framework runs it in a worker thread.

        ``_open_worldpop_raster`` imports rioxarray, registering the ``.rio`` accessor
        used by ``write_crs`` below, so no separate import is needed here.
        """
        country_code = _required_country_code(params)
        year = int(period_id)
        ages = [int(a) for a in _AGE_BANDS]

        # WorldPop publishes every (sex, age) band for a released country-year, so we
        # fetch them all. A missing band (a 404 from a partial/unreleased year) raises
        # and aborts the whole year rather than producing a cube with a hole.
        sex_cubes: list[xr.DataArray] = []
        for sex_token in _SEX_TOKENS.values():
            bands: list[xr.DataArray] = []
            for age in _AGE_BANDS:
                url = _agesex_url(year, country_code, self.revision, sex_token, age, self.constrained)
                da = _open_worldpop_raster(url, chunks="auto")  # lazy — fetched to cache, opened from disk
                bands.append(da.squeeze("band", drop=True))
            sex_cubes.append(xr.concat(bands, dim="age_group").assign_coords(age_group=ages))

        population = xr.concat(sex_cubes, dim="sex").assign_coords(sex=list(_SEX_TOKENS))
        ds = population.to_dataset(name="population").rio.write_crs("EPSG:4326")
        ds = ds.where(ds != _WORLDPOP_NODATA)  # mask the sentinel (lazy)
        return ds.expand_dims(t=[np.datetime64(str(year))])  # type: ignore[no-any-return]


def _resolve_variant(*, product: str, variable: str) -> _WorldPopVariant:
    if product == "total":
        return _WorldPopVariant(product="total", output_variable=variable)
    raise ValueError(f"Unsupported WorldPop product '{product}'; only 'total' (population counts) is supported")


def _required_country_code(params: dict[str, Any]) -> str:
    country_code = params.get("country_code")
    if not isinstance(country_code, str) or not country_code:
        raise ValueError("WorldPop streaming ingest requires country_code in plugin params")
    return country_code


def _population_url(year: int, country_code: str, revision: str, resolution: str, constrained: bool) -> str:
    """Build the public WorldPop hub population-counts GeoTIFF URL for one country-year.

    Verified Global2 layout (``data.worldpop.org/GIS/Population/Global_2015_2030``):
    - 100 m: ``{rev}/{year}/{CC}/v1/100m/{constrained}/{cc}_pop_{year}_{CN}_100m_{rev}_v1.tif``
    - 1 km:  ``{rev}/{year}/{CC}/v1/1km_ua/{constrained}/{cc}_pop_{year}_{CN}_1km_{rev}_UA_v1.tif``
    """
    cc_lower, cc_upper = country_code.lower(), country_code.upper()
    sub = "constrained" if constrained else "unconstrained"
    cn = "CN" if constrained else "UC"
    base = f"{_HUB}/Population/Global_2015_2030/{revision}/{year}/{cc_upper}/v1"
    if resolution == "100m":
        return f"{base}/100m/{sub}/{cc_lower}_pop_{year}_{cn}_100m_{revision}_v1.tif"
    if resolution == "1km":
        return f"{base}/1km_ua/{sub}/{cc_lower}_pop_{year}_{cn}_1km_{revision}_UA_v1.tif"
    raise ValueError(f"Unsupported WorldPop resolution '{resolution}'; expected '100m' or '1km'")


def _agesex_url(year: int, country_code: str, revision: str, sex: str, age: str, constrained: bool) -> str:
    """Build the WorldPop age/sex 100 m GeoTIFF URL for one country-year-sex-age band.

    Verified Global2 layout (``data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030``):
    ``{rev}/{year}/{CC}/v1/100m/{constrained}/{cc}_{sex}_{age}_{year}_{CN}_100m_{rev}_v1.tif``
    """
    cc_lower, cc_upper = country_code.lower(), country_code.upper()
    sub = "constrained" if constrained else "unconstrained"
    cn = "CN" if constrained else "UC"
    return (
        f"{_HUB}/AgeSex_structures/Global_2015_2030/{revision}/{year}/{cc_upper}/v1/100m/{sub}/"
        f"{cc_lower}_{sex}_{age}_{year}_{cn}_100m_{revision}_v1.tif"
    )
