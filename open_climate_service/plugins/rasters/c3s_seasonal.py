"""C3S seasonal forecast anomalies, from the Copernicus Climate Data Store.

A forecast plugin on the same two axes as every other (see
:mod:`open_climate_service.shared.forecast`), but the first one whose lead is counted in
**months** rather than days: six lead months per run, one run per month.

**Anomalies rather than absolute values.** A seasonal forecast of absolute temperature at 1°
carries model bias that has to be removed before the number means anything locally. C3S publishes
the anomaly against each system's own 1993–2016 hindcast climatology, which is both the
bias-corrected quantity and the decision-relevant one: "1.5 °C warmer than normal for October"
is actionable in a way that "23.4 °C at 1° resolution" is not.

**Ensemble members become quantiles.** Real-time runs carry 51 members but the hindcasts only 25.
Quantiles are comparable across that change; a member index is not — a store keyed on member
number would quietly mean something different either side of 2017.

The upstream product is GRIB, and ECMWF states that CDS's NetCDF conversion is experimental and
not recommended for operational use, so this decodes GRIB via cfgrib. Read with
``time_dims=("forecastMonth", "time")`` the fields come back as
``(number, forecastMonth, latitude, longitude)`` with ``time`` a scalar issue time.

Note the resolution: 1° (~110 km) for every C3S seasonal product, so a country like Malawi is a
handful of cells. Sound for a national outlook, not for district maps.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.shared import forecast
from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_COLLECTION = "seasonal-postprocessed-single-levels"
_CDS_CATALOGUE_URL = "https://cds.climate.copernicus.eu/api"
# Six lead months, and lead 0 does not exist upstream: C3S numbers `forecastMonth` from 1, where
# 1 is the month the run was issued in. Normalised to 0-based here so the generic rule
# `valid = reference + lead x unit` holds for a seasonal cube exactly as it does for a daily one.
_UPSTREAM_FIRST_LEAD = 1
_MAX_LEAD_MONTHS = 6
# SEAS5.1. ECMWF also still publishes systems 4 and 5; the system number is part of the data's
# identity, not an implementation detail, so it stays an explicit template setting.
_DEFAULT_SYSTEM = "51"
_DEFAULT_CENTRE = "ecmwf"
# `monthly_mean` carries every ensemble member; `ensemble_mean` is the members already averaged,
# which throws away the spread this plugin exists to keep.
_PRODUCT_TYPE = "monthly_mean"
_FIRST_YEAR = 2017


class C3SSeasonalAnomalyPlugin(BaseDatasetPlugin):
    """One instance per variable. Streams one monthly initialisation per fetch."""

    max_concurrency = 1
    # One run is one commit: a six-month block for a country bbox is tiny, and a CDS request is
    # slow enough that losing a completed run to a later failure is the expensive outcome.
    commit_batch_size = 1
    # The append axis. Without this the framework appends along `t` and the store stops being
    # able to say which run a value came from.
    time_dim = forecast.REFERENCE_DIM

    def __init__(
        self,
        *,
        variable: str,
        originating_centre: str = _DEFAULT_CENTRE,
        system: str = _DEFAULT_SYSTEM,
        max_lead_months: int = _MAX_LEAD_MONTHS,
        quantiles: list[float] | None = None,
        unit_transform: str | None = None,
    ) -> None:
        if not 1 <= max_lead_months <= _MAX_LEAD_MONTHS:
            raise ValueError(f"max_lead_months must lie in 1..{_MAX_LEAD_MONTHS}, got {max_lead_months}")
        if quantiles is not None and not all(0.0 <= q <= 1.0 for q in quantiles):
            raise ValueError(f"Quantiles must lie in 0..1, got {quantiles!r}")
        self._variable = variable
        self._centre = originating_centre
        self._system = system
        self._max_lead_months = max_lead_months
        self._quantiles = quantiles
        self._unit_transform = unit_transform

    async def periods(self, start: str, end: str) -> list[str]:
        """Issue months in ``[start, end]`` as ``YYYY-MM``, oldest first.

        Clipped at both ends to what the CDS actually publishes. The upper bound matters:
        ``temporal_direction: future`` means a request without an end gets a horizon a year ahead,
        and the framework leaves it to the plugin to clip to its source (see
        ``ingestions.services._forecast_horizon``). Enumerating the calendar to that horizon would
        ask for twelve runs that do not exist yet and fail twelve times over.

        Availability comes from the catalogue rather than a rule about publication dates, because
        the rule differs per centre — the 6th of the month for ECMWF, the 10th for the others.
        """
        first = _as_month(start)
        last = _as_month(end)
        earliest = date(_FIRST_YEAR, 1, 1)
        if first < earliest:
            logger.info("C3S seasonal forecasts start at %s; clamping requested start %s", earliest, first)
            first = earliest
        available = _availability_cutoff()
        if last > available:
            logger.info("Latest published C3S seasonal run is %s; clipping requested end %s", available, last)
            last = available
        months = []
        cursor = first
        while cursor <= last:
            months.append(f"{cursor.year:04d}-{cursor.month:02d}")
            cursor = _next_month(cursor)
        return months

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """One initialisation as ``(reference_time=1, lead_time=N[, quantile], y, x)``."""
        month = _as_month(period_id)
        xmin, ymin, xmax, ymax = map(float, bbox)
        leads = [str(n) for n in range(_UPSTREAM_FIRST_LEAD, _UPSTREAM_FIRST_LEAD + self._max_lead_months)]
        request = {
            "originating_centre": self._centre,
            "system": self._system,
            "variable": [self._variable],
            "product_type": [_PRODUCT_TYPE],
            "year": [f"{month.year:04d}"],
            "month": [f"{month.month:02d}"],
            "leadtime_month": leads,
            # CDS takes the bbox as north, west, south, east.
            "area": [ymax, xmin, ymin, xmax],
            "data_format": "grib",
        }
        logger.info("Requesting %s %s/%s %s for %s", _COLLECTION, self._centre, self._system, self._variable, period_id)
        with tempfile.TemporaryDirectory() as workdir:
            target = str(Path(workdir) / f"{self._variable}-{period_id}.grib")
            remote = _cds_client().submit(_COLLECTION, request)
            remote.download(target)
            # Loaded eagerly inside the temporary directory: cfgrib reads lazily off the file and
            # builds an index beside it, both of which vanish with the directory.
            run = xr.open_dataset(
                target,
                engine="cfgrib",
                backend_kwargs={"time_dims": ("forecastMonth", "time"), "indexpath": ""},
            ).load()

        # Read before reducing: `quantile` drops the scalar `time` coordinate that carries it.
        issued = np.datetime64(np.asarray(run["time"].values).item(), "ns").astype("datetime64[M]")
        run = self._reduce_members(run)
        return self._as_forecast_cube(run, issued)

    def _reduce_members(self, run: xr.Dataset) -> xr.Dataset:
        """Collapse the ensemble to a mean, or to the requested quantiles."""
        if "number" not in run.dims:
            return run
        if self._quantiles is None:
            return run.mean("number", keep_attrs=True)
        # Named `quantile` to match xclim, so ensemble statistics computed later land on the
        # same axis rather than a parallel one.
        return run.quantile(self._quantiles, dim="number", keep_attrs=True)

    def _as_forecast_cube(self, run: xr.Dataset, issued: np.datetime64) -> xr.Dataset:
        """Rename to the published axes and declare the lead unit, issue time and valid times."""
        renamed = run.rename({"latitude": "y", "longitude": "x", "forecastMonth": forecast.LEAD_DIM})
        # `surface`/`step`/`valid_time` are cfgrib's own scalars — dropped rather than carried,
        # because `valid_time` is in `get_time_dim`'s lookup list and would make the cube's
        # temporal axis ambiguous.
        renamed = renamed.drop_vars(["time", "surface", "step", "valid_time", "number"], errors="ignore")

        leads = forecast.lead_values(renamed) - _UPSTREAM_FIRST_LEAD
        renamed = renamed.assign_coords({forecast.LEAD_DIM: leads.astype("int32")})
        renamed[forecast.LEAD_DIM].attrs["units"] = "months"
        reference = np.asarray([issued], dtype="datetime64[ns]")
        renamed = renamed.expand_dims({forecast.REFERENCE_DIM: reference})
        renamed = renamed.assign_coords(
            {
                forecast.VALID_COORD: (
                    (forecast.REFERENCE_DIM, forecast.LEAD_DIM),
                    forecast.valid_times_for(reference, leads, "month"),
                )
            }
        )
        for name, standard in forecast.CF_STANDARD_NAMES.items():
            if name in renamed.coords:
                renamed[name].attrs["standard_name"] = standard
        renamed = self._apply_unit_transform(renamed)
        ordered = renamed.transpose(forecast.REFERENCE_DIM, forecast.LEAD_DIM, ..., "y", "x", missing_dims="ignore")
        return self._store_ready(ordered)

    def _apply_unit_transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert the stored values to the template's units, as the ERA5-Land plugins do.

        Converted at ingest rather than on read, so the store holds the units the catalogue
        advertises and every consumer — viewer, openEO export, DHIS2 push — agrees without each
        having to know the source's.

        Note ``kelvin_difference_to_celsius`` rather than ``kelvin_to_celsius`` for temperature:
        these are anomalies, so the values are already degrees Celsius of difference and
        subtracting 273.15 would turn +1.4 into −271.75.
        """
        if self._unit_transform in (None, "", "none"):
            return ds
        from open_climate_service.transforms import unit_conversion

        known = {
            "kelvin_difference_to_celsius": unit_conversion.kelvin_difference_to_celsius,
            "metres_per_second_to_mm_per_day": unit_conversion.metres_per_second_to_mm_per_day,
            "kelvin_to_celsius": unit_conversion.kelvin_to_celsius,
            "metres_to_mm": unit_conversion.metres_to_mm,
        }
        convert = known.get(str(self._unit_transform))
        if convert is None:
            raise ValueError(f"Unknown unit_transform {self._unit_transform!r}; expected one of {sorted(known)}")
        # Keyed on the store's variable name, which for a GRIB source is the short name cfgrib
        # produced (`t2a`, `tpara`) rather than the long CDS request name.
        for name in list(ds.data_vars):
            ds = convert(ds, {"variable": name})
        return ds

    @staticmethod
    def _store_ready(ds: xr.Dataset) -> xr.Dataset:
        """One chunk per issue time, which is how the store grows.

        Unwriteable attributes — cfgrib attaches plenty — are dropped at the write boundary for
        every plugin alike, in ``shared.cf.drop_unserializable_attrs``.
        """
        chunks: dict[str, int] = {forecast.REFERENCE_DIM: 1}
        for dim in (forecast.LEAD_DIM, "quantile", "y", "x"):
            if dim in ds.sizes:
                chunks[dim] = int(ds.sizes[dim])
        rechunked: xr.Dataset = ds.chunk(chunks)
        return rechunked


def _cds_client() -> Any:
    """The CDS client, resolving credentials the way the ERA5-Land plugins do.

    Imported lazily so that merely loading the dataset registry does not require the CDS
    dependency, and constructed per request so a rotated key is picked up without a restart.
    """
    from ecmwf.datastores import Client

    return Client()


def _availability_cutoff() -> date:
    """The latest issue month the CDS has published, from the collection's declared end.

    The same catalogue lookup the ERA5-Land plugins use for their cutoffs, so "what exists" has
    one source of truth rather than a per-plugin guess. Raises rather than assuming a bound: a
    wrong guess either silently truncates the archive or asks for runs that do not exist.
    """
    from ecmwf.datastores import Client

    collection = Client(url=_CDS_CATALOGUE_URL, key="").get_collection(_COLLECTION)
    end = collection.end_datetime
    if end is None:
        raise RuntimeError(f"CDS collection '{_COLLECTION}' returned no end_datetime")
    return date(end.year, end.month, 1)


def _as_month(value: str) -> date:
    """First of the month for a ``YYYY-MM`` or longer ISO period string."""
    return date(int(value[:4]), int(value[5:7]), 1)


def _next_month(value: date) -> date:
    return date(value.year + 1, 1, 1) if value.month == 12 else date(value.year, value.month + 1, 1)
