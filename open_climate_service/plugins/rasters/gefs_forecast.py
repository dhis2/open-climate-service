"""NOAA GEFS 35-day ensemble forecast, from dynamical.org's public Zarr store.

A forecast plugin, so it works the way :mod:`open_climate_service.shared.forecast` describes:
``periods()`` enumerates *issue times* and ``fetch_period()`` returns that one run's whole lead
block, which the framework appends along ``reference_time``. Every refresh therefore adds a run
rather than overwriting the previous one.

Chosen as the reference forecast source because it needs no credentials, is already Zarr (so no
GRIB decode), reaches 35 days, and carries an archive back to 2020 — enough history to exercise
verification rather than only "what is the outlook". ECMWF's open data would be the European
equivalent but caps at 15 days, with extended-range and seasonal products explicitly excluded.

The upstream store is ``(init_time, ensemble_member, lead_time, latitude, longitude)`` — already
the forecast shape, so nothing needs reshaping. Two reductions do happen:

* **Ensemble members** collapse to the mean, or to a ``quantile`` axis when the template asks
  for one. A mean discards the spread that is the point of an ensemble, so quantiles are the
  better product wherever a consumer can use them.
* **Lead time** resamples to whole days. Upstream leads are mixed 3-hourly and 6-hourly, which
  is neither a lead in days nor something DHIS2 periods can express.

Rate variables then need integrating: upstream precipitation is an average ``kg m-2 s-1`` over
each step, so a daily total in millimetres is the day's mean rate times the length of a day.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.shared import forecast
from open_climate_service.streaming import BaseDatasetPlugin, bbox_slices

logger = logging.getLogger(__name__)

_STORE_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"
_MAX_LEAD_DAYS = 35
# Upstream leads run 0h to 840h at 3- then 6-hourly spacing, so a "day" is a group of steps
# rather than a single one.
_RESAMPLE = {"mean", "min", "max", "sum"}
_SECONDS_PER_DAY = 86400
# Accumulating a rate means integrating it over the day, so only a mean is a valid starting
# point. Summing per-second rates would report the daily mean multiplied by however many steps
# the day happened to hold — a number with no unit.
_RATE_SUFFIX = "s-1"
# Zarr accepts uniform chunks with a smaller final one, and nothing else. Grouping by forecast
# day and slicing a bbox out of the upstream chunking both produce ragged dask chunks, so the
# cube is rechunked before it is handed back to the writer.
_SPATIAL_CHUNK = 256


class GefsForecastPlugin(BaseDatasetPlugin):
    """One instance per variable. Streams one initialisation per fetch."""

    # Deliberately serial. zarr already fetches a shard's inner chunks concurrently within one
    # read, so fetching several initialisations at once only contends for the same connection:
    # measured over four initialisations, four workers took 73s each against 13.8s serial. What
    # remains is per-request latency (~1.3s per ranged GET), not bandwidth, so more workers
    # cannot help until that changes.
    max_concurrency = 1
    # One run is one commit: a 35-day block for a country bbox is small, and batching would
    # only delay the point at which a partially fetched archive becomes resumable.
    commit_batch_size = 1
    # The append axis. Without this the framework would append along `t` and the store would
    # lose the distinction between when a forecast was made and what it is about.
    time_dim = forecast.REFERENCE_DIM

    def __init__(
        self,
        *,
        variable: str,
        store_url: str = _STORE_URL,
        max_lead_days: int = _MAX_LEAD_DAYS,
        resample: str = "mean",
        accumulate: bool = False,
        accumulated_units: str = "mm/d",
        quantiles: list[float] | None = None,
    ) -> None:
        if resample not in _RESAMPLE:
            raise ValueError(f"Unknown resample {resample!r}; expected one of {sorted(_RESAMPLE)}")
        if quantiles is not None and not all(0.0 <= q <= 1.0 for q in quantiles):
            raise ValueError(f"Quantiles must lie in 0..1, got {quantiles!r}")
        if accumulate and resample != "mean":
            raise ValueError(f"accumulate integrates the daily mean rate, so resample must be 'mean', not {resample!r}")
        self._variable = variable
        self._store_url = store_url
        self._max_lead_days = max_lead_days
        self._resample = resample
        self._accumulate = accumulate
        self._accumulated_units = accumulated_units
        self._quantiles = quantiles
        self._ds: xr.Dataset | None = None

    def _open(self) -> xr.Dataset:
        """Open the remote store once and reuse it.

        Only the store handle is cached, never a derived dataset: a shared `xr.Dataset` that a
        caller may close poisons the cache for every later fetch.
        """
        if self._ds is None:
            logger.info("Opening GEFS forecast store %s", self._store_url)
            # `chunks=None` — no dask. Still lazy: xarray defers to zarr's own indexing, which
            # is what performs the partial shard reads this store needs.
            #
            # `chunks={}` would adopt the store's chunking, which sounds right and is a trap
            # here. The store is sharded: the shard is (1, 31, 192, 374, 368) but `array.chunks`
            # reports the *inner* chunk (1, 31, 64, 17, 16), so dask graphs the whole
            # (2148, 31, 181, 721, 1440) variable at inner-chunk granularity — 24.9 million
            # chunks — before any selection narrows it. Measured for one initialisation over a
            # country bbox: 24.4s and 5.9 GB peak RSS with `chunks={}`, against 13.8s and
            # 328 MB with `chunks=None`, for an identical result. The cost was graph
            # construction, not transfer, which is why asking for a single lead step cost the
            # same as asking for all 181.
            #
            # Chunking at the shard instead is not the fix: dask chunks that span shards fail
            # with a Blosc "Stored and computed checksum do not match".
            self._ds = xr.open_zarr(
                self._store_url,
                decode_timedelta=True,
                # `None` is xarray's documented "do not use dask" value; its stub types the
                # parameter as `str`, so the annotation is narrower than the API.
                chunks=None,  # pyright: ignore[reportArgumentType]
            )
        return self._ds

    async def periods(self, start: str, end: str) -> list[str]:
        """Issue times available in ``[start, end]``, newest last.

        The upstream archive runs to thousands of initialisations, so a caller asking for "the
        current forecast" wants a narrow range — a blank start resolves to now, per
        `temporal_direction: future`.
        """
        ds = self._open()
        inits = np.asarray(ds[self._upstream_init].values, dtype="datetime64[ns]")
        first = np.datetime64(date.fromisoformat(start[:10]), "ns")
        last = np.datetime64(date.fromisoformat(end[:10]), "ns") + np.timedelta64(1, "D")
        selected = inits[(inits >= first) & (inits < last)]
        return [str(np.datetime64(value, "s")).replace("T", " ")[:19] for value in selected]

    @property
    def _upstream_init(self) -> str:
        return "init_time"

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """One initialisation as ``(reference_time=1, lead_time=N[, quantile], y, x)``."""
        ds = self._open()
        xmin, ymin, xmax, ymax = map(float, bbox)
        init = np.datetime64(period_id.replace(" ", "T"), "ns")

        run = ds[[self._variable]].sel({self._upstream_init: init})
        # Subset before reducing: the global grid is 721 x 1440 across 31 members and 181 leads,
        # so reducing first would pull the whole world to produce one country.
        #
        # `bbox_slices` rather than `slice(ymax, ymin)`: a plain label slice keeps only cells
        # whose *centre* falls inside the bbox, so on this 0.25° grid the store ended up ~14 km
        # short of the requested extent on each edge — an uncovered strip on the map and border
        # districts aggregated from partial data. It also handles the descending latitude axis.
        run = run.sel(bbox_slices(run, [xmin, ymin, xmax, ymax], x_dim="longitude", y_dim="latitude"))
        run = run.sel(lead_time=slice(None, np.timedelta64(self._max_lead_days, "D")))

        units = str(run[self._variable].attrs.get("units", ""))
        if self._resample == "sum" and units.endswith(_RATE_SUFFIX):
            raise ValueError(
                f"{self._variable!r} is a rate ({units}); summing its steps has no meaning. "
                "Use resample='mean', or accumulate=True for a daily total."
            )

        run = self._reduce_members(run)
        daily = self._resample_leads(run, init)
        daily = self._integrate_over_the_day(daily)
        return self._as_forecast_cube(daily, init)

    def _integrate_over_the_day(self, daily: xr.Dataset) -> xr.Dataset:
        """Turn a mean rate into a daily total, when the template asked for one.

        Upstream precipitation is an average rate in ``kg m-2 s-1`` over each step, but a health
        consumer wants millimetres per day. Over water's density the two are numerically the
        same, so the whole conversion is a multiplication by the length of a day.
        """
        if not self._accumulate:
            return daily
        attrs = dict(daily[self._variable].attrs)
        integrated = daily[self._variable] * _SECONDS_PER_DAY
        attrs["units"] = self._accumulated_units
        attrs["cell_methods"] = "time: sum"
        integrated.attrs = attrs
        return daily.assign({self._variable: integrated})

    def _reduce_members(self, run: xr.Dataset) -> xr.Dataset:
        """Collapse the 31 ensemble members to a mean, or to the requested quantiles."""
        if "ensemble_member" not in run.dims:
            return run
        if self._quantiles is None:
            return run.mean("ensemble_member", keep_attrs=True)
        # `quantile` as one axis rather than a variable per quantile: one coordinate serves
        # every variable and extends without a schema change. Named to match xclim, so
        # ensemble statistics computed later land on the same axis.
        return run.quantile(self._quantiles, dim="ensemble_member", keep_attrs=True)

    def _resample_leads(self, run: xr.Dataset, init: np.datetime64) -> xr.Dataset:
        """Aggregate mixed 3/6-hourly leads into whole forecast days.

        Grouped by the *date* each step is valid for rather than by a fixed number of steps,
        because the upstream spacing changes partway through the run — counting steps would
        silently make later "days" twice as long.
        """
        valid = run.coords["valid_time"] if "valid_time" in run.coords else None
        if valid is None:
            valid = init + run["lead_time"]
        # Materialised: xarray refuses to group by a chunked array without explicit labels, and
        # this is one value per lead step — 181 of them — so there is nothing to defer.
        day = valid.dt.floor("D").compute()
        labels = np.asarray(day.values)
        grouped = run.groupby(day.rename("valid_day"))
        aggregated = getattr(grouped, self._resample)(keep_attrs=True)

        keep = self._complete_days(np.asarray(valid.values, dtype="datetime64[ns]"), labels)
        if not keep.all():
            dropped = np.unique(labels)[~keep]
            logger.debug(
                "Dropping %d forecast day(s) the run only partly covers: %s",
                len(dropped),
                ", ".join(str(value)[:10] for value in dropped),
            )
            aggregated = aggregated.isel(valid_day=np.flatnonzero(keep))
        # Lead in whole days from the initialisation, which is what the forecast axis means.
        leads = ((aggregated["valid_day"] - init.astype("datetime64[ns]")) / np.timedelta64(1, "D")).astype("int32")
        daily: xr.Dataset = aggregated.assign_coords({"lead_time": ("valid_day", leads.values)}).swap_dims(
            {"valid_day": forecast.LEAD_DIM}
        )
        return daily

    @staticmethod
    def _complete_days(stamps: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Flag which forecast days the run's steps actually span, in the group order.

        A day counts as covered when its steps reach the end of it: the last step's offset into
        the day, plus the spacing between steps, must be a whole day. Testing the span rather
        than the number of steps matters because upstream spacing widens from 3- to 6-hourly
        partway through the run, so a complete later day holds half as many steps as a complete
        early one — a count comparison would discard 25 good days.

        A 00z GEFS run reaches 840h, one step past day 35, so the tail group holds a single
        instant. Left in, it reads as a whole day: harmless-looking as a mean, and an eightfold
        understatement as a total.
        """
        one_day = np.timedelta64(1, "D")
        overall = np.diff(np.unique(stamps))
        # A lone step carries no spacing of its own, so fall back to the run's finest.
        finest = overall.min() if overall.size else one_day
        covered = []
        for value in np.unique(labels):
            in_day = np.sort(stamps[labels == value])
            within = np.diff(in_day)
            spacing = within.min() if within.size else finest
            covered.append((in_day[-1] - value) + spacing >= one_day)
        return np.array(covered, dtype=bool)

    def _as_forecast_cube(self, daily: xr.Dataset, init: np.datetime64) -> xr.Dataset:
        """Rename to the published axes and add the issue time and valid-time coordinate."""
        renamed = daily.rename({"latitude": "y", "longitude": "x"})
        renamed = renamed.drop_vars(["valid_day", "valid_time"], errors="ignore")
        renamed = renamed.expand_dims({forecast.REFERENCE_DIM: [init]})
        leads = np.asarray(renamed[forecast.LEAD_DIM].values, dtype="int32")
        offsets = leads.astype("timedelta64[D]").astype("timedelta64[ns]")
        renamed = renamed.assign_coords(
            {
                forecast.VALID_COORD: (
                    (forecast.REFERENCE_DIM, forecast.LEAD_DIM),
                    (init + offsets)[None, :],
                )
            }
        )
        for name, standard in forecast.CF_STANDARD_NAMES.items():
            if name in renamed.coords:
                renamed[name].attrs["standard_name"] = standard
        renamed[forecast.LEAD_DIM].attrs["units"] = "days"
        ordered = renamed.transpose(forecast.REFERENCE_DIM, forecast.LEAD_DIM, ..., "y", "x", missing_dims="ignore")
        return self._store_ready_chunks(ordered)

    @staticmethod
    def _store_ready_chunks(ds: xr.Dataset) -> xr.Dataset:
        """Rechunk so the writer can store the cube.

        Grouping by forecast day leaves one chunk per lead, and slicing a bbox out of the global
        grid leaves ragged spatial chunks like (10, 17, 4) — which `to_zarr` rejects outright.
        One chunk per issue time matches how the store grows, and a fixed spatial chunk stays
        uniform for a country extent as much as for a global one.
        """
        sizes = ds.sizes
        chunks: dict[str, int] = {forecast.REFERENCE_DIM: 1}
        for dim in (forecast.LEAD_DIM, "quantile"):
            if dim in sizes:
                chunks[dim] = int(sizes[dim])
        for dim in ("y", "x"):
            if dim in sizes:
                chunks[dim] = min(int(sizes[dim]), _SPATIAL_CHUNK)
        rechunked: xr.Dataset = ds.chunk(chunks)
        return rechunked

    def _iso(self, value: Any) -> str:
        return datetime.fromisoformat(str(value)[:19]).isoformat()
