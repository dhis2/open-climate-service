# Adding custom datasets

This guide explains how to add a new dataset source to your Open Climate Service instance — for example a national meteorological service, a regional satellite product, or a custom model output.

The built-in dataset templates (CHIRPS3, ERA5-Land, WorldPop) ship as package data. Custom datasets are layered on top by pointing `plugins_dir` in your `climate-service.yaml` at a plugins directory.

## Overview

Adding a custom dataset involves two things:

1. **A streaming plugin** — a Python class that enumerates periods and fetches one period at a time as an `xarray.Dataset`.
2. **A dataset template YAML** — a file that describes the dataset and tells the API which plugin class to use.

Place both in your `plugins/datasets/` directory:

```
plugins/
└── datasets/
    ├── enacts_rainfall.yaml
    └── enacts.py          # the plugin class
```

## Step 1: Write the streaming plugin

Subclass `BaseDatasetPlugin` and implement just two methods. The base class supplies
the concurrency defaults and the canonical dimension names; the framework handles resume,
concurrency, store commits, artifact registration, and publication.

```python
# plugins/datasets/enacts.py
# Everything you need to write a plugin is importable from open_climate_service.streaming.
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, daily_period_ids, normalize_period


class ENACTSRainfallPlugin(BaseDatasetPlugin):
    async def periods(self, start: str, end: str) -> list[str]:
        """Return the ordered list of period ids available between start and end."""
        ...

    def fetch_period(self, period_id: str, bbox: list[float], **params) -> xr.Dataset:
        """Fetch one period and return it as an xarray Dataset."""
        da = ...  # read the source raster for this period
        return normalize_period(da, variable="rainfall", period=period_id, bbox=bbox)
```

**`periods`** — returns an ordered list of period identifiers (typically ISO 8601 date
strings) the source has available between `start` and `end`. The framework uses it to
determine which periods are missing and need to be fetched.

**`fetch_period`** — fetches exactly one period and returns it as an `xarray.Dataset`
normalized to `(t, y, x)`. Write it as a regular (blocking) method and the framework runs
it in a worker thread, so ordinary blocking I/O is fine; the framework appends the result
directly to the Icechunk-backed Zarr store, so the function should not write to disk. For
a natively-async source (e.g. lazy Zarr access), declare it `async def fetch_period(...)`
instead — the orchestrator awaits it directly.

The framework **closes the dataset you return** after writing it (releasing the
`open_rasterio` / `open_dataset` handles), so return a self-contained dataset — not a lazy
view that shares a backing handle with a long-lived cache. A plugin that caches a fetched
month/region should `.load()` it into memory so the per-period slices it returns are
independent.

`**params` receives the `params` dict from the YAML template, so the same class can serve
multiple variables.

### Helpers, grid inference, and tuning

The helpers below are all importable from `open_climate_service.streaming` (the single
plugin import surface), alongside `BaseDatasetPlugin`.

- **`normalize_period(obj, *, variable, period=None, nodata=None, bbox=None, bbox_crs="EPSG:4326", ...)`**
  — turns a freshly read raster/dataset into the canonical
  `(t, y, x)` single-variable shape. It drops curvilinear 2-D `lon`/`lat` helper coordinates,
  renames the source axes (`lon`/`longitude`/`X` → `x`, `lat`/`latitude`/`Y` → `y`,
  `time`/`valid_time` → `t`), clips to `bbox` (reprojecting the bbox from `bbox_crs` — WGS84
  by default — onto the source CRS, so a projected/UTM grid clips correctly), drops a
  singleton `band`, masks the nodata sentinel, and stamps the period onto the time axis.
- **`daily_period_ids(start, end)`** — enumerate the inclusive ISO day strings for a daily
  `periods()` implementation; apply your own availability clamp around it (accepts ISO
  strings or `date` objects, returns `[]` when `start > end`).
- **Tuning** — set the class attributes `max_concurrency` (default 1) and
  `commit_batch_size` (default 1) only when the defaults don't fit.
- **Grid inference** — the framework infers the store grid from the
  **first fetched period** — shape and dtype from the array, the nodata sentinel from the
  source `_FillValue`, and the CRS from the data. CRS inference falls back to **EPSG:4326**
  when the data carries none, so a **projected-grid** source should declare its CRS with the
  **`crs` class attribute** (an EPSG int or string).

### Projected-grid (non-WGS84) sources

For a source on a projected grid (e.g. a national UTM product), set the `crs` class attribute
and write that CRS onto the data before normalizing. `normalize_period` then reprojects the
(WGS84) request bbox onto the grid for the spatial clip, so no manual coordinate transform is
needed — and the declared `crs` is what the grid inference records for the store:

```python
class SeNorgePlugin(BaseDatasetPlugin):
    crs = 32633  # UTM33 (EPSG) — drives both grid inference and the normalize_period clip

    def fetch_period(self, period_id, bbox, **params):
        import rioxarray  # noqa: F401  # activates the .rio accessor for write_crs

        ds = read_source(period_id).rio.write_crs(self.crs)
        return normalize_period(ds, variable="tg", bbox=bbox)
```

### Extra (non-spatial) dimensions

A dataset is not limited to `(t, y, x)`. `fetch_period` may return a single variable with
additional non-spatial dimensions — for example a `dayofyear` climatology axis, or `sex`
and `age_group` disaggregation axes (as the WorldPop age/sex plugin does).

STAC declares each non-spatial dimension under `cube:dimensions`, and the map viewer builds
one control per dimension: a **slider** for a temporal or evenly-spaced ordinal axis (one
with a regular numeric `step`, e.g. `dayofyear`), and a **dropdown** for a categorical or
irregularly-spaced one (e.g. `sex`, or the irregular age bands). The control type follows
from the dimension's metadata, so there's nothing extra to configure.

## Step 2: Create a dataset template YAML

```yaml
# plugins/datasets/enacts_rainfall.yaml
- id: enacts_rainfall_daily
  name: ENACTS Rainfall (daily)
  short_name: Rainfall
  variable: rainfall
  period_type: daily
  sync:
    kind: temporal
    execution: append
  ingestion:
    plugin: datasets.enacts.ENACTSRainfallPlugin
  units: mm
  resolution: 4 km x 4 km
  source: ENACTS
  source_url: https://enacts.example.org
```

### Template field reference

**Identity**

| Field        | Required | Description                                                                    |
| ------------ | -------- | ------------------------------------------------------------------------------ |
| `id`         | Yes      | Unique template identifier. This becomes the dataset ID in the API             |
| `name`       | Yes      | Full human-readable name shown in API responses and STAC metadata              |
| `short_name` | No       | Short label used in compact displays                                           |
| `variable`   | Yes      | Name of the data variable in the Zarr store (e.g. `precip`, `t2m`, `rainfall`) |
| `source`     | No       | Name of the upstream data source                                               |
| `source_url` | No       | URL to the upstream dataset documentation or landing page                      |

**Period and sync**

| Field                | Required | Description                                                                                                                                                                                                                                       |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `period_type`        | Yes      | Temporal resolution: `hourly`, `daily`, `dekadal`, `weekly`, `monthly`, `yearly`, or `climatology`. Validated at registration — an unrecognised value is rejected rather than silently ignored, and it is required unless `sync.kind` is `static` |
| `sync.kind`          | Yes      | `temporal` — data grows over time; `release` — versioned releases; `static` — never synced                                                                                                                                                        |
| `sync.execution`     | No       | `append` — new time steps appended to existing store; `rematerialize` — full rebuild on each sync                                                                                                                                                 |
| `temporal_direction` | No       | Which way the periods run relative to now: `past` (default), `future` (a forecast), or `spanning` (crosses now, e.g. WorldPop 2015–2030). See below                                                                                               |

### Dekads: a period type with no fixed length

`dekadal` is 10-daily data, and is the one cadence whose periods differ in length: a dekad
runs day 1–10, then 11–20, then **21 to the end of the month**, so the third is 8, 9, 10
or 11 days long. There are 36 in a year.

Two consequences worth knowing:

- **Period ids are the dekad's first day** — `2026-01-01`, `2026-01-11`, `2026-01-21`. They
  sort chronologically and parse as ordinary dates. Any date within a dekad normalises to
  its start, so `2026-01-15` becomes `2026-01-11`.
- **There is no ISO 8601 duration for a dekad**, so the STAC temporal dimension declares
  `step: null` — the datacube extension's encoding for irregular spacing — rather than a
  fictional `P10D` that would be wrong for every third dekad. Clients read the timestamps
  instead of extrapolating from a step.

A plugin enumerating dekads should use `shared.time.dekad_period_ids(start, end)`, the
dekadal counterpart of `daily_period_ids`. `dekad_bounds(period_id)` gives the inclusive
first and last day, which is the only complete description of a dekad's extent.

### Which way the periods run: `temporal_direction`

Most datasets are historical, and the default (`past`) suits them. Two other shapes exist, and they behave differently at ingest time:

| Value            | Periods                                                       | `start` on an ingestion          |
| ---------------- | ------------------------------------------------------------- | -------------------------------- |
| `past` (default) | All historical                                                | Required                         |
| `future`         | All ahead of now — a forecast                                 | **Optional**, meaning "from now" |
| `spanning`       | Cross now — WorldPop Global2 (2015–2030), climate projections | Required                         |

`spanning` requires a start _on purpose_. Defaulting it to "now" would ingest only the projected years and silently drop every historical one, which is usually the half you actually want. What declaring it does buy you: the ingest form prefills the end from the dataset's declared `extents.temporal.end`, so selecting WorldPop offers the full range through 2030 instead of truncating at today.

### Forecast datasets (`temporal_direction: future`)

A forecast's periods lie in the _future_, which changes what an ingestion request means. Declare it:

```yaml
- id: tmax_forecast_daily
  period_type: daily
  temporal_direction: future
  sync:
    kind: temporal # still temporal: re-running fetches a fresher forecast
  ingestion:
    plugin: datasets.my_forecast.MyForecastPlugin
    params:
      max_lead_days: 7
```

With that declared, **`start` may be omitted** from an ingestion request, and means "from now":

```bash
curl -X POST http://localhost:9000/ingestions \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id": "tmax_forecast_daily"}'
```

Omitting it is usually what you want. A fixed `start` is only correct on the day it is written — tomorrow it under-requests — so a scheduled refresh with hardcoded dates drifts out of the forecast window and then quietly fetches nothing. Supplying `start`/`end` still works, and narrows the window when you deliberately want a subset.

**What your `periods()` receives.** For a historical dataset an omitted end is filled in with "now" — "through the latest available period". A forecast cannot use that, because "now" is the _start_ of its window: filling it in would hand you `start == end == today` and collapse a seven-day forecast to one day. So a forecast instead receives a **forward horizon** — the template's declared `extents.temporal.end` if it has one, otherwise a year ahead. It is deliberately generous: the real limit is your plugin's lead time, and a tighter bound in core would silently truncate a longer forecast.

Your plugin clips to what it actually publishes:

```python
async def periods(self, start: str, end: str) -> list[str]:
    base = date.fromisoformat(start[:10])  # core resolves this to today
    days = [(base + timedelta(days=i)).isoformat() for i in range(self.max_lead_days)]
    return [d for d in days if d <= end[:10]]  # clip to the requested window
```

Note `end` stays a plain `str`, so there is no missing-value case to handle.

**Honour `end` when it is given.** If your plugin returns periods outside the requested scope, the ingestion is refused with `Materialized artifact coverage does not match the requested scope`. That guard is helpful — it catches a plugin that ignores the range rather than silently storing more than was asked for — but it means a lead-day plugin has to filter rather than ignore.

`temporal_direction` is separate from `sync.kind` on purpose: a forecast is still `temporal` for sync (re-run it and you get fresher data); what differs is which way its periods run. It cannot be combined with `sync.kind: static`, which has no upstream to look ahead into.

**How far ahead belongs in the template, not the request.** A source often publishes further out than is useful — 40 days when only 7 verify well. That cap is a property of the dataset, so express it in `ingestion.params` (as `max_lead_days` above) and let your plugin's `periods()` honour it. The request then narrows _within_ that window rather than re-deciding it on every run.

The response reports the window that was actually ingested, under `dataset.extent.temporal`, so you can confirm what an omitted `start` resolved to.

**Ingestion**

| Field                  | Required | Description                                                                                                                                     |
| ---------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `ingestion.plugin`     | Yes      | Dotted path to the streaming plugin class                                                                                                       |
| `ingestion.params`     | No       | Extra keyword arguments forwarded to `fetch_period` as `**params`, and to the plugin constructor                                                |
| `ingestion.resampling` | No       | Pyramid coarsening for large layers: `mean` (default; continuous data), `max`/`min`/`sum`, or `mode`/`nearest` for categorical data — see below |

Multiple templates can share the same plugin class and differ only in `params`:

```yaml
- id: era5land_temperature_hourly
  ingestion:
    plugin: open_climate_service.plugins.datasets.era5_land.ERA5LandHourlySingleBandPlugin
    params:
      variable: 2m_temperature

- id: era5land_precipitation_hourly
  ingestion:
    plugin: open_climate_service.plugins.datasets.era5_land.ERA5LandPrecipitationPlugin
    params:
      variable: total_precipitation
```

#### Pyramid resampling for categorical layers

Layers whose grid exceeds 1024×1024 cells (about 1.05 megapixels, counted as `nx × ny` rather than per axis) are stored as a multiscale pyramid so the map stays fast when zoomed out. Coarser levels sit beside the full-resolution level 0, which analytics keeps reading, so **analysis results do not change** and the persistent cost is roughly +33% storage.

Building one is not free, though. Level 0 is rewritten rather than kept as-is — the pyramid writer re-chunks it to its own target chunk size — the rebuild goes to a sibling store and is swapped in, so it needs transient disk for a second copy, and it takes longer than a flat write (measured at roughly 2.4× on a 2 GB store, in exchange for around 6× less peak memory).

Levels are aggregated from the full-resolution data, and `ingestion.resampling` controls how:

- **Continuous data** (temperature, precipitation, NDVI, …) — leave the default `mean`.
- **Binary masks** (0/1 presence) — use `max` ("present anywhere in the block"). Averaging turns a mask into meaningless fractions.
- **Multi-class categorical** (land-cover class codes, etc.) — use `mode` (majority class). `mean` would average class codes into a _different, non-existent_ class (e.g. `mean(10, 80) = 45`).

`mean`/`max`/`min`/`sum`/`nearest` are computed by [topozarr](https://github.com/carbonplan/topozarr), which builds each level from the one above. That is valid for all five because they are _composable_ — for `nearest`, taking the corner of each corner gives the same cell as taking every nth cell of the original.

`mode` is not composable: mode-of-modes is not mode-of-native, since a locally dominant class can win at coarse zoom even when it is globally rare. So Open Climate Service resamples `mode` levels from the native resolution itself. A first-class `mode` upstream is still open as [carbonplan/topozarr#26](https://github.com/carbonplan/topozarr/issues/26); when it lands, that local path can go.

**Spatial and temporal extents** — declares what the source dataset covers. Used to validate ingest requests before hitting the provider:

```yaml
extents:
  spatial:
    bbox: [-180, -50, 180, 50] # [xmin, ymin, xmax, ymax] in WGS84
    crs: http://www.opengis.net/def/crs/OGC/1.3/CRS84
  temporal:
    begin: "1981-01-01"
    end: "2030-12-31" # omit if ongoing
    trs: http://www.opengis.net/def/uom/ISO-8601/0/Gregorian
    resolution: P1D # ISO 8601 duration: PT1H, P1D, P1M, P1Y
```

**CF metadata** — stamped onto the stored variable at ingest so the GeoZarr store is
CF-compliant on disk and CF-aware tools (xclim climate indices, cf-xarray, QGIS) work
without per-process glue. These fields take effect when the store is written, so changing
them requires re-ingesting the dataset:

| Field           | Required | Description                                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `units`         | No       | Physical units, as a **CF/udunits** string (e.g. `mm`, `mm/d`, `degC`, `kg m-2 s-1`). Validated at registration — a non-udunits value (e.g. `people`) is logged as a warning. Use `""` for a dimensionless quantity (e.g. a standardized index). For unit-aware processes (e.g. SPI) the unit must be _dimensionally_ correct — a precipitation **rate** is `mm/d`, not bare `mm`. |
| `standard_name` | No       | CF [standard name](https://cfconventions.org/standard-names.html) (e.g. `air_temperature`, `lwe_thickness_of_precipitation_amount`).                                                                                                                                                                                                                                               |
| `cell_methods`  | No       | CF cell methods describing the temporal aggregation (e.g. `time: mean`, `time: sum`).                                                                                                                                                                                                                                                                                              |

**Display**

| Field              | Required | Description                                              |
| ------------------ | -------- | -------------------------------------------------------- |
| `resolution`       | No       | Human-readable spatial resolution (e.g. `5 km x 5 km`)   |
| `display.colormap` | No       | Colormap name for map rendering (e.g. `blues`, `rdbu_r`) |
| `display.range`    | No       | `[min, max]` display range for the colormap              |
| `display.nodata`   | No       | No-data / fill value                                     |

## Step 3: Point the instance at your plugins directory

Add `plugins_dir` to your `climate-service.yaml`:

```yaml
extent:
  name: Rwanda
  bbox: [28.8, -2.9, 30.9, -1.0]

data_dir: ./data
plugins_dir: ./plugins/
```

All `*.yaml` files in `plugins_dir/datasets/` are loaded and merged with the built-in templates. Custom templates are additive — the built-ins remain available unless you deliberately override one by using the same `id`.

Since `plugins_dir` is added to `sys.path`, the plugin class at `datasets.enacts.ENACTSRainfallPlugin` is importable without installing a package.

## Step 4: Ingest and publish

Once the API is running with `CLIMATE_SERVICE_CONFIG` pointing to your updated config:

```bash
curl -s -X POST http://127.0.0.1:9000/ingestions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "enacts_rainfall_daily",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "publish": true
  }' | jq
```

Verify it appears in the STAC catalog:

```bash
curl -s http://127.0.0.1:9000/stac/catalog.json | jq '.links[] | select(.rel == "child")'
```

## Distributing a plugin as an installable package

The `plugins_dir` above is ideal for instance-specific customisation. To make a plugin
**reusable across instances** — packaged and installed with `uv add`, no path wiring — see the
[Installable plugins](installable_plugins.md) guide. The layout mirrors `plugins_dir`, so migrating
is mostly moving the files into a package and declaring one entry point.

The [seNorge plugin](https://github.com/dhis2/open-climate-service-senorge-plugin) is the reference
implementation.
