# Workflows

A **workflow** is a reusable, named openEO process graph stored on the server. Instead of repeating the same chain of processes every time, you define it once, give it a name and parameters, and call it by name from any openEO client in any language.

Workflows appear in `GET /process_graphs` and are callable directly by `process_id` in a process graph — exactly like any standard openEO process.

---

## Anatomy of a workflow

A workflow is a JSON file with an `id`, a list of `parameters`, and a `process_graph`:

```json
{
  "id": "monthly_precipitation",
  "summary": "Aggregate daily precipitation to monthly totals",
  "parameters": [
    {
      "name": "collection_id",
      "description": "Dataset id to load",
      "schema": { "type": "string" }
    }
  ],
  "process_graph": {
    "load": {
      "process_id": "load_collection",
      "arguments": { "id": { "from_parameter": "collection_id" } }
    },
    "aggregate": {
      "process_id": "aggregate_temporal_period",
      "arguments": {
        "data": { "from_node": "load" },
        "period": "month",
        "reducer": {
          "process_graph": {
            "sum": {
              "process_id": "sum",
              "arguments": { "data": { "from_parameter": "data" } },
              "result": true
            }
          }
        }
      },
      "result": true
    }
  }
}
```

Calling it from a process graph is a single node:

```json
{
  "process_graph": {
    "1": {
      "process_id": "monthly_precipitation",
      "arguments": { "collection_id": "chirps3_precipitation_daily" },
      "result": true
    }
  }
}
```

---

## Built-in workflows

Open Climate Service ships with ready-to-use workflows for aggregating **any published GeoZarr dataset** to a set of **GeoJSON features** (typically DHIS2 organisation units) and exporting DHIS2-ready output:

| Workflow | Output |
|---|---|
| `aggregate_to_dhis2_json` | DHIS2 `dataValueSet` JSON |
| `aggregate_to_chap_csv` | CHAP wide CSV (`time_period`, `location`, one column per variable) |

Both run `load_collection → aggregate_spatial → save_result`: they load the dataset over a time range, compute a spatial statistic of the variable within each feature, and emit one value per feature per time step. Each feature's GeoJSON `id` becomes the DHIS2 `orgUnit` (CHAP `location`), and each time step becomes the DHIS2 `period` (CHAP `time_period`).

### Parameters

| Name | Workflows | Default | Description |
|---|---|---|---|
| `dataset_id` | both | — | Published GeoZarr collection to aggregate (see `/datasets`) |
| `temporal_extent` | both | — | `[start, end]` ISO-8601 dates |
| `geometries` | both | — | GeoJSON `FeatureCollection`; each feature's `id` is the org unit / location |
| `data_element_id` | DHIS2 only | — | DHIS2 data element id assigned to every value |
| `method` | both | `mean` | Spatial aggregation method: `mean`, `min`, `max`, or `sum` |
| `period_type` | both | `month` | Period type used to format each time step: `day`, `week`, `month`, `quarter`, `year` |

> `period_type` **formats** each native time step into a DHIS2 period — it does not re-aggregate in time. Pick a dataset whose native temporal resolution matches the period you want (e.g. a monthly dataset for monthly values).

### Example

Mean monthly precipitation per district, as DHIS2 data values:

```json
{
  "process": {
    "process_graph": {
      "agg": {
        "process_id": "aggregate_to_dhis2_json",
        "arguments": {
          "dataset_id": "era5land_precipitation_monthly",
          "temporal_extent": ["2025-01-01", "2025-12-31"],
          "geometries": { "type": "FeatureCollection", "features": [ "...org units..." ] },
          "data_element_id": "fbfJHSPpUQD",
          "method": "mean",
          "period_type": "month"
        },
        "result": true
      }
    }
  }
}
```

Submit it to `POST /result` (synchronous) or `POST /jobs` (batch); the result is a DHIS2 `dataValueSet` ready to POST to the DHIS2 Web API. For CHAP CSV, call `aggregate_to_chap_csv` with the same arguments minus `data_element_id`.

---

## Mapping change between two periods

`temporal_change` computes the per-pixel **net change** of a variable between the first and last time step in a range (`last − first`) and publishes it as a new single-band GeoZarr dataset — for example population change between two census years, or NDVI change between two dekads. Positive values are increases and negative values decreases, so the result suits a diverging colormap centred on zero.

It runs `load_collection → reduce_dimension → save_result`, reducing the time dimension away to a 2-D `(y, x)` raster. Only the earliest and latest time step in `temporal_extent` contribute; any steps in between are ignored.

### Parameters

| Name | Description |
|---|---|
| `dataset_id` | Published GeoZarr collection to load (see `/datasets`) |
| `output_dataset_id` | Id of the change dataset to publish (needs a dataset template — see below) |
| `variable` | Variable/band name carried through to the published dataset |
| `temporal_extent` | `[start, end]` ISO-8601 dates; the change is `value(last) − value(first)` within this range |

The `output_dataset_id` must have a **dataset template** registered on the instance: a YAML in the built-in `plugins/datasets/` folder (or an instance's `plugins_dir/datasets/`) with `sync: {kind: static}` and a `display` block. No ingestion plugin (`.py`) is needed — the data is produced by the workflow, not ingested. Open Climate Service bundles `worldpop_population_change` (a second entry in `worldpop.yaml`) for the population example below:

```yaml
- id: worldpop_population_change
  name: Population change (WorldPop Global2)
  short_name: Population change
  variable: pop_change
  period_type: yearly
  sync:
    kind: static
  units: people
  display:
    colormap: RdBu
    range: [-50.0, 50.0]
```

### Example

Zarr output cannot be produced synchronously, so submit it as a **batch job** (`POST /jobs`, then `POST /jobs/{id}/results`):

```json
{
  "process": {
    "process_graph": {
      "change": {
        "process_id": "temporal_change",
        "arguments": {
          "dataset_id": "worldpop_population_global2_R2025A_100m",
          "output_dataset_id": "worldpop_population_change",
          "variable": "pop_change",
          "temporal_extent": ["2015-01-01", "2030-12-31"]
        },
        "result": true
      }
    }
  }
}
```

When the job finishes, the new change dataset appears under `/datasets` and on the map viewer.

## Climate normals

`climate_normal` computes a **climatological normal** — a WMO-style multi-year average of a variable for each **day-of-year** (1–366) or **month** (1–12) over a reference period (e.g. 1991–2020) — and publishes it as a new managed dataset. Use it to derive a "what's typical" baseline from a collection you have already ingested for the reference period; the anomaly workflow below then compares observations against it.

### Parameters

| Name | Description |
|---|---|
| `dataset_id` | Published collection to load; must cover the reference period |
| `output_dataset_id` | Id of the normal dataset to publish (needs a static template — see below) |
| `variable` | Variable/band name carried through to the published dataset |
| `temporal_extent` | Reference period `[start, end]`, e.g. `["1991-01-01", "2020-12-31"]` |
| `frequency` | `dayofyear` (1–366, default) or `month` (1–12) |
| `smoothing_window` | Circular day-of-year smoothing window in days (0 disables, default 31; ignored for `month`) |

The `output_dataset_id` needs a **static dataset template** (`sync: {kind: static}`, `period_type: climatology`, a `display` block; no ingestion plugin) — auto-registered from the result and source metadata if none exists. Open Climate Service bundles the from-store monthly templates (`era5land_temperature_monthly_normal_1991_2020`, …) plus EDH-direct **day-of-year** ERA5-Land normals (`era5land_*_daily_normal_1991_2020`) that read the reference period straight from Earth Data Hub — no 30-year ingest needed.

### Example

```json
{
  "process": {
    "process_graph": {
      "normal": {
        "process_id": "climate_normal",
        "arguments": {
          "dataset_id": "era5land_temperature_monthly",
          "output_dataset_id": "era5land_temperature_monthly_normal_1991_2020",
          "variable": "t2m",
          "temporal_extent": ["1991-01-01", "2020-12-31"],
          "frequency": "month"
        },
        "result": true
      }
    }
  }
}
```

## Climate anomalies

`climate_anomaly` computes an **anomaly** — how far observations depart from a climatological normal — and publishes it as a new managed dataset that keeps the observed time axis. Positive values are above-normal and negative below-normal, so the result suits a diverging colormap centred on zero.

### Parameters

| Name | Description |
|---|---|
| `observed_dataset_id` | Published observed collection (datetime axis), e.g. `era5land_temperature_daily` |
| `normal_dataset_id` | Published climatological normal (with a `dayofyear`/`month` axis) |
| `output_dataset_id` | Id of the anomaly dataset to publish (needs a static template — see below) |
| `variable` | Variable/band name carried through to the published dataset |
| `temporal_extent` | Observed range `[start, end]` |
| `method` | `absolute` (observed − normal, default) or `relative` (percent of normal — for ratio-scale variables like precipitation only, **not** temperature) |

The `output_dataset_id` needs a static template like the normals above (auto-registered if missing). Use a diverging colormap centred on zero, running the way the variable is read: `RdBu` for precipitation, where wet is blue and dry is red, and `rdbu_r` for temperature, where warm is red. Pair a **daily** observed dataset with a day-of-year normal, or a **monthly** observed dataset with a month normal — `compute_anomaly` aligns on the matching axis automatically.

The two methods publish **different units**, so a pre-registered template belongs to one of them:

| `method` | Published units | Suitable range |
|---|---|---|
| `absolute` | the observed variable's unit (e.g. `mm/d`) | precipitation `[-20, 20]` daily, `[-10, 10]` monthly — a monthly mean rate averages the extremes away |
| `relative` | `%` | `[-100, 100]` |

An auto-registered template takes its units from the computed result, so it is correct either way. A pre-registered one is authoritative, and publishing a relative anomaly into a template declaring `mm/d` is **refused** rather than relabelled — otherwise 20 % of normal would be stored as 20 mm of rain per day, which is plausible enough that nothing downstream could catch it. The shipped templates come in both forms: `*_anomaly_1991_2020` declares the observed unit for `absolute`, and `*_relative_anomaly_1991_2020` declares `%` for `relative`. Target the one matching the `method` you pass — precipitation only, since a relative anomaly is refused for temperature.

The observed and normal cubes must be on the **same grid** — same cell count, same spacing, same positions — otherwise the anomaly is refused.

### Example

```json
{
  "process": {
    "process_graph": {
      "anomaly": {
        "process_id": "climate_anomaly",
        "arguments": {
          "observed_dataset_id": "era5land_temperature_daily",
          "normal_dataset_id": "era5land_temperature_daily_normal_1991_2020",
          "output_dataset_id": "era5land_temperature_daily_anomaly_1991_2020",
          "variable": "t2m",
          "temporal_extent": ["2026-01-01", "2026-06-24"],
          "method": "absolute"
        },
        "result": true
      }
    }
  }
}
```

When the job finishes, the normal/anomaly dataset appears under `/datasets` and on the map viewer.

---

## Aggregating dekads to months or weeks

`aggregate_dekads_to_period` turns a published **dekadal** (10-daily) dataset into a monthly
or weekly one, weighting each dekad by the number of days it shares with the target period.

The weighting is the point. Dekads run day 1–10, 11–20, then 21 to the end of the month, so
the third is 8, 9, 10 or 11 days long. Three of them tile a calendar month exactly, but they
are not equal, so a plain `mean` over-weights a short third dekad — February's 8-day dekad by
4.8 percentage points, a 14% relative error on its contribution. The error is systematic
rather than noise: it always favours short dekads, so an unweighted monthly series carries a
seasonal artefact that follows month length.

Pick `method` by what the variable is:

| `method` | For | Result |
|---|---|---|
| `mean` (default) | a per-day **rate**, e.g. CLMS GPP in `gC/m²/day` | the target period's average daily value, same units |
| `sum` | a per-dekad **total** | the target period's total, conserved exactly regardless of dekad length |

Passing `sum` for a per-day rate is logged as a warning — adding daily rates does not produce
a total.

`period: week` exists but is rarely what you want: dekads and ISO weeks never align (36
against 52 or 53), so a weekly series has an effective resolution of about 10 days however it
is derived, and the weights must be recomputed per year. Monthly is exact by comparison.

**Not interpolation, deliberately.** Splining through dekad midpoints does not conserve the
annual total, invents sub-dekad structure the sensor never observed, and can undershoot below
zero for a non-negative quantity. If a smooth daily curve is ever genuinely needed, the
defensible route is mean-preserving interpolation, not a plain spline.

The output variable carries a `cell_methods` recording the weighting, so a consumer can tell a
day-weighted aggregate from an observation. A partially covered period at either end of the
record is computed from the dekads that exist rather than returning empty, and a dekad missing
over part of the grid is dropped from the weights *there* and the remainder renormalised.

## Three sources of workflows

Workflows are loaded from three places, each overriding the previous on id collision:

| Source | Location | Loaded |
|---|---|---|
| Built-in | `open_climate_service/plugins/workflows/` | At startup |
| Instance plugin | `plugins_dir/workflows/` | On each request |
| Runtime-registered | `PUT /process_graphs/{id}` | Immediately |

**Instance plugin workflows** (files in `plugins_dir/workflows/`) are re-read on every `GET /process_graphs` call — no restart needed to pick up changes.

**Runtime-registered workflows** are created via the `PUT /process_graphs/{id}` API and stored in the instance data directory. They disappear if the data directory is wiped.

---

## Creating a workflow

### Via file (recommended for instance repos)

Add a `.json` file to your instance `plugins/workflows/` directory:

```
my-instance/
└── plugins/
    └── workflows/
        └── aggregate_for_dhis2.json
```

Configure `plugins_dir` in `climate-service.yaml`:

```yaml
plugins_dir: ./plugins/
```

The workflow appears in `GET /process_graphs` immediately on the next request — no restart required.

### Via API

```bash
curl -X PUT http://127.0.0.1:9000/process_graphs/monthly_precipitation \
  -H "Content-Type: application/json" \
  -d @monthly_precipitation.json
```

---

## Using a workflow

### Python (openEO client)

```python
from openeo import connect

conn = connect("http://127.0.0.1:9000")

result = conn.execute(
    {
        "process_graph": {
            "1": {
                "process_id": "monthly_precipitation",
                "arguments": {"collection_id": "chirps3_precipitation_daily"},
                "result": True,
            }
        }
    }
)
```

### JavaScript (openEO JS client)

```javascript
import { OpenEO } from "@openeo/js-client";

const conn = await OpenEO.connect("http://127.0.0.1:9000");

const process = await conn.buildProcess((builder) =>
  builder.monthly_precipitation("chirps3_precipitation_daily")
);

const result = await conn.computeResult(process);
```

The JS client discovers available workflows via `GET /process_graphs` automatically — `monthly_precipitation` appears alongside standard openEO processes.

---

## Listing workflows

```bash
GET /process_graphs
```

Returns all available workflows: built-ins, instance plugins, and runtime-registered, merged together.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/process_graphs` | List all workflows |
| `GET` | `/process_graphs/{id}` | Get one workflow |
| `PUT` | `/process_graphs/{id}` | Create or replace a workflow |
| `DELETE` | `/process_graphs/{id}` | Delete a runtime-registered workflow |

> Note: `DELETE` only removes runtime-registered workflows. Workflows loaded from `plugins_dir/workflows/` files persist until the file is removed — deleting them via the API has no effect.
