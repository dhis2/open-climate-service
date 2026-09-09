# openEO

openEO is an **open standard API** for accessing and processing Earth Observation (EO) data. Instead of downloading raw climate or satellite data and writing custom processing scripts, you describe _what you want to compute_ as a process graph, and the server runs it for you on its own data.

---

## Why openEO?

Traditional EO data access is fragmented: each data provider has its own API, format, and tools. openEO solves this by defining a vendor-neutral HTTP API so the same client code works against any compliant backend.

## Why openEO for the Open Climate Service?

The Open Climate Service stores climate datasets — precipitation, temperature, population — as managed Zarr stores. openEO gives us a standardised, well-documented way to query and transform those datasets without building a bespoke query language.

Concretely it means:

- **DHIS2 analytics apps** can request district-level climate aggregates (monthly sum, seasonal mean) without downloading raw daily rasters — the computation runs server-side and returns a small result.
- **Data scientists** can use the standard openEO Python client or web editor directly against the service without learning a DHIS2-specific API.
- **New datasets** added to the service are immediately queryable through the same process graph interface, with no additional API work.
- **Interoperability** — process graphs written for the Open Climate Service work, with minor configuration changes, against any other openEO-compliant backend, and vice versa.

---

## Key concepts

| Concept                | Description                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Collection**         | A published dataset, equivalent to a STAC collection. Has spatial/temporal extent, variables (bands), and dimension metadata. |
| **Process**            | A single named operation — `load_collection`, `filter_temporal`, `aggregate_temporal_period`, `save_result`, etc.             |
| **Process graph**      | Connected processes describing the full computation.                                                                          |
| **Batch job**          | Asynchronous execution of a process graph. Create → start → poll → download results.                                          |
| **Synchronous result** | `POST /result` — executes immediately and returns output in the HTTP response body.                                           |
| **UDP**                | User-Defined Process — a named, reusable process graph stored server-side; callable like any built-in process.                |

---

## Connecting

```python
import openeo

conn = openeo.connect("http://127.0.0.1:9000")
print(conn.capabilities().api_version())  # 1.2.0
```

No authentication is required for local deployments. `openeo.connect` discovers the API via `GET /.well-known/openeo` and negotiates the version automatically.

The web editor at [editor.openeo.org](https://editor.openeo.org) can also connect directly. Use `GET /openeo` as a shortcut — it redirects to the editor pre-configured with the correct server URL.

---

## Available collections

Collections map 1:1 to published datasets. They are exposed at `/collections` and are compatible with both openEO clients and STAC browsers.

```python
for c in conn.list_collections():
    print(c["id"], "—", c["title"])
```

Each collection includes `cube:dimensions` (spatial `x`/`y`, temporal `t`, `bands`), extent, and variable metadata.

---

## Building a process graph

Process graphs are composable operations. The openEO Python client builds them lazily — no data moves until you call `execute()` or `download()`.

```python
cube = conn.load_collection(
    "worldpop_population_global2_R2025A_100m",
    spatial_extent={"west": -13.3, "south": 7.0, "east": -10.3, "north": 10.0},
    temporal_extent=["2015-01-01", "2021-01-01"],
    bands=["pop_total"],
)
```

Chain operations exactly as in the [openEO Python client docs](https://open-eo.github.io/openeo-python-client/):

```python
# Scale values and take the temporal maximum across the loaded years
cube = cube.apply(lambda x: x / 1_000_000).max_time()
```

---

## Synchronous execution

`POST /result` executes a process graph in the foreground and returns the result immediately. Synchronous raster execution is intended for concrete export formats such as `NetCDF`, `GeoTIFF`, `PNG`, or `CSV`. Zarr datacube output is not served synchronously; use a batch job for that.

```python
result = conn.execute(cube.save_result(format="NetCDF"))
print(type(result))
```

Equivalent with curl:

```bash
curl -s -X POST http://127.0.0.1:9000/result \
  -H "Content-Type: application/json" \
  -d '{
    "process": {
      "process_graph": {
        "load": {
          "process_id": "load_collection",
          "arguments": {
            "id": "worldpop_population_global2_R2025A_100m",
            "temporal_extent": ["2020-01-01", "2021-01-01"],
            "spatial_extent": {"west": -13.3, "south": 7.0, "east": -10.3, "north": 10.0}
          }
        },
        "result": {
          "process_id": "save_result",
          "arguments": {"data": {"from_node": "load"}, "format": "NetCDF"},
          "result": true
        }
      }
    }
  }'
```

---

## Batch jobs

For long-running computations, create a batch job and poll its status.

```python
job = cube.create_job(title="worldpop-max-2015-2020")
job.start_job()

# Poll until finished
import time

while (status := job.status()) not in ("finished", "error"):
    print("status:", status)
    time.sleep(2)

# Retrieve result asset links
print(job.get_results().get_assets())
```

REST equivalent:

```bash
# 1 — create
curl -s -X POST http://127.0.0.1:9000/jobs \
  -H "Content-Type: application/json" \
  -d '{"process": {"process_graph": {...}}, "title": "my-job"}'

# 2 — start
curl -s -X POST http://127.0.0.1:9000/jobs/{job_id}/results

# 3 — poll
curl -s http://127.0.0.1:9000/jobs/{job_id}

# 4 — download result
curl -s http://127.0.0.1:9000/jobs/{job_id}/results
```

Completed batch jobs write their output to disk and expose it as an asset link at `GET /jobs/{id}/results/{filename}`. The output format is controlled by the `format` argument of `save_result` — see [Export formats](#export-formats) below.

---

## Available processes

`GET /processes` returns all 120+ standard openEO processes from [openeo-processes-dask](https://github.com/Open-EO/openeo-processes-dask), plus `load_collection` and `save_result` which are implemented by this backend. All processes listed are callable from process graphs.

Key processes for climate work:

| Process                     | What it does                                              |
| --------------------------- | --------------------------------------------------------- |
| `load_collection`           | Open a published dataset as an openEO data cube           |
| `filter_temporal`           | Restrict the time dimension to an interval                |
| `filter_bbox`               | Restrict the spatial extent                               |
| `filter_bands`              | Select a subset of variables/bands                        |
| `apply`                     | Apply an element-wise callback to every pixel             |
| `reduce_dimension`          | Collapse a dimension with a reducer (e.g. mean, sum)      |
| `aggregate_temporal_period` | Group by calendar period (month, season, year) and reduce |
| `aggregate_spatial`         | Zonal statistics over GeoJSON geometries                  |
| `resample_cube_spatial`     | Reproject and resample to a target grid                   |
| `merge_cubes`               | Combine two aligned cubes                                 |
| `save_result`               | Finalise the result — controls the output format          |

---

## Export formats

The `format` argument of `save_result` controls what the server writes. `GET /file_formats` advertises all supported formats to clients.

| Format key | Title      | Output type     | Notes                                                                                  |
| ---------- | ---------- | --------------- | -------------------------------------------------------------------------------------- |
| `ZARR`     | Zarr       | Raster          | Default. Zarr v3 directory store; served chunk-by-chunk                                |
| `NETCDF`   | NetCDF     | Raster          | Raw float values — compatible with CDO, NCO, xarray, R                                 |
| `GTIFF`    | GeoTIFF    | Raster          | Raw float values with embedded CRS — compatible with QGIS, GDAL                        |
| `PNG`      | PNG        | Raster          | Styled image using the collection's colormap and rescale range; transparent background |
| `CSV`      | CSV        | Raster / Vector | Tabular — ideal for time series and zonal statistics output                            |
| `GEOJSON`  | GeoJSON    | Vector          | Default for `aggregate_spatial` results; one feature per geometry                      |
| `PARQUET`  | GeoParquet | Vector          | Columnar binary — efficient for large vector datasets                                  |
| `DHIS2JSON` | DHIS2 JSON | Tabular        | DHIS2 `dataValueSet` — one value per org unit, period and data element                 |
| `CHAPCSV`  | CHAP CSV   | Tabular         | Wide CSV for CHAP: `time_period`, `location`, one column per variable                   |

For aggregating a dataset to DHIS2 org units and producing `DHIS2JSON` or `CHAPCSV` directly, see the built-in [org-unit aggregation workflows](workflows.md#built-in-workflows).

```bash
# Monthly precipitation totals as NetCDF
curl -X POST http://127.0.0.1:9000/result \
  -H "Content-Type: application/json" \
  -d '{
    "process": {
      "process_graph": {
        "load": { "process_id": "load_collection", "arguments": { "id": "chirps3_precipitation_daily", "temporal_extent": ["2026-01-01", "2026-03-31"] } },
        "agg":  { "process_id": "aggregate_temporal_period", "arguments": { "data": {"from_node": "load"}, "period": "month", "reducer": { "process_graph": { "sum": { "process_id": "sum", "arguments": { "data": {"from_parameter": "data"} }, "result": true } } } } },
        "save": { "process_id": "save_result", "arguments": { "data": {"from_node": "agg"}, "format": "NetCDF" }, "result": true }
      }
    }
  }' --output monthly_precip.nc
```

---

## User-defined processes (UDPs)

UDPs are named, parameterized process graphs stored server-side. They let you define reusable pipelines and invoke them by name from any other process graph.

```bash
# Store a UDP
curl -s -X PUT http://127.0.0.1:9000/process_graphs/pop_millions \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Load WorldPop population in millions",
    "parameters": [
      {"name": "temporal_extent", "schema": {"type": "array"}}
    ],
    "process_graph": {
      "load": {
        "process_id": "load_collection",
        "arguments": {
          "id": "worldpop_population_global2_R2025A_100m",
          "temporal_extent": {"from_parameter": "temporal_extent"}
        }
      },
      "scale": {
        "process_id": "apply",
        "arguments": {
          "data": {"from_node": "load"},
          "process": {
            "process_graph": {
              "div": {
                "process_id": "divide",
                "arguments": {"x": {"from_parameter": "x"}, "y": 1000000},
                "result": true
              }
            }
          }
        }
      },
      "result": {
        "process_id": "save_result",
        "arguments": {"data": {"from_node": "scale"}, "format": "Zarr"},
        "result": true
      }
    }
  }'

# Invoke it from another process graph
curl -s -X POST http://127.0.0.1:9000/result \
  -H "Content-Type: application/json" \
  -d '{
    "process": {
      "process_graph": {
        "run": {
          "process_id": "pop_millions",
          "arguments": {"temporal_extent": ["2020-01-01", "2025-01-01"]},
          "result": true
        }
      }
    }
  }'
```

---

## Custom processes (plugins)

Processing plugins are Python functions registered via YAML that extend the process library. A plugin with the same `id` as a standard process shadows the built-in. See [Extensibility — Processes](extensibility.md#processes) for the plugin contract.

---

## How the Open Climate Service implements openEO

```
openEO client
      │
      ▼
POST /result  ──────────────────────────────────────► immediate response
POST /jobs → POST /jobs/{id}/results → GET /jobs/{id}/results
      │
      ▼
openeo-pg-parser-networkx   ← parses the process graph DAG
      │
      ▼
openeo-processes-dask       ← executes each node (120+ standard processes)
      │
      ▼
load_collection             ← reads from Icechunk/Zarr managed dataset store
      │
      ▼
save_result                 ← writes output file; returns asset href
```

openEO is an additional access layer on top of the existing dataset store — the same data served via the native ingestion and sync endpoints is available through process graphs with no duplication.

---

## Examples

- [`examples/openeo_process_graph.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/openeo_process_graph.py) — full end-to-end walkthrough using the openEO Python client
- [`examples/zonal_statistics.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/zonal_statistics.py) — district-level statistics with DHIS2 organisation unit IDs via `aggregate_spatial` and `rename_labels`
- [`examples/aggregate_and_import_to_dhis2.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/aggregate_and_import_to_dhis2.py) — fetch org units from DHIS2, run the `aggregate_to_dhis2_json` workflow, and import the result back into DHIS2

---

## Resources

| Resource                                 | Link                                               |
| ---------------------------------------- | -------------------------------------------------- |
| openEO.org — overview and use cases      | <https://openeo.org>                               |
| API specification (v1.2.0)               | <https://openeo.org/documentation/1.0/api/>        |
| Standard process catalogue               | <https://processes.openeo.org>                     |
| Python client documentation              | <https://open-eo.github.io/openeo-python-client/>  |
| Web editor (hosted)                      | <https://editor.openeo.org>                        |
| openEO cookbook (Python examples)        | <https://openeo.org/documentation/1.0/cookbook/>   |
| openeo-processes-dask (execution engine) | <https://github.com/Open-EO/openeo-processes-dask> |
