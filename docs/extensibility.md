# Extensibility

The Open Climate Service supports three plugin types, all following the same pattern: place files in the appropriate subdirectory of `plugins_dir` and the service picks them up automatically — no forking or patching of core code required.

| Plugin type | Location | Format |
| ----------- | -------- | ------ |
| Datasets | `plugins_dir/datasets/` | `.yaml` + `.py` |
| Processes | `plugins_dir/processes/` | `.py` |
| Workflows | `plugins_dir/workflows/` | `.json` |

The same three plugin types can also be packaged and **installed** (`uv add`), so a reusable plugin
is shared across instances without any `plugins_dir` wiring — see [Installable plugins](installable_plugins.md).
`plugins_dir` still takes precedence, so it can override an installed plugin locally.

---

## Datasets

Dataset templates are YAML files that describe a data source. Built-ins live in the package (`open_climate_service/plugins/datasets/`). Custom templates are loaded from `plugins_dir/datasets/`.

```
plugins/
└── datasets/
    ├── enacts_rainfall.yaml    # dataset template
    └── enacts.py               # streaming plugin class
```

```yaml
# climate-service.yaml
plugins_dir: ./plugins/
```

All `*.yaml` files in `plugins_dir/datasets/` are merged with the built-ins. A custom template with the same `id` as a built-in overrides it — useful for adjusting display ranges or availability settings on an existing dataset.

A Python plugin class is declared alongside the YAML using the `ingestion.plugin` dotted path. Plugins subclass `BaseDatasetPlugin` and implement just `periods()` and `fetch_period()` — the base class provides the concurrency defaults and canonical dimension names. `fetch_period` is a regular (blocking) method run in a worker thread, or an `async def` for natively-async sources. Any data transformations (unit conversion, dimension renaming, nodata masking, bbox clipping) are applied inside the fetch before the `xr.Dataset` is returned, typically via the `normalize_period` helper. The grid (shape, dtype, nodata, CRS) is inferred from the first fetched period; a projected-grid source declares its CRS via the `crs` class attribute.

See [Adding custom datasets](adding_custom_datasets.md) for the full template field reference, the streaming plugin contract, and the available helpers.

---

## Processes

Custom processes are Python functions decorated with `@process` and placed in `plugins_dir/processes/`. They appear in `GET /processes` alongside standard openEO processes and are callable directly by `process_id` in any process graph.

```python
# plugins/processes/indices.py
import xarray as xr
from open_climate_service.process import process


@process(summary="Precipitation anomaly relative to a baseline mean")
def precip_anomaly(pr: xr.DataArray, baseline: float = 0.0) -> xr.DataArray:
    """Deviation of precipitation from a long-term baseline mean."""
    return pr - baseline
```

The `@process` decorator derives the process id (function name), summary, parameter names, types, and defaults from the function signature and docstring. Use explicit metadata to override descriptions:

```python
@process(
    summary="Precipitation anomaly relative to a baseline mean",
    parameters={"baseline": {"description": "Long-term mean precipitation (kg m-2 s-1)."}},
)
def precip_anomaly(pr: xr.DataArray, baseline: float = 0.0) -> xr.DataArray: ...
```

A plugin process with the same id as an existing process overrides it. The server must be restarted to pick up new process files. For built-in climate indices, see [Climate indices](climate_indices.md).

### Scientific libraries available to your process

These are declared dependencies of the server, so you can import them in a plugin process without adding anything to your instance's `pyproject.toml`:

| Library | Use it for |
|---|---|
| [xarray](https://docs.xarray.dev) | The cube type every process takes and returns |
| [numpy](https://numpy.org), [pandas](https://pandas.pydata.org) | Array and time-series primitives |
| [xclim](https://xclim.readthedocs.io) | Peer-reviewed climate indices — 179 indicators, already exposed as processes (see [Climate indices](climate_indices.md)) |
| [earthkit](https://earthkit.ecmwf.int) | ECMWF's toolkit: `earthkit.transforms` (climatologies, anomalies, deaccumulation), `earthkit.meteo` (thermodynamics) |
| [rioxarray](https://corteva.github.io/rioxarray), [metpy](https://unidata.github.io/MetPy) | Raster/CRS operations and meteorological calculations |

Anything outside this list belongs in your own instance's dependencies — the plugin that ships the import owns it. Note that plugin modules are imported **lazily, when a process runs or an ingest starts**, so a missing dependency will not show up at startup: the API boots cleanly and the dataset still lists in `/collections`, then the run fails. Declaring it is the only reliable fix.

### Choosing between a standard process and a library

Reach for the standard openEO process first. Roughly 130 are available out of the box, and they are what every openEO client, tutorial and process graph already speaks — a locally named equivalent fragments that vocabulary.

- **openEO names it** → use it. Temporal and spatial aggregation, reductions, resampling, and the `mean`/`median`/`sd`/`quantiles` reducers are all standard: `aggregate_temporal`, `aggregate_temporal_period`, `aggregate_spatial`, `reduce_dimension`, `resample_cube_temporal`.
- **openEO does not name it** → wrap a library in a `@process` function. Climatological normals, anomalies and deaccumulation have no standard equivalent, which is why they exist here as named processes.

This is also why the two libraries are surfaced differently. xclim's indicators and earthkit-meteo's thermodynamic functions are auto-registered, so they appear in `GET /processes` individually — each is a distinct scientific quantity openEO does not define. earthkit-transforms is deliberately *not* auto-registered: most of it duplicates the standard aggregation processes, and its public functions are decorator-wrapped down to `(*args, **kwargs)`, leaving no signature from which to derive a usable process description. Call it from a hand-written `@process` instead, where you control the parameters and documentation.

### Units are part of the contract

Processes that wrap unit-sensitive physics should validate their inputs rather than trust them. Stored variables are CF-stamped from the `units` field of their dataset template, and those units are whatever the dataset declares — ERA5-Land temperature, for example, is converted to `degC` at ingest, while ECMWF library functions expect kelvin. Passing one for the other raises no error and produces a plausible, wrong number.

The auto-registered earthkit-meteo processes handle this for you: each reads the unit its upstream function documents, converts a compatible cube (`degC` → `K`, `hPa` → `Pa`), and refuses a cube whose units are missing or incompatible. The check does not trust upstream docstrings to be well-formed: a parameter documented as taking a cube but carrying no unit the adapter can enforce is *refused registration* rather than quietly advertised as a plain number, and a small override table supplies units for known upstream documentation defects. If you write a process with the same sensitivity, do the same — and make sure your dataset templates declare `units`, since that is what makes the check possible.

---

## Workflows

Reusable pipeline compositions are implemented as **UDPs** (User Defined Processes) — JSON process graph files placed in `plugins_dir/workflows/`. A UDP is a named, parameterised composition of existing openEO processes callable by name from any openEO client.

```
plugins/
└── workflows/
    └── monthly_rainfall.json
```

### Example: monthly rainfall totals

```json
{
  "id": "monthly_rainfall",
  "summary": "Monthly total precipitation for a collection and time range",
  "parameters": [
    {"name": "collection_id", "description": "Collection to load", "schema": {"type": "string"}},
    {"name": "temporal_extent", "description": "Time range [start, end]", "schema": {"type": "array"}}
  ],
  "process_graph": {
    "load": {
      "process_id": "load_collection",
      "arguments": {
        "id": {"from_parameter": "collection_id"},
        "temporal_extent": {"from_parameter": "temporal_extent"}
      }
    },
    "aggregate": {
      "process_id": "aggregate_temporal_period",
      "arguments": {
        "data": {"from_node": "load"},
        "period": "month",
        "reducer": {
          "process_graph": {
            "sum": {
              "process_id": "sum",
              "arguments": {"data": {"from_parameter": "data"}},
              "result": true
            }
          }
        }
      }
    },
    "save": {
      "process_id": "save_result",
      "arguments": {"data": {"from_node": "aggregate"}, "format": "Zarr"},
      "result": true
    }
  }
}
```

Calling the workflow from any openEO client:

```python
import openeo

conn = openeo.connect("http://your-instance:9000")
job = conn.execute_batch_job(
    {
        "process_graph": {
            "result": {
                "process_id": "monthly_rainfall",
                "arguments": {
                    "collection_id": "chirps_rainfall_daily",
                    "temporal_extent": ["2020-01-01", "2023-12-31"],
                },
                "result": true,
            }
        }
    }
)
```

Workflow JSON files are loaded on each request to `GET /process_graphs`, so changes on disk take effect without restarting the server. A plugin workflow with the same `id` as a built-in overrides it.
