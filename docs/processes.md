# Processes

The Open Geo Service exposes two complementary processing interfaces:

- **openEO process graphs** — the primary interface for data analysis. Submit a DAG of composable operations via `POST /result` (synchronous) or `POST /jobs` (batch). 120+ standard processes are available out of the box. See the [openEO guide](openeo.md).
- **Plugin processes** — custom named processes registered via the `@process` decorator, discoverable at `GET /processes` and callable directly by `process_id` in any openEO process graph. See [Extensibility — Processes](extensibility.md#processes).

---

## Temporal resampling

Temporal resampling (e.g. daily → monthly, hourly → daily) is handled by the standard openEO `aggregate_temporal_period` process:

```json
{
  "process_graph": {
    "load": {
      "process_id": "load_collection",
      "arguments": { "id": "era5land_temperature_hourly" }
    },
    "resample": {
      "process_id": "aggregate_temporal_period",
      "arguments": {
        "data": { "from_node": "load" },
        "period": "day",
        "reducer": {
          "process_graph": {
            "mean": {
              "process_id": "mean",
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

See the [openEO guide](openeo.md) and the [openEO process specification](https://processes.openeo.org/#aggregate_temporal_period) for the full parameter reference.
