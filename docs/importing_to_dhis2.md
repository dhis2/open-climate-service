# Importing data to DHIS2 and Chap

This guide shows the full round-trip: pull organisation unit boundaries from DHIS2, aggregate a published dataset to those org units with Open Climate Service, and import the result back into DHIS2 as data values.

The spatial aggregation happens via the built-in [`aggregate_to_dhis2_json`](workflows.md#built-in-workflows) workflow. It saves its result through a named export declared in the instance configuration, which maps the aggregate to DHIS2 data elements and periods. Because each org unit's GeoJSON `id` is its DHIS2 UID, the result imports without any remapping.

There are two ways to get the values into DHIS2:

- **Client import.** Run the workflow synchronously and import the returned `dataValueSet` with a DHIS2 client.
- **Server delivery.** Run the workflow as a batch job and let OCS deliver the saved result. See [Automated delivery with a named export](#automated-delivery-with-a-named-export).

## Prerequisites

- A running Open Climate Service instance with the dataset published (see [Accessing data](user_guide.md)).
- A DHIS2 instance whose organisation units have geometry, plus a data element to import into. The data element referenced in the payload must already exist in DHIS2 — see the DHIS2 Climate Tools [Prepare metadata](https://climate-tools.dhis2.org/guides/import-data/prepare-metadata/) guide for creating it.
- The two clients, installed together through the `dhis2` extra:

  ```bash
  pip install "open-climate-service[dhis2]>=0.1.1"
  ```

  `open-climate-service` ships the `ClimateService` client; the extra adds
  [dhis2-client](https://pypi.org/project/dhis2-client/), which handles the DHIS2 Web API calls.
  From a checkout of this repository, use `uv run --extra dhis2` for a client-side script.

## Named connections for server-side plugins

Exporters and feature providers running inside an OCS instance can share a named
DHIS2 connection. Configure it in the file referenced by `CLIMATE_SERVICE_CONFIG`:

```yaml
dhis2_connections:
  - id: national-hmis
    url: https://hmis.example.org/dhis
    token_env: DHIS2_IMPORT_TOKEN
    timeout: 30
    connect_timeout: 10
    retries: 3
```

Use the instance URL, including its deployment path if any, without appending
`/api`. The timeout fields are positive seconds; `retries` is an integer from 0 to
10. The shown values are defaults. TLS verification is enabled. Connection IDs
must be unique and contain letters, digits, underscores, or hyphens.

Supply the raw personal access token in the server process's `DHIS2_IMPORT_TOKEN`
environment variable. Connection fields are literal: use `token_env: DHIS2_IMPORT_TOKEN`,
not `token_env: ${DHIS2_IMPORT_TOKEN}`. Embedded credentials, token/password fields,
and URLs with query strings or fragments are rejected. Configuration using this
section must be valid YAML before environment substitution; quote placeholders in
other sections when necessary. Existing interpolation outside this section remains
available.

The client is the `dhis2` extra, `dhis2-client>=0.3.2` from PyPI. The official Docker image
installs it. An instance managed with uv names the extra beside `server` in its dependency,
`open-climate-service[server,dhis2]`, and re-locks; without it, named connections fail at use
time with a `RuntimeError` naming the extra.

An independently installed provider can use the public accessor without reading
OCS configuration or handling credentials itself:

```python
from contextlib import closing

from open_climate_service.exports.dhis2 import get_connection

with closing(get_connection("national-hmis")) as dhis2:
    org_units = dhis2.get_org_units_geojson(level=2)
```

Each call creates a client and resolves the current token. The caller must close
the client; `contextlib.closing` also closes it if the operation fails. Creating a
client sends no network requests. A new call picks up token rotation, while an
already open client retains its original token. `get_connection_config(id)` returns
only non-secret settings and works without the optional client or a configured token.

The tested client retries GET responses with server errors; it does not
automatically replay POST requests or retry transport exceptions. Delivery jobs
will own import recovery and reports. This connection helper does not introduce
an export endpoint, bypass read-only guards, or authorize operations: the calling
operator command or future HTTP route remains responsible for those controls.

## 1. Configure a named export

Declare the export in the file referenced by `CLIMATE_SERVICE_CONFIG`:

```yaml
exports:
  - id: rainfall-monthly
    plugin: dhis2
    dataset: era5land_precipitation_monthly
    connection: national-hmis
    aggregation: mean
    period_type: monthly
    series:
      - select: {}
        data_element: BXgDHhPdFVU
```

Replace the data element UID with one that exists in your DHIS2 instance. The
`period_type` must match the dataset's native temporal resolution.

The other fields are optional for client import:

- `dataset` is checked against the dataset the job actually loaded.
- `aggregation` must equal the workflow's `method`; a run with a different method fails, synchronously or as a batch job.
- `connection` names a [DHIS2 connection](#named-connections-for-server-side-plugins) and is required for server delivery.

See [Export plugins and named mappings](export_plugins.md) for all mapping fields,
including several series and category option combos. Restart the service after
changing the configuration.

## 2. Fetch organisation units from DHIS2

For reusable destination mappings and pure rendering through `save_result`, see
[Export plugins and named mappings](export_plugins.md).

Pull the org unit boundaries as GeoJSON. Each feature's `id` is the org unit UID, which the workflow uses as the `orgUnit`.

!!! tip
    On a running instance, let OCS fetch and store them instead, with the built-in `dhis2` feature provider: see [DHIS2 organisation units](built_in_datasets.md#dhis2-organisation-units-feature-collection). Workflows then refer to them by id rather than carrying the GeoJSON.

```python
from dhis2_client import DHIS2Client
from dhis2_client.settings import ClientSettings

dhis2 = DHIS2Client(
    settings=ClientSettings(
        base_url="https://my-dhis2.example.org",
        username="admin",
        password="district",
    )
)

org_units = dhis2.get_org_units_geojson(level=2)
print(len(org_units["features"]), "org units")
```

!!! note
`admin` / `district` are the public DHIS2 demo credentials, shown for illustration — point the client at your own instance and credentials.

## 3. Aggregate on Open Climate Service

Run the `aggregate_to_dhis2_json` workflow with `ClimateService.execute()`. It loads the dataset over the time range, aggregates it within each org-unit polygon, and returns the `dataValueSet` rendered by the named export.

```python
from open_climate_service import ClimateService

service = ClimateService("http://127.0.0.1:9000")

data_value_set = service.execute(
    {
        "agg": {
            "process_id": "aggregate_to_dhis2_json",
            "arguments": {
                "dataset_id": "era5land_precipitation_monthly",  # a published collection
                "temporal_extent": ["2025-01-01", "2025-12-31"],
                "geometries": org_units,
                "export": "rainfall-monthly",  # the named export from step 1
                "method": "mean",  # mean (default), min, max, sum, or median
            },
            "result": True,
        }
    }
)

print(len(data_value_set["dataValues"]), "data values")
```

The export fills in `orgUnit`, `period`, `value`, and `dataElement` for every cell — a valid DHIS2 `dataValueSet`, ready to import as-is. Every feature's `id` must be its DHIS2 organisation unit UID. Features with a missing, duplicate, or non-UID `id` are rejected before aggregation starts.

## 4. Import into DHIS2

```python
report = dhis2.post_data_value_set(data_value_set)
print(report["response"]["importCount"])
```

## Automated delivery with a named export

Instead of importing the payload from a client, OCS can deliver it. This needs a
`connection` on the export from step 1 and a writable instance.

### Run the workflow as a batch job

Submit the same process graph to `POST /jobs` and start it with
`POST /jobs/{job_id}/results`:

```json
{
  "process": {
    "process_graph": {
      "agg": {
        "process_id": "aggregate_to_dhis2_json",
        "arguments": {
          "dataset_id": "era5land_precipitation_monthly",
          "temporal_extent": ["2025-01-01", "2025-12-31"],
          "geometries": { "type": "FeatureCollection", "features": ["...org units..."] },
          "export": "rainfall-monthly",
          "method": "mean"
        },
        "result": true
      }
    }
  }
}
```

The finished job exposes the payload and a delivery manifest as result assets.
The manifest binds the payload to the export mapping, renderer version, DHIS2
target, observed dataset, and executed aggregation method. A synchronous
`POST /result` returns the same payload without a manifest, so its result cannot be
delivered by the server.

To run the workflow whenever a dataset updates, reference it from an
`automation.workflow_triggers` entry with the same arguments. See
`climate-service.yaml.example`.

### Deliver the result

Submit the finished job to its export. Start with a dry run:

```http
POST /exports/rainfall-monthly
Idempotency-Key: rainfall-2025-dry-run
Content-Type: application/json

{"job_id": "<job_id>", "dry_run": true}
```

The response is `202 Accepted` with a delivery job ID and status URL. When the
dry-run report's `outcome` is `dry_run`, repeat the request with `"dry_run": false`
and a new idempotency key. The server process needs the connection's token in its
environment, for example `DHIS2_IMPORT_TOKEN`.

Delivery refuses a result if its payload, mapping, renderer, or target has changed
since the job ran. Re-run the workflow after changing the export configuration.
[Export plugins and named mappings](export_plugins.md#deliver-a-saved-export)
describes chunking, recovery, and reports.

## Ad-hoc DHIS2 JSON without a named export

To render a payload without instance configuration, write the graph yourself and
pass the mapping as `save_result` options:

```json
{
  "load": {
    "process_id": "load_collection",
    "arguments": {"id": "era5land_precipitation_monthly", "temporal_extent": ["2025-01-01", "2025-12-31"]}
  },
  "zonal": {
    "process_id": "aggregate_spatial",
    "arguments": {
      "data": {"from_node": "load"},
      "geometries": {"type": "FeatureCollection", "features": ["...org units..."]},
      "reducer": {
        "process_graph": {
          "mean": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": true}
        }
      }
    }
  },
  "save": {
    "process_id": "save_result",
    "arguments": {
      "data": {"from_node": "zonal"},
      "format": "DHIS2JSON",
      "options": {"data_element_id": "BXgDHhPdFVU", "org_unit_field": "geometry", "period_type": "month"}
    },
    "result": true
  }
}
```

This is a lower-level escape hatch for experiments and one-off imports. The payload
is downloadable but has no delivery manifest, so OCS cannot deliver it.

## Producing a CHAP CSV instead

To feed the [Chap Modeling Platform](https://chap.dhis2.org/chap-modeling-platform/) rather than DHIS2 data values, use the companion [`aggregate_to_chap_csv`](workflows.md#built-in-workflows) workflow — same call, but pass `period_type` instead of `export`; it needs no named export and returns a Chap-ready CSV instead of a `dataValueSet`. Pass `path=` to write it to disk:

```python
csv_path = service.execute(
    {
        "agg": {
            "process_id": "aggregate_to_chap_csv",
            "arguments": {
                "dataset_id": "era5land_precipitation_monthly",
                "temporal_extent": ["2025-01-01", "2025-12-31"],
                "geometries": org_units,
                "method": "mean",
                "period_type": "month",
            },
            "result": True,
        }
    },
    path="climate-monthly.csv",
)
```

## See also

- [Workflows](workflows.md) — reference for the `aggregate_to_dhis2_json` and `aggregate_to_chap_csv` workflows.
- [Export plugins and named mappings](export_plugins.md) — named export fields and delivery behaviour.
- [openEO](openeo.md) — process graphs, export formats, and more examples.
