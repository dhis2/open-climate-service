# Open Climate Service Managed Data Guide

This guide describes the current native FastAPI surface for Open Climate Service and how datasets are published as STAC collections.

The current public story is:

- run and inspect ingestion operations with `/ingestions`
- discover the configured extent with `/extent`
- discover managed datasets with `/datasets`
- discover published GeoZarr datasets with `/stac/catalog.json`
- access raw Zarr data with `/zarr/{dataset_id}` (vanilla zarr clients, web maps)
- access the native Icechunk store with `/icechunk/{dataset_id}` (Icechunk SDK)

Internal artifacts still exist as a storage and provenance model, but they are not part of the public API contract.

Operational note:

- `/ingestions` is the execution and admin-facing surface for ingestion runs
- `/datasets` is the canonical managed-data surface for consumers

## Main Public Endpoints

- `POST /ingestions`
- `GET /ingestions`
- `GET /ingestions/{ingestion_id}`
- `GET /extent`
- `GET /datasets`
- `GET /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/download`
- `GET /stac`
- `GET /stac/catalog.json`
- `GET /stac/collections/{dataset_id}`
- `GET /zarr/{dataset_id}`
- `GET /zarr/{dataset_id}/{relative_path}`
- `GET /icechunk/{dataset_id}/{path}`
- `GET /sync/{dataset_id}/plan`
- `POST /sync/{dataset_id}`

## 1. Discover the configured extent

The configured extent is setup-time Open Climate Service configuration. It is read-only at runtime.

Example:

```bash
curl -s http://127.0.0.1:9000/extent | jq
```

Example response:

```json
{
  "name": "Sierra Leone",
  "description": null,
  "bbox": [-13.5, 6.9, -10.1, 10.0]
}
```

What this means:

- `bbox` is the resolved spatial extent exposed publicly
- provider-specific hints may exist internally in extent config, but they are not part of the public extent response

## 2. Ingest a dataset

The public ingestion contract takes:

- `dataset_id`
- `start`
- optional `end`
- `overwrite`
- `publish`

Raw `bbox` and `country_code` are not part of the public ingestion payload — the API resolves them from the configured extent. All datasets are stored as Icechunk stores; there is no format selection parameter.

### Example: CHIRPS3

```bash
curl -s -X POST http://127.0.0.1:9000/ingestions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "chirps3_precipitation_daily",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "overwrite": false,
    "publish": true
  }' | jq
```

### Example: WorldPop

```bash
curl -s -X POST http://127.0.0.1:9000/ingestions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "worldpop_population_global2_R2025A_100m",
    "start": "2020",
    "end": "2020",
    "overwrite": false,
    "publish": true
  }' | jq
```

Example response:

```json
{
  "ingestion_id": "a7e06c93-ba78-4c74-b772-160927fdb463",
  "status": "completed",
  "dataset": {
    "dataset_id": "chirps3_precipitation_daily",
    "source_dataset_id": "chirps3_precipitation_daily",
    "dataset_name": "Total precipitation (CHIRPS3)",
    "short_name": "Total precipitation",
    "description": "CHIRPS v3 daily precipitation in mm.",
    "variable": "precip",
    "period_type": "daily",
    "units": "mm",
    "resolution": "5 km x 5 km",
    "source": "CHIRPS v3",
    "source_url": "https://www.chc.ucsb.edu/data/chirps3",
    "extent": {
      "spatial": {
        "xmin": -13.52499751932919,
        "ymin": 6.92499920912087,
        "xmax": -10.124997468665242,
        "ymax": 10.02499925531447
      },
      "temporal": {
        "start": "2024-01-01",
        "end": "2024-01-31"
      }
    },
    "last_updated": "2026-04-01T09:03:28.691120Z",
    "links": [
      {
        "href": "/datasets/chirps3_precipitation_daily",
        "rel": "self",
        "title": "Dataset detail"
      },
      {
        "href": "/zarr/chirps3_precipitation_daily",
        "rel": "zarr",
        "title": "Zarr store"
      },
      {
        "href": "/stac/collections/chirps3_precipitation_daily",
        "rel": "stac",
        "title": "STAC collection"
      }
    ],
    "publication": {
      "status": "published",
      "published_at": "2026-04-01T09:03:28.692230Z"
    }
  }
}
```

What this means:

- `ingestion_id` is the handle for the ingestion event lookup route
- `status = "completed"` means this branch still treats ingestion synchronously
- `/ingestions` is an operational/admin surface, not the main managed-data catalog
- `dataset` is a public managed dataset summary, not an internal artifact record
- `extent` is realized data coverage, not just the configured bbox
- `links` point to the native dataset metadata, native Zarr access, and STAC collection metadata

## 3. List ingestion runs

`GET /ingestions` returns ingestion run records for operational and admin use.

Example:

```bash
curl -s http://127.0.0.1:9000/ingestions | jq
```

What this means:

- this route is for execution lookup and operational visibility
- it is not intended to replace `/datasets` as the primary data discovery surface
- items are ordered from most recent ingestion to oldest
## 4. Ingestion failure behavior

Ingestion should fail gracefully with a structured API error, not a raw 500 stack trace.

Current behavior:

- invalid or missing spatial/config inputs return `400`
- dataset/provider execution failures return `502`

Example cases:

- a provider requires a country code and the resolved extent config does not provide one
- a dataset requires a bbox and no bbox can be resolved
- the upstream provider fails at download time

Example error response:

```json
{
  "detail": "Upstream dataset download failed: provider timeout"
}
```

## 5. Discover managed datasets

`GET /datasets` is the native managed-data catalog and the main consumer-facing data surface.

Example:

```bash
curl -s http://127.0.0.1:9000/datasets | jq
```

Example response:

```json
{
  "kind": "DatasetList",
  "items": [
    {
      "dataset_id": "chirps3_precipitation_daily",
      "source_dataset_id": "chirps3_precipitation_daily",
      "dataset_name": "Total precipitation (CHIRPS3)",
      "short_name": "Total precipitation",
      "description": "CHIRPS v3 daily precipitation in mm.",
      "variable": "precip",
      "period_type": "daily",
      "units": "mm",
      "resolution": "5 km x 5 km",
      "source": "CHIRPS v3",
      "source_url": "https://www.chc.ucsb.edu/data/chirps3",
      "extent": {
        "spatial": {
          "xmin": -13.52499751932919,
          "ymin": 6.92499920912087,
          "xmax": -10.124997468665242,
          "ymax": 10.02499925531447
        },
        "temporal": {
          "start": "2024-01-01",
          "end": "2024-01-31"
        }
      },
      "last_updated": "2026-04-01T09:03:28.691120Z",
      "links": [
        {
          "href": "/datasets/chirps3_precipitation_daily",
          "rel": "self",
          "title": "Dataset detail"
        },
        {
          "href": "/zarr/chirps3_precipitation_daily",
          "rel": "zarr",
          "title": "Zarr store"
        }
      ],
      "publication": {
        "status": "published",
        "published_at": "2026-04-01T09:03:28.692230Z"
      }
    }
  ]
}
```

What this means:

- `/datasets` is the public native catalog of managed datasets
- `description` carries the dataset template's own prose, and is `null` when the template
  declares none. It is where a dataset states what its values actually mean, so it is worth
  reading before using one: `chirps3_precipitation_monthly` is a mean daily rate rather than a
  monthly total. The same text is published as the STAC collection `description`.
- `items` is wrapped in a `kind` envelope for consistency and self-description
- dataset items contain public metadata and access links only
- internal artifact ids, filesystem paths, and downloader implementation details are intentionally omitted

## 6. Get dataset detail

`GET /datasets/{dataset_id}` returns the full managed dataset detail view.

Example:

```bash
curl -s http://127.0.0.1:9000/datasets/chirps3_precipitation_daily | jq
```

What this adds beyond the list response:

- full dataset metadata
- publication summary
- slim `versions` history derived from internal records

The detailed dataset response is where version history belongs. The ingestion response stays as a summary.

## 7. Access raw Zarr data

`/zarr/{dataset_id}` bridges the Icechunk store into standard zarr HTTP semantics. It works with any vanilla zarr client including xarray.

Examples:

```bash
curl -s http://127.0.0.1:9000/zarr/chirps3_precipitation_daily/zarr.json | jq
```

`/zarr/{dataset_id}` is the store prefix; the root metadata is at `/zarr/{dataset_id}/zarr.json`.

## 8. Access the Icechunk store natively

`/icechunk/{dataset_id}/{path}` serves raw Icechunk store files for native SDK access. Use this when you need versioning or want to avoid the zarr proxy layer.

```python
import icechunk, xarray as xr

repo = icechunk.Repository.open(icechunk.http_storage("http://127.0.0.1:9000/icechunk/chirps3_precipitation_daily"))
ds = xr.open_zarr(repo.readonly_session("main").store, zarr_format=3, consolidated=False)
```

Both endpoints are advertised as `assets` in the STAC collection:

```json
"assets": {
  "zarr": {
    "href": "https://host/zarr/chirps3_precipitation_daily",
    "type": "application/vnd.zarr; version=3",
    "xarray:open_kwargs": { "consolidated": true }
  },
  "icechunk": {
    "href": "https://host/icechunk/chirps3_precipitation_daily",
    "type": "application/octet-stream",
    "xarray:open_kwargs": { "zarr_format": 3, "consolidated": false }
  }
}
```

A **pyramided** store advertises one extra media type parameter:

```json
"type": "application/vnd.zarr; version=3; profile=multiscales"
```

This is how a client learns the store has resolution levels — the plain Zarr media type cannot
express it, and clients that only render pyramided stores (STAC Browser, via
`ol/source/GeoZarr`) will not offer to display a store without it. The parameter is only added
when the store genuinely has a non-empty `multiscales.layout`; claiming it for a flat store
would send a renderer looking for levels that do not exist.

It is matched as a literal, not parsed, so the string is byte-for-byte fixed: same parameter
order, one space after each `;`. Reformatting it silently disables rendering.

## 10. Access published STAC collections

Published Zarr-backed datasets are exposed through `/stac` for discovery.

STAC examples:

```bash
curl -s "http://127.0.0.1:9000/stac/catalog.json" | jq
curl -s "http://127.0.0.1:9000/stac/collections/chirps3_precipitation_daily" | jq
```

What this means:

- `/stac` is the public STAC discovery surface for published Zarr-backed datasets
- native FastAPI no longer exposes `/collections`
- dataset responses include `/stac/collections/{dataset_id}`

## 11. `/sync`

`/sync` advances an existing managed dataset from its latest local coverage toward a requested upstream period.

Available operations:

- `GET /sync/{dataset_id}/plan?end={period}` returns the planned sync action without downloading or writing data
- `POST /sync/{dataset_id}` executes the plan when a new version is needed

Implemented behavior:

- `temporal` datasets compare the next missing period with the requested or metadata-clamped latest period
- `release` datasets compare the current materialized release with the requested or metadata-clamped latest release
- `static` datasets return `not_syncable`
- preserve stable managed dataset identity
- use template-level `sync_execution`
- `append` execution downloads only the missing period range, then rebuilds the canonical artifact from the local cache
- `rematerialize` execution downloads the full original request range through the requested end period
- return the updated dataset view plus structured `sync_detail`
- reject a rebuilt artifact before storing or publishing it if realized temporal coverage does not match the requested scope

Current sync constraints:

- append execution is a delta-download plus canonical rebuild, not in-place Zarr mutation
- upstream availability is determined by each plugin's `periods()` method

Configured availability policies:

- CHIRPS3 daily uses `open_climate_service.providers.availability.chirps3_daily_latest_available`; this clamps sync targets to the latest complete released source month
- ERA5-Land hourly uses `open_climate_service.providers.availability.lagged_latest_available` with a YAML-declared `lag_hours`
- WorldPop yearly uses `open_climate_service.providers.availability.worldpop_release_latest_available`; this can allow configured future projection years

Example dry-run plan:

```bash
curl -s "http://127.0.0.1:9000/sync/chirps3_precipitation_daily/plan?end=2024-02-10" | jq
```

Example execution:

```bash
curl -s -X POST "http://127.0.0.1:9000/sync/chirps3_precipitation_daily" \
  -H "Content-Type: application/json" \
  -d '{"end":"2024-02-10","publish":true}' | jq
```

## Manual Test Sequence

These commands assume the API is running on `http://127.0.0.1:9000` and that
`jq` is available.

### 1. Confirm configured extent

```bash
curl -s "http://127.0.0.1:9000/extent" | jq
```

### 2. Create an initial CHIRPS3 managed dataset

```bash
curl -s -X POST "http://127.0.0.1:9000/ingestions" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "chirps3_precipitation_daily",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "publish": true
  }' | jq
```

Expected:

- `status` is `completed`
- `dataset.dataset_id` is `chirps3_precipitation_daily`
- `dataset.extent.temporal.end` is `2024-01-31`

### 3. Inspect the managed dataset and publication

```bash
curl -s "http://127.0.0.1:9000/datasets/chirps3_precipitation_daily" | jq
curl -s "http://127.0.0.1:9000/stac/collections/chirps3_precipitation_daily" | jq
curl -s "http://127.0.0.1:9000/zarr/chirps3_precipitation_daily" | jq
```

### 4. Dry-run a CHIRPS3 append sync

```bash
curl -s "http://127.0.0.1:9000/sync/chirps3_precipitation_daily/plan?end=2024-02-10" | jq
```

Expected planning response:

- `sync_kind` is `temporal`
- `action` is `append`
- `reason` is `new_periods_available_for_append`
- `message` explains that existing data is present and which missing period range will be downloaded
- `current_start` is `2024-01-01`
- `current_end` is `2024-01-31`
- `target_end` is `2024-02-10`
- `target_end_source` is `request`
- `delta_start` is `2024-02-01`
- `delta_end` is `2024-02-10`

`append` here means Open Climate Service downloads only the missing period range and then
rebuilds the canonical artifact from local cache. It is not in-place Zarr mutation.

Where these timestamps come from:

- `current_start` and `current_end` come from the latest stored artifact coverage
- `target_end` comes from the explicit `end` query parameter, or defaults to today in the dataset-native period format when omitted
- `target_end_source` tells you whether `target_end` came from `request`, `default_today`, `current_coverage`, or was clamped by source availability
- `delta_start` is the first period after `current_end`
- `delta_end` is the resolved target period after any availability clamping

If `end` is omitted, the planner defaults to the current date. For example, calling
`/sync/chirps3_precipitation_daily/plan` on `2026-04-20` after ingesting
through `2024-01-31` first resolves the target from today's date, then applies
CHIRPS3 availability. Because CHIRPS3 daily sync is configured to use complete
released source months, the target may be clamped below today's date and
`target_end_source` will be `default_today_clamped_by_availability`.

For controlled tests, always pass an explicit `end`. If the explicit `end`
extends beyond the configured provider availability, `target_end_source` will be
`request_clamped_by_availability`.

### Availability clamping example

If CHIRPS3 currently has complete released data through `2026-03-31` and you ask
for:

```bash
curl -s "http://127.0.0.1:9000/sync/chirps3_precipitation_daily/plan?end=2026-04-21" | jq
```

Expected:

- `target_end` is `2026-03-31`
- `target_end_source` is `request_clamped_by_availability`
- the sync does not ask the upstream downloader for unavailable April daily data

### 5. Execute the CHIRPS3 sync

```bash
curl -s -X POST "http://127.0.0.1:9000/sync/chirps3_precipitation_daily" \
  -H "Content-Type: application/json" \
  -d '{
    "end": "2024-02-10",
    "publish": true
  }' | jq
```

Expected:

- `status` is `completed`
- `sync_detail.action` is `append`
- `sync_detail.current_end` was `2024-01-31`
- `sync_detail.delta_start` is `2024-02-01`
- `sync_detail.delta_end` is `2024-02-10`
- `sync_detail.target_end` is `2024-02-10`
- the returned `dataset.dataset_id` is still `chirps3_precipitation_daily`
- the returned dataset has a newer version in `versions`
- the returned dataset coverage ends at `2024-02-10`, even if the upstream downloader cached the full February source month

You can then extend the same managed dataset again:

```bash
curl -s -X POST "http://127.0.0.1:9000/sync/chirps3_precipitation_daily" \
  -H "Content-Type: application/json" \
  -d '{
    "end": "2024-02-20",
    "publish": true
  }' | jq
```

Expected:

- `sync_detail.current_end` was `2024-02-10`
- `sync_detail.delta_start` is `2024-02-11`
- `sync_detail.delta_end` is `2024-02-20`
- returned dataset coverage ends at `2024-02-20`
- execution may be fast when the provider cache already contains the needed source files

### 6. Confirm no-op behavior

Run the same plan again with the current end:

```bash
curl -s "http://127.0.0.1:9000/sync/chirps3_precipitation_daily/plan?end=2024-02-20" | jq
```

Expected:

- `action` is `no_op`
- `reason` is `no_new_period`
- `message` explains that the requested target is already covered locally

### 7. Test release-style sync with WorldPop

Create an initial WorldPop managed dataset:

```bash
curl -s -X POST "http://127.0.0.1:9000/ingestions" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "worldpop_population_global2_R2025A_100m",
    "start": "2020",
    "end": "2020",
    "publish": true
  }' | jq
```

Plan a later release:

```bash
curl -s "http://127.0.0.1:9000/sync/worldpop_population_global2_R2025A_100m/plan?end=2021" | jq
```

Expected:

- `sync_kind` is `release`
- `action` is `rematerialize`
- `reason` is `new_release_available`
- `target_end` is `2021`

Execute the release sync:

```bash
curl -s -X POST "http://127.0.0.1:9000/sync/worldpop_population_global2_R2025A_100m" \
  -H "Content-Type: application/json" \
  -d '{
    "end": "2021",
    "publish": true
  }' | jq
```

Expected:

- `status` is `completed`
- `sync_detail.action` is `rematerialize`
- `dataset.dataset_id` is `worldpop_population_global2_R2025A_100m`

## Summary

The current branch is no longer an artifact-first API.

The public contract is now:

- ingest with `/ingestions`
- discover the configured extent with `/extent`
- discover managed datasets with `/datasets`
- discover published Zarr-backed datasets with `/stac/catalog.json`
- access raw native data with `/zarr/{dataset_id}`

Artifacts remain internal because Open Climate Service still needs storage and provenance records behind ingestion and publication, but those internals are no longer exposed as first-class public resources.
