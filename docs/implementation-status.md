# Implementation Status

## Purpose

This note captures the current implementation state of the branch after the API consolidation around ingestions, datasets, extents, raw Zarr access, and STAC discovery and publication.

It is intended to answer:

1. what the main branch now exposes
2. what is intentionally internal
3. how the current pieces fit together
4. what remains to be refined

## Current API surface

The main branch now centers on one narrow vertical slice:

1. define dataset templates in the Open Climate Service registry
2. define configured extents for the Open Climate Service instance
3. ingest data into a managed dataset for one dataset template plus one extent
4. publish that managed dataset as a STAC collection under `/stac`
5. expose native metadata under `/datasets`, STAC discovery under `/stac`, and raw Zarr access under `/zarr`
6. sync existing managed datasets forward through `/sync`

The public surface is intentionally small:

- `/ingestions`
- `/extent`
- `/datasets`
- `/stac/...`
- `/zarr/{dataset_id}`
- `/sync/{dataset_id}`

## Main Code References

- [open_climate_service/main.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/main.py)
  - app assembly and router mounting
- [open_climate_service/ingestions/routes.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/ingestions/routes.py)
  - ingestion, dataset, zarr, and sync routes
- [open_climate_service/ingestions/services.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/ingestions/services.py)
  - internal artifact persistence, dataset grouping, sync service wiring, Zarr browsing
- [open_climate_service/ingestions/sync_engine.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/ingestions/sync_engine.py)
  - sync planning and execution engine
- [open_climate_service/ingestions/schemas.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/ingestions/schemas.py)
  - public ingestion, dataset, and sync contracts
- [open_climate_service/extents/routes.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/extents/routes.py)
  - extent discovery endpoint
- [open_climate_service/extents/services.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/extents/services.py)
  - extent registry backed by CLIMATE_SERVICE_CONFIG
- [open_climate_service/publications/services.py](https://github.com/dhis2/open-climate-service/blob/main/open_climate_service/publications/services.py)
  - artifact publication and stable managed dataset id logic
- `extent:` block in `climate-service.yaml` (CLIMATE_SERVICE_CONFIG)
  - configured spatial extent for this Open Climate Service instance

## What Was Achieved

### 1. Public ingestion contract now uses `extent_id`

`POST /ingestions` now takes:

- `dataset_id`
- `start`
- `end`
- `extent_id`
- `overwrite`
- `prefer_zarr`
- `publish`

Raw `bbox` and `country_code` are no longer part of the public ingestion payload.

The route resolves `extent_id` inside Open Climate Service and then calls the downloader with concrete spatial inputs.

### 2. Public ingestion responses now return datasets, not artifacts

`POST /ingestions`, `GET /ingestions`, and `GET /ingestions/{ingestion_id}` now define the operational ingestion surface.

`POST /ingestions` and `GET /ingestions/{ingestion_id}` return:

- `ingestion_id`
- `status`
- `dataset`

The `dataset` field uses the public dataset summary model from `/datasets`, not the full dataset detail view with version history.

Internal artifact records still exist, but they no longer define the public response story.

`GET /ingestions` lists ingestion run records for admin and operational use. `/datasets` remains the canonical managed-data surface for consumers.

### 3. Extents are now a first-class read-only part of the native API

The branch exposes:

- `GET /extent`

Extents are configured in YAML and currently include:

- `extent_id`
- `name`
- `description`
- `bbox`

This keeps spatial configuration explicit without turning it into a runtime write API.

### 4. `/datasets` is now the native managed-data catalog

`GET /datasets` returns a public dataset catalog envelope:

- `kind`
- `items`

Each dataset item includes:

- public dataset id
- source dataset template id
- dataset metadata from the registry
- current extent
- last updated timestamp
- public links
- publication status

The public dataset response no longer exposes internal artifact ids, artifact counts, filesystem paths, or downloader implementation details.

### 5. Raw Zarr access is now canonical under `/zarr/{dataset_id}`

The raw data surface is:

- `GET /zarr/{dataset_id}`
- `GET /zarr/{dataset_id}/{relative_path}`

The public Zarr listing response now avoids leaking internal artifact ids and raw filesystem roots. It returns:

- `kind`
- `dataset_id`
- `path`
- `entries`

Entry links point back into the canonical `/zarr/{dataset_id}/...` namespace.

### 6. STAC is now the public discovery surface for published Zarr datasets

The branch exposes a dedicated STAC surface under:

- `/stac`
- `/stac/catalog.json`
- `/stac/collections/{dataset_id}`

Published Zarr-backed managed datasets appear there as one STAC Collection per dataset. The `zarr` asset points to the canonical native `/zarr/{dataset_id}` route.

`xstac` derives Datacube metadata from the opened Zarr-backed dataset, while the Open Climate Service service layer remains responsible for publication filtering, link construction, and Zarr asset metadata.

Current STAC details:

- the Zarr asset href is `/zarr/{dataset_id}` — the store root — for flat and pyramided stores alike. A pyramided root has no data variables of its own: it carries `multiscales` (with the levels under `0/`, `1/`, …) plus the time coordinate and `spatial_ref`, and its media type gains `profile=multiscales`, which is how a client knows to resolve a level rather than expecting arrays at the root. Python readers do that descent for you via `open_icechunk_dataset` / `open_zarr_dataset`, which always land on level 0
- temporal extents are normalized to RFC 3339 in both STAC and Datacube temporal extent fields
- STAC collection `license` currently defaults to `various`
- spatial `step` values are rounded for readability while preserving axis direction
- an opt-in live interoperability smoke test exists at `tests/integration/test_stac_interop.py`

### 7. STAC is the publication surface

Published datasets are exposed as STAC collections:

- `/stac/catalog.json`
- `/stac/collections/{dataset_id}`

Dataset responses on the native FastAPI side include publication state and a link to the STAC collection. (The previous OGC API surface has been removed.)

### 8. Internal artifacts still exist as a storage/provenance model

The branch still persists internal artifact records in `data/artifacts/records.json`.

Those internal records retain:

- exact request scope
- stored format
- creation time
- publication mapping
- deduplication and sync history inputs

This internal model remains necessary for provenance and sync behavior, but it is no longer a public API concept.

The current JSON-backed store is still an interim persistence layer. Record mutations now use file locking to avoid lost updates during concurrent writes, but the long-term direction should be a proper transactional store.

### 9. `/sync` is now a testable managed dataset update path

The sync API now exposes:

- `GET /sync/{dataset_id}/plan?end={period}`
- `POST /sync/{dataset_id}`

The plan endpoint returns a dry-run `SyncDetail` without downloading or writing data. The post endpoint executes the same plan through the existing artifact creation path when work is required.

Implemented sync behavior:

- temporal datasets can append missing periods
- release datasets rematerialize when a newer requested release exists
- static datasets return `not_syncable`
- plugin `periods()` clamps target end to actually available data before execution
- append execution reuses the planned source delta and writes only missing periods to the committed Icechunk store
- rematerialization builds and validates the complete contiguous union in a sibling store before publication
- artifact records keep caller request scope as provenance and cumulative realized store coverage separately
- existing stores are updated from a planned contiguous union; earlier or gapped ranges rematerialize in ascending order

## How The Current Flow Works

### Ingestion

1. client submits `dataset_id`, `start`, optional `end`, and optional `extent_id`
2. Open Climate Service resolves the dataset template from the registry
3. Open Climate Service resolves `extent_id` to a concrete bbox or other configured spatial input
4. Open Climate Service checks for an existing matching internal artifact
5. if needed, Open Climate Service downloads the source data
6. Open Climate Service prefers Zarr materialization and falls back to NetCDF when needed
7. Open Climate Service computes realized coverage metadata
8. Open Climate Service stores an internal artifact record
9. if `publish=true`, Open Climate Service publishes the dataset as a STAC collection
10. the route returns the public managed dataset view

### Dataset publication

1. publication derives a stable managed dataset id
2. STAC collection documents are derived dynamically from the published artifact state
3. the dataset becomes available immediately under `/stac/collections/{dataset_id}`

### Raw data access

1. `/datasets/{dataset_id}` exposes native metadata and version summary
2. `/stac/collections/{dataset_id}` exposes standards-friendly discovery metadata for direct Zarr-opening clients
3. `/zarr/{dataset_id}` exposes the raw Zarr store layout when the latest version is Zarr-backed

### Sync

1. `GET /sync/{dataset_id}/plan` resolves the latest local artifact and source template
2. `sync_engine.plan_sync(...)` computes the action, target, and delta range
3. plugin `periods()` is queried to clamp the target end to available data
4. `POST /sync/{dataset_id}` returns `up_to_date` or `not_syncable` without writes when applicable
5. otherwise, sync calls the existing artifact creation path
6. the new version is optionally published under the same stable managed dataset id

## Current Public Surface

### Native FastAPI

- `POST /ingestions`
- `GET /ingestions`
- `GET /ingestions/{ingestion_id}`
- `GET /extent`
- `GET /datasets`
- `GET /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/download`
- `GET /zarr/{dataset_id}`
- `GET /zarr/{dataset_id}/{relative_path}`
- `POST /sync/{dataset_id}`
- `GET /sync/{dataset_id}/plan`

### Standards-facing

- `GET /stac`
- `GET /stac/catalog.json`
- `GET /stac/collections/{dataset_id}`

## What Is Still Deferred

1. a final decision on how much version history to expose publicly
2. richer extent configuration shapes beyond `id + bbox + optional metadata`
3. any runtime write API for extents
4. multi-version publication resolution behind one dataset id
5. true in-place Zarr append, if storage semantics require it later

## Short Summary

The branch now presents a much cleaner product story:

1. run ingestions through `/ingestions` as an execution and admin surface
2. return datasets, not artifacts
3. discover managed data under `/datasets`
4. discover published Zarr-backed datasets under `/stac/catalog.json`
5. access raw Zarr under `/zarr/{dataset_id}`
6. sync managed datasets through `/sync/{dataset_id}`

Internal artifacts still exist, but only as a storage and provenance model.
