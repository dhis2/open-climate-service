# Steps: automating a flow from source to DHIS2

> Reference implementation of the model proposed in
> [CLIM-1378](https://dhis2.atlassian.net/browse/CLIM-1378), for testing the approach. Not a
> settled design.

A **step** is one unit of work the instance does on its own. There are four kinds, and every kind
says *when* it runs in the same ways: on a **cron**, **after** something it depends on changed, or
**by hand** (`POST /steps/{id}/run`).

| Kind       | Target               | On a cron | After                       |
| ---------- | -------------------- | --------- | --------------------------- |
| `sync`     | a managed dataset    | yes       | –                           |
| `refresh`  | a feature collection | yes       | –                           |
| `workflow` | an openEO workflow   | yes       | a dataset or a collection   |
| `deliver`  | a named export       | –         | a workflow step             |

Every step that changes something says so: a sync or a workflow that publishes a dataset emits
`dataset.updated`, and a feature refresh emits `collection.updated`. A step can wait for either, so
steps chain: sync daily data, derive a monthly dataset, then aggregate and deliver that. A chain
stops after 5 workflow runs in a row, and a workflow step that would write the dataset it waits for
is refused.

All steps live in one store, `<data_dir>/steps.json`, and are managed through `/steps`. A change
applies at once, with no restart. `climate-service.yaml` keeps only the infrastructure: the clock's
`scheduler.enabled` and `timezone`, `dhis2_connections`, and the named `exports`.

## Example: daily rainfall to DHIS2

```bash
# 1. Keep the dataset current: sync it every morning.
curl -X POST $OCS/steps -H 'Content-Type: application/json' -d '{
  "id": "sync-chirps3_precipitation_daily", "kind": "sync",
  "target": "chirps3_precipitation_daily", "cron": "0 6 * * *"}'

# 2. Keep the org units current: refresh them every Monday.
curl -X POST $OCS/steps -H 'Content-Type: application/json' -d '{
  "id": "districts-weekly", "kind": "refresh", "target": "districts", "cron": "0 5 * * 1"}'

# 3. Aggregate to the districts after each update.
curl -X POST $OCS/steps -H 'Content-Type: application/json' -d '{
  "id": "chirps-to-districts", "kind": "workflow", "target": "aggregate_to_dhis2_json",
  "after": {"dataset": "chirps3_precipitation_daily"},
  "arguments": {
    "dataset_id": "$event.dataset_id",
    "temporal_extent": ["$event.previous_end", "$event.current_end"],
    "geometries": {"from_features": "districts"},
    "export": "chirps-daily", "method": "mean"}}'

# 4. Deliver each result through the named export, as a dry run first.
curl -X POST $OCS/steps -H 'Content-Type: application/json' -d '{
  "id": "chirps-to-dhis2", "kind": "deliver", "target": "chirps-daily",
  "after": {"step": "chirps-to-districts"}, "dry_run": true}'

# Check the reports, then go live.
curl -X PUT $OCS/steps/chirps-to-dhis2 -H 'Content-Type: application/json' -d '{"dry_run": false}'
```

`GET /steps` lists every step with how it starts, its next run on the clock and its latest result.
`POST /steps/{id}/pause` and `/resume` stop and restart one without removing it.

## How it maps onto what existed

* A `sync` step is a sync schedule: `/schedules/sync/{dataset_id}` and the dataset page's Schedule
  tab still work, and read and write the same store.
* A `workflow` step after a dataset is what `automation.workflow_triggers` was, and a `deliver` step
  after it is that trigger's `deliver` block. Events, replay after a restart, retries, the export
  gate and delivery are unchanged; only where the configuration comes from moved.
* An `automation` block in `climate-service.yaml` is refused at startup.

## Not in this reference implementation

* A web page for steps. The Schedules page shows the sync steps; `/steps` is API only.
* One run record linking a step's run to the jobs and deliveries it caused
  ([CLIM-1222](https://dhis2.atlassian.net/browse/CLIM-1222)).
* Exports in the store ([CLIM-1089](https://dhis2.atlassian.net/browse/CLIM-1089)) and SQLite as
  the store ([CLIM-927](https://dhis2.atlassian.net/browse/CLIM-927)).
* A deliver step on its own cron, and `after` a delivery.
