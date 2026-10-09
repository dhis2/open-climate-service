# Automating a flow from source to DHIS2

> Reference implementation of the model proposed in
> [CLIM-1378](https://dhis2.atlassian.net/browse/CLIM-1378), for testing the approach. Not a
> settled design.

An OCS instance keeps data current and sends it on by itself through **tasks**. A task is one piece
of work and the rule for when it runs:

| Kind       | Does                                  | On a schedule | After a change              |
| ---------- | ------------------------------------- | ------------- | --------------------------- |
| `sync`     | fetches new periods of a dataset      | yes           | –                           |
| `refresh`  | re-fetches a feature collection       | yes           | –                           |
| `workflow` | runs an openEO workflow               | yes           | a dataset or a collection   |
| `deliver`  | sends a workflow's result on          | –             | a workflow task             |

Every task can also be run by hand. Every task says what it did: a sync, or a workflow that
publishes a dataset, emits `dataset.updated`; a refresh emits `collection.updated`; a delivery emits
`export.delivered`. A task that waits for a change runs when that event arrives, so tasks chain:
sync daily data, derive a monthly dataset from it, aggregate that to districts and send it to DHIS2.
A chain stops after 5 workflow runs in a row, and a workflow task that would write the dataset it
waits for is refused.

## The quick way: from a dataset's page

On a dataset's page:

1. **Sync** schedules the dataset to stay current.
2. **Send to DHIS2** asks for a connection, the org units, a data element and a statistic, and
   creates the three things a delivery needs: a named export (the mapping), a workflow task that
   aggregates the dataset to the org units after each update, and a deliver task, a dry run.
3. **Flow** draws where the dataset comes from and what it feeds, with each step's latest run.

When the dry-run reports look right, open the deliver task's page and press **Go live**.

## The Tasks pages

* **Tasks** lists every task as a tile: what it does, how it starts and its last run, with a
  search and filters by kind and status. **Flow** switches the list to a diagram of how the tasks
  connect: from each dataset, through the workflows that read it, to where the result goes, each
  box with its latest run, so a failed box is where a broken flow starts. **New task** adds any
  kind of task.
* **A task's page** shows what it does, when it runs next, the part of the flow it is in and its
  recent runs. Run it, pause it, delete it, or switch a delivery between dry run and live there.
* A **dataset's page** draws the part of the flow through that dataset, and a **workflow's page**
  lists the tasks that run it.

## The API

| Route | What |
| --- | --- |
| `GET`, `POST /tasks`; `GET`, `PUT`, `DELETE /tasks/{id}` | tasks |
| `POST /tasks/{id}/run`, `/pause`, `/resume` | run by hand, pause, resume |
| `GET /exports`; `GET`, `PUT`, `DELETE /exports/{id}` | named exports, validated by their plugin |
| `GET /runs`, `GET /runs/{id}` | what each task did; a run with the runs it set off |
| `GET /flows` | the flow graph as nodes and edges |
| `GET`, `PUT /configuration` | every task and export as one document |
| `GET /ingestions/jobs` | every native job |

```bash
# Keep the dataset current, and the org units.
curl -X POST $OCS/tasks -H 'Content-Type: application/json' -d '{
  "id": "sync-chirps", "kind": "sync", "target": "chirps3_precipitation_daily", "cron": "0 6 * * *"}'
curl -X POST $OCS/tasks -H 'Content-Type: application/json' -d '{
  "id": "districts-weekly", "kind": "refresh", "target": "districts", "cron": "0 5 * * 1"}'

# The mapping to DHIS2.
curl -X PUT $OCS/exports/chirps-daily -H 'Content-Type: application/json' -d '{
  "plugin": "dhis2", "dataset": "chirps3_precipitation_daily", "connection": "national-hmis",
  "aggregation": "mean", "period_type": "daily",
  "series": [{"select": {}, "data_element": "BXgDHhPdFVU"}]}'

# Aggregate after each update, and deliver the result, as a dry run first.
curl -X POST $OCS/tasks -H 'Content-Type: application/json' -d '{
  "id": "chirps-to-districts", "kind": "workflow", "target": "aggregate_to_dhis2_json",
  "after": {"dataset": "chirps3_precipitation_daily"},
  "arguments": {
    "dataset_id": "$event.dataset_id",
    "temporal_extent": ["$event.previous_end", "$event.current_end"],
    "geometries": {"from_features": "districts"},
    "export": "chirps-daily", "method": "mean"}}'
curl -X POST $OCS/tasks -H 'Content-Type: application/json' -d '{
  "id": "chirps-to-dhis2", "kind": "deliver", "target": "chirps-daily",
  "after": {"task": "chirps-to-districts"}, "dry_run": true}'

# Go live.
curl -X PUT $OCS/tasks/chirps-to-dhis2 -H 'Content-Type: application/json' -d '{"dry_run": false}'
```

## Configuration: one source, and a document to carry it

`climate-service.yaml` holds the infrastructure only: the clock (`scheduler.enabled`, `timezone`),
`dhis2_connections` with their tokens in the environment, plugin folders, read-only. Tasks and
exports live in the operational database and change at once, without a restart. An `automation` or
`exports` block in the file is refused.

To keep the configuration in git, or to copy it to another instance, take it out as one document
and put it back:

```bash
curl "$OCS/configuration?format=yaml" > automation.yaml
curl -X PUT $OCS/configuration -H 'Content-Type: application/yaml' --data-binary @automation.yaml
```

An import is validated as a whole before anything changes.

## State

Tasks, exports, run records, the automation's activation boundaries and the clock's lease are in
`<data_dir>/ocs.db`, SQLite in WAL mode: one writer and any number of readers across processes, and
no whole-file rewrite that a full disk can truncate. All access goes through
`open_climate_service/state/db.py`, in SQL that Postgres also runs, so a networked database can
replace it for a deployment of several machines. Job and artifact records are still JSON files;
moving them is the rest of [CLIM-927](https://dhis2.atlassian.net/browse/CLIM-927).

## Capacity

Heavy work is bounded however many tasks fire at once. Ingestion, sync, refresh and workflow jobs
share the compute slots, `CLIMATE_SERVICE_MAX_CONCURRENT_JOBS` (default 2), and the rest queue.
Deliveries only send a payload, so they have slots of their own,
`CLIMATE_SERVICE_MAX_CONCURRENT_DELIVERIES` (default 2): a delivery never waits behind an hour of
aggregation, nor takes the slot that aggregation needs. All dask computation shares one thread pool
(half the cores by default).

## Several processes, or several machines

Every process with the scheduler enabled starts a clock, but only the one holding the clock's lease
runs schedules; the others stand by and bid for it every 30 seconds. If the holder stops, one of
them takes over within 90 seconds. So several workers on one machine fire each schedule once.

For several machines, the operational database must be one they all reach (Postgres, rather than
SQLite on a shared disk, whose locks are unreliable over NFS), and jobs still run in the process
that started them: a shared job queue across machines is
[CLIM-998](https://dhis2.atlassian.net/browse/CLIM-998).

## Not in this reference implementation

* Job and artifact records in the database ([CLIM-927](https://dhis2.atlassian.net/browse/CLIM-927)).
* A Postgres backend for the operational database, and a job queue shared across machines.
* Notifications: a webhook on a run's outcome ([CLIM-919](https://dhis2.atlassian.net/browse/CLIM-919)).
  The `export.delivered` event and the failure count are what it would send.
* A deliver task on its own schedule, and typed connections for plugins other than DHIS2
  ([CLIM-1289](https://dhis2.atlassian.net/browse/CLIM-1289)).
