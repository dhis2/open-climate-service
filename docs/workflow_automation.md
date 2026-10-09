# Dataset-update workflow automation

> In the [tasks](tasks.md) reference implementation (CLIM-1378), workflow triggers and their deliveries are workflow and deliver tasks managed through `/tasks`, and a workflow that publishes a dataset emits `dataset.updated`, so the two-trigger chain described below runs. The `automation` block is no longer read.

An OCS instance can run an existing openEO workflow when a successful ingestion or sync operation
changes stored data. This is event-driven: it does not guess that an operation has finished by
scheduling a second cron expression.

## Configure a trigger

Triggers are instance-owned bindings in `climate-service.yaml`:

```yaml
automation:
  workflow_triggers:
    - id: chirps-to-chap
      on_update_of: chirps3_precipitation_daily
      workflow_id: aggregate_to_chap_csv
      arguments:
        dataset_id: $event.dataset_id
        temporal_extent: [$event.previous_end, $event.current_end]
        geometries: { from_features: districts }
        method: mean
        period_type: day
```

`on_update_of` names the managed source dataset and `workflow_id` names a workflow already
available through `GET /process_graphs`. Trigger IDs must be unique within an instance.

Arguments are the parameters passed to the workflow. Literal YAML values are preserved. These
exact event references can be used at any nesting level:

- `$event.dataset_id`
- `$event.artifact_id`
- `$event.action`: `ingest` for an ingestion, or the sync planner's `append` or `rematerialize`
- `$event.previous_end`: the coverage end before the update, or `null` when the whole store is new
  or rewritten
- `$event.current_start`: the coverage start after the update
- `$event.current_end`: the coverage end after the update

After an initial ingestion, `previous_end` is JSON/YAML `null`, because no periods were stored
before. A temporal interval of `[$event.previous_end, $event.current_end]` is therefore open at
the start and reads through `current_end` from the dataset's earliest available period. This
means the first triggered workflow run processes the complete ingested history. An
ingestion that rewrites the whole store, including `overwrite`, also uses `null` because every
period was rewritten. After an append, `previous_end` is the old coverage boundary, so that
interval includes the boundary and the newly appended periods. A workflow that always needs an
explicit, non-null start can use `$event.current_start`, the start of the complete coverage after
the update.

The workflow definition remains reusable and deployment-independent. Operational bindings such
as output dataset IDs, geometries, and DHIS2 identifiers remain in instance configuration.

### Referencing a feature collection

Within an automation trigger, `geometries: { from_features: districts }` names a feature
collection declared under `plugins/vectors/` (see
[Installable plugins](installable_plugins.md#package-layout)) instead of embedding a
`FeatureCollection` inline. This shorthand is resolved by the automation service and is not a
general openEO process-graph argument. Direct process graphs should call `load_features`
explicitly. An inline `FeatureCollection` remains valid in either context.

The reference is never resolved into geometry at submission. Instead OCS rewrites it into a
sibling node in the process graph that the workflow calls `load_features` on:

```json
{
  "features_districts": {
    "process_id": "load_features",
    "arguments": { "id": "districts", "version": "2026-09-01T06:00:00+00:00" }
  },
  "workflow": {
    "process_id": "aggregate_to_chap_csv",
    "arguments": { "geometries": { "from_node": "features_districts" } },
    "result": true
  }
}
```

This keeps a country's whole org unit hierarchy out of the persisted job record. At submission,
OCS resolves the current feature record and stores only the collection id, its refresh timestamp,
and a `load_features` call. The timestamp pins execution to that record: if the collection is
refreshed while the job is queued, the job refuses the stale binding instead of silently running
against different geometry. The job description carries the same version (for example
`against features districts@2026-09-01T06:00:00+00:00`), so it accurately explains why one run
covered 47 districts and the next covered 48.

A malformed reference or an id that no template declares fails at startup, like an unknown
`workflow_id`. A declared collection must also be registered before an event can submit a job.

## Delivery behavior

A workflow is considered only after a successful ingestion or sync has persisted a
`dataset.updated` event. Failed, cancelled and no-op runs do not trigger workflows, and neither
does re-ingesting a dataset that is already current. Ingestion and sync, whether queued,
scheduled, run inline, or started from the admin pages, all use the same path. Work run inline
is recorded as a completed job under `/ingestions/jobs`, because that record is what makes
its event durable after the operation returns. Unlike queued ingestion, inline execution has
no pre-write job checkpoint: if the process stops after committing data but before recording
the completed job, that update event cannot be recovered automatically.

An ingestion job notes in its checkpoint that it is about to change stored data, before
fetching anything. If OCS stops after the data is committed but before the job completes,
the recovered job finds the data current and still emits the event it owed, exactly once.

Each event and trigger pair produces a deterministic openEO job ID. OCS replays persisted events
at startup, but an already created, queued, running, or completed job is not duplicated. If OCS
stopped after creating a job but before queueing it, startup queues that existing job.

A trigger is activated the first time OCS starts with it configured. By default
(`replay_existing: false`), startup replay ignores events that predate the trigger's activation, so
adding a new trigger does not backfill the workflow over every historical update. Set
`replay_existing: true` to opt into replaying all persisted events for that trigger. Newly
persisted events are always dispatched immediately, regardless of this flag.

Workflow execution and status remain owned by the openEO batch-job service and are visible under
`GET /jobs`. Workflows may also be submitted directly without a preceding dataset sync.

Triggered jobs share the existing openEO job pool with manually submitted jobs — there is no
separate automation queue or concurrency limit. A fan-out of several triggers therefore runs
independently in that pool. Bounding total workflow concurrency is a general resource-policy
concern deferred to CLIM-845; until then the per-store lock remains the write-safety boundary for
workflows that publish managed datasets.

## Retries and restarts

A triggered workflow job runs up to `max_attempts` times, the first attempt included. The
default is 3 and the bound is 10:

```yaml
automation:
  workflow_triggers:
    - id: chirps-to-dhis2
      on_update_of: chirps3_precipitation_daily
      workflow_id: aggregate_to_dhis2_json
      max_attempts: 3
      arguments: { ... }
```

Every attempt runs under the same deterministic job ID, so a retry is still the one job for
that event and trigger, and replaying the event does not create another.

- **A failure while the workflow runs is retried after a backoff** of 1, 2, then 4 minutes.
  This covers an unreachable source, a timeout, a server error, a store another writer is
  busy with, and also errors such as a truncated remote response, which look like invalid
  input but may not repeat. While it waits the job is `queued`, holding no worker, and no
  process runs it before its backoff has passed. An invalid argument discovered only while
  the graph runs may therefore use the full attempt budget; only graph validation and known
  save-result configuration errors can be classified as permanent before execution.
- **A permanent error is not retried**, because it fails the same way on every attempt: an
  invalid process graph, a request a process refuses (an unknown collection, for example),
  or invalid configuration found while saving the result, such as an unknown export or a
  units mismatch.
- **A restart during an attempt** requeues the job if it has attempts left. The interruption
  counts as an attempt, so a job that keeps crashing the server still stops.
- **A job still backing off at shutdown** waits out the rest of its backoff after the next
  start.
- **Cancelling during a backoff** takes effect at once, and the attempt history records it.
  Re-running a cancelled job with `POST /jobs/{job_id}/results` runs it; the earlier
  cancellation does not carry over.

When the attempts run out, the job stays `error`. Its error names the attempt, for example
`OSError: connection reset (attempt 3 of 3)`, and `GET /jobs/{job_id}/results` answers with it.
The job's `logs` field lists every attempt with its time and outcome.

Only an attempt that finishes can deliver, so a trigger with `deliver` imports the result of
the successful attempt, once. Re-running a failed job with `POST /jobs/{job_id}/results`
starts a fresh attempt budget.

Jobs submitted directly, not by a trigger, run once as before. A job still running in another
OCS process, for example one that has not finished shutting down, is not marked failed at
startup: it is left to that process, and taken over if the process exits without finishing it.

## Deliver the result to DHIS2

A trigger can deliver its job's result once the job finishes. The workflow must save through a
[named export](export_plugins.md), and `deliver.export` names that same export:

```yaml
automation:
  workflow_triggers:
    - id: chirps-monthly-to-districts
      on_update_of: chirps3_precipitation_monthly
      workflow_id: aggregate_to_dhis2_json
      arguments:
        dataset_id: $event.dataset_id
        temporal_extent: [$event.previous_end, $event.current_end]
        geometries: { from_features: districts }
        export: chirps-monthly-districts
      deliver:
        export: chirps-monthly-districts
        dry_run: true   # the default; set false to import into DHIS2
```

When a job created by the trigger finishes, OCS submits a delivery exactly as
[`POST /exports/{export_id}`](export_plugins.md#deliver-a-saved-export) does. Verification,
chunking, checkpoints, the per-export lock, and the import report all come from that delivery
job. The source job lists it under `usage.deliveries`. A job that fails or is canceled
delivers nothing.

`dry_run` defaults to `true`, so writing to DHIS2 is an explicit opt-in. Inspect the dry-run
reports first, then set `dry_run: false`. Only jobs that finish after the change are imported.
Jobs that finished earlier are never imported live, whether or not their dry run was delivered.

Each finished job is delivered at most once per export and mode. The idempotency key is
`auto:{job_id}:{export}:{dry-run|live}`, so replaying an event or restarting OCS creates no
second delivery. A job records the delivery it owes in the same write that marks it finished.
At startup OCS submits any recorded delivery missed because the process stopped after a job
finished. If submission fails, the source job exposes the latest failure under
`usage.delivery_error`; OCS retries it during the next startup reconciliation, not on an
in-process timer. A workflow finishing during shutdown may likewise defer submission until
the next startup reconciliation. A job keeps its delivery links when re-run by hand, so a
delivered job is not delivered again, even after switching to live; submit the new result through
`POST /exports/{export_id}` instead.

Adding `deliver` to an existing trigger does not deliver its history. The delivery step has
its own activation time, and only jobs that finish after it are delivered. Changing
`deliver.export` or `deliver.dry_run`, or removing `deliver` and adding it back, starts a new
activation. A job that finishes while its trigger has no active delivery step, for example
while `deliver` is absent or the instance is read-only, owes nothing and is never delivered
later. `replay_existing` applies to workflow submission only, never to delivery.

Startup fails, naming the trigger, when `deliver.export` is not a configured export, is
configured more than once, its plugin is not `dhis2`, it has no `connection`, that connection
is not configured, its mapping is invalid, or a literal `arguments.export` differs from
`deliver.export`. A read-only instance validates the same configuration but leaves delivery
inactive, allowing writable and serving instances to share one tracked configuration.

The export delivers what exists at its period. To deliver a monthly export from a daily
dataset, let one trigger derive and publish the monthly dataset, and a second trigger on that
dataset's update run the export. See
[the export's period must be reachable](export_plugins.md#the-exports-period-must-be-reachable-from-the-datasets-cadence).

Delivery needs the `dhis2` extra (`open-climate-service[dhis2]`) and the connection's token in
the server environment. See
[named connections](importing_to_dhis2.md#named-connections-for-server-side-plugins).

## Current boundary

This mechanism dispatches workflows owned by the same OCS instance. It does not provide workflow
dependency graphs, cross-service retries, webhooks, or distributed event consumption. A delivery
that ends partial, rejected, or unknown is reported on its delivery job but does not yet notify
anyone (CLIM-919), and a workflow job that exhausts its attempts is not reported beyond its own
status. Exactly one writable OCS process should perform automation until the stores and
leadership model become shared and transactional.
