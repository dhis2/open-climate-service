# Dataset-update workflow automation

An OCS instance can run an existing openEO workflow after a successful dataset sync changes
stored data. This is event-driven: it does not guess that a sync has finished by scheduling a
second cron expression.

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
- `$event.action`
- `$event.previous_end`
- `$event.current_end`

The workflow definition remains reusable and deployment-independent. Operational bindings such
as output dataset IDs, geometries, and DHIS2 identifiers remain in instance configuration.

### Referencing a feature collection

Within an automation trigger, `geometries: { from_features: districts }` names a feature
collection declared under `plugins/features/` (see
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

A workflow is considered only after the native sync job has successfully persisted a
`dataset.updated` event. Failed and no-op syncs do not trigger workflows. Manual and scheduled
syncs use the same path.

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
second delivery. At startup OCS submits any delivery missed because the process stopped
after a job finished. Re-running a delivered job by hand does not deliver it again; submit
the new result through `POST /exports/{export_id}` instead.

Adding `deliver` to an existing trigger does not deliver its history. The delivery step has
its own activation time, and only jobs that finish after it are delivered. Changing
`deliver.export` or `deliver.dry_run`, or removing `deliver` and adding it back, starts a new
activation.
`replay_existing` applies to workflow submission only, never to delivery.

Startup fails, naming the trigger, when `deliver.export` is not a configured export, its
plugin is not `dhis2`, it has no `connection`, or that connection is not configured. A
read-only instance refuses triggers with `deliver`, because delivery writes to DHIS2.

Delivery needs the optional `dhis2-client` package and the connection's token in the
server environment. See
[named connections](importing_to_dhis2.md#named-connections-for-server-side-plugins).

## Current boundary

This mechanism dispatches workflows owned by the same OCS instance. It does not provide workflow
dependency graphs, cross-service retries, webhooks, or distributed event consumption. A delivery
that ends partial, rejected, or unknown is reported on its delivery job but does not yet notify
anyone, and failed workflow jobs are not retried (CLIM-919). Exactly
one writable OCS process should perform automation until the stores and leadership model become
shared and transactional.
