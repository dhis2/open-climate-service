# Dataset-update workflow automation

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

## Current boundary

This mechanism dispatches workflows owned by the same OCS instance. It does not provide workflow
dependency graphs, cross-service retries, webhooks, or distributed event consumption. Exactly
one writable OCS process should perform automation until the stores and leadership model become
shared and transactional.
