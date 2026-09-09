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
        geometries:
          type: FeatureCollection
          features: []
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

## Current boundary

This mechanism dispatches workflows owned by the same OCS instance. It does not provide workflow
dependency graphs, cross-service retries, webhooks, or distributed event consumption. Exactly
one writable OCS process should perform automation until the stores and leadership model become
shared and transactional.
