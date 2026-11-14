# Dataset-update automation

When a successful ingestion or sync changes stored data, OCS records a durable `dataset.updated`
event. Two things act on it, each with its own page:

- **Keep a derived dataset current.** An anomaly, a normal or another dataset a workflow
  produces is computed again when the dataset it derives from updates. Give its dataset
  template a recipe; see
  [Keep a derived dataset current with a recipe](adding_custom_datasets.md#keep-a-derived-dataset-current-with-a-recipe).
- **Deliver with a pipeline.** A [pipeline](web_interface.md#the-pipeline-pages-pipelines) runs
  one workflow on its input dataset's updates, or by hand, and delivers the result to a
  destination such as DHIS2, with validation, a dry-run gate and run history; see
  [Importing to DHIS2](importing_to_dhis2.md).

A refreshed derived dataset announces its own update, so a pipeline on it runs next: a sync of
ERA5-Land monthly precipitation refreshes the anomaly, and the anomaly's update runs the
pipeline that delivers it per district.

Workflows can still be run directly, without either: submit a process graph to `POST /jobs`,
and save through a pipeline's named export with `save_result(..., options={"export":
"<pipeline id>"})` to deliver the result with `POST /exports/{pipeline id}`.

## When an update is announced

An update is recorded only after a successful ingestion, sync or openEO publication has
persisted it. Failed, cancelled and no-op runs announce nothing, and neither does re-ingesting
a dataset that is already current. Ingestion and sync, whether queued, scheduled, run inline,
or started from the admin pages, all use the same path. Work run inline is recorded as a
completed job under `/ingestions/jobs`, because that record is what makes its event durable
after the operation returns. Unlike queued ingestion, inline execution has no pre-write job
checkpoint: if the process stops after committing data but before recording the completed job,
that update event cannot be recovered automatically.

An ingestion job notes in its checkpoint that it is about to change stored data, before
fetching anything. If OCS stops after the data is committed but before the job completes,
the recovered job finds the data current and still emits the event it owed, exactly once.

The event says what changed: `dataset_id`, `artifact_id`, `action` (`ingest`, the sync
planner's `append` or `rematerialize`, or `full_rewrite` for a dataset an openEO job
published), `previous_end` (the coverage end before the update, or `null` when the whole store
is new or rewritten), and `current_start` and `current_end` (the coverage after it). A pipeline
runs the interval the update affected; a recipe always recomputes the whole coverage.

## Updates from derived datasets

An openEO job whose graph ends in `save_result` as Zarr with a `dataset_id` publishes a managed
dataset, and owns that publication and its `dataset.updated` event. Startup replay and the live
consumers read events from the native and the openEO job stores alike, so a pipeline on a
derived dataset needs nothing special.

- **Only a published output is announced.** The event is recorded once the Icechunk commit has
  landed, the artifact is registered and its publication is settled, and it follows the stored
  record: an artifact that was already published stays published, and is announced, even when
  the job did not ask to publish. An unpublished output is stored and registered without one.
- **It describes a full rewrite.** The managed writer replaces the whole store, so the event's
  `action` is `full_rewrite`, `previous_end` is empty and `current_start`/`current_end` are the
  committed dataset's coverage. It does not claim which periods changed, so a pipeline on the
  dataset re-runs its whole coverage on every publication.
- **A rewrite never shortens a published dataset.** One that would start later or end earlier
  than the published coverage is refused before anything is committed. It may cover more.
- **The producing job owns it, one event per commit.** The event carries `producer_type: openeo`,
  `producer_job_id` and `output_snapshot_id`, and its id is derived from the job, the artifact
  and the committed snapshot. Completing the same commit again, in a retry or at a restart, is
  the same logical update; a deliberate re-run of the job commits a new snapshot and is a new
  update, which recipes and pipelines act on again.
- **The snapshot is provenance, not a version.** `output_snapshot_id` is the physical Icechunk
  snapshot the job committed. The artifact records it under `provenance`, with the producing
  job and the execution evidence (process digest and input artifacts, source datasets and input
  snapshots); its logical `version` stays empty unless OCS releases the dataset.
- **Provenance is shown, and so is an input that has moved on.** `GET /datasets/{id}` returns
  it under `provenance`: the producing job, the process digest, the OCS version that ran it and
  each input with the snapshot the job read. An input is `changed` when what `load_collection`
  would read now is another artifact or another snapshot, so a dataset computed before its
  input's last sync says so; the dataset page shows the same. The STAC collection links each
  input as `derived_from` and names this instance as its `processor` provider, which carries the
  processing time and OCS version.

The job records its intent before writing, marks the publication `committing` at the
cancellation gate just before the commit, and writes the outcome and event together after
registration. A restart that finds it `committing` searches the store history above the
attempt's starting snapshot for the commit message naming the job and attempt: found, it
completes the publication with that snapshot; not found, nothing was published and the job runs
again, whoever submitted it. A job that recorded its event but stopped before finishing is
finished at restart and its event dispatched; one that finished but stopped before dispatching
is found by startup replay. Cancellation before the gate publishes nothing; after it, and while
a commit is unresolved, cancellation is refused.

An ordinary error after the gate (registering the artifact, publishing it, recording the
outcome) never loses a commit that landed. The failing attempt completes the publication if it
can, and the job finishes. If it cannot, the publication stays `committing`: a retry completes
it from the existing commit rather than computing and committing again. Startup recovery keeps
watching a pending final attempt while its dataset lock or another transient dependency is
unavailable; once completion succeeds, the job is marked finished and its one event is emitted.

## Feature collections in workflow arguments

A pipeline binds a workflow's geometry parameter as `{ from_features: districts }`, naming a
feature collection declared under `plugins/vectors/` (see
[Installable plugins](installable_plugins.md#package-layout)) instead of embedding a
`FeatureCollection`. This shorthand is resolved when the pipeline submits its job and is not a
general openEO process-graph argument: direct process graphs call `load_features` explicitly.

The reference is never resolved into geometry at submission. Instead OCS rewrites it into a
sibling node in the process graph that the workflow calls `load_features` on:

```json
{
  "features_districts": {
    "process_id": "load_features",
    "arguments": { "id": "districts", "version": "2026-09-01T06:00:00+00:00" }
  },
  "workflow": {
    "process_id": "aggregate_by_features",
    "arguments": { "geometries": { "from_node": "features_districts" } },
    "result": true
  }
}
```

This keeps a country's whole org unit hierarchy out of the persisted job record. At submission,
OCS resolves the current feature record and stores only the collection id, its refresh timestamp,
and a `load_features` call. The timestamp pins execution to that record: if the collection is
refreshed while the job is queued, the job refuses the stale binding instead of silently running
against different geometry.

## Jobs, retries and restarts

A pipeline run and a derived dataset refresh each run as an openEO batch job, visible under
`GET /jobs` and sharing the job pool with jobs submitted directly; there is no separate
automation queue or concurrency limit. Each job's id is derived from the event it answers and
its owner, so replaying an event at startup finds the existing job instead of creating another.
If OCS stopped after creating a job but before queueing it, startup queues that existing job.

Such a job runs up to 3 times, the first attempt included, under the same job id.

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
The job's `logs` field lists every attempt with its time and outcome. Re-running a failed job
with `POST /jobs/{job_id}/results` starts a fresh attempt budget.

Jobs submitted directly run once. A job still running in another OCS process, for example one
that has not finished shutting down, is not marked failed at startup: it is left to that
process, and taken over if the process exits without finishing it.

## Current boundary

Automation dispatches workflows owned by the same OCS instance. It does not provide
cross-service retries or distributed event consumption. A failing scheduled sync or pipeline
can post to a webhook (see [Scheduled sync](scheduled_sync.md)). Exactly one writable OCS
process should perform automation until the stores and leadership model become shared and
transactional.
