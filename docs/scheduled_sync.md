# Scheduled dataset synchronization

Open Climate Service can periodically check whether an existing managed dataset has new
source periods and submit an asynchronous sync job when work is required. The scheduler is
a clock only: planning, retries, execution, progress, and restart recovery remain owned by
the native OCS job service.

## Configure schedules

Schedules are an instance-level operational choice in `climate-service.yaml`:

```yaml
scheduler:
  enabled: true
  timezone: UTC
  dataset_sync:
    - dataset_id: chirps3_precipitation_daily
      cron: "0 6 * * *"
      publish: true
      max_attempts: 3
```

`cron` is a standard five-field expression interpreted in the configured IANA timezone.
UTC is the default and is recommended for checks that follow upstream publication times.
Only one entry is allowed per dataset.

The target dataset must already have been ingested. When a schedule becomes due, APScheduler
queues work through the same native job path as an asynchronous `POST /sync/{dataset_id}` request
and returns immediately. Planning and synchronization happen in the native queue. An up-to-date
check therefore completes as a normal no-op job record instead of doing upstream work in the
clock callback.

Scheduled work uses the same native job pool as manual ingestion and sync requests. Scheduling
does not change that pool's concurrency. An active manual ingestion or sync for the same dataset
suppresses duplicate scheduled submission; the per-store lock remains the final write-safety
boundary.

## Manage schedules without a restart

Schedules can also be added, changed, paused and removed from the **Sync schedules** page and
the `/schedules` API. These schedules are stored beside the configuration file, under
`<data_dir>/schedules.json`, and the clock runs the two sources merged:

- File entries are listed and visible but read-only in the page and the API.
- The file wins. A stored entry for a dataset the file also configures is kept and listed as
  shadowed, and does not run. It stays editable and pausable, so it is ready for the day the
  file entry goes. Creating a stored entry for a file-configured dataset is refused.
- The configuration file is read once, at startup. A reload re-reads the store only, so a
  change to `scheduler.dataset_sync` in the file, including removing an entry so a shadowed
  stored one becomes effective, takes effect after a restart.
- A stored entry has a pause switch, `enabled`. Pausing keeps the entry and stops it firing.
- Every change reloads the clock. The reload parses the store, then resolves every effective
  entry's dataset and builds its trigger, before any job is touched. When the store cannot be
  read, the previous schedules stay in force and `GET /schedules` reports `reload_error` until
  the cause is fixed. An entry whose dataset no longer resolves is taken off the clock and
  listed with the reason, so no job keeps firing with settings the status no longer shows. If
  the clock itself refuses a sound plan, the previous jobs are put back and the error says so;
  should that fail too, the error says the clock may run settings the list does not show.
- A change made through another process reaches the clock. The process that owns the clock
  watches the store file and reloads within 30 seconds of a write from anywhere on the shared
  data directory, so an API-only replica can pause or edit a schedule without knowing which
  replica runs the clock. The request's own process reloads at once.
- `effective` on a listed entry means enabled and not shadowed: the entry the clock would run.
  `registered` means a clock job exists for it in this process now, which also needs the
  scheduler enabled and the dataset to resolve.
- APScheduler job ids are stable, `dataset-sync:<dataset_id>`, so a reload adds, replaces and
  removes jobs rather than rebuilding the clock.
- The timezone stays the instance's `scheduler.timezone`; a stored entry cannot set its own.
- Missed fires are not replayed: the job store is in memory, so a fire that falls into a
  restart is lost, and the next fire catches up because the sync plans from the store.

The API:

| Method and path | Effect |
| --- | --- |
| `GET /schedules` | The merged list with runtime status; the page for a browser |
| `GET /schedules/{dataset_id}` | One dataset's effective schedule and status |
| `POST /schedules` | Add a stored schedule: `dataset_id`, `cron`, `publish`, `max_attempts`, `enabled` |
| `PUT /schedules/{dataset_id}` | Replace a stored schedule's settings |
| `POST /schedules/{dataset_id}/pause`, `/resume` | Flip the pause switch |
| `DELETE /schedules/{dataset_id}` | Remove a stored schedule |

A dataset that is static, forecast-facing or has no registered data source is refused with the
reason. The target must have been ingested for a check to submit a sync; a stored schedule for
a dataset not yet ingested is saved and its checks finish as `not_materialized` until it is.

Changing schedules changes when data moves into the instance, and a sync can trigger
workflows and deliveries configured under `automation`. These routes are closed on a
read-only instance, but read-only mode is not authentication: an instance that exposes the
page must be deployed privately or behind reverse-proxy authentication until OCS has its
own write authentication.

## Inspect status

`GET /schedules` reports whether the process-local clock is running and, for each schedule,
its next check, latest enqueue outcome, message, and submitted native job ID. This status does
not mirror the terminal job state; follow `last_job_id` through the native jobs API. Check state
is volatile and resets when the process restarts; submitted jobs remain durable.

Every due check creates a native job, including checks that finish without finding new periods.
Native job history is currently retained in `jobs.json`; retention and indexed active-job lookup
are follow-up work for the persistent job-store implementation.

## Deployment constraints

Exactly one OCS process or replica may set `scheduler.enabled: true`. Active-job detection
and submission are not an atomic cross-process operation, and the store write lock is
process-local. Enabling the clock in multiple replicas could therefore submit concurrent
writers for the same store. API-only replicas must leave scheduling disabled; they may still
serve the schedules page and API, and the clock owner picks their changes up from the shared
store file.

A read-only instance never starts its scheduler. Use a separate writable operator instance
to maintain data served by a structurally read-only public instance.

## Current scope

Scheduled sync supports previously ingested historical or otherwise append/rematerialize
datasets. A static or future-facing entry is skipped and remains visible in `GET /schedules`
with an error, without preventing unrelated API routes from starting. Refreshing a forecast
requires replacing an overlapping forward window, not merely appending periods after the
latest stored timestamp.

Successful update events can drive openEO workflows through instance-owned bindings; see
[Dataset-update workflow automation](workflow_automation.md). Cross-service orchestration,
distributed leader election, and forecast-window refresh remain later automation phases.
