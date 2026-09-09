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
writers for the same store. API-only replicas must leave scheduling disabled.

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
schedule mutation APIs, distributed leader election, and forecast-window refresh remain later
automation phases.
