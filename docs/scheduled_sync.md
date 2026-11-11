# Scheduled dataset synchronization

Open Climate Service can periodically check whether an existing managed dataset has new
source periods and submit an asynchronous sync job when work is required. The scheduler is
a clock only: planning, retries, execution, progress, and restart recovery remain owned by
the native OCS job service.

## Configure schedules

`climate-service.yaml` controls whether the instance clock runs and its timezone:

```yaml
scheduler:
  enabled: true
  timezone: UTC
```

Create and edit schedules from each dataset's **Schedule** tab, or through the
`/schedules/sync` API. All dataset schedules live together in
`<data_dir>/schedules.json`, keyed by dataset id. This file can be prepared before
starting OCS, though the UI/API is recommended for validation and atomic writes.
Only one schedule is allowed per dataset. The schedule's `cron` is a standard
five-field expression interpreted in the configured IANA timezone. UTC is the
default and is recommended for checks that follow upstream publication times.

For pre-start provisioning, one file contains all datasets, for example:

```json
{
  "chirps3_precipitation_daily": {
    "dataset_id": "chirps3_precipitation_daily",
    "cron": "0 6 * * *",
    "publish": true,
    "max_attempts": 3,
    "enabled": true
  },
  "era5land_precipitation_monthly": {
    "dataset_id": "era5land_precipitation_monthly",
    "cron": "0 8 5 * *",
    "publish": true,
    "max_attempts": 3,
    "enabled": false
  }
}
```

The map key and `dataset_id` must match. The optional `enabled` field defaults to
true; the second example stays paused until an operator resumes it. When editing
the JSON outside the UI, stop OCS or replace the complete file atomically so the
clock never reads a partial write.

For a development instance with `scheduler.dataset_sync` in YAML, remove that block
before starting the updated server. Those entries are not imported automatically:
recreate them from each dataset's Schedule tab or through `POST /schedules/sync`
after startup. To provision schedules before startup instead, use the single JSON
file format above. A leftover YAML schedule block is rejected explicitly rather
than silently ignored. Keep `scheduler.enabled` and `scheduler.timezone` in YAML.

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

Schedules can be added, changed, paused and removed without a restart: on the dataset
page, whose Sync panel holds the dataset's schedule; on the **Schedules** page, which lists
everything on the clock and pauses, resumes or deletes any saved entry; and through the
`/schedules/sync` API. The clock reads only `<data_dir>/schedules.json`:

On the dataset page, **Sync now** and **Schedule** share one card; Sync now opens by default,
and an edit link or validation error opens Schedule. Without JavaScript both sections remain
visible and their forms still work. On the Schedule tab, choose **Every day**, **Every week**,
or **Every month**, then set a check time in `scheduler.timezone`. Custom cron remains
available under **Custom (advanced)**.
The page suggests daily checks for hourly, daily and weekly data; weekly checks for dekadal
and monthly data; and monthly checks for yearly data. These are starting points, **not**
publication dates: the dataset period describes what a value represents, while the upstream
source may publish that value later. Choose a check time after its usual release, or check
more often if its delay varies. A check submits a native sync job; if nothing new is available,
the job completes without changing the dataset. Monthly presets use days 1–28 so they do not
skip shorter months. Neither the preset nor a custom cron changes which source periods the
sync engine ingests.

- The configuration file controls the global clock switch and timezone, read at startup;
  the JSON store controls every dataset schedule. Editing that store through the UI/API
  takes effect without restarting OCS.
- A stored entry has a pause switch, `enabled`. Pausing keeps the entry and stops it firing.
- Every change reloads the clock. The reload parses the store, then resolves every effective
  entry's dataset and builds its trigger, before any job is touched. When the store cannot be
  read, the previous schedules stay in force and `GET /schedules` reports `reload_error` until
  the cause is fixed. An entry whose dataset no longer resolves is taken off the clock and
  listed with the reason, so no job keeps firing with settings the status no longer shows. If
  the clock itself refuses a sound plan, the previous jobs are put back and the error says so;
  should that fail too, the error says the clock may run settings the list does not show.
  If a save or delete persists but that reload fails, the API answers 409 with an explicit
  stored-versus-applied result; the page shows the failure rather than claiming the clock
  accepted it. The previous clock plan stays in force until a successful reload.
- A change made through another process reaches the clock. The process that owns the clock
  watches the store file and reloads within 30 seconds of a write from anywhere on the shared
  data directory, so an API-only replica can pause or edit a schedule without knowing which
  replica runs the clock. The request's own process reloads at once. Every process, with or
  without the clock, compares a digest of the store file before it lists schedules, so a
  replica's page and API show a change saved elsewhere straight away.
- An unreadable store file is reported, not hidden. Listings carry `reload_error` with the
  reason, and writes answer 503 with it, until the file is fixed.
- `effective` on a listed entry means enabled: the entry the clock would run.
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
| `GET /schedules` | Everything on the clock with runtime status, each row with its `kind`; the page for a browser |
| `GET /schedules/sync/{dataset_id}` | One dataset's effective sync schedule and status |
| `POST /schedules/sync` | Add a stored sync schedule: `dataset_id`, `cron`, `publish`, `max_attempts`, `enabled`; answers `201` with the schedule in `Location` |
| `PUT /schedules/sync/{dataset_id}` | Change a stored sync schedule; settings left out of the body are kept |
| `POST /schedules/sync/{dataset_id}/pause`, `/resume` | Flip the pause switch |
| `DELETE /schedules/sync/{dataset_id}` | Remove a stored sync schedule |

Sync schedules sit under `/schedules/sync` because they are keyed by the dataset they sync,
one per dataset. Another kind of schedule would get its own path under `/schedules` and appear
in the same listing.

A dataset that is static, forecast-facing or has no registered data source is refused with the
reason. The target must have been ingested for a check to submit a sync; a stored schedule for
a dataset not yet ingested is saved and its checks finish as `not_materialized` until it is.

Changing schedules changes when data moves into the instance, and a sync can refresh derived
datasets and run the pipelines on them, including live deliveries. These routes are closed on a
read-only instance, but read-only mode is not authentication: an instance that exposes the
page must be deployed privately or behind reverse-proxy authentication until OCS has its
own write authentication.

## Inspect status

`GET /schedules` reports, for each schedule, its next check, latest enqueue outcome, message,
and submitted native job ID. This status does not mirror the terminal job state; follow
`links.last_job`, which leads to `GET /ingestions/jobs/{job_id}`, and poll it.

The process that runs the clock records each check's result, and a heartbeat every 30 seconds,
in `<data_dir>/schedule_results.json`. Every process on the same data directory reports from
that file, so a read-only public instance beside a writable worker shows the worker's checks,
and the clock's own process still shows them after a restart. `enabled` and `running` describe
the answering process; `clock` says whether a clock runs on the data directory, whether it is
this process (`here`) and when it last reported (`beat_at`), and is `null` when none has
reported in the last 90 seconds. Which host and process run it stays in the file. A process
that does not run the clock computes the next check from the cron, in the timezone the clock
reports, and only while a clock reports, since otherwise nothing fires.

Every due check creates a native job, including checks that finish without finding new periods.
Native job history is currently retained in `jobs.json`; retention and indexed active-job lookup
are follow-up work for the persistent job-store implementation.

## Failure alerts

OCS can post to a webhook when a scheduled sync or a pipeline starts failing, and again when
that ends. Set the URL in `climate-service.yaml`:

```yaml
notifications:
  webhook_url: ${OCS_WEBHOOK_URL}
```

OCS sends an HTTP JSON POST. A directly compatible receiver displays the `text` field, ignores
fields it does not recognise, and carries its authentication in the webhook URL, because custom
headers cannot be configured. Slack and Mattermost incoming webhooks meet this contract. Strict
receivers, and services that require their own payload format such as Google Chat, Microsoft
Teams and Discord, need a small relay that translates the request.

Each schedule and each pipeline posts once when it starts failing and once when that ends, not
once per failure: a daily sync that keeps failing alerts on the first morning only.

A scheduled sync fails when its job fails after its last attempt, and also when its check fails
before any job exists: the check errors, the dataset has not been ingested, or the clock refuses
the schedule when it loads it (an unknown data source, a cron it cannot read). Such a check is
recorded as a failed job, so it shows in the jobs list too; a check that fails the same way
again links that job rather than adding another. The schedule recovers when a sync
job next succeeds.

A pipeline run fails when its job cannot be submitted, its workflow job errors or disappears, or
its delivery fails or ends partial, rejected or unknown; it succeeds when the delivery is
confirmed or validated. A cancelled job is neither. The pipeline is failing while it has a failed
run that no newer successful run superseded. It recovers when a newer run succeeds. It is
resolved when every failure since the last success is a delivery that ended unconfirmed and an
operator has acknowledged each of them. A submission or workflow-job failure stays open until a
newer run succeeds, and acknowledging an older run while a newer one still fails changes nothing.

The body is JSON, posted with the header `Idempotency-Key` set to the outcome's event id:

```json
{
  "status": "failed",
  "event": "8f1c...:0",
  "time": "2026-11-06T06:00:12+00:00",
  "automation": "pipeline",
  "id": "rainfall-to-dhis2",
  "link": "https://ocs.example.org/pipelines/rainfall-to-dhis2",
  "run": "8f1c...",
  "details": "https://ocs.example.org/pipelines/rainfall-to-dhis2/runs/8f1c...",
  "stage": "delivery",
  "message": "DHIS2 refused the payload",
  "text": "Pipeline 'rainfall-to-dhis2' failed at delivery: DHIS2 refused the payload https://ocs.example.org/pipelines/rainfall-to-dhis2"
}
```

`status` is `failed`, `recovered` (a run succeeded) or `resolved` (an operator acknowledged the
failure, which is not a success; only pipelines have this). A receiver that only opens and closes
alerts can treat `recovered` and `resolved` alike. `automation` is `pipeline` or `schedule`; a scheduled sync
names its `job` instead of a `run`. `text` is the one-line summary that chat tools show. Links
are absolute when `CLIMATE_SERVICE_BASE_URL` is set, and paths otherwise; either way they carry
the deployment prefix, the base URL's path or else `ROOT_PATH`.

The outcomes are durable events, persisted with the job or run that reached them, and each alert
is queued in `<data_dir>/notifications.json` before it is posted. An alert the receiver does not
answer with 2xx is retried on the 15-minute maintenance interval with a growing delay, for about
five hours. Each schedule's and pipeline's alerts arrive in order, and one that is refused holds
up only its own. A failure alert given up on drops the alert that would end it too, so a
receiver never gets a recovery or resolution alone; a later failure is still posted. After a restart, outcomes that no process decided are decided then.
Outcomes from before the webhook was first configured are not posted. A read-only instance posts
nothing.

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
