# Using the web interface

Every Open Climate Service instance ships with a small built-in web interface — no
separate application to install. Once the instance is running, open its root URL
(`http://127.0.0.1:9000` by default) and the landing page links to everything below.

| Page           | URL       | What it does                                                                       |
| -------------- | --------- | ---------------------------------------------------------------------------------- |
| Landing page   | `/`       | Instance overview, the datasets it holds, the templates and workflows it offers |
| **Map viewer** | `/map`    | View published datasets on an interactive map                                      |
| openEO editor  | `/openeo` | Redirects to the openEO Web Editor, pre-connected to this instance                 |
| API docs       | `/docs`   | Interactive Swagger documentation for the REST API                                 |

The interface is intended for operators setting up and curating an instance. Everything
it does is also available through the REST API, so the same operations can be scripted or
scheduled — see the [API reference](managed_data_api_guide.md).

---

## The overview and the collections

Every page carries the same navigation on the left (a row of tabs on a narrow screen), and each
entry is a page of its own — an address that can be bookmarked, shared or opened without
JavaScript.

| Page                      | What it shows                                                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Overview** (`/`)        | The instance's extent on a globe, counts of datasets, data sources and workflows, the size of everything stored, plus access mode and version |
| **Datasets** (`/datasets`) | The data this instance holds, with temporal coverage and publication status                                                        |
| **Data sources** (`/data-sources`) | Data the instance can fetch from outside providers, rasters and feature collections, titled by dataset with the provider beneath |
| **Workflows** (`/workflows`) | Each workflow and what it makes: a published dataset or an exported file                                                         |
| **Schedules** (`/schedules`) | Everything on the instance's clock: each dataset's sync schedule, with pause, resume and delete; set up and edited on the dataset page |
| **Pipelines** (`/pipelines`) | Run a registered workflow over its bound inputs and deliver the result to a destination such as DHIS2: validate, dry-run, go live, run, pause and delete |
| **Processes** (`/processes`) | The processes this instance can run, tagged by origin and filterable by it                                                       |

The overview counts the collections and links to them rather than listing them, so the root
stays small however much the instance holds.

The Datasets and Data sources lists can be searched and filtered, and are shown a page at a
time. Without JavaScript each list is complete.

Datasets and data sources can be shown as **tiles** or as a **list**; the choice is
remembered in the browser. Data sources show their provider, description and details, with no
preview, since they hold no data yet.
Each dataset shows its thumbnail, source, a short description, publication status, period,
temporal coverage and units. A dataset ingested before thumbnails existed shows its colour
scale instead, until its next sync renders one.

### The data source page (`/data-sources/{dataset_id}`)

Selecting a data source opens its page: the description, what the data is (variable, units,
period, available range, resolution and coverage), the provider and licence, how it updates,
and its colour scale. If it has already been ingested, the page links to that dataset.

The page also has an **ingest form**. It takes a start and an optional end, prefilled for the
kind of source — the past year for historical data, blank (meaning "from now") for a forecast,
and the full declared range for a source that runs into the future. Progress is shown on the
page; when ingestion finishes the dataset page opens, and an error is shown in place. Data is
always fetched for the instance's configured extent. The form is not shown on a read-only
instance or when no extent is configured.

A **feature collection template** (boundaries or points, labelled *Features* in the list) has a
page of its own: the provider, the pinned release, any filters such as the administrative level,
and the licence with what it requires when the data is served. Its form has no date range, since
a collection has no time axis: **Fetch** runs the provider and opens the collection when it
finishes, and fetching again replaces it. A template whose provider this instance lacks is not
listed.

### The API page (`/api`)

The API page is built from the instance's own OpenAPI schema, so it lists the endpoints that
instance actually serves, grouped by area — datasets, ingestion, sync, Zarr and Icechunk
access, STAC, openEO, and the rest — each with what it does. It also links the STAC catalogue,
the openEO capabilities and collections, the Swagger documentation and the OpenAPI schema. On
a read-only instance the endpoints that refuse are marked.

### The workflow page (`/workflows/{workflow_id}`)

Selecting a workflow opens its page: what it does, including its usage example, whether it
publishes a dataset or exports a file, its parameters with their types and defaults, the
datasets it produces (linked where they have been made), and any automation configured to
run it when a dataset is updated. The process graph itself is at
`/process_graphs/{workflow_id}`. Workflows cannot yet be started from the page.

### The process page (`/processes/{process_id}`)

Selecting a process opens its page: its description, origin and categories, its parameters
and return value, the workflows on this instance that use it, and its reference links. As
with datasets, the URL still returns the openEO process description as JSON to API clients;
a browser gets the page, and `?f=json` and `?f=html` choose explicitly.

### The Schedules page (`/schedules`)

The Schedules page lists everything on the instance's clock. Today that is one kind, a sync
schedule: at each check time OCS submits a sync job, which adds new source data if any is
available. Each row names its kind in the first column, so the list can carry other kinds later.

A sync schedule belongs to exactly one dataset, so it is set up and edited on the dataset page,
next to the manual sync it automates (see below). The Schedules page is where they are seen
together, and it covers what a dataset page cannot: a saved schedule whose
dataset is no longer on the instance, which can still be paused or deleted; and the
instance-wide reload error. Each saved row offers Edit, which opens the dataset page, Pause or
Resume, and Delete, which asks for confirmation in a dialog.

Only one schedule is allowed per dataset. All schedules are stored in one file,
`<data_dir>/schedules.json`, and can be managed from the UI or API. The instance-wide
clock switch and timezone remain in `climate-service.yaml`. A static or forecast
dataset cannot take a schedule, and the API refuses it with the reason.

Each row shows a readable check time (or cron for a custom schedule) and timezone,
its status (scheduled, paused, or unable to run
with the reason), the next check and the last one. Check
state is kept in memory and resets on restart; the sync jobs a check submits are durable and
linked from the row. The timezone is the instance's `scheduler.timezone`. When the scheduler is
disabled, schedules can still be saved and are listed, and the page explains that an operator
enables checks by setting `enabled: true` under `scheduler:` in `climate-service.yaml` and
restarting OCS. If a change cannot be applied, the page shows why.
### The pipeline pages (`/pipelines`)

For a new setup that runs a workflow on a dataset and delivers the result, use a pipeline. The
older [workflow triggers](workflow_automation.md) and the named `exports` in
`climate-service.yaml` keep working for existing configurations, and react to the same dataset
updates, but are due to be retired in favour of pipelines.

A **workflow** is a reusable, parameterized sequence of processes that defines how data is
transformed or analyzed. A **pipeline** is a managed configuration that connects one workflow to
its inputs and a destination, governing when it runs, how results are delivered, and how
execution is monitored. The pipeline holds no processing of its own: it names a registered
workflow and binds its arguments, names an export plugin as its destination with that plugin's
settings, and sets a delivery mode and policy. Several pipelines can use the same workflow with
different arguments, destinations and policies.

```yaml
id: rainfall-to-dhis2
name: Monthly rainfall to DHIS2
workflow:
  id: aggregate_by_features
  arguments:
    dataset_id: era5land_precipitation_monthly
    geometries: { from_features: districts }
    method: mean
destination:
  plugin: dhis2
  settings:
    connection: national_hmis
    data_set: BfMAe6Itzgt        # optional, checked for period type and assignment
    series:
      - data_element: LeePHYkZwBk
delivery:
  mode: dry_run                  # dry_run | live | paused
  policy: on_update              # on_update | manual
```

The workflow's arguments are checked against its declared parameters: unknown, missing or
ill-typed arguments are refused with the reason. A feature collection is bound with
`{from_features: <id>}`, as for workflow triggers, and pinned to its current version when a run
is submitted.

In this first iteration a pipeline supports workflows that meet a narrow contract. It is a
limit of what pipelines support today, not of workflows in general:

- exactly one `temporal-interval` parameter is left unbound, and every run fills it with its own
  interval (bind any other interval parameters explicitly);
- the workflow returns data and does not end in its own `save_result`, because a pipeline saves
  through its destination;
- a source dataset is a parameter declared with `subtype: collection-id`. It is checked to be
  published, and its updates start runs. A pipeline that runs on updates must bind one; a
  dataset passed through a plain string parameter is not recognised as a source.

Of the built-in workflows only `aggregate_by_features` meets it: the `aggregate_to_*` workflows
save their own result, and the climate normal, anomaly, change and dekad workflows write a
managed dataset. The create form lists only the workflows that meet the contract.

The destination's settings belong to the export plugin. The plugin validates them, compiles
them into the named export the pipeline's results are saved and delivered through, and runs its
own destination checks. For DHIS2 these are the connection, the data element and data set UIDs,
that the dataset's cadence is a DHIS2 period, that the organisation units come from a DHIS2
collection fetched over the same connection, and, when a data set is declared, its period type,
data elements and organisation-unit assignments. The pipeline core knows none of this.

The create form is driven by the same metadata: choose a workflow and a destination, and the form
shows a field per workflow parameter (a dataset list, a feature collection list, a choice, a
value, or JSON for anything else) and per destination setting. **Create pipeline** on a dataset's
page opens the form with that dataset bound. **Save** validates before it stores and refuses a
pipeline that fails any check, keeping the form. The validation also reports how each source
dataset is kept current: the pipeline owns no sync clock. **Re-check** repeats the validation.

A **dry run** executes the exact graph a run would submit over a bounded range, renders it through
the destination and, where the destination supports it, submits it as a dry run (DHIS2 validates
with `dryRun=true` and stores nothing). A passed dry run approves the bindings it ran on: the
workflow reference and arguments, the workflow's own definition, and the destination settings.
Changing any of them needs a new dry run before live delivery; renaming the pipeline or switching
its mode does not. **Go live** re-runs every check, the destination's included, and needs that
approval.

Every run goes through one path, whoever starts it: **Run once** over a chosen interval, **Run
again** for a failed run, and a source dataset update for a pipeline whose policy is `on_update`.
The run is stored with its
interval, trigger, workflow arguments, bindings and deterministic job id before the job is
created, then submitted as an openEO batch job of the workflow saved through the pipeline's
export. The workflow is frozen into that graph: the call, and any workflow it calls, is inlined
from the registry as it was when the run was submitted, with declared defaults applied, so a
queued or retried job runs the definition that was approved even if the workflow is edited
meanwhile. When the job finishes, its result is delivered through the same delivery path as
`POST /exports/{id}`, under the run's idempotency key, so a repeated completion or a restart does
not deliver twice.

Each run moves through these states; the page shows them and the action each needs:

| State | Meaning | What happens next |
|---|---|---|
| submission `pending` | stored, job not created yet | retried automatically, up to 5 attempts |
| submission `failed` | refused (an invalid graph), or the attempts ran out | terminal: **Run again** |
| job `error`, `canceled`, deleted | the job ended without a result (the job service already retried transient failures) | terminal: **Run again** |
| delivery `awaiting` | the job is running, or a transient hand-off failure is being retried | handed off when the job finishes; retried automatically, up to 5 attempts, always under the same idempotency key |
| delivery `held` | the pipeline was paused, or a live result waits for the pipeline to be live with the same approved bindings | released automatically on resume once eligible; **Deliver** for a held dry run |
| delivery `failed` | the hand-off was refused (a changed export, a missing result) or its attempts ran out | **Deliver** retries it under the same key, so nothing is sent twice |
| delivery `submitted` | handed to the delivery framework, not yet delivered | the delivery job's outcome is recorded on the run: `delivered` (or `validated` for a dry run) confirms it; `partial` (conflicts), `rejected`, `unknown` (no answer after sending, or the delivery job's record is gone), `failed` and `cancelled` need an operator: **Send again** where that is safe, or **Checked at the destination**, which records `acknowledged` and sends nothing |

OCS recovers the automatic states at startup and every 15 minutes: it submits pending runs,
hands off finished jobs whose completion was missed, records jobs that ended without a result,
retries transient hand-off failures, and releases held deliveries that have become eligible.
A failure is permanent when repeating it can only fail the same way: an HTTP 4xx refusal (a
changed export, a missing result, a reused key) or an invalid value. A 408, 425 or 429, a
conflict that only says the job's result or the export is busy right now, and any other error
(the store, a lock, the network) are retried within the bound.

Retries never send twice. A hand-off is retried under the run's one idempotency key, which the
delivery framework reserves before it queues anything, so a repeated hand-off returns the
delivery already queued. A run counts as delivered only when its delivery job confirms it; until
then it stays in the history. **Send again** is the only way a payload is sent a second time: an
operator action, only for a delivery that ended unconfirmed (`failed`, `rejected`, `partial`,
`unknown`, `cancelled`), only where the destination plugin declares that a repeat is harmless
(`redelivery_safe`; DHIS2 imports with `CREATE_AND_UPDATE`, which overwrites the same values), and
under a new key. Elsewhere it is refused: check what arrived at the destination, then acknowledge
the delivery or run the interval again. **Send again** checks everything first (the outcome, the
destination, the source job, and whether the pipeline may deliver now: a live run needs the
pipeline live) and records the new round only together with the delivery that opens it, so a
refusal leaves the earlier delivery exactly as it was. If sending fails after those checks, the
earlier delivery is kept and only the error is noted; the next attempt reuses the same new key,
so a delivery the framework had already reserved is found rather than sent twice.

**Dataset updates.** An update describes itself in one of two ways, and a run covers what it
says changed, not the dataset's whole coverage:

- an *append* (`previous_end` set) added the periods after `previous_end` and kept the earlier
  ones as they were. The run covers `previous_end` to `current_end`, re-sending the last period
  already stored, an overwrite at the destination;
- a *rewrite* (`previous_end` empty: a new store or a rematerialisation) may have changed any
  period, so the run covers the whole coverage after it, `current_start` to `current_end`.

The run records which (`source.update`). Updates do not describe a correction of individual
stored periods: OCS has no such update, so a pipeline cannot follow a revision of past data at
the source (for example a provisional value replaced by a final one) unless the dataset is
rematerialised. Run that interval by hand. The interval's ends are passed to the workflow as the
updates give them, as for workflow triggers.

Each pipeline keeps a durable checkpoint of the updates it has decided on, written in the same
write as the run an update starts. What happens to an update:

- **run**: at most once per pipeline, whatever the mode, retries or restarts. Pruning the run
  history does not change that: the checkpoint, not the history, records the decision;
- **ignored**: an update for another dataset, or older than the moment the pipeline started
  running on updates with its current mode and bindings. Switching mode or changing the workflow
  or destination moves that moment, so earlier updates are never delivered live or run with other
  bindings;
- **not run, and listed**: the pipeline refused it (invalid, or changed since it was validated:
  re-check, then run its interval by hand), it names no interval, starting its run failed for a
  reason that cannot pass (a missing feature collection, an unresolvable export), a transient
  failure persisted over 5 attempts, it arrived while the pipeline was paused, or it arrived after
  the checkpoint had passed its time. The page lists these with their interval under **Updates
  not run**; none is run later by itself. The list is bounded to the 50 most recent: it is a
  worklist for an operator, not an audit trail. When older entries are discarded, the pipeline
  counts them and records the time of the newest discarded update, and the page says so; the
  updates themselves remain in the native job history;
- **retried**: starting its run failed for a transient reason. It stays undecided, holds the
  checkpoint back and is tried again at the next start, up to 5 attempts.

A destination that cannot be reached does not stop an update's run: starting a run never
contacts the destination, and its delivery retries transient failures. Updates that arrive while
paused are not run on resume: the checkpoint starts afresh, and the page lists them so an operator
can run their intervals.

At startup OCS decides any persisted update a pipeline missed, then moves the checkpoint forward
only to the earliest of: the start of that pass, the creation of any ingestion still running, and
any update still being retried; less 15 minutes. An ingestion's update carries its completion
time, which is never earlier than the job's creation, so an update from a job that was running
or not yet created at that pass is always after the checkpoint, however late it arrives. The
15 minutes cover the moment between an update's time being taken and its record being written,
for work recorded inline rather than through a queued job. An update can be missed only if that
moment exceeds 15 minutes, or if instances sharing a data directory disagree on the time by more
than that; such an update, when it arrives, is listed as arriving after the checkpoint rather than
lost silently. Exactly once is therefore not unconditional: at most once always holds, at least
once holds within those limits. Updates are read from both job stores, the native one (ingestion
and sync) and the openEO one (a published derived dataset), neither of which has a cursor, so the
startup pass reads them once, as workflow automation does; the checkpoint makes each pipeline
skip decided updates without further work. An openEO job's update needs no unfinished-job bound:
its time is taken inside the store write that records it.

**Pause** stops new automatic runs and holds deliveries: a job already running finishes and keeps
its result, and the delivery it owes is held. An operator can still run and deliver a dry run
while paused. Resuming releases what is eligible: held dry-run deliveries on any resume, held live
ones once the pipeline is live again, approved for the bindings they were computed with. Resuming
does not replay the updates that arrived while paused; run their interval by hand. A scheduled
release time is not implemented yet and is not accepted.

A pipeline's export id is its own id, unique across the pipelines and the `exports` in
`climate-service.yaml`. An id configured in both is refused, when the pipeline is validated and
again when the export is resolved, rather than one silently winning. Pipeline records are kept
under `<data_dir>/pipelines`; every change is a locked read-modify-write, and an edit names the
revision it was based on, so a stale edit is refused rather than overwriting a newer one. A
check, dry run or mode switch computed while the pipeline changed is refused too, and only a
validation of the current revision publishes the pipeline's export. The run history keeps the
newest 50 runs, and every older run until it is delivered or run again: a run that is pending,
running, held, failed or ended without a result stays, whatever its age. A run's process graph is
kept only until its job holds it.

**Delete** removes the pipeline and its run history after a confirmation; jobs and deliveries
already made are kept. The API offers the same: `POST /pipelines` and `POST /pipelines/{id}` to
save (with an optional `revision`), `DELETE /pipelines/{id}`, `POST /pipelines/{id}/validate`,
`POST /pipelines/{id}/dry-run` and `POST /pipelines/{id}/runs` with `start` and `end` (and `mode`
for a run), `POST /pipelines/{id}/runs/{run_id}/deliver`, `POST /pipelines/{id}/runs/{run_id}/redeliver`,
`POST /pipelines/{id}/runs/{run_id}/retry`,
`POST /pipelines/{id}/runs/{run_id}/acknowledge`,
and `POST /pipelines/{id}/mode/{dry_run|live|paused}`.

### The dataset page (`/datasets/{dataset_id}`)

Selecting a dataset opens its page: a larger preview, the full description, and everything
known about it — variable, units, period, coverage, resolution and bounding box; source,
licence, providers and how it is made (fetched, or produced by a named workflow); publication
and update status; colour scale; and its version history. Links lead to the map viewer, the
Zarr store, the STAC collection and the JSON metadata.

The same URL still returns JSON to API clients. A browser, which asks for HTML first, gets the
page; `?f=json` and `?f=html` choose explicitly.

Data sources and Workflows together cover every data source registered on the instance:
one that can be ingested is listed under Data sources, and one that a workflow writes is
listed under that workflow (see [Templates that are produced, not
ingested](adding_custom_datasets.md#templates-that-are-produced-not-ingested)).

---

## Ingesting and syncing data

There is no separate console: data is added from the page of the thing it concerns.

- **Ingest** from a data source page (`/data-sources/{dataset_id}`): enter a start and an
  optional end, choose whether to publish and whether to overwrite an existing store, and
  start. Progress streams on the page; the dataset page opens when it finishes.
- **Fetch** a feature collection from its data source page: choose whether to publish and start.
  Progress streams on the page; the collection opens when it finishes.
- **Sync** from a dataset page (`/datasets/{dataset_id}`): the page shows what the source
  has published since the last sync, and **Start sync** fetches it, optionally only up to a
  cutoff date. The page reloads with the new coverage when it finishes. A sync keeps the
  dataset's current publication state — a published dataset stays published — because an
  incremental sync appends to the store the published version already points at.
- **Schedule** the sync on the same dataset page: **Sync now** and **Schedule** are tabs in
  one card, with Sync now shown first. On Schedule, choose daily, weekly or monthly checks,
  a time, and up to how many attempts a sync job may make. Custom cron is available for
  advanced needs. The suggested check frequency reflects the data's period type, but the
  operator should choose a time after the source normally publishes; period type does not
  tell OCS that release time. A schedule can be saved paused. Once saved, it shows the next
  and last check and offers Edit, Pause or Resume, and Delete behind a confirmation.
  Editing does not silently change its publication setting. Every schedule on the
  instance is listed on the [Schedules page](#the-schedules-page-schedules).

You do **not** enter a bounding box — ingestion always uses the spatial extent configured
for the instance in `climate-service.yaml`. On a read-only instance neither form is shown.

---

## The map viewer (`/map`)

The map viewer renders **published** datasets directly in the browser: rasters from their
GeoZarr stores, vector datasets from their GeoParquet. Only published datasets appear here.

- **Dataset selector** — pick any published dataset from the dropdown, grouped into raster and
  vector datasets when the instance has both.
- **Vector datasets** — drawn as filled areas, lines or points, with the feature's name shown
  on click. The viewer reads the collection's stored file (`/features/{id}/data.parquet`),
  which has to be in WGS 84.
- **Dimension controls** — the viewer builds one control per non-spatial dimension of the
  dataset, choosing the type from the dimension's metadata: a **slider** for a continuous,
  evenly-spaced axis (time, or a regular ordinal axis like day-of-year) and a **dropdown**
  for a categorical or irregular one.
- **Legend** — a colour bar with the value range and units, derived from the dataset's
  metadata (including the colour scheme defined in its template).
- **Source and units** — shown alongside the legend for context.
- **Link to the dataset** — opens the dataset's page.
- The map fits to the instance's configured extent on load.

The viewer sits under the same header and navigation as the other pages. To link to one
dataset, use `/map?dataset={dataset_id}`: the viewer opens with that dataset selected, and
choosing another dataset updates the address, so what is in the address bar can always be
shared. A dataset page's **Open in map viewer** button uses the same link.

If a dataset doesn't show up, confirm it was ingested with **Publish** enabled — only
published datasets are listed.

---

## When to use the API instead

The web interface is the quickest way to set up and curate an instance by hand. For
**automation, scheduling, or programmatic access** — recurring ingestion, scripted sync,
or integrating with other systems — use the REST API directly. The management forms map
one-to-one onto the `POST /ingestions` and `POST /sync/{dataset_id}` endpoints documented
in the [API reference](managed_data_api_guide.md). To read and analyse published data, see
[Accessing data](user_guide.md).
