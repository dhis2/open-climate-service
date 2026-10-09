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
| **Pipelines** (`/pipelines`) | Save, dry-run, run, pause and delete a climate-to-DHIS2 delivery: a dataset aggregated to organisation units and sent to a data element |
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

A pipeline binds one published climate dataset to polygon organisation units, a named DHIS2
connection, a spatial reducer, and one or more DHIS2 data elements, and decides whether, which
and when values are delivered. It never fetches data: keeping the dataset current is the
dataset's own sync schedule, configured under `scheduler.dataset_sync`, and the pipeline reacts
to the dataset's updates. The create page offers only stored polygon collections; a DHIS2
collection also has to come from the same named connection as the destination. The
**Configured pipelines** tab lists saved pipelines, while **Create pipeline** opens the form at
`/pipelines/new` without mixing it into the listing.

**Save** validates before it stores. Every binding is checked, in OCS and in DHIS2: the dataset
and its cadence, the collection's ids, the connection, the DHIS2 data set's period type, data
elements and organisation-unit assignments when a data set is declared, and that no configured
export or trigger already uses the same id differently. The validation also reports how the
source dataset is kept current: synced on a schedule, scheduled but with the instance scheduler
disabled, or not scheduled at all, in which case the pipeline runs after a manual sync or when
run once. A pipeline that fails any check is not saved, and the form comes back with the failed
checks. The same holds for an edit: **Edit** opens the saved pipeline in the form, and a change
that fails a check is refused and the stored version kept. A change to the bindings clears the
last dry run, because that run was judged on bindings that no longer hold. The pipeline page
keeps the validation it was saved on; **Re-check** repeats it, since the dataset, the collection
and DHIS2's metadata can all move afterwards.

Delivery is controlled by three settings. The **mode** says whether anything goes out: `dry_run`
sends every payload with `dryRun=true`, so DHIS2 validates and stores nothing; `live` writes
values, and is allowed only after a dry run has passed; `paused` keeps the pipeline valid and
the dataset current and sends nothing. The **values** setting says which periods go: only the
updated interval, or the complete stored history on every update. The **policy** says when:
after each dataset update, only when an operator runs or delivers by hand, or at a release time
of its own. The scheduled release is accepted so the shape is final, but it is not applied yet;
until stored pipelines are read by the automation it behaves as manual, and the validation says
so. The page switches the mode with **Go live**, **Pause**, **Back to dry run** and the resume
buttons; each switch re-validates and is refused when a check fails.

A bounded dry run renders the exact generated named-export mapping and sends it through the DHIS2
export plugin with `dryRun=true`. **Run once** goes one step further: it submits the aggregation
over a chosen range as a batch job through the pipeline's own named export, and delivers the
result when the job finishes, through the same delivery path as `POST /exports/{id}`. A run is
delivered as a dry run by default; a live run, which writes values into DHIS2, is offered only
after a dry run has passed. Run once works in every mode, paused included, because it is the
operator's own action. The page lists each run with its job and its delivery, and offers
**Deliver** for a finished run whose hand-off did not happen, for example because the instance
restarted. A saved pipeline whose validation passed is resolvable as a named export without any
change to the instance configuration.

Unattended delivery stays configuration-based. The page shows the generated `exports` entry and,
for a pipeline that delivers after each update and is not paused, the `automation` trigger, to
merge into `climate-service.yaml` before a restart. The dataset's sync schedule is not part of
the fragment; it is managed with the instance's schedules, and a successful scheduled sync
triggers processing and delivery, so DHIS2 has no separate delivery clock.

**Delete** removes the pipeline and its run history from OCS after a confirmation. Anything
already merged into the instance configuration stays until an operator removes it. The API offers
the same: `POST /pipelines` and `POST /pipelines/{id}` to save, `DELETE /pipelines/{id}`,
`POST /pipelines/{id}/mode/{dry_run|live|paused}`, `POST /pipelines/{id}/runs` with `start`,
`end` and `mode`, and `POST /pipelines/{id}/runs/{job_id}/deliver`.

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
