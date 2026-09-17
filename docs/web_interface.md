# Using the web interface

Every Open Climate Service instance ships with a small built-in web interface — no
separate application to install. Once the instance is running, open its root URL
(`http://127.0.0.1:9000` by default) and the landing page links to everything below.

| Page           | URL       | What it does                                                                       |
| -------------- | --------- | ---------------------------------------------------------------------------------- |
| Landing page   | `/`       | Instance overview, the datasets it holds, the data sources and workflows it offers |
| **Map viewer** | `/map`    | View published datasets on an interactive map                                      |
| openEO editor  | `/openeo` | Redirects to the openEO Web Editor, pre-connected to this instance                 |
| API docs       | `/docs`   | Interactive Swagger documentation for the REST API                                 |

The interface is intended for operators setting up and curating an instance. Everything
it does is also available through the REST API, so the same operations can be scripted or
scheduled — see the [API reference](managed_data_api_guide.md).

---

## The landing page (`/`)

The landing page is split into five areas, chosen from the navigation on the left (a row of tabs
on a narrow screen). One area shows at a time, and each has its own address, such as
`/#data-sources`, so it can be bookmarked or linked.

| Area             | What it shows                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| **Overview**     | Counts of datasets, published datasets, data sources and workflows, plus the extent and access mode |
| **Explore**      | The map viewer, openEO editor, STAC catalog, API docs and the JSON root                             |
| **Datasets**     | The data this instance holds, with temporal coverage and publication status                         |
| **Data sources** | Data the instance can fetch from outside providers, titled by dataset with the provider beneath     |
| **Workflows**    | Each workflow and what it makes: a published dataset or an exported file                            |

The Datasets and Data sources lists can be searched and filtered, and are shown a page at a
time. Without JavaScript every area is shown in sequence and each list is complete.

Datasets and data sources can be shown as **tiles** or as a **list**; the choice is remembered
in the browser. Data sources show their provider, description and details, with no preview,
since they hold no data yet.
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

### The workflow page (`/workflows/{workflow_id}`)

Selecting a workflow opens its page: what it does, including its usage example, whether it
publishes a dataset or exports a file, its parameters with their types and defaults, the
datasets it produces (linked where they have been made), and any automation configured to
run it when a dataset is updated. The process graph itself is at
`/process_graphs/{workflow_id}`. Workflows cannot yet be started from the page.

### The dataset page (`/datasets/{dataset_id}`)

Selecting a dataset opens its page: a larger preview, the full description, and everything
known about it — variable, units, period, coverage, resolution and bounding box; source,
licence, providers and how it is made (fetched, or produced by a named workflow); publication
and update status; colour scale; and its version history. Links lead to the map viewer, the
Zarr store, the STAC collection and the JSON metadata.

The same URL still returns JSON to API clients. A browser, which asks for HTML first, gets the
page; `?f=json` and `?f=html` choose explicitly.

Data sources and Workflows together cover every dataset template registered on the
instance: a template that can be ingested is a data source, and one that a workflow writes is
listed under that workflow (see [Templates that are produced, not
ingested](adding_custom_datasets.md#templates-that-are-produced-not-ingested)).

---

## Ingesting and syncing data

There is no separate console: data is added from the page of the thing it concerns.

- **Ingest** from a data source page (`/data-sources/{dataset_id}`): enter a start and an
  optional end, choose whether to publish and whether to overwrite an existing store, and
  start. Progress streams on the page; the dataset page opens when it finishes.
- **Sync** from a dataset page (`/datasets/{dataset_id}`): the page shows what the source
  has published since the last sync, and **Start sync** fetches it, optionally only up to a
  cutoff date. The page reloads with the new coverage when it finishes.

You do **not** enter a bounding box — ingestion always uses the spatial extent configured
for the instance in `climate-service.yaml`. On a read-only instance neither form is shown.

---

## The map viewer (`/map`)

The map viewer renders **published** datasets directly in the browser from their GeoZarr
stores (using MapLibre and zarr-layer), so only datasets ingested with publishing enabled
appear here.

- **Dataset selector** — pick any published dataset from the dropdown.
- **Dimension controls** — the viewer builds one control per non-spatial dimension of the
  dataset, choosing the type from the dimension's metadata: a **slider** for a continuous,
  evenly-spaced axis (time, or a regular ordinal axis like day-of-year) and a **dropdown**
  for a categorical or irregular one.
- **Legend** — a colour bar with the value range and units, derived from the dataset's
  metadata (including the colour scheme defined in its template).
- **Source and units** — shown alongside the legend for context.
- The map fits to the instance's configured extent on load.

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
