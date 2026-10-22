# Open Climate Service

Agent context for the `open-climate-service` repository. Provider-agnostic — intended to be readable by any AI coding assistant.

## Project overview

The Open Climate Service is a FastAPI-based REST API that downloads, processes, and serves climate and Earth Observation data as GeoZarr stores.

Key concepts:

- **Data sources** — upstream datasets this instance can fetch, each described by a **dataset template**: a YAML entry in `plugins/rasters/` (variable, period type, download plugin) or `plugins/vectors/` (feature provider and its params). These are blueprints; the data sources are listed at `GET /data-sources` and on the Data sources page.
- **Artifacts / managed datasets** — ingested instances of a template for a specific spatial extent and time range. Exposed under `/datasets` and `/zarr/{dataset_id}`.
- **Extent** — a single named spatial bounding box configured at instance setup time (`id`, `bbox`, optional `country_code`). Exposed at `GET /extent`.
- **GeoZarr stores** — datasets are stored as chunked Zarr v3 archives with GeoZarr spatial attributes. Flat stores for small extents; multiscale pyramids for large ones. Served chunk-by-chunk over HTTP with no specialised server middleware.
- **Feature collections** — vector datasets (org unit polygons, facility points) stored as GeoParquet under `<data_dir>/vectors`, tracked by the same `ArtifactRecord` as rasters and discriminated by `itemType: "feature"`. A record is what makes a collection exist: the listing reads records, never the filesystem, so the store directory is not an inbox.

## Repository layout

```
open_climate_service/
  data_manager/     # download and zarr build (downloader.py)
  data_accessor/    # open zarr / netcdf for read (accessor.py)
  data_registry/    # raster dataset template loading
  ingestions/       # artifact lifecycle: create, list, sync, publish
  features/         # feature collection store, GeoParquet reader, GET /features
  publications/     # STAC publication metadata
  extents/          # spatial extent config
  shared/           # dhis2 adapter, time utils
  main.py           # FastAPI app, CORS middleware, route registration
open_climate_service/plugins/rasters/   # built-in raster dataset templates (chirps3.yaml, worldpop.yaml, …)
open_climate_service/plugins/vectors/   # built-in vector dataset templates and providers
tests/
docs/
```

## Development

```bash
make run      # start uvicorn with --reload
make fix      # ruff autofix + format, whole tree
make lint     # ruff check + ruff format + mypy + pyright
make test     # pytest
make start    # docker compose up --build
```

The `.env` file is required for `make run`. Copy `.env.example` if it exists.

Before handing work back, fix and format only the files you changed (`uv run ruff check --fix <files>`, `uv run ruff format <files>`), then run `make lint` and `make test`. Do not run `make fix`: it rewrites the whole tree, including a developer's uncommitted work. All of `make lint` must pass, not only ruff and mypy: pyright runs in strict mode and rejects code mypy accepts, such as a parameter reassigned after its `None` check.

## Working rules

- **Test through the entry point.** Drive the route, `fetch_period`, or the publish path, not the helper underneath. Bugs found in review usually sit in a caller that bypassed a correct, well-tested helper.
- **Count the places that enforce a rule.** If the same rule is checked in more than one place, decide it once and call that.
- **Run the real path once.** Ingest a period or call the route on a local instance and read the output. Implausible values and leaked metadata show up there, not in unit tests.
- **Size defaults for a laptop.** OCS normally runs on one. Concurrency, thread pools and memory-heavy work default low; servers raise them through settings.
- **Build the simplest version first.** Add server-side machinery for speed or size only when the plain approach is too slow where OCS runs, and name the optimisation as a follow-up instead.
- **No backward compatibility yet.** Renames, schema changes and forced re-downloads on existing instances are fine; say so in one line of the PR.
- **Suppress narrowly.** A `# type: ignore`, `# pyright: ignore` or `# noqa` names the specific code it silences. Lint rejects a blanket `# type: ignore` or `# noqa`; a blanket `# pyright: ignore` is not caught, so review for it. Fix the type rather than suppress it where you can.
- **Wording.** "Data source" in the API and web interface; "dataset template" for the YAML files that describe one.

## Known pitfalls

Silent failures found the hard way. Each passed its tests at first.

- **Timestamps in openEO process arguments lose their offset.** Every argument that parses as an ISO 8601 timestamp is made timezone-naive before the process sees it (`_normalise_temporal_arguments` in `openeo/execution.py`). Never pass one through as an opaque token and compare it as a string; compare parsed instants.
- **`TestClient(root_path=...)` does not prefix the path** the way uvicorn does, so a test of prefix handling passes with the fix removed. Use the `mounted_client_factory` fixture in `tests/conftest.py`.
- **topozarr above 0.1.8 shards pyramids along time by default.** On an appending store every daily sync then rewrites the partial shard, about 25 times more data written. Set the shard size explicitly before upgrading.
- **Tests that assert on a submitted request** (a process graph, a job body) do not show that it runs. Execute it once.

For adding a data source, follow the checklist in [docs/adding_custom_datasets.md](docs/adding_custom_datasets.md#contributing-a-data-source).

## Dataset templates

Each YAML file in `plugins/rasters/` holds raster dataset templates, one per data source. The `ingestion` block names the streaming plugin class, a `BaseDatasetPlugin` subclass implementing `periods()` and `fetch_period()`:

```yaml
ingestion:
  plugin: open_climate_service.plugins.rasters.worldpop.WorldPopYearlyPlugin
  params: {} # passed to the plugin
```

`write_to_icechunk_store` in `data_manager/services/downloader.py` builds a multiscale Zarr pyramid when the grid has more than 1024×1024 pixels; otherwise it writes a flat store. Ingestion and sync write one time step per chunk; see `docs/zarr_and_geozarr.md`.

## Planned work

Tracked in Jira, project [CLIM](https://dhis2.atlassian.net/jira/software/c/projects/CLIM/issues), component "Climate Service", grouped under one epic per area (data ingestion, processing and workflows, openEO, catalogue and metadata, GeoZarr and Icechunk storage, vector data, DHIS2 integration, and so on). GitHub issues are not used.

When writing a Jira epic or issue, follow the [write-issue guide](https://github.com/dhis2-chap/claude-plugins/blob/main/plugins/write-issue/skills/write-issue/SKILL.md). Filing details for this repo:

- Project `CLIM`, component `Climate Service`.
- Parent: the area epic, named "OCS: <area>" (for example CLIM-802 "OCS: Processing and workflows").
- Label: `ch-obj-open-climate-service` on every issue, as a placeholder until the OCS objectives are in Jira.
- Sprint: the current sprint for work being done now; no sprint for work filed for later.
- Assignee: none.

To install the guide as a skill in Claude Code, Codex, Copilot or another agent: `npx skills add dhis2-chap/claude-plugins --skill write-issue`.

## Commit conventions

- **Conventional Commits** for all git activity — commit messages, branch names, and PR titles.
  - Format: `<type>(<scope>)?: <description>` (e.g. `feat(ci): add docker publish workflow`, `fix(main): correct db path creation`).
  - Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`, `style`, `revert`.
  - Branch names: `<type>/<short-description>` (e.g. `feat/makefile-and-ci`, `fix/sqlite-path`).
- **No attribution.** Do not add `Co-Authored-By: Claude ...`, "Generated with Claude Code", or any similar attribution to commits, PRs, or files.
- **No emojis** anywhere — not in commits, code, comments, or documentation.
