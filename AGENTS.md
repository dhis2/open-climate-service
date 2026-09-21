# Open Climate Service

Agent context for the `open-climate-service` repository. Provider-agnostic — intended to be readable by any AI coding assistant.

## Project overview

The Open Climate Service is a FastAPI-based REST API that downloads, processes, and serves climate and Earth Observation data as GeoZarr stores.

Key concepts:

- **Dataset templates** — YAML files in `plugins/datasets/` describing a data source (variable, period type, download function). These are blueprints.
- **Artifacts / managed datasets** — ingested instances of a template for a specific spatial extent and time range. Exposed under `/datasets` and `/zarr/{dataset_id}`.
- **Extent** — a single named spatial bounding box configured at instance setup time (`id`, `bbox`, optional `country_code`). Exposed at `GET /extent`.
- **GeoZarr stores** — datasets are stored as chunked Zarr v3 archives with GeoZarr spatial attributes. Flat stores for small extents; multiscale pyramids for large ones. Served chunk-by-chunk over HTTP with no specialised server middleware.
- **Feature collections** — vector datasets (org unit polygons, facility points) stored as GeoParquet under `<data_dir>/features`, tracked by the same `ArtifactRecord` as rasters and discriminated by `itemType: "feature"`. A record is what makes a collection exist: the listing reads records, never the filesystem, so the store directory is not an inbox.

## Repository layout

```
open_climate_service/
  data_manager/     # download and zarr build (downloader.py)
  data_accessor/    # open zarr / netcdf for read (accessor.py)
  data_registry/    # dataset template YAML loading
  ingestions/       # artifact lifecycle: create, list, sync, publish
  features/         # feature collection store, GeoParquet reader, GET /features
  publications/     # STAC publication metadata
  extents/          # spatial extent config
  shared/           # dhis2 adapter, time utils
  main.py           # FastAPI app, CORS middleware, route registration
data/datasets/      # dataset template YAMLs (chirps3.yaml, worldpop.yaml, …)
tests/
docs/
```

## Development

```bash
make run      # start uvicorn with --reload
make lint     # ruff check + ruff format + mypy + pyright
make test     # pytest
make start    # docker compose up --build
```

The `.env` file is required for `make run`. Copy `.env.example` if it exists.

## Dataset templates

Each YAML in `plugins/datasets/` defines a dataset template. The `ingestion` block controls download and zarr build behaviour:

```yaml
ingestion:
  function: dhis2eo.data.worldpop.pop_total.yearly.download
  default_params: {} # passed to the download function
```

`build_dataset_zarr` in `data_manager/downloader.py` builds a multiscale Zarr pyramid when the spatial dimensions exceed 2048×2048 pixels; otherwise it writes a flat chunked zarr with chunk sizes derived from the dataset's temporal resolution.

The ingestion interface is being redesigned as a plugin protocol (see GitHub issue #64) — the `ingestion.function` convention will be replaced by a three-method async plugin (`probe`, `periods`, `fetch_period`).

## Active design work

- **#64** — streaming ingest via Icechunk; per-period writes; no intermediate files
- **#111** — async job execution; OGC API Processes; progress reporting
- **#137** — spatial aggregation to DHIS2 org units; multi-dataset grid alignment

## Commit conventions

- **Conventional Commits** for all git activity — commit messages, branch names, and PR titles.
  - Format: `<type>(<scope>)?: <description>` (e.g. `feat(ci): add docker publish workflow`, `fix(main): correct db path creation`).
  - Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`, `style`, `revert`.
  - Branch names: `<type>/<short-description>` (e.g. `feat/makefile-and-ci`, `fix/sqlite-path`).
- **No attribution.** Do not add `Co-Authored-By: Claude ...`, "Generated with Claude Code", or any similar attribution to commits, PRs, or files.
- **No emojis** anywhere — not in commits, code, comments, or documentation.
