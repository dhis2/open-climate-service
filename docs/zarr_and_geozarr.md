# Zarr and GeoZarr

This document explains why the Open Climate Service uses Zarr as its primary storage format, how Zarr stores are structured and served, and how GeoZarr root attributes enable map rendering.

---

## What is Zarr?

[Zarr](https://zarr.dev) is an open storage format for chunked, compressed N-dimensional arrays. A Zarr store is a directory tree: array metadata lives in `zarr.json` files, and the data itself is split into independent chunk files. Each chunk is compressed independently and can be read in a single HTTP request.

Zarr is designed to work natively in cloud object stores as well as on local disk — the directory layout is the same in both cases. The [Zarr v3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) is the current standard.

---

## What is GeoZarr?

[GeoZarr](https://github.com/zarr-developers/geozarr-spec) is a draft convention that adds spatial context to Zarr stores. A plain Zarr array has no concept of geography — it is just numbers in a grid. GeoZarr defines a small set of root attributes (`spatial:bbox`, `proj:code`, `zarr_conventions`) that tell a client where the grid is located on Earth and in which coordinate reference system.

---

## Why Zarr

Climate datasets are large, multi-dimensional arrays: a daily precipitation dataset covering a country at 5 km resolution for 10 years has roughly 3 600 time steps and hundreds of thousands of spatial pixels. Serving this efficiently from a REST API requires a format that supports:

- **Chunk-level random access** — a client requesting one time step should not have to read the entire file. Zarr stores data in independent, addressable chunks; a request for a single date reads only the relevant chunk.
- **HTTP-native serving** — each chunk is a separate file on disk. A standard `GET /zarr/{dataset_id}/{chunk_path}` serves it with a regular `FileResponse`. No specialised server software is needed.
- **Cloud compatibility** — the same directory layout works on local disk and cloud storage without code changes.
- **Multiscale pyramids** — GeoZarr defines a multiscales convention that allows a store to contain multiple resolution levels. Map clients request only the level that matches their current zoom, avoiding full-resolution downloads.

---

## ARCO: Analysis-Ready, Cloud-Optimized

The stores produced by the Open Climate Service are an instance of the **ARCO** pattern — a term from the climate science community describing datasets that are simultaneously ready for analysis and optimised for cloud access.

The two halves of the term map directly onto the choices described in this document:

**Analysis-ready** means a consumer can open the data and start computing without preprocessing:

- Dimension names are normalised to `(time, x, y)` regardless of the source convention.
- All datasets in an instance share a single coordinate reference system.
- Units are standardised by the transform pipeline (e.g. Kelvin → Celsius).

**Cloud-optimized** means the data can be accessed efficiently over HTTP without downloading the whole file. The Zarr and GeoZarr formats provide all the necessary properties — chunk-level access, HTTP-native serving, multiscale pyramids, and cloud compatibility.

The Open Climate Service targets the same access pattern at country scale for arbitrary source datasets.

---

## Store layout on disk

Each managed dataset has exactly one store on disk, under `{data_dir}/downloads/{dataset_id}.icechunk`. All stores use the [Icechunk](https://icechunk.io) versioned Zarr v3 format.

Inside the store, the layout is either:

- **Flat** — a single-resolution Zarr group with dimensions `(time, x, y)`
- **Pyramid** — a multi-resolution Zarr group with levels `0/`, `1/`, `2/`, … where `0/` is full resolution

The flat vs. pyramid decision is made at build time based on spatial size (see [Multiscale pyramids](#multiscale-pyramids) below).

---

## Chunk sizing

Chunks are sized to match expected access patterns. The goal is that reading one time step for the full spatial extent fits in one round-trip, and that full time series for a small area also fits in one round-trip.

Time chunk sizes are derived from the dataset's `extents.temporal.resolution` field, an ISO 8601 duration (e.g. `P1D`, `PT1H`, `P1M`). When present and valid, the duration is converted to approximate hours and mapped to a natural analysis window:

| Duration tier       | Approximate hours | Target window | Example                     |
| ------------------- | ----------------- | ------------- | --------------------------- |
| Sub-daily           | < 24 h            | ~1 week       | `PT1H` (hourly) → 168 steps |
| Daily to sub-weekly | 24 h – 168 h      | ~1 month      | `P1D` (daily) → 30 steps    |
| Weekly and coarser  | ≥ 168 h           | ~1 year       | `P1M` (monthly) → 12 steps  |

This calculation is fully data-driven: any dataset — including custom or plugin datasets — only needs to declare `extents.temporal.resolution` and the correct chunk size is computed automatically. If the field is absent or not a valid ISO 8601 duration, a warning is logged and the time chunk falls back to the dataset's `period_type`.

Spatial chunks are capped at 512 × 512 pixels — a pragmatic compromise between tile rendering (which benefits from smaller chunks) and analysis workloads (which benefit from larger ones). For small extents where the full spatial dimension is smaller than 512 pixels, the entire dimension fits in one chunk.

Dimension names are normalised to `(time, x, y)` before writing, regardless of the source naming convention (`lat`/`lon`, `latitude`/`longitude`, etc.).

---

## Multiscale pyramids

For large spatial extents, a flat zarr would require a map viewer to download the entire spatial extent at full resolution on every tile request. The platform builds a multiscale pyramid when the spatial dimensions exceed **2048 × 2048 pixels**.

Pyramid levels are computed as:

```
levels = ceil(log2(max_dim / 512))   # clamped to [2, 8]
```

Where 512 is the target tile size in pixels. Each level halves the resolution in both spatial dimensions using mean downsampling. Level `0/` is always the full resolution.

Both flat and pyramid stores are written in **Zarr v3** format using regular chunks with zstd compression.

---

## GeoZarr root attributes

A plain Zarr store has no concept of spatial coordinates. A map viewer opening it has no way to know where to position tiles on a map. GeoZarr addresses this by writing a small set of attributes into `zarr.json` at the store root:

| Attribute            | Example value                              | Purpose                                          |
| -------------------- | ------------------------------------------ | ------------------------------------------------ |
| `spatial:transform`  | `[0.05, 0, 2.95, 0, -0.05, 60.0]`          | Affine placing the grid: cell size and origin     |
| `spatial:dimensions` | `["y", "x"]`                               | Names of the row and column axes, in array order |
| `spatial:shape`      | `[60, 581]`                                | Grid size as `[height, width]`                    |
| `spatial:bbox`       | `[2.95, 57.0, 32.0, 60.0]`                 | Bounding box in the stored CRS                    |
| `proj:code`          | `EPSG:4326`                                | CRS of the stored coordinates                     |
| `zarr_conventions`   | `[{...}]`                                  | Convention declarations                           |

Two details matter to any client that reads the store directly:

- **Axis order is array order, `(y, x)`.** `spatial:dimensions` and `spatial:shape` are read
  positionally — the second-to-last entry is the row axis, the last is the column axis. Naming
  them x-first, or writing the shape as `[width, height]`, transposes the grid.
- **`spatial:transform` is what actually places the raster.** It is `[stepX, rotX, originX,
  rotY, stepY, originY]`, with the origin on the outer *edge* of the first cell (pixel
  registration) and `stepY` negative for a north-up grid. Clients that find no affine fall
  back to inferring one from the coordinate arrays, and typically assume EPSG:4326 while doing
  so — which silently mislocates a store held in projected metres.

The same affine is also written to the `spatial_ref` grid-mapping variable as a GDAL
`GeoTransform`, in GDAL's own coefficient order (`originX stepX rotX originY rotY stepY`).
That is what makes the store self-describing to GDAL/QGIS, and it is how a viewer recognises
a projected grid rather than degrees.

These attributes are computed from the **stored coordinate arrays** — not from the requested
bbox. A source may return less than was asked for (CHIRPS ends at 60°N, so a request reaching
72.5°N yields a store that stops at 60°N), and describing the request would stretch the raster
over ground the store does not cover. They are always written by the framework after any
transforms have run, so they reflect the final stored data.

`zarr_conventions` for a flat store contains the base GeoZarr convention declaration. For pyramid stores it also includes a `multiscales` entry that declares the level structure.

The pyramid metadata follows the **GeoZarr `multiscales.layout` format** (not OME-NGFF). Each level is described as a `layout` entry with an `asset` key pointing to the level path, plus `transform.scale` and `transform.translation` values for that level.

---

## CRS handling

The Open Climate Service **does not reproject** data during ingestion — each dataset is
stored as its source delivers it, so the stored coordinates keep the source's native CRS.
The framework records that CRS in the GeoZarr metadata (the `proj:code` root attribute and
the per-variable `spatial_ref` coordinate) so clients and the map viewer can position the
data; it does not transform the grid.

The recorded CRS is the one the ingestion **declares** — a dataset plugin reports it for its
source (the built-in CHIRPS, ERA5-Land and WorldPop datasets are all `EPSG:4326`), and when
none is declared the instance's configured `crs:` is used as the default. The framework does
**not** auto-detect a per-dataset CRS or reconcile differing coordinate systems, so the
declared CRS must match the coordinates the plugin emits.

```yaml
extent:
  bbox: [3.0, 57.0, 32.0, 72.5]
  crs: EPSG:32633 # optional default tag when a source CRS is not declared; defaults to EPSG:4326
```

The stored `spatial:bbox` is in that stored CRS — degrees for a geographic dataset,
eastings and northings for a projected one. STAC metadata also stores the WGS84 bounding box
alongside it, so catalogue clients that expect geographic coordinates always get one.

---

## How a client is told whether a store is pyramided

STAC has no way to distinguish a flat store from a pyramided one through the plain Zarr media
type, so the collection's `zarr` asset carries an extra parameter when — and only when — the
store really has levels:

| Store     | Advertised media type                                    |
| --------- | -------------------------------------------------------- |
| flat      | `application/vnd.zarr; version=3`                        |
| pyramided | `application/vnd.zarr; version=3; profile=multiscales`   |

Detection requires **both** the multiscales convention in the root `zarr_conventions` *and* a
non-empty `multiscales.layout` — the same two conditions a pyramid-aware renderer checks, so the
claim always matches what a renderer can use. Any failure to inspect the store degrades to the
flat type: understating is harmless, since a client then opens the store the ordinary way, while
overstating sends a renderer looking for levels that do not exist.

The `profile` parameter is a
[STAC Zarr best practices](https://github.com/radiantearth/stac-best-practices/blob/main/best-practices-zarr.md)
recommendation, not part of the official Zarr media type registration. Consumers match it as a
literal rather than parsing parameters, so the exact string matters; the constraint is recorded
next to it in `stac/media_types.py`.

## CF metadata in the catalog

Variables carry their CF semantics into `cube:variables`, named per the
[STAC CF extension](https://github.com/stac-extensions/cf):

```json
"cube:variables": {
  "tg": {
    "type": "data",
    "dimensions": ["t", "y", "x"],
    "unit": "degree_Celsius",
    "cf:standard_name": "air_temperature",
    "cf:cell_methods": "time: mean",
    "attrs": { "long_name": "...", "units": "degree_Celsius", "standard_name": "air_temperature" }
  }
}
```

The `cf:` prefix matters: the extension explicitly lists `cube:variables` among the places its
fields may be used, so a prefixed field is one a client is defined to understand, while a bare
`standard_name` at that level is not. The unprefixed spellings remain inside `attrs`, which
passes through the store's own CF attribute names verbatim. The extension is declared in
`stac_extensions` only when a `cf:` field was actually emitted.

## Opening a local store in a hosted browser viewer

Hosted tools like [zarr-viewer](https://source-cooperative.github.io/zarr-viewer/) and
[inspect.geozarr.org](https://inspect.geozarr.org) read a store directly over HTTP, which means a
public page fetching `http://localhost:9000`. Browsers gate that, and there are **two** separate
gates — only one of which the server controls.

**Server side, handled for you.** The instance answers both spellings of the opt-in Chrome has
shipped as the spec evolved — Private Network Access and its successor Local Network Access — for
an allowlist of origins that already includes the two viewers above. Add others with:

```
CLIMATE_SERVICE_ZARR_BROWSER_ORIGINS="https://inspect.geozarr.org,https://your-viewer.example"
```

**Browser side, yours to grant.** Current Chrome additionally requires *user permission* to reach
the loopback address space, and refuses by default. When that is what blocks the request, the
console says:

```
Access to fetch at 'http://localhost:9000/zarr/...' from origin 'https://...'
has been blocked by CORS policy: Permission was denied for this request
to access the `loopback` address space.
```

Note the wording — `Permission was denied`, not a missing header. No server change fixes it. The
options are to allow local network access for the site when the browser prompts (or via the icon
in the address bar), to run the viewer from `localhost` as well so both are in the same address
space, or to expose the instance on a public HTTPS origin.

The viewer's own error message is unhelpful here: it reports *"your connection looks slow or
unstable"* for what is a permission refusal, so check the browser console before suspecting the
network.

## How Zarr stores are served

The Open Climate Service provides two endpoints for accessing the same Icechunk store:

### `/zarr/{dataset_id}` — vanilla zarr clients (web maps, xarray)

```
GET /zarr/{dataset_id}/zarr.json          → root metadata with consolidated metadata injected
GET /zarr/{dataset_id}/precip/c/0/0/0     → chunk at time=0, x=0, y=0
GET /zarr/{dataset_id}/time/c/0           → time coordinate chunk
```

Metadata files (`zarr.json`) are returned as `application/json`. Chunk data is returned as `application/octet-stream`.

### `/icechunk/{dataset_id}` — Icechunk SDK clients

This endpoint serves raw Icechunk store files for native SDK access. The Icechunk SDK uses HTTP range requests to fetch only the byte ranges it needs from manifests and chunks.

```
GET /icechunk/{dataset_id}/repo              → store configuration
GET /icechunk/{dataset_id}/snapshots/...     → snapshot metadata
GET /icechunk/{dataset_id}/manifests/...     → chunk manifests (HTTP range requests)
GET /icechunk/{dataset_id}/chunks/...        → chunk data (HTTP range requests)
```

SDK usage:

```python
import icechunk, xarray as xr

repo = icechunk.Repository.open(icechunk.http_storage("https://host/icechunk/era5land_precipitation_daily"))
ds = xr.open_zarr(repo.readonly_session("main").store, zarr_format=3, consolidated=False)
```

Both endpoints are advertised as assets in the STAC collection so clients can choose the appropriate one. Use `/zarr` for web maps and standard xarray workflows; use `/icechunk` when you need versioning or SDK-level access.

---

## Fill values and NaN handling

When writing float data to Zarr, missing data is stored as IEEE `NaN`. The map viewer uses the zarr `fill_value` attribute (which defaults to `NaN` for float arrays) to render missing pixels as transparent.
