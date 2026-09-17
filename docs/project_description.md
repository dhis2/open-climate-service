# Open Climate Service

## Climate Data & Earth Observation Integration Platform

_Project Description · 2026_

---

## 1. Background

Climate change and extreme weather events pose severe threats to low- and middle-income countries, but data fragmentation makes it difficult to analyze these threats and forecast future impacts in an accurate and timely way. Climate and earth observation data is distributed across dozens of providers — each with different APIs, data formats, and access mechanisms. Integrating this fragmented landscape into operational systems requires specialised expertise for every data source, a barrier that limits its use across sectors in low- and middle-income countries.

This project develops the Open Climate Service — an open-source, standards-based, and decentralised integration platform that unifies this fragmented space behind a single, consistent interface. By abstracting data access across heterogeneous sources and harmonising outputs into a common format, the platform enables organisations to automatically ingest, process, and analyse global, national, and local climate and environmental data without requiring specialised expertise for each data provider. Although developed in close alignment with DHIS2 to support seamless integration with national health information systems, the platform is designed to operate independently of any specific software platform, and can serve as shared infrastructure for any sector that requires systematic access to harmonised climate and environmental data — including health, agriculture, land management, and forest monitoring.

A key design principle is data sovereignty: the platform is deployable on national or regional infrastructure without dependency on proprietary services, ensuring that countries retain full control over their data. Outputs are stored and served using open geospatial standards, making the platform an open foundation rather than a closed system. Local data providers and developers can connect their own data sources, build custom analytical workflows, and extend the platform to address specific needs. This openness and interoperability is a deliberate design choice — the goal is not a single monolithic tool, but shared infrastructure that countries and communities can adapt and innovate upon.

The platform is being developed in close collaboration with HISP groups in the countries themselves — the people who understand local data landscapes, institutional arrangements, and technical constraints. The initial dataset catalogue will include open data on climate, earth observation, and population. By making this data systematically accessible through open, sovereign infrastructure, the platform lowers the barrier for countries to conduct integrated analysis, run predictive models, and establish early warning systems.

---

## 2. Overview

The Open Climate Service is a data integration platform that enables earth observation (EO) and climate data from multiple upstream sources to be downloaded, processed, harmonised, and loaded into DHIS2 and the CHAP Modelling Platform.

The platform is built as a Python-based REST API (FastAPI) and exposes native REST endpoints together with a STAC catalogue for discovery. Data is stored in cloud-native GeoZarr format and can be consumed by the DHIS2 Climate App, the DHIS2 Maps App, DHIS2 Climate Tools, the CHAP Modelling Platform, and third-party tools such as QGIS.

The Open Climate Service is envisioned as the shared data infrastructure layer for the DHIS2 climate and health ecosystem — a single, well-defined source of spatiotemporal raster data that any DHIS2 application or external tool can build on.

### 2.1 Relationship to existing DHIS2 climate work

The Open Climate Service supplements and extends the existing DHIS2 climate data integration work documented at dhis2.org/climate/climate-data/. The platform wraps climate and earth observation workflows in a standardised API so that data access, processing, and publication can be configured and run without writing code.

### 2.2 Scope of this document

This document describes the project vision, design constraints, user stories, functional requirements, technical architecture, and data pipeline approach. It is intended for technical contributors, DHIS2 country implementers, and stakeholders evaluating the Open Climate Service for deployment.

---

## 3. Vision and goals

The Open Climate Service aims to:

- Provide a unified API through which EO and climate data can be requested, downloaded, processed, and uploaded to DHIS2 — with all complexity handled behind the scenes.
- Serve as a no-code alternative to DHIS2 Climate Tools for standard data integration workflows, built on the same underlying libraries.
- Allow DHIS2 Climate/Maps app and CHAP to act as frontends consuming the Open Climate Service.
- Support custom orchestration — users can build pipelines with pre- and post-processing steps.
- Work independently of a DHIS2 instance.
- Follow the requirements for being a Digital Public Good (DPG) and adhere to the FAIR principles (Findable, Accessible, Interoperable, Reusable).

---

## 4. User stories

| ID   | Actor        | Goal                                                                                                                                                   |
| ---- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| US-A | Data manager | Import daily temperature and precipitation data into DHIS2 at a user-defined scheduled interval (e.g. nightly), automatically aggregated to org units. |
| US-B | Data manager | Import population data for the current year, automatically aggregated to org units.                                                                    |
| US-C | Analyst      | Visualise high-resolution population data on DHIS2 Maps with styles adapted to the population density of the country.                                  |
| US-D | Analyst      | Preview climate data for an org unit of interest before importing it to DHIS2.                                                                         |
| US-E | Data manager | Add a custom pre- or post-processing step (e.g. calculate consecutive rainy days) before importing the result to DHIS2.                                |

---

## 5. Design constraints

The following constraints apply to the first version of the Open Climate Service. They represent deliberate architectural decisions and are open for discussion as the platform matures.

### 5.1 Single spatial extent

Each Open Climate Service instance is configured with one or more named extents, defined at setup time. Each extent has a required `id` and `bbox`, and an optional `org_unit_id` for linking to a DHIS2 org unit.

Extents are not expected to change after setup. For the first version, only a single extent is supported. Larger countries may configure a sub-national extent (e.g. a district) to limit initial download volume. The `extent_id` is passed as a parameter to ingestion alongside the `dataset_id` — ingestion is not tied to a DHIS2 instance.

### 5.2 No temporal gaps

Downloaded datasets must not contain temporal gaps, unless gaps exist in the original upstream data source. All subsequent scheduled updates look at the last period with data and import from there until today, ensuring continuity. The `/sync` endpoint validates temporal continuity before appending new time steps.

### 5.3 One period type per dataset

Each dataset has a single period type (daily, weekly, monthly, yearly, etc.). The period type is included in the dataset ID (e.g. `chirps3_precipitation_daily_sle`). It is possible to construct derived datasets with a different period type from an existing dataset (e.g. daily → weekly aggregation), which will result in a separate dataset ID.

### 5.4 One artifact per dataset

Each dataset ID maps to exactly one output artifact in the form of a GeoZarr store. New time steps are appended to the existing store on sync rather than creating parallel stores.

### 5.5 DHIS2-independent operation

Core parts of the Open Climate Service must function without a connected DHIS2 instance. Spatial extent is defined via instance configuration rather than a DHIS2 org unit query. Aggregation accepts GeoJSON features from any source and outputs CSV or JSON as well as DHIS2 data values.

### 5.6 Dataset templates and published datasets

Internally, the Open Climate Service distinguishes between dataset templates and published datasets:

- **Dataset templates** — YAML definitions describing a dataset type (source, variable, period type, processing steps). These are internal and act as blueprints for ingestion.
- **Published datasets** — actualised, ingested datasets for a specific extent and time range, exposed under `/datasets` and `/stac`. These are what end users and client applications discover and consume.

This mirrors the approach used in the CHAP Modelling Platform, where generic model template YAMLs are distinguished from specific initialised instances.

---

## 6. Functional requirements

### 6.1 Data pipeline

- Allow EO data to be requested through a unified API where download, processing, and optional upload to DHIS2 happen behind the scenes.
- Each step in the pipeline is also available as a separate API endpoint with clear input and output definitions: data extraction, aggregation, and upload to DHIS2.
- Support scheduling — data can be downloaded, processed, and imported at fixed user-defined intervals.
- Support orchestration — users can compose custom data pipelines, including pre- and post-processing steps.

### 6.2 Data storage and serving

- Store all datasets as GeoZarr — cloud-native, chunked, and multiscale — without reprojecting: the stored coordinates keep the source's native CRS, tagged in the GeoZarr metadata.
- Expose datasets via a `/zarr/{dataset_id}` endpoint using HTTP range requests, enabling chunk-level access by any compatible client.
- Expose dataset discovery metadata via a `/datasets` endpoint and a STAC catalogue.

### 6.3 Visualisation

- Support direct browser rendering of Zarr data via zarr-layer (MapLibre custom layer) with GPU reprojection from the stored CRS to Spherical Mercator and client-side dynamic colour classification.
- Support point queries (single location time series) for preview before import.

### 6.4 Aggregation

- Aggregate raster data to org unit polygons (or any GeoJSON feature collection).
- Support async execution for long-running aggregation jobs (openEO batch jobs).
- Support demographic disaggregation for WorldPop data (age/sex bands as additional Zarr dimensions).

### 6.5 Integration

- Upload aggregated data values directly to DHIS2 (optional — can be skipped for standalone use).
- Accept GeoJSON features from external sources (not only DHIS2 org units).
- Output results as DHIS2 data values, CSV, or JSON.
- Provide a client library / SDK for programmatic access by DHIS2 apps and third-party tools.

### 6.6 Non-functional requirements

- Handle simultaneous and long-running requests without blocking.
- Follow FAIR principles: Findable, Accessible, Interoperable and Reusable.
- Build on existing open-source solutions — the team is small and sustainability matters.
- Support deployment via Docker for local, cloud-hosted, and sovereign country environments.
- Storage backend configurable via environment variables — no code changes required to switch between local filesystem, different cloud providers, AWS S3 (including Africa and Asia regions), and self-hosted Ceph/RGW for sovereign deployments.

---

## 7. Supported data sources

The Open Climate Service ingests data from multiple upstream Earth Observation and climate sources. Current and planned sources include:

- **CHIRPS** (Climate Hazards Group InfraRed Precipitation with Station data) — daily and pentadal precipitation.
- **ERA5 / Climate Data Store** (Copernicus CDS) — temperature, humidity, wind, and other atmospheric variables at multiple temporal resolutions.
- **WorldPop** — annual gridded population estimates, with optional age and sex disaggregation at 5-year intervals.

The dataset ID schema encodes the source, variable, period type, and spatial extent ID. The extent ID is an ISO country code or named sub-national identifier defined in the instance configuration. Examples:

- `chirps3_precipitation_daily_sle`
- `era5_temperature_daily_sle`
- `worldpop_population_global2_100m_sle`

Sub-national extents use the same schema (e.g. `chirps3_precipitation_daily_bo` for the Bo district of Sierra Leone), allowing larger countries to configure a district-level extent to limit initial download volume.

---

## 8. Technical architecture

### 8.1 API layer

The API is built on FastAPI and exposes the following endpoint groups:

| Endpoint             | Description                                                                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/ingestions`        | Trigger data download from an upstream source for the configured extent and date range. Parameters: `dataset_id`, `start`, `end`, `extent_id`. Creates or updates the corresponding Zarr store. |
| `/sync`              | Check for more recent data from the upstream source and append new time steps to the existing Zarr store. Validates temporal continuity before writing.                                         |
| `/datasets`          | List and describe available published datasets — metadata, period type, extent, last updated, and access links.                                                                                 |
| `/zarr/{dataset_id}` | Serve the GeoZarr store by returning a directory listing at the dataset path and file responses for Zarr contents beneath it.                                                                   |

### 8.2 Storage layer

All datasets are stored as GeoZarr. Key properties:

- Stored without reprojection — coordinates keep the source's native CRS, recorded in the GeoZarr `proj:code` metadata from the CRS the ingestion declares, falling back to what rioxarray detects on the data and then to `EPSG:4326`. The instance-wide `crs:` is deliberately never a fallback: using it stamped a projected code onto degree coordinates and put the store at the projection's origin. A declared CRS that contradicts the coordinates is rejected in favour of the data's own. The framework does not otherwise reconcile mixed coordinate systems.
- CF-compliant coordinate attributes and `_ARRAY_DIMENSIONS` metadata.
- Multiscale pyramid overview levels declared under the `multiscales` key in `.zattrs` — required for efficient zoom-level-aware chunk fetching by zarr-layer.
- Chunk shape tuned per dataset to balance three access patterns: time series queries, polygon aggregation, and browser tile rendering.
- Blosc/Zstd compression.

Managed datasets are written through Icechunk, which opens every store on the local filesystem — the only backend implemented today. Icechunk uses Apache Arrow's `object_store` rather than fsspec, so an alternative backend is a constructor change (`s3_storage`, `gcs_storage`) rather than a path rewrite. Making the backend a per-deployment configuration choice is tracked in CLIM-555, targeting:

| Backend                  | Notes                                                                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local filesystem         | The only backend implemented today.                                                                                                                                        |
| European S3-compatible   | Hetzner, Scaleway, IONOS, OVHcloud. GDPR-native.                                                                                                                           |
| AWS S3 (af-south-1)      | Cape Town region — lowest latency for Southern/Eastern Africa deployments.                                                                                                 |
| AWS S3 (ap-southeast-1)  | Singapore — lowest latency for Laos, Sri Lanka, and Southeast Asia deployments.                                                                                            |
| Ceph / RGW (self-hosted) | S3-compatible. For sovereign deployments requiring data to remain within national borders. Runs on university and research network infrastructure (AfricaConnect / GÉANT). |

### 8.3 Data pipeline model

The Open Climate Service follows an ETL (Extract, Transform, Load) pattern — transformation occurs on the processing server before data is loaded into DHIS2. An ELT approach (transformation in a cloud data warehouse) may be supported in a future version.

The pipeline stages are:

- **Extract** — download raw data from the upstream source for the configured extent and time range.
- **Transform** — reproject, rechunk, apply temporal aggregation (if needed), compute derived variables, and write to GeoZarr.
- **Load** — aggregate to org unit polygons and upload data values to DHIS2, or output as CSV/JSON for standalone use.

Each stage is independently accessible as an API endpoint, allowing custom pipelines to be constructed by combining steps in different sequences.

Long-running jobs (ingestion, sync, aggregation) are executed asynchronously. This ensures the API remains responsive under concurrent load. Dask is used for parallel computation within each job, processing Zarr chunks concurrently across CPU cores or threads.

### 8.4 Technology stack

| Technology                                        | Role in the Open Climate Service                                                                                                                                                              |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FastAPI (Python)                                  | Core REST API framework. Handles ingestion, sync, dataset, processing, and STAC endpoints. Each pipeline step is exposed as a separate endpoint.                                              |
| Xarray + Zarr                                     | In-memory dataset model and cloud-native chunked storage format. GeoZarr conventions applied for geospatial metadata and multiscale pyramid support.                                          |
| Dask                                              | Parallel computation within jobs — processes Zarr chunks concurrently for aggregation, reprojection, and derived variable computation. Works natively with Xarray.                            |
| rioxarray                                         | Raster operations on Xarray datasets — reprojection, clipping, resampling, and CRS management.                                                                                                |
| openeo-pg-parser-networkx + openeo-processes-dask | openEO process-graph parsing and execution — the processing and aggregation engine (temporal/spatial aggregation, derived variables, and reusable workflows).                                 |
| topozarr                                          | Builds multiscale pyramid overview levels at ingest time, for zarr-layer zoom-level-aware chunk fetching.                                                                                     |
| geozarr-toolkit                                   | Applies GeoZarr metadata conventions (multiscales, CRS, spatial) to stored datasets.                                                                                                          |
| icechunk                                          | Transactional, versioned storage engine for the managed Zarr datasets.                                                                                                                        |
| xclim                                             | Climate-index library (179 CF-compliant indicators) exposed automatically as openEO processes.                                                                                                |
| xstac                                             | Derives STAC and datacube metadata from the published Zarr datasets.                                                                                                                          |
| zarr-layer (MapLibre)                             | TypeScript library for rendering Zarr directly as a native MapLibre Custom Layer in the browser. GPU reprojection from the stored CRS to Spherical Mercator; uses multiscale levels per zoom. |
| Docker                                            | Containerised deployment. Supports local, cloud-hosted, and country sovereign deployments.                                                                                                    |
| dhis2-python-client                               | Planned DHIS2 Web API integration library for future data value push and related DHIS2 write workflows.                                                                                       |
| STAC                                              | Complementary discovery and metadata catalogue layer. Each dataset exposed as a STAC Item with temporal, spatial, and access metadata.                                                        |

---

## 9. Standards compliance

The Open Climate Service favours open, widely-supported standards over bespoke interfaces:

- **GeoZarr specification**: geospatial metadata conventions for cloud-native, chunked, multiscale Zarr stores.
- **STAC** (SpatioTemporal Asset Catalog): dataset discovery and asset linking.
- **openEO**: a standard HTTP API and process-graph model for querying and processing datasets (see [openEO](openeo.md)).
- **CF Conventions**: coordinate metadata for Xarray/Zarr datasets.
- **FAIR principles**: datasets are Findable (STAC + `/datasets`), Accessible (open HTTP range requests), Interoperable (standard formats and APIs), and Reusable (documented metadata and provenance).

> An earlier design targeted OGC API profiles (Coverages, EDR, Processes, Tiles and Maps, Collections). These were dropped in favour of the lighter-weight combination above — STAC for discovery, Zarr over HTTP for access, and openEO for processing.

---

## 10. Deployment and sovereignty

The Open Climate Service is distributed as a Docker image and can be deployed in several configurations:

- **Hosted by HISP Centre** — a centrally managed instance for demo purposes.
- **Country-hosted** — deployed within a country's own infrastructure, with local storage or the nearest available regional cloud provider (AWS af-south-1 for Africa, AWS ap-southeast-1 for Southeast Asia).
- **Sovereign** — deployed on local or research network infrastructure. Data never leaves national borders. Suitable for countries with data residency requirements.

The storage backend is configured entirely via environment variables — no code changes are required to switch between backends. This ensures the same Docker image can be deployed across all contexts.

---

## 11. Related work and repositories

| Resource                    | Link / Description                            |
| --------------------------- | --------------------------------------------- |
| Open Climate Service GitHub | https://github.com/dhis2/open-climate-service |
| DHIS2 climate data          | https://dhis2.org/climate/climate-data/       |
| CHAP Modelling Platform     | https://chap.dhis2.org/                       |
| dhis2-python-client         | https://github.com/dhis2/dhis2-python-client  |
| GeoZarr roadmap             | https://geozarr.org/roadmap.html              |
