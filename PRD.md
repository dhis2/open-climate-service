# Open Geo Service — Product Requirements Document (PRD)

> **Internal working document.** This PRD captures the original MVP/hackathon scope and is kept for historical context. The published, maintained project overview is [`docs/project_description.md`](docs/project_description.md) (the "Project overview" page in the documentation site).

## 1) Overview
Open Geo Service is a geospatial data integration platform that enables users to discover, fetch, process, harmonize, and load earth observation and related datasets into DHIS2 and the CHAP Modelling Platform.

DHIS2 Maps app and DHIS2 Climate app are core downstream consumers of this API, with `geoservice` replacing functionality currently sourced from Google Earth Engine.

This PRD defines the MVP scope for the hackathon and a near-term path toward a production-ready platform.

## 2) Problem Statement
Current EO data workflows are fragmented across tools and scripts, making them hard to repeat, schedule, and maintain. Teams need a unified API that supports:
- Dataset discovery
- On-demand processing
- Aggregation to DHIS2 org units
- Scheduled ingestion
- Optional custom pre/post-processing

## 3) Goals
- Provide a unified API for EO data retrieval and processing.
- Enable no-code workflows comparable to DHIS2 Climate Tools, using existing DHIS2 EO libraries.
- Support map-serving and analytics use cases without Google Earth Engine lock-in.
- Provide stable API capabilities consumed by both DHIS2 Maps app and DHIS2 Climate app.
- Deliver an MVP that proves core end-to-end flows during the hackathon.

## 4) Non-Goals (MVP)
- Full enterprise IAM/SSO implementation.
- Comprehensive billing/tenancy model.
- Complete OGC suite compliance beyond selected process endpoints.
- Replacing all existing tooling in one release.

## 5) Users and Key User Stories
### Primary users
- DHIS2 implementers and analysts
- Climate/health data teams
- GIS/data engineering teams

### User stories
- User A: Import daily temperature and precipitation into DHIS2 on a user-defined schedule, aggregated to org units.
- User B: Import annual population data, automatically aggregated to org units.
- User C: Visualize high-resolution population data in DHIS2 Maps with meaningful density-based styles.
- User D: Preview climate data for a selected org unit before committing import.
- User E: Add custom pre/post-processing (e.g., consecutive rainy days) before import.

## 6) Functional Requirements
### FR1 — Unified data/process API
- Expose a single API surface for dataset listing, request submission, processing, and retrieval.
- Use `dhis2eo` as the default EO/climate processing integration library.

### FR2 — Core process execution
- Support process discovery and execution using an OGC API - Processes-aligned model.
- Minimum endpoints:
  - `GET /processes`
  - `GET /processes/{process-id}`
  - `POST /processes/{process-id}/execution`

### FR3 — Raster and tiling capabilities
- Support on-the-fly image tiling and styling from raster sources (COGs/Zarr where feasible).
- Enable:
  - Value retrieval at a location
  - Aggregation over org unit geometries
- Ensure parity for the core map/preview capabilities currently implemented with Google Earth Engine for DHIS2 Maps and Climate clients.

### FR4 — Dataset discovery endpoint
- Provide endpoint(s) to list available datasets for DHIS2 Maps and related clients.
- Prefer STAC-compatible metadata where possible.

### FR8 — Consumer app compatibility
- API contracts and outputs must support DHIS2 Maps app (https://github.com/dhis2/maps-app) and DHIS2 Climate app (https://github.com/dhis2/climate-app).
- Changes to Open Geo Service contracts should be assessed for impact on both apps before release.

### FR5 — Scheduling and orchestration
- Allow fixed-interval runs for recurring ingestion.
- Support simple pipeline orchestration with optional pre/post-processing step hooks.

### FR6 — DHIS2 integration
- Push processed/aggregated outputs to DHIS2 database through the DHIS2 Web API.
- Use `dhis2-python-client` as the default DHIS2 Web API integration library.

### FR9 — Caching and configuration storage
- Support file-based caching of downloaded and/or intermediate processed data when needed for performance, resiliency, or replay.
- Support use of DHIS2 Data Store for storing Open Geo Service configuration metadata where appropriate.

### FR7 — Upstream library evolution
- Treat `dhis2-python-client` and `dhis2eo` as strategic dependencies that can be changed to fulfill `geoservice` requirements.
- When required functionality is missing, define and implement upstream changes rather than introducing long-term local forks in `geoservice`.

## 7) Non-Functional Requirements
- Reliability: Handle simultaneous and long-running requests.
- Extensibility: Process catalog must support adding new EO pipelines with minimal API changes.
- Interoperability: Align with open geospatial standards where practical (OGC API, STAC).
- Portability: Containerized deployment via Docker.
- Governance: Align with FAIR principles and Digital Public Good expectations.
- Dependency sustainability: Upstream contributions to `dhis2-python-client` and `dhis2eo` should preserve backward compatibility where feasible.

## 8) Proposed Technical Direction
- API framework: FastAPI
- Process API: native openEO `/processes` and `/jobs` (OGC API - Processes-aligned)
- Raster/tile API: TiTiler (`/cog/*`, `/stac/*`)
- DHIS2 API integration library: `dhis2-python-client` (https://github.com/dhis2/dhis2-python-client)
- EO/climate processing library: `dhis2eo` (https://github.com/dhis2/dhis2eo)
- Caching approach: file-based cache for selected source/intermediate artifacts
- Configuration persistence: DHIS2 Data Store can be used for Open Geo Service configuration state
- Orchestration/scheduling: Apache Airflow (or Prefect for evaluation)
- Data formats: Cloud Optimized GeoTIFF (COG), Zarr
- Deployment: Docker-based services

## 9) MVP Scope (Hackathon)
### In scope
- Draft unified API interface
- Process catalog + at least one executable process flow
- Climate variable import path (temperature/precipitation) aggregated to org units
- Preview of data for org unit
- Basic scheduled run capability (or simulated scheduler integration)

### Out of scope
- Full production-grade authz/authn stack
- Broad catalog of EO sources
- Advanced UI beyond thin client integration points

## 10) Success Metrics
- Time-to-first-ingestion: User can configure and run first import quickly.
- Pipeline success rate for scheduled runs.
- Latency for preview and aggregation requests within acceptable operational bounds.
- Number of reusable process definitions in the process catalog.

## 11) Google Earth Engine Parity Checklist
- [ ] On-the-fly raster tiling available via Open Geo Service endpoints for Maps/Climate use cases.
- [ ] Styling controls exposed for map visualization workflows used by DHIS2 Maps app.
- [ ] Value retrieval at a single location (point query) supported for climate/map inspection.
- [ ] Aggregation to DHIS2 org unit geometries supported for required metrics.
- [ ] Dataset discovery endpoint provides metadata needed by Maps app and Climate app.
- [ ] Preview workflow for org-unit-focused inspection available before import.
- [ ] Scheduled ingestion workflow supports recurring climate/population imports.
- [ ] Import-ready output format validated for DHIS2 ingestion paths.
- [ ] Core Maps app workflows run without Google Earth Engine dependency.
- [ ] Core Climate app workflows run without Google Earth Engine dependency.

### Parity verification criteria
- Functional parity: Equivalent user-visible outcome for each checklist item in Maps and Climate integrations.
- Data parity: Results are within agreed tolerance for value/aggregation comparisons.
- Operational parity: Throughput and latency are acceptable for expected production usage.

## 12) Risks and Open Questions
- Should TiTiler and Open Geo Service run in same container or separate services?
- Should openEO be explored as a strategic integration path?
- What level of OGC API - Processes compliance is required for MVP vs production?

## 13) Upstream Contribution Strategy
- Preferred path: implement missing cross-project capabilities in `dhis2-python-client` and/or `dhis2eo`, then consume released versions in `geoservice`.
- Avoid permanent private forks; use short-lived patches only when release timing requires temporary workarounds.
- Track upstream gaps as explicit requirements with owner, milestone, and compatibility impact.

## 14) Milestones
1. Domain and technology landscape assessment
2. Unified API draft finalized
3. MVP process flow implemented
4. End-to-end demo: dataset → process → aggregated output → DHIS2 import path

## 15) Acceptance Criteria (MVP)
- A user can discover available process(es) and execute one through API.
- A climate dataset can be previewed and aggregated to org units.
- A recurring run can be configured and triggered on schedule.
- Output is available in a DHIS2-compatible import format and at least one ingestion path is demonstrated.
- Core map/preview and aggregation workflows required by Maps app and Climate app are available via `geoservice` without dependence on Google Earth Engine.
