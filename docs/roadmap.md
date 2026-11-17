# Roadmap

The Open Geo Service has delivered its foundation. An instance can ingest climate
and Earth Observation data, keep it current, store it as cloud-native GeoZarr, make it
discoverable through STAC, derive new products from it through openEO processes
(including climate indices), and chain those steps into reusable workflows.

The focus now shifts from building the core platform to **integrating it with the DHIS2
ecosystem, hardening it for production, and broadening the data and analysis it offers.**

---

## Delivered

**Data ingestion and sync**

- Ingest climate and Earth Observation datasets for a configured spatial extent
- Keep datasets current — append new periods and rematerialise releases on sync
- Built-in dataset catalogue (CHIRPS, ERA5 / ERA5-Land, WorldPop)

**GeoZarr storage and discovery**

- All datasets stored as cloud-native GeoZarr, with multiscale pyramids for efficient
  browser-based rendering
- Local filesystem storage; S3-compatible and self-hosted (sovereign) object storage for
  cloud and country deployments
- STAC catalogue plus raw Zarr access — direct `xarray` / `stackstac` use without API
  knowledge

**Processing and analysis**

- openEO adopted as the process execution and chaining standard, with both synchronous
  and batch execution, backed by openeo-pg-parser-networkx and openeo-processes-dask
  (120+ standard processes)
- Custom processes, including climate indices
- Derived datasets materialised back into managed GeoZarr stores
  (e.g. change between two periods, population-weighted exposure)

**Workflows**

- Named, reusable workflows that chain processes end to end — e.g. aggregate to org-unit
  boundaries, population-weighted aggregation, and change between periods
- Workflow outputs published as managed datasets, discoverable like any other

---

## Focus areas

These are the priorities for taking the Open Geo Service from a working platform to
production-ready shared infrastructure for the DHIS2 climate and health ecosystem. They
are largely parallel rather than strictly sequential.

### DHIS2 integration

The highest-value next step is direct integration with the DHIS2 ecosystem so that climate
and EO data flows into the tools country teams already use:

- **DHIS2 Climate App** — consume Open Geo Service datasets and aggregation directly
- **DHIS2 Maps App** — render GeoZarr layers and import aggregated values
- **CHAP Modelling Platform** — supply harmonised climate inputs to models
- Push aggregated values into DHIS2 data elements against org-unit hierarchies

### Stability

- Harden ingestion, sync, and job execution against partial failures and stale jobs
- Replace the interim JSON-backed artifact store with a proper transactional store
- Broaden automated test coverage, including integration tests against running instances

### Security and authentication

- Authentication and authorisation for the API
- Per-instance access control for write and admin operations
- Secure handling of upstream provider credentials

### Performance and scalability

- Faster ingestion and processing for large extents and long time ranges
- Tune chunking, pyramids, and Dask execution for concurrent, long-running jobs
- Scale-out execution for batch jobs

### Automation and scheduling

- Scheduled sync and recurring derived-product updates (e.g. monthly normals, weekly
  anomaly refreshes)
- Event-driven cascades — a sync that updates a base dataset triggers downstream derived
  products automatically

### More datasets, processing and workflows

- Expand the built-in dataset catalogue — more sources, variables, and resolutions
- More built-in analysis: climate normals, anomalies, exposure indices, and additional
  climate indices
- More reusable workflows for common climate-and-health use cases

### Building competence and partnerships

- Build climate-data competence within HISP groups and country teams
- Grow partnerships with data providers and the wider climate / EO and DHIS2 communities

### Knowledge sharing

- Documentation, guides, and worked examples
- Training materials and presentations for implementers and analysts
