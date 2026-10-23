# Built-in datasets

The Open Climate Service ships with built-in data sources for commonly requested climate and population data. Each one describes an upstream dataset and the rules for downloading, transforming, and syncing it. They are available in every instance without any additional configuration.

To ingest a built-in dataset for your configured extent, see the [API reference](managed_data_api_guide.md). To add datasets beyond these, see [Adding custom datasets](adding_custom_datasets.md).

---

## ERA5-Land — temperature and precipitation

ERA5-Land provides temperature and precipitation at hourly, daily, and monthly resolution. Nine data sources are available covering both variables and all resolutions, with options for UTC or local-timezone daily aggregation.

See **[ERA5-Land datasets](era5_land_datasets.md)** for the full reference, including dataset IDs, coverage, lag times, and guidance on choosing the right dataset for your use case.

---

## CHIRPS v3 — daily precipitation

| Property               | Value                                              |
| ---------------------- | -------------------------------------------------- |
| **Dataset ID**         | `chirps3_precipitation_daily`                      |
| **Variable**           | `precip`                                           |
| **Units**              | mm                                                 |
| **Period**             | Daily                                              |
| **Spatial coverage**   | Global land, 50°S–50°N                             |
| **Spatial resolution** | ~5 km                                              |
| **Record start**       | 1981-01-01                                         |
| **Source**             | [CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3) |

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) v3 is a quasi-global daily precipitation dataset merging satellite thermal infrared imagery with station observations. It is widely used for drought monitoring, food security analysis, and WASH planning in low- and middle-income countries.

**Sync behaviour** — new data is ingested incrementally as it becomes available. The final product is published a month at a time, a month or more after it, so the current month is never present. The API probes the CHIRPS server for the newest complete month before each sync. For the current month, use the preliminary dataset below.

**Transforms** — none applied; data is stored as received in mm.

---

## CHIRPS v3 — preliminary daily precipitation

| Property               | Value                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **Dataset ID**         | `chirps3_precipitation_daily_prelim`                                               |
| **Variable**           | `precip`                                                                           |
| **Units**              | mm/d                                                                               |
| **Period**             | Daily                                                                              |
| **Spatial coverage**   | Global land, 50°S–50°N                                                             |
| **Spatial resolution** | ~5 km                                                                              |
| **Record start**       | 2025-01-01                                                                         |
| **Source**             | [CHIRPS v3 preliminary](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/) |

Daily rainfall a few days after it falls, including the current month, for in-season monitoring. It is satellite-only, without the station blending of the final product, and **provisional**: the final daily dataset replaces these values a month or so later.

It is a separate dataset from `chirps3_precipitation_daily` on purpose. A store holding both would rewrite its own history when the final values arrive, with nothing to say which values were provisional. A client wanting a continuous series uses the final dataset up to its last month and the preliminary one after it.

**Sync behaviour** — the newest published day is found by probing the CHIRPS server day by day from today, up to 21 days back. The lag is typically 4–5 days.

**Preliminary first, final later, in DHIS2** — push both datasets to the same data element. The DHIS2 export imports with `CREATE_AND_UPDATE`, so a value for an org unit and period that already exists is overwritten:

1. Daily, push `chirps3_precipitation_daily_prelim` for the days after the final dataset's last date, so a late run never puts provisional values back over final ones.
2. When the final product publishes a month, push `chirps3_precipitation_daily` for that month; its values replace the preliminary ones.

DHIS2 then holds the best value available for each day. It does not show which values are still provisional; to mark them, push the preliminary dataset with its own attribute option combo or data element instead, which gives up the overwrite.

---

## CHIRPS v3 — monthly precipitation

| Property               | Value                                              |
| ---------------------- | -------------------------------------------------- |
| **Dataset ID**         | `chirps3_precipitation_monthly`                    |
| **Variable**           | `precip`                                           |
| **Units**              | mm/d                                               |
| **Period**             | Monthly                                            |
| **Spatial coverage**   | Global land, 50°S–50°N                             |
| **Spatial resolution** | ~5 km                                              |
| **Record start**       | 1981-01                                            |
| **Source**             | [CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3) |

The monthly CHIRPS product, published as one global raster per calendar month. Prefer it over aggregating the daily dataset when you only need monthly values: it is roughly 30× less data to ingest, and drought indices such as SPI and SPEI are conventionally computed monthly.

**Sync behaviour** — as for the daily dataset, the latest published month is discovered by probing the CHIRPS server rather than assuming a fixed lag. The monthly final product typically trails the calendar by a month or two.

**Transforms** — the source raster is a monthly **total** in mm; it is divided by the number of days in the month and stored as a mean daily rate (`mm/d`). Unlike the daily dataset, where mm and mm/day are the same number, this is a real conversion.

That choice keeps every monthly precipitation dataset on the same units, which matters because `chirps3_precipitation_monthly_normal_1991_2020` is also `mm/d` and is the natural partner for a monthly anomaly — storing the raw total under the same label would make that comparison wrong by a factor of about 30. It is also the form xclim's drought indices expect.

---

## WorldPop Global2 — total population (yearly)

| Property               | Value                                                                |
| ---------------------- | -------------------------------------------------------------------- |
| **Dataset ID**         | `worldpop_population_global2_100m`                                   |
| **Variable**           | `pop_total`                                                          |
| **Units**              | people                                                               |
| **Period**             | Yearly                                                               |
| **Spatial coverage**   | Global                                                               |
| **Spatial resolution** | ~100 m                                                               |
| **Record start**       | 2015                                                                 |
| **Record end**         | 2030                                                                 |
| **Source**             | [WorldPop Global2](https://hub.worldpop.org/project/categories?id=3) |

WorldPop Global2 provides gridded population estimates and projections at 100 m resolution. Each raster year represents estimated residential population counts. Years up to and including the present are backward-modelled estimates; years beyond the present are forward projections.

**Sync behaviour** — population data is released year by year, not as a continuous stream. The API uses a `release`-kind sync that checks each calendar year separately. Future years (projections) are also requestable, since the underlying data covers through 2030.

**Transforms** — none applied; values are stored as received (population counts per pixel).

---

## ESA CCI / C3S — land cover class (yearly)

| Property               | Value                                                                 |
| ---------------------- | --------------------------------------------------------------------- |
| **Dataset ID**         | `esacci_landcover_class_yearly`                                       |
| **Variable**           | `lccs_class`                                                          |
| **Units**              | dimensionless class codes                                             |
| **Period**             | Yearly                                                                |
| **Spatial coverage**   | Global                                                                |
| **Spatial resolution** | 300 m                                                                 |
| **Record start**       | 1992                                                                  |
| **Record end**         | Currently 2022; available years are discovered from the CDS catalogue |
| **Source**             | [ESA CCI / C3S land cover](https://cds.climate.copernicus.eu/datasets/satellite-land-cover) |

Each yearly raster contains the ESA CCI / C3S land-cover map in its 37 Land Cover Classification
System (LCCS) classes. This is a categorical dataset which uses categorical **mode**
resampling when building overviews rather than averaging the class codes. The no-data code (0) is
stored as missing data.

**Setup** — register for a Copernicus Climate Data Store account, configure the CDS credentials,
and accept the dataset licence in the browser before ingesting. See the
[CDS setup instructions](era5_land_datasets.md#copernicus-climate-data-store-monthly-datasets).

**Availability and sync** — the plugin discovers the allowed years from the CDS catalogue when
listing available periods, so newly published years do not require a hardcoded plugin year limit.
The CDS publishes version 2.0.7cds for 1992–2015 and version 2.1.1 from 2016 onward. Sync tracks
the declared release; new catalogue years can be ingested when requested.

---

## Copernicus — elevation (static)

| Property               | Value                                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| **Dataset ID**         | `copdem30_elevation_static`                                                                          |
| **Variable**           | `elevation`                                                                                          |
| **Units**              | meters                                                                                               |
| **Period**             | None: no time axis                                                                                   |
| **Spatial coverage**   | Global                                                                                               |
| **Spatial resolution** | ~30 m                                                                                                |
| **Source**             | [Copernicus / DestinE](https://earthdatahub.destine.eu/collections/copernicus-dem/datasets/GLO-30)   |

Copernicus DEM Global 30m, recording elevation height above sea level. 

The dataset is static: it has no time axis, so it is ingested whole and shown without a time slider.

It needs an [Earth Data Hub](https://earthdatahub.destine.eu) account. Add your key to `~/.netrc`
as for [ERA5-Land](era5_land_datasets.md), with the host `data.earthdatahub.destine.eu`:

```
machine data.earthdatahub.destine.eu
  password <your-standard-api-key>
```

---

## Overture Maps — administrative divisions (feature collection)

| Property             | Value                                                                   |
| -------------------- | ----------------------------------------------------------------------- |
| **Collection ID**    | `overture_divisions`                                                    |
| **Item type**        | Feature collection (GeoParquet), not a raster                           |
| **Identity**         | `id_property: id` — Overture's own division id                          |
| **Spatial coverage** | Global; extracted for this instance's configured extent                 |
| **Version**          | The pinned Overture release, e.g. `2026-09-23.0`                        |
| **Licence**          | ODbL-1.0 — attribution **and** share-alike                              |
| **Source**           | [Overture Maps divisions](https://docs.overturemaps.org/guides/divisions/) |

Registered collections are served under `GET /features` and loaded inside a process graph by id
with `load_features`.

**Pick one administrative level.** A bounding-box window returns every level that overlaps it, so
the shipped template filters to a single `subtype`, coarsest first: `country`, `dependency`,
`region`, `county`, `localadmin`, `locality`, `macrohood`, `neighborhood`.

**A window crosses borders.** The bbox is a rectangle, not a country outline, so a neighbouring
country's divisions come back too. Add `country` to `filters` when that matters.

```yaml
- id: overture_divisions
  name: Administrative divisions (Overture Maps)
  license: ODbL-1.0
  attribution: © OpenStreetMap contributors, © Overture Maps Foundation
  id_property: id
  provider: overture
  params:
    release: 2026-09-23.0
    theme: divisions
    filters: { subtype: county, country: SL }
```

**Refreshing** — Overture publishes monthly and the release id *is* the recorded version, so
comparing what an instance holds against the latest release answers "is this stale" without
reading the data. Bumping `release` selects which release the **next** provider run extracts; it
does not itself trigger one. A feature collection is not on the raster sync path — `POST
/sync/{id}` refuses it — so refreshing means re-running the provider: **Fetch** on its template
page, or `POST /features/{id}/refresh` (add `Prefer: respond-async` to run it as a background
job). The release is pinned
rather than resolved to `latest`, so an upgrade is a visible configuration change and an extract
is reproducible.

**Other themes** — the provider is generic: `theme` selects the family (`divisions`, `buildings`,
`places`, `transportation`, `addresses`, `base`) and resolves to the type worth extracting, or
name `type` outright. Only `divisions` is shipped as a template.

**Licence obligation** — ODbL because divisions incorporate OpenStreetMap. The `license` and
`attribution` fields are the obligation, not decoration: publishing an extract without surfacing
them is a breach, and share-alike attaches to a publicly served derived database.


## DHIS2 — organisation units (feature collection)

The built-in `dhis2` provider fetches organisation units with their boundaries from a DHIS2
instance and stores them as a feature collection (GeoParquet). Each feature's id is the DHIS2 UID
(`id_property: id`), so values aggregated to these boundaries can go straight back into DHIS2 as
`orgUnit`. It needs the `dhis2` extra (`open-climate-service[dhis2]`) and a named connection. No
template is shipped, since the DHIS2 instance is yours: you declare one, as below.

**1. Configure the connection** in `climate-service.yaml`, with the token in the environment (see
[named connections](importing_to_dhis2.md#named-connections-for-server-side-plugins)):

```yaml
dhis2_connections:
  - id: national-hmis
    url: https://hmis.example.org/dhis
    token_env: DHIS2_IMPORT_TOKEN
```

**2. Declare a template** in the `vectors/` folder of your `plugins_dir`. Templates there are
read only when `plugins_dir` is set in `climate-service.yaml`, for example `plugins_dir: ./plugins`;
then add `plugins/vectors/districts.yaml`:

```yaml
- id: districts
  name: Districts (DHIS2 level 2)
  description: >-
    The districts of the national HMIS hierarchy, fetched with their boundaries from DHIS2.
    Each feature's id is the organisation unit UID, so values aggregated to a district go
    back to DHIS2 as its orgUnit.
  providers:
    - name: National HMIS (DHIS2)
      url: https://hmis.example.org/dhis
      roles: [producer, host]
  id_property: id
  provider: dhis2
  params:
    connection: national-hmis
    level: 2
```

`level` and `parent` select the organisation units as DHIS2's own queries do: `level: 2` for every
unit at level 2, `parent: <UID>` for the units under one parent, or both. With neither, DHIS2
returns level 1. `description` and the first `providers` entry are what the Datasets page shows
under the collection's name; both are optional.

**3. Fetch it** with **Fetch** on the template's page, or `POST /features/districts/refresh` (add
`Prefer: respond-async` to run it as a background job). Fetch again when the hierarchy changes in
DHIS2.

**What it checks.** DHIS2 leaves out organisation units that have no geometry. The provider counts
them with a second request and logs how many were skipped, naming a few, so a gap is visible
rather than silent; if none of the selected units has a geometry, the fetch fails. A unit with no
UID, or the same UID twice, is refused with its name. The other DHIS2 properties (`code`, `name`,
`level`, `parent`, `parentGraph`, `groups`) are kept on each feature.

---

## Temporal resampling

Any ingested dataset can be resampled to a coarser temporal resolution (e.g. hourly → daily, daily → monthly) using the standard openEO `aggregate_temporal_period` process in a process graph. See [Processes](processes.md) for an example.
