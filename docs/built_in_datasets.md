# Built-in datasets

The Open Climate Service ships with built-in dataset templates covering precipitation, temperature, and population. Each template describes an upstream data source and the rules for downloading, transforming, and syncing it. They are available in every instance without any additional configuration.

To ingest a built-in dataset for your configured extent, see the [API reference](managed_data_api_guide.md). To add datasets beyond these, see [Adding custom datasets](adding_custom_datasets.md).

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

**Sync behaviour** — new data is ingested incrementally as it becomes available. CHIRPS has a nominal publication lag of around 3–7 days, so data through yesterday is not always present. The API uses a custom availability function that checks the actual latest available date from the CHIRPS server before each sync.

**Transforms** — none applied; data is stored as received in mm.

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

## ERA5-Land — temperature and precipitation

ERA5-Land provides temperature and precipitation at hourly, daily, and monthly resolution. Nine dataset templates are available covering both variables and all resolutions, with options for UTC or local-timezone daily aggregation.

See **[ERA5-Land datasets](era5_land_datasets.md)** for the full reference, including dataset IDs, coverage, lag times, and guidance on choosing the right dataset for your use case.

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

## WorldPop Global2 — population by age and sex (yearly)

| Property               | Value                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **Dataset ID**         | `worldpop_agesex_global2_100m`                                                            |
| **Variable**           | `population` (over `sex` and `age_group` dimensions)                                      |
| **Units**              | people                                                                                    |
| **Period**             | Yearly                                                                                    |
| **Spatial coverage**   | Per-country (set `extent.country_code`)                                                   |
| **Spatial resolution** | ~100 m                                                                                    |
| **Record start**       | 2015                                                                                      |
| **Record end**         | 2030                                                                                      |
| **Source**             | [WorldPop Global2 age & sex structures](https://hub.worldpop.org/project/categories?id=8) |

Population disaggregated by sex and 5-year age band. Population is the quantity; sex and age are both disaggregation dimensions of it — so WorldPop's ~40 per-(sex, age) GeoTIFFs per country-year are combined into a **single `population` variable** over a `sex` dimension (`female`, `male`) and an ordinal `age_group` dimension (the lower bound of each band: 0, 1, 5, 10, … 90).

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

Administrative boundaries for countries with no usable DHIS2 hierarchy of their own — the polygons
an aggregation runs over when `dhis2` org units are unavailable. Registered collections are served
under `GET /features` and loaded inside a process graph by id with `load_features`.

**Pick one administrative level.** A bounding-box window returns every level that overlaps it.
Measured for the Sierra Leone extent: 201 features spanning `country` (6), `region` (18), `county`
(44), `localadmin` (15), `locality` (43), `macrohood` (24), `neighborhood` (49) and `dependency`
(2). Aggregating over that mixture is meaningless, so the shipped template filters to a single
`subtype`, coarsest first: `country`, `dependency`, `region`, `county`, `localadmin`, `locality`,
`macrohood`, `neighborhood`.

**A window crosses borders.** The bbox is a rectangle, not a country outline, so a neighbouring
country's divisions come back too — a Sierra Leone extent also returns Coyah and Conakry in
Guinea. Add `country` to `filters` when that matters.

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

**Sync behaviour** — Overture publishes monthly and the release id *is* the version, so bumping
`release` is what forces a re-extract. It is pinned rather than resolved to `latest`, so an
upgrade is a visible configuration change and an extract is reproducible.

**Other themes** — the provider is generic: `theme` selects the family (`divisions`, `buildings`,
`places`, `transportation`, `addresses`, `base`) and resolves to the type worth extracting, or
name `type` outright. Only `divisions` is shipped as a template. Large themes such as `buildings`
are a different problem — millions of features per country, needing tile generation rather than a
feature collection — and are out of scope here.

**Licence obligation** — ODbL because divisions incorporate OpenStreetMap. The `license` and
`attribution` fields are the obligation, not decoration: publishing an extract without surfacing
them is a breach, and share-alike attaches to a publicly served derived database.

---

## Temporal resampling

Any ingested dataset can be resampled to a coarser temporal resolution (e.g. hourly → daily, daily → monthly) using the standard openEO `aggregate_temporal_period` process in a process graph. See [Processes](processes.md) for an example.
