# ERA5-Land datasets

ERA5-Land is a global reanalysis of land surface variables produced by ECMWF. It provides a consistent record from 1950 to the near-present at ~9 km resolution. The Open Climate Service ships six ERA5-Land data sources covering temperature and precipitation at daily and monthly resolution.

## Setup

ERA5-Land data is fetched from two sources depending on the dataset. Both are free but require registration.

### Earth Data Hub (hourly and daily datasets)

1. Create a free account at [earthdatahub.destine.eu](https://earthdatahub.destine.eu).
2. Go to **Account settings → My API keys → Standard API keys** and generate a key.
3. Add it to `~/.netrc`:

```
machine api.earthdatahub.destine.eu
  password <your-standard-api-key>
```

```bash
chmod 600 ~/.netrc
```

**Windows:** use `%USERPROFILE%\_netrc` (underscore prefix) with the same content.

### Copernicus Climate Data Store (monthly datasets)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) and get your API key from your profile.
2. Create `~/.ecmwfdatastoresrc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-cds-api-key>
```

### Ingest your first ERA5-Land dataset

Once authenticated, ingest ERA5-Land daily temperature:

```bash
curl -s -X POST http://127.0.0.1:9000/ingestions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "era5land_temperature_daily",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "publish": true
  }' | jq
```

---

## Temperature datasets

### Daily (pre-computed)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_temperature_daily` |
| **Variable** | `t2m` |
| **Units** | °C |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/era5-land-daily) |
| **Lag** | ~4 weeks |

Daily mean temperature for the **UTC calendar day** (00:00–23:59 UTC). For most instances this is the right choice.

For instances where health data is reported using a local calendar that does not align with UTC (e.g. UTC+3 East Africa), use `era5land_temperature_daily_from_hourly` instead.

**Sync** — new days are appended incrementally. Earth Data Hub provides pre-computed daily means; this dataset uses those directly for efficiency and falls back to a CDS-derived daily mean for the most recent ~4 weeks not yet published by Earth Data Hub.

---

### Daily (from hourly, timezone-aware)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_temperature_daily_from_hourly` |
| **Variable** | `t2m` |
| **Units** | °C |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-land) |
| **Lag** | ~4 weeks |

Daily mean temperature computed by averaging the 24 hourly values that fall within the **local calendar day**. The local day is defined by `utc_offset_hours` in `climate-service.yaml` (default: 0 = UTC).

For a UTC+3 instance, "January 1" covers 21:00 UTC December 31 through 20:59 UTC January 1 — matching the calendar day as experienced locally.

**When to use this instead of `era5land_temperature_daily`:** when your health data is reported by local date and your instance has a non-zero `utc_offset_hours`.

---

### Monthly

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_temperature_monthly` |
| **Variable** | `t2m` |
| **Units** | °C |
| **Period** | Monthly |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS reanalysis-era5-land-monthly-means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) |
| **Lag** | ~2 months |

Monthly mean 2 m temperature, downloaded directly from the Copernicus monthly averages product. Each value is the mean of all hourly temperature readings in the calendar month.

---

## Precipitation datasets

Precipitation values represent **total precipitation reaching the surface**, including rain and snow as liquid-equivalent water. Units are always **mm** (millimetres of water depth).

!!! note "About daily totals"
    Daily precipitation values represent the **total water that fell during that calendar day**. Summing the daily values over a month matches the monthly total dataset to within rounding.

### Daily (UTC calendar day)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_precipitation_daily` |
| **Variable** | `tp` |
| **Units** | mm |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-land) |
| **Lag** | ~4 weeks |

Total precipitation for the **UTC calendar day**. Each value is the sum of all precipitation that fell within 00:00–23:59 UTC.

For instances where health data is reported using a local calendar, use `era5land_precipitation_daily_from_hourly` instead.

**Sync** — Earth Data Hub covers history. For the most recent ~4 weeks not yet published by Earth Data Hub, daily totals are derived from an hourly download from CDS.

---

### Daily (timezone-aware)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_precipitation_daily_from_hourly` |
| **Variable** | `tp` |
| **Units** | mm |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-land) |
| **Lag** | ~4 weeks |

Total precipitation summed over the **local calendar day**, defined by `utc_offset_hours` in `climate-service.yaml`.

**When to use this instead of `era5land_precipitation_daily`:** when your health data is reported by local date and your instance has a non-zero `utc_offset_hours`.

---

### Monthly

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_precipitation_monthly` |
| **Variable** | `tp` |
| **Units** | mm |
| **Period** | Monthly |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS reanalysis-era5-land-monthly-means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) |
| **Lag** | ~2 months |

Total precipitation for the calendar month in mm. The CDS monthly-means product provides the mean daily precipitation; this is multiplied by the number of days in the month at ingest time to give a directly comparable monthly total.

This is the recommended dataset for monthly precipitation analysis and climate–health modelling. It is produced by Copernicus and represents the most authoritative ERA5-Land precipitation product.

---

## Choosing the right dataset

| Use case | Recommended dataset |
| --- | --- |
| Daily health modelling, UTC-aligned | `era5land_temperature_daily` / `era5land_precipitation_daily` |
| Daily health modelling, local calendar | `era5land_temperature_daily_from_hourly` / `era5land_precipitation_daily_from_hourly` |
| Monthly climate–health modelling | `era5land_temperature_monthly` / `era5land_precipitation_monthly` |
| CHAP integration (South Sudan, East Africa) | `era5land_precipitation_monthly` |

## UTC offset configuration

For instances in non-UTC time zones, set `utc_offset_hours` in `climate-service.yaml`:

```yaml
utc_offset_hours: 3   # East Africa (UTC+3)
```

This affects all daily datasets — both the UTC and local-time variants use the offset when computing day boundaries.
