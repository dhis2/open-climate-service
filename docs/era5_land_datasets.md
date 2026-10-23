# ERA5-Land datasets

ERA5-Land is a global reanalysis of land surface variables produced by ECMWF. It provides a consistent record from 1950 to the near-present at ~9 km resolution. The Open Climate Service ships six ERA5-Land data sources covering temperature and precipitation at daily and monthly resolution.

## Setup

ERA5-Land data is fetched from two sources depending on the dataset. Both are free but require registration.

### Earth Data Hub (monthly datasets and daily normals)

1. Create a free account at [earthdatahub.destine.eu](https://earthdatahub.destine.eu).
2. Go to **Account settings → My API keys → Standard API keys** and generate a key.
3. Add it to `~/.netrc`:

```
machine data.earthdatahub.destine.eu
  password <your-standard-api-key>
```

```bash
chmod 600 ~/.netrc
```

**Windows:** use `%USERPROFILE%\_netrc` (underscore prefix) with the same content.

### Copernicus Climate Data Store (daily datasets)

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

### Daily (from hourly)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_temperature_daily` |
| **Variable** | `t2m` |
| **Units** | °C |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS ERA5-Land analysis-ready data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data) |
| **Lag** | ~5 days |

Daily mean temperature, averaged from the 24 hourly values of the **UTC calendar day** (00:00–23:59 UTC).

`era5land_temperature_daily_min` and `era5land_temperature_daily_max` are the same, with the daily minimum and maximum of the hourly values instead of the mean.

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
| **Source** | [Earth Data Hub ERA5-Land daily](https://earthdatahub.destine.eu/collections/era5/datasets/era5-land-daily) |
| **Lag** | Available after each month is complete in the source |

Monthly mean 2 m temperature, averaged from the Earth Data Hub daily means of the calendar month. Only complete months are ingested.

---

## Precipitation datasets

Precipitation values represent **total precipitation reaching the surface**, including rain and snow as liquid-equivalent water. Units are **mm/d**, a daily rate: the daily value is the day's total in mm, and the monthly value is the mean of the daily totals. Multiply a monthly value by the number of days in the month for the monthly total.

### Daily (from hourly)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_precipitation_daily` |
| **Variable** | `tp` |
| **Units** | mm/d |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS ERA5-Land analysis-ready data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data) |
| **Lag** | ~5 days |

Total precipitation for the **UTC calendar day**, summed from the 24 hourly values within 00:00–23:59 UTC.

---

### Monthly

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_precipitation_monthly` |
| **Variable** | `tp` |
| **Units** | mm/d |
| **Period** | Monthly |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [Earth Data Hub ERA5-Land daily](https://earthdatahub.destine.eu/collections/era5/datasets/era5-land-daily) |
| **Lag** | Available after each month is complete in the source |

Mean daily precipitation for the calendar month, averaged from the Earth Data Hub daily totals. Only complete months are ingested.

---

## Choosing the right dataset

| Use case | Recommended dataset |
| --- | --- |
| Daily health modelling | `era5land_temperature_daily` / `era5land_precipitation_daily` |
| Monthly climate–health modelling | `era5land_temperature_monthly` / `era5land_precipitation_monthly` |
| CHAP integration (South Sudan, East Africa) | `era5land_precipitation_monthly` |
