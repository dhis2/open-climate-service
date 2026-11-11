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

**Set up before this release?** Change `machine api.earthdatahub.destine.eu` to `machine data.earthdatahub.destine.eu`. The stores moved to the new host, and `.netrc` matches on host, so the old entry gets a 401.

### Copernicus Climate Data Store (daily datasets)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) and get your API key from your profile.
2. Create `~/.ecmwfdatastoresrc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-cds-api-key>
```

Alternatively, set the `CDS_API_KEY` environment variable to the same key.

**Set up before this release?** The daily datasets used to need only Earth Data Hub credentials. They now read from CDS, so they need these CDS credentials as well.

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
| **Lag** | ~6 days |

Total precipitation for the **UTC calendar day**, summed from the 24 hourly values of that day.

Each hourly precipitation value is the total for the hour *ending* at its timestamp, so the value stamped 00:00 belongs to the previous day. A day therefore sums the values stamped 01:00 through 00:00 the next day. The CDS publishes whole days of timestamps (00:00–23:00), so the hour that closes the latest day only arrives with the next day's update: precipitation always ends one day earlier than temperature, and that day is appended on the next sync.

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

## Humidity datasets

### Daily dewpoint (from hourly)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_dewpoint_daily` |
| **Variable** | `d2m` |
| **Units** | °C |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS ERA5-Land analysis-ready data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data) |
| **Lag** | ~5 days |

Daily mean 2 m dewpoint temperature, averaged from the 24 hourly values of the **UTC calendar day**. Dewpoint is the temperature at which water vapor will start condensing as dew.

Together with `era5land_temperature_daily`, it can be used as input to the [`relative_humidity_from_daily_dewpoint`](workflows.md#relative-humidity-from-daily-dewpoint) workflow, which publishes an estimated daily relative humidity where only daily inputs exist.

---

### Daily relative humidity (from hourly)

| Property | Value |
| --- | --- |
| **Dataset ID** | `era5land_relative_humidity_daily` |
| **Variable** | `rh` |
| **Units** | % |
| **Period** | Daily |
| **Coverage** | Global, 1950–present |
| **Resolution** | ~9 km |
| **Source** | [CDS ERA5-Land analysis-ready data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data) |
| **Lag** | ~5 days |

Daily mean 2 m relative humidity for the **UTC calendar day**. It is computed for each hour from the hourly temperature and dewpoint, then averaged. Computing it from daily means instead underestimates it, most where days are hot and nights cool, because saturation vapour pressure rises exponentially with temperature. Ingesting it reads both hourly variables.

`era5land_relative_humidity_daily_min` and `era5land_relative_humidity_daily_max` are the same, with the daily minimum and maximum of the hourly values instead of the mean. The `relative_humidity_from_daily_dewpoint` workflow cannot produce these, as it estimates relative humidity from daily means only.

---

## Choosing the right dataset

| Use case | Recommended dataset |
| --- | --- |
| Daily health modelling | `era5land_temperature_daily` / `era5land_precipitation_daily` |
| Monthly climate–health modelling | `era5land_temperature_monthly` / `era5land_precipitation_monthly` |
| CHAP integration (South Sudan, East Africa) | `era5land_precipitation_monthly` |
