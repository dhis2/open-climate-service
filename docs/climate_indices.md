# Climate Indices

The Open Climate Service exposes the full [xclim](https://xclim.readthedocs.io) indicator library — 179 peer-reviewed, CF-compliant climate extreme indices — as native openEO processes. Every indicator appears automatically in `GET /processes` and is callable by name from any openEO process graph or client, with no additional configuration.

---

## What is available

All xclim indicator modules are covered:

| Module | Domain | Examples |
|---|---|---|
| `atmos` | Atmospheric | SPI, CDD, CWD, TX days above, heat wave frequency, frost days, growing degree days, … |
| `land` | Land surface | Snow water equivalent, soil freeze/thaw, … |
| `seaIce` | Sea ice | Ice extent, concentration trends |

Browse the full list from any connected client:

```python
import openeo

conn = openeo.connect("http://your-instance:9000")
processes = conn.list_processes()
print([p.id for p in processes])
```

Or call `GET /processes` directly and search by keyword. In the [openEO Web Editor](https://editor.openeo.org), the search box filters by id and summary.

---

## Standardized Precipitation Index (SPI)

SPI is the primary drought monitoring index. The `window` parameter selects the accumulation timescale:

| Process call | Timescale |
|---|---|
| `spi(pr, window=1)` | SPI-1 — short-term moisture anomaly |
| `spi(pr, window=3)` | SPI-3 — seasonal drought |
| `spi(pr, window=6)` | SPI-6 — medium-term drought |
| `spi(pr, window=12)` | SPI-12 — long-term / groundwater recharge |

### Python client

```python
import openeo

conn = openeo.connect("http://your-instance:9000")

precip = conn.load_collection(
    "chirps_rainfall_daily",
    temporal_extent=["2000-01-01", "2023-12-31"],
)

spi3 = precip.process(
    "spi",
    arguments={
        "pr": precip,
        "window": 3,
        "cal_start": "2000-01-01",
        "cal_end": "2020-12-31",
    },
)

job = spi3.save_result("Zarr").create_job()
job.start_and_wait()
```

### Process graph (JSON)

```json
{
  "process_graph": {
    "load": {
      "process_id": "load_collection",
      "arguments": {
        "id": "chirps_rainfall_daily",
        "temporal_extent": ["2000-01-01", "2023-12-31"]
      }
    },
    "spi": {
      "process_id": "spi",
      "arguments": {
        "pr": { "from_node": "load" },
        "window": 3,
        "cal_start": "2000-01-01",
        "cal_end": "2020-12-31"
      },
      "result": true
    }
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pr` | datacube | — | Daily precipitation (kg m⁻² s⁻¹ or mm/day) |
| `window` | integer | `1` | Accumulation window in months (1=SPI-1, 3=SPI-3, 6=SPI-6) |
| `cal_start` | string | start of record | Calibration period start (YYYY-MM-DD) |
| `cal_end` | string | end of record | Calibration period end (YYYY-MM-DD) |
| `freq` | string | `"MS"` | Output frequency — `"MS"` = monthly, `None` = pre-aggregated input |

SPI values are standardised: `0` = climatological median, `±1` = moderate anomaly, `±2` = extreme anomaly.

---

## Other common indices

### Consecutive dry / wet days

```python
# Annual maximum consecutive dry days (CDD)
precip.process("cdd", arguments={"pr": precip, "thresh": "1 mm/day", "freq": "YS"})

# Monthly maximum consecutive wet days (CWD)
precip.process("cwd", arguments={"pr": precip, "thresh": "1 mm/day", "freq": "MS"})
```

### Days above temperature threshold

```python
# TX days above 35°C per year
tmax.process("tx_days_above", arguments={"tasmax": tmax, "thresh": "35 degC", "freq": "YS"})
```

### Heat wave frequency

```python
# Annual count of heat waves (≥3 consecutive days with Tmax>35°C and Tmin>20°C)
tmax.process(
    "heat_wave_frequency",
    arguments={
        "tasmin": tmin,
        "tasmax": tmax,
        "thresh_tasmin": "20 degC",
        "thresh_tasmax": "35 degC",
        "freq": "YS",
    },
)
```

---

## Derived meteorological variables (earthkit-meteo)

Alongside the xclim indicators, the thermodynamic functions of ECMWF's [earthkit-meteo](https://earthkit.ecmwf.int) are auto-registered as processes. These are not indices but *derivations* — quantities computed from other quantities, useful as inputs to an index or a health model:

| Process | Computes |
|---|---|
| `relative_humidity_from_dewpoint` | Relative humidity (%) from temperature and dewpoint |
| `dewpoint_from_relative_humidity` | Dewpoint from temperature and relative humidity |
| `specific_humidity_from_dewpoint` | Specific humidity from dewpoint and pressure |
| `wet_bulb_temperature_from_dewpoint` | Wet-bulb temperature — a heat-stress input |
| `saturation_vapour_pressure` | Saturation vapour pressure over water/ice/mixed phase |
| `potential_temperature`, `virtual_temperature` | Thermodynamic temperatures |

Relative humidity is the common case, since the ERA5-Land source carries temperature and dewpoint but not humidity directly:

```python
t2m = conn.load_collection("era5land_temperature_daily")
d2m = conn.load_collection("era5land_dewpoint_daily")

rh = t2m.process("relative_humidity_from_dewpoint", arguments={"t": t2m, "td": d2m})
```

Two things worth noting. Both cubes are passed **explicitly** — the collection you call
`.process()` on is only the entry point, not an implicit first argument. And
`era5land_dewpoint_daily` is **not a built-in**: the shipped ERA5-Land templates cover
temperature (`t2m`) and precipitation (`tp`) only. Dewpoint needs a template of its own,
which is a copy of the temperature one with `variable: d2m` — the same
one-plugin-many-templates pattern described in
[Architecture](architecture.md#streaming-plugin).

Browse the full set in `GET /processes`; every function earthkit-meteo documents with a single cube output is registered.

### Units are converted for you

These functions expect ECMWF's native units — temperature in **K**, pressure in **Pa** — and do no checking of their own. Our stores are usually not in those units: ERA5-Land temperature is converted to `degC` at ingest. Passing `degC` where `K` is expected produces no error, just a wrong number.

The registered processes therefore enforce units on the way in. A cube in a compatible unit is converted (`degC` → `K`, `hPa` → `Pa`); a cube whose `units` attribute is missing or incompatible is rejected with an explanatory error.

So the call above needs no change on a `degC` store: `era5land_temperature_daily` and `era5land_dewpoint_daily` are both converted to `K` on the way in, and the answer is correct without a manual conversion step.

If a cube is rejected for missing units, the fix is to declare `units` on the variable in its dataset template — that value is what gets CF-stamped at ingest.

---

## Customising an index

The auto-registered xclim processes can be overridden per-instance by placing a `@process`-decorated function with the same id in `plugins_dir/processes/`. This is useful for adjusting default parameters or adding domain-specific documentation without modifying shared code.

```python
# plugins/processes/climate_indices.py
import xarray as xr
import xclim.indicators.atmos as xclim_atmos
from open_climate_service.process import process


@process(summary="Consecutive dry days (Rwanda threshold)")
def cdd(pr: xr.DataArray, thresh: str = "0.5 mm/day", freq: str = "MS") -> xr.DataArray:
    """CDD with a lower threshold suited to Rwanda's dry season definition."""
    return xclim_atmos.maximum_consecutive_dry_days(pr, thresh=thresh, freq=freq)
```

See [Extensibility — Processes](extensibility.md#processes) for the full `@process` decorator reference.

---

## References

- [xclim documentation](https://xclim.readthedocs.io)
- [xclim indicator catalogue](https://xclim.readthedocs.io/en/stable/indicators.html)
- [earthkit-meteo documentation](https://earthkit-meteo.readthedocs.io)
- [openEO process graph specification](https://processes.openeo.org)
