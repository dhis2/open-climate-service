# Processes

The Open Climate Service exposes two complementary processing interfaces:

- **openEO process graphs** — the primary interface for data analysis. Submit a DAG of composable operations via `POST /result` (synchronous) or `POST /jobs` (batch). 120+ standard processes are available out of the box. See the [openEO guide](openeo.md).
- **Plugin processes** — custom named processes registered via the `@process` decorator, discoverable at `GET /processes` and callable directly by `process_id` in any openEO process graph. See [Extensibility — Processes](extensibility.md#processes).

---

## Temporal resampling

Temporal resampling (e.g. daily → monthly, hourly → daily) is handled by the standard openEO `aggregate_temporal_period` process:

```json
{
  "process_graph": {
    "load": {
      "process_id": "load_collection",
      "arguments": { "id": "era5land_temperature_hourly" }
    },
    "resample": {
      "process_id": "aggregate_temporal_period",
      "arguments": {
        "data": { "from_node": "load" },
        "period": "day",
        "reducer": {
          "process_graph": {
            "mean": {
              "process_id": "mean",
              "arguments": { "data": { "from_parameter": "data" } },
              "result": true
            }
          }
        }
      },
      "result": true
    }
  }
}
```

See the [openEO guide](openeo.md) and the [openEO process specification](https://processes.openeo.org/#aggregate_temporal_period) for the full parameter reference.

---

## Spatial aggregation

`aggregate_spatial` reduces a cube to one value per geometry and time step: the input to every org-unit workflow, the DHIS2 and CHAP exports included. It diverges from the openEO specification on purpose, so the same graph can give different numbers here than on another openEO backend.

### Polygons are area-weighted

The specification counts a cell when its centre falls inside the polygon. That returns no value for a zone smaller than a cell, and puts every zone edge up to half a cell off. Open Climate Service instead weights each cell by the fraction of it the polygon covers, computed by [exactextract](https://github.com/isciences/exactextract):

| Reducer | Result |
| --- | --- |
| `mean` | Covered-area-weighted mean |
| `sum` | Sum of each value times its covered fraction, a total over the covered area |
| `min`, `max` | Over every cell the polygon touches |
| `median` | Weighted median; equal weights give what `np.median` gives |

On seNorge monthly temperature over Norway's 357 kommuner, the specification's rule returns no value for 1 kommune at the native 1 km grid and for 13 at an ERA5-Land-like 11 km grid. Area weighting returns a value for all of them.

The weighted path applies when the reducer is exactly one of the statistics above, written as `reduce_by_method` or as openEO's own `mean`, `sum`, `min`, `max` or `median` (without `ignore_nodata: false`). It is recognised from the graph's structure, never from its output. A reducer that does anything more, such as a mean multiplied by 2 or capped with `clip`, runs as given over the specification's pixel-centre rule, the log says so, and no method is recorded for a named export to check against. A geometry that captures no data at all is named in a warning rather than returned as a silent NaN.

### Categorical data is never averaged

Each variable's dataset declares categorical data through `ingestion.resampling` (see [Pyramid resampling](adding_custom_datasets.md#pyramid-resampling)): `mode` for class codes such as land cover, `max` for presence masks, or `nearest`. For such a dataset:

- `mean`, `median` and `sum` become the **area-weighted majority** class, with a warning, since an average of class codes names no class. `min` and `max` keep their meaning.
- `reduce_by_method` with `method: "majority"` asks for the majority directly.
- `reduce_by_method` with `method: "fractions"` returns each class's share of the zone, on a new `class` dimension shared by every variable, so a class absent from one variable's zones reads 0. More than 256 distinct values is refused, since that is continuous data rather than class codes. "What share of this district is forest" is usually the more useful question for health data.

The declaration travels as a cube attribute from `load_collection`. An operation between the two that drops attributes leaves the continuous default.

### Points are interpolated

A `Point` samples the surface at its location instead of taking whichever cell contains it:

- continuous data: bilinear interpolation between the four nearest cell centres;
- categorical data: the nearest cell.

The optional `method` argument, an Open Climate Service extension using openEO's `resample_spatial` names, overrides this: `near`, `bilinear` or `cubic` (which needs at least 4 cells along each axis). Interpolation requested over categorical data is allowed but logged, since it treats class codes as ordinal. Where interpolation has nothing to work with, within half a cell of the grid edge or next to a missing cell such as sea beside a coastal facility, the point takes its containing cell's value. A point outside the grid is NaN.

Points and polygons can be mixed in one feature collection; results keep the input order. Lines and multipoints are refused.

### Values differ from earlier versions

Area weighting changes numbers that were already delivered. Most means move by hundredths of a degree at 1 km, and by up to about a degree for small zones on coarse grids. `sum` moves the most, because partly covered cells now count partly. Re-deliver affected periods if the change matters for the data element.

### Grid alignment

Aggregating to polygons needs no alignment between datasets: each is aggregated on its own grid, and the polygons are the common reference. Aligning datasets to one grid is needed only for pixel-level analysis across datasets, such as a feature stack for a spatial model. It belongs in the process graph (`resample_cube_spatial` to a target cube), or, for a known national grid, at ingest time so the cost is paid once.

Geometries must be in the cube's CRS; reprojection inside the process is CLIM-892.
