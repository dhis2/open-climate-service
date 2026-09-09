# User Guide

This guide shows how to discover and access data from a running Open Climate Service instance. It assumes the API is running locally at `http://127.0.0.1:9000` and that at least one dataset has been ingested and published.

For configuring a new instance for your country, see the [instance guide](instance_guide.md). To browse datasets visually instead of through code, use the built-in [map viewer](web_interface.md#the-map-viewer-map). For the full ingestion and sync API reference, see [managed_data_api_guide.md](managed_data_api_guide.md).

## Discovering datasets

The STAC catalog is the starting point for data discovery. It lists all published GeoZarr datasets as STAC Collections.

```bash
curl -s http://127.0.0.1:9000/stac/catalog.json | jq
```

Each entry in `links` with `"rel": "child"` points to one dataset collection. Use the `href` from the catalog to fetch it:

```bash
# Replace {dataset_id} with any id from the catalog above, e.g. chirps3_precipitation_daily
curl -s http://127.0.0.1:9000/stac/collections/{dataset_id} | jq
```

The `assets.zarr` field contains everything needed to open the dataset:

```json
{
  "assets": {
    "zarr": {
      "href": "http://127.0.0.1:9000/zarr/chirps3_precipitation_daily",
      "xarray:open_kwargs": { "consolidated": true }
    }
  }
}
```

## The Python client

`open_climate_service` ships a lightweight `ClimateService` client. The base install
needs only `httpx` and `pystac`; opening datasets as xarray adds the `xarray` extra:

```bash
pip install open-climate-service            # client only — talk to an instance over HTTP
pip install open-climate-service[xarray]    # + open published datasets as xarray
```

```python
from open_climate_service import ClimateService

service = ClimateService("http://127.0.0.1:9000")

# Discover published datasets (STAC catalog)
for link in service.datasets():
    print(link["id"], "—", link["title"])

# List the processes and reusable workflows the instance exposes
service.processes()  # standard openEO processes
service.workflows()  # stored workflows / UDPs (e.g. aggregate_to_dhis2_json)

# Run a process graph synchronously. JSON results (e.g. a DHIS2 dataValueSet) come
# back as a dict; file results (CSV, GeoJSON, GeoTIFF) come back as bytes.
data_value_set = service.execute(
    {
        "agg": {
            "process_id": "aggregate_to_dhis2_json",
            "arguments": {
                "dataset_id": "era5land_temperature_monthly",
                "temporal_extent": ["2025-01-01", "2025-12-31"],
                "geometries": {"type": "FeatureCollection", "features": []},
                "data_element_id": "fbfJHSPpUQD",
                "method": "mean",
            },
            "result": True,
        }
    }
)
```

## Opening a dataset with xarray

`open_dataset` reads the Zarr asset advertised in the dataset's STAC collection and returns a lazy `xarray.Dataset` (requires the `xarray` extra):

```python
from open_climate_service import ClimateService

service = ClimateService("http://127.0.0.1:9000")

datasets = service.datasets()
ds = service.open_dataset(datasets[0]["id"])  # open whichever dataset is published first
print(ds)
```

The `base_url` defaults to the `CLIMATE_SERVICE_BASE_URL` environment variable (falling back to `http://127.0.0.1:9000`), so module-level functions work without any argument when the env var is set:

```python
from open_climate_service.client import list_datasets, open_dataset  # reads CLIMATE_SERVICE_BASE_URL

dataset_id = list_datasets()[0]["id"]
ds = open_dataset(dataset_id)
```

Each dataset has a `t` dimension (time), `x` and `y` spatial dimensions, and a data variable matching the variable (e.g. `precip` for CHIRPS, `t2m` for ERA5-Land temperature).

Select the first time step:

```python
snapshot = ds.isel(t=0)
print(snapshot)
```

Select a spatial point by sampling the centre of the domain:

```python
variable = list(ds.data_vars)[0]  # precip, t2m, tp, or pop_total depending on the dataset
centre_y = ds.y.mean().item()
centre_x = ds.x.mean().item()
point = ds.sel(y=centre_y, x=centre_x, method="nearest")
print(point[variable].values)
```

Compute the spatial mean over the first 10 time steps (slicing first avoids reading the full dataset over HTTP):

```python
spatial_mean = ds[variable].isel(t=slice(10)).mean(dim=["y", "x"])
print(spatial_mean.to_dataframe())
```

## What's next

- **Process data** — run temporal aggregations, spatial filters, and custom calculations via openEO process graphs. See [openeo.md](openeo.md) and [`examples/openeo_process_graph.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/openeo_process_graph.py).
- **Built-in datasets** — coverage, units, and sync behaviour of each source. See [built_in_datasets.md](built_in_datasets.md).
- **Discovery scripts** — [`examples/stac_discover_and_open.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/stac_discover_and_open.py) and [`examples/zarr_direct_access.py`](https://github.com/dhis2/open-climate-service/blob/main/examples/zarr_direct_access.py).
- **Admin API** — ingestion, sync, and publication reference. See [managed_data_api_guide.md](managed_data_api_guide.md).
