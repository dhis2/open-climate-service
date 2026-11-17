"""Discover published datasets via the STAC catalog and open one with xarray.

Requires a running Open Geo Service instance and at least one published dataset.
Adjust BASE_URL if the API is not running on the default local address.
"""

import json

from open_geo_service import GeoService

BASE_URL = "http://127.0.0.1:9000"


def main() -> None:
    """Discover and open the first published dataset."""
    service = GeoService(BASE_URL)
    datasets = service.datasets()

    if not datasets:
        print("No published datasets found. Run an ingestion first.")
        return

    print(json.dumps(datasets, indent=2))

    first = datasets[0]
    print(f"\nOpening: {first['title']}")

    ds = service.open_dataset(first["id"])
    print(ds)

    print(f"\nTime range: {ds['t'].values[0]}  →  {ds['t'].values[-1]}")
    print(f"Time steps: {ds.sizes['t']}")
    print(f"Latitude:  {ds['y'].min().item():.4f}  →  {ds['y'].max().item():.4f}")
    print(f"Longitude: {ds['x'].min().item():.4f}  →  {ds['x'].max().item():.4f}")

    variable = list(ds.data_vars)[0]
    centre_y = ds["y"].mean().item()
    centre_x = ds["x"].mean().item()
    sample = ds[variable].isel(t=0).sel(y=centre_y, x=centre_x, method="nearest")
    print(f"\n{variable} at domain centre, t=0: {sample.compute().item()}")


if __name__ == "__main__":
    main()
