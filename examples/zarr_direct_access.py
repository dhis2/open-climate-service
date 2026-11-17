"""Open a published GeoZarr dataset directly and demonstrate spatial and temporal subsetting.

Requires a running Open Geo Service instance with at least one published dataset.
Adjust BASE_URL if the API is not running on the default local address.
"""

from open_geo_service import GeoService

BASE_URL = "http://127.0.0.1:9000"


def main() -> None:
    """Open a Zarr store directly and demonstrate spatial and temporal subsetting."""
    service = GeoService(BASE_URL)
    datasets = service.datasets()
    if not datasets:
        print("No published datasets found. Run an ingestion first.")
        return

    first = datasets[0]
    print(f"Opening: {first['title']}\n")

    ds = service.open_dataset(first["id"])
    print(ds)

    print(f"\nDimensions:  {dict(ds.sizes)}")
    print(f"Time range:  {ds['t'].values[0]}  →  {ds['t'].values[-1]}")
    print(f"Latitude:    {ds['y'].min().item():.4f}  →  {ds['y'].max().item():.4f}")
    print(f"Longitude:   {ds['x'].min().item():.4f}  →  {ds['x'].max().item():.4f}")

    variable = list(ds.data_vars)[0]

    # Select a single time step
    t0 = ds["t"].values[0]
    snapshot = ds[variable].isel(t=0).compute()
    print(f"\n{variable} snapshot at {t0}:")
    print(f"  shape: {snapshot.shape},  min: {snapshot.min().item()},  max: {snapshot.max().item()}")

    # Select the point closest to the spatial centre of the domain
    centre_y = ds["y"].mean().item()
    centre_x = ds["x"].mean().item()
    point = ds[variable].sel(y=centre_y, x=centre_x, method="nearest")
    print(f"\n{variable} at domain centre ({centre_y:.2f}, {centre_x:.2f}):")
    print(point.to_dataframe()[[variable]].head(10))

    # Spatial mean over the full domain — first 10 time steps
    spatial_mean = ds[variable].isel(t=slice(10)).mean(dim=["y", "x"])
    print(f"\nSpatial mean {variable} time series (first 10 steps):")
    print(spatial_mean.to_dataframe()[[variable]])


if __name__ == "__main__":
    main()
