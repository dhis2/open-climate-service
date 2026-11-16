"""FastAPI router exposing dataset endpoints."""

from fastapi import APIRouter
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..data_registry.routes import _get_dataset_or_404
from .services.accessor import cleanup_file, get_data, xarray_to_temporary_netcdf

router = APIRouter()


@router.get("/{dataset_id}")
def get_file(
    dataset_id: str,
    start: str,
    end: str,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
    format: str = "netcdf",
) -> FileResponse:
    """Get a dataset filtered to a timeperiod and bbox as a downloadable raster file."""
    dataset = _get_dataset_or_404(dataset_id)

    # get filtered data
    if all(value is not None for value in (xmin, ymin, xmax, ymax)):
        assert xmin is not None and ymin is not None and xmax is not None and ymax is not None
        bbox: list[float] | None = [xmin, ymin, xmax, ymax]
    else:
        bbox = None
    ds = get_data(dataset, start, end, bbox)

    # save to temporary file
    if format.lower() == "netcdf":
        # convert to netcdf
        file_path = xarray_to_temporary_netcdf(ds)

    else:
        raise ValueError(f"Unsupported output format: {format}")

    # return as file
    return FileResponse(
        file_path,
        media_type="application/x-netcdf",
        filename="geoservice-raster-download.nc",
        background=BackgroundTask(cleanup_file, file_path),
    )
