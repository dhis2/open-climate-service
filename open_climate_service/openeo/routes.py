"""openEO HTTP routes."""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from open_climate_service.openeo import collections as collections_service
from open_climate_service.openeo import processes as processes_service
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.openeo.capabilities import build_capabilities
from open_climate_service.openeo.jobs import get_openeo_job_service
from open_climate_service.openeo.schemas import (
    OpenEOJobCreate,
    OpenEOJobListResponse,
    OpenEOJobRecord,
    OpenEOJobResults,
    OpenEOJobUpdate,
    WorkflowListResponse,
    WorkflowRecord,
)

capabilities_router = APIRouter(tags=["openEO"])
collections_router = APIRouter(tags=["openEO"])
processes_router = APIRouter(tags=["openEO"])
jobs_router = APIRouter(tags=["openEO"])
udp_router = APIRouter(tags=["openEO"])
result_router = APIRouter(tags=["openEO"])


# ---------------------------------------------------------------------------
# Capabilities  GET /
# (registered without prefix in main.py so it overlays the system route)
# ---------------------------------------------------------------------------


def get_openeo_capabilities(request: Request) -> JSONResponse:
    """Return openEO capabilities when the client requests JSON."""
    base_url = _abs_base(request)
    caps = build_capabilities(base_url)
    return JSONResponse(caps.model_dump())


@capabilities_router.get("/.well-known/openeo")
def well_known_openeo(request: Request) -> dict[str, Any]:
    """OpenEO service discovery — lets clients find the versioned API URL."""
    base_url = _abs_base(request)
    return {
        "versions": [
            {
                "url": base_url,
                "api_version": "1.2.0",
                "production": False,
            }
        ]
    }


@capabilities_router.get("/credentials/oidc")
def credentials_oidc() -> dict[str, Any]:
    """Return OIDC authentication providers (empty = open/anonymous access)."""
    return {"providers": []}


@capabilities_router.get("/file_formats")
def file_formats() -> dict[str, Any]:
    """Return supported input and output file formats."""
    output_formats: dict[str, Any] = {
        "ZARR": {
            "title": "Zarr",
            "description": "Zarr v3 chunked array store — cloud-native format for multi-dimensional data",
            "gis_data_types": ["raster"],
            "parameters": {},
            "links": [{"rel": "about", "href": "https://zarr.dev"}],
        },
        "NETCDF": {
            "title": "Network Common Data Form (NetCDF)",
            "description": "Self-describing format for climate and Earth science, supported by CDO, NCO, xarray and R",
            "gis_data_types": ["raster"],
            "parameters": {},
            "links": [{"rel": "about", "href": "https://www.unidata.ucar.edu/software/netcdf/"}],
        },
        "GTIFF": {
            "title": "GeoTIFF",
            "description": "Georeferenced TIFF raster with embedded CRS — compatible with QGIS and GDAL",
            "gis_data_types": ["raster"],
            "parameters": {},
            "links": [{"rel": "about", "href": "https://www.ogc.org/standards/geotiff"}],
        },
        "PNG": {
            "title": "Portable Network Graphics (PNG)",
            "description": "Colourised raster image of a single 2-D slice, suitable for quick previews and web display",
            "gis_data_types": ["raster"],
            "parameters": {},
            "links": [],
        },
        "GEOJSON": {
            "title": "GeoJSON",
            "description": "Standard JSON encoding of geographic features — default for vector results",
            "gis_data_types": ["vector"],
            "parameters": {},
            "links": [{"rel": "about", "href": "https://geojson.org"}],
        },
        "CSV": {
            "title": "Comma Separated Values (CSV)",
            "description": "Plain text tabular format — suitable for time series and spreadsheet import",
            "gis_data_types": ["raster", "vector", "table"],
            "parameters": {},
            "links": [],
        },
        "PARQUET": {
            "title": "GeoParquet",
            "description": "Apache Parquet with geometry column — efficient for large vector datasets",
            "gis_data_types": ["vector"],
            "parameters": {},
            "links": [{"rel": "about", "href": "https://geoparquet.org"}],
        },
        "CHAPCSV": {
            "title": "CHAP CSV",
            "description": (
                "Wide-format CSV for CHAP imports with one row per location-period and one column per variable."
            ),
            "gis_data_types": ["table"],
            "parameters": {
                "period_field": {"type": "string", "default": "t"},
                "location_field": {"type": "string", "default": "geometry"},
                "period_type": {"type": "string"},
            },
            "links": [],
        },
        "DHIS2JSON": {
            "title": "DHIS2 JSON",
            "description": (
                "DHIS2 import-ready JSON dataValues envelope for aggregated org-unit results. "
                "Requires save_result options such as data_element_id, org_unit_field, and period_type."
            ),
            "gis_data_types": ["table", "vector"],
            "parameters": {
                "data_element_id": {"type": "string"},
                "org_unit_field": {"type": "string"},
                "period_field": {"type": "string", "default": "t"},
                "period_type": {"type": "string"},
                "category_option_combo": {"type": "string"},
            },
            "links": [],
        },
    }
    from open_climate_service.exports.registry import load_export_plugins

    for plugin in load_export_plugins().values():
        if plugin.format not in output_formats:
            output_formats[plugin.format] = {
                "title": plugin.format,
                "description": "Pure export renderer; requires a configured export ID in save_result options.",
                "gis_data_types": ["table"],
                "parameters": {},
                "links": [],
            }
        output_formats[plugin.format]["parameters"]["export"] = {
            "type": "string",
            "description": "Named export mapping; use this option without per-request mapping overrides.",
        }
    return {
        "input": {},
        "output": output_formats,
    }


@capabilities_router.get("/service_types")
def service_types() -> dict[str, Any]:
    """Return supported secondary web service types (none currently)."""
    return {}


@capabilities_router.get("/me")
def me() -> dict[str, Any]:
    """Return basic account info for this open/anonymous backend."""
    return {"user_id": "anonymous", "links": []}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


@collections_router.get("")
def list_collections(request: Request) -> dict[str, Any]:
    """Return all published collections (openEO + STAC compatible)."""
    return collections_service.list_collections(request)


@collections_router.get("/{collection_id}")
def get_collection(collection_id: str, request: Request) -> dict[str, Any]:
    """Return one published collection."""
    return collections_service.get_collection(collection_id, request)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


@processes_router.get("")
def list_processes(request: Request) -> dict[str, Any]:
    """Return all available openEO processes."""
    procs = processes_service.list_openeo_processes()
    base_url = _abs_base(request)
    return {
        "processes": procs,
        "links": [{"rel": "self", "href": f"{base_url}/processes", "type": "application/json"}],
    }


@processes_router.get("/{process_id}")
def get_process_spec(process_id: str) -> dict[str, Any]:
    """Return one openEO process description by id."""
    p = processes_service.get_openeo_process(process_id)
    if p is not None:
        return p
    raise HTTPException(status_code=404, detail=f"Process '{process_id}' not found")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@jobs_router.get("", response_model=OpenEOJobListResponse)
def list_jobs() -> OpenEOJobListResponse:
    """Return all openEO jobs."""
    return get_openeo_job_service().list_jobs()


@jobs_router.post("", status_code=201)
def create_job(body: OpenEOJobCreate, response: Response) -> OpenEOJobRecord:
    """Create a new openEO job."""
    record = get_openeo_job_service().create_job(body)
    response.headers["Location"] = f"/jobs/{record.id}"
    response.headers["OpenEO-Identifier"] = record.id
    return record


@jobs_router.get("/{job_id}", response_model=OpenEOJobRecord)
def get_job(job_id: str) -> OpenEOJobRecord:
    """Return one openEO job."""
    return get_openeo_job_service().get_job_or_404(job_id)


@jobs_router.patch("/{job_id}", status_code=204)
def update_job(job_id: str, body: OpenEOJobUpdate) -> Response:
    """Update job metadata (title, description, process, etc.)."""
    get_openeo_job_service().update_job(job_id, body)
    return Response(status_code=204)


@jobs_router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    """Delete a job and its results."""
    get_openeo_job_service().delete_job(job_id)
    return Response(status_code=204)


@jobs_router.post("/{job_id}/results", status_code=202)
def start_job(job_id: str) -> Response:
    """Queue a job for processing."""
    get_openeo_job_service().start_job(job_id)
    return Response(status_code=202)


@jobs_router.get("/{job_id}/results", response_model=OpenEOJobResults)
def get_results(job_id: str) -> OpenEOJobResults:
    """Return result links for a finished job."""
    return get_openeo_job_service().get_results(job_id)


@jobs_router.delete("/{job_id}/results", status_code=204)
def cancel_job(job_id: str) -> Response:
    """Cancel a running or queued job."""
    get_openeo_job_service().cancel_job(job_id)
    return Response(status_code=204)


@jobs_router.get("/{job_id}/results/result.geojson")
def download_geojson_result(job_id: str) -> FileResponse:
    """Serve the GeoJSON result file for a finished batch job."""
    from open_climate_service.openeo.jobs import _JOBS_DIR

    path = _JOBS_DIR / job_id / "results" / "result.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(str(path), media_type="application/geo+json", filename="result.geojson")


@jobs_router.get("/{job_id}/results/result.zarr/{zarr_path:path}")
def download_zarr_chunk(job_id: str, zarr_path: str) -> FileResponse:
    """Serve one file from within the Zarr result store (chunk or metadata)."""
    from open_climate_service.openeo.jobs import _JOBS_DIR

    store_dir = _JOBS_DIR / job_id / "results" / "result.zarr"
    if not store_dir.is_dir():
        raise HTTPException(status_code=404, detail="Result Zarr store not found")

    # Reject path traversal attempts.
    try:
        chunk_path = (store_dir / zarr_path).resolve()
        chunk_path.relative_to(store_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Zarr path")

    if not chunk_path.exists() or not chunk_path.is_file():
        raise HTTPException(status_code=404, detail="Zarr chunk not found")

    return FileResponse(str(chunk_path))


_RESULT_MEDIA_TYPES: dict[str, str] = {
    ".nc": "application/netcdf",
    ".tif": "image/tiff; subtype=geotiff",
    ".png": "image/png",
    ".csv": "text/csv",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".parquet": "application/vnd.apache.parquet",
}


def _json_tabular_payload_response(frame: Any, options: dict[str, Any]) -> JSONResponse:
    from open_climate_service.openeo.jobs import _build_dhis2_json_payload

    try:
        payload = _build_dhis2_json_payload(frame, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(payload)


@jobs_router.get("/{job_id}/results/{filename}")
def download_result_file(job_id: str, filename: str) -> FileResponse:
    """Serve a result file (NetCDF, GeoTIFF, PNG, CSV, GeoParquet) for a finished batch job."""
    from open_climate_service.openeo.jobs import _JOBS_DIR

    results_dir = _JOBS_DIR / job_id / "results"
    try:
        path = (results_dir / filename).resolve()
        path.relative_to(results_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")

    suffix = path.suffix.lower()
    media_type = _RESULT_MEDIA_TYPES.get(suffix, "application/octet-stream")
    from open_climate_service.exports.service import read_export_metadata

    metadata = read_export_metadata(path)
    if metadata is not None:
        media_type = metadata["media_type"]
    return FileResponse(str(path), media_type=media_type, filename=filename)


def _sync_export_response(
    write_output: Callable[[Any], str | None],
    fmt: str,
    media_type: str | None = None,
) -> Response:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            output = write_output(Path(tmp))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if output is None:
            raise HTTPException(status_code=500, detail=f"Format '{fmt}' produced no output")
        path = Path(output)
        data = path.read_bytes()
        resolved_media_type = media_type or _RESULT_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return Response(content=data, media_type=resolved_media_type)


# ---------------------------------------------------------------------------
# Synchronous result  POST /result
# ---------------------------------------------------------------------------


@result_router.post("")
def execute_synchronous(
    body: dict[str, Any] = Body(...),
    request: Request = None,  # type: ignore[assignment]
) -> Any:
    """Execute a process graph synchronously and return the result.

    Returns JSON for scalar/dict results, or an exported file for
    synchronously-supported non-Zarr raster/vector formats.
    """
    from open_climate_service.openeo.execution import run_process_graph

    # Accept both { "process": { "process_graph": ... } } (spec)
    # and   { "process_graph": ... } (editor direct export)
    if "process" in body:
        process = body["process"]
    elif "process_graph" in body:
        process = body
    else:
        raise HTTPException(status_code=422, detail="Body must contain a 'process' object or 'process_graph' directly")
    if not isinstance(process, dict):
        raise HTTPException(status_code=422, detail="Body must contain a 'process' object")
    result = run_process_graph(process, request)

    import tempfile

    import xarray as xr

    from open_climate_service.openeo.execution import SaveResultEnvelope
    from open_climate_service.openeo.jobs import (
        _VECTOR_FORMATS,
        _write_dataset_tabular_export,
        _write_raster,
        _write_tabular_export,
        _write_vector,
    )

    # Unwrap save_result envelope to get requested format
    fmt = "ZARR"
    options: dict[str, Any] = {}
    if isinstance(result, SaveResultEnvelope):
        fmt = result.format
        options = result.options
        result = result.data

    if "export" in options:
        from open_climate_service.exports.service import render_named_export

        try:
            plugin, rendered = render_named_export(result, fmt, options)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=rendered.content, media_type=plugin.media_type)

    if isinstance(result, xr.DataArray):
        result = result.to_dataset(name=result.name or "result")

    if isinstance(result, xr.Dataset):
        if fmt == "DHIS2JSON":
            return _json_tabular_payload_response(result.to_dataframe().reset_index(), options)
        if fmt == "ZARR":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Synchronous datacube results do not support ZARR output; "
                    "use a non-ZARR format or submit a batch job"
                ),
            )
        if fmt == "CHAPCSV":
            return _sync_export_response(
                lambda tmp_dir: _write_dataset_tabular_export(result, tmp_dir, fmt, options),
                fmt,
                "text/csv",
            )
        return _sync_export_response(
            lambda tmp_dir: _write_raster(result, tmp_dir, fmt),
            fmt,
        )

    # Try vector
    try:
        import dask_geopandas

        if isinstance(result, dask_geopandas.GeoDataFrame):
            result = result.compute()
    except ImportError:
        pass

    try:
        import geopandas as gpd
        import pandas as pd

        if isinstance(result, gpd.GeoDataFrame):
            if fmt == "CHAPCSV":
                return _sync_export_response(
                    lambda tmp_dir: _write_tabular_export(
                        result.drop(columns="geometry", errors="ignore"),
                        tmp_dir,
                        fmt,
                        options,
                    ),
                    fmt,
                    "text/csv",
                )
            if fmt == "DHIS2JSON":
                frame = pd.DataFrame(result.drop(columns="geometry", errors="ignore"))
                return _json_tabular_payload_response(frame, options)
            if fmt == "GEOJSON":
                if result.crs is not None and result.crs.to_epsg() != 4326:
                    result = result.to_crs("EPSG:4326")
                return Response(content=result.to_json(), media_type="application/geo+json")
            if fmt in _VECTOR_FORMATS:
                with tempfile.TemporaryDirectory() as tmp:
                    from pathlib import Path

                    output = _write_vector(result, Path(tmp), fmt)
                    if output is None:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Format '{fmt}' produced no output",
                        )
                    data = Path(output).read_bytes()
                    mime = _VECTOR_FORMATS[fmt][1]
                    return Response(content=data, media_type=mime)
            info = {
                "type": "vector",
                "features": len(result),
                "columns": list(result.columns),
            }
            return JSONResponse(info)
    except ImportError:
        pass

    return result


def _coord_summary(coord: Any) -> Any:
    """Compact coordinate summary safe for JSON serialization."""
    import numpy as np

    vals = coord.values
    if vals.ndim == 0:
        v = vals.item()
        return str(v) if not isinstance(v, (int, float, bool)) else v
    if vals.size == 0:
        return []

    def _fmt(v: Any) -> Any:
        return v.item() if isinstance(v, np.generic) else str(v)

    if vals.size <= 4:
        return [_fmt(v) for v in vals]
    return {"first": _fmt(vals[0]), "last": _fmt(vals[-1]), "size": int(vals.size)}


# ---------------------------------------------------------------------------
# Workflows (user-defined processes)  /process_graphs
# ---------------------------------------------------------------------------


@udp_router.get("", response_model=WorkflowListResponse)
def list_workflows() -> WorkflowListResponse:
    """Return all stored workflows (user-defined processes)."""
    return workflow_store.list_workflows()


@udp_router.get("/{process_graph_id}", response_model=WorkflowRecord)
def get_workflow(process_graph_id: str) -> WorkflowRecord:
    """Return one workflow (user-defined process)."""
    record = workflow_store.get_workflow(process_graph_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Process graph '{process_graph_id}' not found")
    return record


@udp_router.put("/{process_graph_id}", status_code=200)
def put_workflow(process_graph_id: str, body: dict[str, Any] = Body(...)) -> WorkflowRecord:
    """Store or replace a workflow (user-defined process)."""
    if process_graph_id in _reserved_process_ids():
        raise HTTPException(
            status_code=400,
            detail=f"Process graph id '{process_graph_id}' conflicts with a predefined process",
        )
    return workflow_store.put_workflow(process_graph_id, body)


@udp_router.delete("/{process_graph_id}", status_code=204)
def delete_workflow(process_graph_id: str) -> Response:
    """Delete a workflow (user-defined process)."""
    found = workflow_store.delete_workflow(process_graph_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Process graph '{process_graph_id}' not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_base(request: Request) -> str:
    base_url = os.getenv("CLIMATE_SERVICE_BASE_URL")
    if base_url:
        return base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def _reserved_process_ids() -> set[str]:
    """Return process ids that must not be replaced by a workflow."""
    return {proc["id"] for proc in processes_service.list_openeo_processes() if isinstance(proc.get("id"), str)}
