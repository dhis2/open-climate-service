"""FastAPI routes for table datasets (CLIM-1553)."""

import csv
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response

from open_geo_service.ingestions import services as ingestion_services
from open_geo_service.ingestions.schemas import ArtifactRecord
from open_geo_service.shared.geoparquet import PARQUET_MEDIA_TYPE
from open_geo_service.tables import store

router = APIRouter()


@router.get("/{dataset_id}/rows", response_model=None)
def get_table_rows(
    dataset_id: str,
    collection: str | None = Query(default=None, description="Feature collection id, e.g. districts."),
    feature: str | None = Query(default=None, description="Only the rows of this feature id."),
    start: date | None = Query(default=None, description="Earliest period to return, by its first day, inclusive."),
    end: date | None = Query(default=None, description="Latest period to return, by its first day, inclusive."),
    reducer: str | None = Query(default=None, description="Only rows whose reducer column holds this, e.g. mean."),
    weights: str | None = Query(
        default=None, description="Only rows whose weights column holds this, e.g. population."
    ),
    output_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> Response:
    """Return a published table dataset's rows, as JSON or CSV.

    One row per key of the table's schema, with its current values; `written_at` is when they
    last changed. In a table with a period column, `period` names the period as raster datasets
    do (`2026-01`, `2026-W01`, `2026`), and `period_end` is its last day, beside the stored first
    day. A missing value is null in JSON and empty in CSV. A filter on a column the table does
    not have is refused with 400.
    """
    record = _published_table_or_404(dataset_id)
    equals = {name: value for name, value in (("reducer", reducer), ("weights", weights)) if value is not None}
    try:
        table = store.read_rows(
            record, collection_id=collection, feature_id=feature, start=start, end=end, equals=equals
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    period_column = store.schema_of(record).role("period")
    rows = _labelled(table.to_pylist(), period_column, record.period_type)
    if output_format == "csv":
        return Response(
            content=_csv(table.column_names, rows, period_column),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{dataset_id}.csv"'},
        )
    return JSONResponse(jsonable_encoder({"dataset_id": dataset_id, "items": rows}))


@router.get("/{dataset_id}/data.parquet", response_class=FileResponse)
# HEAD as well, for a client that sizes the file before reading ranges of it. Kept out of the
# OpenAPI schema so the operation is listed once.
@router.head("/{dataset_id}/data.parquet", response_class=FileResponse, include_in_schema=False)
def download_table(dataset_id: str) -> FileResponse:
    """Serve the Parquet file holding a published table dataset's current values.

    The `data` asset of its STAC collection: the one file the store keeps, as the record names it.
    """
    record = _published_table_or_404(dataset_id)
    path = Path(record.asset_paths[0]) if record.asset_paths else None
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Table dataset '{dataset_id}' is registered but its file is missing"
        )
    return FileResponse(path, media_type=PARQUET_MEDIA_TYPE, filename=f"{dataset_id}.parquet")


def _published_table_or_404(dataset_id: str) -> ArtifactRecord:
    record = ingestion_services.published_table_artifacts_by_dataset().get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Table dataset '{dataset_id}' not found")
    return record


def _labelled(rows: list[dict[str, Any]], period_column: str | None, period_type: str | None) -> list[dict[str, Any]]:
    """Each row with `period` and `period_end` around its stored period start; computed here, never stored."""
    labelled: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for name, value in row.items():
            if name == period_column and value is not None:
                item["period"] = store.period_label(value, period_type)
                item[name] = value
                item["period_end"] = store.end_of_period(value, period_type)
            else:
                item[name] = value
        labelled.append(item)
    return labelled


def _csv(names: list[str], rows: list[dict[str, Any]], period_column: str | None) -> str:
    """One line per row, ISO dates and times, durations in seconds, and an empty cell for NULL."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    columns = [
        label for name in names for label in (("period", name, "period_end") if name == period_column else (name,))
    ]
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(name)) for name in columns])
    return out.getvalue()


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    return value
