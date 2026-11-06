"""FastAPI routes for table datasets (CLIM-1553)."""

import csv
import io
from datetime import date, datetime, timedelta
from typing import Literal

import pyarrow as pa
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactRecord
from open_climate_service.tables import store

router = APIRouter()


@router.get("/{dataset_id}/rows", response_model=None)
def get_table_rows(
    dataset_id: str,
    collection: str | None = Query(default=None, description="Feature collection id, e.g. districts."),
    start: date | None = Query(default=None, description="First period start to return, inclusive."),
    end: date | None = Query(default=None, description="Last period start to return, inclusive."),
    reducer: str | None = Query(default=None, description="Only rows computed with this reducer, e.g. mean."),
    weights: str | None = Query(default=None, description="Only rows weighted by this dataset, e.g. population."),
    history: bool = Query(
        default=False,
        description="Return every row ever written, not only the current value (latest written_at) per key.",
    ),
    output_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> Response:
    """Return a published table dataset's rows, as JSON or CSV.

    One row per feature, period and value key. A missing value is null in JSON and empty in CSV.
    """
    table = store.read_rows(
        _published_table_or_404(dataset_id),
        collection_id=collection,
        start=start,
        end=end,
        reducer=reducer,
        weights=weights,
        history=history,
    )
    if output_format == "csv":
        return Response(
            content=_csv(table),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{dataset_id}.csv"'},
        )
    return JSONResponse(jsonable_encoder({"dataset_id": dataset_id, "history": history, "items": table.to_pylist()}))


def _published_table_or_404(dataset_id: str) -> ArtifactRecord:
    record = ingestion_services.published_table_artifacts_by_dataset().get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Table dataset '{dataset_id}' not found")
    return record


def _csv(table: pa.Table) -> str:
    """One line per row, ISO dates and times, lead_time in seconds, and an empty cell for NULL."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(table.column_names)
    for row in table.to_pylist():
        writer.writerow([_cell(value) for value in row.values()])
    return out.getvalue()


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    return value
