"""FastAPI routes for table datasets (CLIM-1553)."""

import csv
import io
from datetime import date, datetime, timedelta
from typing import Any, Literal

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
    start: date | None = Query(default=None, description="Earliest period_start to return, inclusive."),
    end: date | None = Query(default=None, description="Latest period_start to return, inclusive."),
    reducer: str | None = Query(default=None, description="Only rows computed with this reducer, e.g. mean."),
    weights: str | None = Query(default=None, description="Only rows weighted by this dataset, e.g. population."),
    output_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> Response:
    """Return a published table dataset's rows, as JSON or CSV.

    One row per feature, period and value key, with its current value; `written_at` is when it
    last changed. `period` names the period as raster datasets do (`2026-01`, `2026-W01`,
    `2026`); `period_start` and `period_end` are its first and last day. A missing value is null
    in JSON and empty in CSV.
    """
    record = _published_table_or_404(dataset_id)
    table = store.read_rows(
        record,
        collection_id=collection,
        start=start,
        end=end,
        reducer=reducer,
        weights=weights,
    )
    rows = _labelled(table.to_pylist(), record.period_type)
    if output_format == "csv":
        return Response(
            content=_csv(rows),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{dataset_id}.csv"'},
        )
    return JSONResponse(jsonable_encoder({"dataset_id": dataset_id, "items": rows}))


def _published_table_or_404(dataset_id: str) -> ArtifactRecord:
    record = ingestion_services.published_table_artifacts_by_dataset().get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Table dataset '{dataset_id}' not found")
    return record


def _labelled(rows: list[dict[str, Any]], period_type: str | None) -> list[dict[str, Any]]:
    """Each row with `period` and `period_end` around the stored `period_start`; computed here, never stored."""
    labelled: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for name, value in row.items():
            if name == "period_start":
                item["period"] = store.period_label(value, period_type)
                item["period_start"] = value
                item["period_end"] = store.end_of_period(value, period_type)
            else:
                item[name] = value
        labelled.append(item)
    return labelled


def _csv(rows: list[dict[str, Any]]) -> str:
    """One line per row, ISO dates and times, lead_time in seconds, and an empty cell for NULL."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    columns = [
        label
        for name in store.SCHEMA.names
        for label in (("period", "period_start", "period_end") if name == "period_start" else (name,))
    ]
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row[name]) for name in columns])
    return out.getvalue()


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    return value
