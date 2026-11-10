"""Table datasets as a pipeline destination: a workflow's value per feature and period, kept in OCS (CLIM-1554).

The `table` destination renders a workflow's vector cube to the rows of a table dataset and
delivers them through the one writer, `tables.store.write_rows`. The payload is those rows as a
Parquet file, checked against the store when rendered, so a dry run reports what the writer
would accept and a delivery appends exactly what was rendered.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from open_climate_service.exports.base import (
    BaseExportPlugin,
    DeliveryContext,
    DestinationCheck,
    DestinationContext,
    PipelineSettingsForm,
    RenderedExport,
    SettingsFormField,
)
from open_climate_service.exports.report import ExportOutcome, ExportReport
from open_climate_service.shared.geoparquet import PARQUET_MEDIA_TYPE
from open_climate_service.shared.urls import SEGMENT_SAFE_ID_PATTERN
from open_climate_service.shared.vectors import GEOMETRY_FIELD, SPATIAL_REDUCER_ATTR

_PERIOD_FIELD = "t"
_MAPPING_FIELDS = ("dataset_id", "collection_id", "name", "period_type", "unit")
# Dataset-level facts travel in the Parquet footer, so `send` needs nothing but the payload.
_NAME_KEY = b"ocs:dataset_name"


class TableDestinationSettings(BaseModel):
    """The table dataset a pipeline writes its rows to."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(
        pattern=SEGMENT_SAFE_ID_PATTERN.pattern,
        description="Id of the table dataset the rows are written to, created on the first delivery.",
    )
    name: str | None = Field(default=None, description="Display name of the table dataset. Defaults to its id.")


class TableExportPlugin(BaseExportPlugin):
    """Write a workflow's per-feature result into a table dataset."""

    id = "table"
    format = "TABLEPARQUET"
    extension = ".parquet"
    media_type = PARQUET_MEDIA_TYPE
    version = "1"
    supports_delivery = True
    supports_dry_run = True
    period_from_result = True
    # Rows carry the `written_at` they were rendered at, so a payload sent again repeats the same
    # rows: current values are unchanged, and history lists them once.
    redelivery_safe = True

    # --- delivery target: the instance's own table store -------------------------------------

    def target_binding(self, references: dict[str, str]) -> dict[str, str] | None:
        return {"store": "tables"}

    def delivery_target(self, binding: dict[str, str]) -> Any:
        return binding["store"]

    # --- pipeline destination hooks ----------------------------------------------------------

    def destination_settings_schema(self) -> dict[str, Any] | None:
        return TableDestinationSettings.model_json_schema()

    def pipeline_settings_form(self) -> PipelineSettingsForm | None:
        return PipelineSettingsForm(
            title="Table dataset",
            description=(
                "Keep the workflow's value per feature and period in this instance, as a table dataset "
                "served at /tables/{id}/rows. Revised values are added as new rows; history is kept."
            ),
            fields=(
                SettingsFormField(name="dataset_id", label="Dataset id", help_text="Letters, digits, '.', '_', '-'."),
                SettingsFormField(name="name", label="Name", help_text="Optional display name."),
            ),
        )

    def export_definition(self, settings: dict[str, Any], context: DestinationContext) -> dict[str, Any]:
        parsed = _parse_settings(settings)
        if len(context.feature_collections) != 1:
            raise ValueError(
                "a table destination needs exactly one feature collection, the workflow binds "
                f"{len(context.feature_collections)}"
            )
        source = context.datasets[0] if len(context.datasets) == 1 else None
        # No period_type: the period is the result's own cadence, which a temporal aggregation
        # makes coarser than the source's, so the source's cadence would be wrong to declare.
        definition = {
            "dataset_id": parsed.dataset_id,
            "collection_id": context.feature_collections[0],
            "name": parsed.name,
            "unit": source.units if source else None,
        }
        return {key: value for key, value in definition.items() if value is not None}

    def check_destination(self, settings: dict[str, Any], context: DestinationContext) -> list[DestinationCheck]:
        from open_climate_service.features.services import registered_collections
        from open_climate_service.tables import store

        try:
            parsed = _parse_settings(settings)
        except ValueError as exc:
            return [DestinationCheck(id="table_settings", status="fail", message=str(exc))]
        checks: list[DestinationCheck] = []
        try:
            store.require_table_id(parsed.dataset_id)
            checks.append(
                DestinationCheck(id="table_dataset", status="pass", message=f"rows go to '{parsed.dataset_id}'")
            )
        except ValueError as exc:
            checks.append(DestinationCheck(id="table_dataset", status="fail", message=str(exc)))
        collections = context.feature_collections
        if len(collections) != 1:
            checks.append(
                DestinationCheck(
                    id="feature_collection",
                    status="fail",
                    message=(
                        "a table destination needs exactly one feature collection, the workflow binds "
                        f"{len(collections)}"
                    ),
                )
            )
        elif collections[0] not in registered_collections():
            checks.append(
                DestinationCheck(
                    id="feature_collection",
                    status="fail",
                    message=f"'{collections[0]}' is not a registered feature collection; fetch it first",
                )
            )
        else:
            checks.append(
                DestinationCheck(
                    id="feature_collection", status="pass", message=f"one row per feature of '{collections[0]}'"
                )
            )
        return checks

    def preview(
        self,
        content: bytes,
        settings: dict[str, Any],
        context: DestinationContext,
        labels: Mapping[str, Mapping[str, str]] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """The first rows by period, then feature, each feature shown by name when it has one."""
        from open_climate_service.tables.store import period_label

        try:
            table, _name, period_type = _read_payload(content)
        except Exception:
            return []
        rows = table.to_pylist()
        rows.sort(key=lambda row: (row["period_start"], row["feature_id"]))
        sample: list[dict[str, Any]] = []
        for row in rows[:limit]:
            name = row.get("feature_name")
            sample.append(
                {
                    "feature": {"value": row["feature_id"], "label": name} if name else row["feature_id"],
                    "period": period_label(row["period_start"], period_type),
                    "value": row["value"],
                    "reducer": row["reducer"],
                }
            )
        return sample

    # --- render and send ---------------------------------------------------------------------

    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(mapping) - set(_MAPPING_FIELDS))
        if unknown:
            raise ValueError(f"Unsupported table mapping fields {unknown}")
        for field in ("dataset_id", "collection_id"):
            value = mapping.get(field)
            if not isinstance(value, str) or not SEGMENT_SAFE_ID_PATTERN.fullmatch(value):
                raise ValueError(f"table mapping requires {field}: letters, digits, '.', '_' or '-'")
        for field in ("name", "period_type", "unit"):
            if field in mapping and not isinstance(mapping[field], str):
                raise ValueError(f"table mapping {field} must be a string")
        return dict(mapping)

    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        """The result's rows, checked by the table store's writer, as a Parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from open_climate_service.features.services import feature_names
        from open_climate_service.shared.time import cadence_of
        from open_climate_service.tables import store

        # The result's own cadence: a graph that aggregates in time makes it coarser than its source.
        period_type = cadence_of(data) or mapping.get("period_type")
        frame, reducer = _frame(data)
        id_field, value_field = _fields(frame)
        names = feature_names(mapping["collection_id"])
        rows = [
            {
                "feature_id": str(record[id_field]),
                "feature_name": names.get(str(record[id_field])),
                "period_start": _period(record[_PERIOD_FIELD]),
                "value": _value(record[value_field]),
                "reducer": reducer,
                "unit": mapping.get("unit"),
            }
            for record in frame.to_dict(orient="records")
        ]
        table = store.check_rows(
            dataset_id=mapping["dataset_id"],
            collection_id=mapping["collection_id"],
            rows=rows,
            period_type=period_type,
            written_at=datetime.now(UTC),
        )
        # The store's metadata carries the period type; the payload adds the dataset name.
        metadata = {**(table.schema.metadata or {}), _NAME_KEY: mapping.get("name", mapping["dataset_id"]).encode()}
        sink = pa.BufferOutputStream()
        pq.write_table(table.replace_schema_metadata(metadata), sink)
        # The period type the store settled on, which may be the dataset's own.
        resolved = metadata.get(store.PERIOD_TYPE_KEY)
        period_type = resolved.decode() if resolved else None
        return RenderedExport(
            content=sink.getvalue().to_pybytes(),
            record_count=table.num_rows,
            periods=tuple(
                store.period_label(start, period_type)
                for start in sorted(set(table.column("period_start").to_pylist()))
            ),
        )

    def send(
        self,
        payload: bytes,
        target: Any,
        *,
        dry_run: bool = False,
        context: DeliveryContext | None = None,
    ) -> ExportReport:
        """Append the rendered rows, or with *dry_run* only check them; a refused write writes nothing."""
        import hashlib

        from open_climate_service.tables import store

        created_at = datetime.now(UTC).isoformat()
        table, name, period_type = _read_payload(payload)
        rows = table.drop_columns(["written_at"]).to_pylist()
        arguments: dict[str, Any] = {
            "dataset_id": table.column("dataset_id")[0].as_py(),
            "collection_id": table.column("collection_id")[0].as_py(),
            "rows": rows,
            "written_at": table.column("written_at")[0].as_py(),
        }
        outcome = ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS
        message: str | None = None
        try:
            if dry_run:
                store.check_rows(**arguments, period_type=period_type)
            else:
                store.write_rows(**arguments, dataset_name=name, period_type=period_type)
        except ValueError as exc:
            outcome, message = ExportOutcome.REJECTED, str(exc)
        accepted = len(rows) if outcome is not ExportOutcome.REJECTED else 0
        return ExportReport(
            plugin_id=self.id,
            dry_run=dry_run,
            outcome=outcome,
            message=message,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=len(rows),
            imported=0 if dry_run else accepted,
            ignored=len(rows) - accepted,
            created_at=created_at,
            finished_at=datetime.now(UTC).isoformat(),
        )


def _parse_settings(settings: dict[str, Any]) -> TableDestinationSettings:
    try:
        return TableDestinationSettings.model_validate(settings)
    except ValidationError as exc:
        error = exc.errors()[0]
        location = ".".join(str(part) for part in error["loc"])
        raise ValueError(f"{location + ': ' if location else ''}{error['msg']}") from None


def _read_payload(payload: bytes) -> tuple[Any, str | None, str | None]:
    """The rows of a rendered payload, with the dataset name and period type from its footer."""
    import pyarrow.parquet as pq

    from open_climate_service.tables.store import PERIOD_TYPE_KEY

    table = pq.read_table(io.BytesIO(payload))
    metadata = table.schema.metadata or {}
    name, period_type = metadata.get(_NAME_KEY), metadata.get(PERIOD_TYPE_KEY)
    return (
        table.replace_schema_metadata(None),
        name.decode() if name else None,
        period_type.decode() if period_type else None,
    )


def _frame(data: Any) -> tuple[Any, str | None]:
    """The result as one long table, and the statistic it was reduced with when it says."""
    import pandas as pd
    import xarray as xr

    if isinstance(data, pd.DataFrame):
        return data.copy(), data.attrs.get(SPATIAL_REDUCER_ATTR)
    if isinstance(data, xr.DataArray):
        reducer = data.attrs.get(SPATIAL_REDUCER_ATTR)
        data = data.to_dataset(name=data.name or "value")
    elif isinstance(data, xr.Dataset):
        reducer = data.attrs.get(SPATIAL_REDUCER_ATTR) or next(
            (var.attrs.get(SPATIAL_REDUCER_ATTR) for var in data.data_vars.values()), None
        )
    else:
        raise ValueError("a table destination takes a vector cube or a table, one value per feature and period")
    scalars = [name for name, coordinate in data.coords.items() if not coordinate.dims]
    return data.drop_vars(scalars).to_dataframe().reset_index(), reducer


def _fields(frame: Any) -> tuple[str, str]:
    """The feature id column and the one value column of *frame*."""
    from open_climate_service.exports.tabular import non_value_fields
    from open_climate_service.shared.vectors import feature_id_field

    id_field = feature_id_field(frame, GEOMETRY_FIELD)
    if id_field not in frame.columns or _PERIOD_FIELD not in frame.columns:
        raise ValueError("a table destination needs a result with feature ids and a time dimension 't'")
    excluded = non_value_fields(frame) | {id_field, _PERIOD_FIELD}
    values = [str(column) for column in frame.columns if str(column) not in excluded]
    if len(values) != 1:
        raise ValueError(
            f"a table destination takes one value per feature and period; the result has {values or 'none'}"
        )
    return id_field, values[0]


def _period(value: Any) -> date:
    import pandas as pd

    return pd.Timestamp(value).date()


def _value(value: Any) -> Any:
    """A number (numpy's too) as a float; the writer reads NaN as missing and refuses anything else."""
    import numbers

    import pandas as pd

    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real):
        return float(value)
    return None if value is pd.NA or value is pd.NaT else value


plugin = TableExportPlugin()
