"""Strict single-series DHIS2 rendering for named export mappings."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from decimal import Decimal
from typing import Any

from open_climate_service.exports.base import BaseExportPlugin, RenderedExport
from open_climate_service.exports.report import ExportOutcome, ExportReport
from open_climate_service.exports.tabular import (
    _build_dhis2_json_payload,
    _is_nullish,
    _normalise_period_type,
    _select_dhis2_value_field,
    _to_dhis2_period_string,
)


class Dhis2ExportPlugin(BaseExportPlugin):
    """Map one prepared aggregate series to one DHIS2 data element."""

    id = "dhis2"
    format = "DHIS2JSON"
    extension = ".json"
    media_type = "application/json"
    version = "1"
    supports_delivery = True

    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        allowed = {"series", "period_type", "org_unit_field", "period_field", "aggregation"}
        if set(mapping) - allowed:
            raise ValueError("Unsupported DHIS2 mapping fields")
        period_type = mapping.get("period_type")
        if not isinstance(period_type, str) or not period_type.strip():
            raise ValueError("DHIS2 mapping requires period_type; temporal aggregation must happen upstream")
        kind = _normalise_period_type(period_type)
        series = mapping.get("series")
        if not isinstance(series, list) or len(series) != 1 or not isinstance(series[0], dict):
            raise ValueError("This phase supports exactly one DHIS2 series per export")
        entry = series[0]
        if set(entry) - {"select", "data_element", "category_option_combo", "attribute_option_combo"}:
            raise ValueError("Unsupported DHIS2 series fields")
        select = entry.get("select", {})
        if not isinstance(select, dict) or set(select) - {"variable"}:
            raise ValueError("Single-series select supports only an optional 'variable' name")
        if "variable" in select and (not isinstance(select["variable"], str) or not select["variable"].strip()):
            raise ValueError("select.variable must be a non-empty variable name")
        _uid(entry.get("data_element"), "series.data_element")
        for field in ("category_option_combo", "attribute_option_combo"):
            if field in entry:
                _uid(entry[field], f"series.{field}")
        for field, default in (("org_unit_field", "geometry"), ("period_field", "t")):
            value = mapping.get(field, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty field name")
        if mapping.get("org_unit_field", "geometry") == mapping.get("period_field", "t"):
            raise ValueError("Organisation unit and period fields must be distinct")
        if "aggregation" in mapping and mapping["aggregation"] not in ("mean", "sum", "min", "max"):
            raise ValueError("aggregation must be mean, sum, min, or max; it declares upstream computation")
        return {**mapping, "period_type": kind, "series": [{**entry, "select": select}]}

    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        import numpy as np
        import pandas as pd
        import xarray as xr

        entry = mapping["series"][0]
        org_field = mapping.get("org_unit_field", "geometry")
        period_field = mapping.get("period_field", "t")
        kind = mapping["period_type"]
        variable = entry["select"].get("variable")
        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name=data.name or "result")
        if isinstance(data, xr.Dataset):
            if variable is not None:
                if variable not in data.data_vars:
                    raise ValueError(f"Selected variable '{variable}' is not present in the result")
                data = data[[variable]]
            if set(data.dims) - {org_field, period_field}:
                raise ValueError("DHIS2 export requires aggregates with only organisation-unit and period dimensions")
            declared_period = data.attrs.get("period_type")
            if declared_period is not None and _normalise_period_type(str(declared_period)) != kind:
                raise ValueError("Result period_type differs from the mapping; aggregate temporally before exporting")
            frame = data.to_dataframe().reset_index()
        elif isinstance(data, pd.DataFrame):
            frame = pd.DataFrame(data).copy()
            if variable is not None:
                required = [org_field, period_field, variable]
                if not all(field in frame.columns for field in required):
                    raise ValueError("Selected variable or identity fields are missing from the result")
                frame = frame[required]
        else:
            raise ValueError("DHIS2 export requires an xarray aggregate or a pandas/GeoPandas table")
        if org_field not in frame or period_field not in frame:
            raise ValueError("DHIS2 result is missing organisation-unit or period fields")
        if not frame.columns.is_unique:
            raise ValueError("DHIS2 result contains duplicate column names")
        value_field = _select_dhis2_value_field(frame, org_field, period_field)
        keys: set[tuple[str, str]] = set()
        for index, record in enumerate(frame.to_dict(orient="records")):
            org = _uid(record[org_field], f"row {index} organisation unit (feature.id)")
            period = _to_dhis2_period_string(record[period_field], kind)
            _validate_period(period, kind)
            key = (org, period)
            if key in keys:
                raise ValueError(f"Duplicate DHIS2 value for organisation unit '{org}' and period '{period}'")
            keys.add(key)
            value = record[value_field]
            if not _is_nullish(value) and (
                not isinstance(value, (int, float, Decimal, np.integer, np.floating, bool, np.bool_))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Row {index} must contain a finite scalar numeric value")
        options = {
            "data_element_id": entry["data_element"],
            "org_unit_field": org_field,
            "period_field": period_field,
            "period_type": kind,
        }
        if "category_option_combo" in entry:
            options["category_option_combo"] = entry["category_option_combo"]
        payload = _build_dhis2_json_payload(frame, options)
        values = payload["dataValues"]
        if "attribute_option_combo" in entry:
            for value in values:
                value["attributeOptionCombo"] = entry["attribute_option_combo"]
        return RenderedExport(
            content=json.dumps(payload, allow_nan=False).encode(),
            record_count=len(values),
            skipped_count=len(frame) - len(values),
            periods=tuple(sorted({value["period"] for value in values})),
        )

    def send(self, payload: bytes, target: Any, *, dry_run: bool = False) -> ExportReport:
        """POST the rendered dataValueSet and summarize the import result."""
        import hashlib
        from contextlib import closing

        from open_climate_service.exports.dhis2 import get_connection
        from open_climate_service.shared.time import utc_now

        if not isinstance(target, str) or not target.strip():
            raise ValueError("DHIS2 delivery requires a named connection")
        created_at = utc_now().isoformat()
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            body = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise ValueError("DHIS2 delivery payload is not valid JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("dataValues"), list):
            raise ValueError("DHIS2 delivery payload must be a dataValueSet")
        submitted = len(body["dataValues"])

        with closing(get_connection(target)) as client:
            params: dict[str, str] = {"importStrategy": "CREATE_AND_UPDATE"}
            if dry_run:
                params["dryRun"] = "true"
            try:
                response = client.post("/api/dataValueSets", json=body, params=params)
            except Exception as exc:
                # The POST may have reached DHIS2 before failing (e.g. timeout).
                # Record an unknown outcome rather than fabricating a rejection.
                return ExportReport(
                    plugin_id=self.id,
                    connection_id=target,
                    dry_run=dry_run,
                    outcome=ExportOutcome.UNKNOWN,
                    message=f"Transport error before the import result could be read: {type(exc).__name__}",
                    payload_sha256=payload_sha256,
                    submitted=submitted,
                    created_at=created_at,
                    finished_at=utc_now().isoformat(),
                )
        return build_dhis2_report(
            _response_status(response),
            _response_json(response),
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            payload_sha256=payload_sha256,
            submitted=submitted,
            created_at=created_at,
        )


def build_dhis2_report(
    status_code: int,
    summary: dict[str, Any],
    *,
    plugin_id: str,
    connection_id: str,
    dry_run: bool,
    payload_sha256: str,
    submitted: int,
    created_at: str,
) -> ExportReport:
    """Translate a DHIS2 dataValueSets import response into an ExportReport."""
    from open_climate_service.shared.time import utc_now

    finished_at = utc_now().isoformat()
    status = summary.get("status")
    message = summary.get("message") or summary.get("httpStatus")
    import_count = summary.get("importCount") or {}
    if not isinstance(import_count, dict):
        import_count = {}
    conflicts = summary.get("conflicts") or []
    if not isinstance(conflicts, list):
        conflicts = []
    remote_task_ids: list[str] = []
    task_id = summary.get("id") or summary.get("taskId")
    if isinstance(task_id, str) and task_id:
        remote_task_ids.append(task_id)

    def count(key: str) -> int:
        value = import_count.get(key, 0)
        return value if isinstance(value, int) else 0

    base: dict[str, Any] = dict(
        plugin_id=plugin_id,
        connection_id=connection_id,
        dry_run=dry_run,
        payload_sha256=payload_sha256,
        submitted=submitted,
        imported=count("imported"),
        updated=count("updated"),
        ignored=count("ignored"),
        deleted=count("deleted"),
        conflicts=conflicts,
        remote_task_ids=remote_task_ids,
        created_at=created_at,
        finished_at=finished_at,
    )

    if dry_run:
        # A dry run reached DHIS2 validation; the summary describes what would
        # happen, not persisted writes.
        return ExportReport(outcome=ExportOutcome.DRY_RUN, message=message, **base)

    if not 200 <= status_code < 300:
        return ExportReport(outcome=ExportOutcome.REJECTED, message=message or f"HTTP {status_code}", **base)

    if status == "ERROR":
        return ExportReport(outcome=ExportOutcome.REJECTED, message=message, **base)
    if status == "WARNING" or conflicts:
        return ExportReport(outcome=ExportOutcome.PARTIAL, message=message, **base)
    if status == "SUCCESS":
        return ExportReport(outcome=ExportOutcome.SUCCESS, message=message, **base)
    # A 2xx without a recognizable import summary cannot be trusted as success.
    return ExportReport(outcome=ExportOutcome.UNKNOWN, message="Unrecognized import summary", **base)


def _response_status(response: Any) -> int:
    return response.status_code if isinstance(getattr(response, "status_code", None), int) else 0


def _response_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _uid(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{10}", value):
        raise ValueError(f"{field} must be a DHIS2 UID; supply real feature IDs instead of positional labels")
    return value


def _validate_period(period: str, kind: str) -> None:
    try:
        if kind == "weekly":
            year, week = period.split("W")
            dt.date.fromisocalendar(int(year), int(week), 1)
        elif kind == "quarterly":
            year, quarter = period.split("Q")
            dt.date(int(year), (int(quarter) - 1) * 3 + 1, 1)
        else:
            pattern = {"daily": "%Y%m%d", "monthly": "%Y%m", "yearly": "%Y"}[kind]
            dt.datetime.strptime(period, pattern)
    except ValueError:
        raise ValueError(f"Invalid DHIS2 {kind} period '{period}'") from None
