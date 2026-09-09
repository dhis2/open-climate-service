"""DHIS2 rendering for named single- and multi-series export mappings."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from decimal import Decimal
from typing import Any

from open_climate_service.exports.base import BaseExportPlugin, DeliveryContext, RenderedExport
from open_climate_service.exports.report import ExportOutcome, ExportReport, merge_chunk_reports
from open_climate_service.exports.tabular import (
    _is_nullish,
    _normalise_period_type,
    _to_dhis2_period_string,
    _to_dhis2_value_string,
)


class _RetryableTransportError(Exception):
    """A connection failure that occurred before the request reached DHIS2."""


def _is_retryable_transport_error(exc: BaseException) -> bool:
    """Return True for failures that occur before the request is sent.

    A refused connection, DNS failure, or connect timeout never reached DHIS2, so
    the chunk can be safely retried. Read/write timeouts happen after the request
    may have reached the server and stay ``unknown``.
    """
    import httpx

    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol))


class Dhis2ExportPlugin(BaseExportPlugin):
    """Map prepared aggregate series to DHIS2 data elements."""

    id = "dhis2"
    format = "DHIS2JSON"
    extension = ".json"
    media_type = "application/json"
    version = "1"
    supports_delivery = True
    # Upper bound for one dataValueSets POST; larger payloads are split into
    # deterministic chunks. Tune against the supported DHIS2 versions.
    max_chunk_size: int = 1000
    # Bounded backoff for reconciling an async import task after submission.
    max_poll_attempts: int = 10
    poll_backoff_base: float = 1.0

    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        allowed = {"series", "period_type", "org_unit_field", "period_field", "aggregation"}
        if set(mapping) - allowed:
            raise ValueError("Unsupported DHIS2 mapping fields")
        period_type = mapping.get("period_type")
        if not isinstance(period_type, str) or not period_type.strip():
            raise ValueError("DHIS2 mapping requires period_type; temporal aggregation must happen upstream")
        kind = _normalise_period_type(period_type)
        series = mapping.get("series")
        if not isinstance(series, list) or not series or not all(isinstance(entry, dict) for entry in series):
            raise ValueError("DHIS2 mapping requires a non-empty list of series mappings")
        validated_series: list[dict[str, Any]] = []
        for index, entry in enumerate(series):
            prefix = f"series[{index}]"
            if set(entry) - {"select", "data_element", "category_option_combo", "attribute_option_combo"}:
                raise ValueError(f"Unsupported {prefix} fields")
            select = entry.get("select", {})
            if not isinstance(select, dict) or set(select) - {"variable", "quantile"}:
                raise ValueError(f"{prefix}.select supports only 'variable' and 'quantile'")
            variable = select.get("variable")
            if "variable" in select and (not isinstance(variable, str) or not variable.strip()):
                raise ValueError(f"{prefix}.select.variable must be a non-empty variable name")
            quantile = select.get("quantile")
            if "quantile" in select and (
                isinstance(quantile, bool)
                or not isinstance(quantile, (int, float))
                or not math.isfinite(float(quantile))
            ):
                raise ValueError(f"{prefix}.select.quantile must be a finite number")
            validated: dict[str, Any] = {
                "data_element": _uid(entry.get("data_element"), f"{prefix}.data_element"),
                "select": dict(select),
            }
            for field in ("category_option_combo", "attribute_option_combo"):
                if field in entry:
                    validated[field] = _uid(entry[field], f"{prefix}.{field}")
            validated_series.append(validated)
        for field, default in (("org_unit_field", "geometry"), ("period_field", "t")):
            value = mapping.get(field, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty field name")
        if mapping.get("org_unit_field", "geometry") == mapping.get("period_field", "t"):
            raise ValueError("Organisation unit and period fields must be distinct")
        if "aggregation" in mapping and mapping["aggregation"] not in ("mean", "sum", "min", "max"):
            raise ValueError("aggregation must be mean, sum, min, or max; it declares upstream computation")
        # Default combo UIDs belong to target metadata, which pure rendering does
        # not fetch. Avoid treating an omitted combo as distinct from an explicit
        # one when several series target the same data element.
        for element in {entry["data_element"] for entry in validated_series}:
            group = [entry for entry in validated_series if entry["data_element"] == element]
            for field in ("category_option_combo", "attribute_option_combo"):
                if any(field in entry for entry in group) and not all(field in entry for entry in group):
                    raise ValueError(
                        f"Specify {field} on every series for data element '{element}' to resolve defaults"
                    )
        return {**mapping, "period_type": kind, "series": validated_series}

    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        import numpy as np

        entries = mapping["series"]
        org_field = mapping.get("org_unit_field", "geometry")
        period_field = mapping.get("period_field", "t")
        kind = mapping["period_type"]

        frame, value_columns = self._to_frame(data, org_field, period_field, kind)
        if org_field not in frame.columns or period_field not in frame.columns:
            raise ValueError("DHIS2 result is missing organisation-unit or period fields")
        if not frame.columns.is_unique:
            raise ValueError("DHIS2 result contains duplicate column names")

        residual_dims: list[str] = []
        if value_columns is not None:
            residual_dims = [
                str(column)
                for column in frame.columns
                if column not in {org_field, period_field}
                and str(column) not in value_columns
                and str(column) not in {"geometry", "spatial_ref", "index", "band", "bands"}
            ]

        data_values: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str, str, str, str]] = set()
        periods: set[str] = set()
        record_count = 0
        skipped_count = 0

        for entry in entries:
            select = entry.get("select", {})
            selected = frame
            if "quantile" in select:
                selected = self._select_quantile_rows(selected, select["quantile"])
            elif "quantile" in residual_dims:
                raise ValueError(
                    "DHIS2 export requires aggregates with only organisation-unit and period dimensions; "
                    "the result contains a 'quantile' dimension — select a quantile or aggregate before exporting"
                )
            unhandled = [column for column in residual_dims if column != "quantile"]
            if unhandled:
                raise ValueError(
                    "DHIS2 export requires aggregates with only organisation-unit and period dimensions; "
                    f"found residual dimensions {unhandled}"
                )
            column = self._resolve_series_column(selected, select, value_columns, org_field, period_field)
            for index, record in enumerate(selected.to_dict(orient="records")):
                org = _uid(record[org_field], f"row {index} organisation unit (feature.id)")
                period = _to_dhis2_period_string(record[period_field], kind)
                _validate_period(period, kind)
                value = record[column]
                if _is_nullish(value):
                    skipped_count += 1
                    continue
                if not isinstance(
                    value, (int, float, Decimal, np.integer, np.floating, bool, np.bool_)
                ) or not math.isfinite(value):
                    raise ValueError(f"Row {index} must contain a finite scalar numeric value")
                category_combo = entry.get("category_option_combo")
                attribute_combo = entry.get("attribute_option_combo")
                key = (entry["data_element"], org, period, category_combo or "", attribute_combo or "")
                if key in seen_keys:
                    raise ValueError(
                        f"Duplicate DHIS2 value for data element '{entry['data_element']}', "
                        f"organisation unit '{org}', and period '{period}'"
                    )
                seen_keys.add(key)
                item: dict[str, str] = {
                    "dataElement": entry["data_element"],
                    "orgUnit": org,
                    "period": period,
                    "value": _to_dhis2_value_string(value),
                }
                if category_combo is not None:
                    item["categoryOptionCombo"] = category_combo
                if attribute_combo is not None:
                    item["attributeOptionCombo"] = attribute_combo
                data_values.append(item)
                periods.add(period)
                record_count += 1

        return RenderedExport(
            content=json.dumps({"dataValues": data_values}, allow_nan=False).encode(),
            record_count=record_count,
            skipped_count=skipped_count,
            periods=tuple(sorted(periods)),
        )

    def _to_frame(self, data: Any, org_field: str, period_field: str, kind: str) -> tuple[Any, list[str] | None]:
        """Normalize an xarray object or DataFrame to one wide table.

        Returns ``(frame, value_columns)``. ``value_columns`` names the value
        columns when the input carried xarray data-variable metadata; it is
        ``None`` for a plain DataFrame, whose value columns are derived later by
        exclusion. A merged cube's synthetic ``__cubes__`` dimension is pivoted
        into one column per cube label so selectors address source names rather
        than the internal dimension name.
        """
        import pandas as pd
        import xarray as xr

        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name=data.name or "result")
        if isinstance(data, xr.Dataset):
            declared_period = data.attrs.get("period_type")
            if declared_period is not None and _normalise_period_type(str(declared_period)) != kind:
                raise ValueError("Result period_type differs from the mapping; aggregate temporally before exporting")
            value_columns = [str(name) for name in data.data_vars]
            if not value_columns:
                raise ValueError("DHIS2 result contains no data variables")
            frame = data.to_dataframe().reset_index()
            if "__cubes__" in frame.columns:
                if len(value_columns) != 1:
                    raise ValueError("Merged-cube results require exactly one value column")
                value_column = value_columns[0]
                index_columns = [column for column in frame.columns if column != value_column and column != "__cubes__"]
                if not index_columns:
                    raise ValueError("Merged-cube result has no organisation-unit or period columns")
                frame = frame.pivot(index=index_columns, columns="__cubes__", values=value_column).reset_index()
                frame.columns.name = None
                value_columns = [str(column) for column in frame.columns if column not in index_columns]
            return frame, value_columns
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(data).copy(), None
        raise ValueError("DHIS2 export requires an xarray aggregate or a pandas/GeoPandas table")

    def _resolve_series_column(
        self,
        frame: Any,
        select: dict[str, Any],
        value_columns: list[str] | None,
        org_field: str,
        period_field: str,
    ) -> str:
        """Return the single value column a series selector resolves to."""
        variable = select.get("variable")
        if variable is not None:
            if not isinstance(variable, str):
                raise ValueError("select.variable must be a string")
            if variable not in self._candidate_value_fields(frame, value_columns, org_field, period_field):
                raise ValueError(f"Selected variable '{variable}' is not present in the result")
            return variable
        fields = self._candidate_value_fields(frame, value_columns, org_field, period_field)
        if len(fields) != 1:
            raise ValueError(
                "DHIS2 export with an empty series selector requires exactly one value column "
                f"after excluding '{org_field}' and '{period_field}'; found {fields}"
            )
        return fields[0]

    @staticmethod
    def _candidate_value_fields(
        frame: Any, value_columns: list[str] | None, org_field: str, period_field: str
    ) -> list[str]:
        if value_columns is not None:
            return [column for column in value_columns if column in frame.columns]
        excluded = {org_field, period_field, "geometry", "spatial_ref", "index", "band", "bands"}
        excluded.add("quantile")
        return [
            str(column) for column in frame.columns if column not in excluded and not str(column).startswith("level_")
        ]

    @staticmethod
    def _select_quantile_rows(frame: Any, quantile: Any) -> Any:
        """Filter a table to one quantile along its ``quantile`` dimension."""
        if "quantile" not in frame.columns:
            raise ValueError("select.quantile requires a 'quantile' dimension in the result")
        mask = frame["quantile"] == quantile
        if not bool(mask.any()):
            raise ValueError(f"select.quantile value {quantile!r} is not present in the result")
        return frame.loc[mask]

    def send(
        self,
        payload: bytes,
        target: Any,
        *,
        dry_run: bool = False,
        context: DeliveryContext | None = None,
    ) -> ExportReport:
        """POST the rendered dataValueSet in resumable chunks and reconcile each.

        Chunking, checkpointing, and response interpretation stay inside the
        plugin. A timeout after POST cannot be reconciled, so it is recorded as
        ``unknown`` and never replayed automatically; a completed chunk is
        check pointed and skipped on restart.
        """
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
        values = body["dataValues"]
        chunks = self._chunk_values(values)

        reports: list[ExportReport] = []
        cancelled_early = False
        terminal_message: str | None = None
        with closing(get_connection(target)) as client:
            for index, chunk_values in enumerate(chunks):
                if context is not None and context.is_cancel_requested():
                    cancelled_early = True
                    break
                chunk_report = self._deliver_chunk(
                    client,
                    target,
                    chunk_values,
                    index=index,
                    dry_run=dry_run,
                    created_at=created_at,
                    context=context,
                )
                reports.append(chunk_report)
                if context is not None:
                    context.report_progress(index + 1, len(chunks), "Delivering export chunks")
                # A rejected chunk is a hard stop: later chunks would fail for the
                # same reason, and it must remain visible in the merged report.
                if chunk_report.outcome == ExportOutcome.REJECTED:
                    terminal_message = chunk_report.message
                    break

        finished_at = utc_now().isoformat()
        message: str | None
        if cancelled_early:
            message = f"Cancelled after {len(reports)} of {len(chunks)} chunks"
        else:
            message = terminal_message
        return merge_chunk_reports(
            reports,
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            payload_sha256=payload_sha256,
            created_at=created_at,
            finished_at=finished_at,
            cancelled_early=cancelled_early,
            message=message,
        )

    def _chunk_values(self, values: list[Any]) -> list[list[Any]]:
        """Split a dataValues list into deterministic, bounded chunks."""
        size = max(1, self.max_chunk_size)
        return [values[offset : offset + size] for offset in range(0, len(values), size)] or [[]]

    def _deliver_chunk(
        self,
        client: Any,
        target: str,
        chunk_values: list[Any],
        *,
        index: int,
        dry_run: bool,
        created_at: str,
        context: DeliveryContext | None,
    ) -> ExportReport:
        import hashlib

        chunk_payload = json.dumps({"dataValues": chunk_values}, allow_nan=False, separators=(",", ":")).encode()
        digest = hashlib.sha256(chunk_payload).hexdigest()
        checkpoint = self._load_chunk_checkpoint(context, index, digest)
        if checkpoint is not None:
            if checkpoint.connection_id != target or checkpoint.dry_run != dry_run:
                raise ValueError("Delivery checkpoint target or mode changed")
            if checkpoint.outcome == ExportOutcome.UNKNOWN and checkpoint.remote_task_ids:
                checkpoint = self._poll_task(
                    client,
                    checkpoint.remote_task_ids[0],
                    target=target,
                    dry_run=dry_run,
                    submitted=len(chunk_values),
                    payload_sha256=digest,
                    created_at=checkpoint.created_at,
                    context=context,
                )
                self._save_chunk_checkpoint(context, index, digest, checkpoint)
            return checkpoint
        # Commit intent before POST. A process death at any later point must not
        # turn an uncertain remote write into a fresh chunk on recovery.
        self._save_chunk_checkpoint(
            context,
            index,
            digest,
            ExportReport(
                plugin_id=self.id,
                connection_id=target,
                dry_run=dry_run,
                outcome=ExportOutcome.UNKNOWN,
                payload_sha256=digest,
                submitted=len(chunk_values),
                created_at=created_at,
                finished_at=created_at,
            ),
        )
        try:
            report = self._submit_chunk(
                client,
                target,
                chunk_values,
                dry_run=dry_run,
                created_at=created_at,
                chunk_sha256=digest,
                context=context,
                index=index,
            )
        except _RetryableTransportError as exc:
            # Nothing reached DHIS2. Record the failed attempt so successful
            # preceding chunks remain visible in the merged delivery report.
            from open_climate_service.shared.time import utc_now

            report = ExportReport(
                plugin_id=self.id,
                connection_id=target,
                dry_run=dry_run,
                outcome=ExportOutcome.REJECTED,
                message=f"Connection failed before submission: {exc}",
                payload_sha256=digest,
                submitted=0,
                created_at=created_at,
                finished_at=utc_now().isoformat(),
            )
        self._save_chunk_checkpoint(context, index, digest, report)
        return report

    def _load_chunk_checkpoint(self, context: DeliveryContext | None, index: int, digest: str) -> ExportReport | None:
        """Restore a chunk from its checkpoint, or None to send it fresh."""
        if context is None:
            return None
        state = context.load_checkpoint(self._chunk_checkpoint_key(index))
        if state is None:
            return None
        # Checkpoint providers can be external plugins without runtime type checks.
        if not isinstance(state, dict) or state.get("digest") != digest:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Delivery checkpoint is corrupt or chunk boundaries changed; refusing to resend")
        if state.get("status") == "completed":
            report = ExportReport.model_validate(state.get("report"))
            if report.payload_sha256 != digest or report.plugin_id != self.id:
                raise ValueError("Delivery checkpoint report does not match the chunk")
            return report
        if state.get("status") == "submitted_unknown":
            # A previous POST timed out without a reconcilable remote task ID.
            # Do not replay; record the uncertainty so the overall report is honest.
            from open_climate_service.shared.time import utc_now

            return ExportReport(
                plugin_id=self.id,
                connection_id=state.get("connection_id"),
                dry_run=bool(state.get("dry_run")),
                outcome=ExportOutcome.UNKNOWN,
                message="Previously submitted; the import result could not be reconciled",
                payload_sha256=digest,
                submitted=int(state.get("submitted") or 0),
                remote_task_ids=list(state.get("remote_task_ids") or []),
                created_at=str(state.get("created_at") or utc_now().isoformat()),
                finished_at=str(state.get("finished_at") or utc_now().isoformat()),
            )
        raise ValueError("Unrecognized delivery checkpoint state; refusing to resend")

    def _save_chunk_checkpoint(
        self, context: DeliveryContext | None, index: int, digest: str, report: ExportReport
    ) -> None:
        if context is None:
            return
        status = "submitted_unknown" if report.outcome == ExportOutcome.UNKNOWN else "completed"
        context.save_checkpoint(
            self._chunk_checkpoint_key(index),
            {
                "digest": digest,
                "status": status,
                "report": report.model_dump(mode="json"),
                "connection_id": report.connection_id,
                "dry_run": report.dry_run,
                "submitted": report.submitted,
                "remote_task_ids": report.remote_task_ids,
                "created_at": report.created_at,
                "finished_at": report.finished_at,
            },
        )

    @staticmethod
    def _chunk_checkpoint_key(index: int) -> str:
        return f"chunk:{index}"

    def _submit_chunk(
        self,
        client: Any,
        target: str,
        chunk_values: list[Any],
        *,
        dry_run: bool,
        created_at: str,
        chunk_sha256: str,
        context: DeliveryContext | None = None,
        index: int = 0,
    ) -> ExportReport:
        """POST one bounded chunk and reconcile its response, polling async tasks."""
        from open_climate_service.shared.time import utc_now

        chunk_body = {"dataValues": chunk_values}
        submitted = len(chunk_values)
        params: dict[str, str] = {"importStrategy": "CREATE_AND_UPDATE"}
        if dry_run:
            params["dryRun"] = "true"
        try:
            response = client.post("/api/dataValueSets", json=chunk_body, params=params)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            error_payload = getattr(exc, "payload", None)
            if isinstance(status_code, int) and 400 <= status_code < 500 and isinstance(error_payload, dict):
                return build_dhis2_report(
                    status_code,
                    error_payload,
                    plugin_id=self.id,
                    connection_id=target,
                    dry_run=dry_run,
                    payload_sha256=chunk_sha256,
                    submitted=submitted,
                    created_at=created_at,
                )
            if _is_retryable_transport_error(exc):
                # The request never reached DHIS2 (refused connection, DNS failure,
                # or connect timeout). Raise so the chunk is not checkpointed as
                # submitted and the caller can retry.
                raise _RetryableTransportError(str(exc)) from exc
            # The POST may have reached DHIS2 before failing (e.g. timeout).
            # Record an unknown outcome rather than fabricating a rejection, and
            # never auto-replay this chunk.
            return ExportReport(
                plugin_id=self.id,
                connection_id=target,
                dry_run=dry_run,
                outcome=ExportOutcome.UNKNOWN,
                message=f"Transport error before the import result could be read: {type(exc).__name__}",
                payload_sha256=chunk_sha256,
                submitted=submitted,
                created_at=created_at,
                finished_at=utc_now().isoformat(),
            )

        status_code = _response_status(response)
        summary = _response_json(response)
        task_id = self._remote_task_id(summary)
        if self._is_async_acceptance(status_code, summary):
            if task_id is None:
                return ExportReport(
                    plugin_id=self.id,
                    connection_id=target,
                    dry_run=dry_run,
                    outcome=ExportOutcome.UNKNOWN,
                    message="Async import accepted without a reconcilable remote task ID",
                    payload_sha256=chunk_sha256,
                    submitted=submitted,
                    created_at=created_at,
                    finished_at=utc_now().isoformat(),
                )
            self._save_chunk_checkpoint(
                context,
                index,
                chunk_sha256,
                ExportReport(
                    plugin_id=self.id,
                    connection_id=target,
                    dry_run=dry_run,
                    outcome=ExportOutcome.UNKNOWN,
                    payload_sha256=chunk_sha256,
                    submitted=submitted,
                    remote_task_ids=[task_id],
                    created_at=created_at,
                    finished_at=utc_now().isoformat(),
                ),
            )
            return self._poll_task(
                client,
                task_id,
                target=target,
                dry_run=dry_run,
                submitted=submitted,
                payload_sha256=chunk_sha256,
                created_at=created_at,
                context=context,
            )

        return build_dhis2_report(
            status_code,
            summary,
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            payload_sha256=chunk_sha256,
            submitted=submitted,
            created_at=created_at,
        )

    def _poll_task(
        self,
        client: Any,
        task_id: str,
        *,
        target: str,
        dry_run: bool,
        submitted: int,
        payload_sha256: str,
        created_at: str,
        context: DeliveryContext | None = None,
    ) -> ExportReport:
        """Poll a submitted async import task with bounded backoff.

        The first status check happens immediately after acceptance; the backoff
        applies between subsequent attempts.
        """
        import time

        from open_climate_service.shared.time import utc_now

        for attempt in range(max(0, self.max_poll_attempts)):
            if context is not None and context.is_cancel_requested():
                break
            try:
                response = client.get(f"/api/system/taskSummaries/DATAVALUE_IMPORT/{task_id}")
                status_code = _response_status(response)
                summary = _response_json(response)
            except Exception:
                status_code = None
                summary = {}
            if status_code is not None and self._task_is_terminal(summary):
                report = build_dhis2_report(
                    status_code,
                    _extract_import_summary(summary),
                    plugin_id=self.id,
                    connection_id=target,
                    dry_run=dry_run,
                    payload_sha256=payload_sha256,
                    submitted=submitted,
                    created_at=created_at,
                )
                if report.remote_task_ids == []:
                    report = report.model_copy(update={"remote_task_ids": [task_id]})
                return report
            time.sleep(min(30.0, self.poll_backoff_base * (2**attempt)))

        # The task never reached a terminal state within the poll budget.
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.UNKNOWN,
            message=f"Async import task '{task_id}' did not reach a terminal state",
            payload_sha256=payload_sha256,
            submitted=submitted,
            remote_task_ids=[task_id],
            created_at=created_at,
            finished_at=utc_now().isoformat(),
        )

    @staticmethod
    def _remote_task_id(summary: dict[str, Any]) -> str | None:
        nested = summary.get("response")
        task = nested if isinstance(nested, dict) else summary
        value = task.get("id") or task.get("taskId")
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) else None

    @staticmethod
    def _is_async_acceptance(status_code: int, summary: dict[str, Any]) -> bool:
        nested = summary.get("response")
        if isinstance(nested, dict) and nested.get("jobType") == "DATAVALUE_IMPORT":
            return 200 <= status_code < 300
        if status_code == 202:
            return True
        status = summary.get("status")
        return isinstance(status, str) and status in {"PENDING", "RUNNING", "SCHEDULED"}

    @staticmethod
    def _task_is_terminal(summary: dict[str, Any]) -> bool:
        status = summary.get("status") or summary.get("state")
        if not isinstance(status, str):
            # Without a recognizable status the summary cannot be trusted as final.
            return bool(summary.get("importCount") or summary.get("conflicts"))
        return status.upper() in {"SUCCESS", "ERROR", "WARNING", "FAILED", "COMPLETED", "COMPLETE"}


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
    summary = _extract_import_summary(summary)
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

    # Classify by the import summary first: DHIS2 2.38+ reports a partially
    # successful import as HTTP 409 with status WARNING and real importCount
    # figures. Only fall back to the HTTP code when no summary status is present.
    if status in {"ERROR", "FAILED"}:
        return ExportReport(outcome=ExportOutcome.REJECTED, message=message, **base)
    if status == "WARNING" or conflicts:
        return ExportReport(outcome=ExportOutcome.PARTIAL, message=message, **base)
    if status == "SUCCESS":
        outcome = ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS
        return ExportReport(outcome=outcome, message=message, **base)
    if not 200 <= status_code < 300:
        return ExportReport(outcome=ExportOutcome.REJECTED, message=message or f"HTTP {status_code}", **base)
    # A 2xx without a recognizable import summary cannot be trusted as success.
    return ExportReport(outcome=ExportOutcome.UNKNOWN, message="Unrecognized import summary", **base)


def _extract_import_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Surface an import summary from a DHIS2 async task response shape.

    Task responses nest the import result under ``importSummary``, ``summary``,
    or ``taskSummary`` in different DHIS2 versions; a synchronous summary may be
    returned directly. Keep the remote task ID visible on whichever shape wins.
    """
    for key in ("importSummary", "summary", "taskSummary", "response"):
        nested = summary.get(key)
        if isinstance(nested, dict):
            result = dict(nested)
            if "id" not in result and "taskId" not in result:
                task_id = summary.get("id") or summary.get("taskId")
                if task_id is not None:
                    result["taskId"] = task_id
            return result
    return summary


def _response_status(response: Any) -> int:
    # The supported DHIS2 client returns decoded JSON after checking HTTP status.
    if isinstance(response, dict):
        status = response.get("httpStatusCode", 200)
        return status if isinstance(status, int) else 200
    return response.status_code if isinstance(getattr(response, "status_code", None), int) else 0


def _response_json(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
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
