"""Slice 5: chunking, checkpoints, async polling, and third-party delivery plugins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from open_climate_service.exports import ExportOutcome, ExportReport, merge_chunk_reports
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.post_handler: Any = None
        self.get_handler: Any = None

    def close(self) -> None:
        pass

    def post(self, path: str, json: Any = None, params: Any = None) -> FakeResponse:
        self.posts.append({"path": path, "json": json, "params": params})
        if self.post_handler is None:
            raise AssertionError("Unexpected POST")
        return self.post_handler(path, json, params)

    def get(self, path: str) -> FakeResponse:
        self.gets.append(path)
        if self.get_handler is None:
            raise AssertionError("Unexpected GET")
        return self.get_handler(path)


class FakeContext:
    def __init__(self) -> None:
        self.checkpoints: dict[str, Any] = {}
        self.progress: list[tuple[Any, Any, Any]] = []
        self.cancelled = False

    def report_progress(self, done: Any = None, total: Any = None, message: Any = None) -> None:
        self.progress.append((done, total, message))

    def is_cancel_requested(self) -> bool:
        return self.cancelled

    def save_checkpoint(self, key: str, state: dict[str, Any]) -> None:
        self.checkpoints[key] = state

    def load_checkpoint(self, key: str) -> dict[str, Any] | None:
        return self.checkpoints.get(key)


def _values(count: int) -> list[dict[str, Any]]:
    return [
        {
            "dataElement": "BXgDHhPdFVU",
            "period": f"2025{(index % 12) + 1:02d}",
            "orgUnit": "DiszpKrYNg8",
            "value": str(index),
        }
        for index in range(count)
    ]


def _payload(values: list[dict[str, Any]]) -> bytes:
    return json.dumps({"dataValues": values}).encode()


def _plugin(**attrs: Any) -> Dhis2ExportPlugin:
    plugin = Dhis2ExportPlugin()
    for key, value in attrs.items():
        setattr(plugin, key, value)
    return plugin


def _chunk_digest(chunk_values: list[dict[str, Any]]) -> str:
    raw = json.dumps({"dataValues": chunk_values}, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _success_report(
    plugin: Dhis2ExportPlugin, *, submitted: int, imported: int, payload_sha256: str, created_at: str
) -> ExportReport:
    return ExportReport(
        plugin_id=plugin.id,
        connection_id="hmis",
        dry_run=False,
        outcome=ExportOutcome.SUCCESS,
        payload_sha256=payload_sha256,
        submitted=submitted,
        imported=imported,
        created_at=created_at,
        finished_at=created_at,
    )


def test_merge_chunk_reports_precedence() -> None:
    now = "2026-01-01T00:00:00+00:00"

    def report(outcome: ExportOutcome) -> ExportReport:
        return ExportReport(
            plugin_id="dhis2",
            connection_id="hmis",
            dry_run=False,
            outcome=outcome,
            payload_sha256="a" * 64,
            submitted=1,
            created_at=now,
            finished_at=now,
        )

    merged = merge_chunk_reports(
        [report(ExportOutcome.SUCCESS), report(ExportOutcome.PARTIAL), report(ExportOutcome.UNKNOWN)],
        plugin_id="dhis2",
        connection_id="hmis",
        dry_run=False,
        payload_sha256="a" * 64,
        created_at=now,
        finished_at=now,
        submitted=3,
    )
    assert merged.outcome == ExportOutcome.UNKNOWN
    assert merged.chunks == 3
    assert merged.attempts == 3
    assert merged.submitted == 3

    cancelled = merge_chunk_reports(
        [report(ExportOutcome.SUCCESS)],
        plugin_id="dhis2",
        connection_id="hmis",
        dry_run=False,
        payload_sha256="a" * 64,
        created_at=now,
        finished_at=now,
        submitted=5,
        cancelled_early=True,
    )
    assert cancelled.outcome == ExportOutcome.CANCELLED

    dry = merge_chunk_reports(
        [],
        plugin_id="dhis2",
        connection_id="hmis",
        dry_run=True,
        payload_sha256="a" * 64,
        created_at=now,
        finished_at=now,
    )
    assert dry.outcome == ExportOutcome.DRY_RUN


def test_send_splits_large_payload_into_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=2, poll_backoff_base=0.0, max_poll_attempts=1)
    values = _values(5)
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        return FakeResponse(200, {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"])}})

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis")

    assert len(client.posts) == 3
    assert report.outcome == ExportOutcome.SUCCESS
    assert report.chunks == 3
    assert report.submitted == 5
    assert report.imported == 5


def test_send_reuses_completed_checkpoint_and_skips_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=2, poll_backoff_base=0.0, max_poll_attempts=1)
    values = _values(5)
    now = "2026-01-01T00:00:00+00:00"

    context = FakeContext()
    context.checkpoints["chunk:0"] = {
        "digest": _chunk_digest(values[0:2]),
        "status": "completed",
        "report": _success_report(
            plugin, submitted=2, imported=2, payload_sha256=_chunk_digest(values[0:2]), created_at=now
        ).model_dump(mode="json"),
    }
    context.checkpoints["chunk:1"] = {
        "digest": _chunk_digest(values[2:4]),
        "status": "submitted_unknown",
        "connection_id": "hmis",
        "dry_run": False,
        "submitted": 2,
        "remote_task_ids": [],
        "created_at": now,
        "finished_at": now,
    }

    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        return FakeResponse(200, {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"])}})

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis", context=context)

    # Chunk 0 was reused, chunk 1 was not re-sent (unknown), only chunk 2 went out.
    assert len(client.posts) == 1
    assert client.posts[0]["json"]["dataValues"] == values[4:5]
    assert report.outcome == ExportOutcome.UNKNOWN
    assert report.chunks == 3
    assert report.imported == 3  # chunk 0 (2) + chunk 2 (1)
    assert report.submitted == 5


def test_send_stops_when_cancellation_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=2, poll_backoff_base=0.0, max_poll_attempts=1)
    values = _values(5)
    context = FakeContext()
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        if len(client.posts) == 1:
            context.cancelled = True
        return FakeResponse(200, {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"])}})

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis", context=context)

    assert report.outcome == ExportOutcome.CANCELLED
    assert len(client.posts) == 1
    assert report.submitted == 5
    assert report.imported == 2


def test_send_records_unknown_on_transport_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=1000)
    values = _values(3)
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        raise TimeoutError("read timed out")

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis")

    assert report.outcome == ExportOutcome.UNKNOWN
    assert report.remote_task_ids == []
    assert report.submitted == 3


def test_send_polls_async_import_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=1000, poll_backoff_base=0.0, max_poll_attempts=5)
    values = _values(2)
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        return FakeResponse(202, {"id": "task-1", "status": "PENDING"})

    client.post_handler = post

    def get(path: str) -> FakeResponse:
        if len(client.gets) == 1:
            return FakeResponse(200, {"status": "RUNNING"})
        return FakeResponse(200, {"status": "SUCCESS", "importCount": {"imported": 2}})

    client.get_handler = get
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis")

    assert report.outcome == ExportOutcome.SUCCESS
    assert report.remote_task_ids == ["task-1"]
    assert len(client.posts) == 1
    assert len(client.gets) >= 2


def test_send_poll_budget_exhaustion_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=1000, poll_backoff_base=0.0, max_poll_attempts=3)
    values = _values(1)
    client = FakeClient()

    client.post_handler = lambda path, json, params: FakeResponse(202, {"id": "task-1", "status": "PENDING"})
    client.get_handler = lambda path: FakeResponse(200, {"status": "RUNNING"})
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    report = plugin.send(_payload(values), "hmis")

    assert report.outcome == ExportOutcome.UNKNOWN
    assert report.remote_task_ids == ["task-1"]


def _install_delivery_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "chunky_export_" + uuid4().hex
    package = tmp_path / name
    (package / "exports").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "exports" / "__init__.py").write_text("", encoding="utf-8")
    (package / "exports" / "chunky.py").write_text(_EXTERNAL_DELIVERY_SOURCE, encoding="utf-8")
    metadata = tmp_path / f"{name}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    (metadata / "entry_points.txt").write_text(f"[open_climate_service.plugins]\n{name} = {name}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


_EXTERNAL_DELIVERY_SOURCE = """
import datetime as dt
import hashlib

from open_climate_service.exports import (
    BaseExportPlugin,
    RenderedExport,
    ExportOutcome,
    ExportReport,
    merge_chunk_reports,
)


class ChunkedDelivery(BaseExportPlugin):
    id = "chunky"
    format = "CHUNKY"
    extension = ".chunky"
    media_type = "application/x-chunky"
    version = "1"
    supports_delivery = True

    def validate_mapping(self, mapping):
        return mapping

    def render(self, data, mapping):
        return RenderedExport(data.encode() if isinstance(data, str) else b"rendered", 1)

    def send(self, payload, target, *, dry_run=False, context=None):
        chunk_size = 3
        chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)] or [b""]
        reports = []
        for index, chunk in enumerate(chunks):
            if context is not None and context.is_cancel_requested():
                break
            key = "chunk:" + str(index)
            digest = hashlib.sha256(chunk).hexdigest()
            checkpoint = context.load_checkpoint(key) if context is not None else None
            if isinstance(checkpoint, dict) and checkpoint.get("digest") == digest and checkpoint.get("report"):
                reports.append(ExportReport.model_validate(checkpoint["report"]))
                continue
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            reports.append(ExportReport(
                plugin_id=self.id,
                connection_id=target,
                dry_run=dry_run,
                outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
                payload_sha256=digest,
                submitted=len(chunk),
                imported=len(chunk),
                created_at=now,
                finished_at=now,
            ))
            if context is not None:
                context.save_checkpoint(key, {"digest": digest, "report": reports[-1].model_dump(mode="json")})
                context.report_progress(index + 1, len(chunks), "delivering")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        return merge_chunk_reports(
            reports,
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            created_at=now,
            finished_at=now,
            submitted=len(payload),
        )


plugin = ChunkedDelivery()
"""


def test_third_party_delivery_plugin_uses_only_public_apis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from open_climate_service import config
    from open_climate_service.exports.registry import load_export_plugins
    from open_climate_service.exports.service import render_named_export

    _install_delivery_fixture(tmp_path, monkeypatch)
    config.get_config()["exports"] = [{"id": "chunky", "plugin": "chunky"}]
    plugin = load_export_plugins()["chunky"]
    assert plugin.supports_delivery is True

    _, rendered = render_named_export("hello world", "CHUNKY", {"export": "chunky"})

    context = FakeContext()
    report = plugin.send(rendered.content, "some-target", context=context)

    assert isinstance(report, ExportReport)
    assert report.outcome == ExportOutcome.SUCCESS
    assert report.chunks == 4  # "hello world" (11 bytes) in chunks of 3
    assert report.imported == 11
    assert report.submitted == 11
    assert len(context.progress) == 4
    assert all(key.startswith("chunk:") for key in context.checkpoints)

    # A second run over the same context resumes from checkpoints without work.
    context.progress.clear()
    again = plugin.send(rendered.content, "some-target", context=context)
    assert again.outcome == ExportOutcome.SUCCESS
    assert again.imported == 11
