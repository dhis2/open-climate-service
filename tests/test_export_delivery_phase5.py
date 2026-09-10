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
    def __init__(self: Any, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self: Any) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self: Any) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.post_handler: Any = None
        self.get_handler: Any = None

    def close(self: Any) -> None:
        pass

    def post(self: Any, path: str, json: Any = None, params: Any = None) -> FakeResponse:
        self.posts.append({"path": path, "json": json, "params": params})
        if self.post_handler is None:
            raise AssertionError("Unexpected POST")
        return self.post_handler(path, json, params)

    def get(self: Any, path: str) -> FakeResponse:
        self.gets.append(path)
        if self.get_handler is None:
            raise AssertionError("Unexpected GET")
        return self.get_handler(path)


class FakeContext:
    def __init__(self: Any) -> None:
        self.checkpoints: dict[str, Any] = {}
        self.progress: list[tuple[Any, Any, Any]] = []
        self.cancelled = False

    def report_progress(self: Any, done: Any = None, total: Any = None, message: Any = None) -> None:
        self.progress.append((done, total, message))

    def is_cancel_requested(self: Any) -> bool:
        return self.cancelled

    def save_checkpoint(self: Any, key: str, state: dict[str, Any]) -> None:
        self.checkpoints[key] = state

    def load_checkpoint(self: Any, key: str) -> dict[str, Any] | None:
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
    assert report.submitted == 2
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


@pytest.mark.parametrize("error_type", ["connect_error", "connect_timeout"])
def test_send_reports_pre_submission_connection_failure(monkeypatch: pytest.MonkeyPatch, error_type: str) -> None:
    import httpx

    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=1000)
    values = _values(1001)
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        if len(client.posts) == 1:
            return FakeResponse(200, {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"])}})
        if error_type == "connect_timeout":
            raise httpx.ConnectTimeout("connection timed out")
        raise httpx.ConnectError("connection refused")

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    context = FakeContext()
    report = plugin.send(_payload(values), "hmis", context=context)

    assert report.outcome == ExportOutcome.REJECTED
    assert report.chunks == 2
    assert report.submitted == 1000
    assert report.imported == 1000
    assert "Connection failed before submission" in str(report.message)
    assert context.checkpoints["chunk:1"]["report"]["outcome"] == ExportOutcome.REJECTED
    assert context.checkpoints["chunk:1"]["status"] == "retryable_failed"

    client.post_handler = lambda path, json, params: FakeResponse(
        200, {"status": "SUCCESS", "importCount": {"imported": len(json["dataValues"])}}
    )
    resumed = plugin.send(_payload(values), "hmis", context=context)

    assert resumed.outcome == ExportOutcome.SUCCESS
    assert resumed.imported == 1001
    assert len(client.posts) == 3


def test_send_keeps_unknown_checkpoint_on_ambiguous_requests_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    from open_climate_service.exports import dhis2 as dhis2_module

    plugin = _plugin(max_chunk_size=1000)
    client = FakeClient()

    def post(path: str, json: Any, params: Any) -> FakeResponse:
        raise requests.exceptions.ConnectionError("connection reset while reading response")

    client.post_handler = post
    monkeypatch.setattr(dhis2_module, "get_connection", lambda target: client)

    context = FakeContext()
    report = plugin.send(_payload(_values(3)), "hmis", context=context)

    assert report.outcome == ExportOutcome.UNKNOWN
    assert context.checkpoints["chunk:0"]["report"]["outcome"] == ExportOutcome.UNKNOWN


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


def test_restart_after_remote_acceptance_does_not_resend(monkeypatch: pytest.MonkeyPatch):
    from open_climate_service.exports import dhis2

    plugin = _plugin(poll_backoff_base=0)
    client = FakeClient()
    context = FakeContext()
    payload = _payload(_values(1))
    monkeypatch.setattr(dhis2, "get_connection", lambda _: client)

    def crash_after_acceptance(path: Any, json: Any, params: Any):
        raise SystemExit("process died after the remote write")

    client.post_handler = crash_after_acceptance
    with pytest.raises(SystemExit):
        plugin.send(payload, "hmis", context=context)
    client.post_handler = lambda *args: pytest.fail("Uncertain chunk must not be resent")
    report = plugin.send(payload, "hmis", context=context)
    assert report.outcome == ExportOutcome.UNKNOWN
    assert len(client.posts) == 1


@pytest.mark.parametrize(
    "state",
    [
        {"status": "completed", "report": {}},
        {"status": "unrecognized"},
        {"status": "completed", "digest": "changed", "report": {}},
    ],
)
def test_corrupt_checkpoint_refuses_resubmission(monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]):
    from open_climate_service.exports import dhis2

    values = _values(1)
    client = FakeClient()
    context = FakeContext()
    context.checkpoints["chunk:0"] = {"digest": _chunk_digest(values), **state}
    monkeypatch.setattr(dhis2, "get_connection", lambda _: client)
    with pytest.raises(ValueError):
        _plugin().send(_payload(values), "hmis", context=context)
    assert client.posts == []


def test_nested_async_task_is_checkpointed_and_polled_on_restart(monkeypatch: pytest.MonkeyPatch):
    from open_climate_service.exports import dhis2

    plugin = _plugin(max_poll_attempts=1, poll_backoff_base=0)
    client = FakeClient()
    client.post_handler = lambda *args, **kwargs: {
        "httpStatusCode": 200,
        "status": "OK",
        "response": {"jobType": "DATAVALUE_IMPORT", "id": "task1"},
    }
    client.get_handler = lambda _: {}
    context = FakeContext()
    monkeypatch.setattr(dhis2, "get_connection", lambda _: client)
    payload = _payload(_values(1))
    first = plugin.send(payload, "hmis", context=context)
    assert first.outcome == ExportOutcome.UNKNOWN
    assert context.checkpoints["chunk:0"]["remote_task_ids"] == ["task1"]
    client.get_handler = lambda _: {"status": "SUCCESS", "importCount": {"imported": 1}}
    second = plugin.send(payload, "hmis", context=context)
    assert second.outcome == ExportOutcome.SUCCESS
    assert len(client.posts) == 1
    assert client.gets == ["/api/system/taskSummaries/DATAVALUE_IMPORT/task1"] * 2


@pytest.mark.parametrize("status_code,outcome", [(200, ExportOutcome.SUCCESS), (409, ExportOutcome.REJECTED)])
def test_pinned_client_response_contract(monkeypatch: pytest.MonkeyPatch, status_code: int, outcome: ExportOutcome):
    import httpx

    from open_climate_service import config

    pytest.importorskip("dhis2_client")
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "dhis2_connections": [
                {"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "REVIEW_TEST_TOKEN"},
            ]
        },
    )
    monkeypatch.setenv("REVIEW_TEST_TOKEN", "test-token")
    requests = []

    def handler(request: httpx.Request):
        requests.append(request)
        assert request.url.path == "/dhis/api/dataValueSets"
        return httpx.Response(
            status_code,
            json={
                "response": {"status": "SUCCESS" if status_code == 200 else "ERROR", "importCount": {"imported": 1}},
            },
        )

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    report = _plugin().send(_payload(_values(1)), "hmis")
    assert report.outcome == outcome
    assert len(requests) == 1


def test_warning_summary_is_partial_not_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from open_climate_service import config

    pytest.importorskip("dhis2_client")
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "dhis2_connections": [
                {"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "REVIEW_TEST_TOKEN"},
            ]
        },
    )
    monkeypatch.setenv("REVIEW_TEST_TOKEN", "test-token")

    def handler(request: httpx.Request):
        return httpx.Response(
            409,
            json={
                "response": {
                    "status": "WARNING",
                    "importCount": {"imported": 999},
                    "conflicts": [{"object": "value"}],
                }
            },
        )

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    report = _plugin(max_chunk_size=1000).send(_payload(_values(1)), "hmis")
    assert report.outcome == ExportOutcome.PARTIAL
    assert report.imported == 999


def test_installed_plugin_can_deliver_through_framework(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from open_climate_service import config
    from open_climate_service.exports.delivery import deliver_named_export
    from open_climate_service.exports.delivery_input import lease_export_input
    from open_climate_service.exports.service import write_named_export
    from open_climate_service.openeo import jobs
    from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
    from open_climate_service.shared.provenance import json_digest
    from open_climate_service.shared.time import utc_now

    _install_delivery_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, "_JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path)
    config.get_config()["exports"] = [{"id": "chunky", "plugin": "chunky", "connection": "hmis"}]
    config.get_config()["dhis2_connections"] = [
        {"id": "hmis", "url": "https://example.org", "token_env": "UNSET_TEST_TOKEN"},
    ]
    directory = tmp_path / "jobs" / "source" / "results"
    directory.mkdir(parents=True)
    path = write_named_export("hello", directory, "CHUNKY", {"export": "chunky"}, job_id="source")
    jobs.store_create_job(
        OpenEOJobRecord(
            id="source",
            status=OpenEOJobStatus.FINISHED,
            created=utc_now(),
            usage={"output_path": path},
        )
    )
    with lease_export_input("chunky", "source") as verified:
        digest = json_digest(verified.manifest.model_dump(mode="json"))
    report = deliver_named_export("chunky", "source", expected_manifest_sha256=digest)
    assert report["outcome"] == "success"
    assert report["imported"] == 5
