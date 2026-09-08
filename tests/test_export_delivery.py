"""Slice 4: explicit delivery, reports, idempotency, and operator routes."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service import config
from open_climate_service.exports import ExportReport
from open_climate_service.exports.delivery import deliver_named_export, submit_delivery
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin, build_dhis2_report
from open_climate_service.exports.report import ExportOutcome
from open_climate_service.exports.service import write_named_export
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.jobs.models import JobRecord, JobStatus
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now


def _data() -> pd.DataFrame:
    return pd.DataFrame({"geometry": ["DiszpKrYNg8"] * 2, "t": ["202501", "202502"], "rain": [0.0, float("nan")]})


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "exports": [
                {
                    "id": "rain",
                    "plugin": "dhis2",
                    "connection": "hmis",
                    "period_type": "monthly",
                    "series": [{"data_element": "BXgDHhPdFVU"}],
                }
            ],
            "dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "TEST_TOKEN"}],
        },
    )
    monkeypatch.delenv("TEST_TOKEN", raising=False)
    directory = tmp_path / "openeo_jobs" / "source" / "results"
    directory.mkdir(parents=True)
    path = write_named_export(_data(), directory, "DHIS2JSON", {"export": "rain"}, job_id="source")
    openeo_jobs.store_create_job(
        OpenEOJobRecord(
            id="source",
            status=OpenEOJobStatus.FINISHED,
            created=utc_now(),
            usage={"output_path": path},
        )
    )
    yield Path(path)
    job_service_module.reset_job_service()


@pytest.fixture
def fake_send(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def send(self: Dhis2ExportPlugin, payload: bytes, target: Any, *, dry_run: bool = False) -> ExportReport:
        calls.append({"payload": payload, "target": target, "dry_run": dry_run})
        return ExportReport(
            plugin_id=self.id,
            connection_id=target,
            dry_run=dry_run,
            outcome=ExportOutcome.DRY_RUN if dry_run else ExportOutcome.SUCCESS,
            message="accepted",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            submitted=len(payload),
            imported=1,
            created_at=utc_now().isoformat(),
            finished_at=utc_now().isoformat(),
        )

    monkeypatch.setattr(Dhis2ExportPlugin, "send", send)
    return calls


def test_build_dhis2_report_outcomes() -> None:
    base: dict[str, Any] = dict(plugin_id="dhis2", connection_id="hmis", payload_sha256="a" * 64, submitted=2)
    created = "2026-01-01T00:00:00+00:00"

    def build(status_code: int, summary: dict[str, Any]) -> ExportReport:
        return build_dhis2_report(status_code, summary, dry_run=False, created_at=created, **base)

    success = build(200, {"status": "SUCCESS", "importCount": {"imported": 2}})
    assert success.outcome == ExportOutcome.SUCCESS
    assert (success.imported, success.updated, success.ignored, success.deleted) == (2, 0, 0, 0)

    partial = build(200, {"status": "WARNING", "conflicts": [{"object": "value"}]})
    assert partial.outcome == ExportOutcome.PARTIAL

    rejected = build(200, {"status": "ERROR", "message": "no such data element"})
    assert rejected.outcome == ExportOutcome.REJECTED

    http_rejected = build(400, {"message": "bad request"})
    assert http_rejected.outcome == ExportOutcome.REJECTED

    dry = build_dhis2_report(200, {"status": "SUCCESS"}, dry_run=True, created_at=created, **base)
    assert dry.outcome == ExportOutcome.DRY_RUN

    unknown = build(200, {"unexpected": True})
    assert unknown.outcome == ExportOutcome.UNKNOWN


def test_deliver_named_export_calls_plugin_send(saved: Path, fake_send: list[dict[str, Any]]) -> None:
    report = deliver_named_export("rain", "source", dry_run=True)
    assert report["outcome"] == ExportOutcome.DRY_RUN
    assert len(fake_send) == 1
    assert fake_send[0]["target"] == "hmis"
    assert fake_send[0]["dry_run"] is True
    assert fake_send[0]["payload"] == saved.read_bytes()


def test_submit_delivery_deduplicates_by_idempotency_key(saved: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[dict[str, Any]] = []

    class FakeService:
        def submit_callable_job(self, *, func: Any, label: str, request: dict[str, Any], **_: Any) -> JobRecord:
            submitted.append({"func": func, "label": label, "request": request})
            job_id = f"delivery-{len(submitted)}"
            return JobRecord(
                job_id=job_id, process_id=label, status=JobStatus.ACCEPTED, created_at=utc_now(), request=request
            )

    monkeypatch.setattr(job_service_module, "get_job_service", lambda: FakeService())

    first, reused = submit_delivery("rain", "source", dry_run=False, idempotency_key="key-1")
    assert (first, reused) == ("delivery-1", False)
    second, reused = submit_delivery("rain", "source", dry_run=False, idempotency_key="key-1")
    assert (second, reused) == ("delivery-1", True)
    assert len(submitted) == 1
    assert submitted[0]["label"] == "export:rain"
    assert submitted[0]["request"] == {"export_id": "rain", "job_id": "source", "dry_run": False}


def test_submit_delivery_conflicts_on_reused_key_with_different_content(
    saved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeService:
        def submit_callable_job(self, **kwargs: Any) -> JobRecord:
            return JobRecord(
                job_id="delivery-1", process_id="export:rain", status=JobStatus.ACCEPTED, created_at=utc_now()
            )

    monkeypatch.setattr(job_service_module, "get_job_service", lambda: FakeService())

    submit_delivery("rain", "source", dry_run=False, idempotency_key="key-1")
    with pytest.raises(HTTPException) as error:
        submit_delivery("rain", "source", dry_run=True, idempotency_key="key-1")
    assert error.value.status_code == 409


def test_delivery_route_end_to_end(saved: Path, fake_send: list[dict[str, Any]], client: TestClient) -> None:
    response = client.post(
        "/exports/rain",
        json={"job_id": "source", "dry_run": False},
        headers={"Idempotency-Key": "e2e-key"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["delivery_job_id"]
    assert body["status_url"] == f"/exports/rain/jobs/{body['delivery_job_id']}"
    assert body["reused"] is False

    service = job_service_module.get_job_service()
    record = _await_terminal(service, body["delivery_job_id"])
    assert record.status == JobStatus.SUCCESSFUL
    assert isinstance(record.result, dict)
    assert record.result["outcome"] == ExportOutcome.SUCCESS
    assert len(fake_send) == 1

    detail = client.get(body["status_url"])
    assert detail.status_code == 200
    assert detail.json()["report"]["outcome"] == ExportOutcome.SUCCESS

    # The source openEO job advertises the delivery.
    linked = openeo_jobs.store_get_job("source")
    assert linked is not None
    deliveries = (linked.usage or {}).get("deliveries")
    assert isinstance(deliveries, list) and deliveries
    assert deliveries[0]["delivery_job_id"] == body["delivery_job_id"]

    # Reusing the same key returns the existing delivery without a second send.
    again = client.post(
        "/exports/rain",
        json={"job_id": "source", "dry_run": False},
        headers={"Idempotency-Key": "e2e-key"},
    )
    assert again.status_code == 202
    assert again.json()["delivery_job_id"] == body["delivery_job_id"]
    assert again.json()["reused"] is True
    assert len(fake_send) == 1


def test_delivery_route_requires_idempotency_key(
    saved: Path, fake_send: list[dict[str, Any]], client: TestClient
) -> None:
    response = client.post("/exports/rain", json={"job_id": "source"})
    assert response.status_code == 400
    assert "Idempotency-Key" in response.json()["detail"]


def test_delivery_route_refused_in_read_only(saved: Path, fake_send: list[dict[str, Any]], client: TestClient) -> None:
    config.get_config()["read_only"] = True
    response = client.post(
        "/exports/rain",
        json={"job_id": "source"},
        headers={"Idempotency-Key": "ro-key"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "PermissionsInsufficient"


def _await_terminal(service: Any, job_id: str, timeout: float = 5.0) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.get_job_or_404(job_id)
        if record.status in {JobStatus.SUCCESSFUL, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    pytest.fail(f"Delivery job '{job_id}' did not finish within {timeout}s")
