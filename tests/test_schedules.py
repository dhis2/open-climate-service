"""Sync schedules in the single shared store (CLIM-1242)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.scheduler import service as scheduler_service
from open_climate_service.scheduler import store
from open_climate_service.scheduler.config import (
    DatasetSyncSchedule,
    SchedulerConfig,
    effective_schedules,
    get_scheduler_config,
)
from open_climate_service.scheduler.presets import (
    cron_from_form,
    form_values,
    schedule_description,
    suggested_frequency,
)
from open_climate_service.scheduler.service import SchedulerService
from open_climate_service.scheduler.store import ScheduleStoreUnreadable, StoredSchedule

BROWSER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
_TEMPLATES: dict[str, dict[str, Any]] = {
    "chirps": {"id": "chirps", "sync": {"kind": "temporal", "execution": "append"}},
    "era5": {"id": "era5", "sync": {"kind": "temporal", "execution": "append"}},
    "worldpop": {"id": "worldpop", "sync": {"kind": "static"}},
}


def _stored(dataset_id: str = "chirps", **updates: Any) -> StoredSchedule:
    values: dict[str, Any] = {"dataset_id": dataset_id, "cron": "0 6 * * *"}
    values.update(updates)
    return StoredSchedule.model_validate(values)


def _file(dataset_id: str = "chirps", cron: str = "0 5 * * *") -> DatasetSyncSchedule:
    return DatasetSyncSchedule(dataset_id=dataset_id, cron=cron)


def _dataset(dataset_id: str) -> Any:
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_name=dataset_id.upper(),
        item_type="coverage",
        source_dataset_id=dataset_id,
        period_type="monthly",
        extent=SimpleNamespace(temporal=SimpleNamespace(start="2025-01", end="2025-02")),
    )


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A data root, syncable templates, a disabled scheduler and a fresh singleton."""
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"enabled": False}})
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: _TEMPLATES.get(dataset_id))
    import open_climate_service.ingestions.services as ingestion_services

    monkeypatch.setattr(
        ingestion_services, "list_datasets", lambda: SimpleNamespace(items=[_dataset("chirps"), _dataset("era5")])
    )

    def no_managed_dataset(dataset_id: str) -> None:
        raise HTTPException(status_code=404, detail=dataset_id)

    monkeypatch.setattr(
        ingestion_services,
        "get_latest_artifact_for_dataset_or_404",
        no_managed_dataset,
    )
    monkeypatch.setattr(scheduler_service, "_service", None)


@pytest.fixture
def client(instance: None) -> TestClient:
    from open_climate_service.main import app

    return TestClient(app)


# --- store -----------------------------------------------------------------------------------


def test_store_creates_updates_pauses_and_deletes(instance: None) -> None:
    created = store.save_schedule(_stored(), create=True)
    assert created.enabled and store.list_schedules() == [created]
    with pytest.raises(ValueError, match="already exists"):
        store.save_schedule(_stored(cron="0 7 * * *"), create=True)
    with pytest.raises(ValueError, match="No stored schedule"):
        store.save_schedule(_stored("era5"), create=False)
    updated = store.save_schedule(_stored(cron="0 7 * * *"), create=False)
    assert updated.cron == "0 7 * * *" and updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    paused = store.set_enabled("chirps", False)
    assert paused.enabled is False and store.get_schedule("chirps") == paused
    assert store.set_enabled("chirps", False) == paused
    assert store.delete_schedule("chirps") is True and store.delete_schedule("chirps") is False
    assert store.list_schedules() == []


def test_store_refuses_an_unreadable_file(instance: None) -> None:
    path = store.schedules_path()
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ScheduleStoreUnreadable, match="not valid JSON"):
        store.list_schedules()
    path.write_text(
        '{"sync-chirps": {"id": "sync-era5", "kind": "sync", "target": "era5", "cron": "0 6 * * *"}}',
        encoding="utf-8",
    )
    with pytest.raises(ScheduleStoreUnreadable, match="has id"):
        store.get_schedule("chirps")


def test_store_stamp_detects_equal_size_replacement_with_unchanged_mtime(instance: None) -> None:
    path = store.schedules_path()
    path.parent.mkdir(parents=True)
    path.write_text("a", encoding="utf-8")
    first = store.store_stamp()
    mtime_ns = path.stat().st_mtime_ns
    path.write_text("b", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    assert path.stat().st_size == 1 and path.stat().st_mtime_ns == mtime_ns
    assert store.store_stamp() != first


def test_legacy_yaml_is_rejected_at_runtime_with_recreation_instruction(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"dataset_sync": []}})
    with pytest.raises(ValueError, match="recreate schedules"):
        get_scheduler_config()


# --- effective store entries ---------------------------------------------------------------------


def test_stored_entries_are_the_only_effective_schedules() -> None:
    entries = effective_schedules([_stored("era5"), _stored("chirps"), _stored("x", enabled=False)])
    assert [(item.dataset_id, item.effective) for item in entries] == [
        ("chirps", True),
        ("era5", True),
        ("x", False),
    ]


# --- service -------------------------------------------------------------------------------------


def _added(scheduler: MagicMock, start: int = 0) -> list[str]:
    """Ids of the schedule jobs added, in order, leaving out the store watch."""
    ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list if c.kwargs["id"].startswith("dataset-sync:")]
    return ids[start:]


def _fake_scheduler(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    scheduler = MagicMock()
    scheduler.get_jobs.return_value = []
    scheduler.get_job.return_value = None
    monkeypatch.setattr("open_climate_service.scheduler.service.AsyncIOScheduler", lambda **_: scheduler)
    monkeypatch.setattr("open_climate_service.scheduler.service.api_config.is_read_only", lambda: False)
    return scheduler


def test_start_registers_active_stored_entries_only(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored("era5"), _stored("chirps", cron="0 9 * * *"), _stored("x", enabled=False)],
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id, {"id": dataset_id, "sync": {"kind": "temporal"}}),
    )
    service.start()
    registered = _added(scheduler)
    assert registered == ["dataset-sync:chirps", "dataset-sync:era5"]
    watch = next(call for call in scheduler.add_job.call_args_list if call.kwargs["id"] == "scheduler:store-watch")
    assert watch.args[0] == service.reload_if_changed
    status = service.status()
    assert [(item.dataset_id, item.effective) for item in status.schedules] == [
        ("chirps", True),
        ("era5", True),
        ("x", False),
    ]
    assert status.reload_error is None


def test_start_survives_an_unreadable_store_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)

    def broken() -> list[StoredSchedule]:
        raise ScheduleStoreUnreadable("schedules.json is not valid JSON")

    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=broken,
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    assert _added(scheduler) == []
    status = service.status()
    assert status.reload_error and "not valid JSON" in status.reload_error
    assert status.schedules == []


def test_reload_adds_replaces_and_removes_jobs_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    stored: list[StoredSchedule] = [_stored("era5")]
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: list(stored),
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id, {"id": dataset_id, "sync": {"kind": "temporal"}}),
    )
    service.start()
    assert _added(scheduler) == ["dataset-sync:era5"]

    stored[:] = [_stored("era5", cron="0 7 * * *"), _stored("chirps")]
    scheduler.get_job.side_effect = lambda job_id: MagicMock() if job_id == "dataset-sync:era5" else None
    service.reload()
    assert _added(scheduler, 1) == ["dataset-sync:chirps", "dataset-sync:era5"]
    assert all(call.kwargs["replace_existing"] for call in scheduler.add_job.call_args_list)
    scheduler.remove_job.assert_not_called()

    stored[:] = [_stored("era5", cron="0 7 * * *", enabled=False)]
    scheduler.get_job.side_effect = lambda job_id: MagicMock()
    service.reload()
    removed = sorted(call.args[0] for call in scheduler.remove_job.call_args_list)
    assert removed == ["dataset-sync:chirps", "dataset-sync:era5"]
    status = service.status()
    assert [(item.dataset_id, item.effective) for item in status.schedules] == [("era5", False)]


def test_failed_reload_keeps_the_previous_schedules(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    loads: list[list[StoredSchedule]] = [[_stored("era5")]]

    def load() -> list[StoredSchedule]:
        if not loads:
            raise ScheduleStoreUnreadable("schedules.json: schedule for 'x' is invalid")
        return loads.pop()

    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=load,
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    registered_before = scheduler.add_job.call_count
    service.reload()
    status = service.status()
    assert status.reload_error and "invalid" in status.reload_error and "stay in force" in status.reload_error
    assert [item.dataset_id for item in status.schedules] == ["era5"]
    assert scheduler.add_job.call_count == registered_before
    scheduler.remove_job.assert_not_called()
    loads.append([_stored("era5"), _stored("chirps")])
    service.reload()
    assert service.status().reload_error is None
    assert [item.dataset_id for item in service.status().schedules] == ["chirps", "era5"]


def test_reload_takes_an_entry_whose_dataset_no_longer_resolves_off_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    templates: dict[str, dict[str, Any]] = dict(_TEMPLATES)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored("era5"), _stored("chirps")],
        template_loader=lambda dataset_id: templates.get(dataset_id),
    )
    service.start()
    assert _added(scheduler) == [
        "dataset-sync:chirps",
        "dataset-sync:era5",
    ]

    del templates["era5"]
    scheduler.get_job.side_effect = lambda job_id: MagicMock()
    service.reload()
    scheduler.remove_job.assert_called_once_with("dataset-sync:era5")
    assert _added(scheduler, 2) == ["dataset-sync:chirps"]
    status = service.status()
    assert status.reload_error is None
    era5 = next(item for item in status.schedules if item.dataset_id == "era5")
    assert era5.effective and era5.last_outcome == "error" and "no registered data source" in (era5.last_message or "")


def test_a_refused_apply_restores_the_previous_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    stored: list[StoredSchedule] = [_stored("era5")]
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: list(stored),
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    stored[:] = [_stored("era5", cron="0 7 * * *"), _stored("chirps")]

    def refuse_the_new_entry(*args: Any, **kwargs: Any) -> None:
        if kwargs["id"] == "dataset-sync:chirps":
            raise RuntimeError("scheduler is shutting down")

    scheduler.add_job.side_effect = refuse_the_new_entry
    scheduler.get_job.side_effect = lambda job_id: MagicMock()
    service.reload()
    status = service.status()
    assert status.reload_error and "were restored" in status.reload_error
    assert [(item.dataset_id, item.cron) for item in status.schedules] == [("era5", "0 6 * * *")]
    era5_crons = [
        c.kwargs["args"][0].cron for c in scheduler.add_job.call_args_list if c.kwargs["id"] == "dataset-sync:era5"
    ]
    assert era5_crons[-1] == "0 6 * * *"
    scheduler.remove_job.assert_called_with("dataset-sync:chirps")


def test_a_failed_restore_says_the_clock_may_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    stored: list[StoredSchedule] = [_stored("era5")]
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: list(stored),
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    stored.append(_stored("chirps"))
    scheduler.add_job.side_effect = RuntimeError("scheduler is shutting down")
    service.reload()
    status = service.status()
    assert status.reload_error and "could not be restored" in status.reload_error
    assert "may run settings" in status.reload_error
    assert [item.dataset_id for item in status.schedules] == ["era5"]


def test_the_clock_owner_picks_up_a_change_made_by_another_process(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"enabled": True}})
    service = SchedulerService(template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id))
    service.start()
    assert _added(scheduler) == []
    assert service.reload_if_changed() is False

    store.save_schedule(_stored("era5"), create=True)  # as another replica would, through the shared file
    assert service.reload_if_changed() is True
    assert _added(scheduler) == ["dataset-sync:era5"]
    assert service.reload_if_changed() is False
    store.set_enabled("era5", False)
    scheduler.get_job.return_value = MagicMock()
    assert service.reload_if_changed() is True
    scheduler.remove_job.assert_called_once_with("dataset-sync:era5")


def test_active_and_paused_rows_have_separate_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    next_check = datetime(2026, 10, 8, 6, tzinfo=timezone.utc)
    scheduler.get_jobs.return_value = [MagicMock(id="dataset-sync:chirps", next_run_time=next_check)]
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True),
        store_loader=lambda: [_stored("chirps", cron="0 9 * * *"), _stored("era5", enabled=False)],
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    service.check_now(_file("chirps"))
    rows = {item.dataset_id: item for item in service.status().schedules}
    active = rows["chirps"]
    assert active.registered and active.next_check == next_check and active.last_outcome is not None
    paused = rows["era5"]
    assert not paused.effective and not paused.registered and paused.next_check is None


def test_reload_on_a_disabled_scheduler_only_updates_the_listing(instance: None) -> None:
    service = SchedulerService(template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id))
    service.start()
    assert service.status().schedules == []
    store.save_schedule(_stored("era5"), create=True)
    service.reload()
    status = service.status()
    assert status.running is False and [item.dataset_id for item in status.schedules] == ["era5"]
    assert service.schedule_for("era5") is not None and service.schedule_for("chirps") is None


# --- API -----------------------------------------------------------------------------------------


def test_api_creates_reads_updates_pauses_and_deletes(client: TestClient) -> None:
    created = client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"})
    assert created.status_code == 201, created.text
    assert created.json()["source"] == "store" and created.json()["enabled"] is True
    assert client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"}).status_code == 409
    listed = client.get("/schedules").json()
    assert [item["dataset_id"] for item in listed["schedules"]] == ["era5"] and listed["reload_error"] is None
    one = client.get("/schedules/sync/era5")
    assert one.status_code == 200 and one.json()["cron"] == "0 6 * * *"
    assert client.get("/schedules/sync/missing").status_code == 404

    updated = client.put("/schedules/sync/era5", json={"cron": "0 7 * * *", "max_attempts": 5, "publish": False})
    assert updated.status_code == 200 and updated.json()["cron"] == "0 7 * * *"
    assert updated.json()["max_attempts"] == 5 and updated.json()["publish"] is False
    assert client.put("/schedules/sync/missing", json={"cron": "0 7 * * *"}).status_code == 404

    paused = client.post("/schedules/sync/era5/pause", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["enabled"] is False and paused.json()["effective"] is False
    resumed = client.post("/schedules/sync/era5/resume", headers={"Accept": "application/json"})
    assert resumed.status_code == 200 and resumed.json()["effective"] is True

    assert client.delete("/schedules/sync/era5").status_code == 204
    assert client.delete("/schedules/sync/era5").status_code == 404
    assert client.get("/schedules").json()["schedules"] == []


def test_api_refuses_what_the_clock_could_not_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    static = client.post("/schedules/sync", json={"dataset_id": "worldpop", "cron": "0 6 * * *"})
    assert static.status_code == 422 and "not syncable" in static.text
    unknown = client.post("/schedules/sync", json={"dataset_id": "nope", "cron": "0 6 * * *"})
    assert unknown.status_code == 422 and "no registered data source" in unknown.text
    bad_cron = client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "every day"})
    assert bad_cron.status_code == 422 and "cron" in bad_cron.text

    assert client.put("/schedules/sync/chirps", json={"cron": "0 6 * * *"}).status_code == 404
    assert client.delete("/schedules/sync/chirps").status_code == 404


def test_managed_schedule_uses_its_source_template(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import open_climate_service.ingestions.services as ingestion_services

    managed = _dataset("managed-era5")
    managed.source_dataset_id = "era5"
    monkeypatch.setattr(ingestion_services, "list_datasets", lambda: SimpleNamespace(items=[managed]))
    monkeypatch.setattr(
        ingestion_services,
        "get_latest_artifact_for_dataset_or_404",
        lambda dataset_id: SimpleNamespace(dataset_id=dataset_id, source_dataset_id="era5"),
    )
    created = client.post("/schedules/sync", json={"dataset_id": "managed-era5", "cron": "0 6 * * *"})
    assert created.status_code == 201, created.text
    assert (
        scheduler_service.get_scheduler_service()
        ._plan(
            SchedulerConfig(enabled=True),
            effective_schedules([_stored("managed-era5")]),
        )
        .refused
        == {}
    )
    updated = client.put("/schedules/sync/managed-era5", json={"cron": "0 7 * * *"})
    assert updated.status_code == 200, updated.text


def test_a_stored_schedule_stays_editable(client: TestClient) -> None:
    store.save_schedule(_stored("chirps", cron="0 9 * * *"), create=True)
    updated = client.put("/schedules/sync/chirps", json={"cron": "0 10 * * *"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["source"] == "store"
    assert updated.json()["cron"] == "0 10 * * *" and updated.json()["effective"] is True
    assert client.get("/schedules/sync/chirps").json()["source"] == "store"
    paused = client.post("/schedules/sync/chirps/pause", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["enabled"] is False
    assert client.delete("/schedules/sync/chirps").status_code == 204
    assert client.get("/schedules").json()["schedules"] == []


def test_a_saved_schedule_reaches_the_running_clock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"enabled": True}})
    monkeypatch.setattr(scheduler_service, "_service", None)
    scheduler_service.get_scheduler_service().start()
    assert _added(scheduler) == []

    assert client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"}).status_code == 201
    assert _added(scheduler) == ["dataset-sync:era5"]
    scheduler.get_job.return_value = MagicMock()
    assert client.post("/schedules/sync/era5/pause", headers={"Accept": "application/json"}).status_code == 200
    scheduler.remove_job.assert_called_once_with("dataset-sync:era5")


# --- pages ---------------------------------------------------------------------------------------


@pytest.fixture
def dataset_pages(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The dataset page for era5 and chirps, served the way the app serves it."""
    import open_climate_service.ingestions.services as ingestion_services
    from tests.test_dataset_page import _record

    def record(dataset_id: str) -> Any:
        if dataset_id not in {"era5", "chirps"}:
            raise HTTPException(status_code=404, detail=dataset_id)
        return _record(dataset_id)

    monkeypatch.setattr(ingestion_services, "get_dataset_or_404", record)
    return client


def test_the_dataset_page_creates_edits_pauses_and_deletes_its_schedule(dataset_pages: TestClient) -> None:
    client = dataset_pages
    page = client.get("/datasets/era5", headers={"Accept": BROWSER})
    assert page.status_code == 200 and 'id="schedule"' in page.text
    assert "Not synced on a schedule" in page.text and "Schedule automatic sync" in page.text
    assert 'action="/schedules/sync"' in page.text and 'name="return_to" value="dataset"' in page.text
    assert 'role="tablist" aria-label="Sync options" hidden' in page.text
    assert 'id="sync-tab" role="tab"' in page.text and 'id="schedule-tab" role="tab"' in page.text
    assert 'id="sync-panel" role="tabpanel"' in page.text
    assert 'id="schedule" role="tabpanel"' in page.text
    assert page.text.index('id="schedule" role="tabpanel"') < page.text.index('id="access-title"')
    assert 'name="frequency"' in page.text and 'name="check_time"' in page.text
    assert "Suggested starting point for monthly data: weekly" in page.text
    assert "scheduler is off" in page.text

    created = client.post(
        "/schedules/sync",
        data={"dataset_id": "era5", "cron": "0 6 * * *", "publish": "on", "enabled": "on", "return_to": "dataset"},
        follow_redirects=False,
    )
    assert created.status_code == 303 and created.headers["location"].endswith("/datasets/era5#schedule")
    page = client.get("/datasets/era5", headers={"Accept": BROWSER})
    assert "Checked on" in page.text and "0 6 * * *" in page.text and "Edit schedule" in page.text
    assert 'action="/schedules/sync/era5"' in page.text and ">Pause<" in page.text
    assert 'data-delete="era5"' in page.text and '<dialog id="delete-dialog"' in page.text
    assert 'name="return_to" value="dataset"' in page.text and "<noscript>" in page.text

    saved = client.post(
        "/schedules/sync/era5",
        data={"cron": "0 7 * * *", "max_attempts": "4", "enabled": "on", "return_to": "dataset"},
        follow_redirects=False,
    )
    assert saved.status_code == 303 and saved.headers["location"].endswith("/datasets/era5#schedule")
    stored = store.get_schedule("era5")
    assert stored is not None and stored.cron == "0 7 * * *" and stored.max_attempts == 4
    assert stored.publish is True, "editing the check time must not silently change publication behavior"

    paused = client.post("/schedules/sync/era5/pause", data={"return_to": "dataset"}, follow_redirects=False)
    assert paused.status_code == 303 and paused.headers["location"].endswith("/datasets/era5#schedule")
    assert "are paused" in client.get("/datasets/era5", headers={"Accept": BROWSER}).text
    assert ">Resume<" in client.get("/datasets/era5", headers={"Accept": BROWSER}).text

    unconfirmed = client.post("/schedules/sync/era5/delete", data={"return_to": "dataset"}, follow_redirects=False)
    assert unconfirmed.status_code == 400 and store.get_schedule("era5") is not None
    deleted = client.post(
        "/schedules/sync/era5/delete", data={"confirm": "yes", "return_to": "dataset"}, follow_redirects=False
    )
    assert deleted.status_code == 303 and deleted.headers["location"].endswith("/datasets/era5#schedule")
    assert store.get_schedule("era5") is None


def test_a_refused_save_shows_the_dataset_page_with_the_reason_and_the_draft(dataset_pages: TestClient) -> None:
    refused = dataset_pages.post(
        "/schedules/sync", data={"dataset_id": "era5", "cron": "every day", "return_to": "dataset"}
    )
    assert refused.status_code == 422 and "cron" in refused.text and 'value="every day"' in refused.text
    assert 'id="schedule"' in refused.text and store.get_schedule("era5") is None
    assert 'data-initial-tab="schedule"' in refused.text


def test_bad_form_dataset_id_keeps_validation_error_instead_of_becoming_404(dataset_pages: TestClient) -> None:
    response = dataset_pages.post(
        "/schedules/sync", data={"dataset_id": "", "cron": "0 6 * * *", "return_to": "dataset"}
    )
    assert response.status_code == 422
    assert "dataset_id" in response.text and "Schedules" in response.text


def test_a_nonexistent_form_target_keeps_the_syncability_error(dataset_pages: TestClient) -> None:
    response = dataset_pages.post(
        "/schedules/sync", data={"dataset_id": "missing", "cron": "0 6 * * *", "return_to": "dataset"}
    )
    assert response.status_code == 422
    assert "no registered data source" in response.text


def test_concurrent_removal_during_update_returns_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    store.save_schedule(_stored("era5"), create=True)

    def removed(_: StoredSchedule, *, create: bool) -> StoredSchedule:
        assert create is False
        raise ValueError("No stored schedule for dataset 'era5'")

    monkeypatch.setattr(store, "save_schedule", removed)
    response = client.put("/schedules/sync/era5", json={"cron": "0 7 * * *"})
    assert response.status_code == 404 and "No stored schedule" in response.text


def test_concurrent_removal_during_pause_or_resume_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def removed(_: str, __: bool) -> StoredSchedule:
        raise ValueError("No stored schedule for dataset 'era5'")

    monkeypatch.setattr(store, "set_enabled", removed)
    for operation in ("pause", "resume"):
        response = client.post(f"/schedules/sync/era5/{operation}", headers={"Accept": "application/json"})
        assert response.status_code == 404 and "No stored schedule" in response.text


def test_concurrent_removal_during_delete_returns_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "delete_schedule", lambda _: False)
    assert client.delete("/schedules/sync/era5").status_code == 404
    response = client.post("/schedules/sync/era5/delete", data={"confirm": "yes"})
    assert response.status_code == 404


def test_an_unticked_box_saves_the_schedule_paused(dataset_pages: TestClient) -> None:
    created = dataset_pages.post(
        "/schedules/sync",
        data={"dataset_id": "era5", "cron": "0 6 * * *", "return_to": "dataset"},
        follow_redirects=False,
    )
    assert created.status_code == 303
    saved = store.get_schedule("era5")
    assert saved is not None and saved.enabled is False and saved.publish is False


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"frequency": "daily", "check_time": "06:15"}, "15 6 * * *"),
        ({"frequency": "weekly", "check_time": "09:00", "weekday": "fri"}, "0 9 * * fri"),
        ({"frequency": "monthly", "check_time": "12:30", "month_day": "28"}, "30 12 28 * *"),
        ({"frequency": "custom", "cron": "0 */6 * * *"}, "0 */6 * * *"),
    ],
)
def test_simple_schedule_choices_compile_to_cron(fields: dict[str, str], expected: str) -> None:
    assert cron_from_form(fields) == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"frequency": "weekly", "check_time": "25:00", "weekday": "mon"},
        {"frequency": "weekly", "check_time": "06:00", "weekday": "not-a-day"},
        {"frequency": "monthly", "check_time": "06:00", "month_day": "31"},
        {"frequency": "custom", "cron": ""},
    ],
)
def test_simple_schedule_choices_reject_invalid_values(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        cron_from_form(fields)


def test_cadence_suggestions_are_not_assumed_publication_times() -> None:
    assert suggested_frequency("daily") == "daily"
    assert suggested_frequency("weekly") == "daily"
    assert suggested_frequency("dekadal") == "weekly"
    assert suggested_frequency("monthly") == "weekly"
    assert suggested_frequency("yearly") == "monthly"
    assert form_values("0 6 * * fri", "daily")["frequency"] == "weekly"
    assert form_values("0 */6 * * *", "daily")["frequency"] == "custom"


@pytest.mark.parametrize(
    ("cron", "expected"),
    [
        ("0 6 * * *", "Every day at 06:00 (UTC)"),
        ("15 9 * * fri", "Every Friday at 09:15 (UTC)"),
        ("30 8 5 * *", "Day 5 of every month at 08:30 (UTC)"),
        ("*/2 * * * *", None),
    ],
)
def test_schedule_description_only_labels_simple_frequencies(cron: str, expected: str | None) -> None:
    assert schedule_description(cron, "UTC") == expected


def test_simple_schedule_form_saves_preset_and_keeps_existing_publication(dataset_pages: TestClient) -> None:
    created = dataset_pages.post(
        "/schedules/sync",
        data={
            "dataset_id": "era5",
            "frequency": "weekly",
            "check_time": "09:15",
            "weekday": "fri",
            "enabled": "on",
            "return_to": "dataset",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    saved = store.get_schedule("era5")
    assert saved is not None and saved.cron == "15 9 * * fri" and saved.publish is False
    page = dataset_pages.get("/datasets/era5", headers={"Accept": BROWSER})
    assert 'value="weekly" selected' in page.text and 'value="fri" selected' in page.text

    updated = dataset_pages.post(
        "/schedules/sync/era5",
        data={"frequency": "monthly", "check_time": "08:30", "month_day": "5", "enabled": "on"},
        follow_redirects=False,
    )
    assert updated.status_code == 303
    saved = store.get_schedule("era5")
    assert saved is not None and saved.cron == "30 8 5 * *" and saved.publish is False


def test_a_persisted_but_unapplied_schedule_is_reported_as_a_conflict(
    dataset_pages: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = scheduler_service.get_scheduler_service()

    def fail_reload() -> None:
        service._reload_error = "clock refused the new trigger"

    monkeypatch.setattr(service, "reload", fail_reload)
    response = dataset_pages.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"})
    assert response.status_code == 409
    assert response.json()["stored"] is True and response.json()["applied"] is False
    assert store.get_schedule("era5") is not None

    page = dataset_pages.post(
        "/schedules/sync/era5",
        data={"cron": "0 7 * * *", "enabled": "on", "return_to": "dataset"},
        follow_redirects=False,
    )
    assert page.status_code == 409
    assert "was saved, but the scheduler could not apply it" in page.text
    assert "clock refused the new trigger" in page.text


def test_deletion_refuses_to_report_success_when_clock_kept_old_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = scheduler_service.get_scheduler_service()
    store.save_schedule(_stored("era5"), create=True)

    def fail_reload() -> None:
        service._reload_error = "clock refused removal"

    monkeypatch.setattr(service, "reload", fail_reload)
    response = client.delete("/schedules/sync/era5")
    assert response.status_code == 409 and response.json()["stored"] is False
    assert store.get_schedule("era5") is None


def test_the_dataset_page_edits_the_single_stored_schedule(dataset_pages: TestClient) -> None:
    page = dataset_pages.get("/datasets/era5", headers={"Accept": BROWSER})
    assert "Not synced on a schedule" in page.text
    assert '<details class="schedule-editor"' in page.text
    assert ">Cancel</a>" not in page.text

    store.save_schedule(_stored("era5", cron="0 9 * * *"), create=True)
    page = dataset_pages.get("/datasets/era5", headers={"Accept": BROWSER})
    assert "0 9 * * *" in page.text and "overridden" not in page.text.lower()
    assert 'value="0 9 * * *"' in page.text and "Edit schedule" in page.text
    assert 'href="/datasets/era5?schedule_view=status#schedule-status" data-schedule-cancel>Cancel</a>' in page.text
    assert 'id="schedule-status">Automatic sync</h4>' in page.text
    assert 'url.searchParams.set("schedule_refresh", String(Date.now()))' in page.text

    cancelled = dataset_pages.get("/datasets/era5?schedule_view=status", headers={"Accept": BROWSER})
    assert 'value="0 9 * * *"' in cancelled.text
    assert '<details class="schedule-editor" open' not in cancelled.text
    assert store.get_schedule("era5").cron == "0 9 * * *"  # type: ignore[union-attr]


def test_the_schedules_page_lists_everything_and_sends_edits_to_the_dataset_page(
    dataset_pages: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = dataset_pages
    page = client.get("/schedules", headers={"Accept": BROWSER})
    assert page.status_code == 200 and "<h2" in page.text and ">Schedules</h2>" in page.text
    assert "No schedules yet" in page.text and "Scheduler off" in page.text
    assert "Add schedule" not in page.text and "<select" not in page.text and "pipeline" not in page.text.lower()
    assert client.get("/schedules/new", headers={"Accept": BROWSER}).status_code in {404, 405}

    store.save_schedule(_stored("era5"), create=True)
    store.save_schedule(_stored("gone", cron="*/2 * * * *"), create=True)
    listed = client.get("/schedules", headers={"Accept": BROWSER})
    assert "<td>Sync</td>" in listed.text
    assert "Every day at 06:00 (UTC)" in listed.text
    assert "*/2 * * * *" in listed.text
    assert 'href="/datasets/era5#schedule">Edit</a>' in listed.text
    assert 'action="/schedules/sync/era5/pause"' in listed.text and 'data-delete="era5"' in listed.text
    assert "no longer on the instance" in listed.text and 'data-delete="gone"' in listed.text
    assert 'href="/datasets/gone#schedule">Edit' not in listed.text
    assert 'href="/schedules/sync/gone/delete"' in listed.text
    confirmation = client.get("/schedules/sync/gone/delete", headers={"Accept": BROWSER})
    assert confirmation.status_code == 200
    assert 'action="/schedules/sync/gone/delete"' in confirmation.text
    assert 'name="confirm" value="yes"' in confirmation.text

    # The page's Pause form has no fields; a browser still posts it as a form, which this mirrors.
    paused = client.post(
        "/schedules/sync/era5/pause",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert paused.status_code == 303 and paused.headers["location"].endswith("/schedules")
    deleted = client.post("/schedules/sync/gone/delete", data={"confirm": "yes"}, follow_redirects=False)
    assert deleted.status_code == 303 and deleted.headers["location"].endswith("/schedules")
    assert store.get_schedule("gone") is None


def test_the_schedules_page_shows_all_entries_as_manageable(client: TestClient) -> None:
    store.save_schedule(_stored("chirps", cron="0 9 * * *"), create=True)
    page = client.get("/schedules", headers={"Accept": BROWSER})
    assert 'href="/datasets/chirps#schedule">Edit</a>' in page.text
    assert "Edit in config file" not in page.text and "Overridden" not in page.text


def test_the_menu_entry_is_schedules(client: TestClient) -> None:
    page = client.get("/schedules", headers={"Accept": BROWSER})
    assert ">Schedules<" in page.text and "Sync schedules" not in page.text


# --- review findings -----------------------------------------------------------------------------


def test_a_process_without_the_clock_lists_what_another_process_saved(instance: None) -> None:
    writer = SchedulerService(template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id))
    reader = SchedulerService(template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id))
    writer.start()
    reader.start()
    assert reader.status().schedules == []
    store.save_schedule(_stored("era5"), create=True)
    writer.reload()
    assert [item.dataset_id for item in reader.status().schedules] == ["era5"]
    assert reader.schedule_for("era5") is not None
    store.set_enabled("era5", False)
    assert reader.schedule_for("era5") is not None and reader.schedule_for("era5").enabled is False  # type: ignore[union-attr]


def test_an_unreadable_store_is_reported_and_writes_answer_503(client: TestClient) -> None:
    assert client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"}).status_code == 201
    store.schedules_path().write_text("not json", encoding="utf-8")
    listed = client.get("/schedules").json()
    assert listed["reload_error"] and "not valid JSON" in listed["reload_error"]
    assert [item["dataset_id"] for item in listed["schedules"]] == ["era5"]
    created = client.post("/schedules/sync", json={"dataset_id": "chirps", "cron": "0 6 * * *"})
    assert created.status_code == 503 and "cannot be read" in created.text
    paused = client.post("/schedules/sync/era5/pause", headers={"Accept": "application/json"})
    assert paused.status_code == 503
    assert client.put("/schedules/sync/era5", json={"cron": "0 7 * * *"}).status_code == 503
    assert client.delete("/schedules/sync/era5").status_code == 503


def test_a_partial_put_keeps_every_setting_it_does_not_name(client: TestClient) -> None:
    created = client.post(
        "/schedules/sync",
        json={"dataset_id": "era5", "cron": "0 6 * * *", "max_attempts": 5, "publish": False, "enabled": False},
    )
    assert created.status_code == 201
    updated = client.put("/schedules/sync/era5", json={"cron": "0 7 * * *"})
    assert updated.status_code == 200
    body = updated.json()
    assert body["cron"] == "0 7 * * *" and body["max_attempts"] == 5
    assert body["publish"] is False and body["enabled"] is False


def test_status_rows_name_their_kind(client: TestClient) -> None:
    client.post("/schedules/sync", json={"dataset_id": "era5", "cron": "0 6 * * *"})
    assert client.get("/schedules").json()["schedules"][0]["kind"] == "sync"
    assert client.get("/schedules/sync/era5").json()["kind"] == "sync"


def test_stored_schedule_timestamps_are_timezone_aware(instance: None) -> None:
    saved = store.save_schedule(_stored("era5"), create=True)
    assert saved.created_at.tzinfo is not None and saved.created_at <= datetime.now(timezone.utc)
