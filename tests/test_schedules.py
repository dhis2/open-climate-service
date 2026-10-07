"""Sync schedules saved from the page and the API, merged with the file's (CLIM-1242)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.scheduler import service as scheduler_service
from open_climate_service.scheduler import store
from open_climate_service.scheduler.config import DatasetSyncSchedule, SchedulerConfig, merge_schedules
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
    path.write_text('{"chirps": {"dataset_id": "era5", "cron": "0 6 * * *"}}', encoding="utf-8")
    with pytest.raises(ScheduleStoreUnreadable, match="names dataset"):
        store.get_schedule("chirps")


# --- merge ---------------------------------------------------------------------------------------


def test_file_entries_come_first_and_shadow_stored_ones() -> None:
    config = SchedulerConfig(enabled=True, dataset_sync=[_file("chirps")])
    merged = merge_schedules(
        config, [_stored("era5"), _stored("chirps", cron="0 9 * * *"), _stored("x", enabled=False)]
    )
    assert [(item.dataset_id, item.source, item.shadowed, item.effective) for item in merged] == [
        ("chirps", "file", False, True),
        ("chirps", "store", True, False),
        ("era5", "store", False, True),
        ("x", "store", False, False),
    ]
    assert merged[0].schedule.cron == "0 5 * * *" and merged[1].schedule.cron == "0 9 * * *"


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


def test_start_registers_file_and_active_stored_entries_only(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True, dataset_sync=[_file("chirps")]),
        store_loader=lambda: [_stored("era5"), _stored("chirps", cron="0 9 * * *"), _stored("x", enabled=False)],
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id, {"id": dataset_id, "sync": {"kind": "temporal"}}),
    )
    service.start()
    registered = _added(scheduler)
    assert registered == ["dataset-sync:chirps", "dataset-sync:era5"]
    watch = next(call for call in scheduler.add_job.call_args_list if call.kwargs["id"] == "scheduler:store-watch")
    assert watch.args[0] == service.reload_if_changed
    status = service.status()
    assert [(item.dataset_id, item.source, item.effective) for item in status.schedules] == [
        ("chirps", "file", True),
        ("chirps", "store", False),
        ("era5", "store", True),
        ("x", "store", False),
    ]
    assert status.reload_error is None


def test_start_survives_an_unreadable_store_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)

    def broken() -> list[StoredSchedule]:
        raise ScheduleStoreUnreadable("schedules.json is not valid JSON")

    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True, dataset_sync=[_file("chirps")]),
        store_loader=broken,
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    assert _added(scheduler) == ["dataset-sync:chirps"]
    status = service.status()
    assert status.reload_error and "not valid JSON" in status.reload_error
    assert [item.dataset_id for item in status.schedules] == ["chirps"]


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


def test_shadowed_and_paused_rows_carry_no_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    next_check = datetime(2026, 10, 8, 6, tzinfo=timezone.utc)
    scheduler.get_jobs.return_value = [MagicMock(id="dataset-sync:chirps", next_run_time=next_check)]
    service = SchedulerService(
        config_loader=lambda: SchedulerConfig(enabled=True, dataset_sync=[_file("chirps")]),
        store_loader=lambda: [_stored("chirps", cron="0 9 * * *"), _stored("era5", enabled=False)],
        template_loader=lambda dataset_id: _TEMPLATES.get(dataset_id),
    )
    service.start()
    service.check_now(_file("chirps"))
    rows = {(item.dataset_id, item.source): item for item in service.status().schedules}
    winner = rows[("chirps", "file")]
    assert winner.registered and winner.next_check == next_check and winner.last_outcome is not None
    shadowed = rows[("chirps", "store")]
    assert shadowed.shadowed and not shadowed.registered and shadowed.next_check is None
    assert shadowed.last_outcome is None and shadowed.last_check is None
    paused = rows[("era5", "store")]
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
    created = client.post("/schedules", json={"dataset_id": "era5", "cron": "0 6 * * *"})
    assert created.status_code == 201, created.text
    assert created.json()["source"] == "store" and created.json()["enabled"] is True
    assert client.post("/schedules", json={"dataset_id": "era5", "cron": "0 6 * * *"}).status_code == 409
    listed = client.get("/schedules").json()
    assert [item["dataset_id"] for item in listed["schedules"]] == ["era5"] and listed["reload_error"] is None
    one = client.get("/schedules/era5")
    assert one.status_code == 200 and one.json()["cron"] == "0 6 * * *"
    assert client.get("/schedules/missing").status_code == 404

    updated = client.put("/schedules/era5", json={"cron": "0 7 * * *", "max_attempts": 5, "publish": False})
    assert updated.status_code == 200 and updated.json()["cron"] == "0 7 * * *"
    assert updated.json()["max_attempts"] == 5 and updated.json()["publish"] is False
    assert client.put("/schedules/missing", json={"cron": "0 7 * * *"}).status_code == 404

    paused = client.post("/schedules/era5/pause", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["enabled"] is False and paused.json()["effective"] is False
    resumed = client.post("/schedules/era5/resume", headers={"Accept": "application/json"})
    assert resumed.status_code == 200 and resumed.json()["effective"] is True

    assert client.delete("/schedules/era5").status_code == 204
    assert client.delete("/schedules/era5").status_code == 404
    assert client.get("/schedules").json()["schedules"] == []


def test_api_refuses_what_the_clock_could_not_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    static = client.post("/schedules", json={"dataset_id": "worldpop", "cron": "0 6 * * *"})
    assert static.status_code == 422 and "not syncable" in static.text
    unknown = client.post("/schedules", json={"dataset_id": "nope", "cron": "0 6 * * *"})
    assert unknown.status_code == 422 and "no registered data source" in unknown.text
    bad_cron = client.post("/schedules", json={"dataset_id": "era5", "cron": "every day"})
    assert bad_cron.status_code == 422 and "cron" in bad_cron.text

    monkeypatch.setattr(
        api_config,
        "_cache",
        {"scheduler": {"enabled": False, "dataset_sync": [{"dataset_id": "chirps", "cron": "0 5 * * *"}]}},
    )
    monkeypatch.setattr(scheduler_service, "_service", None)
    shadowed = client.post("/schedules", json={"dataset_id": "chirps", "cron": "0 6 * * *"})
    assert shadowed.status_code == 409 and "climate-service.yaml" in shadowed.text
    assert client.put("/schedules/chirps", json={"cron": "0 6 * * *"}).status_code == 409
    assert client.delete("/schedules/chirps").status_code == 409
    listed = client.get("/schedules").json()["schedules"]
    assert [(item["dataset_id"], item["source"]) for item in listed] == [("chirps", "file")]


def test_a_shadowed_stored_schedule_stays_editable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_config,
        "_cache",
        {"scheduler": {"enabled": False, "dataset_sync": [{"dataset_id": "chirps", "cron": "0 5 * * *"}]}},
    )
    monkeypatch.setattr(scheduler_service, "_service", None)
    store.save_schedule(_stored("chirps", cron="0 9 * * *"), create=True)
    updated = client.put("/schedules/chirps", json={"cron": "0 10 * * *"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["source"] == "store" and updated.json()["shadowed"] is True
    assert updated.json()["cron"] == "0 10 * * *" and updated.json()["effective"] is False
    assert client.get("/schedules/chirps").json()["source"] == "file"
    paused = client.post("/schedules/chirps/pause", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["enabled"] is False
    edit = client.get("/schedules/chirps/edit", headers={"Accept": BROWSER})
    assert edit.status_code == 200 and 'value="0 10 * * *"' in edit.text
    assert client.delete("/schedules/chirps").status_code == 204
    assert [item["source"] for item in client.get("/schedules").json()["schedules"]] == ["file"]


def test_a_saved_schedule_reaches_the_running_clock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _fake_scheduler(monkeypatch)
    monkeypatch.setattr(api_config, "_cache", {"scheduler": {"enabled": True}})
    monkeypatch.setattr(scheduler_service, "_service", None)
    scheduler_service.get_scheduler_service().start()
    assert _added(scheduler) == []

    assert client.post("/schedules", json={"dataset_id": "era5", "cron": "0 6 * * *"}).status_code == 201
    assert _added(scheduler) == ["dataset-sync:era5"]
    scheduler.get_job.return_value = MagicMock()
    assert client.post("/schedules/era5/pause", headers={"Accept": "application/json"}).status_code == 200
    scheduler.remove_job.assert_called_once_with("dataset-sync:era5")


# --- pages ---------------------------------------------------------------------------------------


def test_page_lists_adds_edits_and_deletes(client: TestClient) -> None:
    page = client.get("/schedules", headers={"Accept": BROWSER})
    assert page.status_code == 200 and "Sync schedules" in page.text and "No schedules yet" in page.text
    assert "Scheduler off" in page.text
    assert "Each check starts a sync job" in page.text
    assert "enabled: true" in page.text and "climate-service.yaml" in page.text
    assert "pipeline" not in page.text.lower()

    form = client.get("/schedules/new?dataset=era5", headers={"Accept": BROWSER})
    assert form.status_code == 200 and "New schedule" in form.text
    assert '<option value="era5" selected' in form.text and "CHIRPS" in form.text

    created = client.post(
        "/schedules",
        data={"dataset_id": "era5", "cron": "0 6 * * *", "publish": "on", "max_attempts": "2"},
        follow_redirects=False,
    )
    assert created.status_code == 303 and created.headers["location"].endswith("/schedules")
    listed = client.get("/schedules", headers={"Accept": BROWSER})
    assert "0 6 * * *" in listed.text and "this page" in listed.text and ">Pause<" in listed.text
    # Delete from the list opens a confirmation; without scripts it lands on the edit view's panel.
    assert 'data-delete="era5"' in listed.text and 'href="/schedules/era5/edit#delete"' in listed.text
    assert '<dialog id="delete-dialog"' in listed.text and 'name="confirm" value="yes"' in listed.text

    refused = client.post("/schedules", data={"dataset_id": "worldpop", "cron": "0 6 * * *"})
    assert refused.status_code == 400 and "not syncable" in refused.text and 'value="0 6 * * *"' in refused.text

    assert ">Cancel<" in form.text
    edit = client.get("/schedules/era5/edit", headers={"Accept": BROWSER})
    assert edit.status_code == 200 and "Edit schedule" in edit.text and "readonly" in edit.text
    assert 'id="delete"' in edit.text and "I understand" in edit.text and "I understand" not in listed.text
    saved = client.post("/schedules/era5", data={"cron": "0 7 * * *", "max_attempts": "3"}, follow_redirects=False)
    assert saved.status_code == 303 and store.get_schedule("era5").cron == "0 7 * * *"  # type: ignore[union-attr]
    assert store.get_schedule("era5").publish is False  # type: ignore[union-attr]

    unconfirmed = client.post("/schedules/era5/delete", data={}, follow_redirects=False)
    assert unconfirmed.status_code == 400 and store.get_schedule("era5") is not None
    confirmed = client.post("/schedules/era5/delete", data={"confirm": "yes"}, follow_redirects=False)
    assert confirmed.status_code == 303 and store.get_schedule("era5") is None


def test_page_shows_file_entries_read_only_and_shadowing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_config,
        "_cache",
        {"scheduler": {"enabled": False, "dataset_sync": [{"dataset_id": "chirps", "cron": "0 5 * * *"}]}},
    )
    monkeypatch.setattr(scheduler_service, "_service", None)
    store.save_schedule(_stored("chirps", cron="0 9 * * *"), create=True)
    page = client.get("/schedules", headers={"Accept": BROWSER})
    assert "climate-service.yaml" in page.text and "Edit in config file" in page.text and "Overridden" in page.text
    form = client.get("/schedules/new", headers={"Accept": BROWSER})
    assert '<option value="chirps" disabled' in form.text or 'value="chirps"  disabled' in form.text


def test_dataset_page_names_the_schedule_or_offers_one(instance: None) -> None:
    from open_climate_service.system import templates as landing
    from tests.test_dataset_page import _record

    html = landing.render_dataset_page(_record("era5"), "/ocs")
    assert "Not synced on a schedule" in html and "/ocs/schedules/new?dataset=era5" in html
    store.save_schedule(_stored("era5", cron="0 6 * * *"), create=True)
    scheduler_service.get_scheduler_service().reload()
    html = landing.render_dataset_page(_record("era5"), "/ocs")
    assert "Checked on" in html and "0 6 * * *" in html
    store.set_enabled("era5", False)
    scheduler_service.get_scheduler_service().reload()
    assert "paused" in landing.render_dataset_page(_record("era5"), "/ocs")


def test_stored_schedule_timestamps_are_timezone_aware(instance: None) -> None:
    saved = store.save_schedule(_stored("era5"), create=True)
    assert saved.created_at.tzinfo is not None and saved.created_at <= datetime.now(timezone.utc)
