from collections.abc import Callable, Coroutine
from html.parser import HTMLParser
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.datastructures import URL
from starlette.responses import StreamingResponse

from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.system import routes as system_routes
from open_climate_service.system import templates as system_templates


class _ManageHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_sync_form: dict[str, str] | None = None
        self.sync_forms_by_dataset_id: dict[str, dict[str, str]] = {}
        self.sync_triggers: dict[str, dict[str, str]] = {}
        self.cancel_buttons: dict[str, dict[str, str]] = {}

    @staticmethod
    def _has_class(attr_map: dict[str, str], class_name: str) -> bool:
        return class_name in attr_map.get("class", "").split()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value for key, value in attrs if value is not None}
        if tag == "form" and self._has_class(attr_map, "sync-form") and "data-trigger-id" in attr_map:
            self.current_sync_form = attr_map
        if (
            tag == "input"
            and self.current_sync_form is not None
            and attr_map.get("type") == "hidden"
            and attr_map.get("name") == "dataset_id"
            and "value" in attr_map
        ):
            self.sync_forms_by_dataset_id[attr_map["value"]] = self.current_sync_form
        if tag == "button" and "data-dataset-id" in attr_map and attr_map.get("id", "").startswith("sync-trigger-"):
            self.sync_triggers[attr_map["data-dataset-id"]] = attr_map
        if tag == "button" and self._has_class(attr_map, "secondary-btn") and "data-dataset-id" in attr_map:
            self.cancel_buttons[attr_map["data-dataset-id"]] = attr_map

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.current_sync_form = None


class _FakeRequest:
    """Enough of a Request for the /manage form handlers.

    `scope` is part of that surface, not an extra: the handlers read `root_path` from it to
    build mount-relative redirects, so a double without it passes tests the real object would
    fail. Defaults to an unmounted instance; pass `root_path` for a prefixed deployment.
    """

    def __init__(self, form_data: dict[str, str], root_path: str = "") -> None:
        self._form_data = form_data
        self.base_url = URL("http://testserver/")
        self.scope = {"root_path": root_path}

    async def form(self) -> dict[str, str]:
        return self._form_data


@pytest.fixture(autouse=True)
def _clear_template_cache() -> None:
    system_templates._cache.clear()


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_sync_forwards_provided_end(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    scheduled: list[Coroutine[object, object, None]] = []

    def fake_sync_dataset(
        *,
        dataset_id: str,
        end: str | None,
        publish: bool,
        on_progress: Callable[[int | None, int | None, str | None], None],
    ) -> None:
        captured["dataset_id"] = dataset_id
        captured["end"] = end
        captured["publish"] = publish
        on_progress(1, 1, "done")

    async def fake_to_thread(func: Callable[[], None]) -> None:
        func()

    def fake_create_task(coro: Coroutine[object, object, None]) -> None:
        scheduled.append(coro)
        return None

    monkeypatch.setattr(ingestion_services, "sync_dataset", fake_sync_dataset)
    monkeypatch.setattr(system_routes.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(system_routes.asyncio, "create_task", fake_create_task)

    response = await system_routes.manage_sync(
        cast(Request, _FakeRequest({"dataset_id": "chirps3_precipitation_daily", "end": "2026-02-10", "publish": "on"}))
    )

    assert isinstance(response, StreamingResponse)
    assert len(scheduled) == 1
    await scheduled[0]
    assert captured == {
        "dataset_id": "chirps3_precipitation_daily",
        "end": "2026-02-10",
        "publish": True,
    }


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_sync_treats_blank_end_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    scheduled: list[Coroutine[object, object, None]] = []

    def fake_sync_dataset(
        *,
        dataset_id: str,
        end: str | None,
        publish: bool,
        on_progress: Callable[[int | None, int | None, str | None], None],
    ) -> None:
        captured["dataset_id"] = dataset_id
        captured["end"] = end
        captured["publish"] = publish
        on_progress(1, 1, "done")

    async def fake_to_thread(func: Callable[[], None]) -> None:
        func()

    def fake_create_task(coro: Coroutine[object, object, None]) -> None:
        scheduled.append(coro)
        return None

    monkeypatch.setattr(ingestion_services, "sync_dataset", fake_sync_dataset)
    monkeypatch.setattr(system_routes.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(system_routes.asyncio, "create_task", fake_create_task)

    response = await system_routes.manage_sync(
        cast(Request, _FakeRequest({"dataset_id": "  chirps3_precipitation_daily  ", "end": "", "publish": "on"}))
    )

    assert isinstance(response, StreamingResponse)
    assert len(scheduled) == 1
    await scheduled[0]
    assert captured == {
        "dataset_id": "chirps3_precipitation_daily",
        "end": None,
        "publish": True,
    }


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_sync_rejects_blank_dataset_id() -> None:
    response = await system_routes.manage_sync(
        cast(
            Request,
            _FakeRequest(
                {
                    "dataset_id": "   ",
                    "end": "",
                }
            ),
        )
    )

    assert response.status_code == 303
    # Relative on purpose: a redirect back to the console must land on the origin the operator
    # actually reached, not on the configured public one (CLIM-974 review).
    assert response.headers["location"] == "/manage?error=Dataset%20ID%20is%20required"


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_ingest_strips_string_inputs_and_treats_blank_end_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    scheduled: list[Coroutine[object, object, None]] = []

    def fake_create_artifact(
        *,
        dataset: dict[str, object],
        start: str,
        end: str | None,
        bbox: list[float],
        country_code: str | None,
        overwrite: bool,
        publish: bool,
        on_progress: Callable[[int | None, int | None, str | None], None],
    ) -> None:
        captured["dataset"] = dataset
        captured["start"] = start
        captured["end"] = end
        captured["bbox"] = bbox
        captured["country_code"] = country_code
        captured["overwrite"] = overwrite
        captured["publish"] = publish
        on_progress(1, 1, "done")

    async def fake_to_thread(func: Callable[[], None]) -> None:
        func()

    def fake_create_task(coro: Coroutine[object, object, None]) -> None:
        scheduled.append(coro)
        return None

    template = {"id": "chirps3_precipitation_daily", "name": "CHIRPS3 precipitation"}
    monkeypatch.setattr(system_routes.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(system_routes.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(ingestion_services, "create_artifact", fake_create_artifact)
    monkeypatch.setattr(
        "open_climate_service.data_registry.services.datasets.get_dataset",
        lambda dataset_id: template if dataset_id == template["id"] else None,
    )
    monkeypatch.setattr(
        "open_climate_service.extents.services.get_extent_or_404",
        lambda: {"bbox": [0.0, 1.0, 2.0, 3.0], "country_code": "SLE"},
    )

    response = await system_routes.manage_ingest(
        cast(
            Request,
            _FakeRequest(
                {
                    "dataset_id": "  chirps3_precipitation_daily  ",
                    "start": " 2024-02-01 ",
                    "end": "   ",
                    "publish": "on",
                }
            ),
        )
    )

    assert isinstance(response, StreamingResponse)
    assert len(scheduled) == 1
    await scheduled[0]
    assert captured == {
        "dataset": template,
        "start": "2024-02-01",
        "end": None,
        "bbox": [0.0, 1.0, 2.0, 3.0],
        "country_code": "SLE",
        "overwrite": False,
        "publish": True,
    }


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_ingest_rejects_blank_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "open_climate_service.data_registry.services.datasets.get_dataset",
        lambda dataset_id: {"id": dataset_id, "name": "CHIRPS3 precipitation"},
    )
    monkeypatch.setattr(
        "open_climate_service.extents.services.get_extent_or_404",
        lambda: {"bbox": [0.0, 1.0, 2.0, 3.0], "country_code": "SLE"},
    )

    response = await system_routes.manage_ingest(
        cast(
            Request,
            _FakeRequest(
                {
                    "dataset_id": "chirps3_precipitation_daily",
                    "start": "   ",
                    "end": "",
                }
            ),
        )
    )

    assert response.status_code == 303
    # The rejection is now dataset-aware: only a forecast (temporal_direction: future) may
    # omit the start, so a historical template still gets a redirect with an error.
    assert "Start%20period%20is%20required" in response.headers["location"]
    assert "chirps3_precipitation_daily" in response.headers["location"]


def test_manage_page_shows_split_publication_and_sync_columns(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = "chirps3_precipitation_daily'quoted"
    monkeypatch.setattr(system_templates, "_load_templates", lambda: [])
    monkeypatch.setattr(system_templates, "_load_extent", lambda: {"id": "sle", "name": "Sierra Leone", "bbox": []})
    monkeypatch.setattr(
        system_templates,
        "_load_datasets",
        lambda: [
            type(
                "Dataset",
                (),
                {
                    "dataset_id": dataset_id,
                    "dataset_name": "CHIRPS3 precipitation",
                    "period_type": "daily",
                    "extent": type(
                        "Extent",
                        (),
                        {"temporal": type("Temporal", (), {"start": "2026-01-01", "end": "2026-01-10"})()},
                    )(),
                    "publication": type("Publication", (), {"status": "published"})(),
                },
            )()
        ],
    )

    response = client.get("/manage")

    assert response.status_code == 200

    parser = _ManageHtmlParser()
    parser.feed(response.text)
    sync_form_attrs = parser.sync_forms_by_dataset_id.get(dataset_id)
    sync_trigger_attrs = parser.sync_triggers.get(dataset_id)
    cancel_button_attrs = parser.cancel_buttons.get(dataset_id)

    assert "<th>Publication</th>" in response.text
    assert "<th>Sync</th>" in response.text
    assert "Start sync" in response.text
    assert "Cutoff end" in response.text
    assert sync_form_attrs is not None
    assert sync_form_attrs["data-trigger-id"].startswith("sync-trigger-sync-row-")
    assert sync_form_attrs["data-progress-id"].startswith("sync-progress-sync-row-")
    assert sync_form_attrs["data-status-id"].startswith("sync-status-sync-row-")
    assert "runJob(" in sync_form_attrs["onsubmit"]
    assert "this.dataset.triggerId" in sync_form_attrs["onsubmit"]
    assert "this.dataset.progressId" in sync_form_attrs["onsubmit"]
    assert "this.dataset.statusId" in sync_form_attrs["onsubmit"]
    assert sync_trigger_attrs is not None
    assert sync_trigger_attrs["data-dataset-id"] == dataset_id
    assert sync_trigger_attrs["data-sync-dom-id"].startswith("sync-row-")
    assert sync_trigger_attrs["onclick"] == "openSyncPanel(this.dataset.syncDomId)"
    assert cancel_button_attrs is not None
    assert cancel_button_attrs["data-dataset-id"] == dataset_id
    assert cancel_button_attrs["data-sync-dom-id"] == sync_trigger_attrs["data-sync-dom-id"]
    assert cancel_button_attrs["onclick"] == "closeSyncPanel(this.dataset.syncDomId)"
    assert "function restoreJobControls(controls, btn, status)" in response.text
    assert "label.textContent = 'Error: Sync ended unexpectedly.';" in response.text
    assert "const message = err instanceof Error ? err.message : String(err);" in response.text


def test_ingestable_templates_excludes_static_workflow_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Templates with no ingestion.plugin (e.g. workflow outputs) are not ingestable."""
    monkeypatch.setattr(
        system_templates,
        "_load_templates",
        lambda: [
            {"id": "chirps3_precipitation_daily", "ingestion": {"plugin": "pkg.Plugin"}},
            {"id": "worldpop_population_yearly", "ingestion": {"plugin": "pkg.Plugin"}},
            {"id": "worldpop_population_change", "sync": {"kind": "static"}},  # derived, no plugin
        ],
    )

    ingestable_ids = [t["id"] for t in system_templates._ingestable_templates()]

    assert "worldpop_population_change" not in ingestable_ids
    assert ingestable_ids == ["chirps3_precipitation_daily", "worldpop_population_yearly"]


def test_manage_page_dropdown_excludes_static_templates(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The /manage ingest dropdown lists only ingestable templates."""
    monkeypatch.setattr(
        system_templates,
        "_load_templates",
        lambda: [
            {
                "id": "chirps3_precipitation_daily",
                "name": "Total precipitation (CHIRPS3)",
                "ingestion": {"plugin": "p"},
            },
            {
                "id": "worldpop_population_change",
                "name": "Population change (WorldPop Global2)",
                "sync": {"kind": "static"},
            },
        ],
    )
    monkeypatch.setattr(system_templates, "_load_extent", lambda: {"id": "sle", "name": "Sierra Leone", "bbox": []})
    monkeypatch.setattr(system_templates, "_load_datasets", lambda: [])

    response = client.get("/manage")

    assert response.status_code == 200
    assert 'value="chirps3_precipitation_daily"' in response.text
    assert 'value="worldpop_population_change"' not in response.text


def test_map_viewer_initializes_at_latest_timestep(client: TestClient) -> None:
    response = client.get("/map")

    assert response.status_code == 200
    # The viewer renders one control per non-spatial dimension...
    assert "function renderDimControls()" in response.text
    # ...and a slider-type dimension (e.g. time) defaults to its last index (latest step).
    assert 'control === "slider" ? Math.max(0, count - 1) : 0' in response.text


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_manage_redirects_keep_the_mount_prefix() -> None:
    """A POST to `/ocs/manage/sync` that redirects to `/manage` drops the prefix and the proxy
    returns 404. The Location has to be mount-relative — and still carry no origin, so it
    lands on the host the operator actually reached."""
    response = await system_routes.manage_sync(
        cast("Request", _FakeRequest({"dataset_id": "  "}, root_path="/ocs")),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/ocs/manage?error=")
    assert "://" not in response.headers["location"]


@pytest.mark.anyio
async def test_a_successful_sync_redirects_under_the_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """The success redirect is the one a mounted deployment actually reaches on the happy path.

    The error redirects were mount-relative while this one was not, so under `/ocs` a sync that
    worked sent the browser to `/manage` and the proxy 404'd — a failure only visible when nothing
    had gone wrong.
    """
    scheduled: list[Coroutine[object, object, None]] = []

    def fake_sync_dataset(
        *,
        dataset_id: str,
        end: str | None,
        publish: bool,
        on_progress: Callable[[int | None, int | None, str | None], None],
    ) -> None:
        on_progress(1, 1, "done")

    async def fake_to_thread(func: Callable[[], None]) -> None:
        func()

    def fake_create_task(coro: Coroutine[object, object, None]) -> None:
        scheduled.append(coro)
        return None

    monkeypatch.setattr(ingestion_services, "sync_dataset", fake_sync_dataset)
    monkeypatch.setattr(system_routes.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(system_routes.asyncio, "create_task", fake_create_task)

    response = await system_routes.manage_sync(
        cast(Request, _FakeRequest({"dataset_id": "chirps3_precipitation_daily"}, root_path="/ocs"))
    )
    await scheduled[0]
    payload = "".join([chunk.decode() if isinstance(chunk, bytes) else chunk async for chunk in response.body_iterator])

    assert "/ocs/manage?message=Sync+completed" in payload
    assert '"/manage?message' not in payload, "the bare path would 404 behind the proxy"
