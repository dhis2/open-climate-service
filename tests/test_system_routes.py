import json
from collections.abc import Callable, Coroutine
from html.parser import HTMLParser
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
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
    monkeypatch.setattr(ingestion_services, "get_latest_artifact_for_dataset_or_404", lambda dataset_id: object())
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
    # Resolved before the stream opens, so an unknown id is a refusal rather than an event.
    monkeypatch.setattr(ingestion_services, "get_latest_artifact_for_dataset_or_404", lambda dataset_id: object())
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

    # A refusal, not a redirect: the page that posted shows the message in place, so the
    # stream only ever opens once the request is known to be runnable.
    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {"error": "Dataset ID is required"}


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

    # Ingestability is checked before the stream opens, so the stub needs a plugin.
    template = {
        "id": "chirps3_precipitation_daily",
        "name": "CHIRPS3 precipitation",
        "ingestion": {"plugin": "some.Plugin"},
    }
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
        # Ingestability is checked before the stream opens, so the stub needs a plugin.
        lambda dataset_id: {
            "id": dataset_id,
            "name": "CHIRPS3 precipitation",
            "ingestion": {"plugin": "some.Plugin"},
        },
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

    # Refused before the stream opens: inside it, this would arrive as an error event on a 200.
    assert response.status_code == 400
    # The rejection is dataset-aware: only a forecast (temporal_direction: future) may omit the
    # start, so a historical template is still refused.
    detail = json.loads(bytes(response.body))["error"]
    assert "Start period is required" in detail
    assert "chirps3_precipitation_daily" in detail


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


def test_the_shared_navigation_marks_the_current_page_and_carries_the_mount() -> None:
    nav = str(system_templates.page_nav("/ocs", "map"))

    assert 'href="/ocs/map"' in nav, "every link resolves under the mount prefix"
    assert 'aria-current="page"' in nav
    # The openEO editor is hosted elsewhere, so it opens in a new tab and is set apart.
    assert 'target="_blank" rel="noopener"' in nav
    assert '<li class="gap">' in nav


def test_the_navigation_links_only_where_this_instance_serves_a_page() -> None:
    """The rail grows as the pages it names arrive; a link to a 404 is worse than no link."""
    nav = str(system_templates.page_nav("", "map"))

    for path in ('href="/map"', 'href="/openeo"'):
        assert path in nav


def test_the_map_viewer_wears_the_shared_chrome(client: TestClient) -> None:
    response = client.get("/map")

    assert response.status_code == 200
    assert '<svg class="logo"' in response.text, "the mark is inline, so the page needs no asset"
    assert 'class="rail"' in response.text
    # The stylesheet is inlined rather than linked: the page must work with no outside request.
    assert "--theme-primary600" in response.text
    assert "{{" not in response.text, "no unrendered placeholder reaches the browser"


def test_the_header_is_a_way_back_to_the_landing_page(client: TestClient) -> None:
    """The mark and the instance name link home, as the title did before the header was shared."""
    body = client.get("/map").text

    assert '<a class="home" href="/">' in body
    assert body.index('<a class="home"') < body.index('<svg class="logo"'), "the mark is inside the link"


def test_the_map_viewer_opens_framed_on_the_instance_extent(client: TestClient) -> None:
    """The extent is read *before* the map is built, so the view never starts global.

    Fitting after construction showed a world view that then moved — which reads as the map
    wandering off while a dataset is already being requested.
    """
    body = client.get("/map").text

    assert "const bounds = await extentBounds();" in body
    assert "...(bounds ? { bounds } : { center: [20, 20], zoom: 1.5 })" in body
    # No animation: the opening view is the destination, not somewhere to travel to.
    assert "animate: false" in body
    assert "fitToExtent" not in body, "the post-build fit is what this replaces"


def test_the_map_viewer_reads_and_writes_the_dataset_in_the_address(client: TestClient) -> None:
    """`/map?dataset=<id>` opens on one dataset, and choosing one writes the parameter back."""
    body = client.get("/map", params={"dataset": "chirps3_precipitation_daily"}).text

    assert 'new URLSearchParams(window.location.search).get("dataset")' in body
    assert 'url.searchParams.set("dataset", id)' in body
    # Each option carries its collection id, which is what a deep link names.
    assert "opt.dataset.id = col.id;" in body


def test_the_map_panel_links_to_the_dataset_page(client: TestClient) -> None:
    """The other half of the round trip: the dataset page links to the map, and the map back."""
    body = client.get("/map").text

    assert 'id="dataset-link"' in body
    assert "datasetLink.href = `/datasets/${encodeURIComponent(id)}`" in body


def test_an_unpublished_dataset_in_the_address_is_reported_not_ignored(client: TestClient) -> None:
    body = client.get("/map").text

    assert "is not published, so it cannot be shown on the map" in body


def test_a_chosen_dataset_waits_for_the_style_as_a_deep_link_does(client: TestClient) -> None:
    """`initMap` returns once the map is constructed, not once its style has loaded.

    The catalogue can populate first, so a selection made straight away would reach
    `addLayer` mid-load, which MapLibre rejects. Both paths go through the same guard.
    """
    body = client.get("/map").text

    assert "whenMapReady(() => loadDataset(e.target.value));" in body
    assert "whenMapReady(() => loadDataset(option.value));" in body


def test_only_the_latest_selection_survives_the_wait_for_the_style(client: TestClient) -> None:
    """One listener, not one per selection.

    Registering a callback per selection meant two choices made before the style loaded both
    ran on `load`, each clearing and then racing to add the same layer id.
    """
    body = client.get("/map").text

    assert "let pendingWhenReady = null;" in body
    assert "if (alreadyWaiting) return;" in body
    # Emptying the selection cancels a load still waiting, rather than letting it arrive later.
    assert "pendingWhenReady = null;\n          clearDataset();" in body


def test_the_console_understands_the_stream_it_is_served(client: TestClient) -> None:
    """`/manage` is still served until it is removed, and consumes the same endpoints.

    It recognised only the old `redirect` event, so a successful run under the new contract
    reached EOF and reported "Sync ended unexpectedly", and a pre-stream refusal lost its
    reason to a generic "Request failed".
    """
    body = client.get("/manage").text

    assert "if (evt.finished)" in body
    assert "if (evt.error)" in body
    assert "refusal.error" in body, "the refusal's reason is read from the body"


def test_a_superseded_load_does_not_reach_the_map(client: TestClient) -> None:
    """The guard after the style has loaded, where `whenMapReady` no longer helps.

    Two selections in quick succession both run: whichever finishes last adds the shared
    `zarr-layer`, so without this the map can settle on the dataset that was not chosen.
    """
    body = client.get("/map").text

    assert "const generation = ++loadGeneration;" in body
    assert "const superseded = () => generation !== loadGeneration;" in body
    # Checked after every await, and again at the step that would actually go wrong.
    assert body.count("if (superseded()) return;") >= 4
    # Emptying the selection supersedes an in-flight load too, not only a pending one.
    assert "loadGeneration++;" in body


def test_map_viewer_initializes_at_latest_timestep(client: TestClient) -> None:
    response = client.get("/map")

    assert response.status_code == 200
    # The viewer renders one control per non-spatial dimension...
    assert "function renderDimControls()" in response.text
    # ...and a slider-type dimension (e.g. time) defaults to its last index (latest step).
    assert 'control === "slider" ? Math.max(0, count - 1) : 0' in response.text


@pytest.mark.anyio  # pyright: ignore[reportUntypedFunctionDecorator]
async def test_a_refusal_names_no_url_at_all() -> None:
    """What replaced the console redirects, and why the mount-prefix bug cannot return.

    A refusal used to be a 303 to `/manage?error=...`, which had to be built mount-relative or
    it 404'd behind a proxy (CLIM-974). The page that posts now shows the message itself, so the
    refusal carries a message and no location — there is no URL left to get wrong.
    """
    response = await system_routes.manage_sync(
        cast("Request", _FakeRequest({"dataset_id": "  "}, root_path="/ocs")),
    )

    assert response.status_code == 400
    assert "location" not in response.headers
    body = bytes(response.body).decode()
    assert json.loads(body) == {"error": "Dataset ID is required"}
    assert "/manage" not in body and "://" not in body
