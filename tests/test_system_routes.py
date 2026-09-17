import json
from collections.abc import Callable, Coroutine
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.system import routes as system_routes
from open_climate_service.system import templates as system_templates


class _FakeRequest:
    """Enough of a Request for the ingest and sync form handlers.

    `scope` is part of the real object's surface; kept so the double does not quietly narrow it.
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

    # A refusal, not a redirect: the page that posted shows the message in place.
    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {"error": "Dataset ID is required"}
    assert "location" not in response.headers


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

    # Only a forecast (temporal_direction: future) may omit the start, so a historical source is
    # refused before any stream starts.
    assert response.status_code == 400
    error = json.loads(bytes(response.body))["error"]
    assert "Start period is required" in error
    assert "chirps3_precipitation_daily" in error


def test_map_viewer_initializes_at_latest_timestep(client: TestClient) -> None:
    response = client.get("/map")

    assert response.status_code == 200
    # The viewer renders one control per non-spatial dimension...
    assert "function renderDimControls()" in response.text
    # ...and a slider-type dimension (e.g. time) defaults to its last index (latest step).
    assert 'control === "slider" ? Math.max(0, count - 1) : 0' in response.text


@pytest.mark.anyio
async def test_a_successful_sync_ends_the_stream_with_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stream reports completion itself rather than naming a page to go to.

    The page that posted decides what happens next (the dataset page reloads), so no URL is sent,
    and nothing in the stream can point at a path that a mounted deployment does not serve.
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
    # Narrowed rather than accessed directly: the endpoint is typed `-> Response`, and only a
    # StreamingResponse carries the SSE body this assertion reads.
    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]
    payload = "".join(chunk.decode() if isinstance(chunk, bytes) else str(chunk) for chunk in chunks)

    events = [json.loads(line[len("data: ") :]) for line in payload.splitlines() if line.startswith("data: ")]
    assert events[-1] == {"finished": True, "message": "Sync completed"}
    assert "/manage" not in payload
