import os
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from open_climate_service import config as api_config
from open_climate_service.main import app
from open_climate_service.openeo import earthkit_processes, xclim_processes
from open_climate_service.shared import compute, urls

_TEST_CONFIG = """\
extent:
  name: Sierra Leone
  bbox: [-13.5, 6.9, -10.1, 10.0]
  country_code: SLE
data_dir: ./data
"""


@pytest.fixture(autouse=True, scope="session")
def _test_open_climate_service_config(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    config_file = tmp_path_factory.mktemp("config") / "climate-service.yaml"
    config_file.write_text(_TEST_CONFIG, encoding="utf-8")
    old = os.environ.get("CLIMATE_SERVICE_CONFIG")
    os.environ["CLIMATE_SERVICE_CONFIG"] = str(config_file)
    yield
    if old is None:
        os.environ.pop("CLIMATE_SERVICE_CONFIG", None)
    else:
        os.environ["CLIMATE_SERVICE_CONFIG"] = old


@pytest.fixture(autouse=True)
def _reset_config_cache() -> Generator[None, None, None]:
    api_config._cache = None
    xclim_processes._cache = None
    earthkit_processes._cache = None
    # Warnings about an unusable base URL are emitted once per distinct value for the life of
    # the process, so without this a test asserting on one depends on whether an earlier test
    # happened to use the same value.
    urls._forget_base_url_warnings()
    # A job a test leaves running would otherwise hold its slot for every later test.
    compute.reset_job_slots()
    yield
    api_config._cache = None
    xclim_processes._cache = None
    earthkit_processes._cache = None
    urls._forget_base_url_warnings()


@pytest.fixture(autouse=True)
def _own_operational_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give each test an empty operational database: tasks, exports, runs and the clock lease.

    The session shares one data directory, so without this a task, an export or an unexpired
    clock lease one test leaves behind would decide what the next test sees.
    """
    from open_climate_service.state import db

    path = tmp_path / "ocs.db"
    monkeypatch.setattr(db, "database_path", lambda: path)


@pytest.fixture(autouse=True)
def _exports_declared_in_the_config_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read named exports from the config cache when a test declares them there.

    Exports live in the operational database (CLIM-1378), and ``climate-service.yaml`` refuses an
    ``exports`` block. Many tests predate that and declare, then mutate, exports in the config
    cache; for those, this reads the cache. A test that does not put exports there uses the store.
    """
    from open_climate_service.exports import store

    stored = store.list_definitions

    def definitions() -> list[dict[str, object]]:
        cache = api_config._cache
        if isinstance(cache, dict) and "exports" in cache:
            return cache["exports"]  # type: ignore[no-any-return]
        return stored()

    monkeypatch.setattr(store, "list_definitions", definitions)


@pytest.fixture(autouse=True)
def _unset_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve tests from the request's own origin, whatever the developer's environment says.

    `CLIMATE_SERVICE_BASE_URL` is documented in `.env.example` and loaded by compose via
    `env_file`, so it is routinely exported on a machine that runs an instance. Every route
    that builds an absolute URL reads it, so with it set the assertions that pin
    `http://testserver/...` fail — a test run whose result depends on the shell it was started
    from. Tests that need a configured origin set it themselves.

    `ROOT_PATH` is the same hazard one step earlier: `create_app()` reads it when `main` is
    imported, so the shared `app` above already carries the developer's prefix by the time this
    fixture runs, and unsetting the variable alone leaves every route rendering it. The
    attribute is reset too, since FastAPI copies `app.root_path` into each request scope.
    """
    monkeypatch.delenv("CLIMATE_SERVICE_BASE_URL", raising=False)
    monkeypatch.delenv("ROOT_PATH", raising=False)
    monkeypatch.setattr(app, "root_path", "")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


MountedClientFactory = Callable[..., TestClient]


@pytest.fixture
def mounted_client_factory() -> MountedClientFactory:
    """Build a client that reaches the app the way uvicorn does under `--root-path`.

    Uvicorn sets `scope["root_path"]` *and* prepends it to `scope["path"]`, so `/manage` under
    `/ocs` arrives as `/ocs/manage`. `TestClient(root_path=...)` only sets the former, which
    makes any prefix-stripping code a no-op under test and lets a test pass with the strip
    removed. The prefix is passed verbatim, so a trailing slash reaches the app as it would in
    production.
    """

    def make(root_path: str = "/ocs", target: ASGIApp = app) -> TestClient:
        async def uvicorn_shaped(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                scope = {**scope, "root_path": root_path, "path": root_path + scope["path"]}
            await target(scope, receive, send)

        return TestClient(uvicorn_shaped)

    return make
