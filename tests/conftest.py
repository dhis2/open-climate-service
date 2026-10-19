import atexit
import os
import shutil
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

_TEST_CONFIG = """\
extent:
  name: Sierra Leone
  bbox: [-13.5, 6.9, -10.1, 10.0]
  country_code: SLE
data_dir: ./data
"""

# The suite's own instance, set before anything from open_climate_service is imported. Store
# paths such as ARTIFACTS_DIR are resolved from the config when their module is imported, so
# a config set later, by a fixture, comes too late: they would already name the developer's
# `data_dir` from `.env`, or the XDG fallback without one. `load_dotenv` does not overwrite a
# variable that is already set, so `.env` cannot take this back.
_TEST_INSTANCE = Path(tempfile.mkdtemp(prefix="ocs-test-instance-"))
atexit.register(shutil.rmtree, _TEST_INSTANCE, ignore_errors=True)
TEST_CONFIG_FILE = _TEST_INSTANCE / "climate-service.yaml"
TEST_DATA_ROOT = _TEST_INSTANCE / "data"
# The caller's own value, restored when a session ends, so a process that runs pytest and then
# carries on (an IDE calling pytest.main(), a notebook) is not left pointing at the test instance.
_CALLERS_CONFIG = os.environ.get("CLIMATE_SERVICE_CONFIG")


def _use_test_instance() -> None:
    TEST_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEST_CONFIG_FILE.write_text(_TEST_CONFIG, encoding="utf-8")
    os.environ["CLIMATE_SERVICE_CONFIG"] = str(TEST_CONFIG_FILE)


_use_test_instance()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.types import ASGIApp, Receive, Scope, Send  # noqa: E402

from open_climate_service import config as api_config  # noqa: E402
from open_climate_service.main import app  # noqa: E402
from open_climate_service.openeo import earthkit_processes, xclim_processes  # noqa: E402
from open_climate_service.shared import compute, urls  # noqa: E402


def pytest_sessionstart(session: pytest.Session) -> None:
    # A repeated pytest.main() in one process can reuse this module without running its top
    # level again, so the test instance is set again here, after the previous session's restore.
    _use_test_instance()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _CALLERS_CONFIG is None:
        os.environ.pop("CLIMATE_SERVICE_CONFIG", None)
    else:
        os.environ["CLIMATE_SERVICE_CONFIG"] = _CALLERS_CONFIG


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
