import atexit
import ipaddress
import os
import shutil
import socket
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

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
from open_climate_service import plugin_discovery  # noqa: E402
from open_climate_service.features import tiles  # noqa: E402
from open_climate_service.main import app  # noqa: E402
from open_climate_service.openeo import earthkit_processes, xclim_processes  # noqa: E402
from open_climate_service.shared import compute, urls  # noqa: E402

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def _local_only(address: Any) -> None:
    """Refuse a connection that would leave the machine.

    A test that reaches a data source fails when the network does, and every developer and
    CI run sends it again, so parallel workers multiply the load on servers that may block
    us. Stub the source instead, as the sync tests do with `_query_available_periods`.
    """
    host = address[0] if isinstance(address, tuple) else None  # a Unix socket's address is a path
    if host is None or host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass  # a host name other than localhost
    raise OSError(f"tests must not connect outside localhost; stub the call that reached {address}")


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    # Before the resolver, so an unstubbed host name fails at once instead of after a DNS timeout.
    name = host.decode() if isinstance(host, bytes) else host
    if isinstance(name, str) and name != "localhost":
        try:
            ipaddress.ip_address(name)
        except ValueError:
            raise OSError(f"tests must not resolve host names; stub the call that looked up {name!r}") from None
    return _getaddrinfo(host, *args, **kwargs)


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _local_only(address)
    _connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _local_only(address)
    return _connect_ex(self, address)


def pytest_sessionstart(session: pytest.Session) -> None:
    # A repeated pytest.main() in one process can reuse this module without running its top
    # level again, so the test instance is set again here, after the previous session's restore.
    _use_test_instance()
    # For the whole session rather than per test, so threads a test leaves running are covered too.
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _getaddrinfo
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
    # A test that installs a plugin rescans; the plugins found before it are put back after.
    installed_plugins = plugin_discovery._installed_plugins()
    yield
    api_config._cache = None
    xclim_processes._cache = None
    earthkit_processes._cache = None
    urls._forget_base_url_warnings()
    plugin_discovery._installed = installed_plugins


@pytest.fixture(autouse=True)
def _no_vector_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh feature collections without building their PMTiles archive.

    A build takes about a second, and most tests refresh a collection for its GeoParquet only.
    The archive is display output whose failure never fails a refresh, so skipping it changes
    nothing else; `test_feature_tiles.py` overrides this fixture to build it.
    """
    monkeypatch.setattr(tiles, "build_archive", lambda *_args, **_kwargs: None)


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


class WebhookReceiver:
    """The configured notification webhook: what it was sent, and how it answers."""

    def __init__(self) -> None:
        self.enabled = True
        self.status = 200
        self.refused: set[str] = set()
        """Automation ids whose posts are answered 503 whatever ``status`` is."""
        self.posts: list[tuple[int, str, dict[str, Any]]] = []
        """Every post as (the status it was answered with, its Idempotency-Key, its body)."""


@pytest.fixture
def webhook(monkeypatch: pytest.MonkeyPatch) -> WebhookReceiver:
    """Notifications configured, posted in the calling thread to a receiver that records them."""
    import httpx

    from open_climate_service.notifications import webhook as notifications

    receiver = WebhookReceiver()

    def post(url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
        status = 503 if json.get("id") in receiver.refused else receiver.status
        receiver.posts.append((status, headers["Idempotency-Key"], json))
        return httpx.Response(status)

    monkeypatch.setattr(notifications, "webhook_url", lambda: "http://hooks.test/ocs" if receiver.enabled else None)
    monkeypatch.setattr(notifications.httpx, "post", post)
    monkeypatch.setattr(notifications, "_kick", notifications.send_due)
    return receiver
