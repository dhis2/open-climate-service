"""Named connections are reusable without copying credentials into plugin inputs."""

import builtins
import importlib.util
import json
import os
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from open_climate_service import config
from open_climate_service.exports.dhis2 import get_connection, get_connection_config


def _write_config(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream)


@pytest.fixture
def connection_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "climate-service.yaml"
    path.write_text(
        "dhis2_connections:\n"
        "  - id: national-hmis\n"
        "    url: https://hmis.example.org/dhis/\n"
        "    token_env: TEST_DHIS2_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(path))
    monkeypatch.delenv("TEST_DHIS2_TOKEN", raising=False)
    return path


def test_config_is_available_without_credentials_or_client(connection_file: Path, monkeypatch: pytest.MonkeyPatch):
    original = builtins.__import__

    def forbid_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dhis2_client":
            pytest.fail("Reading configuration must not import the client")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_client)
    connection = get_connection_config("national-hmis")
    assert connection.url == "https://hmis.example.org/dhis"
    assert connection.token_env == "TEST_DHIS2_TOKEN"


@pytest.mark.parametrize(
    "patch",
    [
        {"id": ""},
        {"url": "file:///tmp/dhis2"},
        {"url": "https://user:literal-secret@example.org"},
        {"url": "https://example.org?token=literal-secret"},
        {"url": "https://example.org/#literal-secret"},
        {"url": "https://example.org:99999"},
        {"url": "https://example.org\\other"},
        {"url": "https://"},
        {"token_env": "${TEST_DHIS2_TOKEN}"},
        {"token_env": ""},
        {"token": "literal-secret"},
        {"password": "literal-secret"},
        {"verify_ssl": False},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": True},
        {"connect_timeout": "30"},
        {"retries": -1},
        {"retries": True},
        {"retries": 11},
    ],
)
def test_invalid_config_is_rejected_before_caching(connection_file: Path, patch: dict[str, Any]):
    item = {"id": "national-hmis", "url": "https://example.org", "token_env": "TEST_DHIS2_TOKEN"} | patch
    _write_config(connection_file, {"dhis2_connections": [item]})
    with pytest.raises(ValueError) as error:
        config.get_config()
    assert "literal-secret" not in str(error.value)
    assert config._cache is None


@pytest.mark.parametrize("raw", [None, {}, "national-hmis", [None]])
def test_connections_must_be_a_list_of_mappings(connection_file: Path, raw: object):
    _write_config(connection_file, {"dhis2_connections": raw})
    with pytest.raises(ValueError, match="dhis2_connections"):
        config.get_config()


def test_duplicate_ids_are_rejected(connection_file: Path):
    raw = yaml.safe_load(connection_file.read_text())
    assert isinstance(raw, dict)
    raw["dhis2_connections"] *= 2
    _write_config(connection_file, raw)
    with pytest.raises(ValueError, match="duplicates"):
        config.get_config()


def test_interpolated_secret_is_never_parsed_or_cached(connection_file: Path, monkeypatch: pytest.MonkeyPatch):
    secret = "secret: [invalid yaml"
    monkeypatch.setenv("TEST_DHIS2_TOKEN", secret)
    connection_file.write_text(
        connection_file.read_text().replace("token_env: TEST_DHIS2_TOKEN", "token_env: ${TEST_DHIS2_TOKEN}")
    )
    with pytest.raises(ValueError, match="literal") as error:
        config.get_config()
    assert secret not in str(error.value)
    assert config._cache is None


def test_other_config_interpolation_still_works(connection_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSTANCE_NAME", "Test instance")
    connection_file.write_text(connection_file.read_text() + "name: ${INSTANCE_NAME}\n")
    assert config.get_name() == "Test instance"


def test_legacy_yaml_fragment_interpolation_without_connections(connection_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEST", "-13.5")
    connection_file.write_text("extent:\n  bbox: [${WEST}, 6.9, -10.1, 10.0]\n")
    assert config.get_config()["extent"]["bbox"] == [-13.5, 6.9, -10.1, 10.0]


def test_unknown_connection(connection_file: Path):
    with pytest.raises(ValueError, match="Unknown DHIS2 connection"):
        get_connection("unknown")


@pytest.mark.parametrize("token", [None, "", " ", "ApiToken secret", "secret\n", "secret\x00", "secret\x7f", "é"])
def test_missing_or_invalid_token(connection_file: Path, monkeypatch: pytest.MonkeyPatch, token: str | None):
    if token is not None:
        if "\x00" in token:
            # Real process environments cannot contain NUL; exercise header validation directly.
            monkeypatch.setattr(
                "open_climate_service.exports.dhis2.os.environ", dict(os.environ) | {"TEST_DHIS2_TOKEN": token}
            )
        else:
            monkeypatch.setenv("TEST_DHIS2_TOKEN", token)
    with pytest.raises(ValueError, match="token") as error:
        get_connection("national-hmis")
    assert "secret" not in str(error.value)


def test_missing_optional_dependency_is_actionable(connection_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_DHIS2_TOKEN", "test-token")
    original = builtins.__import__

    def missing_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dhis2_client":
            raise ModuleNotFoundError("missing", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_client)
    with pytest.raises(RuntimeError, match="optional dhis2-client"):
        get_connection("national-hmis")


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    pytest.importorskip("dhis2_client", reason="Install the documented optional DHIS2 client to run transport tests")
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    original = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        return original(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "Client", client)
    return captured


def test_real_client_auth_rotation_and_config_redaction(
    connection_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    requests: list[httpx.Request],
    caplog: pytest.LogCaptureFixture,
):
    for token in ["first-test-secret", "rotated-test-secret"]:
        monkeypatch.setenv("TEST_DHIS2_TOKEN", token)
        with closing(get_connection("national-hmis")) as client:
            assert len(requests) == (0 if token.startswith("first") else 1)
            client.get("/api/me")
            assert requests[-1].headers["Authorization"] == f"ApiToken {token}"
            assert str(requests[-1].url) == "https://hmis.example.org/dhis/api/me"
            assert requests[-1].extensions["timeout"]["connect"] == 10.0
            assert requests[-1].extensions["timeout"]["read"] == 30.0
        assert token not in json.dumps(config.get_config())
        assert token not in repr(get_connection_config("national-hmis"))
        assert token not in caplog.text


def test_external_provider_uses_only_public_accessor(
    connection_file: Path, monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request], tmp_path: Path
):
    monkeypatch.setenv("TEST_DHIS2_TOKEN", "provider-test-token")
    module_file = tmp_path / "external_provider.py"
    module_file.write_text(
        "from contextlib import closing\n"
        "from open_climate_service.exports.dhis2 import get_connection\n"
        "def fetch(connection_id, level):\n"
        "    with closing(get_connection(connection_id)) as client:\n"
        "        return client.get_org_units_geojson(level=level)\n"
    )
    spec = importlib.util.spec_from_file_location("external_provider", module_file)
    assert spec is not None and spec.loader is not None
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    assert provider.fetch("national-hmis", 2) == {"type": "FeatureCollection", "features": []}
    assert requests[0].url.path == "/dhis/api/organisationUnits.geojson"
    assert requests[0].url.params["level"] == "2"
