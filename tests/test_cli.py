"""What the server entry point passes to uvicorn (CLIM-974).

`forwarded_allow_ips` only takes effect if uvicorn is *told* about it, and it does not fail
loudly when it is not: a proxied deployment simply emits `http://` links. `ROOT_PATH` is
read by the app itself (see `test_absolute_urls`), so the entry point does not forward it.
"""

from typing import Any

import pytest


@pytest.fixture
def uvicorn_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The keyword arguments `main()` hands to `uvicorn.run`, without starting a server."""
    import open_climate_service.cli as cli

    captured: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    # `.env` was loaded when `cli` imported `startup`; clearing after that keeps a developer's
    # machine from deciding what these assertions see.
    for name in ("HOST", "PORT", "FORWARDED_ALLOW_IPS"):
        monkeypatch.delenv(name, raising=False)
    return captured


def test_the_defaults_bind_locally(uvicorn_kwargs: dict[str, Any]) -> None:
    from open_climate_service.cli import DEFAULT_HOST, DEFAULT_PORT, main

    main()

    assert uvicorn_kwargs["app"] == "open_climate_service.main:app"
    assert uvicorn_kwargs["host"] == DEFAULT_HOST
    assert uvicorn_kwargs["port"] == DEFAULT_PORT
    # None, not "": uvicorn's own default is 127.0.0.1 and an empty string would trust nothing.
    assert uvicorn_kwargs["forwarded_allow_ips"] is None


def test_the_environment_reaches_uvicorn(monkeypatch: pytest.MonkeyPatch, uvicorn_kwargs: dict[str, Any]) -> None:
    from open_climate_service.cli import main

    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    main()

    assert uvicorn_kwargs["host"] == "127.0.0.1"
    assert uvicorn_kwargs["port"] == 8080
    assert uvicorn_kwargs["forwarded_allow_ips"] == "10.0.0.0/8"


def test_the_entry_point_does_not_own_the_root_path(
    monkeypatch: pytest.MonkeyPatch, uvicorn_kwargs: dict[str, Any]
) -> None:
    """One owner: the app reads `ROOT_PATH`, so it applies under `make run` and bare uvicorn
    too, and the entry point passing it as well would only be a second copy of the value."""
    from open_climate_service.cli import main

    monkeypatch.setenv("ROOT_PATH", "/ocs")
    main()

    assert "root_path" not in uvicorn_kwargs
