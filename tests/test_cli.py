"""What the server entry point passes to uvicorn (CLIM-974).

Three of the four settings only take effect if uvicorn is *told* about them, and none of them
fails loudly when it is not: without `forwarded_allow_ips` a proxied deployment emits `http://`
links, and without `root_path` a prefixed one has to declare its prefix in
`CLIMATE_SERVICE_BASE_URL` and gets prefixed links on a direct port-forward too.
"""

from typing import Any

import pytest


@pytest.fixture
def uvicorn_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The keyword arguments `main()` hands to `uvicorn.run`, without starting a server."""
    import dotenv

    import open_climate_service.cli as cli

    captured: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    # `.env` on a developer's machine would otherwise decide what these assertions see. `main`
    # imports `load_dotenv` when it runs, so patching the attribute is enough.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for name in ("HOST", "PORT", "ROOT_PATH", "FORWARDED_ALLOW_IPS"):
        monkeypatch.delenv(name, raising=False)
    return captured


def test_the_defaults_bind_locally_with_no_prefix(
    monkeypatch: pytest.MonkeyPatch, uvicorn_kwargs: dict[str, Any]
) -> None:
    from open_climate_service.cli import DEFAULT_HOST, DEFAULT_PORT, main

    main()

    assert uvicorn_kwargs["host"] == DEFAULT_HOST
    assert uvicorn_kwargs["port"] == DEFAULT_PORT
    assert uvicorn_kwargs["root_path"] == ""
    # None, not "": uvicorn's own default is 127.0.0.1 and an empty string would trust nothing.
    assert uvicorn_kwargs["forwarded_allow_ips"] is None


def test_the_environment_reaches_uvicorn(monkeypatch: pytest.MonkeyPatch, uvicorn_kwargs: dict[str, Any]) -> None:
    """`ROOT_PATH` is the whole point: nothing else in the process can set ASGI `root_path`."""
    from open_climate_service.cli import main

    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("ROOT_PATH", "/ocs")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    main()

    assert uvicorn_kwargs["host"] == "127.0.0.1"
    assert uvicorn_kwargs["port"] == 8080
    assert uvicorn_kwargs["root_path"] == "/ocs"
    assert uvicorn_kwargs["forwarded_allow_ips"] == "10.0.0.0/8"
