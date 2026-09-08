"""Command-line entry point for running the Open Climate Service with uvicorn."""

import os

import uvicorn

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000


def main() -> None:
    """Start the Open Climate Service server.

    Host, port, deployment prefix and trusted proxy addresses come from HOST, PORT, ROOT_PATH
    and FORWARDED_ALLOW_IPS, defaulting to 0.0.0.0:9000, no prefix, and uvicorn's own
    127.0.0.1.

    `.env` is loaded here, before the four are read, because uvicorn builds its config from the
    process environment as `run()` is called — earlier than the app import that loads `.env` for
    everything else. Without this, `FORWARDED_ALLOW_IPS` in `.env` is ignored and only compose's
    `env_file` works, with no error to go on. `load_dotenv` does not overwrite variables already
    set, so compose and the shell still win over the file.

    `ROOT_PATH` is how a deployment prefix should be declared: it sets ASGI `root_path`, which
    `shared.urls` prefers over its fallback of reading the path of `CLIMATE_SERVICE_BASE_URL` as
    a mount. That fallback cannot distinguish a proxied request from a direct one, so it also
    prefixes in-page links on a port-forward straight to the port.
    """
    from dotenv import load_dotenv

    load_dotenv()

    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS")
    root_path = os.environ.get("ROOT_PATH", "")

    uvicorn.run(
        "open_climate_service.main:app",
        host=host,
        port=port,
        forwarded_allow_ips=forwarded_allow_ips,
        root_path=root_path,
    )
