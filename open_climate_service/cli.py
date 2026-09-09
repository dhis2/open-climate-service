"""Command-line entry point for running the Open Climate Service with uvicorn."""

import os

import uvicorn

import open_climate_service.startup  # noqa: F401  # pyright: ignore[reportUnusedImport]

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000


def main() -> None:
    """Start the Open Climate Service server.

    Host, port and trusted proxy addresses come from HOST, PORT and FORWARDED_ALLOW_IPS,
    defaulting to 0.0.0.0:9000 and uvicorn's own 127.0.0.1. ROOT_PATH is read by the app
    itself (`create_app`), so it applies under every launcher, not only this one.

    `startup` is imported above, before the environment is read, because uvicorn builds its
    config from the process environment as `run()` is called, earlier than the app import that
    loads `.env` for everything else. Without it, `FORWARDED_ALLOW_IPS` in `.env` is silently
    ignored. `load_dotenv` does not overwrite variables already set, so compose and the shell
    still win over the file.
    """
    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS")

    uvicorn.run(
        "open_climate_service.main:app",
        host=host,
        port=port,
        forwarded_allow_ips=forwarded_allow_ips,
    )
