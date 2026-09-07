"""Command-line entry point for running the Open Climate Service with uvicorn."""

import os

import uvicorn

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000


def main() -> None:
    """Start the Open Climate Service server.

    Host, port and trusted proxy addresses are read from HOST, PORT and
    FORWARDED_ALLOW_IPS, defaulting to 0.0.0.0:9000 and uvicorn's own 127.0.0.1.

    `FORWARDED_ALLOW_IPS` is passed explicitly rather than left to uvicorn. Uvicorn reads it
    from the process environment while building its config, which happens before the app
    imports and calls `load_dotenv`, so a value in `.env` was silently ignored here and worked
    only through compose's `env_file`. An operator who set it and still got `http://` links had
    no error to go on.
    """
    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))

    from dotenv import load_dotenv

    load_dotenv()
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS")

    uvicorn.run(
        "open_climate_service.main:app",
        host=host,
        port=port,
        forwarded_allow_ips=forwarded_allow_ips,
    )
