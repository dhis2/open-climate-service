"""Public named DHIS2 clients shared by exporters and external feature providers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from open_climate_service import config
from open_climate_service.exports.dhis2_config import Dhis2ConnectionConfig, parse_connections
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

if TYPE_CHECKING:
    from dhis2_client import DHIS2Client

__all__ = ["Dhis2ConnectionConfig", "Dhis2ExportPlugin", "get_connection", "get_connection_config"]


def get_connection_config(connection_id: str) -> Dhis2ConnectionConfig:
    """Resolve non-secret configuration without importing the optional client."""
    connections = parse_connections(config.get_config().get("dhis2_connections", []))
    try:
        return connections[connection_id]
    except KeyError:
        raise ValueError("Unknown DHIS2 connection; configure its ID in dhis2_connections") from None


def get_connection(connection_id: str) -> DHIS2Client:
    """Create a configured client, resolving its token at use time.

    The caller owns the returned client and must call ``close()`` in a ``finally``
    block (or use ``contextlib.closing``). Each call creates a fresh client so token
    rotation takes effect without resetting configuration. Construction sends no
    network requests. Access control for operations belongs to their callers.
    """
    connection = get_connection_config(connection_id)
    token = os.environ.get(connection.token_env)
    if not token or not token.strip():
        raise ValueError("The DHIS2 connection's token environment variable is unset or empty")
    if any(not 33 <= ord(character) <= 126 for character in token):
        raise ValueError("The DHIS2 token must be a raw ASCII token without whitespace or an authentication prefix")
    try:
        from dhis2_client import DHIS2Client
    except ModuleNotFoundError as exc:
        if exc.name != "dhis2_client":
            raise
        raise RuntimeError(
            "Named DHIS2 connections require the optional dhis2-client package. "
            "See docs/importing_to_dhis2.md for the tested installation command."
        ) from None

    return DHIS2Client(
        base_url=connection.url,
        token=token,
        timeout=connection.timeout,
        connect_timeout=connection.connect_timeout,
        retries=connection.retries,
        verify_ssl=True,
    )
