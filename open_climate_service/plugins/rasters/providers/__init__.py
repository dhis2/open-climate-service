"""Helpers shared by the built-in raster plugins that read from the same upstream provider.

Internal to the built-in raster plugins: external plugins neither import nor follow these. A credential
is read from the environment, which OCS fills from ``.env``, when a plugin opens a source, so
listing data sources needs no keys.
"""

import os


def require_env(name: str, *, hint: str) -> str:
    """Return a non-empty environment variable, or raise naming it and how to get one."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise PermissionError(f"{name} is not set. {hint} Then set {name} in .env and restart OCS.")
    return value
