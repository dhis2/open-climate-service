"""Validate named DHIS2 connections without resolving credentials or opening clients."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Dhis2ConnectionConfig:
    """Non-secret configuration for one DHIS2 instance."""

    id: str
    url: str
    token_env: str
    timeout: float = 30.0
    connect_timeout: float = 10.0
    retries: int = 3


def parse_connections(raw: object) -> dict[str, Dhis2ConnectionConfig]:
    """Validate a list of connection definitions, without including input values in errors."""
    if not isinstance(raw, list):
        raise ValueError("dhis2_connections must be a list")
    connections: dict[str, Dhis2ConnectionConfig] = {}
    allowed = {"id", "url", "token_env", "timeout", "connect_timeout", "retries"}
    for index, item in enumerate(raw):
        prefix = f"dhis2_connections[{index}]"
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError(f"{prefix} must be a mapping containing only supported connection fields")
        identifier = _string(item.get("id"), f"{prefix}.id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError(f"{prefix}.id must contain only letters, digits, underscores, or hyphens")
        if identifier in connections:
            raise ValueError(f"{prefix}.id duplicates an earlier connection")
        url = _string(item.get("url"), f"{prefix}.url").rstrip("/")
        if not _valid_url(url):
            raise ValueError(f"{prefix}.url must be an HTTP(S) instance URL without credentials, query, or fragment")
        token_env = _string(item.get("token_env"), f"{prefix}.token_env")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
            raise ValueError(f"{prefix}.token_env must be a literal environment variable name")
        retries = item.get("retries", 3)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError(f"{prefix}.retries must be an integer between 0 and 10")
        connections[identifier] = Dhis2ConnectionConfig(
            id=identifier,
            url=url,
            token_env=token_env,
            timeout=_timeout(item.get("timeout", 30.0), f"{prefix}.timeout"),
            connect_timeout=_timeout(item.get("connect_timeout", 10.0), f"{prefix}.connect_timeout"),
            retries=retries,
        )
    return connections


def _string(raw: object, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or "${" in raw:
        raise ValueError(f"{field} must be a non-empty literal string (no environment interpolation)")
    return raw.strip()


def _timeout(raw: object, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (float, int)) or raw <= 0:
        raise ValueError(f"{field} must be a finite positive number of seconds")
    try:
        value = float(raw)
    except OverflowError:
        raise ValueError(f"{field} must be a finite positive number of seconds") from None
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite positive number of seconds")
    return value


def _valid_url(url: str) -> bool:
    if any(character.isspace() for character in url) or "\\" in url:
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
        return bool(
            parts.scheme in {"http", "https"}
            and parts.hostname
            and not parts.username
            and not parts.password
            and "@" not in parts.netloc
            and "?" not in url
            and "#" not in url
            and (port is None or port > 0)
        )
    except ValueError:
        return False
