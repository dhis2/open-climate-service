"""The ASGI middleware that decides every request before the application reads its body."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Match
from starlette.types import ASGIApp, Receive, Scope, Send

from open_geo_service import config as api_config
from open_geo_service import read_only
from open_geo_service.auth.route_table import listed_tier
from open_geo_service.auth.tokens import ANONYMOUS, OPEN, Principal, Tier, authenticate, configured_tokens
from open_geo_service.shared.urls import strip_mount

logger = logging.getLogger(__name__)

_FAILURE_LOG_INTERVAL = 60.0
_failures_since_log = 0
_last_failure_log = float("-inf")


def principal_of(request: Request) -> Principal:
    """The principal the access middleware resolved for this request."""
    principal = getattr(request.state, "principal", ANONYMOUS)
    assert isinstance(principal, Principal)
    return principal


class AccessMiddleware:
    """Resolve the caller, apply read-only mode, and compare the caller's tier with the route's.

    Registered inside CORS, so a refusal still carries the headers a browser needs to read it.
    `OPTIONS` preflights pass, so a browser learns about the refusal from the real request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        method: str = scope["method"]
        # Matched as the routes see the path, so a deployment prefix cannot hide a gated route.
        path = strip_mount(scope["path"], str(scope.get("root_path", "")))
        required = listed_tier(method, path)
        if required is None and not _served(scope):
            # Answered 404 by the application, rather than refused as if it were a gated route.
            await self.app(scope, receive, send)
            return
        if api_config.is_read_only() and read_only.is_blocked(method, path):
            message = read_only.MESSAGE.format(method=method, path=path)
            await _refuse(403, "read-only", "PermissionsInsufficient", message)(scope, receive, send)
            return
        principal = OPEN
        tokens = configured_tokens()
        if tokens:
            resolved = authenticate(Headers(scope=scope).get("authorization"), tokens)
            if resolved is None:
                _log_failure()
                message = "The credential is not valid for this instance."
                await _refuse(401, "authentication", "TokenInvalid", message)(scope, receive, send)
                return
            principal = resolved
            needed = Tier.OPERATOR if required is None else required
            if principal.tier < needed:
                tier = needed.name.lower()
                if principal is ANONYMOUS:
                    message = f"{method} {path} needs a token with the {tier} tier."
                    await _refuse(401, "authentication", "AuthenticationRequired", message)(scope, receive, send)
                else:
                    message = f"{method} {path} needs the {tier} tier, which {principal.id} does not have."
                    await _refuse(403, "authentication", "PermissionsInsufficient", message)(scope, receive, send)
                return
        scope.setdefault("state", {})["principal"] = principal
        await self.app(scope, receive, send)


def _served(scope: Scope) -> bool:
    """Whether a route of the application matches the request, by path whatever the method."""
    application: Starlette = scope["app"]
    return any(route.matches(scope)[0] is not Match.NONE for route in application.routes)


def _refuse(status: int, error_id: str, code: str, message: str) -> JSONResponse:
    # `detail` matches FastAPI's HTTPException shape; `code` and `message` are the openEO error
    # fields, so openEO clients show the reason rather than a bare status.
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    content = {"id": error_id, "code": code, "message": message, "detail": message}
    return JSONResponse(status_code=status, content=content, headers=headers)


def _log_failure() -> None:
    """Log rejected credentials at most once a minute, so a client retrying a bad token cannot flood the log."""
    global _failures_since_log, _last_failure_log
    _failures_since_log += 1
    now = time.monotonic()
    if now - _last_failure_log >= _FAILURE_LOG_INTERVAL:
        logger.warning("Rejected %d request(s) with an invalid credential", _failures_since_log)
        _failures_since_log = 0
        _last_failure_log = now
