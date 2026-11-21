"""The ASGI middleware that decides every request before the application reads its body."""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
from contextvars import ContextVar
from http.cookies import SimpleCookie

from fastapi import Request
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from open_geo_service import config as api_config
from open_geo_service import read_only
from open_geo_service.auth import sessions
from open_geo_service.auth.route_table import listed_tier
from open_geo_service.auth.tokens import ANONYMOUS, OPEN, Principal, Tier, authenticate, configured_tokens
from open_geo_service.shared.urls import BASE_URL_ENV, mount_prefix, strip_mount

logger = logging.getLogger(__name__)

_FAILURE_LOG_INTERVAL = 60.0
_failures_since_log = 0
_last_failure_log = float("-inf")


_SAFE_METHODS = frozenset({"GET", "HEAD"})
# Posted by a browser before it holds a session, so checked for origin like a cookie's writes.
_SIGN_IN_FORMS = frozenset({"/sign-in", "/sign-out"})

_viewer: ContextVar[Principal | None] = ContextVar("viewer", default=None)


def principal_of(request: Request) -> Principal:
    """The principal the access middleware resolved for this request."""
    principal = getattr(request.state, "principal", ANONYMOUS)
    assert isinstance(principal, Principal)
    return principal


def current_viewer() -> Principal:
    """The principal of the request being handled, for page parts every page renders.

    Outside a request, as when a test renders a page directly, it is `ANONYMOUS` with
    authentication on, so such a page shows no operator forms, and `OPEN` with it off.
    """
    viewer = _viewer.get()
    if viewer is not None:
        return viewer
    return ANONYMOUS if configured_tokens() else OPEN


def session_cookie_path(request: Request) -> str:
    """The cookie path: the mount prefix, so another app on the same host never receives it."""
    return mount_prefix(request) or "/"


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
            headers = Headers(scope=scope)
            authorization = headers.get("authorization")
            session_id = _session_id(headers)
            if authorization is None and session_id is not None:
                session = sessions.resolve(session_id, tokens)
                if session is None:
                    # An expired or unknown cookie is no credential, not a refusal: refusing it
                    # would lock the browser out of every page, the sign-in page included.
                    send = _clearing_cookie(send, session_cookie_path(Request(scope)))
                principal = session or ANONYMOUS
                by_cookie = session is not None
            else:
                resolved = authenticate(authorization, tokens)
                if resolved is None:
                    log_rejected_credential()
                    message = "The credential is not valid for this instance."
                    await _refuse(401, "authentication", "TokenInvalid", message)(scope, receive, send)
                    return
                principal = resolved
                by_cookie = False
            browser_form = by_cookie or (path in _SIGN_IN_FORMS and authorization is None)
            if method not in _SAFE_METHODS and browser_form and not _same_origin(scope, headers):
                message = f"{method} {path} from a page on another site is refused."
                if path in _SIGN_IN_FORMS:
                    # A cookie's own writes are rarely misdirected; a refused sign-in usually means
                    # a proxy hides the address the browser used.
                    message += " If this instance runs behind a proxy, set GEOSERVICE_BASE_URL to its public address."
                await _refuse(403, "authentication", "PermissionsInsufficient", message)(scope, receive, send)
                return
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
        reset = _viewer.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            _viewer.reset(reset)


def _session_id(headers: Headers) -> str | None:
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get("cookie", ""))
    except Exception:  # noqa: BLE001 - a malformed Cookie header carries no session
        return None
    morsel = cookie.get(sessions.COOKIE)
    return morsel.value if morsel is not None and morsel.value else None


def _same_origin(scope: Scope, headers: Headers) -> bool:
    """Whether a browser request comes from a page of this instance.

    `Origin` must be one of `page_origins`, scheme, host and port alike. Without it,
    `Sec-Fetch-Site` must say same-origin. A request with neither is refused, since only a
    browser sends the cookie.
    """
    origin = headers.get("origin")
    if origin is None:
        return headers.get("sec-fetch-site") == "same-origin"
    return origin in page_origins(scope)


def page_origins(scope: Scope) -> set[str]:
    """The origins this instance's pages are served from.

    The one the request addressed, and `GEOSERVICE_BASE_URL`'s, which behind a TLS proxy is the
    one the browser sees.
    """
    origins = {f"{scope.get('scheme', 'http')}://{Headers(scope=scope).get('host', '')}"}
    configured = urllib.parse.urlsplit(os.getenv(BASE_URL_ENV, "").strip())
    if configured.scheme and configured.netloc:
        origins.add(f"{configured.scheme}://{configured.netloc}")
    return origins


def _clearing_cookie(send: Send, path: str) -> Send:
    """`send`, adding a header that clears the session cookie to the response."""

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = MutableHeaders(scope=message)
            # A sign-in sent with a stale cookie sets a new one, which must not be cleared too.
            if not any(value.startswith(f"{sessions.COOKIE}=") for value in headers.getlist("set-cookie")):
                headers.append("set-cookie", sessions.expired_cookie(path))
        await send(message)

    return wrapped


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


def log_rejected_credential() -> None:
    """Log rejected credentials at most once a minute, so a client retrying a bad token cannot flood the log."""
    global _failures_since_log, _last_failure_log
    _failures_since_log += 1
    now = time.monotonic()
    if now - _last_failure_log >= _FAILURE_LOG_INTERVAL:
        logger.warning("Rejected %d request(s) with an invalid credential", _failures_since_log)
        _failures_since_log = 0
        _last_failure_log = now
