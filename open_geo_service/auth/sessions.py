"""Web interface sessions: a random id in a cookie, mapped in memory to the token that signed in.

The cookie never holds the token. Sessions live in this process, so a restart signs everyone
out; Open Geo Service runs as one process. A session ends at sign-out, after `LIFETIME`, or when
its token is no longer configured with the same secret.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie

from open_geo_service.auth.tokens import NamedToken, Principal, principal_for

COOKIE = "geoservice_session"
LIFETIME_SECONDS = 8 * 60 * 60


@dataclass(frozen=True)
class _Session:
    token: NamedToken
    expires: float


_sessions: dict[str, _Session] = {}
_lock = threading.Lock()


def start(token: NamedToken) -> str:
    """Open a session for a token that signed in, and return the id the cookie carries."""
    session_id = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _lock:
        # Expired sessions are dropped here, so the table stays bounded by recent sign-ins.
        for stale in [key for key, session in _sessions.items() if session.expires <= now]:
            del _sessions[stale]
        _sessions[session_id] = _Session(token, now + LIFETIME_SECONDS)
    return session_id


def resolve(session_id: str, tokens: tuple[NamedToken, ...]) -> Principal | None:
    """The principal a session id stands for; None when it is unknown, expired or its token changed."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        if session.expires <= time.monotonic() or session.token not in tokens:
            del _sessions[session_id]
            return None
    return principal_for(session.token)


def end(session_id: str) -> None:
    """End a session at sign-out; an unknown id is ignored."""
    with _lock:
        _sessions.pop(session_id, None)


def cookie(session_id: str, path: str) -> str:
    """The `Set-Cookie` value for a session, scoped to *path*."""
    return _cookie(session_id, path, LIFETIME_SECONDS)


def expired_cookie(path: str) -> str:
    """The `Set-Cookie` value that removes the session cookie at *path*."""
    return _cookie("", path, 0)


def _cookie(value: str, path: str, max_age: int) -> str:
    # Secure: browsers keep it over https and on localhost, so sign-in needs one of the two.
    jar = SimpleCookie()
    jar[COOKIE] = value
    morsel = jar[COOKIE]
    morsel["path"] = path
    morsel["max-age"] = max_age
    morsel["httponly"] = True
    morsel["secure"] = True
    morsel["samesite"] = "Lax"
    return morsel.OutputString()
