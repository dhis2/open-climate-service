"""Web interface sign-in: a token posted once becomes a session cookie (see `auth/sessions.py`)."""

from __future__ import annotations

import urllib.parse
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from open_geo_service.auth import sessions
from open_geo_service.auth.middleware import log_rejected_credential, page_origins, session_cookie_path
from open_geo_service.auth.tokens import authenticate, configured_tokens
from open_geo_service.shared.urls import mount_prefix

router = APIRouter(include_in_schema=False)


def _require_authentication() -> None:
    # With authentication off there is nothing to sign in to.
    if not configured_tokens():
        raise HTTPException(status_code=404)


_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _secure_page(request: Request) -> bool:
    """Whether the browser reached this page over HTTPS or on this machine, as far as the server can tell.

    Elsewhere the form would send the token in clear, and the browser would not keep the
    `Secure` cookie anyway. The origin is the one the request addressed, or
    `GEOSERVICE_BASE_URL`'s when it names the same host, as behind a TLS proxy. A proxy that
    rewrites `Host` to a loopback address passes this check, so the page also shows the form
    only in a secure context, which only the browser can see.
    """
    host = request.headers.get("host", "")
    for origin in page_origins(request.scope):
        split = urllib.parse.urlsplit(origin)
        name = split.hostname or ""
        if split.netloc == host and (split.scheme == "https" or name in _LOOPBACK or name.endswith(".localhost")):
            return True
    return False


def _local_path(candidate: str | None, mount: str) -> str:
    """A path on this instance to return to after signing in; the landing page for anything else."""
    split = urllib.parse.urlsplit(candidate or "")
    path = split.path
    # Only a path under the mount, never another origin or a scheme-relative `//host`.
    if split.scheme or split.netloc or not path.startswith(f"{mount}/") or path.startswith("//"):
        return f"{mount}/"
    if path in (f"{mount}/sign-in", f"{mount}/sign-out"):
        return f"{mount}/"
    return path + (f"?{split.query}" if split.query else "")


@router.get("/sign-in", response_class=HTMLResponse)
def sign_in_page(request: Request) -> HTMLResponse:
    """The form that takes a token; it returns to the page the visitor came from."""
    _require_authentication()
    from open_geo_service.system.templates import SIGN_IN_NEEDS_HTTPS, render_sign_in_page

    mount = mount_prefix(request)
    referer = urllib.parse.urlsplit(request.headers.get("referer", ""))
    came_from = referer.path + (f"?{referer.query}" if referer.query else "")
    same_host = referer.netloc == request.headers.get("host")
    next_path = _local_path(came_from if same_host else None, mount)
    if not _secure_page(request):
        return HTMLResponse(render_sign_in_page(mount, next_path=next_path, error=SIGN_IN_NEEDS_HTTPS, form=False))
    return HTMLResponse(render_sign_in_page(mount, next_path=next_path))


@router.post("/sign-in", response_class=HTMLResponse)
def sign_in(request: Request, token: Annotated[str, Form()], next: Annotated[str, Form()] = "") -> Response:
    """Exchange a token for a session cookie, then return to *next*."""
    _require_authentication()
    from open_geo_service.system.templates import SIGN_IN_NEEDS_HTTPS, render_sign_in_page

    mount = mount_prefix(request)
    if not _secure_page(request):
        page = render_sign_in_page(mount, next_path=_local_path(next, mount), error=SIGN_IN_NEEDS_HTTPS, form=False)
        return HTMLResponse(page, status_code=403)
    principal = authenticate(f"Bearer {token.strip()}", configured_tokens())
    if principal is None or principal.token is None:
        log_rejected_credential()
        page = render_sign_in_page(mount, next_path=_local_path(next, mount), error="That token is not valid here.")
        return HTMLResponse(page, status_code=401)
    response = RedirectResponse(_local_path(next, mount), status_code=303)
    session_id = sessions.start(principal.token)
    response.headers.append("set-cookie", sessions.cookie(session_id, session_cookie_path(request)))
    return response


@router.post("/sign-out")
def sign_out(request: Request) -> Response:
    """End the session and clear its cookie."""
    _require_authentication()
    session_id = request.cookies.get(sessions.COOKIE)
    if session_id:
        sessions.end(session_id)
    response = RedirectResponse(f"{mount_prefix(request)}/", status_code=303)
    response.headers.append("set-cookie", sessions.expired_cookie(session_cookie_path(request)))
    return response
