"""The `/mcp` route: an MCP streamable HTTP endpoint mounted on the OCS app (CLIM-851)."""

from contextvars import ContextVar
from functools import cache

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from open_climate_service import config as api_config

# The HTTP request a tool call arrived on. Tools call the OCS API in-process, and need the
# request's app, origin and mount prefix to do it the way an outside client would, so links in
# their results point at the instance's public address rather than an internal one.
current_scope: ContextVar[Scope] = ContextVar("mcp_current_scope")


@cache
def _server() -> MCPServer:
    from open_climate_service.mcp_server.tools import build_server

    return build_server()


class McpEndpoint:
    """ASGI app for `/mcp`, answering 404 unless the instance enables it.

    Stateless, so each request runs its own session manager for as long as it takes: nothing is
    kept between requests, any worker can serve any of them, and the endpoint needs nothing
    from the app's lifespan.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not api_config.is_mcp_enabled():
            response = JSONResponse(
                {"detail": "MCP is not enabled on this instance. Set mcp: true in climate-service.yaml."},
                status_code=404,
            )
            await response(scope, receive, send)
            return
        server = _server()
        # Called for the session manager it creates, not for the Starlette app it returns: the
        # route is OCS's own. No host or origin allowlist. The SDK turns one on for a server
        # bound to localhost, which would refuse an instance served under its public name. /mcp
        # is read-only and offers nothing the unauthenticated API does not, which allows any
        # origin.
        server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        manager = server.session_manager
        token = current_scope.set(scope)
        try:
            async with manager.run():
                await manager.handle_request(scope, receive, send)
        finally:
            current_scope.reset(token)
