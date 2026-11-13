"""DestinE Earth Data Hub (EDH) credentials."""

import base64

from open_climate_service.plugins.rasters.providers import require_env


def api_key() -> str:
    """The EDH API key."""
    hint = "Generate a standard API key at https://earthdatahub.destine.eu under Account settings, My API keys."
    return require_env("EDH_API_KEY", hint=hint)


def auth_headers() -> dict[str, str]:
    """Basic Auth headers for EDH, which takes the login ``edh`` and the API key as password."""
    encoded = base64.b64encode(f"edh:{api_key()}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}
