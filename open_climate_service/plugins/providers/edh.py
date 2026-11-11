"""DestinE Earth Data Hub (EDH) credentials."""

import base64

from open_climate_service.plugins.providers import require_env


def auth_headers() -> dict[str, str]:
    """Basic Auth headers for EDH, which takes the login ``edh`` and the API key as password."""
    key = require_env(
        "EDH_API_KEY",
        hint="Generate a standard API key at https://earthdatahub.destine.eu under Account settings, My API keys.",
    )
    encoded = base64.b64encode(f"edh:{key}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}
