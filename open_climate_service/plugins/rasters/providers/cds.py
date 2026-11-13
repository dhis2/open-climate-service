"""Copernicus Climate Data Store (CDS) credentials."""

from open_climate_service.plugins.rasters.providers import require_env


def api_key() -> str:
    """The CDS API key, which both the retrieve API and the ARCO stores accept."""
    hint = "Copy the API key from your profile at https://cds.climate.copernicus.eu."
    return require_env("CDS_API_KEY", hint=hint)


def auth_headers() -> dict[str, str]:
    """Bearer headers for the CDS ARCO stores."""
    return {"Authorization": f"Bearer {api_key()}"}
