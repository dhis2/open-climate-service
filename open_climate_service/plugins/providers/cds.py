"""Copernicus Climate Data Store (CDS) credentials."""

from open_climate_service.plugins.providers import require_env


def api_key() -> dict[str, str]:
    """The CDS API key, which both the retrieve API and the ARCO stores accept."""
    hint = "Copy the API key from your profile at https://cds.climate.copernicus.eu."
    return {"CDS_API_KEY": require_env("CDS_API_KEY", hint=hint)}


def auth_headers() -> dict[str, str]:
    """Bearer headers for the CDS ARCO stores."""
    return {"Authorization": f"Bearer {api_key()['CDS_API_KEY']}"}
