"""Routes for the STAC catalogue."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from open_climate_service.stac import documents
from open_climate_service.stac import services as stac_services
from open_climate_service.stac.schemas import StacCatalogResponse

router = APIRouter()


@router.get("", response_model=StacCatalogResponse)
def get_stac_landing(request: Request) -> dict[str, object]:
    """Return the STAC catalog landing page."""
    return stac_services.build_catalog(request)


@router.get("/catalog.json", response_model=StacCatalogResponse)
def get_stac_catalog_json(request: Request) -> dict[str, object]:
    """Return the STAC catalog landing page (catalog.json alias)."""
    return stac_services.build_catalog(request)


@router.get("/collections/{dataset_id}")
def get_stac_collection(dataset_id: str, request: Request) -> dict[str, object]:
    """Return one STAC collection document."""
    return stac_services.build_collection(dataset_id, request)


def _markdown(text: str) -> PlainTextResponse:
    return PlainTextResponse(text, media_type=stac_services.MARKDOWN_MEDIA_TYPE)


@router.get("/README.md", response_class=PlainTextResponse)
def get_catalog_readme(request: Request) -> PlainTextResponse:
    """What the catalog holds, as the catalog's `describedby` link."""
    return _markdown(documents.catalog_readme(request))


@router.get("/AGENTS.md", response_class=PlainTextResponse)
def get_catalog_agents(request: Request) -> PlainTextResponse:
    """How an agent reads the catalog, as its `agents` link."""
    return _markdown(documents.catalog_agents(request))


@router.get("/collections/{dataset_id}/README.md", response_class=PlainTextResponse)
def get_collection_readme(dataset_id: str, request: Request) -> PlainTextResponse:
    """A feature collection's title, description, licence and provenance."""
    return _markdown(documents.collection_readme(dataset_id, request))


@router.get("/collections/{dataset_id}/AGENTS.md", response_class=PlainTextResponse)
def get_collection_agents(dataset_id: str, request: Request) -> PlainTextResponse:
    """How an agent reads a feature collection's data."""
    return _markdown(documents.collection_agents(dataset_id, request))
