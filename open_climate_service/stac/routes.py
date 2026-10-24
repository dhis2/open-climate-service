"""Routes for the STAC catalogue.

Laid out as a static catalog, so a client that follows links reads it as files: the root
`catalog.json` with `README.md` and `AGENTS.md` beside it, and one directory per collection holding
`collection.json` and, for a feature collection, its own `README.md` and `AGENTS.md`. The
`describedby` and `agents` links are relative, so `/stac` and `/stac/collections/{id}` redirect to
the document rather than serve it from a path those links would resolve against wrongly. The
redirects are relative, so they stay under whatever prefix a proxy serves the instance at. Every
route answers HEAD, as a static host does and a link checker expects.
"""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, RedirectResponse

from open_climate_service.shared.urls import path_segment
from open_climate_service.stac import documents
from open_climate_service.stac import services as stac_services
from open_climate_service.stac.schemas import StacCatalogResponse

router = APIRouter()

_GET_HEAD = ["GET", "HEAD"]


@router.api_route("", methods=_GET_HEAD, include_in_schema=False)
def get_stac_landing() -> RedirectResponse:
    """Redirect to the root `catalog.json`."""
    return RedirectResponse("stac/catalog.json")


# `response_model_exclude_none`: a link with no title or type leaves the field out. The core
# catalog schema allows `title` only as a string, so a `null` one fails validation.
@router.api_route(
    "/catalog.json", methods=_GET_HEAD, response_model=StacCatalogResponse, response_model_exclude_none=True
)
def get_stac_catalog_json(request: Request) -> dict[str, object]:
    """Return the root STAC catalog."""
    return stac_services.build_catalog(request)


@router.api_route("/{dataset_id}/collection.json", methods=_GET_HEAD)
def get_stac_collection(dataset_id: str, request: Request) -> dict[str, object]:
    """Return one STAC collection document."""
    return stac_services.build_collection(dataset_id, request)


@router.api_route("/collections/{dataset_id}", methods=_GET_HEAD, include_in_schema=False)
def redirect_stac_collection(dataset_id: str) -> RedirectResponse:
    """Redirect the STAC API style path to the collection's `collection.json`."""
    return RedirectResponse(f"../{path_segment(dataset_id)}/collection.json")


def _markdown(text: str) -> PlainTextResponse:
    return PlainTextResponse(text, media_type=stac_services.MARKDOWN_MEDIA_TYPE)


@router.api_route("/README.md", methods=_GET_HEAD, response_class=PlainTextResponse)
def get_catalog_readme(request: Request) -> PlainTextResponse:
    """What the catalog holds, as the catalog's `describedby` link."""
    return _markdown(documents.catalog_readme(request))


@router.api_route("/AGENTS.md", methods=_GET_HEAD, response_class=PlainTextResponse)
def get_catalog_agents(request: Request) -> PlainTextResponse:
    """How an agent reads the catalog, as its `agents` link."""
    return _markdown(documents.catalog_agents(request))


@router.api_route("/{dataset_id}/README.md", methods=_GET_HEAD, response_class=PlainTextResponse)
def get_collection_readme(dataset_id: str, request: Request) -> PlainTextResponse:
    """A feature collection's title, description, licence and provenance."""
    return _markdown(documents.collection_readme(dataset_id, request))


@router.api_route("/{dataset_id}/AGENTS.md", methods=_GET_HEAD, response_class=PlainTextResponse)
def get_collection_agents(dataset_id: str, request: Request) -> PlainTextResponse:
    """How an agent reads a feature collection's data."""
    return _markdown(documents.collection_agents(dataset_id, request))
