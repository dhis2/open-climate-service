"""FastAPI router exposing the data source endpoints (`/data-sources`)."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from open_climate_service.shared.urls import mount_prefix

from .services import datasets

router = APIRouter()


def _with_ingestability(dataset: dict[str, Any]) -> dict[str, Any]:
    """A template plus the derived `ingestable` flag.

    Derived rather than declared, so a template author cannot get it wrong: it is read from
    the same `ingestion.plugin` that `create_artifact` requires. Without it the listing gives
    an operator no way to tell an ingestable template from a workflow output, and the answer
    arrives as a failed ingest (CLIM-912).
    """
    return {**dataset, "itemType": "coverage", "ingestable": datasets.is_ingestable(dataset)}


def _feature_template_view(template: dict[str, Any], providers: dict[str, Any] | None = None) -> dict[str, Any]:
    """A feature collection template, marked as one, with whether its provider can fetch it.

    `itemType` tells the two kinds apart as it does on `GET /datasets`. For a feature template
    `ingestable` means its provider is available here, and it is fetched with
    `POST /features/{id}/refresh` rather than `POST /ingestions`.
    """
    from open_climate_service.features.services import is_refreshable

    return {**template, "itemType": "feature", "ingestable": is_refreshable(template, providers)}


@router.get(
    "",
    response_model=list[dict[str, Any]],
    responses={200: {"content": {"text/html": {"schema": {"type": "string"}}}}},
)
def list_data_sources(request: Request, response: Response) -> list[dict[str, Any]] | HTMLResponse:
    """Return the available data sources from the registry.

    JSON by default. A browser gets the page the rail links to: only a client ranking
    `text/html` above JSON, with `?f=html` and `?f=json` deciding outright.

    The page is a narrower view than the JSON: it lists what this instance can *fetch*, while
    the JSON lists every data source and flags `ingestable`. A data source produced by a
    workflow is shown under Workflows instead, where the thing that makes it can be seen beside it.

    Raster data sources come first, then feature collection data sources, each marked with
    `itemType` (`coverage` or `feature`), as datasets are on `GET /datasets`.
    """
    from open_climate_service.features import providers as feature_providers
    from open_climate_service.features.services import usable_feature_templates
    from open_climate_service.system.templates import prefers_html, render_data_sources_page

    # Two representations share this URL, so a cache keyed on the URL alone would serve one
    # client the other's. Set on both arms: the JSON arm is a value FastAPI serialises, so the
    # header goes on the shared `response` rather than on a response object of our own.
    response.headers["Vary"] = "Accept"
    if prefers_html(request):
        page = HTMLResponse(render_data_sources_page(mount_prefix(request)))
        page.headers["Vary"] = "Accept"
        return page
    rasters = [_with_ingestability(dataset) for dataset in datasets.list_datasets()]
    providers = feature_providers.load_feature_providers()
    return rasters + [_feature_template_view(template, providers) for template in usable_feature_templates()]


def _get_dataset_or_404(dataset_id: str) -> dict[str, Any]:
    """Look up a dataset template by ID or raise 404."""
    dataset = datasets.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Data source '{dataset_id}' not found")
    return dataset


@router.get(
    "/{dataset_id}",
    response_model=dict,
    responses={200: {"content": {"text/html": {"schema": {"type": "string"}}}}},
)
def get_data_source(dataset_id: str, request: Request, response: Response) -> dict[str, Any] | HTMLResponse:
    """Get a single data source by ID with derived coverage metadata.

    JSON by default. A browser gets the page, where the data source can also be ingested;
    `?f=html` and `?f=json` choose explicitly.
    """
    # Note: have to import inside function to avoid circular import
    from open_climate_service.system.templates import prefers_html, render_data_source_page

    from ..data_accessor.services.accessor import get_data_coverage

    response.headers["Vary"] = "Accept"
    if datasets.get_dataset(dataset_id) is None:
        # Not a raster template; a feature collection template is served from the same place.
        from open_climate_service.features import templates as feature_templates
        from open_climate_service.system.templates import render_feature_source_page

        feature = feature_templates.get_feature_template(dataset_id)
        if feature is None:
            raise HTTPException(status_code=404, detail=f"Data source '{dataset_id}' not found")
        if prefers_html(request):
            page = HTMLResponse(render_feature_source_page(feature, mount_prefix(request)))
            page.headers["Vary"] = "Accept"
            return page
        from open_climate_service.features.services import registered_collections

        # The counterpart of a raster template's coverage: whether the collection has been fetched.
        return {**_feature_template_view(feature), "has_data": dataset_id in registered_collections()}
    dataset = _get_dataset_or_404(dataset_id)
    if prefers_html(request):
        # A workflow output is a template too, so resolving by id alone gave one a page with an
        # ingest form it cannot use. Its page is the workflow's, where what makes it is visible
        # beside it; the JSON arm still describes it here, flagged `ingestable: false`.
        if not datasets.is_ingestable(dataset):
            raise HTTPException(status_code=404, detail=f"Data source '{dataset_id}' has no page")
        page = HTMLResponse(render_data_source_page(dataset, mount_prefix(request)))
        page.headers["Vary"] = "Accept"
        return page
    coverage = get_data_coverage(dataset)
    dataset.update(coverage)
    return _with_ingestability(dataset)
