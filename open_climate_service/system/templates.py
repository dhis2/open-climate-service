"""Server-side HTML rendering and root resource representations for the Open Climate Service."""

import importlib.resources
import logging
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import jinja2
from fastapi import Request

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents.services import get_extent
from open_climate_service.ingestions.services import list_datasets

from .schemas import Link, RootResponse

_env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=True)

_cache: dict[str, jinja2.Template] = {}

_log = logging.getLogger(__name__)

try:
    app_version = _pkg_version("open-climate-service")
except PackageNotFoundError:
    app_version = "unknown"

ROOT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Landing page (HTML) or navigation document (JSON)",
        "content": {
            "text/html": {"schema": {"type": "string"}},
            "application/json": {"schema": RootResponse.model_json_schema()},
        },
    }
}


def root_json(base: str) -> RootResponse:
    """Build the root navigation document for the JSON representation."""
    return RootResponse(
        message="Welcome to Open Climate Service",
        links=[
            Link(href=f"{base}/stac/catalog.json", rel="stac", title="STAC Catalog"),
            Link(href=f"{base}/extent", rel="extent", title="Extent"),
            Link(href=f"{base}/ingestions", rel="ingestions", title="Ingestions"),
            Link(href=f"{base}/datasets", rel="datasets", title="Datasets"),
            Link(href=f"{base}/docs", rel="docs", title="API Docs"),
        ],
    )


def get_template(name: str) -> jinja2.Template:
    """Load and cache a Jinja2 template from the bundled templates/ directory."""
    if name not in _cache:
        resource = importlib.resources.files("open_climate_service") / "templates" / name
        _cache[name] = _env.from_string(resource.read_text(encoding="utf-8"))
    return _cache[name]


def _media_type_q(accept: str, media_type: str) -> float:
    """Return the effective q-value for media_type in an Accept header, or -1.0 if absent.

    Handles RFC 7231 wildcards: exact matches beat type/* beats */*. An API client
    sending Accept: */* (e.g. the requests library default) therefore matches any
    media type at q=1.0, and JSON wins over HTML when both match equally.
    """
    media_type_family = media_type.split("/", 1)[0]
    exact_q = -1.0
    family_q = -1.0
    wildcard_q = -1.0
    for item in accept.split(","):
        parts = item.strip().split(";")
        token = parts[0].strip()
        q = 1.0
        for param in parts[1:]:
            param = param.strip()
            if param.startswith("q="):
                try:
                    q = float(param[2:])
                except ValueError:
                    pass
        if token == media_type:
            exact_q = max(exact_q, q)
        elif token == f"{media_type_family}/*":
            family_q = max(family_q, q)
        elif token == "*/*":
            wildcard_q = max(wildcard_q, q)
    if exact_q >= 0:
        return exact_q
    if family_q >= 0:
        return family_q
    return wildcard_q


def wants_json(request: Request) -> bool:
    """Return True if the client prefers a JSON response over HTML.

    JSON is preferred when application/json is present with a q-value greater
    than or equal to text/html. HTML wins only when text/html has a strictly
    higher q-value, matching RFC 7231 content negotiation semantics.
    """
    if request.query_params.get("f") == "json":
        return True
    accept = request.headers.get("accept", "")
    if not accept:
        return False
    json_q = _media_type_q(accept, "application/json")
    html_q = _media_type_q(accept, "text/html")
    return json_q >= 0 and (html_q < 0 or json_q >= html_q)


def render_maps(mount: str) -> str:
    """Render the map viewer page.

    Takes a mount prefix rather than a base URL: every link and fetch target in the page is
    same-origin, so the template emits `{{ mount }}`-prefixed paths. A path carries no scheme
    or host, so it inherits both from the page — which fixes the mixed-content bug in CLIM-974
    without the risk of substituting a *different* origin, as an operator on a port-forward
    would otherwise have the viewer fetch the configured public instance's catalogue. The
    prefix is what keeps those paths resolving under `--root-path`.

    Required rather than defaulted: an omitted mount yields links that work unmounted and 404
    behind a prefix, which is the failure this parameter exists to prevent.
    """
    return get_template("map-viewer.html").render(mount=mount, name=api_config.get_name())


def _load_extent() -> dict[str, Any] | None:
    try:
        return get_extent()
    except ValueError:
        return None
    except Exception:
        _log.exception("Unexpected error loading extent")
        return None


def _load_templates() -> list[dict[str, Any]]:
    try:
        return registry_datasets.list_datasets()
    except Exception:
        _log.exception("Unexpected error loading dataset templates")
        return []


def _ingestable_templates() -> list[dict[str, Any]]:
    """Return only templates that can be ingested from a source.

    Ingestable templates declare an ``ingestion.plugin``. Derived/static templates
    (e.g. workflow outputs published via ``save_result``, which have ``sync.kind:
    static`` and no upstream fetch path) are excluded so they don't appear in the
    ingest form.
    """
    return [t for t in _load_templates() if (t.get("ingestion") or {}).get("plugin")]


def _load_datasets() -> list[Any]:
    try:
        return list_datasets().items
    except Exception:
        _log.exception("Unexpected error loading datasets")
        return []


def render_landing(version: str, mount: str) -> str:
    """Render the root landing page with live instance status."""
    return get_template("landing_page.html").render(
        version=version,
        mount=mount,
        name=api_config.get_name(),
        extent=_load_extent(),
        datasets=_load_datasets(),
        templates=_load_templates(),
        # Read-only instances refuse /manage, so offering the link would advertise a 403.
        read_only=api_config.is_read_only(),
    )


def render_manage(version: str, mount: str, message: str | None = None, error: str | None = None) -> str:
    """Render the management page."""
    today = date.today().isoformat()
    year_ago = date.today().replace(year=date.today().year - 1).isoformat()
    return get_template("manage.html").render(
        version=version,
        mount=mount,
        name=api_config.get_name(),
        extent=_load_extent(),
        templates=_ingestable_templates(),
        datasets=_load_datasets(),
        today=today,
        year_ago=year_ago,
        message=message,
        error=error,
    )
