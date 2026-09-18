"""Server-side HTML rendering and root resource representations for the Open Climate Service."""

import importlib.resources
import logging
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import jinja2
from fastapi import Request
from markupsafe import Markup

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


def _read_asset(name: str) -> str:
    resource = importlib.resources.files("open_climate_service") / "templates" / name
    return resource.read_text(encoding="utf-8")


# The instance mark: a climate over a globe. Monochrome in `currentColor` so it takes the header
# bar's white, with the sun in the DHIS2 yellow400 token; sized for the 48px bar, and legible at
# that size because it is three shapes. Inlined into the page rather than linked, so it costs no
# request and works offline — the rationale lives here rather than in the file, because an XML
# comment in the file would be served to every viewer.
LOGO = Markup(_read_asset("ocs_logo.svg"))

_NAV_ITEMS = (
    ("map", "Map viewer", "/map"),
    ("openeo", "openEO editor", "/openeo"),
)

# Set apart by a gap: the only entry that leaves this instance for another site.
_NAV_GAP_BEFORE = frozenset({"openeo"})

# Leaves the instance for the hosted openEO Web Editor, so it opens in a new tab.
_EXTERNAL_NAV_ITEMS = frozenset({"openeo"})


def page_nav(mount: str, current: str) -> Markup:
    """The left navigation for pages outside the landing page, with *current* marked.

    The landing page renders its own, because its links switch areas in place rather than
    navigate.
    """
    items = Markup("").join(
        Markup('<li{}><a href="{}{}"{}{}>{}</a></li>').format(
            Markup(' class="gap"') if key in _NAV_GAP_BEFORE else "",
            mount,
            path,
            Markup(' aria-current="page"') if key == current else "",
            Markup(' target="_blank" rel="noopener"') if key in _EXTERNAL_NAV_ITEMS else "",
            label,
        )
        for key, label, path in _NAV_ITEMS
    )
    return Markup('<nav class="rail" aria-label="Sections"><ul>{}</ul></nav>').format(items)


def _licence_label(template: dict[str, Any]) -> str | None:
    """The licence as a short label: an SPDX id as written, or a named licence's name."""
    licence = template.get("license")
    if isinstance(licence, str):
        return licence
    name = licence.get("name") if isinstance(licence, dict) else None
    return name if isinstance(name, str) else None


def _colormap_ramp(name: str | None) -> str:
    """A CSS gradient through a colormap, drawn where a dataset has no thumbnail yet.

    The same colormap the map viewer and the thumbnail use, so the placeholder already looks
    like the layer will. Cached: a landing page lists the same few colormaps many times, and
    resolving one imports matplotlib.
    """
    from matplotlib.colors import to_hex

    from open_climate_service.shared.thumbnails import resolve_colormap

    colormap = resolve_colormap(name)
    stops = ", ".join(to_hex(colormap(i / 4)) for i in range(5))
    return f"linear-gradient(90deg, {stops})"


def _coverage_label(start: object, end: object) -> str:
    if not start and not end:
        return ""
    return f"{start or '…'} – {end or '…'}"


def _dataset_view(dataset: Any, template: dict[str, Any] | None) -> dict[str, Any]:
    """A dataset tile or row: thumbnail (or colormap ramp), name, source and description.

    The thumbnail is linked only when its file exists. It is written at ingest and sync, so a
    dataset ingested before thumbnails existed has none until its next sync, and linking a
    missing image would show a broken-image icon rather than the ramp.
    """
    from open_climate_service.shared.thumbnails import thumbnail_path

    display = (template or {}).get("display")
    colormap = display.get("colormap") if isinstance(display, dict) else None
    status = "published" if dataset.publication.status == "published" else "unpublished"
    try:
        has_thumbnail = thumbnail_path(dataset.dataset_id).is_file()
    except OSError:
        has_thumbnail = False
    description = " ".join((dataset.description or "").split())
    return {
        "id": dataset.dataset_id,
        "name": dataset.dataset_name,
        "description": description,
        "source": dataset.source or "",
        "variable": dataset.variable,
        "units": dataset.units or "",
        "period_type": dataset.period_type,
        "coverage": _coverage_label(dataset.extent.temporal.start, dataset.extent.temporal.end),
        "status": status,
        "has_thumbnail": has_thumbnail,
        "ramp": _colormap_ramp(colormap if isinstance(colormap, str) else None),
    }


def _paragraphs(text: str | None) -> list[str]:
    """Split prose on blank lines, joining the wrapped lines within each paragraph."""
    blocks = (text or "").replace("\r\n", "\n").split("\n\n")
    return [" ".join(block.split()) for block in blocks if block.strip()]


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value.strftime("%Y-%m-%d %H:%M UTC"))
    except AttributeError:
        return str(value)


Fact = tuple[str, str, str | None]
"""A dataset page line: label, value, and an optional link."""

_SYNC_KIND_LABELS = {
    "temporal": "New periods are added as the source publishes them",
    "release": "Replaced when the source issues a new release",
    "static": "Does not update",
}


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _record_licence_label(record: Any) -> str:
    # "other" is what a record says when the licence has no SPDX id, including when none was
    # declared at all; shown bare it reads like a licence called "other".
    if record.license == "other":
        return "See licence" if record.license_url else "Not specified"
    return str(record.license)


def _dataset_page_context(record: Any, template: dict[str, Any] | None) -> dict[str, Any]:
    """Everything the dataset page shows: the managed record, plus what its template adds.

    Each fact is a (label, value, href) triple and is dropped when it has no value, so the
    page lists what is known rather than a column of dashes.
    """
    template = template or {}
    summary = _dataset_view(record, template)
    spatial = record.extent.spatial
    display = _mapping(template.get("display"))
    sync = _mapping(template.get("sync"))
    sync_kind = str(sync.get("kind") or "")
    providers = ", ".join(
        str(provider["name"])
        for provider in template.get("providers") or []
        if isinstance(provider, dict) and provider.get("name")
    )
    version = sync.get("version")
    version_label = (
        f"{version.get('authority')}:{version.get('value')}"
        if isinstance(version, dict) and version.get("value")
        else (str(version) if version else "")
    )
    display_range = display.get("range")
    range_label = (
        f"{display_range[0]} – {display_range[1]}"
        if isinstance(display_range, list | tuple) and len(display_range) == 2
        else ""
    )

    if template and not registry_datasets.is_ingestable(template) and template.get("produced_by"):
        origin: Fact = (
            "Produced by",
            f"{template['produced_by']} workflow",
            f"/process_graphs/{template['produced_by']}",
        )
    elif template and registry_datasets.is_ingestable(template):
        origin = ("Origin", "Fetched from the data source", None)
    else:
        origin = ("Origin", "", None)

    about: list[Fact] = [
        ("Identifier", record.dataset_id, None),
        ("Short name", record.short_name or "", None),
        ("Source", record.source or "", record.source_url),
        origin,
        ("Licence", _licence_label(template) or _record_licence_label(record), record.license_url),
        ("Providers", providers, None),
    ]
    data: list[Fact] = [
        ("Variable", record.variable, None),
        ("Standard name", str(template.get("standard_name") or ""), None),
        ("Units", record.units or "", None),
        ("Cell methods", str(template.get("cell_methods") or ""), None),
        ("Period", record.period_type, None),
        ("Temporal coverage", summary["coverage"], None),
        ("Direction", str(template.get("temporal_direction") or ""), None),
        ("Resolution", record.resolution or "", None),
        (
            "Bounding box",
            f"{spatial.xmin:.4f}, {spatial.ymin:.4f}, {spatial.xmax:.4f}, {spatial.ymax:.4f}",
            None,
        ),
    ]
    status: list[Fact] = [
        ("Publication", summary["status"], None),
        ("Published", _format_timestamp(record.publication.published_at), None),
        ("Last updated", _format_timestamp(record.last_updated), None),
        ("Updates", _SYNC_KIND_LABELS.get(sync_kind, sync_kind), None),
        ("Release", version_label, None),
        ("Colour scale", str(display.get("colormap") or ""), None),
        ("Display range", range_label, None),
    ]

    def present(facts: list[Fact]) -> list[Fact]:
        return [fact for fact in facts if fact[1]]

    return {
        "dataset": summary,
        # Static datasets and workflow outputs have no upstream to sync from; the console offers
        # the button for them and the planner answers "not syncable", which is noise here.
        "syncable": bool(template) and sync_kind in {"temporal", "release"},
        "sync_kind": sync_kind,
        "format_hint": _PERIOD_FORMAT_HINTS.get(str(record.period_type), ""),
        "paragraphs": _paragraphs(record.description),
        "about": present(about),
        "data": present(data),
        "status_facts": present(status),
        "links": [link for link in record.links if link.rel != "self"],
        "published": summary["status"] == "published",
        "versions": [
            {
                "created_at": _format_timestamp(version.created_at),
                "format": str(getattr(version.format, "value", version.format)),
                "coverage": _coverage_label(version.coverage.temporal.start, version.coverage.temporal.end),
            }
            for version in sorted(record.versions, key=lambda version: version.created_at, reverse=True)
        ],
    }


def render_dataset_page(record: Any, mount: str) -> str:
    """Render the HTML page for one managed dataset, linked from the landing page."""
    try:
        template = registry_datasets.get_dataset(record.source_dataset_id) or registry_datasets.get_dataset(
            record.dataset_id
        )
    except Exception:
        _log.exception("Unexpected error loading the template for dataset '%s'", record.dataset_id)
        template = None
    return get_template("dataset_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "datasets"),
        job_script=_read_asset("ocs_jobs.js"),
        read_only=api_config.is_read_only(),
        **_dataset_page_context(record, template),
    )


_PERIOD_FORMAT_HINTS = {
    "hourly": "YYYY-MM-DDTHH",
    "daily": "YYYY-MM-DD",
    "dekadal": "YYYY-MM-DD",
    "weekly": "YYYY-MM-DD",
    "monthly": "YYYY-MM",
    "yearly": "YYYY",
}


def prefers_html(request: Request) -> bool:
    """Whether a client asked for HTML over JSON, for an endpoint that is JSON by default.

    Stricter than `wants_json` on purpose. `/` has always defaulted to HTML; a data endpoint
    has always answered JSON, and scripts calling it may send no Accept header or `*/*`.
    Those keep getting JSON, and only a client that ranks `text/html` above JSON — a browser —
    gets the page. `?f=html` and `?f=json` override either way.
    """
    requested = request.query_params.get("f")
    if requested in {"html", "json"}:
        return requested == "html"
    accept = request.headers.get("accept", "")
    if not accept:
        return False
    return _media_type_q(accept, "text/html") > _media_type_q(accept, "application/json")


def render_maps(mount: str) -> str:
    """Render the map viewer page.

    A mount prefix rather than a base URL, because every link and fetch target in the page is
    same-origin. A path carries no scheme or host and inherits both from the page, which fixes
    the mixed-content bug in CLIM-974 without the risk of naming a *different* origin — an
    operator on a port-forward would otherwise have the viewer fetch the public instance's
    catalogue. The prefix keeps those paths resolving under `--root-path`.

    Required rather than defaulted: an omitted mount gives links that work unmounted and 404
    behind a prefix, which is the failure this parameter exists to prevent.
    """
    return get_template("map-viewer.html").render(
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        version=app_version,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "map"),
    )


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

    Templates without an ingestion plugin — typically workflow outputs published via
    ``save_result`` — have no upstream fetch path, so they are excluded from the ingest form. Shares the registry's
    predicate with ``GET /dataset-templates/`` and with the ingest path that refuses them, so
    the form and the API cannot disagree about what is offerable.
    """
    return [t for t in _load_templates() if registry_datasets.is_ingestable(t)]


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
