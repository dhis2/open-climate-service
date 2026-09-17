"""Server-side HTML rendering and root resource representations for the Open Climate Service."""

import functools
import importlib.resources
import json
import logging
import re
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import jinja2
from fastapi import Request
from markupsafe import Markup, escape

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


LOGO = Markup("""<svg class="logo" viewBox="0 0 32 32" width="28" height="28" aria-hidden="true" focusable="false">
  <defs><clipPath id="ocs-logo-globe"><circle cx="15" cy="17" r="12" /></clipPath></defs>
  <g clip-path="url(#ocs-logo-globe)" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <path d="M-1 15c4 0 4 3 8 3s4-3 8-3 4 3 8 3 4-3 8-3" opacity=".9" />
    <path d="M-1 21c4 0 4 3 8 3s4-3 8-3 4 3 8 3 4-3 8-3" opacity=".65" />
    <path d="M-1 27c4 0 4 3 8 3s4-3 8-3 4 3 8 3 4-3 8-3" opacity=".45" />
  </g>
  <circle cx="15" cy="17" r="12" fill="none" stroke="currentColor" stroke-width="2" />
  <circle cx="26" cy="7" r="4.5" fill="#ffa902" />
</svg>""")
"""The instance mark: a climate over a globe, drawn inline so it needs no asset and no request.

Monochrome in `currentColor` so it takes the header bar's white, with the sun in the DHIS2
yellow400 token. Sized for the 48px bar, and legible at that size because it is three shapes.
"""


_NAV_ITEMS = (
    ("overview", "Overview", "/#overview"),
    ("datasets", "Datasets", "/#datasets"),
    ("data-sources", "Data sources", "/#data-sources"),
    ("workflows", "Workflows", "/#workflows"),
    ("processes", "Processes", "/#processes"),
    ("map", "Map viewer", "/map"),
    ("api", "API", "/api"),
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


def _load_datasets() -> list[Any]:
    try:
        return list_datasets().items
    except Exception:
        _log.exception("Unexpected error loading datasets")
        return []


def _load_workflows() -> list[Any]:
    try:
        from open_climate_service.openeo.workflows import list_workflows

        return sorted(list_workflows().processes, key=lambda workflow: workflow.id)
    except Exception:
        _log.exception("Unexpected error loading workflows")
        return []


_PROCESS_ORIGINS = {
    "ocs": "OCS",
    "xclim": "Climate indicators (xclim)",
    "earthkit": "Meteorology (earthkit)",
    "core": "openEO core",
}


def _load_processes() -> list[dict[str, Any]]:
    """The process catalogue this instance actually loaded, each tagged with where it comes from.

    Grouped by origin rather than openEO `categories`, which most processes do not declare.
    Origin is decided by which loader registered the callable that won, so an instance plugin
    overriding an xclim indicator by id counts as OCS, not xclim.
    """
    try:
        from open_climate_service.openeo import earthkit_processes, xclim_processes
        from open_climate_service.openeo.plugin_processes import load_plugin_processes
        from open_climate_service.openeo.processes import list_openeo_processes

        processes = list_openeo_processes()
        xclim = {id(func) for func in xclim_processes.scan()}
        earthkit = {id(func) for func in earthkit_processes.scan()}
        plugins = dict(load_plugin_processes())
    except Exception:
        _log.exception("Unexpected error loading processes")
        return []

    def origin(process_id: str) -> str:
        func = plugins.get(process_id)
        if func is None:
            return "core"
        if id(func) in xclim:
            return "xclim"
        if id(func) in earthkit:
            return "earthkit"
        return "ocs"

    order = list(_PROCESS_ORIGINS)
    views = []
    for process in processes:
        process_id = str(process.get("id") or "")
        if not process_id:
            continue
        kind = origin(process_id)
        views.append(
            {
                "id": process_id,
                "summary": " ".join(str(process.get("summary") or "").split()),
                "categories": [str(category) for category in process.get("categories") or []],
                "origin": kind,
                "origin_label": _PROCESS_ORIGINS[kind],
            }
        )
    return sorted(views, key=lambda view: (order.index(view["origin"]), view["id"]))


def _licence_label(template: dict[str, Any]) -> str | None:
    """The licence as a short label: an SPDX id as written, or a named licence's name."""
    licence = template.get("license")
    if isinstance(licence, str):
        return licence
    name = licence.get("name") if isinstance(licence, dict) else None
    return name if isinstance(name, str) else None


def _source_view(template: dict[str, Any]) -> dict[str, Any]:
    """A data source card: titled by the dataset, with the provider beneath it."""
    return {
        "id": template["id"],
        "name": template.get("name") or template["id"],
        "provider": template.get("source") or "",
        "provider_url": template.get("source_url"),
        "description": " ".join(str(template.get("description") or "").split()),
        "variable": template.get("variable") or "",
        "units": template.get("units") or "",
        "period_type": template.get("period_type") or "",
        "resolution": template.get("resolution") or "",
        "licence": _licence_label(template),
    }


@functools.lru_cache(maxsize=64)
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


def _dataset_views(datasets: list[Any], templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {template["id"]: template for template in templates}
    views = []
    for dataset in datasets:
        template = by_id.get(dataset.source_dataset_id) or by_id.get(dataset.dataset_id)
        try:
            views.append(_dataset_view(dataset, template))
        except Exception:
            _log.exception("Unexpected error preparing dataset '%s' for the landing page", dataset.dataset_id)
    return views


def _landing_catalogue(templates: list[dict[str, Any]], workflows: list[Any]) -> dict[str, Any]:
    """Split templates between Data sources and Workflows by whether they can be ingested.

    A template is fetched or produced, never both (registration refuses `produced_by` beside
    `ingestion.plugin`), so each appears in exactly one area. A non-ingestable template whose
    `produced_by` names no known workflow — or that declares none — is still listed, under
    Workflows as an output of an unknown workflow, rather than silently dropped.
    """
    sources = sorted(
        (_source_view(t) for t in templates if registry_datasets.is_ingestable(t)),
        key=lambda source: (source["provider"].lower(), source["name"].lower()),
    )
    workflow_ids = {workflow.id for workflow in workflows}
    outputs: dict[str, list[dict[str, Any]]] = {workflow_id: [] for workflow_id in workflow_ids}
    unattributed: list[dict[str, Any]] = []
    for template in templates:
        if registry_datasets.is_ingestable(template):
            continue
        view = _source_view(template)
        produced_by = template.get("produced_by")
        if isinstance(produced_by, str) and produced_by in workflow_ids:
            outputs[produced_by].append(view)
        else:
            unattributed.append(view)
    for views in outputs.values():
        views.sort(key=lambda view: view["name"].lower())
    unattributed.sort(key=lambda view: view["name"].lower())
    return {
        "sources": sources,
        "workflows": [
            {
                "record": workflow,
                "title": _workflow_title(workflow.id),
                "results": _workflow_results(workflow),
                "outputs": outputs[workflow.id],
            }
            for workflow in workflows
        ],
        "unattributed_outputs": unattributed,
    }


def render_landing(version: str, mount: str) -> str:
    """Render the root landing page with live instance status."""
    datasets = _load_datasets()
    templates = _load_templates()
    catalogue = _landing_catalogue(templates, _load_workflows())
    return get_template("landing_page.html").render(
        version=version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        extent=_load_extent(),
        datasets=_dataset_views(datasets, templates),
        published_count=sum(1 for dataset in datasets if dataset.publication.status == "published"),
        sources=catalogue["sources"],
        workflows=catalogue["workflows"],
        unattributed_outputs=catalogue["unattributed_outputs"],
        processes=_load_processes(),
        process_origins=_PROCESS_ORIGINS,
        # Shown on the overview, so a visitor knows why no page offers ingest or sync.
        read_only=api_config.is_read_only(),
    )


@functools.lru_cache(maxsize=4)
def _read_asset(name: str) -> str:
    resource = importlib.resources.files("open_climate_service") / "templates" / name
    return resource.read_text(encoding="utf-8")


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


def _ingest_defaults(template: dict[str, Any], today: date) -> dict[str, Any]:
    """Prefilled start and end for the ingest form, following the source's direction.

    History defaults to the past year; a forecast leaves
    both blank, meaning "from now, as far ahead as the source offers"; a span that crosses now
    runs to the declared end, so the projected years are not cut off at today.
    """
    direction = str(template.get("temporal_direction") or "past")
    year_ago = today.replace(year=today.year - 1).isoformat()
    declared_end = _mapping(_mapping(template.get("extents")).get("temporal")).get("end")
    if direction == "future":
        return {"start": "", "end": "", "start_required": False, "direction": direction}
    end = str(declared_end) if direction == "spanning" and declared_end else today.isoformat()
    return {"start": year_ago, "end": end, "start_required": True, "direction": direction}


def _data_source_page_context(
    template: dict[str, Any], datasets: list[Any], *, read_only: bool, has_extent: bool, today: date
) -> dict[str, Any]:
    """Everything the data source page shows, and whether it can offer the ingest form."""
    display = _mapping(template.get("display"))
    sync = _mapping(template.get("sync"))
    sync_kind = str(sync.get("kind") or "")
    extents = _mapping(template.get("extents"))
    temporal = _mapping(extents.get("temporal"))
    bbox = _mapping(extents.get("spatial")).get("bbox")
    version = sync.get("version")
    providers = ", ".join(
        str(provider["name"])
        for provider in template.get("providers") or []
        if isinstance(provider, dict) and provider.get("name")
    )
    ingestable = registry_datasets.is_ingestable(template)
    ingested = next((dataset for dataset in datasets if dataset.dataset_id == template["id"]), None)
    display_range = display.get("range")

    about: list[Fact] = [
        ("Identifier", str(template["id"]), None),
        ("Short name", str(template.get("short_name") or ""), None),
        ("Provider", str(template.get("source") or ""), template.get("source_url")),
        ("Providers", providers, None),
        ("Licence", _licence_label(template) or "", None),
    ]
    data: list[Fact] = [
        ("Variable", str(template.get("variable") or ""), None),
        ("Standard name", str(template.get("standard_name") or ""), None),
        ("Units", str(template.get("units") or ""), None),
        ("Cell methods", str(template.get("cell_methods") or ""), None),
        ("Period", str(template.get("period_type") or ""), None),
        ("Available", _coverage_label(temporal.get("begin"), temporal.get("end")), None),
        ("Direction", str(template.get("temporal_direction") or ""), None),
        ("Resolution", str(template.get("resolution") or ""), None),
        (
            "Coverage",
            ", ".join(str(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else "",
            None,
        ),
    ]
    status: list[Fact] = [
        ("Updates", _SYNC_KIND_LABELS.get(sync_kind, sync_kind), None),
        (
            "Release",
            f"{version.get('authority')}:{version.get('value')}" if isinstance(version, dict) else str(version or ""),
            None,
        ),
        ("Colour scale", str(display.get("colormap") or ""), None),
        (
            "Display range",
            f"{display_range[0]} – {display_range[1]}"
            if isinstance(display_range, list | tuple) and len(display_range) == 2
            else "",
            None,
        ),
    ]

    def present(facts: list[Fact]) -> list[Fact]:
        return [fact for fact in facts if fact[1]]

    period = str(template.get("period_type") or "")
    return {
        "source": {
            "id": template["id"],
            "name": template.get("name") or template["id"],
            "provider": template.get("source") or "",
            "ramp": _colormap_ramp(display.get("colormap") if isinstance(display.get("colormap"), str) else None),
        },
        "paragraphs": _paragraphs(str(template.get("description") or "")),
        "about": present(about),
        "data": present(data),
        "status_facts": present(status),
        "ingestable": ingestable,
        "produced_by": template.get("produced_by") if not ingestable else None,
        "ingested": (
            {
                "coverage": _coverage_label(ingested.extent.temporal.start, ingested.extent.temporal.end),
                "status": "published" if ingested.publication.status == "published" else "unpublished",
            }
            if ingested is not None
            else None
        ),
        "can_ingest": ingestable and not read_only and has_extent,
        "read_only": read_only,
        "has_extent": has_extent,
        "defaults": _ingest_defaults(template, today),
        "format_hint": _PERIOD_FORMAT_HINTS.get(period, ""),
    }


def render_data_source_page(template: dict[str, Any], mount: str) -> str:
    """Render the page for one data source, with the form that ingests it."""
    read_only = api_config.is_read_only()
    return get_template("data_source_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "data-sources"),
        job_script=_read_asset("ocs_jobs.js"),
        **_data_source_page_context(
            template,
            _load_datasets(),
            read_only=read_only,
            has_extent=_load_extent() is not None,
            today=date.today(),
        ),
    )


_TITLE_WORDS = {"chap": "CHAP", "csv": "CSV", "dhis2": "DHIS2", "json": "JSON"}

_RESULT_FORMATS = {
    "zarr": ("publish", "Publishes a dataset"),
    "geozarr": ("publish", "Publishes a dataset"),
    "chapcsv": ("export", "Exports CHAP CSV"),
    "dhis2json": ("export", "Exports DHIS2 JSON"),
}


def _workflow_title(workflow_id: str) -> str:
    """`aggregate_to_chap_csv` → `Aggregate to CHAP CSV`: workflows declare no title of their own."""
    words = [_TITLE_WORDS.get(word, word) for word in workflow_id.split("_")]
    return " ".join([words[0][:1].upper() + words[0][1:], *words[1:]]) if words else workflow_id


def _workflow_results(record: Any) -> list[tuple[str, str]]:
    """What the workflow's `save_result` nodes produce, as (kind, label) pairs."""
    results = []
    for node in (getattr(record, "process_graph", None) or {}).values():
        if not isinstance(node, dict) or node.get("process_id") != "save_result":
            continue
        fmt = str(_mapping(node.get("arguments")).get("format") or "")
        results.append(_RESULT_FORMATS.get(fmt.lower(), ("other", f"Returns {fmt}" if fmt else "Returns a result")))
    return list(dict.fromkeys(results))


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _inline_code(text: str) -> Markup:
    """Escape *text*, rendering the inline markdown descriptions use: code spans and web links.

    Only `code`, ``code``, **bold** and [text](http…) are recognised; anything else stays
    literal text.
    Links are limited to http(s), so a description cannot smuggle in a `javascript:` URL.
    """
    parts = str(text).replace("``", "`").split("`")
    rendered = []
    for index, part in enumerate(parts):
        if index % 2:
            rendered.append(Markup("<code>{}</code>").format(part))
            continue
        pieces, last = [], 0
        for match in _MARKDOWN_LINK.finditer(part):
            pieces.append(_bold(part[last : match.start()]))
            pieces.append(Markup('<a href="{}">{}</a>').format(match.group(2), match.group(1)))
            last = match.end()
        pieces.append(_bold(part[last:]))
        rendered.append(Markup("").join(pieces))
    return Markup("").join(rendered)


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _bold(text: str) -> Markup:
    # Applied to escaped text, so the markup it adds is the only markup there is.
    return Markup(_BOLD.sub(r"<strong>\1</strong>", str(escape(text))))


_LIST_ITEM = re.compile(r"^\s*(?:[*-]|\d+\.)\s+")


def _description_blocks(text: str | None) -> list[dict[str, Any]]:
    """Blocks of a workflow or process description: paragraphs, bullet lists and code.

    A block that is a JSON example, or fenced with backticks, becomes code; a block whose lines
    all start with a list marker becomes a list.
    """
    blocks: list[dict[str, Any]] = []
    for block in (text or "").replace("\r\n", "\n").split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            code = stripped.strip("`").split("\n", 1)
            blocks.append({"code": code[1] if len(code) > 1 else code[0]})
        elif stripped[0] in "{[" and not _MARKDOWN_LINK.match(stripped):
            blocks.append({"code": stripped})
        elif all(_LIST_ITEM.match(line) for line in stripped.splitlines()):
            blocks.append(
                {"bullets": [_inline_code(_LIST_ITEM.sub("", line).strip()) for line in stripped.splitlines()]}
            )
        else:
            blocks.append({"html": _inline_code(" ".join(stripped.split()))})
    return blocks


def _parameter_type(schema: object) -> str:
    """A short label for a parameter's schema: type or subtype, and the allowed values."""
    schemas = schema if isinstance(schema, list) else [schema]
    labels = []
    for item in schemas:
        item = _mapping(item)
        label = str(item.get("subtype") or item.get("type") or "")
        if isinstance(item.get("enum"), list):
            label = " | ".join(str(value) for value in item["enum"])
        if label:
            labels.append(label)
    return " or ".join(dict.fromkeys(labels))


def _parameter_views(record: Any) -> list[dict[str, Any]]:
    """Rows for a parameters table, from an openEO process or workflow description."""
    return [
        {
            "name": str(parameter.get("name") or ""),
            "required": not parameter.get("optional", False),
            "type": _parameter_type(parameter.get("schema")),
            "default": json.dumps(parameter["default"]) if "default" in parameter else "",
            "description": _inline_code(" ".join(str(parameter.get("description") or "").split())),
        }
        for parameter in record.parameters
        if isinstance(parameter, dict)
    ]


def _workflow_page_context(
    record: Any, templates: list[dict[str, Any]], datasets: list[Any], triggers: list[Any]
) -> dict[str, Any]:
    held = {dataset.dataset_id for dataset in datasets}
    outputs = sorted(
        (
            {"id": t["id"], "name": t.get("name") or t["id"], "ingested": t["id"] in held}
            for t in templates
            if t.get("produced_by") == record.id and not registry_datasets.is_ingestable(t)
        ),
        key=lambda output: str(output["name"]).lower(),
    )
    parameters = _parameter_views(record)
    return {
        "workflow": {
            "id": record.id,
            "title": _workflow_title(record.id),
            "summary": record.summary or "",
            "results": _workflow_results(record),
        },
        "blocks": _description_blocks(record.description),
        "parameters": parameters,
        "outputs": outputs,
        "triggers": [
            {
                "id": trigger.id,
                "on_update_of": trigger.on_update_of,
                "held": trigger.on_update_of in held,
                "arguments": json.dumps(trigger.arguments, indent=2) if trigger.arguments else "",
            }
            for trigger in triggers
            if trigger.workflow_id == record.id
        ],
    }


def _load_triggers() -> list[Any]:
    try:
        from open_climate_service.automation.config import get_automation_config

        return list(get_automation_config().workflow_triggers)
    except Exception:
        _log.exception("Unexpected error loading workflow triggers")
        return []


def render_workflow_page(record: Any, mount: str) -> str:
    """Render the page for one workflow: what it does, its parameters and what it produces."""
    return get_template("workflow_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "workflows"),
        **_workflow_page_context(record, _load_templates(), _load_datasets(), _load_triggers()),
    )


def _uses_process(graph: object, process_id: str) -> bool:
    """Whether a process graph calls *process_id*, including inside callbacks."""
    if isinstance(graph, dict):
        if graph.get("process_id") == process_id:
            return True
        return any(_uses_process(value, process_id) for value in graph.values())
    if isinstance(graph, list):
        return any(_uses_process(value, process_id) for value in graph)
    return False


def _process_page_context(process: dict[str, Any], origin_label: str, workflows: list[Any]) -> dict[str, Any]:
    returns = _mapping(process.get("returns"))
    record = type("ProcessParameters", (), {"parameters": process.get("parameters") or []})
    return {
        "process": {
            "id": process["id"],
            "summary": " ".join(str(process.get("summary") or "").split()),
            "origin": origin_label,
            "categories": [str(category) for category in process.get("categories") or []],
            "experimental": bool(process.get("experimental")),
            "deprecated": bool(process.get("deprecated")),
        },
        "blocks": _description_blocks(process.get("description")),
        "parameters": _parameter_views(record),
        "returns": {
            "type": _parameter_type(returns.get("schema")),
            "description": _inline_code(" ".join(str(returns.get("description") or "").split())),
        },
        "links": [
            {"href": str(link["href"]), "title": str(link.get("title") or link["href"])}
            for link in process.get("links") or []
            if isinstance(link, dict) and str(link.get("href", "")).startswith(("http://", "https://"))
        ],
        "used_by": [
            {"id": workflow.id, "title": _workflow_title(workflow.id)}
            for workflow in workflows
            if _uses_process(workflow.process_graph, process["id"])
        ],
    }


def render_process_page(process: dict[str, Any], mount: str) -> str:
    """Render the page for one process: what it does, its parameters and where it is used."""
    origins = {view["id"]: view["origin_label"] for view in _load_processes()}
    return get_template("process_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "processes"),
        **_process_page_context(process, origins.get(process["id"], ""), _load_workflows()),
    )


_API_GROUP_NOTES = {
    "Datasets": "What this instance holds, and the metadata for each dataset.",
    "Dataset templates": "The data sources this instance can ingest, and whether each one is ingestable.",
    "Ingestions": "Fetch a data source into this instance, and follow the job it starts.",
    "Sync": "Bring an ingested dataset up to date, or ask what a sync would do.",
    "Zarr": "The datasets themselves, as Zarr over HTTP for any Zarr-aware client.",
    "Icechunk": "The same stores for the Icechunk SDK, with version history.",
    "STAC": "Catalogue metadata for discovery, one collection per published dataset.",
    "openEO": "Process graphs: collections, processes, stored workflows, jobs and synchronous results.",
    "Extent": "The area this instance covers.",
    "Schedules": "Scheduled dataset refreshes, as configured for this instance.",
    "Exports": "Deliver an export to its destination, and follow the delivery job.",
    "System": "Health, version and the landing page's JSON form.",
}

_API_GROUP_ORDER = list(_API_GROUP_NOTES)


def _endpoint_summary(operation: dict[str, Any]) -> str:
    """One line for an endpoint: its docstring's first line, or the generated summary.

    FastAPI derives `summary` from the function name, so `read_index` becomes "Read Index".
    The docstring says something, and is what the API docs show as the description.
    """
    description = str(operation.get("description") or "").strip()
    if description:
        return description.splitlines()[0].strip()
    return str(operation.get("summary") or "")


def _api_page_context(schema: dict[str, Any], *, read_only: bool) -> dict[str, Any]:
    """Group the instance's own OpenAPI paths for the API page.

    Built from the served schema rather than a written list, so the page describes the routes
    this instance actually exposes — including those a plugin or an optional dependency adds.
    """
    from open_climate_service.read_only import is_blocked

    groups: dict[str, list[dict[str, Any]]] = {}
    for path, operations in _mapping(schema.get("paths")).items():
        for method, operation in _mapping(operations).items():
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags") or ["Other"]
            group = str(tags[0])
            groups.setdefault(group, []).append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": _endpoint_summary(operation),
                    "closed": read_only and is_blocked(method.upper(), path),
                }
            )
    ordered = sorted(
        groups.items(),
        key=lambda item: _API_GROUP_ORDER.index(item[0]) if item[0] in _API_GROUP_ORDER else len(_API_GROUP_ORDER),
    )
    entry_points = [
        ("STAC catalogue", "/stac/catalog.json", "Browsable metadata for every published dataset"),
        ("openEO capabilities", "/?f=json", "What this backend supports, for an openEO client"),
        ("openEO collections", "/collections", "The datasets an openEO process graph can load"),
        ("API documentation", "/docs", "Interactive Swagger UI for every endpoint below"),
        ("OpenAPI schema", "/openapi.json", "The machine-readable description this page is built from"),
    ]
    return {
        "entry_points": [{"title": title, "path": path, "note": note} for title, path, note in entry_points],
        "groups": [
            {
                "name": name,
                "note": _API_GROUP_NOTES.get(name, ""),
                "endpoints": sorted(endpoints, key=lambda endpoint: (endpoint["path"], endpoint["method"])),
            }
            for name, endpoints in ordered
        ],
        "read_only": read_only,
    }


def render_api_page(schema: dict[str, Any], mount: str) -> str:
    """Render the page listing this instance's API endpoints."""
    return get_template("api_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "api"),
        **_api_page_context(schema, read_only=api_config.is_read_only()),
    )


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
