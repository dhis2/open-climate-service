"""Server-side HTML rendering and root resource representations for the Open Climate Service."""

import functools
import importlib.resources
import json
import logging
import math
import os
import re
import time
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from textwrap import dedent
from typing import Any

import jinja2
from fastapi import Request
from markupsafe import Markup, escape

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.extents.services import get_extent
from open_climate_service.ingestions.services import list_datasets
from open_climate_service.shared.time import datetime_to_period_string

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


@functools.lru_cache(maxsize=4)
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
    ("overview", "Overview", "/"),
    ("datasets", "Datasets", "/datasets"),
    ("data-sources", "Data sources", "/data-sources"),
    ("workflows", "Workflows", "/workflows"),
    ("processes", "Processes", "/processes"),
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


def _licence_label(template: dict[str, Any]) -> str | None:
    """The licence as a short label: an SPDX id as written, or a named licence's name."""
    licence = template.get("license")
    if isinstance(licence, str):
        return licence
    name = licence.get("name") if isinstance(licence, dict) else None
    return name if isinstance(name, str) else None


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
        # Linked, now that the data source has a page: the source page already links to the
        # dataset it produced, so this closes that pair rather than leaving it one-way.
        origin = ("Origin", "Fetched from the data source", f"/data-sources/{template['id']}")
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
        ("Period type", record.period_type, None),
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
        # Above the scale it describes, and above the bar below the list: the range is what makes
        # the colours mean anything, so it reads before the name of the ramp rather than after.
        ("Display range", range_label, None),
        ("Colour scale", str(display.get("colormap") or ""), None),
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
        # One list: what the dataset measures, then where it came from. Two panels of facts side
        # by side in the same column read as one region anyway, and splitting them meant a
        # reader hunting two places for "what is this".
        "data": present(data + about),
        "status_facts": present(status),
        "links": [link for link in record.links if link.rel != "self"],
        "published": summary["status"] == "published",
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
    "weekly": "YYYY-Www",
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


def _period_value(value: str, period: str) -> str:
    """A date as the source's own period identifier.

    The field asks for a period, not a date: a monthly source takes `2025-09`, a yearly one
    `2025`, a weekly one `2025-W38`. Through the shared converter rather than by trimming the
    string, because a week's identifier is not a prefix of the date it falls in — truncating
    left `2026-09-17` unchanged and the form then asked for something the source cannot read.
    """
    if not value or not period:
        return value
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value
    try:
        return datetime_to_period_string(moment, period)
    except Exception:
        _log.exception("Unexpected error formatting %r as a %s period", value, period)
        return value


def _ingest_defaults(template: dict[str, Any], today: date) -> dict[str, Any]:
    """Prefilled start and end for the ingest form, following the source's direction.

    History defaults to the past year; a forecast leaves
    both blank, meaning "from now, as far ahead as the source offers"; a span that crosses now
    runs to the declared end, so the projected years are not cut off at today.
    """
    direction = str(template.get("temporal_direction") or "past")
    period = str(template.get("period_type") or "")
    # Keep the day, and only give it up where it does not exist: clamping every date to the
    # 28th moved the default start back by up to three days for most of each month, which a
    # daily or dekadal source ingests as real extra periods.
    try:
        year_ago = today.replace(year=today.year - 1).isoformat()
    except ValueError:
        # 29 February, where the previous year has none: the 28th is the nearest real date.
        year_ago = today.replace(year=today.year - 1, month=2, day=28).isoformat()
    declared_end = _mapping(_mapping(template.get("extents")).get("temporal")).get("end")
    if direction == "future":
        return {"start": "", "end": "", "start_required": False, "direction": direction}
    end = str(declared_end) if direction == "spanning" and declared_end else today.isoformat()
    return {
        "start": _period_value(year_ago, period),
        "end": _period_value(end, period),
        "start_required": True,
        "direction": direction,
    }


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
    # A managed dataset records the template it came from in `source_dataset_id`; match on that
    # first and fall back to `dataset_id`, as the dataset page and the STAC path both do.
    #
    # Nothing writes an aliased record today — `create_artifact` persists under the template id
    # and no path assigns `source_dataset_id` — so the two are always equal in practice. The
    # ingest form below depends on that: it posts `source.id`, which is the only id
    # `create_artifact` can act on. Introducing aliasing therefore means changing the ingest
    # path to carry a managed id, not just this lookup.
    ingested = next(
        (dataset for dataset in datasets if (dataset.source_dataset_id or dataset.dataset_id) == template["id"]),
        None,
    )
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
        ("Period type", str(template.get("period_type") or ""), None),
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
            f"{version.get('authority')}:{version.get('value')}"
            if isinstance(version, dict) and version.get("value")
            else str(version or ""),
            None,
        ),
        # Above the scale it describes, as on the dataset page: the range is what makes the
        # colours mean anything, so it reads before the name of the ramp rather than after.
        (
            "Display range",
            f"{display_range[0]} – {display_range[1]}"
            if isinstance(display_range, list | tuple) and len(display_range) == 2
            else "",
            None,
        ),
        ("Colour scale", str(display.get("colormap") or ""), None),
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
        # One list, as on the dataset page: what the source measures, then where it comes from.
        "data": present(data + about),
        "status_facts": present(status),
        "ingestable": ingestable,
        "produced_by": template.get("produced_by") if not ingestable else None,
        "ingested": (
            {
                # The dataset's own id, not the template's: the two differ whenever a source was
                # ingested under a different name, and the links below have to reach the dataset.
                "id": ingested.dataset_id,
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


def render_datasets_page(mount: str) -> str:
    """Render the list of datasets this instance holds.

    The area that lived at `/#datasets` as its own page, so it can be linked to, bookmarked and
    reached without JavaScript. `GET /datasets` still answers JSON to anything that does not ask
    for a page.
    """
    return get_template("datasets_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        list_script=_read_asset("ocs_list.js"),
        nav=page_nav(mount, "datasets"),
        datasets=_dataset_views(_load_datasets(), _load_templates()),
    )


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


def render_data_sources_page(mount: str) -> str:
    """Render the list of data sources this instance can fetch from.

    HTML only, like a single data source: the machine-readable list of the same thing is
    `GET /dataset-templates/`, which this does not rename or duplicate.
    """
    templates = _load_templates()
    return get_template("data_sources_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        list_script=_read_asset("ocs_list.js"),
        nav=page_nav(mount, "data-sources"),
        sources=[_source_view(t) for t in _ingestable_templates(templates)],
    )


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


_RST_ROLE = re.compile(r":[a-z]+:`([^`]+)`")
# Not a run of three: a Markdown fence is backticks too, and matching inside one would break
# a description that was already Markdown.
_RST_LITERAL = re.compile(r"(?<!`)``([^`]+)``(?!`)")
_RST_DIRECTIVE = re.compile(r"^\s*\.\. [a-z]+::\s*$")

# Rendered as tables further down the page, so repeating them as prose only duplicates the
# content and leaves the numpydoc underline showing as a row of dashes. Only these two: no page
# renders Raises or Yields, so dropping those would lose what they document rather than repeat it.
_DOC_SECTIONS_SHOWN_ELSEWHERE = frozenset({"parameters", "returns"})


def _is_section_heading(lines: list[str], index: int) -> bool:
    """A numpydoc heading: a title on its own line, underlined with --- or ===."""
    if index + 1 >= len(lines) or not lines[index].strip():
        return False
    rule = lines[index + 1].strip()
    return len(rule) >= 3 and set(rule) in ({"-"}, {"="})


def _from_rst(text: str) -> str:
    """Turn the reStructuredText in a Python docstring into the Markdown the renderer reads.

    Process descriptions are docstrings, and the ones from earthkit are numpydoc: section
    headings underlined with dashes, `.. math::` directives, ``literals`` and :role:`links`.
    Rendered as Markdown those leak — a row of dashes in a paragraph, a bare ".. math::", and
    every role name printed before its argument.

    Only the constructs that actually occur are handled. Anything else passes through, because
    a docstring that is already Markdown must come out unchanged.
    """
    # Inline first: ``literal`` and :role:`x` become `x` before any fence exists, or the
    # literal pattern would match the pair of backticks inside a fence this function adds.
    text = _RST_LITERAL.sub(r"`\1`", text.replace("\r\n", "\n"))
    text = _RST_ROLE.sub(r"`\1`", text)
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        if _is_section_heading(lines, index):
            title = lines[index].strip()
            index += 2
            body: list[str] = []
            # The body runs to the next section, or to a blank-line break — numpydoc separates
            # the last section from any closing prose that way, and that prose is real content.
            while index < len(lines) and not _is_section_heading(lines, index):
                if not lines[index].strip() and index + 1 < len(lines) and not lines[index + 1].strip():
                    break
                body.append(lines[index])
                index += 1
            if title.lower() in _DOC_SECTIONS_SHOWN_ELSEWHERE:
                continue
            out += ["", f"**{title}**", ""] + body
            continue
        if _RST_DIRECTIVE.match(lines[index]):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            body = []
            while index < len(lines) and (lines[index].startswith((" ", "\t")) or not lines[index].strip()):
                body.append(lines[index])
                index += 1
            block = dedent("\n".join(body)).strip()
            # A formula is not prose: keep it whole rather than folding its whitespace away.
            if block:
                out += ["", "```", block, "```", ""]
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out).strip()


def _split_fences(text: str) -> list[tuple[str, bool]]:
    """Split *text* into runs, flagging which are fenced code.

    Taken before paragraphs are, because a fence may contain a blank line: splitting on blank
    lines first cut such a block in two, leaving the remainder as prose with a stray closing
    fence in it.
    """
    runs: list[tuple[str, bool]] = []
    buffer: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            runs.append(("\n".join(buffer), fenced))
            buffer = []
            fenced = not fenced
            continue
        buffer.append(line)
    runs.append(("\n".join(buffer), fenced))
    return [(run, is_code) for run, is_code in runs if run.strip()]


def _description_blocks(text: str | None) -> list[dict[str, Any]]:
    """Blocks of a workflow or process description: paragraphs, bullet lists and code.

    A fenced region is one block whatever it contains; outside them, a block that is a JSON
    example becomes code, and a block whose lines all start with a list marker becomes a list.
    """
    blocks: list[dict[str, Any]] = []
    for run, is_code in _split_fences(_from_rst(text or "")):
        if is_code:
            blocks.append({"code": run.strip("\n") + "\n"})
            continue
        for block in run.split("\n\n"):
            stripped = block.strip()
            if not stripped:
                continue
            if stripped[0] in "{[" and not _MARKDOWN_LINK.match(stripped):
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
    summary = " ".join(str(process.get("summary") or "").split())
    # A docstring's first line is its summary, so for every process that has one the page led
    # with the same sentence twice — once as the lead, once as the opening paragraph. Compared
    # before rendering: the block is Markup by then, so a summary carrying a link, bold or an
    # escaped character would no longer match itself and the repeat would come back.
    description = str(process.get("description") or "")
    first, _, rest = description.partition("\n\n")
    if summary and " ".join(first.split()) == summary:
        description = rest
    blocks = _description_blocks(description)
    return {
        "process": {
            "id": process["id"],
            "summary": summary,
            "origin": origin_label,
            "categories": [str(category) for category in process.get("categories") or []],
            "experimental": bool(process.get("experimental")),
            "deprecated": bool(process.get("deprecated")),
        },
        "blocks": blocks,
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


# coastlines curve the way they do on a globe without the page showing a ball on a background.
# Drawn in a 144x100 viewBox: wider than tall, so the view carries the region around the extent.
_GLOBE_WIDTH, _GLOBE_HEIGHT = 144, 100
_GLOBE_RADIUS = 48
# Below this the sphere's edge enters the corners (the half-diagonal is 87.7, and 48 * 1.85 is
# above it), and the map would read as a ball on a background rather than a piece of the world.
_MIN_ZOOM = 1.85


@functools.lru_cache(maxsize=1)
def _world_rings() -> list[list[tuple[float, float]]]:
    """Coastlines as (lon, lat) rings; see `templates/world_land.json` for what they are."""
    data = json.loads(_read_asset("world_land.json"))
    return [[(point[0], point[1]) for point in ring] for ring in data["rings"]]


def _globe_zoom(width: float, height: float, latitude: float) -> float:
    """How much to magnify the sphere, so the extent sits in its region rather than filling the view.

    Bounded below so the sphere's edge stays outside the frame, and above at 2.2, which shows
    roughly 40 degrees of longitude either side — a country with its continent around it.
    """
    span = max(width * math.cos(math.radians(latitude)), height, 0.5)
    return max(_MIN_ZOOM, min(0.1 * 180 / span, 2.2))


def _project(lon: float, lat: float, lon0: float, lat0: float, scale: float) -> tuple[float, float] | None:
    """Orthographic projection onto the viewBox, or None for a point on the far side."""
    lam, phi = math.radians(lon - lon0), math.radians(lat)
    phi0 = math.radians(lat0)
    cos_c = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(lam)
    if cos_c <= 0:
        return None
    x = math.cos(phi) * math.sin(lam)
    y = math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(lam)
    return _GLOBE_WIDTH / 2 + scale * x, _GLOBE_HEIGHT / 2 - scale * y


def _path(points: list[tuple[float, float] | None], *, close: bool) -> str:
    """An SVG path through *points*, starting a new subpath wherever the horizon cut them."""
    parts: list[str] = []
    run: list[tuple[float, float]] = []
    for point in [*points, None]:
        if point is None:
            if len(run) > 1:
                parts.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in run) + ("Z" if close else ""))
            run = []
        else:
            run.append(point)
    return "".join(parts)


def _densify(bbox: tuple[float, float, float, float], steps: int = 24) -> list[tuple[float, float]]:
    """The bbox as a ring with points along each side, so its edges bend with the sphere."""
    xmin, ymin, xmax, ymax = bbox
    ring: list[tuple[float, float]] = []
    for index in range(steps):
        ring.append((xmin + (xmax - xmin) * index / steps, ymin))
    for index in range(steps):
        ring.append((xmax, ymin + (ymax - ymin) * index / steps))
    for index in range(steps):
        ring.append((xmax - (xmax - xmin) * index / steps, ymax))
    for index in range(steps):
        ring.append((xmin, ymax - (ymax - ymin) * index / steps))
    return ring


@functools.lru_cache(maxsize=8)
def _globe(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """The globe for one extent: the land it shows, and the extent on it."""
    xmin, ymin, xmax, ymax = bbox
    lon0, lat0 = (xmin + xmax) / 2, (ymin + ymax) / 2
    scale = _GLOBE_RADIUS * _globe_zoom(abs(xmax - xmin), abs(ymax - ymin), lat0)
    land = "".join(
        _path([_project(lon, lat, lon0, lat0, scale) for lon, lat in ring], close=True) for ring in _world_rings()
    )
    marker = _path([_project(lon, lat, lon0, lat0, scale) for lon, lat in _densify(bbox)], close=True)
    return {"land": land, "extent": marker, "width": _GLOBE_WIDTH, "height": _GLOBE_HEIGHT}


def _extent_globe(extent: dict[str, Any] | None) -> dict[str, Any] | None:
    """The configured extent drawn on a globe, or None when there is no usable extent."""
    bbox = (extent or {}).get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    xmin, ymin, xmax, ymax = values
    if xmax <= xmin or ymax <= ymin:
        return None
    return _globe((xmin, ymin, xmax, ymax))


_SIZE_CACHE_SECONDS = 60.0


def _directory_bytes(path: Path) -> int:
    """Bytes held under a store directory, following none of its symlinks."""
    total = 0
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


_stored_bytes_cache: tuple[float, int] | None = None


def _stored_bytes() -> int:
    """Total size on disk of every store this instance's artifacts point at.

    Walked rather than read from a record: nothing stores a size, and an Icechunk store grows
    with each sync, so a recorded one would be stale. Distinct paths only — successive
    ingestions of the same dataset append to a single store. Cached for a minute, because a
    store is tens of thousands of chunk files and the overview is reloaded far more often than
    the data changes.
    """
    global _stored_bytes_cache
    now = time.monotonic()
    if _stored_bytes_cache is not None and now - _stored_bytes_cache[0] < _SIZE_CACHE_SECONDS:
        return _stored_bytes_cache[1]
    try:
        from open_climate_service.ingestions.services import list_artifacts

        paths = {artifact.path for artifact in list_artifacts().items if artifact.path}
        total = sum(_directory_bytes(Path(path)) if Path(path).is_dir() else _file_bytes(Path(path)) for path in paths)
    except Exception:
        _log.exception("Unexpected error measuring stored data")
        total = 0
    _stored_bytes_cache = (now, total)
    return total


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _format_bytes(total: int) -> str:
    """A size a reader can take in at a glance: three significant figures at most."""
    size = float(total)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.0f} {unit}" if size >= 100 else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.0f} TB" if size >= 100 else f"{size:.1f} TB"


def _landing_catalogue(templates: list[dict[str, Any]], workflows: list[Any]) -> dict[str, Any]:
    """Split templates between Data sources and Workflows by whether they can be ingested.

    A template is fetched or produced, never both (registration refuses `produced_by` beside
    `ingestion.plugin`), so each appears in exactly one area. A non-ingestable template whose
    `produced_by` names no known workflow — or that declares none — is still listed, under
    Workflows as an output of an unknown workflow, rather than silently dropped.
    """
    sources = sorted(
        (_source_view(t) for t in _ingestable_templates(templates)),
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


def render_workflows_page(mount: str) -> str:
    """Render the list of workflows, with the datasets each one produces.

    HTML only: the machine-readable list stays at `GET /process_graphs`.
    """
    catalogue = _landing_catalogue(_load_templates(), _load_workflows())
    return get_template("workflows_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        list_script=_read_asset("ocs_list.js"),
        nav=page_nav(mount, "workflows"),
        workflows=catalogue["workflows"],
        unattributed_outputs=catalogue["unattributed_outputs"],
    )


def render_processes_page(mount: str) -> str:
    """Render the process catalogue this instance loaded, tagged by origin.

    `GET /processes` answers the openEO JSON as it always has; only a browser gets this.
    """
    return get_template("processes_page.html").render(
        version=app_version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        list_script=_read_asset("ocs_list.js"),
        nav=page_nav(mount, "processes"),
        processes=_load_processes(),
        process_origins=_PROCESS_ORIGINS,
    )


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


def _ingestable_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The templates in *templates* that can be ingested from a source.

    Templates without an ingestion plugin — typically workflow outputs published via
    ``save_result`` — have no upstream fetch path, so they are not data sources. Shares the
    registry's predicate with ``GET /dataset-templates/`` and with the ingest path that refuses
    them, so every surface agrees about what is offerable.

    Takes the list rather than loading it, because each caller already has one and a second
    load would be a second answer to the same question.
    """
    return [t for t in templates if registry_datasets.is_ingestable(t)]


def _load_datasets() -> list[Any]:
    try:
        return list_datasets().items
    except Exception:
        _log.exception("Unexpected error loading datasets")
        return []


def render_landing(version: str, mount: str) -> str:
    """Render the root overview: what this instance holds, and how much of it.

    The lists it used to carry are pages of their own now, so this counts them and links to
    them rather than repeating them. It still loads each collection, because a count is what
    the overview is for.
    """
    extent = _load_extent()
    templates = _load_templates()
    catalogue = _landing_catalogue(templates, _load_workflows())
    return get_template("landing_page.html").render(
        version=version,
        mount=mount,
        name=api_config.get_name(),
        logo=LOGO,
        styles=_read_asset("ocs_ui.css"),
        nav=page_nav(mount, "overview"),
        extent=extent,
        globe=_extent_globe(extent),
        datasets=_load_datasets(),
        stored_size=_format_bytes(_stored_bytes()),
        sources=catalogue["sources"],
        workflows=catalogue["workflows"],
        # Shown on the overview, so a visitor knows why no page offers ingest or sync.
        read_only=api_config.is_read_only(),
    )
