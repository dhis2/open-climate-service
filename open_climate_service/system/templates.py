"""Server-side HTML rendering and root resource representations for the Open Climate Service."""

import functools
import importlib.resources
import json
import logging
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
        styles=_read_asset("ocs_ui.css"),
        extent=_load_extent(),
        datasets=_dataset_views(datasets, templates),
        published_count=sum(1 for dataset in datasets if dataset.publication.status == "published"),
        sources=catalogue["sources"],
        workflows=catalogue["workflows"],
        unattributed_outputs=catalogue["unattributed_outputs"],
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
        styles=_read_asset("ocs_ui.css"),
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
        styles=_read_asset("ocs_ui.css"),
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


def _inline_code(text: str) -> Markup:
    """Escape *text*, rendering `backticked` spans as code, the way the descriptions are written."""
    parts = str(text).split("`")
    return Markup("").join(
        Markup("<code>{}</code>").format(part) if index % 2 else escape(part) for index, part in enumerate(parts)
    )


def _description_blocks(text: str | None) -> list[dict[str, Any]]:
    """Paragraphs of a workflow description; a block that is a JSON example becomes code."""
    blocks = []
    for block in (text or "").replace("\r\n", "\n").split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped[0] in "{[":
            blocks.append({"code": stripped})
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
    parameters = [
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
        styles=_read_asset("ocs_ui.css"),
        **_workflow_page_context(record, _load_templates(), _load_datasets(), _load_triggers()),
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
