"""The landing page's areas, and the split of dataset templates between them (CLIM-940).

A template is shown as a data source when it can be ingested, and as a workflow output when a
workflow produces it. The split is decided by `_landing_catalogue`, so the unit tests below
drive it directly; the rendering tests go through `GET /`, where the page is actually served.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.openeo.workflows import _load_builtin_workflows
from open_climate_service.system import templates as landing

AREAS = ["overview", "explore", "datasets", "data-sources", "workflows", "operator"]


def _template(template_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": template_id, "name": template_id.replace("_", " "), "sync": {"kind": "static"}, **fields}


def _ingestable(template_id: str, **fields: Any) -> dict[str, Any]:
    return _template(template_id, ingestion={"plugin": "some.Plugin"}, **fields)


def _workflow(workflow_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=workflow_id, summary=f"{workflow_id} summary")


class _VisibleText(HTMLParser):
    """Collects the text a reader sees: no markup, scripts, styles or comments."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return " ".join(parser.parts)


def _area_ids(html: str) -> list[str]:
    return re.findall(r'data-area-link="([^"]+)"', html)


# --- the split ---------------------------------------------------------------------------


def test_ingestable_templates_are_data_sources_and_nothing_else() -> None:
    catalogue = landing._landing_catalogue(
        [_ingestable("chirps"), _template("normal", produced_by="climate_normal")],
        [_workflow("climate_normal")],
    )

    assert [source["id"] for source in catalogue["sources"]] == ["chirps"]
    assert [output["id"] for output in catalogue["workflows"][0]["outputs"]] == ["normal"]
    assert catalogue["unattributed_outputs"] == []


def test_every_template_appears_in_exactly_one_place() -> None:
    templates = [
        _ingestable("a"),
        _template("b", produced_by="w1"),
        _template("c", produced_by="w2"),
        _template("d", produced_by="unregistered"),
        _template("e"),
    ]
    catalogue = landing._landing_catalogue(templates, [_workflow("w1"), _workflow("w2")])

    placed = [s["id"] for s in catalogue["sources"]]
    placed += [o["id"] for w in catalogue["workflows"] for o in w["outputs"]]
    placed += [o["id"] for o in catalogue["unattributed_outputs"]]
    assert sorted(placed) == ["a", "b", "c", "d", "e"]


def test_an_output_of_an_unknown_workflow_is_listed_not_dropped() -> None:
    """A workflow can be registered at runtime, so the template may name one not yet known."""
    catalogue = landing._landing_catalogue(
        [_template("orphan", produced_by="not_registered"), _template("undeclared")],
        [],
    )

    assert {o["id"] for o in catalogue["unattributed_outputs"]} == {"orphan", "undeclared"}


def test_data_sources_are_ordered_by_provider_then_name() -> None:
    catalogue = landing._landing_catalogue(
        [
            _ingestable("z", name="Zeta", source="B provider"),
            _ingestable("y", name="Beta", source="A provider"),
            _ingestable("x", name="Alpha", source="B provider"),
        ],
        [],
    )

    assert [s["id"] for s in catalogue["sources"]] == ["y", "x", "z"]


@pytest.mark.parametrize(
    ("licence", "label"),
    [
        ("CC-BY-4.0", "CC-BY-4.0"),
        ({"name": "Copernicus licence", "url": "https://example.org"}, "Copernicus licence"),
        (None, None),
    ],
)
def test_licence_label(licence: object, label: str | None) -> None:
    template = _ingestable("t") if licence is None else _ingestable("t", license=licence)
    assert landing._landing_catalogue([template], [])["sources"][0]["licence"] == label


# --- produced_by -------------------------------------------------------------------------


def test_every_built_in_derived_template_names_a_built_in_workflow() -> None:
    """The shipped catalogue must not rely on the unknown-workflow fallback."""
    workflow_ids = {workflow["id"] for workflow in _load_builtin_workflows()}
    derived = [t for t in registry_datasets.list_datasets() if not registry_datasets.is_ingestable(t)]

    assert derived, "the built-in catalogue has derived templates"
    unlinked = {t["id"]: t.get("produced_by") for t in derived if t.get("produced_by") not in workflow_ids}
    assert unlinked == {}


@pytest.mark.parametrize("value", ["", "  ", " climate_normal", 3, ["climate_normal"]])
def test_a_malformed_produced_by_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match="invalid produced_by"):
        registry_datasets._validate_dataset_template(
            _template("derived", produced_by=value, license="CC-BY-4.0"), source="derived.yaml"
        )


def test_produced_by_beside_an_ingestion_plugin_is_refused() -> None:
    template = _template(
        "both",
        produced_by="climate_normal",
        ingestion={"plugin": "some.Plugin"},
        license="CC-BY-4.0",
    )

    with pytest.raises(ValueError, match="either ingested or produced"):
        registry_datasets._validate_dataset_template(template, source="both.yaml")


def test_produced_by_is_listed_by_the_template_api(client: TestClient) -> None:
    templates = {t["id"]: t for t in client.get("/dataset-templates/").json()}

    assert templates["worldpop_population_change"]["produced_by"] == "temporal_change"
    assert "produced_by" not in templates["worldpop_population_global2_100m"]


# --- the rendered page -------------------------------------------------------------------


def test_the_page_offers_every_area_in_order(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text

    assert _area_ids(html) == AREAS
    for area in AREAS:
        assert f'id="{area}" data-area' in html


def test_read_only_drops_the_operator_area(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    html = landing.render_landing("0.0.0", "")

    assert _area_ids(html) == AREAS[:-1]
    assert 'id="operator"' not in html
    assert "read-only" in _visible_text(html)


def test_the_word_template_is_not_shown_to_readers(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text

    assert "template" not in _visible_text(html).lower()


def test_lists_carry_the_filter_and_pager_hooks(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    sources = [t for t in registry_datasets.list_datasets() if registry_datasets.is_ingestable(t)]

    assert re.search(r'id="data-sources".*?data-list data-page-size="\d+"', html, re.S)
    assert html.count("data-filter-text") >= 1
    assert html.count("data-pager") >= 1
    # One filterable card per ingestable template, and no card for a workflow output.
    section = html.split('id="data-sources"', 1)[1].split('id="workflows"', 1)[0]
    assert section.count("data-item") == len(sources)
    # Same tiles/list layout as datasets, without the preview: a data source holds no data yet.
    assert 'data-views="sources"' in section
    assert 'data-view-button="list"' in section
    assert 'class="collection view-tiles" data-view-target' in section
    assert 'class="thumb' not in section


def test_workflow_outputs_are_listed_under_their_workflow(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    section = html.split('id="workflows"', 1)[1].split('id="operator"', 1)[0]
    temporal_change = section.split('/process_graphs/temporal_change"', 1)[1].split("</article>", 1)[0]

    assert "Population change (WorldPop Global2)" in temporal_change


def test_colours_come_from_the_token_block(client: TestClient) -> None:
    """Hex literals may appear only where the DHIS2 tokens are declared."""
    html = client.get("/", headers={"Accept": "text/html"}).text
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    rules = style.split("--header-height", 1)[1]

    assert "@dhis2/ui-constants 10.17.0" in style
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b", rules) == []


def test_the_json_representation_is_unchanged(client: TestClient) -> None:
    response = client.get("/", params={"f": "json"})

    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert {"api_version", "backend_version", "endpoints", "links"} <= body.keys()


# --- datasets: tiles, list and the dataset page ------------------------------------------


def _record(dataset_id: str = "chirps_monthly", **fields: Any) -> Any:
    from open_climate_service.ingestions.schemas import DatasetDetailRecord

    values: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_dataset_id": dataset_id,
        "dataset_name": "Precipitation (CHIRPS, monthly)",
        "description": "First line\nwraps here.\n\nSecond paragraph.",
        "variable": "precip",
        "period_type": "monthly",
        "units": "mm/d",
        "source": "CHIRPS v3",
        "source_url": "https://example.org/chirps",
        "extent": {
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": "2020-01", "end": "2026-07"},
        },
        "last_updated": "2026-09-01T07:58:03Z",
        "links": [
            {"href": f"/datasets/{dataset_id}", "rel": "self", "title": "Dataset detail"},
            {"href": f"/zarr/{dataset_id}", "rel": "zarr", "title": "Zarr store"},
        ],
        "publication": {"status": "published", "published_at": "2026-09-01T07:58:03Z"},
        "versions": [],
        **fields,
    }
    return DatasetDetailRecord.model_validate(values)


BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


@pytest.mark.parametrize(
    ("accept", "query", "html"),
    [
        (BROWSER_ACCEPT, "", True),
        ("*/*", "", False),
        ("", "", False),
        ("application/json", "", False),
        (BROWSER_ACCEPT, "?f=json", False),
        ("*/*", "?f=html", True),
    ],
)
def test_the_dataset_endpoint_serves_a_page_only_to_clients_that_ask_for_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, accept: str, query: str, html: bool
) -> None:
    """Scripts that send `*/*` or no Accept header keep the JSON they always got."""
    from open_climate_service.ingestions import services

    monkeypatch.setattr(services, "get_dataset_or_404", lambda dataset_id: _record(dataset_id))
    headers = {"Accept": accept} if accept else {}

    response = client.get(f"/datasets/chirps_monthly{query}", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html" if html else "application/json")
    if not html:
        assert response.json()["dataset_id"] == "chirps_monthly"


def test_a_dataset_without_a_thumbnail_gets_the_colormap_ramp() -> None:
    view = landing._dataset_view(_record("never_rendered"), {"display": {"colormap": "blues"}})

    assert view["has_thumbnail"] is False
    assert view["ramp"].startswith("linear-gradient(90deg, #")
    assert view["description"] == "First line wraps here. Second paragraph."
    assert view["coverage"] == "2020-01 – 2026-07"


def test_a_dataset_with_a_thumbnail_links_it() -> None:
    from open_climate_service.shared.thumbnails import thumbnail_path

    path = thumbnail_path("rendered_dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    try:
        assert landing._dataset_view(_record("rendered_dataset"), None)["has_thumbnail"] is True
    finally:
        path.unlink()


def test_the_dataset_page_names_the_workflow_that_produced_it() -> None:
    template = _template("normal", produced_by="climate_normal", sync={"kind": "static"})
    context = landing._dataset_page_context(_record("normal"), template)

    assert ("Produced by", "climate_normal workflow", "/process_graphs/climate_normal") in context["about"]


def test_the_dataset_page_lists_only_what_is_known() -> None:
    context = landing._dataset_page_context(_record(short_name=None, resolution=None), None)
    labels = [label for label, _, _ in context["about"] + context["data"] + context["status_facts"]]

    assert "Short name" not in labels
    assert "Resolution" not in labels
    assert {"Identifier", "Variable", "Bounding box", "Publication"} <= set(labels)
    assert ("Licence", "Not specified", None) in context["about"]
    assert context["paragraphs"] == ["First line wraps here.", "Second paragraph."]
    assert [link.rel for link in context["links"]] == ["zarr"]


def test_the_dataset_page_renders_under_the_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)

    html = landing.render_dataset_page(_record(), "/ocs")

    assert "<p>Second paragraph.</p>" in html
    assert 'href="/ocs/zarr/chirps_monthly"' in html
    assert 'href="/ocs/datasets/chirps_monthly?f=json"' in html
    assert 'href="/ocs/#datasets" aria-current="page"' in html
    assert "template" not in _visible_text(html).lower()


def test_the_datasets_area_offers_tiles_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(landing, "_load_datasets", lambda: [_record()])

    html = landing.render_landing("0.0.0", "")
    section = html.split('id="datasets"', 1)[1].split('id="data-sources"', 1)[0]

    assert 'data-view-button="tiles" aria-pressed="true"' in section
    assert 'data-view-button="list"' in section
    assert 'class="collection view-tiles" data-view-target' in section
    assert 'href="/datasets/chirps_monthly"' in section
    assert "First line wraps here." in section


# --- the data source page ----------------------------------------------------------------

TODAY = __import__("datetime").date(2026, 9, 17)


def _source_context(template: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {"read_only": False, "has_extent": True, "today": TODAY, **overrides}
    datasets = options.pop("datasets", [])
    return landing._data_source_page_context(template, datasets, **options)


@pytest.mark.parametrize(
    ("direction", "extents", "expected"),
    [
        (None, {}, {"start": "2025-09-17", "end": "2026-09-17", "start_required": True}),
        ("future", {}, {"start": "", "end": "", "start_required": False}),
        ("spanning", {"temporal": {"end": "2030"}}, {"start": "2025-09-17", "end": "2030", "start_required": True}),
    ],
)
def test_the_ingest_form_is_prefilled_for_the_direction_of_the_source(
    direction: str | None, extents: dict[str, Any], expected: dict[str, Any]
) -> None:
    template = _ingestable("src", extents=extents, **({"temporal_direction": direction} if direction else {}))

    defaults = _source_context(template)["defaults"]

    assert {key: defaults[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("template", "overrides", "can_ingest"),
    [
        (_ingestable("src"), {}, True),
        (_ingestable("src"), {"read_only": True}, False),
        (_ingestable("src"), {"has_extent": False}, False),
        (_template("normal", produced_by="climate_normal"), {}, False),
    ],
)
def test_the_form_is_offered_only_where_ingesting_can_work(
    template: dict[str, Any], overrides: dict[str, Any], can_ingest: bool
) -> None:
    assert _source_context(template, **overrides)["can_ingest"] is can_ingest


def test_an_ingested_source_links_to_its_dataset() -> None:
    context = _source_context(_ingestable("chirps_monthly"), datasets=[_record("chirps_monthly")])

    assert context["ingested"] == {"coverage": "2020-01 – 2026-07", "status": "published"}


def test_the_data_source_page_is_served_with_the_form(client: TestClient) -> None:
    response = client.get("/data-sources/chirps3_precipitation_daily")

    assert response.status_code == 200
    assert 'id="ingest-form"' in response.text
    assert 'action="/manage/ingest"' in response.text
    assert '<input type="hidden" name="dataset_id" value="chirps3_precipitation_daily" />' in response.text
    assert "template" not in _visible_text(response.text).lower()


def test_an_unknown_data_source_is_a_404(client: TestClient) -> None:
    assert client.get("/data-sources/does_not_exist").status_code == 404


def test_read_only_pages_offer_no_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)
    template = registry_datasets.get_dataset("chirps3_precipitation_daily")
    assert template is not None

    html = landing.render_data_source_page(template, "")

    assert "/manage" not in html
    assert "read-only" in _visible_text(html)


def test_data_sources_link_to_their_page(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    section = html.split('id="data-sources"', 1)[1].split('id="workflows"', 1)[0]

    assert 'href="/data-sources/chirps3_precipitation_daily"' in section
