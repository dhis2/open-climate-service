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
