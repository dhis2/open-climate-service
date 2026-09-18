"""The landing page's areas, and the split of dataset templates between them (CLIM-940).

A template is shown as a data source when it can be ingested, and as a workflow output when a
workflow produces it. The split is decided by `_landing_catalogue`, so the unit tests below
drive it directly; the rendering tests go through `GET /`, where the page is actually served.
"""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.system import templates as landing

AREAS = ["overview", "datasets", "data-sources", "workflows", "processes"]


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


# --- the rendered page -------------------------------------------------------------------


def test_the_page_offers_every_area_in_order(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text

    assert _area_ids(html) == AREAS
    for area in AREAS:
        assert f'id="{area}" data-area' in html


def test_read_only_keeps_the_areas_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    html = landing.render_landing("0.0.0", "")

    assert _area_ids(html) == AREAS
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


def test_workflow_tiles_link_to_the_page_that_lists_their_outputs(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    section = html.split('id="workflows"', 1)[1].split("</section>", 1)[0]
    tile = section.split('href="/workflows/temporal_change"', 1)[1].split("</article>", 1)[0]

    assert "Temporal change" in tile
    assert "Publishes a dataset" in tile
    # The list of what it produces is on the workflow page, not the tile.
    assert "Population change (WorldPop Global2)" not in section

    page = client.get("/workflows/temporal_change").text
    assert "Population change (WorldPop Global2)" in page.split('id="produces-title"', 1)[1]


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
    assert "#operator" not in html
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

    assert context["ingested"] == {
        "id": "chirps_monthly",
        "coverage": "2020-01 – 2026-07",
        "status": "published",
    }


def test_the_data_source_page_is_served_with_the_form(client: TestClient) -> None:
    response = client.get("/data-sources/chirps3_precipitation_daily")

    assert response.status_code == 200
    assert 'id="ingest-form"' in response.text
    assert 'action="/manage/ingest"' in response.text
    assert '<input type="hidden" name="dataset_id" value="chirps3_precipitation_daily" />' in response.text
    assert "template" not in _visible_text(response.text).lower()


def test_ingesting_a_workflow_output_is_refused_before_the_stream_opens(client: TestClient) -> None:
    """A stream would report this as an event on a 200; it is a client mistake, so it is a 400."""
    response = client.post(
        "/manage/ingest",
        data={"dataset_id": "chirps3_precipitation_daily_normal_1991_2020", "start": "2020-01-01"},
    )

    assert response.status_code == 400, response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert "workflow" in response.json()["error"].lower() or "ingest" in response.json()["error"].lower()


def test_syncing_an_unknown_dataset_is_refused_before_the_stream_opens(client: TestClient) -> None:
    response = client.post("/manage/sync", data={"dataset_id": "not_a_dataset"})

    assert response.status_code == 404, response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_the_ingest_form_survives_a_leap_day(client: TestClient) -> None:
    """The prefilled start is a year back, and 29 February has no anniversary."""
    defaults = landing._ingest_defaults({"period_type": "daily"}, date(2024, 2, 29))

    assert defaults["start"] == "2023-02-28"


def test_the_overview_counts_what_the_instance_holds(client: TestClient) -> None:
    """Published is not a headline figure — how much data is stored is."""
    section = client.get("/", headers={"Accept": "text/html"}).text.split('id="overview"', 1)[1]
    section = section.split("</section>", 1)[0]

    assert "Data stored" in section
    assert "Published" not in section


@pytest.mark.parametrize(
    ("total", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (150_000, "150 KB"), (2_400_000_000, "2.4 GB")],
)
def test_a_stored_size_reads_at_a_glance(total: int, expected: str) -> None:
    assert landing._format_bytes(total) == expected


def test_the_stored_size_sums_each_store_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two artifacts appended to one store must not be counted twice."""
    store = tmp_path / "chirps.icechunk"
    (store / "chunks").mkdir(parents=True)
    (store / "chunks" / "0").write_bytes(b"x" * 1000)

    monkeypatch.setattr(
        landing,
        "_stored_bytes_cache",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[SimpleNamespace(path=str(store)), SimpleNamespace(path=str(store))]),
    )

    assert landing._stored_bytes() == 1000
    landing._stored_bytes_cache = None


def test_the_ingest_form_prefills_dates_in_the_format_it_states(client: TestClient) -> None:
    """The field takes a period identifier, so a monthly source must not be prefilled with a day."""
    body = client.get("/data-sources/era5land_temperature_monthly").text
    form = body.split('id="ingest-form"', 1)[1]

    assert 'value="{}"'.format(date.today().replace(year=date.today().year - 1).strftime("%Y-%m")) in form
    assert "Format YYYY-MM." in form


@pytest.mark.parametrize(
    ("value", "period", "expected"),
    [
        ("2026-09-17", "yearly", "2026"),
        ("2026-09-17", "monthly", "2026-09"),
        ("2026-09-17", "daily", "2026-09-17"),
        ("2026-09-17", "hourly", "2026-09-17T00"),
        ("2026-09-17", "climatology", "2026-09-17"),
        ("", "monthly", ""),
        ("2030", "monthly", "2030"),
    ],
)
def test_a_prefilled_date_is_cut_to_the_period(value: str, period: str, expected: str) -> None:
    assert landing._period_value(value, period) == expected


def test_an_unknown_data_source_is_a_404(client: TestClient) -> None:
    assert client.get("/data-sources/does_not_exist").status_code == 404


def test_a_source_ingested_under_another_id_still_shows_as_ingested(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dataset a source produced need not carry the source's id.

    A managed dataset records the template it came from in `source_dataset_id` and is free to
    use a different `dataset_id`. Matching only on `dataset_id` left the page claiming the
    source had never been ingested, and offering a first ingest for data that already exists.
    """
    ingested = _record("chirps3_nepal", source_dataset_id="chirps3_precipitation_daily")
    monkeypatch.setattr(landing, "_load_datasets", lambda: [ingested])

    html = landing.render_data_source_page(_ingestable("chirps3_precipitation_daily"), "")

    assert "Already ingested" in html
    assert 'href="/datasets/chirps3_nepal"' in html


def test_a_release_version_without_a_value_is_left_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-populated version dict must not reach the reader as "chirps:None"."""
    monkeypatch.setattr(landing, "_load_datasets", list)
    template = _ingestable(
        "chirps3_precipitation_daily",
        sync={"kind": "release", "version": {"authority": "chirps"}},
    )

    text = _visible_text(landing.render_data_source_page(template, ""))

    assert "None" not in text
    assert "chirps:" not in text


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


def test_each_tile_is_one_link_to_its_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole tile opens the page through a stretched title link, so a tile holds no other link."""
    monkeypatch.setattr(landing, "_load_datasets", lambda: [_record()])
    html = landing.render_landing("0.0.0", "")

    for area, prefix in (
        ("datasets", "/datasets/"),
        ("data-sources", "/data-sources/"),
        ("workflows", "/workflows/"),
        ("processes", "/processes/"),
    ):
        section = html.split(f'id="{area}" data-area', 1)[1].split("</section>", 1)[0]
        items = re.split(r'class="panel item[^"]*"', section)[1:]
        assert items, area
        for item in items:
            item = item.split("</article>", 1)[0]
            links = re.findall(r'<a [^>]*href="([^"]+)"', item)
            assert len(links) == 1 and links[0].startswith(prefix), (area, links)
            assert 'class="item-link"' in item


# --- sync from the dataset page ----------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "read_only", "form"),
    [
        (_ingestable("chirps_monthly", sync={"kind": "temporal"}), False, True),
        (_ingestable("chirps_monthly", sync={"kind": "release"}), False, True),
        (_ingestable("chirps_monthly", sync={"kind": "temporal"}), True, False),
        (_template("chirps_monthly", sync={"kind": "static"}), False, False),
        (None, False, False),
    ],
)
def test_the_dataset_page_offers_sync_only_where_it_can_run(
    monkeypatch: pytest.MonkeyPatch, template: dict[str, Any] | None, read_only: bool, form: bool
) -> None:
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: template)
    monkeypatch.setattr(api_config, "is_read_only", lambda: read_only)

    html = landing.render_dataset_page(_record(), "/ocs")

    assert ('id="sync-form"' in html) is form
    if form:
        assert 'action="/ocs/manage/sync"' in html
        # No publish choice: a sync keeps the dataset's current publication state, because
        # nothing here can change it and an incremental sync writes into the published store
        # either way.
        assert 'name="publish" value="on"' in html, "a published dataset stays published"
        assert "Publish after sync" not in html
        assert 'data-plan-url="/ocs/sync/chirps_monthly/plan"' in html
        assert 'placeholder="YYYY-MM"' in html
        assert "function runJobForm" in html
    else:
        assert "/manage" not in html


# --- the workflow page -------------------------------------------------------------------


def _workflow_record(**fields: Any) -> Any:
    from open_climate_service.openeo.schemas import WorkflowRecord

    values: dict[str, Any] = {
        "id": "aggregate_to_chap_csv",
        "summary": "Aggregate and export",
        "description": 'Loads `dataset_id` and exports.\n\nUsage:\n\n{\n  "agg": {"process_id": "x"}\n}',
        "parameters": [
            {"name": "dataset_id", "schema": {"type": "string"}, "description": "The `dataset` to read."},
            {
                "name": "period",
                "optional": True,
                "default": "month",
                "schema": {"type": "string", "enum": ["month", "week"]},
            },
            {"name": "temporal_extent", "schema": {"type": "array", "subtype": "temporal-interval"}},
        ],
        "process_graph": {
            "load": {"process_id": "load_collection", "arguments": {}},
            "save": {"process_id": "save_result", "arguments": {"format": "CHAPCSV"}, "result": True},
        },
        **fields,
    }
    return WorkflowRecord.model_validate(values)


def test_the_workflow_page_describes_the_workflow() -> None:
    context = landing._workflow_page_context(_workflow_record(), [], [], [])

    assert context["workflow"]["title"] == "Aggregate to CHAP CSV"
    assert context["workflow"]["results"] == [("export", "Exports CHAP CSV")]
    assert str(context["blocks"][0]["html"]) == "Loads <code>dataset_id</code> and exports."
    assert context["blocks"][2]["code"].startswith("{")
    params = {p["name"]: p for p in context["parameters"]}
    assert params["dataset_id"]["required"] is True
    assert params["period"] == {**params["period"], "required": False, "type": "month | week", "default": '"month"'}
    assert params["temporal_extent"]["type"] == "temporal-interval"


def test_workflow_descriptions_are_escaped() -> None:
    context = landing._workflow_page_context(_workflow_record(description="<script>x</script> `<b>`"), [], [], [])

    assert str(context["blocks"][0]["html"]) == "&lt;script&gt;x&lt;/script&gt; <code>&lt;b&gt;</code>"


def test_the_workflow_page_lists_outputs_and_triggers() -> None:
    from open_climate_service.automation.config import WorkflowTrigger

    record = _workflow_record(id="climate_normal")
    templates = [
        _template("normal_b", name="B normal", produced_by="climate_normal"),
        _template("normal_a", name="A normal", produced_by="climate_normal"),
        _template("other", produced_by="temporal_change"),
    ]
    triggers = [
        WorkflowTrigger(id="refresh", on_update_of="chirps_monthly", workflow_id="climate_normal", arguments={"x": 1}),
        WorkflowTrigger(id="elsewhere", on_update_of="chirps_monthly", workflow_id="temporal_change"),
    ]

    context = landing._workflow_page_context(record, templates, [_record("normal_b")], triggers)

    assert [(o["id"], o["ingested"]) for o in context["outputs"]] == [("normal_a", False), ("normal_b", True)]
    assert [t["id"] for t in context["triggers"]] == ["refresh"]
    assert context["triggers"][0]["held"] is False


def test_the_workflow_page_is_served(client: TestClient) -> None:
    response = client.get("/workflows/climate_normal")

    assert response.status_code == 200
    assert "Climate normal" in response.text
    assert 'href="/process_graphs/climate_normal"' in response.text
    assert client.get("/workflows/does_not_exist").status_code == 404


# --- processes ---------------------------------------------------------------------------


def test_processes_are_tagged_with_their_origin() -> None:
    processes = {p["id"]: p for p in landing._load_processes()}

    assert processes["load_collection"]["origin"] == "core"
    assert processes["add"]["origin"] == "core"
    assert processes["spi"]["origin"] == "ocs"
    assert {"xclim", "earthkit"} <= {p["origin"] for p in processes.values()}
    # OCS first, openEO core last, so the default view leads with what is specific to OCS.
    origins = [p["origin"] for p in processes.values()]
    assert origins[0] == "ocs" and origins[-1] == "core"


def test_a_plugin_overriding_an_indicator_counts_as_ocs(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import earthkit_processes, plugin_processes, processes, xclim_processes

    def indicator() -> None: ...

    def override() -> None: ...

    monkeypatch.setattr(processes, "list_openeo_processes", lambda: [{"id": "tg_mean"}, {"id": "abs"}])
    monkeypatch.setattr(xclim_processes, "scan", lambda: [indicator])
    monkeypatch.setattr(earthkit_processes, "scan", lambda: [])
    monkeypatch.setattr(plugin_processes, "load_plugin_processes", lambda: [("tg_mean", override)])

    assert {p["id"]: p["origin"] for p in landing._load_processes()} == {"tg_mean": "ocs", "abs": "core"}


def test_the_processes_area_lists_every_process_by_default(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    section = html.split('id="processes" data-area', 1)[1].split("</section>", 1)[0]

    assert '<option value="" selected>All</option>' in section
    assert 'data-origin="core"' in section, "core processes are listed, not filtered out"
    assert '<option value="!core">' in section, "hiding the core processes is still offered"
    assert 'data-views="processes"' in section
    # A leading "!" in a filter value must be understood by the list script.
    assert 'value.charAt(0) === "!"' in html


# --- the process page --------------------------------------------------------------------


def test_process_descriptions_render_their_markdown_safely() -> None:
    blocks = landing._description_blocks(
        "Uses ``eq()`` and **bold**; see [docs](https://openeo.org) or [x](javascript:alert(1)).\n\n"
        "* First <item>\n* Second `code`\n\n"
        "```\nprint(1)\n```"
    )

    assert str(blocks[0]["html"]) == (
        'Uses <code>eq()</code> and <strong>bold</strong>; see <a href="https://openeo.org">docs</a> '
        "or [x](javascript:alert(1))."
    )
    assert [str(item) for item in blocks[1]["bullets"]] == ["First &lt;item&gt;", "Second <code>code</code>"]
    assert blocks[2] == {"code": "print(1)\n"}


def test_the_process_page_lists_the_workflows_that_use_it() -> None:
    process = {
        "id": "reduce_dimension",
        "summary": "Reduce",
        "categories": ["cubes"],
        "parameters": [{"name": "data", "schema": {"type": "object", "subtype": "datacube"}}],
        "returns": {"description": "A cube.", "schema": {"type": "object", "subtype": "datacube"}},
        "links": [{"href": "https://example.org", "title": "About"}, {"href": "javascript:x", "rel": "bad"}],
    }
    nested = _workflow_record(
        id="uses_it",
        process_graph={
            "r": {
                "process_id": "apply",
                "arguments": {"process": {"process_graph": {"x": {"process_id": "reduce_dimension"}}}},
            }
        },
    )
    unrelated = _workflow_record(id="does_not")

    context = landing._process_page_context(process, "openEO core", [nested, unrelated])

    assert [w["id"] for w in context["used_by"]] == ["uses_it"]
    assert context["returns"]["type"] == "datacube"
    assert context["parameters"][0]["type"] == "datacube"
    assert context["links"] == [{"href": "https://example.org", "title": "About"}]


@pytest.mark.parametrize(("accept", "html"), [(BROWSER_ACCEPT, True), ("*/*", False), ("application/json", False)])
def test_the_process_endpoint_serves_a_page_only_to_browsers(client: TestClient, accept: str, html: bool) -> None:
    response = client.get("/processes/load_collection", headers={"Accept": accept})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html" if html else "application/json")
    if html:
        assert 'href="/workflows/climate_normal"' in response.text
        assert 'href="/#processes" aria-current="page"' in response.text
    else:
        assert response.json()["id"] == "load_collection"


# --- the map viewer ----------------------------------------------------------------------


def test_every_page_links_to_the_map_viewer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        client.get("/", headers={"Accept": "text/html"}).text,
        client.get("/data-sources/chirps3_precipitation_daily").text,
        client.get("/workflows/climate_normal").text,
        client.get("/processes/load_collection", headers={"Accept": BROWSER_ACCEPT}).text,
    ]
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)
    pages.append(landing.render_dataset_page(_record(), ""))

    for html in pages:
        assert 'href="/map">Map viewer</a>' in html


def test_the_map_viewer_has_the_page_header_and_navigation(client: TestClient) -> None:
    html = client.get("/map").text

    assert '<body class="map-page">' in html
    assert '<a href="/map" aria-current="page">Map viewer</a>' in html
    assert 'href="/#datasets">Datasets</a>' in html
    assert "--colors-blue800" in html, "the shared design tokens are included"


def test_the_map_viewer_opens_the_dataset_named_in_the_address(client: TestClient) -> None:
    html = client.get("/map", params={"dataset": "chirps3_precipitation_daily"}).text

    assert 'new URLSearchParams(window.location.search).get("dataset")' in html
    assert 'url.searchParams.set("dataset", id)' in html
    assert "`/datasets/${encodeURIComponent(id)}`" in html


def test_the_dataset_page_opens_the_map_on_that_dataset() -> None:
    html = landing.render_dataset_page(_record("chirps monthly"), "/ocs")

    assert 'href="/ocs/map?dataset=chirps%20monthly"' in html


def test_page_nav_marks_only_the_current_page() -> None:
    nav = str(landing.page_nav("/ocs", "workflows"))

    assert nav.count('aria-current="page"') == 1
    assert '<a href="/ocs/#workflows" aria-current="page">Workflows</a>' in nav
    assert '<a href="/ocs/map">Map viewer</a>' in nav
    assert '<a href="/ocs/openeo" target="_blank" rel="noopener">openEO editor</a>' in nav
    assert nav.index("Map viewer") < nav.index("API") < nav.index("openEO editor")
    # Set apart from the pages of this instance, because it leaves for another site.
    assert '<li class="gap"><a href="/ocs/openeo"' in nav


# --- the API page ------------------------------------------------------------------------


def test_the_api_page_lists_the_instance_s_own_endpoints(client: TestClient) -> None:
    html = client.get("/api").text
    schema = client.get("/openapi.json").json()
    served = {(method.upper(), path) for path, ops in schema["paths"].items() for method in ops}

    for method, path in served:
        assert f"<code>{path}</code>" in html, path
        assert f"<code>{method}</code>" in html
    # Entry points for the catalogues and the docs, which have no OpenAPI path of their own.
    for href in ("/stac/catalog.json", "/docs", "/openapi.json", "/collections", "/?f=json"):
        assert f'href="{href}"' in html
    assert '<a href="/api" aria-current="page">API</a>' in html


def test_endpoint_rows_prefer_the_docstring_over_the_generated_summary(client: TestClient) -> None:
    html = client.get("/api").text

    assert "Read Index" not in html, "FastAPI's function-name summary is not what a reader wants"
    assert "Return openEO capabilities (JSON) or the landing page (HTML)." in html


def test_the_api_page_marks_what_read_only_closes(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    context = landing._api_page_context(schema, read_only=True)
    closed = {(e["method"], e["path"]) for group in context["groups"] for e in group["endpoints"] if e["closed"]}
    everything = {(e["method"], e["path"]) for group in context["groups"] for e in group["endpoints"]}

    assert ("POST", "/ingestions") in closed
    assert ("GET", "/jobs") in closed
    assert ("GET", "/datasets") in everything - closed
    assert "closed" in landing.render_api_page(schema, "")


def test_a_writable_instance_marks_nothing_closed(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    context = landing._api_page_context(schema, read_only=False)

    assert not [e for group in context["groups"] for e in group["endpoints"] if e["closed"]]


def test_a_sync_leaves_an_unpublished_dataset_unpublished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry_datasets, "get_dataset", lambda dataset_id: _ingestable("d", sync={"kind": "temporal"})
    )
    record = _record()
    record.publication.status = "unpublished"

    html = landing.render_dataset_page(record, "")

    assert 'name="publish"' not in html
    assert "The dataset stays unpublished." in _visible_text(html)


# --- the extent globe --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bbox", "zoom"),
    [
        ([-180, -90, 180, 90], 1.85),  # a global extent shows as much as the frame can hold
        ([-25, 34, 45, 72], 1.85),  # Europe is wide enough to sit at the same bound
        ([80.05, 26.35, 88.2, 30.45], 2.2),  # Nepal, at the upper bound
        ([-13.5, 6.9, -10.1, 10.0], 2.2),  # a small country keeps its continent around it
    ],
)
def test_the_globe_zooms_to_the_size_of_the_extent(bbox: list[float], zoom: float) -> None:
    width, height = abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])

    assert landing._globe_zoom(width, height, (bbox[1] + bbox[3]) / 2) == pytest.approx(zoom, abs=0.01)


def test_the_globe_is_centred_on_the_extent() -> None:
    """The extent's centre projects to the middle of the globe, wherever on Earth it is."""
    for bbox in ([-13.5, 6.9, -10.1, 10.0], [80.05, 26.35, 88.2, 30.45], [-70, -40, -60, -30]):
        lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        point = landing._project(lon0, lat0, lon0, lat0, landing._GLOBE_RADIUS)
        assert point == (landing._GLOBE_WIDTH / 2, landing._GLOBE_HEIGHT / 2)


def test_the_far_side_of_the_world_is_not_drawn() -> None:
    """Orthographic shows one hemisphere; the antipode must not fold onto the front."""
    assert landing._project(0.0, 0.0, 180.0, 0.0, 48) is None
    assert landing._project(1.0, 1.0, 0.0, 0.0, 48) is not None


def test_the_map_never_shows_the_sphere_s_edge() -> None:
    """Cropped to a rectangle rather than drawn as a ball: the sphere must overflow the corners."""
    half_diagonal = ((landing._GLOBE_WIDTH / 2) ** 2 + (landing._GLOBE_HEIGHT / 2) ** 2) ** 0.5

    assert landing._GLOBE_RADIUS * landing._MIN_ZOOM > half_diagonal


def test_the_globe_draws_land_and_the_extent() -> None:
    globe = landing._extent_globe({"bbox": [-13.5, 6.9, -10.1, 10.0]})

    assert globe is not None
    assert globe["land"].count("M") > 10, "the visible hemisphere's coastlines"
    assert globe["extent"].startswith("M") and globe["extent"].endswith("Z")
    # Every drawn point is inside the viewBox's circle-ish area, so nothing escapes the clip.
    assert "-" not in globe["extent"], "the extent stays on the front of the globe"


@pytest.mark.parametrize(
    "extent",
    [None, {}, {"bbox": "world"}, {"bbox": [1, 2, 3]}, {"bbox": [1, 2, 3, "x"]}, {"bbox": [10, 10, 5, 5]}],
)
def test_no_globe_without_a_usable_extent(extent: dict[str, Any] | None) -> None:
    assert landing._extent_globe(extent) is None


def test_the_overview_draws_the_globe_without_fetching_anything(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    globe = re.search(r'<svg[^>]*class="globe".*?</svg>', html, re.S)

    assert globe is not None
    svg = globe.group(0)
    assert 'class="land"' in svg and 'class="extent"' in svg
    assert "http" not in svg, "no tiles, no external image: the page works offline"
