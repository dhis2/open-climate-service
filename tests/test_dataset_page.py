"""The dataset page, and how `/datasets/{id}` chooses between it and JSON (CLIM-940).

The endpoint has always answered JSON, so it still does unless a client ranks `text/html`
higher — a browser following the landing page's link. `?f=html` and `?f=json` say so outright.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.system import templates as landing


def _template(template_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": template_id, "name": template_id.replace("_", " "), "sync": {"kind": "static"}, **fields}


def _ingestable(template_id: str, **fields: Any) -> dict[str, Any]:
    return _template(template_id, ingestion={"plugin": "some.Plugin"}, **fields)


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
        # Required since GeoParquet feature collections landed: a listing has to tell a
        # raster from a feature without a request per row.
        "itemType": "coverage",
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


def test_both_representations_declare_that_they_vary_by_accept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One URL, two representations: a cache keyed on the URL alone would cross them over.

    Without `Vary: Accept` a shared proxy can hand an API client the HTML page it cached for a
    browser, which is exactly the JSON compatibility this endpoint promises to keep.
    """
    from open_climate_service.ingestions import services

    monkeypatch.setattr(services, "get_dataset_or_404", lambda dataset_id: _record(dataset_id))

    for accept in (BROWSER_ACCEPT, "application/json"):
        response = client.get("/datasets/chirps_monthly", headers={"Accept": accept})

        assert response.status_code == 200
        assert response.headers["vary"] == "Accept", accept


def test_the_page_drops_the_version_table_but_the_json_keeps_the_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact records for one dataset are ingest events, not versions anyone can fetch.

    They all name the same Icechunk store, and a sync appends to it, so a row reading
    "2025-01 – 2025-04" describes a state the store no longer holds and nothing can serve. On
    all but the most-synced dataset the table was also a single row. The field stays in the
    JSON, which is an API contract; only the table is gone.
    """
    from open_climate_service.ingestions import services

    monkeypatch.setattr(services, "get_dataset_or_404", lambda dataset_id: _record(dataset_id))

    page = client.get("/datasets/chirps_monthly", headers={"Accept": BROWSER_ACCEPT}).text
    assert "versions-title" not in page
    assert "Versions" not in _visible_text(page)

    assert "versions" in client.get("/datasets/chirps_monthly").json()


@pytest.mark.parametrize(
    ("accept", "html"),
    [(BROWSER_ACCEPT, True), ("*/*", False), ("", False), ("application/json", False)],
)
def test_the_dataset_list_serves_a_page_only_to_clients_that_ask_for_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, accept: str, html: bool
) -> None:
    """The area that lived at `/#datasets`, as its own URL — linkable and JavaScript-free.

    `GET /datasets` has always answered JSON, so it still does for anything that does not ask
    for a page.
    """
    # Stubbed rather than read from the instance: with nothing ingested the page renders its
    # empty branch, so the assertions below would depend on whoever ran the suite.
    monkeypatch.setattr(landing, "_load_datasets", lambda: [_record()])
    headers = {"Accept": accept} if accept else {}

    response = client.get("/datasets", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html" if html else "application/json")
    assert response.headers["vary"] == "Accept"
    if html:
        assert 'data-list data-page-size="12" data-views="datasets"' in response.text
        # Served from the shared asset, so the list behaves the same here as on the landing page.
        assert "initList" in response.text
    else:
        assert "items" in response.json()


@pytest.mark.parametrize(
    ("item_types", "filter_shown"), [(["coverage", "feature"], True), (["coverage", "coverage"], False)]
)
def test_the_dataset_list_filters_by_type_when_it_holds_both(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, item_types: list[str], filter_shown: bool
) -> None:
    records = [_record(f"ds_{i}", itemType=item_type) for i, item_type in enumerate(item_types)]
    monkeypatch.setattr(landing, "_load_datasets", lambda: records)

    body = client.get("/datasets", headers={"Accept": BROWSER_ACCEPT}).text

    assert ('data-filter-field="kind"' in body) is filter_shown
    assert body.count('data-kind="raster"') == item_types.count("coverage")
    assert body.count('data-kind="vector"') == item_types.count("feature")


def test_the_breadcrumb_returns_to_the_dataset_list(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The list is a page, so the breadcrumb goes to it rather than a landing-page fragment."""
    from open_climate_service.ingestions import services

    monkeypatch.setattr(services, "get_dataset_or_404", lambda dataset_id: _record(dataset_id))

    body = client.get("/datasets/chirps_monthly", headers={"Accept": BROWSER_ACCEPT}).text

    assert '<a href="/datasets">Datasets</a>' in body
    assert "/#datasets" not in body


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
    """The workflow's page, not its process graph.

    The link is named after the workflow, so it opened JSON at a name a reader expects to be
    readable. The process graph stays a click away, labelled as JSON on that page.
    """
    template = _template("normal", produced_by="climate_normal", sync={"kind": "static"})
    context = landing._dataset_page_context(_record("normal"), template)

    assert ("Produced by", "climate_normal workflow", "/workflows/climate_normal") in context["data"]


def test_the_dataset_page_names_the_dataset_template_it_was_fetched_from() -> None:
    context = landing._dataset_page_context(_record("chirps_monthly"), _ingestable("chirps_monthly"))

    assert ("Origin", "Fetched from the data source", "/data-sources/chirps_monthly") in context["data"]


def test_the_dataset_page_lists_only_what_is_known() -> None:
    context = landing._dataset_page_context(_record(short_name=None, resolution=None), None)
    labels = [label for label, _, _ in context["data"] + context["status_facts"]]

    assert "Short name" not in labels
    assert "Resolution" not in labels
    assert {"Identifier", "Variable", "Bounding box", "Publication"} <= set(labels)
    assert ("Licence", "Not specified", None) in context["data"]
    assert context["paragraphs"] == ["First line wraps here.", "Second paragraph."]
    assert [link.rel for link in context["links"]] == ["zarr"]


def test_the_dataset_page_lists_the_identifier_first() -> None:
    context = landing._dataset_page_context(_record("chirps_monthly"), _ingestable("chirps_monthly"))

    assert context["data"][0] == ("Identifier", "chirps_monthly", None)


def test_the_dataset_page_shows_temporal_coverage_under_the_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)

    html = landing.render_dataset_page(_record(), "/ocs")
    title_end = html.index('<div class="dataset-page">')

    assert '<span class="range">2020-01 – 2026-07</span>' in html[:title_end]
    assert '<span class="period">Monthly</span>' in html[:title_end]
    # Moved, not copied: About no longer repeats them.
    about = _visible_text(html[title_end:])
    assert "Temporal coverage" not in about and "Period type" not in about


def test_a_climatology_keeps_its_period_type_in_about() -> None:
    record = _record(
        period_type="climatology",
        extent={
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": None, "end": None},
        },
    )

    assert ("Period type", "climatology", None) in landing._dataset_page_context(record, None)["data"]


def test_a_dataset_without_temporal_coverage_shows_no_coverage_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)
    record = _record(
        itemType="feature",
        period_type=None,
        extent={
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": None, "end": None},
        },
    )

    assert 'class="page-coverage"' not in landing.render_dataset_page(record, "/ocs")


@pytest.mark.parametrize(
    ("period_type", "start", "end", "label"),
    [
        ("monthly", "2026-01-01", "2026-08-01", "2026-01 – 2026-08"),
        ("monthly", "1990-01", "2026-08", "1990-01 – 2026-08"),
        ("yearly", "2015-01-01", "2030-01-01", "2015 – 2030"),
        ("weekly", "2026-09-14", "2026-09-21", "2026-W38 – 2026-W39"),
        # ISO week-year: 2024-12-30 is in week 1 of 2025.
        ("weekly", "2024-12-23", "2024-12-30", "2024-W52 – 2025-W01"),
        ("climatology", "2026-01-01", "2026-12-01", "2026-01-01 – 2026-12-01"),
        ("daily", "2026-01-01", "2026-08-31", "2026-01-01 – 2026-08-31"),
        ("dekadal", "2026-01-01", None, "2026-01-01 – …"),
    ],
)
def test_coverage_reads_at_the_dataset_period(period_type: str, start: str, end: str | None, label: str) -> None:
    """A monthly store a workflow wrote keeps full dates; the label reads like an ingested one."""
    record = _record(
        period_type=period_type,
        extent={
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": start, "end": end},
        },
    )

    assert landing._dataset_view(record, None)["coverage"] == label


_NOT_FETCHED = "not fetched by a provider"
_OWNED_ELSEWHERE = "fetched by provider 'overture', but its data source now names 'fake'"
_NO_PROVIDER = "no provider on this instance fetches it"


@pytest.mark.parametrize(
    ("provider", "owner", "raster_collision", "linked", "blocked"),
    [
        ("fake", "fake", False, True, None),  # the provider that fetched it, and it is installed
        (None, None, False, True, _NO_PROVIDER),  # providerless template
        ("missing", "missing", False, True, _NO_PROVIDER),  # names a provider this instance lacks
        ("fake", "fake", True, False, _NO_PROVIDER),  # the id is a raster data source's
        ("fake", None, False, True, _NOT_FETCHED),  # stored without a provider: none may overwrite it
        ("fake", "overture", False, True, _OWNED_ELSEWHERE),  # template now names another provider
    ],
)
def test_a_vector_dataset_offers_refresh_only_where_the_refresh_would_run(
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    owner: str | None,
    raster_collision: bool,
    linked: bool,
    blocked: str | None,
) -> None:
    """The page agrees with the refresh: an installed provider, and the one that wrote the stored
    collection, since a refresh may only overwrite what its own provider fetched."""
    from types import SimpleNamespace

    from open_climate_service.features import providers as feature_providers
    from open_climate_service.features import services as feature_services
    from open_climate_service.features import templates as feature_templates

    template = {"id": "districts", "name": "Districts", "license": "ODbL-1.0"}
    if provider:
        template["provider"] = provider
    monkeypatch.setattr(feature_templates, "get_feature_template", lambda template_id: dict(template))
    monkeypatch.setattr(feature_providers, "load_feature_providers", lambda: {"fake": object()})
    monkeypatch.setattr(
        feature_services,
        "registered_collections",
        lambda: {"districts": SimpleNamespace(features=SimpleNamespace(provider=owner))},
    )
    monkeypatch.setattr(
        registry_datasets, "get_dataset", lambda dataset_id: {"id": dataset_id} if raster_collision else None
    )

    html = landing.render_dataset_page(_record("districts", itemType="feature", period_type=None), "/ocs")
    text = _visible_text(html)

    assert ('href="/ocs/data-sources/districts"' in html) is linked
    # Never claims how the collection was made: a template is no evidence it was fetched.
    assert "Fetched from" not in text
    assert ("fetches it again from the provider" in text) is (blocked is None)
    if blocked is not None:
        assert "cannot be refreshed here" in text and blocked in text
    assert "no source to sync from" not in text
    assert "Refresh" in text and "Status and display" not in text
    assert 'id="sync-form"' not in html


@pytest.mark.parametrize(("item_type", "raster"), [("coverage", True), ("feature", False)])
def test_the_dataset_page_offers_the_map_viewer_and_colour_scale_only_for_a_raster(
    monkeypatch: pytest.MonkeyPatch, item_type: str, raster: bool
) -> None:
    """The map viewer draws only rasters, and a feature collection has no colour scale."""
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)

    html = landing.render_dataset_page(_record(itemType=item_type), "/ocs")

    assert ("Open in map viewer" in html) is raster
    assert ('class="ramp-bar"' in html) is raster
    assert ("--ramp: linear-gradient" in html) is raster


def test_the_dataset_page_renders_under_the_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: None)

    html = landing.render_dataset_page(_record(), "/ocs")

    assert "<p>Second paragraph.</p>" in html
    assert 'href="/ocs/zarr/chirps_monthly"' in html
    assert 'href="/ocs/datasets/chirps_monthly?f=json"' in html
    assert 'href="/ocs/"' in html, "the header links home under the mount"
    # The rail carries the mount too. It names no Datasets entry yet: that entry is an area of
    # the landing page, which arrives in a later slice, and a link to a 404 is worse than none.
    assert 'href="/ocs/map"' in html
    assert "#operator" not in html
    # "Data sources" is the rail entry; the dataset page itself names no template.
    assert "template" not in _visible_text(html).lower().replace("dataset templates", "")


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


def test_the_dataset_page_opens_the_map_on_that_dataset() -> None:
    html = landing.render_dataset_page(_record("chirps monthly"), "/ocs")

    assert 'href="/ocs/map?dataset=chirps%20monthly"' in html


def test_a_sync_leaves_an_unpublished_dataset_unpublished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry_datasets, "get_dataset", lambda dataset_id: _ingestable("d", sync={"kind": "temporal"})
    )
    record = _record()
    record.publication.status = "unpublished"

    html = landing.render_dataset_page(record, "")

    assert 'name="publish"' not in html
    assert "The dataset stays unpublished." in _visible_text(html)
