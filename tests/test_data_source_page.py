"""The data source page, and the ingest form it offers (CLIM-940).

A data source is a dataset template the instance can fetch from outside. The page is HTML
only — the machine-readable form of the same thing stays at `GET /dataset-templates/{id}`.
"""

from __future__ import annotations

from datetime import date
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
        "publication": {"status": "published", "published_at": "2026-09-01T07:58:03Z"},
        "versions": [],
        **fields,
    }
    return DatasetDetailRecord.model_validate(values)


TODAY = date(2026, 9, 17)


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


def test_the_ingest_form_prefills_dates_in_the_format_it_states(client: TestClient) -> None:
    """The field takes a period identifier, so a monthly source must not be prefilled with a day."""
    body = client.get("/data-sources/era5land_temperature_monthly").text
    form = body.split('id="ingest-form"', 1)[1]

    assert 'value="{}"'.format(date.today().replace(year=date.today().year - 1).strftime("%Y-%m")) in form
    assert "Format YYYY-MM." in form


@pytest.mark.parametrize(
    ("today", "expected_start"),
    [
        (date(2026, 3, 31), "2025-03-31"),
        (date(2026, 12, 31), "2025-12-31"),
        (date(2026, 1, 29), "2025-01-29"),
        # The one date the previous year does not have.
        (date(2024, 2, 29), "2023-02-28"),
    ],
)
def test_the_default_start_keeps_the_day_except_where_it_cannot(today: date, expected_start: str) -> None:
    """Clamping every date to the 28th moved the default back by up to three days a month.

    For a daily or dekadal source those are real extra periods to fetch, so only 29 February —
    which the previous year genuinely lacks — falls back.
    """
    assert _ingest_defaults_start(today) == expected_start


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


def test_one_facts_panel_holds_both_what_and_where(client: TestClient) -> None:
    """About folds into the facts list, as on the dataset page.

    Nothing covered the About facts before, so the merge could have dropped them silently.
    """
    context = _source_context(registry_datasets.get_dataset("chirps3_precipitation_daily") or {})
    labels = [label for label, _, _ in context["data"]]

    assert "about" not in context
    # What it measures, then where it comes from — in that order, in one list.
    assert {"Variable", "Period type"} <= set(labels)
    assert {"Identifier", "Provider", "Licence"} <= set(labels)
    assert labels.index("Variable") < labels.index("Identifier")

    body = client.get("/data-sources/chirps3_precipitation_daily").text
    assert body.count('id="about-title"') == 1
    assert 'id="data-title"' not in body


def test_the_data_source_list_is_its_own_page(client: TestClient) -> None:
    """The area that lived at `/#data-sources`, as a URL that can be linked to.

    HTML only, like a single data source: the machine-readable list of the same thing stays at
    `GET /dataset-templates/`, which this neither renames nor duplicates.
    """
    response = client.get("/data-sources")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'data-views="sources"' in response.text
    assert "initList" in response.text, "the shared list script, not a second copy"
    # Only what can be fetched: a workflow output is not a data source.
    assert 'href="/data-sources/chirps3_precipitation_daily"' in response.text
    assert "chirps3_precipitation_daily_normal_1991_2020" not in response.text

    assert client.get("/dataset-templates/").status_code == 200


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


def test_the_dataset_page_links_back_to_the_data_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pair is two-way: the source page links to its dataset, and the dataset back.

    Only for a source that is fetched — a workflow output points at the workflow instead, and
    a dataset whose template has gone names no origin at all.
    """
    monkeypatch.setattr(registry_datasets, "get_dataset", lambda dataset_id: _ingestable("chirps_monthly"))

    html = landing.render_dataset_page(_record("chirps_monthly"), "/ocs")

    assert 'href="/ocs/data-sources/chirps_monthly"' in html


def _ingest_defaults_start(today: date) -> str:
    return str(landing._ingest_defaults(_ingestable("chirps_daily"), today)["start"])
