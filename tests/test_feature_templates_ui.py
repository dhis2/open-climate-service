"""Feature collection templates on the Dataset templates page and API, and fetching them.

Driven through the HTTP surface a browser and a client use: the template listing and detail
routes, the list and template pages, `POST /features/{id}/refresh` and the page's
`/manage/features/refresh` stream. A fake provider stands in for Overture or DHIS2 so nothing
leaves the machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import providers as feature_providers
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions import services as ingestion_services

TEMPLATES = """
- id: demo_regions
  name: Demo regions
  description: Regions for the tests.
  license: ODbL-1.0
  attribution: © Demo contributors
  id_property: code
  provider: fake
  params:
    release: 2026-09-23.0
    filters:
      subtype: region
- id: orphan_regions
  name: Orphan regions
  id_property: code
  provider: missing
"""

HTML = {"Accept": "text/html"}


def _fake_provider(**_: Any) -> dict[str, Any]:
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"code": "R1"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
        ],
    }


@pytest.fixture(autouse=True)
def _feature_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configs = tmp_path / "feature_templates"
    configs.mkdir()
    (configs / "demo.yaml").write_text(TEMPLATES, encoding="utf-8")
    monkeypatch.setattr(feature_templates, "CONFIGS_DIR", configs)
    feature_templates.reset_feature_template_caches()
    monkeypatch.setattr(
        feature_providers, "get_feature_provider", lambda name: _fake_provider if name == "fake" else None
    )
    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "features")
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")


def _collection_ids(client: TestClient) -> set[str]:
    return {item["id"] for item in client.get("/features").json()["items"]}


# --- listing -------------------------------------------------------------------------------


def test_the_template_listing_includes_feature_templates_marked_by_kind(client: TestClient) -> None:
    listed = {t["id"]: t for t in client.get("/dataset-templates").json()}

    assert listed["demo_regions"]["itemType"] == "feature"
    assert listed["demo_regions"]["ingestable"] is True
    assert listed["orphan_regions"]["ingestable"] is False  # names no provider this instance has
    rasters = [t for t in listed.values() if t["itemType"] == "coverage"]
    assert rasters, "raster templates are still listed"


def test_a_feature_template_is_fetchable_by_id_and_reports_whether_it_was_fetched(client: TestClient) -> None:
    before = client.get("/dataset-templates/demo_regions")
    assert before.status_code == 200
    assert before.json()["itemType"] == "feature" and before.json()["has_data"] is False

    client.post("/features/demo_regions/refresh")

    assert client.get("/dataset-templates/demo_regions").json()["has_data"] is True


def test_the_templates_page_lists_feature_templates_with_a_type_filter(client: TestClient) -> None:
    page = client.get("/dataset-templates", headers=HTML).text

    assert 'href="/dataset-templates/demo_regions"' in page
    assert 'data-kind="features"' in page and 'data-kind="raster"' in page
    assert 'data-filter-field="kind"' in page
    # A template whose provider is missing is not offered for fetching.
    assert "orphan_regions" not in page


# --- the template page ---------------------------------------------------------------------


def test_a_feature_template_page_offers_fetching_without_a_date_range(client: TestClient) -> None:
    page = client.get("/dataset-templates/demo_regions", headers=HTML)

    assert page.status_code == 200
    html = page.text
    assert 'action="/manage/features/refresh"' in html
    assert 'name="start"' not in html and 'name="end"' not in html
    assert "region" in html and "2026-09-23.0" in html  # filters and the pinned release
    assert "ODbL-1.0" in html and "share-alike" in html  # the licence and what it requires


def test_a_read_only_instance_shows_the_page_without_the_form_and_refuses_fetching(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    html = client.get("/dataset-templates/demo_regions", headers=HTML).text
    assert "fetch-form" not in html and "read-only" in html
    assert client.post("/features/demo_regions/refresh").status_code == 403
    assert client.post("/manage/features/refresh", data={"collection_id": "demo_regions"}).status_code == 403


# --- fetching ------------------------------------------------------------------------------


def test_refresh_fetches_the_collection_synchronously(client: TestClient) -> None:
    response = client.post("/features/demo_regions/refresh")

    assert response.status_code == 200
    assert response.json()["id"] == "demo_regions"
    assert "demo_regions" in _collection_ids(client)


def test_refresh_can_run_as_a_background_job(client: TestClient) -> None:
    response = client.post("/features/demo_regions/refresh", headers={"Prefer": "respond-async"})

    assert response.status_code == 202
    location = response.headers["Location"]
    assert location.startswith("/ingestions/jobs/")
    status = "accepted"
    for _ in range(100):
        status = client.get(location).json()["status"]
        if status not in ("accepted", "queued", "running"):
            break
        time.sleep(0.05)
    assert status in ("successful", "completed", "finished")
    assert "demo_regions" in _collection_ids(client)


def test_refresh_refuses_before_queueing_an_unknown_or_unfetchable_template(client: TestClient) -> None:
    assert client.post("/features/nowhere/refresh").status_code == 404
    assert client.post("/features/orphan_regions/refresh", headers={"Prefer": "respond-async"}).status_code == 400


def test_the_page_stream_fetches_and_finishes(client: TestClient) -> None:
    response = client.post("/manage/features/refresh", data={"collection_id": "demo_regions", "publish": "on"})

    assert response.status_code == 200
    events = [json.loads(line[len("data: ") :]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == {"finished": True, "message": "Fetched Demo regions"}
    assert "demo_regions" in _collection_ids(client)


def test_the_page_stream_refuses_an_unknown_template_as_json(client: TestClient) -> None:
    response = client.post("/manage/features/refresh", data={"collection_id": "nowhere"})

    assert response.status_code == 404
    assert "not found" in response.json()["error"]
