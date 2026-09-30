"""A forecast dataset may be ingested without a fixed date range (CLIM-868).

Historical datasets are unchanged: `start` is still required. A dataset declaring
`temporal_direction: future` may omit it, meaning "from now" — a fixed date would be stale
by the next day, and a scheduled re-ingest would silently drift out of the forecast window.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import CreateIngestionRequest
from open_climate_service.shared.time import utc_today


def _write_template(tmp_path: Path, body: str, name: str = "forecast.yaml") -> None:
    (tmp_path / name).write_text(body, encoding="utf-8")


# --- how a template declares direction -----------------------------------------------


def test_defaults_to_past_when_undeclared() -> None:
    assert registry.temporal_direction({"id": "x"}) == "past"
    assert registry.is_future_facing({"id": "x"}) is False


def test_reads_a_declared_direction() -> None:
    assert registry.is_future_facing({"id": "x", "temporal_direction": "future"}) is True
    assert registry.is_future_facing({"id": "x", "temporal_direction": "past"}) is False


def test_accepts_a_future_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        """
- id: tmax_forecast_daily
  name: Max temperature forecast
  variable: tmax
  period_type: daily
  temporal_direction: future
  sync:
    kind: temporal
  ingestion:
    plugin: does.not.matter.ForTemplateValidation
""",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", tmp_path)
    loaded = {d["id"]: d for d in registry.list_datasets()}
    assert registry.is_future_facing(loaded["tmax_forecast_daily"]) is True


def test_spanning_is_not_treated_as_future() -> None:
    """A dataset straddling now must still be given a start, or history is silently lost.

    WorldPop Global2 runs 2015–2030. Defaulting its start to "now" would ingest only the
    projected years and drop every historical one — usually the half that is wanted.
    """
    spanning = {"id": "worldpop_population_global2_100m", "temporal_direction": "spanning"}
    assert registry.temporal_direction(spanning) == "spanning"
    assert registry.is_future_facing(spanning) is False


def test_worldpop_global2_declares_itself_spanning() -> None:
    """The built-in that motivated the third value: declared extent 2015 → 2030."""
    loaded = {d["id"]: d for d in registry.list_datasets()}
    for dataset_id in ("worldpop_population_global2_100m", "worldpop_agesex_global2_100m"):
        dataset = loaded[dataset_id]
        assert registry.temporal_direction(dataset) == "spanning"
        assert registry.declared_temporal_end(dataset) == "2030"
        assert registry.is_future_facing(dataset) is False


def test_declared_temporal_end_is_none_when_absent() -> None:
    assert registry.declared_temporal_end({"id": "x"}) is None
    assert registry.declared_temporal_end({"id": "x", "extents": {}}) is None
    assert registry.declared_temporal_end({"id": "x", "extents": {"temporal": {"begin": "1981"}}}) is None


def test_spanning_dataset_still_requires_a_start_over_http(client: TestClient) -> None:
    response = client.post("/ingestions", json={"dataset_id": "worldpop_population_global2_100m"})
    assert response.status_code == 400, response.text
    assert "requires a start period" in response.json()["detail"]


def test_ingest_form_offers_the_declared_end_for_a_spanning_dataset(client: TestClient) -> None:
    """So WorldPop's page prefills through 2030 rather than truncating at today."""
    body = client.get("/dataset-templates/worldpop_population_global2_100m?f=html").text
    assert 'value="2030"' in body.split('id="end"', 1)[1].split("/>", 1)[0]
    assert "Prefilled to the end of what the source covers." in body


def test_rejects_an_unsupported_direction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_template(
        tmp_path,
        """
- id: sideways
  name: Sideways
  variable: v
  period_type: daily
  temporal_direction: sideways
  sync:
    kind: temporal
  ingestion:
    plugin: a.b.C
""",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="unsupported temporal_direction 'sideways'"):
        registry.list_datasets()


def test_rejects_future_combined_with_static_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A static dataset has no upstream to look ahead into, so the pair is incoherent."""
    _write_template(
        tmp_path,
        """
- id: static_forecast
  name: Static forecast
  variable: v
  period_type: daily
  temporal_direction: future
  sync:
    kind: static
""",
    )
    monkeypatch.setattr(registry, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="has no upstream to look ahead into"):
        registry.list_datasets()


# --- resolving an omitted start -------------------------------------------------------


def test_supplied_start_is_used_unchanged() -> None:
    resolved = ingestion_services._resolve_request_start(
        "2024-01-01", dataset={"id": "x", "period_type": "daily"}, period_type="daily"
    )
    assert resolved == "2024-01-01"


def test_omitted_start_on_a_forecast_resolves_to_today() -> None:
    resolved = ingestion_services._resolve_request_start(
        None,
        dataset={"id": "tmax_forecast_daily", "temporal_direction": "future"},
        period_type="daily",
    )
    assert resolved == utc_today().isoformat()


@pytest.mark.parametrize(
    ("period_type", "expected"),
    [
        ("daily", lambda: utc_today().isoformat()),
        ("monthly", lambda: f"{utc_today().year:04d}-{utc_today().month:02d}"),
        ("yearly", lambda: str(utc_today().year)),
    ],
)
def test_resolved_start_is_in_the_dataset_native_period_format(period_type: str, expected: object) -> None:
    resolved = ingestion_services._resolve_request_start(
        None, dataset={"id": "f", "temporal_direction": "future"}, period_type=period_type
    )
    assert resolved == expected()  # type: ignore[operator]


def test_omitted_start_on_a_historical_dataset_is_a_client_error() -> None:
    with pytest.raises(HTTPException) as exc:
        ingestion_services._resolve_request_start(
            None, dataset={"id": "chirps3_precipitation_daily"}, period_type="daily"
        )
    assert exc.value.status_code == 400
    # The message must name the dataset and how to opt in, not just say "required".
    assert "chirps3_precipitation_daily" in exc.value.detail
    assert "temporal_direction: future" in exc.value.detail


def test_forecast_horizon_reaches_beyond_now() -> None:
    """The bug this guards: an omitted end collapsed a 7-day forecast to a single day.

    An omitted end means "through the latest available period" for a historical dataset, so
    core substitutes now. For a forecast, now is the *start* of the window — substituting it
    there hands the plugin ``start == end == today``, one period comes back, and the whole
    point of an omittable start is cancelled out. A forecast gets a forward horizon instead.
    """
    horizon = ingestion_services._forecast_horizon({"id": "f"}, "daily")
    assert horizon > utc_today().isoformat()


def test_forecast_horizon_prefers_a_declared_end() -> None:
    dataset: dict[str, object] = {"id": "f", "extents": {"temporal": {"end": "2030"}}}
    assert ingestion_services._forecast_horizon(dataset, "yearly") == "2030"


@pytest.mark.parametrize("period_type", ["daily", "monthly", "yearly", "hourly", "weekly"])
def test_forecast_horizon_survives_a_leap_day(monkeypatch: pytest.MonkeyPatch, period_type: str) -> None:
    """29 February has no counterpart in the following year.

    An earlier version advanced the year with ``replace(year=+1)``, which raises "day is
    out of range for month" on a leap day — a crash on one day every four years, for a
    horizon that is approximate by design.
    """
    import datetime as _datetime

    monkeypatch.setattr(ingestion_services, "utc_today", lambda: _datetime.date(2028, 2, 29))
    horizon = ingestion_services._forecast_horizon({"id": "f"}, period_type)
    assert horizon  # did not raise
    assert horizon > "2028"


@pytest.mark.parametrize("period_type", ["daily", "monthly", "yearly", "hourly", "weekly"])
def test_forecast_horizon_is_period_native(period_type: str) -> None:
    """It is handed straight to the plugin, so it has to parse as that period type."""
    from open_climate_service.shared.time import normalize_period_string

    horizon = ingestion_services._forecast_horizon({"id": "f"}, period_type)
    assert normalize_period_string(horizon, period_type) == horizon


# --- the request contract -------------------------------------------------------------


def test_request_schema_allows_an_omitted_start() -> None:
    assert CreateIngestionRequest(dataset_id="x").start is None


def test_request_schema_still_accepts_a_start() -> None:
    assert CreateIngestionRequest(dataset_id="x", start="2024-01-01").start == "2024-01-01"


def test_post_ingestions_without_start_rejects_a_historical_dataset(client: TestClient) -> None:
    """End to end: the API must refuse, with the actionable message rather than a 422."""
    response = client.post("/ingestions", json={"dataset_id": "chirps3_precipitation_daily"})
    assert response.status_code == 400, response.text
    assert "temporal_direction: future" in response.json()["detail"]


# --- the ingest form ------------------------------------------------------------------


def test_ingest_form_requires_a_start_for_a_historical_source(client: TestClient) -> None:
    """A historical source must name its start; only a forecast may leave it blank."""
    body = client.get("/dataset-templates/chirps3_precipitation_daily?f=html").text
    start_input = body.split('id="start"', 1)[1].split("/>", 1)[0]
    assert "required" in start_input


def test_manage_form_start_rejection_is_dataset_aware(client: TestClient) -> None:
    """The unconditional "Start date is required" gate is gone from the ingest route.

    The rejection still happens in ``manage_ingest`` rather than downstream in
    ``create_artifact`` — deliberately, because the ingest runs inside an event stream and a
    failure there arrives after the response has begun. What changed is that it now consults
    ``is_future_facing()`` instead of rejecting every blank start.
    """
    from open_climate_service.system import routes

    source = Path(routes.__file__).read_text(encoding="utf-8")
    assert 'detail="Start date is required"' not in source
    assert "is_future_facing(template)" in source
