"""Pipelines: create, validate and dry-run one from the page and the API, without a server or DHIS2."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.pipelines import service, store
from open_climate_service.pipelines.schemas import Check, PipelineSpec, ValidationResult

_OU = ["ImspTQPwCqd", "O6uvpzGd5pu"]
_ELEMENT = "LeePHYkZwBk"
_SPEC: dict[str, Any] = {
    "id": "rain-to-dhis2",
    "name": "Rain to DHIS2",
    "source": {"dataset": "rain_monthly"},
    "destination": {
        "connection": "local",
        "data_set": "N63vVmt2uY0",
        "organisation_units": {"feature_collection": "districts"},
        "series": [{"data_element": _ELEMENT}],
    },
    "aggregation": {"spatial": {"reducer": "mean"}},
}


def _dataset(period_type: str = "monthly", published: bool = True) -> Any:
    return SimpleNamespace(
        dataset_id="rain_monthly",
        dataset_name="Rain, monthly",
        variable="tp",
        period_type=period_type,
        item_type="coverage",
        extent=SimpleNamespace(temporal=SimpleNamespace(start="2025-01", end="2025-02")),
        publication=SimpleNamespace(status="published" if published else "unpublished"),
    )


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(
        api_config, "_cache", {"dhis2_connections": [{"id": "local", "url": "http://dhis2.test", "token_env": "T"}]}
    )
    monkeypatch.setattr(
        service, "dataset_record", lambda dataset_id: _dataset() if dataset_id == "rain_monthly" else None
    )
    monkeypatch.setattr(
        service,
        "_feature_binding",
        lambda collection_id: (
            service.FeatureBinding(ids=list(_OU), geometry_types=["Polygon"], connection="local", level=2)
            if collection_id == "districts"
            else (_ for _ in ()).throw(LookupError(f"'{collection_id}' is not a registered feature collection"))
        ),
    )
    monkeypatch.setattr(
        service,
        "choices",
        lambda: {
            "datasets": [
                {
                    "id": "rain_monthly",
                    "name": "Rain, monthly",
                    "variable": "tp",
                    "period_type": "monthly",
                    "cadence": "monthly",
                    "coverage": {"start": "2025-01", "end": "2025-02"},
                }
            ],
            "collections": [
                {
                    "id": "districts",
                    "name": "Districts",
                    "feature_count": 2,
                    "geometry_types": ["Polygon"],
                    "connection": "local",
                    "level": 2,
                }
            ],
            "connections": ["local"],
        },
    )


class _FakeClient:
    def __init__(self, data_set: dict[str, Any], summary: dict[str, Any]) -> None:
        self.data_set, self.summary, self.posts = data_set, summary, []

    def get(self, path: str, *, params: Any = None) -> dict[str, Any]:
        return self.data_set

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        self.posts.append(kwargs)
        return self.summary

    def close(self) -> None:
        pass


def _dhis2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    period_type: str = "Monthly",
    assigned: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> _FakeClient:
    client = _FakeClient(
        {
            "id": "N63vVmt2uY0",
            "displayName": "OCS test",
            "periodType": period_type,
            "dataSetElements": [{"dataElement": {"id": _ELEMENT}}],
            "organisationUnits": [{"id": ou} for ou in (assigned if assigned is not None else _OU)],
        },
        summary
        or {
            "status": "SUCCESS",
            "importCount": {"imported": 0, "updated": 4, "ignored": 0, "deleted": 0},
            "conflicts": [],
        },
    )
    import open_climate_service.exports.dhis2 as dhis2_module

    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: client)
    return client


# --- schema and compile ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "change", [{"id": "bad id"}, {"destination": {**_SPEC["destination"], "series": [{"data_element": "nope"}]}}]
)
def test_spec_rejects_bad_ids(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PipelineSpec.model_validate({**_SPEC, **change})


def test_compile_produces_an_export_and_a_trigger() -> None:
    compiled = service.compile_pipeline(PipelineSpec.model_validate(_SPEC), "monthly")
    export = compiled["exports"][0]
    trigger = compiled["automation"]["workflow_triggers"][0]
    assert export == {
        "id": "rain-to-dhis2",
        "plugin": "dhis2",
        "dataset": "rain_monthly",
        "connection": "local",
        "aggregation": "mean",
        "period_type": "monthly",
        "series": [{"select": {}, "data_element": _ELEMENT}],
    }
    assert trigger["on_update_of"] == "rain_monthly"
    assert trigger["arguments"]["geometries"] == {"from_features": "districts"}
    assert trigger["arguments"]["temporal_extent"] == ["$event.previous_end", "$event.current_end"]
    assert trigger["deliver"] == {"export": "rain-to-dhis2", "dry_run": True}


@pytest.mark.parametrize(
    "delivery",
    [{"mode": "paused"}, {"policy": "manual"}, {"policy": "scheduled", "release_cron": "0 2 5 * *"}],
)
def test_compile_emits_no_trigger_unless_delivering_on_update(delivery: dict[str, Any]) -> None:
    compiled = service.compile_pipeline(PipelineSpec.model_validate({**_SPEC, "delivery": delivery}), "monthly")
    assert set(compiled) == {"exports"} and compiled["exports"][0]["id"] == "rain-to-dhis2"


def test_spec_never_carries_a_sync_schedule_and_checks_the_release() -> None:
    with pytest.raises(ValueError, match="maintenance"):
        PipelineSpec.model_validate(
            {**_SPEC, "source": {"dataset": "rain_monthly", "maintenance": {"cron": "0 6 * * *"}}}
        )
    with pytest.raises(ValueError, match="release_cron"):
        PipelineSpec.model_validate({**_SPEC, "delivery": {"policy": "scheduled"}})
    with pytest.raises(ValueError, match="invalid five-field cron"):
        PipelineSpec.model_validate({**_SPEC, "delivery": {"policy": "scheduled", "release_cron": "nope"}})
    with pytest.raises(ValueError, match="only applies"):
        PipelineSpec.model_validate({**_SPEC, "delivery": {"policy": "manual", "release_cron": "0 2 5 * *"}})


# --- validation ------------------------------------------------------------------------------


def test_validation_passes_when_everything_lines_up(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    record = store.create_record(PipelineSpec.model_validate(_SPEC))
    result = service.validate_pipeline(record)
    assert result.valid and result.period_type == "monthly"
    assert {check.id: check.status for check in result.checks} == {
        "dataset": "pass",
        "cadence": "pass",
        "series": "pass",
        "organisation_units": "pass",
        "connection": "pass",
        "spatial_reducer": "pass",
        "dhis2_metadata": "pass",
        "configuration": "pass",
        "schedule": "skip",
        "delivery": "pass",
    }


def test_validation_names_each_mismatch(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch, period_type="Daily", assigned=[_OU[0]])
    spec = {
        **_SPEC,
        "destination": {
            **_SPEC["destination"],
            "connection": "elsewhere",
            "organisation_units": {"feature_collection": "nowhere"},
        },
    }
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(spec)))
    by_id = {check.id: check for check in result.checks}
    assert not result.valid
    assert by_id["organisation_units"].status == "fail" and "not a registered" in by_id["organisation_units"].message
    assert by_id["connection"].status == "fail"
    assert by_id["dhis2_metadata"].status == "skip"  # no connection to ask


def test_validation_reports_dhis2_period_and_assignment(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch, period_type="Daily", assigned=[_OU[0]])
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(_SPEC)))
    message = next(check for check in result.checks if check.id == "dhis2_metadata").message
    assert "period type is 'Daily', the export is Monthly" in message
    assert "1 of 2 organisation units are not assigned" in message


def test_validation_rejects_wrong_connection_or_non_polygon_collection(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(
        service,
        "_feature_binding",
        lambda collection_id: service.FeatureBinding(
            ids=list(_OU), geometry_types=["Point"], connection="another", level=4
        ),
    )
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(_SPEC)))
    check = next(item for item in result.checks if item.id == "organisation_units")
    assert check.status == "fail" and "Polygon" in check.message
    assert "ids are not" not in check.message

    monkeypatch.setattr(
        service,
        "_feature_binding",
        lambda collection_id: service.FeatureBinding(
            ids=list(_OU), geometry_types=["Polygon"], connection="another", level=2
        ),
    )
    other = PipelineSpec.model_validate({**_SPEC, "id": "rain-to-other-dhis2"})
    result = service.validate_pipeline(store.create_record(other))
    check = next(item for item in result.checks if item.id == "organisation_units")
    assert check.status == "fail" and "destination uses 'local'" in check.message

    monkeypatch.setattr(
        service,
        "_feature_binding",
        lambda collection_id: service.FeatureBinding(
            ids=list(_OU), geometry_types=["Polygon"], connection=None, level=None
        ),
    )
    non_dhis2 = PipelineSpec.model_validate({**_SPEC, "id": "rain-to-overture"})
    result = service.validate_pipeline(store.create_record(non_dhis2))
    check = next(item for item in result.checks if item.id == "organisation_units")
    assert check.status == "fail" and "was not fetched from DHIS2" in check.message


def test_validation_does_not_claim_to_reuse_a_disabled_schedule(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(
        api_config,
        "_cache",
        {
            "dhis2_connections": [{"id": "local", "url": "http://dhis2.test", "token_env": "T"}],
            "scheduler": {
                "enabled": False,
                "dataset_sync": [{"dataset_id": "rain_monthly", "cron": "0 6 * * *"}],
            },
        },
    )
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(_SPEC)))
    check = next(item for item in result.checks if item.id == "schedule")
    assert check.status == "skip" and "disabled" in check.message and "0 6 * * *" in check.message


def test_validation_names_the_source_schedule_and_the_delivery_policy(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(
        api_config,
        "_cache",
        {
            "dhis2_connections": [{"id": "local", "url": "http://dhis2.test", "token_env": "T"}],
            "scheduler": {"enabled": True, "dataset_sync": [{"dataset_id": "rain_monthly", "cron": "0 6 * * *"}]},
        },
    )
    spec = {**_SPEC, "delivery": {"mode": "paused", "policy": "scheduled", "release_cron": "0 2 5 * *"}}
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(spec)))
    by_id = {item.id: item for item in result.checks}
    assert result.valid
    assert by_id["schedule"].status == "pass" and "0 6 * * *" in by_id["schedule"].message
    assert by_id["delivery"].status == "pass" and by_id["delivery"].message.startswith("paused")
    assert "not applied yet" in by_id["delivery"].message


def test_validation_fails_when_declared_dhis2_dataset_cannot_be_read(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.dhis2 as dhis2_module

    monkeypatch.setattr(dhis2_module, "get_connection", lambda connection_id: (_ for _ in ()).throw(OSError("down")))
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(_SPEC)))
    check = next(item for item in result.checks if item.id == "dhis2_metadata")
    assert check.status == "fail" and "could not read" in check.message


def test_validation_refuses_a_cadence_dhis2_cannot_take(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "dataset_record", lambda dataset_id: _dataset(period_type="dekadal"))
    _dhis2(monkeypatch)
    result = service.validate_pipeline(store.create_record(PipelineSpec.model_validate(_SPEC)))
    assert next(check for check in result.checks if check.id == "cadence").status == "fail"
    assert result.period_type is None


def test_live_delivery_needs_a_passed_dry_run(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    spec = PipelineSpec.model_validate({**_SPEC, "delivery": {"mode": "live"}})
    record = store.create_record(spec)
    assert next(c for c in service.validate_pipeline(record).checks if c.id == "delivery").status == "fail"


# --- dry run ---------------------------------------------------------------------------------


def _draft_export(values: list[dict[str, str]]) -> tuple[Any, Any]:
    from open_climate_service.exports.base import RenderedExport
    from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin

    plugin = Dhis2ExportPlugin()
    rendered = RenderedExport(
        content=json.dumps({"dataValues": values}).encode(),
        record_count=len(values),
        periods=tuple(sorted({value["period"] for value in values})),
    )
    return SimpleNamespace(references={"connection": "local"}, plugin=plugin), rendered


def test_compiled_named_export_renders_every_series() -> None:
    from open_climate_service.exports.service import resolve_export_definition

    second_element = "K2mt1TTI2Xd"
    spec = PipelineSpec.model_validate(
        {
            **_SPEC,
            "destination": {
                **_SPEC["destination"],
                "series": [{"data_element": _ELEMENT}, {"data_element": second_element}],
            },
        }
    )
    resolved = resolve_export_definition(service.compile_export(spec, "monthly"))
    frame = pd.DataFrame({"geometry": [_OU[0]], "t": ["2025-01-01"], "tp": [1.5]})
    payload = json.loads(resolved.plugin.render(frame, resolved.mapping).content)
    assert [value["dataElement"] for value in payload["dataValues"]] == [_ELEMENT, second_element]


def test_draft_render_uses_named_export_mapping_and_provenance(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import execution
    from open_climate_service.openeo.execution import SaveResultEnvelope

    second_element = "K2mt1TTI2Xd"
    spec = PipelineSpec.model_validate(
        {
            **_SPEC,
            "destination": {
                **_SPEC["destination"],
                "series": [{"data_element": _ELEMENT}, {"data_element": second_element}],
            },
        }
    )
    envelope = SaveResultEnvelope(pd.DataFrame({"geometry": [_OU[0]], "t": ["2025-01-01"], "tp": [1.5]}), "NetCDF")
    envelope.provenance = {
        "sources": [{"collection_id": "rain_monthly"}],
        "spatial_aggregations": ["mean"],
    }
    monkeypatch.setattr(execution, "run_process_graph", lambda graph: envelope)
    _, rendered = service._render_export(spec, "2025-01-01", "2025-01-31")
    payload = json.loads(rendered.content)
    assert [value["dataElement"] for value in payload["dataValues"]] == [_ELEMENT, second_element]


def test_draft_render_applies_the_export_data_cadence_gate(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import execution
    from open_climate_service.openeo.execution import SaveResultEnvelope

    envelope = SaveResultEnvelope(
        pd.DataFrame(
            {
                "geometry": [_OU[0], _OU[0]],
                "t": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "tp": [1.5, 2.0],
            }
        ),
        "NetCDF",
    )
    envelope.provenance = {
        "sources": [{"collection_id": "rain_monthly"}],
        "spatial_aggregations": ["mean"],
    }
    monkeypatch.setattr(execution, "run_process_graph", lambda graph: envelope)
    with pytest.raises(ValueError, match="spaced daily"):
        service._render_export(PipelineSpec.model_validate(_SPEC), "2025-01-01", "2025-01-31")


def test_dry_run_renders_the_payload_and_folds_the_report(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _dhis2(monkeypatch)
    frame = pd.DataFrame(
        {"geometry": _OU * 2, "t": ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"], "tp": [1.0, 2.0, 3.0, 4.0]}
    )
    monkeypatch.setattr(
        service,
        "_render_export",
        lambda spec, start, end: _draft_export(
            [
                {
                    "dataElement": _ELEMENT,
                    "orgUnit": r["geometry"],
                    "period": pd.Timestamp(r["t"]).strftime("%Y%m"),
                    "value": str(r["tp"]),
                }
                for r in frame.to_dict("records")
            ]
        ),
    )
    record = store.create_record(PipelineSpec.model_validate(_SPEC))
    result = service.dry_run_pipeline(record, "2025-01-01", "2025-02-28")
    assert result.error is None
    assert result.values == 4 and result.org_units == 2 and result.periods == ["202501", "202502"]
    assert client.posts[0]["params"] == {"importStrategy": "CREATE_AND_UPDATE", "dryRun": "true"}
    assert result.report is not None
    assert result.report["outcome"] == "dry_run" and result.report["updated"] == 4 and result.passed


def test_dry_run_keeps_the_error_when_the_graph_refuses(instance: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(spec: Any, start: str, end: str) -> tuple[Any, Any]:
        raise ValueError("period_type 'monthly' is coarser than the result's daily spacing")

    monkeypatch.setattr(service, "_render_export", refuse)
    result = service.dry_run_pipeline(
        store.create_record(PipelineSpec.model_validate(_SPEC)), "2025-01-01", "2025-02-28"
    )
    assert result.error is not None and "coarser" in result.error and not result.passed


# --- pages and API ---------------------------------------------------------------------------


@pytest.fixture
def client(instance: None) -> TestClient:
    from open_climate_service.main import app

    return TestClient(app)


def test_page_creates_validates_and_shows_a_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    page = client.get("/pipelines", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Configured pipelines" in page.text and "New pipeline" not in page.text
    create_page = client.get("/pipelines/new", headers={"Accept": "text/html"})
    assert create_page.status_code == 200 and "New pipeline" in create_page.text and "Rain, monthly" in create_page.text
    created = client.post(
        "/pipelines",
        data={
            "id": "rain-to-dhis2",
            "name": "Rain to DHIS2",
            "dataset": "rain_monthly",
            "feature_collection": "districts",
            "connection": "local",
            "data_set": "N63vVmt2uY0",
            "data_element": _ELEMENT,
            "reducer": "mean",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303, created.text[:600]
    assert created.headers["location"].endswith("/pipelines/rain-to-dhis2")
    validated = client.post(
        "/pipelines/rain-to-dhis2/validate", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert validated.status_code == 303
    shown = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert shown.status_code == 200
    assert "dhis2_metadata" in shown.text and "every organisation unit assigned" in shown.text
    assert "on_update_of: rain_monthly" in shown.text
    listed = client.get("/pipelines", headers={"Accept": "application/json"}).json()
    assert [item["spec"]["id"] for item in listed["items"]] == ["rain-to-dhis2"]
    assert listed["items"][0]["validation"]["valid"] is True


def test_detail_get_is_read_only_and_renders_stored_invalid_validation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = store.create_record(PipelineSpec.model_validate(_SPEC))
    record.validation = ValidationResult(
        valid=False,
        checked_at="2026-10-06T10:00:00Z",
        checks=[Check(id="dhis2_metadata", status="fail", message="DHIS2 was unavailable")],
    )
    store.save_record(record)

    def refuse_validation(record: Any) -> Any:
        raise AssertionError("GET must not validate")

    monkeypatch.setattr(service, "validate_pipeline", refuse_validation)
    shown = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert shown.status_code == 200 and "DHIS2 was unavailable" in shown.text
    as_json = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "application/json"})
    assert as_json.status_code == 200 and as_json.json()["validation"]["valid"] is False
    assert store.get_record("rain-to-dhis2") == record


def test_page_keeps_the_draft_on_a_bad_form(client: TestClient) -> None:
    response = client.post(
        "/pipelines",
        data={
            "id": "rain-to-dhis2",
            "dataset": "rain_monthly",
            "feature_collection": "districts",
            "connection": "local",
            "data_element": "bad",
            "reducer": "mean",
        },
    )
    assert response.status_code == 400
    assert "data_element" in response.text and 'value="rain-to-dhis2"' in response.text


def test_api_creates_and_dry_runs_a_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(
        service,
        "_render_export",
        lambda spec, start, end: _draft_export(
            [{"dataElement": _ELEMENT, "orgUnit": _OU[0], "period": "202501", "value": "1"}]
        ),
    )
    created = client.post("/pipelines", json=_SPEC)
    assert created.status_code == 201 and created.json()["compiled"]["exports"][0]["id"] == "rain-to-dhis2"
    assert client.post("/pipelines", json=_SPEC).status_code == 409
    ran = client.post("/pipelines/rain-to-dhis2/dry-run", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert ran.status_code == 200 and ran.json()["values"] == 1 and ran.json()["report"]["outcome"] == "dry_run"
    live = client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"})
    assert live.status_code == 200
    assert live.json()["compiled"]["automation"]["workflow_triggers"][0]["deliver"]["dry_run"] is False
    safe = client.post("/pipelines/rain-to-dhis2/mode/dry_run", headers={"Accept": "application/json"})
    assert safe.status_code == 200 and safe.json()["spec"]["delivery"]["mode"] == "dry_run"
    paused = client.post("/pipelines/rain-to-dhis2/mode/paused", headers={"Accept": "application/json"})
    assert paused.status_code == 200 and paused.json()["spec"]["delivery"]["mode"] == "paused"
    assert "automation" not in paused.json()["compiled"]
    page = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Resume as dry run" in page.text
    assert client.get("/pipelines/missing", headers={"Accept": "application/json"}).status_code == 404


def test_failed_live_validation_does_not_persist_live_mode(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    monkeypatch.setattr(
        service,
        "_render_export",
        lambda spec, start, end: _draft_export(
            [{"dataElement": _ELEMENT, "orgUnit": _OU[0], "period": "202501", "value": "1"}]
        ),
    )
    assert client.post("/pipelines", json=_SPEC).status_code == 201
    assert (
        client.post("/pipelines/rain-to-dhis2/dry-run", json={"start": "2025-01-01", "end": "2025-01-31"}).status_code
        == 200
    )
    monkeypatch.setattr(
        service,
        "validate_pipeline",
        lambda record: ValidationResult(
            valid=False,
            checked_at="2026-10-06T10:00:00Z",
            checks=[Check(id="dhis2_metadata", status="fail", message="DHIS2 was unavailable")],
        ),
    )
    response = client.post("/pipelines/rain-to-dhis2/mode/live", headers={"Accept": "application/json"})
    assert response.status_code == 409 and "DHIS2 was unavailable" in response.text
    assert store.get_record("rain-to-dhis2").spec.delivery.mode == "dry_run"  # type: ignore[union-attr]


# --- save, edit, delete, run ----------------------------------------------------------------


def _valid_validation() -> ValidationResult:
    return ValidationResult(valid=True, checked_at="2026-10-06T10:00:00Z", period_type="monthly", checks=[])


def _invalid_validation() -> ValidationResult:
    return ValidationResult(
        valid=False,
        checked_at="2026-10-06T10:00:00Z",
        checks=[Check(id="dhis2_metadata", status="fail", message="DHIS2 was unavailable")],
    )


def test_a_validated_pipeline_is_a_named_export_and_the_file_wins(
    instance: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.exports.service import resolve_named_export

    with pytest.raises(ValueError, match="Unknown export"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    record = store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    assert resolved.references["dataset"] == "rain_monthly" and resolved.mapping["period_type"] == "monthly"
    record.validation = _invalid_validation()
    store.save_record(record)
    with pytest.raises(ValueError, match="Unknown export"):
        resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"})
    record.validation = _valid_validation()
    store.save_record(record)
    configured = {**service.compile_export(record.spec, "monthly"), "aggregation": "sum"}
    api_config._cache["exports"] = [configured]  # type: ignore[index]
    assert resolve_named_export("DHIS2JSON", {"export": "rain-to-dhis2"}).mapping["aggregation"] == "sum"


def test_save_refuses_an_invalid_pipeline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "validate_pipeline", lambda record: _invalid_validation())
    refused = client.post("/pipelines", json=_SPEC)
    assert refused.status_code == 422
    assert refused.json()["detail"]["checks"]["checks"][0]["id"] == "dhis2_metadata"
    page = client.post(
        "/pipelines",
        data={
            "id": "rain-to-dhis2",
            "dataset": "rain_monthly",
            "feature_collection": "districts",
            "connection": "local",
            "data_element": _ELEMENT,
            "reducer": "mean",
        },
    )
    assert page.status_code == 400 and "not saved" in page.text and "DHIS2 was unavailable" in page.text
    assert 'value="rain-to-dhis2"' in page.text
    assert store.list_records() == []


def test_edit_validates_the_change_and_clears_the_dry_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _dhis2(monkeypatch)
    record = store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())
    record.dry_run = service.DryRunResult(
        ran_at="2026-10-06T10:00:00Z", start="2025-01-01", end="2025-01-31", report={"outcome": "dry_run"}
    )
    store.save_record(record)
    form = client.get("/pipelines/rain-to-dhis2/edit", headers={"Accept": "text/html"})
    assert form.status_code == 200 and "Edit pipeline" in form.text and "readonly" in form.text
    assert 'action="/pipelines/rain-to-dhis2"' in form.text and ">Save<" in form.text

    monkeypatch.setattr(service, "validate_pipeline", lambda record: _invalid_validation())
    bad = client.post("/pipelines/rain-to-dhis2", json={**_SPEC, "aggregation": {"spatial": {"reducer": "sum"}}})
    assert bad.status_code == 422 and "dhis2_metadata" in bad.text
    kept = store.get_record("rain-to-dhis2")
    assert kept is not None and kept.spec.aggregation.spatial.reducer == "mean" and kept.dry_run is not None

    monkeypatch.setattr(service, "validate_pipeline", lambda record: _valid_validation())
    unchanged = client.post("/pipelines/rain-to-dhis2", json=_SPEC)
    assert unchanged.status_code == 200 and unchanged.json()["dry_run"] is not None
    changed = client.post("/pipelines/rain-to-dhis2", json={**_SPEC, "aggregation": {"spatial": {"reducer": "sum"}}})
    assert changed.status_code == 200 and changed.json()["dry_run"] is None
    assert changed.json()["spec"]["aggregation"]["spatial"]["reducer"] == "sum"
    saved = store.get_record("rain-to-dhis2")
    assert saved is not None and saved.spec.id == "rain-to-dhis2" and saved.dry_run is None
    assert client.post("/pipelines/missing", json=_SPEC).status_code == 404


def test_delete_from_api_and_from_the_page(client: TestClient) -> None:
    store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())
    assert client.delete("/pipelines/rain-to-dhis2").status_code == 204
    assert client.delete("/pipelines/rain-to-dhis2").status_code == 404
    assert store.get_record("rain-to-dhis2") is None

    store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())
    unconfirmed = client.post("/pipelines/rain-to-dhis2/delete", data={}, follow_redirects=False)
    assert unconfirmed.status_code == 400 and store.get_record("rain-to-dhis2") is not None
    confirmed = client.post("/pipelines/rain-to-dhis2/delete", data={"confirm": "yes"}, follow_redirects=False)
    assert confirmed.status_code == 303 and confirmed.headers["location"].endswith("/pipelines")
    assert store.get_record("rain-to-dhis2") is None


class _FakeOpenEO:
    """A job service that records what it was asked to run and finishes nothing by itself."""

    def __init__(self) -> None:
        from datetime import UTC, datetime

        from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus

        self.jobs: dict[str, OpenEOJobRecord] = {}
        self.started: list[str] = []
        self._record, self._status, self._now = OpenEOJobRecord, OpenEOJobStatus, lambda: datetime.now(UTC)

    def create_job(self, body: Any) -> Any:
        job = self._record(
            id=f"job-{len(self.jobs) + 1}",
            title=body.title,
            process=body.process,
            status=self._status.CREATED,
            created=self._now(),
        )
        self.jobs[job.id] = job
        return job

    def start_job(self, job_id: str) -> None:
        self.started.append(job_id)
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": self._status.RUNNING})

    def finish(self, job_id: str) -> Any:
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": self._status.FINISHED})
        return self.jobs[job_id]


@pytest.fixture
def openeo(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenEO:
    import open_climate_service.openeo.jobs as jobs_module

    fake = _FakeOpenEO()
    monkeypatch.setattr(jobs_module, "get_openeo_job_service", lambda: fake)
    monkeypatch.setattr(jobs_module, "store_get_job", lambda job_id: fake.jobs.get(job_id))
    return fake


def test_run_once_submits_a_job_and_delivers_when_it_finishes(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    import open_climate_service.exports.delivery as delivery_module

    deliveries: list[tuple[str, str, bool, str]] = []

    def fake_submit(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        deliveries.append((export_id, source_job_id, dry_run, idempotency_key))
        return f"delivery-{len(deliveries)}", False

    monkeypatch.setattr(delivery_module, "submit_delivery", fake_submit)
    store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())

    live_too_early = client.post(
        "/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31", "mode": "live"}
    )
    assert live_too_early.status_code == 409 and "dry run" in live_too_early.text
    assert client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01"}).status_code == 400

    submitted = client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert submitted.status_code == 202, submitted.text
    run = submitted.json()
    assert run["job_id"] == "job-1" and run["mode"] == "dry_run" and run["delivery_job_id"] is None
    assert openeo.started == ["job-1"]
    graph = openeo.jobs["job-1"].process["process_graph"]
    assert graph["agg"]["arguments"]["export"] == "rain-to-dhis2" and graph["agg"]["arguments"]["method"] == "mean"
    assert graph["agg"]["arguments"]["temporal_extent"] == ["2025-01-01", "2025-01-31"]

    too_early = client.post("/pipelines/rain-to-dhis2/runs/job-1/deliver", headers={"Accept": "application/json"})
    assert too_early.status_code == 409 and "not finished" in too_early.text
    assert client.post("/pipelines/rain-to-dhis2/runs/nope/deliver").status_code == 404

    service.on_job_finished(openeo.jobs["job-1"])
    assert deliveries == []
    service.on_job_finished(openeo.finish("job-1"))
    assert deliveries == [("rain-to-dhis2", "job-1", True, "pipeline:rain-to-dhis2:job-1:dry_run")]
    stored = store.get_record("rain-to-dhis2")
    assert stored is not None and stored.runs[0].delivery_job_id == "delivery-1"
    assert stored.runs[0].delivery_status_url == "/exports/rain-to-dhis2/jobs/delivery-1"

    again = client.post("/pipelines/rain-to-dhis2/runs/job-1/deliver", headers={"Accept": "application/json"})
    assert again.status_code == 200 and again.json()["delivery_job_id"] == "delivery-1" and len(deliveries) == 1

    page = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Run once" in page.text and "job-1"[:8] in page.text
    as_json = client.get("/pipelines/rain-to-dhis2", headers={"Accept": "application/json"}).json()
    assert as_json["runs"][0]["job_status"] == "finished"


def test_a_manual_deliver_runs_when_the_hand_off_did_not(
    client: TestClient, openeo: _FakeOpenEO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import open_climate_service.exports.delivery as delivery_module

    def refuse(export_id: str, source_job_id: str, dry_run: bool, idempotency_key: str) -> tuple[str, bool]:
        raise HTTPException(status_code=409, detail="No DHIS2 JSON result for this job")

    monkeypatch.setattr(delivery_module, "submit_delivery", refuse)
    store.create_record(PipelineSpec.model_validate(_SPEC), _valid_validation())
    assert (
        client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"}).status_code
        == 202
    )
    service.on_job_finished(openeo.finish("job-1"))
    stored = store.get_record("rain-to-dhis2")
    assert stored is not None and stored.runs[0].delivery_job_id is None
    assert stored.runs[0].error == "No DHIS2 JSON result for this job"

    monkeypatch.setattr(delivery_module, "submit_delivery", lambda *args: ("delivery-9", False))
    delivered = client.post("/pipelines/rain-to-dhis2/runs/job-1/deliver", headers={"Accept": "application/json"})
    assert delivered.status_code == 200 and delivered.json()["delivery_job_id"] == "delivery-9"
    assert delivered.json()["error"] is None


def test_run_needs_a_valid_pipeline(client: TestClient, openeo: _FakeOpenEO) -> None:
    store.create_record(PipelineSpec.model_validate(_SPEC), _invalid_validation())
    refused = client.post("/pipelines/rain-to-dhis2/runs", json={"start": "2025-01-01", "end": "2025-01-31"})
    assert refused.status_code == 409 and "validate" in refused.text and openeo.started == []
