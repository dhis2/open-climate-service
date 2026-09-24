"""Referencing a declared feature collection from a workflow trigger (CLIM-926, step 7).

The trigger layer's job is to turn `{"from_features": "districts"}` into something the process
graph can resolve, *without* putting geometry into the persisted job record. These cover both
halves of that: the rewrite produces a node the executor evaluates, and the job still records
which version of each collection it ran against; plus the startup validation that catches a typo
in a schedule before it ever fires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from open_climate_service import config as api_config
from open_climate_service.automation import service as automation_service
from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger
from open_climate_service.features import services as feature_services
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus

DISTRICTS_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}


def _box(code: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    }


@pytest.fixture(autouse=True)
def feature_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the store and record index at a temporary directory, isolated from other tests."""
    root = tmp_path / "features"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    return root


def _declare(monkeypatch: pytest.MonkeyPatch, *templates: dict[str, object]) -> None:
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: list(templates))


def _register(dataset_id: str = "districts") -> None:
    feature_services.refresh_feature_collection(
        template={**DISTRICTS_TEMPLATE, "id": dataset_id},
        features={"type": "FeatureCollection", "features": [_box("SL-01")]},
    )


# --- the rewrite itself: pure functions, no store involved -----------------------------------


def test_a_trigger_reference_becomes_a_node_not_geometry() -> None:
    """The whole point: the persisted process graph names the collection, not its polygons."""
    nodes: dict[str, Any] = {}
    resolved = automation_service._resolve_feature_references(
        {"dataset_id": "chirps", "geometries": {"from_features": "districts"}}, nodes
    )

    assert resolved == {"dataset_id": "chirps", "geometries": {"from_node": "features_districts"}}
    assert nodes == {"features_districts": {"process_id": "load_features", "arguments": {"id": "districts"}}}
    assert "coordinates" not in str(resolved) + str(nodes), "geometry must not reach the job record"


def test_the_pinned_node_executes_through_the_real_process_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the executor boundary that normalizes timezone-qualified string arguments."""
    from open_climate_service.openeo.execution import run_process_graph

    _declare(monkeypatch, DISTRICTS_TEMPLATE)
    feature_templates.reset_feature_template_caches()
    _register("districts")
    arguments = {"geometries": {"from_features": "districts"}}
    versions = automation_service._feature_versions(arguments)
    nodes: dict[str, Any] = {}
    automation_service._resolve_feature_references(arguments, nodes, versions)
    nodes["features_districts"]["result"] = True

    result = run_process_graph({"process_graph": nodes})

    assert result["features"][0]["id"] == "SL-01"


def test_the_reference_is_a_node_the_executor_actually_resolves() -> None:
    """An inline `{"process_id": ...}` in an argument is *not* evaluated -- it is passed through.

    The graph parser leaves such a dict untouched, so `aggregate_spatial` would receive it and try
    to read it as GeoJSON. Only a sibling node plus `from_node` becomes a `ResultReference` the
    executor resolves, so this asserts the wiring is that shape rather than the inline one.
    """
    from openeo_pg_parser_networkx import OpenEOProcessGraph
    from openeo_pg_parser_networkx.pg_schema import ResultReference

    nodes: dict[str, Any] = {}
    arguments = automation_service._resolve_feature_references({"geometries": {"from_features": "districts"}}, nodes)
    graph = {"process_graph": {**nodes, "workflow": {"process_id": "add", "arguments": arguments, "result": True}}}

    parsed = OpenEOProcessGraph(pg_data=graph)
    workflow = next(attrs for _, attrs in parsed.nodes if attrs["process_id"] == "add")

    assert isinstance(workflow["resolved_kwargs"]["geometries"], ResultReference)


def test_the_rewritten_node_stays_small() -> None:
    """A level-3 hierarchy is megabytes; the record that replaces it must not grow with it."""
    resolved = automation_service._resolve_feature_references({"geometries": {"from_features": "districts"}})

    assert len(str(resolved)) < 200


def test_a_literal_argument_is_left_alone() -> None:
    inline = {"type": "FeatureCollection", "features": []}
    assert automation_service._resolve_feature_references({"geometries": inline}) == {"geometries": inline}


def test_inline_geojson_may_have_a_property_named_from_features() -> None:
    inline = {
        "type": "FeatureCollection",
        "features": [
            {
                **_box("SL-01"),
                "properties": {"orgUnitCode": "SL-01", "from_features": "source-system-value"},
            }
        ],
    }

    assert automation_service._resolve_feature_references({"geometries": inline}) == {"geometries": inline}
    assert list(automation_service._iter_feature_references({"geometries": inline})) == []


@pytest.mark.parametrize(
    "marker",
    [
        {"from_features": 3},
        {"from_features": ""},
        {"from_features": " districts "},
        {"from_features": "districts", "typo": True},
    ],
)
def test_a_malformed_feature_reference_is_refused(marker: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="from_features"):
        automation_service._resolve_feature_references({"geometries": marker})


def test_multiple_references_to_the_same_collection_share_one_node() -> None:
    nodes: dict[str, Any] = {}
    automation_service._resolve_feature_references(
        {"a": {"from_features": "districts"}, "b": {"from_features": "districts"}}, nodes
    )

    assert nodes == {"features_districts": {"process_id": "load_features", "arguments": {"id": "districts"}}}


# --- provenance: what a job's description records about the collections it ran against --------


def test_the_job_pins_and_records_which_feature_version_it_runs_against() -> None:
    """The graph and description must agree on the exact registered collection version."""
    _register("districts")
    record = feature_services.registered_collections()["districts"]

    versions = automation_service._feature_versions({"geometries": {"from_features": "districts"}})
    provenance = automation_service._feature_provenance(versions)
    nodes: dict[str, Any] = {}
    automation_service._resolve_feature_references(
        {"geometries": {"from_features": "districts"}},
        nodes,
        versions,
    )

    expected = record.created_at.isoformat()
    assert versions == {"districts": expected}
    assert provenance == f" against features districts@{expected}"
    assert nodes["features_districts"]["arguments"]["version"] == expected


def test_an_unregistered_collection_is_refused_before_submission() -> None:
    with pytest.raises(ValueError, match="declared but unregistered"):
        automation_service._feature_versions({"geometries": {"from_features": "districts"}})


def test_feature_versions_skips_registered_collection_lookup_without_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_services,
        "registered_collections",
        lambda: (_ for _ in ()).throw(AssertionError("collection registry should not be loaded")),
    )

    assert automation_service._feature_versions({"dataset_id": "chirps"}) == {}


def test_provenance_is_empty_when_nothing_is_referenced() -> None:
    assert automation_service._feature_provenance({}) == ""


# --- startup validation: an undeclared id is a boot error, not a 3am failure ------------------


def test_startup_skips_the_feature_registry_when_no_trigger_references_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_templates,
        "list_feature_templates",
        lambda: (_ for _ in ()).throw(AssertionError("feature registry should not be loaded")),
    )
    config = AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id="t",
                on_update_of="chirps",
                workflow_id="w",
                arguments={"geometries": {"type": "FeatureCollection", "features": []}},
            )
        ]
    )

    automation_service._validate_feature_references(config)


def test_a_trigger_referencing_an_undeclared_feature_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(monkeypatch, DISTRICTS_TEMPLATE)
    config = AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id="t",
                on_update_of="chirps",
                workflow_id="w",
                arguments={"geometries": {"from_features": "distrcits"}},
            )
        ]
    )

    with pytest.raises(ValueError, match="references feature 'distrcits'.*Declared: districts"):
        automation_service._validate_feature_references(config)


@pytest.mark.parametrize(
    "marker",
    [
        {"from_features": 3},
        {"from_features": "districts", "extra": True},
    ],
)
def test_startup_rejects_a_malformed_feature_reference(
    monkeypatch: pytest.MonkeyPatch,
    marker: dict[str, Any],
) -> None:
    _declare(monkeypatch, DISTRICTS_TEMPLATE)
    config = AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(id="t", on_update_of="chirps", workflow_id="w", arguments={"geometries": marker})
        ]
    )

    with pytest.raises(ValueError, match="Workflow trigger 't'.*invalid feature reference"):
        automation_service._validate_feature_references(config)


def test_a_trigger_referencing_a_declared_feature_passes_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(monkeypatch, DISTRICTS_TEMPLATE)
    config = AutomationConfig(
        workflow_triggers=[
            WorkflowTrigger(
                id="t", on_update_of="chirps", workflow_id="w", arguments={"geometries": {"from_features": "districts"}}
            )
        ]
    )

    automation_service._validate_feature_references(config)  # must not raise


def test_start_rejects_an_undeclared_feature_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same startup-validation door `_validate_event_references` already uses."""
    from open_climate_service.openeo import workflows

    _declare(monkeypatch)  # nothing declared at all
    monkeypatch.setattr(workflows, "get_workflow", lambda _: MagicMock(parameters=[]))
    trigger = WorkflowTrigger(
        id="t", on_update_of="chirps", workflow_id="w", arguments={"geometries": {"from_features": "districts"}}
    )
    service = automation_service.WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=[trigger]), openeo_service=MagicMock()
    )

    with pytest.raises(ValueError, match="references feature 'districts'"):
        service.start()


# --- end to end: a firing trigger submits a graph with a sibling load_features node -----------


def _event(dataset_id: str = "chirps") -> JobEvent:
    from datetime import UTC, datetime

    return JobEvent(
        event_id="native-job:0",
        time=datetime(2026, 8, 19, tzinfo=UTC),
        type=DATASET_UPDATED_EVENT_TYPE,
        source=f"/datasets/{dataset_id}",
        data={"dataset_id": dataset_id},
    )


def test_a_firing_trigger_submits_a_sibling_load_features_node(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    _register("districts")
    openeo = MagicMock()
    openeo.create_triggered_job.return_value = (
        OpenEOJobRecord(
            id="triggered-job",
            process={"process_graph": {}},
            status=OpenEOJobStatus.CREATED,
            created=datetime(2026, 8, 19, tzinfo=UTC),
        ),
        True,
    )
    trigger = WorkflowTrigger(
        id="agg-after-chirps",
        on_update_of="chirps",
        workflow_id="wf",
        arguments={"dataset_id": "$event.dataset_id", "geometries": {"from_features": "districts"}},
    )
    service = automation_service.WorkflowAutomationService(
        config_loader=lambda: AutomationConfig(workflow_triggers=[trigger]), openeo_service=openeo
    )

    service.consume([_event()])

    body = openeo.create_triggered_job.call_args.args[0]
    graph = body.process["process_graph"]
    record = feature_services.registered_collections()["districts"]
    assert graph["features_districts"] == {
        "process_id": "load_features",
        "arguments": {"id": "districts", "version": record.created_at.isoformat()},
    }
    assert graph["workflow"]["arguments"]["geometries"] == {"from_node": "features_districts"}
    assert "coordinates" not in str(graph), "geometry must not reach the persisted job record"
    assert "against features districts@" in body.description
