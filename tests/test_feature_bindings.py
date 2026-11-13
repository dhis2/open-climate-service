"""Binding a workflow argument to a registered feature collection (CLIM-926, step 7).

`{"from_features": "districts"}` must become something the process graph can resolve, without
putting geometry into the persisted job record. These cover both halves of that: the rewrite
produces a node the executor evaluates, and the node is pinned to the collection version the
job was submitted against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from open_climate_service import config as api_config
from open_climate_service.features import bindings
from open_climate_service.features import services as feature_services
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions import services as ingestion_services

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
    root = tmp_path / "vectors"
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


def test_a_reference_becomes_a_node_not_geometry() -> None:
    """The whole point: the persisted process graph names the collection, not its polygons."""
    nodes: dict[str, Any] = {}
    resolved = bindings.resolve_feature_references(
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
    versions = bindings.feature_versions(arguments)
    nodes: dict[str, Any] = {}
    bindings.resolve_feature_references(arguments, nodes, versions)
    nodes["features_districts"]["result"] = True

    result = run_process_graph({"process_graph": nodes})

    assert result["features"][0]["id"] == "SL-01"


def test_the_reference_is_a_node_the_executor_actually_resolves() -> None:
    """An inline `{"process_id": ...}` in an argument is *not* evaluated -- it is passed through.

    The graph parser leaves such dictionaries untouched. A reference only becomes a ResultReference
    when it is represented as a sibling node with from_node, which the executor can then resolve.
    This verifies that resolve_feature_references produces the expected graph wiring.
    """
    from openeo_pg_parser_networkx import OpenEOProcessGraph
    from openeo_pg_parser_networkx.pg_schema import ResultReference

    nodes: dict[str, Any] = {}
    arguments = bindings.resolve_feature_references({"geometries": {"from_features": "districts"}}, nodes)
    graph = {"process_graph": {**nodes, "workflow": {"process_id": "add", "arguments": arguments, "result": True}}}

    parsed = OpenEOProcessGraph(pg_data=graph)
    workflow = next(attrs for _, attrs in parsed.nodes if attrs["process_id"] == "add")

    assert isinstance(workflow["resolved_kwargs"]["geometries"], ResultReference)


def test_the_rewritten_node_stays_small() -> None:
    """A level-3 hierarchy is megabytes; the record that replaces it must not grow with it."""
    resolved = bindings.resolve_feature_references({"geometries": {"from_features": "districts"}})

    assert len(str(resolved)) < 200


def test_a_literal_argument_is_left_alone() -> None:
    inline = {"type": "FeatureCollection", "features": []}
    assert bindings.resolve_feature_references({"geometries": inline}) == {"geometries": inline}


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

    assert bindings.resolve_feature_references({"geometries": inline}) == {"geometries": inline}
    assert list(bindings.iter_feature_references({"geometries": inline})) == []


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
        bindings.resolve_feature_references({"geometries": marker})


def test_multiple_references_to_the_same_collection_share_one_node() -> None:
    nodes: dict[str, Any] = {}
    bindings.resolve_feature_references(
        {"a": {"from_features": "districts"}, "b": {"from_features": "districts"}}, nodes
    )

    assert nodes == {"features_districts": {"process_id": "load_features", "arguments": {"id": "districts"}}}


# --- pinning: which collection version a job runs against ------------------------------------


def test_the_job_pins_the_feature_version_it_runs_against() -> None:
    _register("districts")
    record = feature_services.registered_collections()["districts"]

    versions = bindings.feature_versions({"geometries": {"from_features": "districts"}})
    nodes: dict[str, Any] = {}
    bindings.resolve_feature_references(
        {"geometries": {"from_features": "districts"}},
        nodes,
        versions,
    )

    expected = record.created_at.isoformat()
    assert versions == {"districts": expected}
    assert nodes["features_districts"]["arguments"]["version"] == expected


def test_an_unregistered_collection_is_refused_before_submission() -> None:
    with pytest.raises(ValueError, match="declared but unregistered"):
        bindings.feature_versions({"geometries": {"from_features": "districts"}})


def test_feature_versions_skips_registered_collection_lookup_without_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_services,
        "registered_collections",
        lambda: (_ for _ in ()).throw(AssertionError("collection registry should not be loaded")),
    )

    assert bindings.feature_versions({"dataset_id": "chirps"}) == {}
