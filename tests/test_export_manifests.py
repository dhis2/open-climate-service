"""Payload integrity, frozen bindings, observed provenance, and result retention."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import xarray as xr
from fastapi import HTTPException

from open_climate_service import config
from open_climate_service.exports.delivery_input import lease_export_input
from open_climate_service.exports.manifest import ExportManifest
from open_climate_service.exports.service import write_named_export
from open_climate_service.openeo import jobs
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus, OpenEOJobUpdate
from open_climate_service.shared.provenance import (
    capture_execution,
    json_digest,
    record_features,
    record_snapshot,
    record_source,
)
from open_climate_service.shared.time import utc_now


def _data() -> pd.DataFrame:
    return pd.DataFrame({"geometry": ["DiszpKrYNg8"] * 2, "t": ["202501", "202502"], "rain": [0.0, float("nan")]})


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(jobs, "_JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "exports": [
                {
                    "id": "rain",
                    "plugin": "dhis2",
                    "connection": "hmis",
                    "period_type": "monthly",
                    "series": [{"data_element": "BXgDHhPdFVU"}],
                }
            ],
            "dhis2_connections": [{"id": "hmis", "url": "https://hmis.example.org/dhis", "token_env": "TEST_TOKEN"}],
        },
    )
    monkeypatch.delenv("TEST_TOKEN", raising=False)
    directory = tmp_path / "jobs" / "source" / "results"
    directory.mkdir(parents=True)
    path = write_named_export(_data(), directory, "DHIS2JSON", {"export": "rain"}, job_id="source")
    jobs.store_create_job(
        OpenEOJobRecord(
            id="source",
            status=OpenEOJobStatus.FINISHED,
            created=utc_now(),
            usage={"output_path": path},
        )
    )
    return Path(path)


def _manifest(path: Path) -> ExportManifest:
    metadata = json.loads((path.parent / ".export.json").read_text())
    return ExportManifest.model_validate_json((path.parent / metadata["manifest"]).read_bytes())


def test_manifest_records_exact_bytes_counts_and_missing_evidence(saved: Path):
    manifest = _manifest(saved)
    assert manifest.source_job_id == "source"
    assert manifest.payload_sha256 == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert manifest.payload_size == saved.stat().st_size
    assert (manifest.record_count, manifest.skipped_count) == (1, 1)
    assert manifest.periods == ["202501"]
    assert manifest.provenance["missing"] == ["execution_provenance"]
    assert manifest.plugin.version == "1"
    with lease_export_input("rain", "source") as verified:
        assert verified.content == saved.read_bytes()
        assert verified.manifest == manifest


def test_rotation_and_url_normalization_do_not_invalidate_binding(saved: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_TOKEN", "rotated-secret")
    config.get_config()["dhis2_connections"][0]["url"] = "https://HMIS.EXAMPLE.ORG:443/dhis/"
    with lease_export_input("rain", "source") as verified:
        serialized = verified.manifest.model_dump_json()
        assert "rotated-secret" not in serialized
        assert "TEST_TOKEN" not in serialized
        assert "https://" not in serialized


@pytest.mark.parametrize("change", ["mapping", "target", "connection", "version", "reference", "process"])
def test_changed_bindings_are_conflicts(saved: Path, monkeypatch: pytest.MonkeyPatch, change: str):
    definition = config.get_config()["exports"][0]
    if change == "mapping":
        definition["series"][0]["data_element"] = "Ix2HsbDMLea"
    elif change == "target":
        config.get_config()["dhis2_connections"][0]["url"] = "https://another.example.org"
    elif change == "connection":
        definition["connection"] = "other"
    elif change == "reference":
        definition["dataset"] = "other"
    elif change == "version":
        monkeypatch.setattr("open_climate_service.exports.dhis2_renderer.Dhis2ExportPlugin.version", "2")
    else:
        # Construct a fresh manifest with the original graph digest, then change the job graph.
        path = write_named_export(
            _data(),
            saved.parent,
            "DHIS2JSON",
            {"export": "rain"},
            job_id="source",
            provenance={"process_sha256": json_digest({}), "sources": []},
        )
        jobs.store_update_job(
            "source",
            lambda record: record.model_copy(
                update={
                    "usage": {"output_path": path},
                    "process": {"process_graph": {}},
                }
            ),
        )
    with pytest.raises(HTTPException) as error:
        with lease_export_input("rain", "source"):
            pytest.fail("Changed export must not be accepted")
    assert error.value.status_code == 409


@pytest.mark.parametrize(
    "change",
    ["payload", "manifest", "metadata", "missing", "legacy", "path", "version", "corrupt"],
)
def test_corrupt_missing_and_legacy_assets_are_rejected(saved: Path, change: str):
    metadata_path = saved.parent / ".export.json"
    metadata = json.loads(metadata_path.read_text())
    if change == "payload":
        saved.write_bytes(saved.read_bytes() + b" ")
    elif change == "corrupt":
        metadata_path.write_text("{ this is not valid json", encoding="utf-8")
    elif change == "missing":
        saved.unlink()
    elif change == "legacy":
        del metadata["manifest"]
        metadata_path.write_text(json.dumps(metadata))
    elif change == "path":
        metadata["manifest"] = "../../outside.json"
        metadata_path.write_text(json.dumps(metadata))
    elif change == "metadata":
        metadata["record_count"] += 1
        metadata_path.write_text(json.dumps(metadata))
    else:
        path = saved.parent / metadata["manifest"]
        if change == "manifest":
            path.write_bytes(path.read_bytes() + b" ")
        else:
            value = json.loads(path.read_text())
            value["schema_version"] = 99
            path.write_text(json.dumps(value))
            metadata["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(HTTPException) as error:
        with lease_export_input("rain", "source"):
            pytest.fail("Invalid export must not be accepted")
    assert error.value.status_code == 409


@pytest.mark.parametrize("status", [OpenEOJobStatus.RUNNING, OpenEOJobStatus.ERROR, OpenEOJobStatus.CANCELED])
def test_only_finished_jobs_are_eligible(saved: Path, status: OpenEOJobStatus):
    jobs.store_update_job("source", lambda record: record.model_copy(update={"status": status}))
    with pytest.raises(HTTPException, match="409"):
        with lease_export_input("rain", "source"):
            pytest.fail("Incomplete job must not be accepted")


def test_wrong_export_and_missing_job(saved: Path):
    for export, job_id, status in [("other", "source", 409), ("rain", "missing", 404)]:
        with pytest.raises(HTTPException) as error:
            with lease_export_input(export, job_id):
                pytest.fail("Wrong source binding must not be accepted")
        assert error.value.status_code == status


def test_lease_blocks_mutation_and_releases_after_exception(saved: Path):
    service = jobs.OpenEOJobService()
    try:
        with pytest.raises(RuntimeError, match="consumer failed"):
            with lease_export_input("rain", "source"):
                for operation in (
                    lambda: service.delete_job("source"),
                    lambda: service.start_job("source"),
                    lambda: service.update_job("source", OpenEOJobUpdate(title="changed")),
                ):
                    with pytest.raises(HTTPException) as error:
                        operation()
                    assert error.value.status_code == 409
                raise RuntimeError("consumer failed")
        service.delete_job("source")
        assert not saved.exists()
    finally:
        service.shutdown()


def test_failed_publication_keeps_previous_generation(saved: Path, monkeypatch: pytest.MonkeyPatch):
    from open_climate_service.exports import manifest as module

    original = module.atomic_write
    old_metadata = (saved.parent / ".export.json").read_bytes()

    def fail_manifest(path: Path, content: bytes) -> None:
        if path.name.endswith(".manifest.json"):
            raise OSError("simulated write failure")
        original(path, content)

    monkeypatch.setattr(module, "atomic_write", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        write_named_export(_data(), saved.parent, "DHIS2JSON", {"export": "rain"}, job_id="source")
    assert (saved.parent / ".export.json").read_bytes() == old_metadata
    with lease_export_input("rain", "source") as verified:
        assert verified.content == saved.read_bytes()


def test_unbound_export_remains_downloadable_but_cannot_be_delivered(saved: Path):
    del config.get_config()["exports"][0]["connection"]
    path = write_named_export(_data(), saved.parent, "DHIS2JSON", {"export": "rain"}, job_id="source")
    jobs.store_update_job("source", lambda record: record.model_copy(update={"usage": {"output_path": path}}))
    assert _manifest(Path(path)).target is None
    with pytest.raises(HTTPException, match="without a delivery target"):
        with lease_export_input("rain", "source"):
            pytest.fail("Unbound export must not be accepted")


def test_feature_and_snapshot_evidence_is_execution_scoped():
    feature = {
        "type": "Feature",
        "id": "DiszpKrYNg8",
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "properties": {"secret": "never copied"},
    }
    artifact = SimpleNamespace(path="/private/path", artifact_id="artifact-1", source_dataset_id="rain")
    with capture_execution({"process_graph": {}}) as outer:
        record_snapshot(artifact.path, "actual-readonly-snapshot")
        record_source("managed-rain", artifact)
        record_features(feature)
        with capture_execution({}) as inner:
            record_features({**feature, "id": None})
        assert inner.describe()["sources"] == []
        assert inner.describe()["features"][0]["ids_valid"] is False
        evidence = outer.describe()
    assert evidence["sources"][0]["snapshot_id"] == "actual-readonly-snapshot"
    assert evidence["features"][0]["ids_valid"] is True
    assert "/private/path" not in json.dumps(evidence)
    assert "never copied" not in json.dumps(evidence)


def test_run_graph_attaches_native_observations(monkeypatch: pytest.MonkeyPatch):
    from open_climate_service.openeo import execution
    from open_climate_service.plugins.processes.aggregate_spatial import _parse_geometries

    artifact = SimpleNamespace(path="/source", artifact_id="artifact-1", source_dataset_id="rain")
    monkeypatch.setattr(execution, "_get_published_artifact", lambda _: artifact)
    monkeypatch.setattr(execution, "_open_artifact", lambda _: xr.Dataset({"rain": ("t", [1])}))
    monkeypatch.setattr(execution, "_ensure_crs", lambda data: data)
    monkeypatch.setattr(execution, "_build_process_registry", lambda: {})
    monkeypatch.setattr(execution, "_augment_with_workflows", lambda registry: registry)

    class Graph:
        def __init__(self: Any, graph: Any):
            pass

        def to_callable(self: Any, registry: Any):
            def execute():
                execution._load_collection_impl("managed-rain")
                _parse_geometries(
                    {"type": "Feature", "id": "DiszpKrYNg8", "geometry": {"type": "Point", "coordinates": [0, 0]}}
                )
                return execution.SaveResultEnvelope(_data(), "DHIS2JSON", {"export": "rain"})

            return execute

    monkeypatch.setattr("openeo_pg_parser_networkx.OpenEOProcessGraph", Graph)
    result = execution.run_process_graph({"process_graph": {}})
    assert result.provenance["sources"][0]["artifact_id"] == "artifact-1"
    assert result.provenance["features"][0]["input_sha256"]


def test_export_environment_interpolation_rejected_before_caching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("exports:\n  - id: rain\n    plugin: dhis2\n    data_element: ${SECRET}\n")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(path))
    monkeypatch.setenv("SECRET", "secret: [invalid yaml")
    with pytest.raises(ValueError, match="literal"):
        config.get_config()
    assert config._cache is None


def test_snapshot_paths_are_normalized(tmp_path: Path):
    artifact = SimpleNamespace(path=str(tmp_path) + "/./store/", artifact_id="a", source_dataset_id="rain")
    with capture_execution({}) as evidence:
        record_snapshot(str(tmp_path / "store"), "snapshot")
        record_source("rain", artifact)
    assert evidence.describe()["sources"][0]["snapshot_id"] == "snapshot"


@pytest.mark.parametrize("ids", [[None], ["same", "same"]])
def test_named_dhis2_graph_validates_original_feature_ids(ids: list[str | None]):
    from open_climate_service.plugins.processes.aggregate_spatial import _parse_geometries

    process = {
        "process_graph": {
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "format": "DHIS2JSON",
                    "options": {"export": "rain"},
                },
            }
        }
    }
    features = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": value, "geometry": {"type": "Point", "coordinates": [0, 0]}} for value in ids
        ],
    }
    with capture_execution(process), pytest.raises(ValueError, match="Feature .*feature.id"):
        _parse_geometries(features)
