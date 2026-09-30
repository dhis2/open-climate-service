"""CLIM-1253: moving an instance from datasets/features/downloads to rasters/vectors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_climate_service import config as api_config
from open_climate_service import migrate_layout


@pytest.fixture
def instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An instance on the old layout: data under downloads/ and features/, plugins under datasets/."""
    data = tmp_path / "data"
    (data / "downloads" / "chirps.icechunk").mkdir(parents=True)
    (data / "downloads" / "chirps.icechunk.lock").write_text("")
    (data / "features").mkdir()
    (data / "features" / "districts.abc.parquet").write_bytes(b"PAR1")
    (data / "artifacts").mkdir()
    records = [
        {"artifact_id": "a", "path": "downloads/chirps.icechunk", "asset_paths": ["downloads/chirps.icechunk"]},
        {"artifact_id": "b", "path": str(data / "features" / "districts.abc.parquet"), "asset_paths": []},
        {"artifact_id": "c", "path": "/mnt/elsewhere/downloads/era5.icechunk", "asset_paths": []},
    ]
    (data / "artifacts" / "records.json").write_text(json.dumps(records, indent=2) + "\n")

    plugins = tmp_path / "plugins"
    (plugins / "datasets").mkdir(parents=True)
    (plugins / "datasets" / "chelsa.yaml").write_text(
        "- id: chelsa\n  name: CHELSA\n  variable: tas\n  period_type: monthly\n  sync:\n    kind: static\n"
        "  ingestion:\n    plugin: datasets.chelsa.ChelsaPlugin\n    params: {}\n"
    )
    (plugins / "datasets" / "chelsa.py").write_text(
        "from datasets.helpers import clip\nimport datasets.helpers\n\ndatasets = ['not an import']\n"
    )
    (plugins / "features").mkdir()
    (plugins / "features" / "regions.yaml").write_text("- id: regions\n  name: Regions\n  id_property: id\n")

    config = tmp_path / "climate-service.yaml"
    config.write_text(f"data_dir: {data}\nplugins_dir: ./plugins\n")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config))
    api_config._cache = None
    return tmp_path


def test_moves_data_and_plugins_and_rewrites_what_names_them(instance: Path) -> None:
    assert migrate_layout.main([]) == 0

    data, plugins = instance / "data", instance / "plugins"
    assert sorted(p.name for p in data.iterdir()) == ["artifacts", "rasters", "vectors"]
    assert (data / "rasters" / "chirps.icechunk").is_dir()
    assert (data / "rasters" / "chirps.icechunk.lock").exists(), "a store's sibling files move with it"
    assert (data / "vectors" / "districts.abc.parquet").exists()

    records = json.loads((data / "artifacts" / "records.json").read_text())
    assert records[0]["path"] == "rasters/chirps.icechunk"
    assert records[0]["asset_paths"] == ["rasters/chirps.icechunk"]
    assert records[1]["path"] == str(data / "vectors" / "districts.abc.parquet")
    assert records[2]["path"] == "/mnt/elsewhere/downloads/era5.icechunk", "a store outside the data dir is left alone"

    assert sorted(p.name for p in plugins.iterdir()) == ["rasters", "vectors"]
    assert "plugin: rasters.chelsa.ChelsaPlugin" in (plugins / "rasters" / "chelsa.yaml").read_text()
    module = (plugins / "rasters" / "chelsa.py").read_text()
    assert "from rasters.helpers import clip" in module
    assert "import rasters.helpers" in module
    assert "datasets = ['not an import']" in module, "only imports are rewritten"


def test_running_it_again_changes_nothing(instance: Path, capsys: pytest.CaptureFixture[str]) -> None:
    migrate_layout.main([])
    records = (instance / "data" / "artifacts" / "records.json").read_text()
    capsys.readouterr()

    assert migrate_layout.main([]) == 0
    assert "Nothing to migrate" in capsys.readouterr().out
    assert (instance / "data" / "artifacts" / "records.json").read_text() == records


def test_a_dry_run_reports_and_changes_nothing(instance: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = sorted(str(p.relative_to(instance)) for p in instance.rglob("*"))

    assert migrate_layout.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "would rename" in out and "would rewrite store paths" in out and "would rewrite plugin paths" in out
    assert sorted(str(p.relative_to(instance)) for p in instance.rglob("*")) == before


def test_refuses_when_an_old_and_a_new_folder_both_exist(instance: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (instance / "data" / "rasters").mkdir()

    assert migrate_layout.main([]) == 1

    assert "move their contents by hand" in capsys.readouterr().err
    assert (instance / "data" / "downloads" / "chirps.icechunk").is_dir(), "nothing was moved"
    assert "plugin: datasets." in (instance / "plugins" / "datasets" / "chelsa.yaml").read_text()


def test_the_service_reads_the_migrated_instance(instance: Path) -> None:
    """The point of the move: after it, the loaders find the plugins and the records resolve."""
    from open_climate_service.data_registry.services import datasets as registry
    from open_climate_service.features import templates as feature_templates

    migrate_layout.main([])
    registry.reset_template_caches()

    assert api_config.get_download_root() == instance / "data" / "rasters"
    assert "chelsa" in {t["id"] for t in registry.list_datasets()}
    assert "regions" in {t["id"] for t in feature_templates.list_feature_templates()}
