"""Portability of the artifact index across data directories (CLIM-916)."""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from open_climate_service import config as api_config
from open_climate_service.ingestions import services
from open_climate_service.ingestions.artifact_paths import to_absolute, to_portable
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)


def _use_data_dir(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> Path:
    """Point the instance config at data_dir and return it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config_file = data_dir.parent / "climate-service.yaml"
    config_file.write_text(
        f"data_dir: {data_dir}\nextent:\n  id: test\n  bbox: [0, 0, 1, 1]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    api_config._cache = None
    return data_dir


def _artifact(path: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="a1",
        dataset_id="chirps3_precipitation_daily",
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        format=ArtifactFormat.ICECHUNK,
        path=path,
        asset_paths=[path],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-10", bbox=(1.0, 2.0, 3.0, 4.0)),
        coverage=ArtifactCoverage(
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-10"),
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
        ),
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id="chirps3"),
    )


def test_to_portable_strips_the_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")

    assert to_portable(str(data_dir / "downloads" / "chirps3.icechunk")) == "downloads/chirps3.icechunk"


def test_to_portable_keeps_paths_outside_the_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_data_dir(monkeypatch, tmp_path / "data")
    external = str(tmp_path / "elsewhere" / "chirps3.icechunk")

    assert to_portable(external) == external


def test_to_absolute_resolves_against_the_current_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")

    assert to_absolute("downloads/chirps3.icechunk") == str(data_dir / "downloads" / "chirps3.icechunk")


def test_to_absolute_rebases_a_legacy_container_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A record written inside Docker at /app/data resolves against the host data dir."""
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    store = data_dir / "downloads" / "chirps3.icechunk"
    store.mkdir(parents=True)

    assert to_absolute("/app/data/downloads/chirps3.icechunk") == str(store)


def test_to_absolute_prefers_a_store_under_the_current_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The current root wins even when the recorded location still exists.

    Only data_dir/downloads is a trusted root for managed stores, so a path elsewhere
    is not worth preserving over one here - and deferring to it breaks copied data
    directories (see test_copied_data_dir_does_not_read_the_original).
    """
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    local = data_dir / "downloads" / "chirps3.icechunk"
    local.mkdir(parents=True)
    external = tmp_path / "mnt" / "downloads" / "chirps3.icechunk"
    external.mkdir(parents=True)

    assert to_absolute(str(external)) == str(local)


def test_to_absolute_keeps_an_external_store_with_no_counterpart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With nothing matching under the root, the recorded path is left alone."""
    _use_data_dir(monkeypatch, tmp_path / "data")
    external = tmp_path / "mnt" / "downloads" / "era5.icechunk"
    external.mkdir(parents=True)

    assert to_absolute(str(external)) == str(external)


def test_copied_data_dir_does_not_read_the_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A copied data dir resolves against its own root while the original still exists."""
    origin = _use_data_dir(monkeypatch, tmp_path / "instance-a" / "data")
    (origin / "downloads" / "chirps3.icechunk").mkdir(parents=True)
    index_path = origin / "artifacts" / "records.json"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", index_path.parent)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", index_path)
    services._save_records([_artifact(str(origin / "downloads" / "chirps3.icechunk"))])
    # legacy state: the index carries absolute paths rather than the portable form
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace(
            '"downloads/chirps3.icechunk"', f'"{origin / "downloads" / "chirps3.icechunk"}"'
        ),
        encoding="utf-8",
    )

    copy = tmp_path / "instance-b" / "data"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origin, copy)
    assert (origin / "downloads" / "chirps3.icechunk").exists()  # the original is still there
    _use_data_dir(monkeypatch, copy)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", copy / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", copy / "artifacts" / "records.json")

    loaded = services._load_records()
    assert loaded[0].path == str(copy / "downloads" / "chirps3.icechunk")

    services._save_records(loaded)
    on_disk = json.loads((copy / "artifacts" / "records.json").read_text(encoding="utf-8"))
    assert on_disk[0]["path"] == "downloads/chirps3.icechunk"  # and it heals


def test_shadowing_an_existing_recorded_store_is_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Preferring the current root over a store that still exists is not silent."""
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    (data_dir / "downloads" / "chirps3.icechunk").mkdir(parents=True)
    external = tmp_path / "mnt" / "downloads" / "chirps3.icechunk"
    external.mkdir(parents=True)
    # startup.py detaches the package logger from the root handler caplog installs
    monkeypatch.setattr(logging.getLogger("open_climate_service"), "propagate", True)

    with caplog.at_level("WARNING"):
        to_absolute(str(external))

    assert str(external) in caplog.text


def test_to_absolute_ignores_a_lookalike_outside_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A deeper suffix match loses to the managed store at data_dir/downloads."""
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    managed = data_dir / "downloads" / "chirps3.icechunk"
    managed.mkdir(parents=True)
    lookalike = data_dir / "data" / "downloads" / "chirps3.icechunk"
    lookalike.mkdir(parents=True)

    assert to_absolute("/old/data/downloads/chirps3.icechunk") == str(managed)
    assert to_portable(to_absolute("/old/data/downloads/chirps3.icechunk")) == "downloads/chirps3.icechunk"


def test_to_absolute_will_not_rebase_onto_a_lookalike_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With only a match outside downloads, the recorded path is kept rather than bound to it."""
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    (data_dir / "data" / "downloads" / "chirps3.icechunk").mkdir(parents=True)
    legacy = "/old/data/downloads/chirps3.icechunk"

    assert to_absolute(legacy) == legacy


def test_to_absolute_will_not_rebase_on_a_bare_basename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A match must agree on the containing directory, not just the store name."""
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    (data_dir / "chirps3.icechunk").mkdir(parents=True)
    archived = "/app/data/downloads/v1/chirps3.icechunk"

    assert to_absolute(archived) == archived


def test_to_absolute_keeps_an_unresolvable_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With nothing to rebase onto, the recorded path survives so errors name it."""
    _use_data_dir(monkeypatch, tmp_path / "data")

    assert to_absolute("/app/data/downloads/missing.icechunk") == "/app/data/downloads/missing.icechunk"


def test_records_are_persisted_relative_and_loaded_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = _use_data_dir(monkeypatch, tmp_path / "data")
    index_path = data_dir / "artifacts" / "records.json"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", index_path.parent)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", index_path)
    store = data_dir / "downloads" / "chirps3.icechunk"

    services._save_records([_artifact(str(store))])

    on_disk = json.loads(index_path.read_text(encoding="utf-8"))
    assert on_disk[0]["path"] == "downloads/chirps3.icechunk"
    assert on_disk[0]["asset_paths"] == ["downloads/chirps3.icechunk"]

    loaded = services._load_records()
    assert loaded[0].path == str(store)
    assert loaded[0].asset_paths == [str(store)]


def test_data_dir_survives_being_moved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The whole point: an index written under one root reads correctly under another."""
    origin = _use_data_dir(monkeypatch, tmp_path / "origin" / "data")
    index_path = origin / "artifacts" / "records.json"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", index_path.parent)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", index_path)
    services._save_records([_artifact(str(origin / "downloads" / "chirps3.icechunk"))])

    moved = tmp_path / "moved" / "data"
    moved.parent.mkdir(parents=True, exist_ok=True)
    origin.rename(moved)
    _use_data_dir(monkeypatch, moved)  # data_dir already exists; the helper only rewrites config
    monkeypatch.setattr(services, "ARTIFACTS_DIR", moved / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", moved / "artifacts" / "records.json")

    assert services._load_records()[0].path == str(moved / "downloads" / "chirps3.icechunk")
