"""Entry-point plugin discovery for processes and workflows (#118).

Dataset-template discovery is covered in test_dataset_registry.py; these exercise
the other two extension points a plugin package can ship.
"""

import importlib.resources
from pathlib import Path

import pytest

from open_climate_service import plugin_discovery
from open_climate_service.openeo import plugin_processes, workflows


def test_entry_point_processes_are_discovered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A tiny importable plugin package that ships a @process in processes/.
    pkg = tmp_path / "fakeproc_plugin"
    (pkg / "processes").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "processes" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "processes" / "myproc.py").write_text(
        "from open_climate_service.process import process\n\n"
        "@process\n"
        "def my_plugin_process(data):\n"
        '    """A test plugin process."""\n'
        "    return data\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def fake_iter(subdir: str):
        if subdir == "processes":
            yield "fakeproc", "fakeproc_plugin", importlib.resources.files("fakeproc_plugin") / "processes"

    monkeypatch.setattr(plugin_discovery, "iter_plugin_subdirs", fake_iter)

    ids = [pid for pid, _ in plugin_processes.load_plugin_processes()]
    assert "my_plugin_process" in ids


def test_entry_point_workflows_are_discovered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "wf.json").write_text('{"id": "plugin_workflow", "process_graph": {}}', encoding="utf-8")

    def fake_iter(subdir: str):
        if subdir == "workflows":
            yield "fake", "fake", workflows_dir

    monkeypatch.setattr(plugin_discovery, "iter_plugin_subdirs", fake_iter)

    assert [w["id"] for w in workflows._load_entry_point_workflows()] == ["plugin_workflow"]


def test_no_installed_plugins_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no plugins declaring the entry point, discovery is a clean no-op.
    scans: list[str] = []
    monkeypatch.setattr(plugin_discovery, "entry_points", lambda group: scans.append(group) or [])
    plugin_discovery.rescan()
    assert list(plugin_discovery.iter_plugin_subdirs("processes")) == []
    assert workflows._load_entry_point_workflows() == []
    assert plugin_processes._scan_plugin_package_processes() == []
    # Read once per process, however many lookups follow.
    assert len(scans) == 1
