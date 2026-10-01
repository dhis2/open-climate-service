"""Tests for the startup instance-plugin loading summary."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from open_climate_service import config as api_config
from open_climate_service import plugins_diagnostics as pd


@pytest.fixture
def capture(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    # The "open_climate_service" logger has propagate=False, so attach caplog's
    # handler directly to capture its records.
    original_level = pd.logger.level
    pd.logger.addHandler(caplog.handler)
    pd.logger.setLevel(logging.INFO)
    try:
        yield caplog
    finally:
        pd.logger.removeHandler(caplog.handler)
        pd.logger.setLevel(original_level)


def test_logs_when_plugins_dir_unset(monkeypatch: pytest.MonkeyPatch, capture: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(api_config, "get_config", lambda: {})
    pd.log_plugin_loading()
    assert "no plugins_dir configured" in capture.text


def test_warns_on_missing_subdir_and_counts_files(
    monkeypatch: pytest.MonkeyPatch, capture: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    plugins = tmp_path / "plugins"
    (plugins / "rasters").mkdir(parents=True)
    (plugins / "processes").mkdir()
    (plugins / "rasters" / "x.yaml").write_text("- id: x\n")
    # workflows/ intentionally absent
    config_path = tmp_path / "climate-service.yaml"
    config_path.write_text("plugins_dir: ./plugins/\n")
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": "./plugins/"})
    monkeypatch.setattr(api_config, "get_config_path", lambda: config_path)

    pd.log_plugin_loading()

    text = capture.text
    assert "rasters=1" in text  # counted the one template file
    assert "workflows" in text and "missing" in text  # warned that workflows/ won't load


def test_warns_when_plugins_dir_missing(
    monkeypatch: pytest.MonkeyPatch, capture: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    config_path = tmp_path / "climate-service.yaml"
    config_path.write_text("plugins_dir: ./plugins/\n")
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": "./does-not-exist/"})
    monkeypatch.setattr(api_config, "get_config_path", lambda: config_path)

    pd.log_plugin_loading()

    assert "does not exist or is not a directory" in capture.text
