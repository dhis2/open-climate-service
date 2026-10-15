"""The suite never writes into a developer's instance (CLIM-1395).

Store paths are resolved from the instance config when their module is imported, so conftest
sets the suite's own config before importing anything. This checks that it worked for every
module, including one added later with a store path of its own.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path

import pytest

import open_climate_service

from . import conftest
from .conftest import TEST_DATA_ROOT


def _import_every_module() -> None:
    # No import error is skipped: the suite runs with every extra installed, so a module that
    # fails to import is broken, and skipping it would leave its store paths unchecked.
    for module in pkgutil.walk_packages(open_climate_service.__path__, prefix="open_climate_service."):
        importlib.import_module(module.name)


def test_every_store_path_resolved_at_import_is_inside_the_test_instance() -> None:
    _import_every_module()
    package = Path(open_climate_service.__file__).parent

    outside = [
        f"{name}.{attr} = {value}"
        for name, module in sorted(sys.modules.items())
        if name == "open_climate_service" or name.startswith("open_climate_service.")
        for attr, value in vars(module).items()
        if isinstance(value, Path)
        and value.is_absolute()
        # Paths into the package's own files (templates, built-in plugins) are not data.
        and not value.resolve().is_relative_to(package)
        and not value.resolve().is_relative_to(TEST_DATA_ROOT.resolve())
    ]

    assert outside == []


def test_the_suite_runs_against_its_own_instance() -> None:
    from open_climate_service import config as api_config

    assert api_config.get_data_root() == TEST_DATA_ROOT.resolve()


def test_a_session_gives_the_caller_its_config_back_and_the_next_session_takes_it_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Restores the real environment afterwards, whatever the hooks do to it.
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", os.environ["CLIMATE_SERVICE_CONFIG"])
    monkeypatch.setattr(conftest, "_CALLERS_CONFIG", "/home/dev/my-instance/climate-service.yaml")

    conftest.pytest_sessionfinish(session=None, exitstatus=0)  # type: ignore[arg-type]
    assert os.environ["CLIMATE_SERVICE_CONFIG"] == "/home/dev/my-instance/climate-service.yaml"

    conftest.TEST_CONFIG_FILE.unlink()  # as if the instance were gone by the next session
    conftest.pytest_sessionstart(session=None)  # type: ignore[arg-type]
    assert os.environ["CLIMATE_SERVICE_CONFIG"] == str(conftest.TEST_CONFIG_FILE)
    assert conftest.TEST_CONFIG_FILE.is_file()


def test_a_caller_without_a_config_is_left_without_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", os.environ["CLIMATE_SERVICE_CONFIG"])
    monkeypatch.setattr(conftest, "_CALLERS_CONFIG", None)

    conftest.pytest_sessionfinish(session=None, exitstatus=0)  # type: ignore[arg-type]

    assert "CLIMATE_SERVICE_CONFIG" not in os.environ
