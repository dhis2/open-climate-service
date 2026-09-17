from pathlib import Path

import pytest

from open_climate_service.config import (
    DEFAULT_CRS,
    DEFAULT_ID,
    DEFAULT_NAME,
    get_config,
    get_crs,
    get_data_dir,
    get_id,
    get_name,
)
from open_climate_service.data_registry.services import datasets as dataset_registry
from open_climate_service.extents import services as extent_services


def test_get_data_dir_returns_none_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    assert get_data_dir() is None


def test_get_data_dir_returns_none_when_config_path_set_but_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(tmp_path / "nonexistent.yaml"))
    assert get_data_dir() is None


def test_get_data_dir_raises_when_config_present_but_no_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("extent:\n  name: Norway\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="data_dir is required"):
        get_data_dir()


def test_get_data_dir_resolves_relative_to_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    assert get_data_dir() == tmp_path / "data"


def test_get_config_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    assert get_config() == {}


def test_get_config_raises_when_path_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(tmp_path / "nonexistent.yaml"))
    with pytest.raises(FileNotFoundError, match="CLIMATE_SERVICE_CONFIG not found"):
        get_config()


def test_get_config_loads_extent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(
        """
extent:
  name: Rwanda
  bbox: [28.8, -2.9, 30.9, -1.0]
  country_code: RWA
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    config = get_config()
    assert config["extent"]["name"] == "Rwanda"
    assert config["extent"]["bbox"] == [28.8, -2.9, 30.9, -1.0]


def test_get_config_substitutes_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(
        """
extent:
  name: ${EXTENT_NAME:-Sierra Leone}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    monkeypatch.delenv("EXTENT_NAME", raising=False)

    config = get_config()
    assert config["extent"]["name"] == "Sierra Leone"


def test_extents_service_reads_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(
        """
extent:
  name: Rwanda
  bbox: [28.8, -2.9, 30.9, -1.0]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    extent = extent_services.get_extent()
    assert extent is not None
    assert extent["name"] == "Rwanda"


def test_extent_service_returns_none_when_config_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)

    assert extent_services.get_extent() is None


def test_builtin_datasets_include_chirps_era5_worldpop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)

    ids = {d["id"] for d in dataset_registry.list_datasets()}
    assert "chirps3_precipitation_daily" in ids
    assert "era5land_temperature_daily" in ids
    assert "era5land_precipitation_monthly" in ids
    assert "worldpop_population_global2_100m" in ids


def test_get_crs_defaults_to_wgs84(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    assert get_crs() == DEFAULT_CRS


def test_get_crs_reads_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\ncrs: EPSG:25833\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    assert get_crs() == "EPSG:25833"


def test_get_crs_raises_for_invalid_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\ncrs: 42\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="crs in CLIMATE_SERVICE_CONFIG must be a non-empty string"):
        get_crs()


def test_get_crs_raises_for_unknown_epsg_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\ncrs: EPSG:999999\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="not a valid CRS"):
        get_crs()


def test_get_id_defaults_to_open_climate_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    assert get_id() == DEFAULT_ID


def test_get_id_reads_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nid: kenya-climate-service\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    assert get_id() == "kenya-climate-service"


def test_get_id_raises_for_non_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nid: 42\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="id in CLIMATE_SERVICE_CONFIG must be a non-empty string"):
        get_id()


def test_get_id_raises_for_blank_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nid: '   '\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="id in CLIMATE_SERVICE_CONFIG must be a non-empty string"):
        get_id()


def test_get_name_defaults_to_open_climate_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIMATE_SERVICE_CONFIG", raising=False)
    assert get_name() == DEFAULT_NAME


def test_get_name_reads_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nname: Nepal Climate Service\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    assert get_name() == "Nepal Climate Service"


def test_get_name_trims_whitespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nname: '  My Service  '\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    assert get_name() == "My Service"


def test_get_name_raises_for_non_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nname: 42\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="name in CLIMATE_SERVICE_CONFIG must be a non-empty string"):
        get_name()


def test_get_name_raises_for_blank_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text("data_dir: ./data\nname: '   '\n", encoding="utf-8")
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    with pytest.raises(ValueError, match="name in CLIMATE_SERVICE_CONFIG must be a non-empty string"):
        get_name()


def test_templates_dir_raises_with_migration_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"templates_dir: {tmp_path / 'templates'}\n", encoding="utf-8")
    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    with pytest.raises(ValueError, match="templates_dir"):
        dataset_registry.list_datasets()


def test_plugins_dir_adds_root_to_sys_path_and_makes_modules_importable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugins_dir = tmp_path / "plugins"
    pkg_dir = plugins_dir / "myplugin"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "source.py").write_text("VALUE = 42\n", encoding="utf-8")
    (plugins_dir / "datasets").mkdir()
    (plugins_dir / "datasets" / "custom.yaml").write_text(
        """
- id: plugin_dataset
  name: Plugin dataset
  variable: val
  period_type: daily
  sync:
    kind: static
  ingestion:
    plugin: myplugin.source.Plugin
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")
    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))
    root_str = str(plugins_dir.resolve())
    monkeypatch.syspath_prepend("")  # ensure clean sys.path state without the plugins root

    dataset_registry.list_datasets()

    import sys

    assert root_str in sys.path
    import importlib

    mod = importlib.import_module("myplugin.source")
    assert mod.VALUE == 42


def test_plugins_dir_in_config_adds_to_bundled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    datasets_subdir = tmp_path / "plugins" / "datasets"
    datasets_subdir.mkdir(parents=True)
    (datasets_subdir / "custom.yaml").write_text(
        """
- id: custom_dataset
  name: Custom dataset
  variable: val
  period_type: daily
  sync:
    kind: static
  ingestion:
    plugin: mypackage.sources.Plugin
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {tmp_path / 'plugins'}\n", encoding="utf-8")

    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    ids = {d["id"] for d in dataset_registry.list_datasets()}
    assert "custom_dataset" in ids
    assert "chirps3_precipitation_daily" in ids


def test_plugins_dir_resolved_relative_to_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """plugins_dir is resolved relative to the config file, not CWD.

    This matters when running the installed `climate-service` CLI from a directory
    other than the repo root, where a relative plugins_dir in the config must
    still point at the correct sibling directory.
    """
    deployment_dir = tmp_path / "deployment"
    datasets_subdir = deployment_dir / "plugins" / "datasets"
    datasets_subdir.mkdir(parents=True)
    (datasets_subdir / "custom.yaml").write_text(
        """
- id: deployed_dataset
  variable: val
  period_type: daily
  sync:
    kind: static
  ingestion:
    plugin: mypackage.sources.Plugin
""",
        encoding="utf-8",
    )
    config_file = deployment_dir / "climate-service.yaml"
    config_file.write_text("plugins_dir: ./plugins\n", encoding="utf-8")

    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    ids = {d["id"] for d in dataset_registry.list_datasets()}
    assert "deployed_dataset" in ids


def test_plugins_dir_in_config_overrides_bundled_by_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    datasets_subdir = tmp_path / "plugins" / "datasets"
    datasets_subdir.mkdir(parents=True)
    (datasets_subdir / "chirps3.yaml").write_text(
        """
- id: chirps3_precipitation_daily
  name: Custom CHIRPS override
  variable: precip
  period_type: daily
  sync:
    kind: static
  ingestion:
    plugin: mypackage.sources.Plugin
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {tmp_path / 'plugins'}\n", encoding="utf-8")

    monkeypatch.setattr(dataset_registry, "CONFIGS_DIR", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    datasets = {d["id"]: d for d in dataset_registry.list_datasets()}
    assert datasets["chirps3_precipitation_daily"]["name"] == "Custom CHIRPS override"
