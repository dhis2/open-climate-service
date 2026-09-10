from pathlib import Path

import pytest

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets


def test_dataset_registry_requires_sync_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry_file = tmp_path / "missing_sync_kind.yaml"
    registry_file.write_text(
        """
- id: missing_sync_kind
  name: Missing sync kind
  variable: value
  period_type: daily
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    with pytest.raises(ValueError, match="must define sync.kind"):
        datasets.list_datasets()


def test_dataset_registry_rejects_unsupported_sync_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "invalid_sync_kind.yaml"
    registry_file.write_text(
        """
- id: invalid_sync_kind
  name: Invalid sync kind
  variable: value
  period_type: daily
  sync:
    kind: sometimes
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    with pytest.raises(ValueError, match="unsupported sync.kind 'sometimes'"):
        datasets.list_datasets()


def test_dataset_registry_accepts_supported_sync_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "valid.yaml"
    registry_file.write_text(
        """
- id: valid_temporal
  name: Valid temporal
  variable: value
  period_type: daily
  sync:
    kind: temporal
  ingestion:
    plugin: open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    assert datasets.list_datasets()[0]["id"] == "valid_temporal"


def test_dataset_registry_accepts_ingestion_plugin_without_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "valid_plugin.yaml"
    registry_file.write_text(
        """
- id: valid_plugin
  name: Valid plugin
  variable: value
  period_type: daily
  sync:
    kind: temporal
  ingestion:
    plugin: open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    assert datasets.list_datasets()[0]["ingestion"]["plugin"].endswith("CHIRPS3DailyPlugin")


def test_dataset_registry_rejects_unsupported_sync_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "invalid_sync_execution.yaml"
    registry_file.write_text(
        """
- id: invalid_sync_execution
  name: Invalid sync execution
  variable: value
  period_type: daily
  sync:
    kind: temporal
    execution: sometimes
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    with pytest.raises(ValueError, match="unsupported sync.execution 'sometimes'"):
        datasets.list_datasets()


def test_dataset_registry_rejects_non_string_sync_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "invalid_sync_execution_type.yaml"
    registry_file.write_text(
        """
- id: invalid_sync_execution_type
  name: Invalid sync execution type
  variable: value
  period_type: daily
  sync:
    kind: temporal
    execution:
      - append
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    with pytest.raises(ValueError, match="invalid sync.execution"):
        datasets.list_datasets()


def test_dataset_registry_accepts_supported_sync_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "valid_append.yaml"
    registry_file.write_text(
        """
- id: valid_append
  name: Valid append
  variable: value
  period_type: daily
  sync:
    kind: temporal
    execution: append
  ingestion:
    plugin: open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "CONFIGS_DIR", tmp_path)

    assert datasets.list_datasets()[0]["sync"]["execution"] == "append"


def test_write_dataset_template_persists_into_plugins_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugins_dir = tmp_path / "plugins"
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    path = datasets.write_dataset_template(
        {
            "id": "derived_change",
            "name": "Derived Change",
            "variable": "change",
            "period_type": "yearly",
            "sync": {"kind": "static"},
            "display": {"colormap": "RdBu", "range": [-10.0, 10.0]},
        }
    )

    assert path == plugins_dir / "datasets" / "derived_change.yaml"
    assert path.exists()
    loaded = datasets.get_dataset("derived_change")
    assert loaded is not None
    assert loaded["display"]["colormap"] == "RdBu"


def test_write_dataset_template_rejects_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugins_dir = tmp_path / "plugins"
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    template = {
        "id": "derived_change",
        "name": "Derived Change",
        "variable": "change",
        "period_type": "yearly",
        "sync": {"kind": "static"},
        "display": {"colormap": "RdBu", "range": [-10.0, 10.0]},
    }
    datasets.write_dataset_template(template)

    with pytest.raises(FileExistsError, match="already exists"):
        datasets.write_dataset_template(template)


@pytest.mark.parametrize("bad_id", ["../evil", "nested/file", "/abs/path"])
def test_write_dataset_template_rejects_path_traversal_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_id: str,
) -> None:
    plugins_dir = tmp_path / "plugins"
    config_file = tmp_path / "climate-service.yaml"
    config_file.write_text(f"plugins_dir: {plugins_dir}\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(api_config, "_cache", None)
    monkeypatch.setenv("CLIMATE_SERVICE_CONFIG", str(config_file))

    with pytest.raises(ValueError, match="Invalid dataset id"):
        datasets.write_dataset_template(
            {
                "id": bad_id,
                "name": "Derived Change",
                "variable": "change",
                "period_type": "yearly",
                "sync": {"kind": "static"},
                "display": {"colormap": "RdBu", "range": [-10.0, 10.0]},
            }
        )


def test_entry_point_plugin_dataset_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed plugin's dataset template is merged into list_datasets (#118)."""
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(datasets, "_load_builtin_datasets", lambda: [])
    monkeypatch.setattr(
        datasets,
        "_load_entry_point_datasets",
        lambda: [
            (
                "senorge",
                {
                    "id": "senorge_temperature_daily",
                    "sync": {"kind": "temporal"},
                    "ingestion": {"plugin": "pkg.senorge.SeNorgePlugin"},
                },
            )
        ],
    )
    monkeypatch.setattr(api_config, "get_config", lambda: {})
    ids = [d["id"] for d in datasets.list_datasets()]
    assert "senorge_temperature_daily" in ids


def test_entry_point_plugin_overrides_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin template with a built-in id overrides the built-in one."""
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(
        datasets, "_load_builtin_datasets", lambda: [{"id": "x", "name": "builtin", "sync": {"kind": "static"}}]
    )
    monkeypatch.setattr(
        datasets,
        "_load_entry_point_datasets",
        lambda: [("p", {"id": "x", "name": "plugin", "sync": {"kind": "static"}})],
    )
    monkeypatch.setattr(api_config, "get_config", lambda: {})
    result = {d["id"]: d for d in datasets.list_datasets()}
    assert result["x"]["name"] == "plugin"


def test_plugins_dir_overrides_entry_point_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """plugins_dir takes precedence over an installed plugin on id conflict."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "x.yaml").write_text("- id: x\n  name: plugins_dir\n  sync:\n    kind: static\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(datasets, "_load_builtin_datasets", lambda: [])
    monkeypatch.setattr(
        datasets,
        "_load_entry_point_datasets",
        lambda: [("p", {"id": "x", "name": "plugin", "sync": {"kind": "static"}})],
    )
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": str(tmp_path)})
    monkeypatch.setattr(api_config, "get_config_path", lambda: tmp_path / "climate-service.yaml")
    result = {d["id"]: d for d in datasets.list_datasets()}
    assert result["x"]["name"] == "plugins_dir"


def test_builtin_templates_are_parsed_once_per_process() -> None:
    """The YAML parse is 13.5 ms; it used to run on every call, twice for some requests."""
    datasets.reset_template_caches()
    first = datasets._load_builtin_datasets()
    second = datasets._load_builtin_datasets()

    info = datasets._parse_builtin_datasets.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    assert [d["id"] for d in first] == [d["id"] for d in second]


def test_returned_templates_are_isolated_from_the_cache() -> None:
    """A caller mutating a template must not poison every later request."""
    datasets.reset_template_caches()
    mutated = datasets._load_builtin_datasets()
    mutated[0]["name"] = "clobbered"
    mutated[0].setdefault("display", {})["colormap"] = "clobbered"

    fresh = datasets._load_builtin_datasets()
    assert fresh[0]["name"] != "clobbered"
    assert fresh[0].get("display", {}).get("colormap") != "clobbered"


def test_unrecognised_units_warn_once_per_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """An instance's plugins_dir is re-read per request, so the warning must not repeat.

    Counting calls on the module logger rather than using ``caplog``: validating units imports
    xclim, which pulls in pint, and that import reshuffles root logging handlers so records
    emitted afterwards can escape capture. The assertion here is the intent anyway — warn once.
    """
    datasets.reset_template_caches()
    warnings: list[str] = []
    monkeypatch.setattr(datasets.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))
    template = {
        "id": "counts",
        "name": "Counts",
        "variable": "n",
        "sync": {"kind": "static"},
        "units": "people",
    }

    for _ in range(3):
        datasets._validate_dataset_template(template, source="counts.yaml")

    # Filtered to the units warning: this template also declares no licence, which warns once
    # too (CLIM-946). Counting every warning would make this test fail whenever another
    # validation is added, which is not what it is checking.
    units_warnings = [w for w in warnings if "not a recognised CF/udunits unit" in w]
    assert len(units_warnings) == 1


def test_missing_plugins_dir_serves_built_ins_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A configured plugins_dir that does not exist must not break template listing.

    Startup warns and keeps serving, so raising here left the instance reporting healthy with
    /dataset-templates/ returning 500 — the one route the ingest form needs (CLIM-910).
    """
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(
        datasets, "_load_builtin_datasets", lambda: [{"id": "built_in", "name": "b", "sync": {"kind": "static"}}]
    )
    monkeypatch.setattr(
        datasets,
        "_load_entry_point_datasets",
        lambda: [("p", {"id": "from_plugin", "name": "p", "sync": {"kind": "static"}})],
    )
    absent = tmp_path / "not-created"
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": str(absent)})
    monkeypatch.setattr(api_config, "get_config_path", lambda: tmp_path / "climate-service.yaml")

    assert not absent.exists()
    assert {d["id"] for d in datasets.list_datasets()} == {"built_in", "from_plugin"}


def test_plugins_dir_of_the_wrong_type_still_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A misconfigured type is a config error, not a missing directory — keep failing loudly."""
    monkeypatch.setattr(datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(datasets, "_load_builtin_datasets", lambda: [])
    monkeypatch.setattr(datasets, "_load_entry_point_datasets", lambda: [])
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": ["a", "list"]})
    monkeypatch.setattr(api_config, "get_config_path", lambda: tmp_path / "climate-service.yaml")

    with pytest.raises(ValueError, match="must be a path string"):
        datasets.list_datasets()


def test_concurrent_cold_calls_parse_the_templates_once() -> None:
    """FastAPI runs sync endpoints in worker threads, so a cold cache is hit concurrently.

    ``lru_cache`` calls the wrapped function outside its own lock, so without serialising
    population every thread arriving on a cold cache parses every template.
    """
    from concurrent.futures import ThreadPoolExecutor

    datasets.reset_template_caches()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(datasets._load_builtin_datasets) for _ in range(8)]]

    assert datasets._parse_builtin_datasets.cache_info().misses == 1
    assert all([d["id"] for d in r] == [d["id"] for d in results[0]] for r in results)


def test_concurrent_validation_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent validation of one template yields one warning.

    A smoke test, not a race guard: it also passes without the lock in `_validate_dataset_template`,
    because the check-then-add window is a few bytecodes and the GIL rarely interleaves there. The
    lock is still worth having — it makes "once" true rather than merely probable — but this test
    cannot demonstrate its absence, unlike the parse test above (8 misses instead of 1).
    """
    from concurrent.futures import ThreadPoolExecutor

    datasets.reset_template_caches()
    warnings: list[str] = []
    warn_lock = __import__("threading").Lock()

    def record(msg: str, *args: object) -> None:
        with warn_lock:  # the assertion is about datasets.py's locking, not this list's
            warnings.append(msg % args if args else msg)

    monkeypatch.setattr(datasets.logger, "warning", record)
    template = {
        "id": "counts",
        "name": "Counts",
        "variable": "n",
        "sync": {"kind": "static"},
        "units": "people",
    }
    # Warm xclim outside the threads so import timing does not shape the result.
    datasets._validate_dataset_template(template, source="warmup.yaml")
    datasets.reset_template_caches()
    warnings.clear()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in [pool.submit(datasets._validate_dataset_template, template, source="counts.yaml") for _ in range(8)]:
            f.result()

    # Per distinct warning, not in total: this template also warns about its missing licence
    # (CLIM-946). The property under test is that eight concurrent validations each produce one
    # warning rather than eight, which is about locking, not about how many checks exist.
    units_warnings = [w for w in warnings if "not a recognised CF/udunits unit" in w]
    licence_warnings = [w for w in warnings if "declares no 'license'" in w]
    assert len(units_warnings) == 1
    assert len(licence_warnings) == 1
