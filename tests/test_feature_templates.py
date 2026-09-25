"""Feature template registry, `@feature_provider` discovery, and their shared precedence rules
(CLIM-926).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from open_climate_service import config as api_config
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.features import providers as feature_providers
from open_climate_service.features import services as feature_services
from open_climate_service.features import templates as feature_templates

VALID_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}


@pytest.fixture(autouse=True)
def _isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from an empty templates/providers registry, not the shipped built-ins."""
    monkeypatch.setattr(feature_templates, "CONFIGS_DIR", None)
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [])
    monkeypatch.setattr(feature_templates, "_load_entry_point_feature_templates", lambda: [])
    monkeypatch.setattr(feature_providers, "_scan_builtin_providers", lambda: [])
    monkeypatch.setattr(feature_providers, "_scan_plugin_package_providers", lambda: [])
    monkeypatch.setattr(feature_providers, "_scan_instance_providers", lambda: [])
    monkeypatch.setattr(api_config, "get_config", lambda: {})
    feature_templates.reset_feature_template_caches()


# --- template validation --------------------------------------------------------------------


def test_a_valid_template_loads() -> None:
    feature_templates._validate_feature_template(VALID_TEMPLATE, source="test.yaml")  # does not raise


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"id": None}, "missing or invalid id", id="no_id"),
        pytest.param({"id": ""}, "missing or invalid id", id="blank_id"),
        pytest.param({"id": "a/b"}, "cannot be used in a URL", id="unsafe_id"),
        pytest.param({"name": None}, "non-empty 'name'", id="no_name"),
        pytest.param({"name": "  "}, "non-empty 'name'", id="blank_name"),
        pytest.param({"id_property": None}, "non-empty 'id_property'", id="no_id_property"),
        pytest.param({"id_property": " orgUnitCode "}, "leading or trailing whitespace", id="padded_id_property"),
        pytest.param({"provider": ""}, "invalid 'provider'", id="blank_provider"),
        pytest.param({"provider": " dhis2"}, "leading or trailing whitespace", id="padded_provider"),
        pytest.param({"params": "not-a-mapping"}, "not a mapping", id="params_not_a_dict"),
        pytest.param({"params": {"level": 2}}, "declares 'params' but no 'provider'", id="params_without_provider"),
    ],
)
def test_an_invalid_template_is_refused(overrides: dict[str, Any], expected: str) -> None:
    template = {**VALID_TEMPLATE, **overrides}
    with pytest.raises(ValueError, match=expected):
        feature_templates._validate_feature_template(template, source="test.yaml")


def test_a_template_may_declare_a_provider_with_params() -> None:
    template = {**VALID_TEMPLATE, "provider": "dhis2", "params": {"connection": "national-hmis", "level": 2}}
    feature_templates._validate_feature_template(template, source="test.yaml")  # does not raise


def test_a_template_with_no_provider_is_metadata_only() -> None:
    """A template with no provider is valid -- it names a collection someone registers by hand."""
    feature_templates._validate_feature_template(VALID_TEMPLATE, source="test.yaml")  # does not raise


def test_an_unregistered_provider_name_does_not_fail_template_validation() -> None:
    """The provider registry is populated independently; an unresolved name is a load-time fact,
    not a reason to fail every other template in the same file."""
    template = {**VALID_TEMPLATE, "provider": "not-yet-installed"}
    feature_templates._validate_feature_template(template, source="test.yaml")  # does not raise


def test_a_non_object_template_is_refused() -> None:
    with pytest.raises(ValueError, match="non-object feature template"):
        feature_templates._validate_feature_template(["not", "a", "dict"], source="test.yaml")


# --- precedence: built-in, then installed plugin, then plugins_dir ---------------------------


def test_entry_point_plugin_template_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        feature_templates, "_load_entry_point_feature_templates", lambda: [("overture", VALID_TEMPLATE)]
    )
    ids = [t["id"] for t in feature_templates.list_feature_templates()]
    assert "districts" in ids


def test_entry_point_plugin_template_overrides_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        feature_templates, "_load_builtin_feature_templates", lambda: [{**VALID_TEMPLATE, "name": "builtin"}]
    )
    monkeypatch.setattr(
        feature_templates,
        "_load_entry_point_feature_templates",
        lambda: [("overture", {**VALID_TEMPLATE, "name": "plugin"})],
    )
    result = {t["id"]: t for t in feature_templates.list_feature_templates()}
    assert result["districts"]["name"] == "plugin"


def test_plugins_dir_template_overrides_entry_point_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "districts.yaml").write_text(
        "- id: districts\n  name: plugins_dir\n  id_property: orgUnitCode\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        feature_templates,
        "_load_entry_point_feature_templates",
        lambda: [("overture", {**VALID_TEMPLATE, "name": "plugin"})],
    )
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": str(tmp_path)})
    monkeypatch.setattr(api_config, "get_config_path", lambda: tmp_path / "climate-service.yaml")

    result = {t["id"]: t for t in feature_templates.list_feature_templates()}
    assert result["districts"]["name"] == "plugins_dir"


def test_get_feature_template_returns_none_for_an_unknown_id() -> None:
    assert feature_templates.get_feature_template("nope") is None


def test_get_feature_template_finds_a_registered_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [VALID_TEMPLATE])
    assert feature_templates.get_feature_template("districts") == VALID_TEMPLATE


def test_get_feature_template_reuses_the_cached_id_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def load() -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return [VALID_TEMPLATE]

    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", load)

    assert feature_templates.get_feature_template("districts") == VALID_TEMPLATE
    assert feature_templates.get_feature_template("districts") == VALID_TEMPLATE
    assert calls == 1


def test_a_missing_plugins_dir_features_folder_serves_built_ins_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [VALID_TEMPLATE])
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": "/does/not/exist"})
    monkeypatch.setattr(api_config, "get_config_path", lambda: None)

    ids = [t["id"] for t in feature_templates.list_feature_templates()]
    assert ids == ["districts"]


# --- provider discovery: the same precedence, for @feature_provider callables -----------------


def _provider(**_: Any) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def test_a_builtin_provider_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    decorated = feature_providers.feature_provider("dhis2")(_provider)
    monkeypatch.setattr(feature_providers, "_scan_builtin_providers", lambda: [decorated])

    assert feature_providers.load_feature_providers() == {"dhis2": decorated}
    assert feature_providers.get_feature_provider("dhis2") is decorated
    assert feature_providers.get_feature_provider("unknown") is None


def test_an_installed_plugin_provider_overrides_a_builtin_of_the_same_name(monkeypatch: pytest.MonkeyPatch) -> None:
    builtin = feature_providers.feature_provider("dhis2")(lambda **_: {"source": "builtin"})
    plugin = feature_providers.feature_provider("dhis2")(lambda **_: {"source": "plugin"})
    monkeypatch.setattr(feature_providers, "_scan_builtin_providers", lambda: [builtin])
    monkeypatch.setattr(feature_providers, "_scan_plugin_package_providers", lambda: [plugin])

    assert feature_providers.get_feature_provider("dhis2") is plugin


def test_an_instance_provider_overrides_an_installed_plugin_of_the_same_name(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = feature_providers.feature_provider("dhis2")(lambda **_: {"source": "plugin"})
    instance = feature_providers.feature_provider("dhis2")(lambda **_: {"source": "instance"})
    monkeypatch.setattr(feature_providers, "_scan_plugin_package_providers", lambda: [plugin])
    monkeypatch.setattr(feature_providers, "_scan_instance_providers", lambda: [instance])

    assert feature_providers.get_feature_provider("dhis2") is instance


def test_an_instance_provider_file_is_discovered_from_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercises the real `_load_from_path` path, not the autouse fixture's `_scan_instance_providers` stub."""
    monkeypatch.undo()  # drop the autouse fixture's stub of _scan_instance_providers for this one test
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "dhis2.py").write_text(
        "from open_climate_service.features.providers import feature_provider\n"
        "\n"
        "@feature_provider('dhis2')\n"
        "def load_org_units(**kwargs):\n"
        "    return {'type': 'FeatureCollection', 'features': [], 'kwargs': kwargs}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_config, "get_config", lambda: {"plugins_dir": str(tmp_path)})
    monkeypatch.setattr(api_config, "get_config_path", lambda: tmp_path / "climate-service.yaml")

    provider = feature_providers.get_feature_provider("dhis2")

    assert provider is not None
    assert provider(level=2) == {"type": "FeatureCollection", "features": [], "kwargs": {"level": 2}}


def test_the_instance_provider_module_name_does_not_depend_on_a_per_process_salt(tmp_path: Path) -> None:
    """`hash()` is salted per process (`PYTHONHASHSEED`); the synthetic module name must not be.

    A single test process can't observe `hash()`'s own per-process instability directly -- its
    salt is fixed for the process's lifetime, so two calls in one test always agree regardless of
    which formula is used. This runs the real `_load_from_path` in two interpreters with
    different hash seeds and checks the module name each one derives for the same path agrees.
    """
    provider_file = tmp_path / "dhis2.py"
    provider_file.write_text(
        "from open_climate_service.features.providers import feature_provider\n"
        "\n"
        "@feature_provider('dhis2')\n"
        "def load_org_units(**_):\n"
        "    return {'type': 'FeatureCollection', 'features': []}\n",
        encoding="utf-8",
    )
    script = (
        "from pathlib import Path\n"
        "from open_climate_service.features.providers import _load_from_path\n"
        f"funcs = _load_from_path(Path(r'{provider_file}'))\n"
        "print(funcs[0].__module__)\n"
    )

    first = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "1"}
    )
    second = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "2"}
    )

    assert first.stdout.strip(), first.stderr
    assert first.stdout == second.stdout


# --- @feature_provider decorator itself -------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", " dhis2", "dhis2 "])
def test_feature_provider_rejects_a_blank_or_padded_name(bad: str) -> None:
    with pytest.raises(ValueError, match="@feature_provider name"):
        feature_providers.feature_provider(bad)


def test_feature_provider_marks_the_function_without_changing_its_behaviour() -> None:
    @feature_providers.feature_provider("dhis2")
    def load_org_units(**kwargs: Any) -> dict[str, Any]:
        return {"type": "FeatureCollection", "features": [], "kwargs": kwargs}

    assert feature_providers.get_feature_provider_name(load_org_units) == "dhis2"
    assert load_org_units(level=2) == {"type": "FeatureCollection", "features": [], "kwargs": {"level": 2}}


def test_an_undecorated_function_has_no_provider_name() -> None:
    def plain(**_: Any) -> dict[str, Any]:
        return {}

    assert feature_providers.get_feature_provider_name(plain) is None


# --- raster and feature registries stay type-specific -----------------------------------------


def test_raster_lookup_does_not_return_a_feature_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(registry_datasets, "_load_builtin_datasets", lambda: [])
    monkeypatch.setattr(registry_datasets, "_load_entry_point_datasets", lambda: [])
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [VALID_TEMPLATE])

    assert registry_datasets.get_dataset("districts") is None
    assert feature_templates.get_feature_template("districts") == VALID_TEMPLATE


def test_same_id_templates_resolve_from_their_own_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    raster = {"id": "districts", "name": "Raster districts", "sync": {"kind": "static"}}
    monkeypatch.setattr(registry_datasets, "CONFIGS_DIR", None)
    monkeypatch.setattr(registry_datasets, "_load_builtin_datasets", lambda: [raster])
    monkeypatch.setattr(registry_datasets, "_load_entry_point_datasets", lambda: [])
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [VALID_TEMPLATE])

    assert registry_datasets.get_dataset("districts") == raster
    assert feature_templates.get_feature_template("districts") == VALID_TEMPLATE


def test_list_datasets_never_includes_a_feature_template() -> None:
    """The raster-only enumeration (the ingest form, /dataset-templates) must not see one."""
    ids = [d["id"] for d in registry_datasets.list_datasets()]
    assert "districts" not in ids


# --- a provider may report the release it read (CLIM-893) --------------------------------------


def test_a_provider_may_return_the_collection_alone() -> None:
    """The original single form stays valid: a provider with nothing to report says so."""
    collection = {"type": "FeatureCollection", "features": []}

    unpacked, version = feature_services._unpack_provider_result(collection, provider_name="dhis2")

    assert unpacked is collection
    assert version is None


def test_a_provider_may_return_the_collection_with_its_version() -> None:
    collection = {"type": "FeatureCollection", "features": []}

    unpacked, version = feature_services._unpack_provider_result((collection, "2026-09-23.0"), provider_name="overture")

    assert unpacked is collection
    assert version is not None
    assert (version.value, version.authority) == ("2026-09-23.0", "overture")


def test_the_authority_is_the_registry_name_not_anything_the_provider_said() -> None:
    """A release identity is the pair; a provider naming its own authority could claim another's."""
    _unpacked, version = feature_services._unpack_provider_result(
        ({"type": "FeatureCollection", "features": []}, "R2025A"), provider_name="dhis2"
    )

    assert version is not None and version.authority == "dhis2"


def test_a_reported_version_of_none_records_nothing() -> None:
    _unpacked, version = feature_services._unpack_provider_result(
        ({"type": "FeatureCollection", "features": []}, None), provider_name="overture"
    )

    assert version is None


def test_a_non_string_reported_version_is_refused() -> None:
    with pytest.raises(ValueError, match="a version must be a non-empty string"):
        feature_services._unpack_provider_result(
            ({"type": "FeatureCollection", "features": []}, 7), provider_name="overture"
        )


@pytest.mark.parametrize(
    "reported",
    [pytest.param("", id="empty"), pytest.param("   ", id="whitespace"), pytest.param(" v1 ", id="padded")],
)
def test_a_reported_version_is_not_trimmed_into_validity(reported: str) -> None:
    """`ArtifactVersion.value` rejects padding rather than trimming it, so 'verbatim' stays true.

    Trimming here would let a provider register a value that the same string could not be
    *declared* with in a template — two doors to one field disagreeing about what it accepts.
    """
    with pytest.raises(ValueError, match="not a usable release identifier"):
        feature_services._unpack_provider_result(
            ({"type": "FeatureCollection", "features": []}, reported), provider_name="overture"
        )


def test_a_wrong_sized_tuple_is_refused() -> None:
    with pytest.raises(ValueError, match="3-tuple"):
        feature_services._unpack_provider_result(({}, "v", "extra"), provider_name="overture")


def test_a_provider_returning_something_that_is_not_a_collection_is_refused() -> None:
    with pytest.raises(ValueError, match="returned list; it must return either"):
        feature_services._unpack_provider_result([1, 2], provider_name="overture")


def test_a_pair_whose_first_element_is_not_a_collection_is_refused() -> None:
    with pytest.raises(ValueError, match="first element is list"):
        feature_services._unpack_provider_result(([], "2026-09-23.0"), provider_name="overture")
