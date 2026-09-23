"""Feature collection template registry, backed by YAML config files (CLIM-926).

A feature template is the declarative counterpart to a raster dataset template, in its own
`plugins/features/` folder rather than `plugins/datasets/` — the folder name is what tells the
loader which schema to parse, so the two never need to share a file format or registry lookup.

Structure mirrors `data_registry.services.datasets` closely on purpose: same three-tier
precedence (built-in, then installed plugin package, then instance `plugins_dir`, last wins),
same `CONFIGS_DIR` test override, same cache-and-deep-copy shape for the two immutable stages.
Kept as an independent module rather than folded into that one, so raster-only callers
(`list_datasets`, the ingest form, `/dataset-templates`) are never handed a feature template by
surprise.
"""

from __future__ import annotations

import copy
import functools
import importlib.resources
import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from open_climate_service import config as api_config
from open_climate_service.shared.urls import is_segment_safe_id

logger = logging.getLogger(__name__)

# Overridden in tests via monkeypatch to point to a temporary directory.
# When set, only this directory is loaded (no built-ins, no config override).
CONFIGS_DIR: Path | None = None

_PARSE_LOCK = threading.Lock()


def list_feature_templates() -> list[dict[str, Any]]:
    """Load all feature templates and return a flat list.

    Precedence, increasing (a same-id template at a later stage overrides an earlier one):

    1. Built-in templates from `open_climate_service/plugins/features/`.
    2. Installed plugin packages declaring an `open_climate_service.plugins` entry point
       that ships a `features/` folder (#118).
    3. The instance `plugins_dir` from `CLIMATE_SERVICE_CONFIG`.

    `CONFIGS_DIR` (test override via monkeypatch) bypasses this and loads only from the given
    directory, matching `data_registry.services.datasets.list_datasets`.
    """
    if CONFIGS_DIR is not None:
        return _load_from_dir(CONFIGS_DIR)

    merged: dict[str, dict[str, Any]] = {t["id"]: t for t in _load_builtin_feature_templates()}

    for plugin_name, template in _load_entry_point_feature_templates():
        template_id = template["id"]
        if template_id in merged:
            logger.warning(
                "Plugin '%s' feature template '%s' overrides an existing feature template",
                plugin_name,
                template_id,
            )
        merged[template_id] = template

    config = api_config.get_config()
    config_plugins_dir = config.get("plugins_dir") if config else None
    if config_plugins_dir:
        if not isinstance(config_plugins_dir, (str, Path)):
            raise ValueError(
                f"plugins_dir in CLIMATE_SERVICE_CONFIG must be a path string, got {type(config_plugins_dir).__name__}"
            )
        config_path = api_config.get_config_path()
        base = config_path.parent if config_path else Path()
        root = (base / config_plugins_dir).resolve()
        features_subdir = root / "features"
        if features_subdir.is_dir():
            for template in _load_from_dir(features_subdir):
                template_id = template["id"]
                if template_id in merged:
                    logger.info("plugins_dir feature template '%s' overrides an existing one", template_id)
                merged[template_id] = template

    return list(merged.values())


def get_feature_template(template_id: str) -> dict[str, Any] | None:
    """Return one feature template by id, or None if no feature template declares it."""
    template = _feature_templates_by_id_cached(*_registry_cache_key()).get(template_id)
    return copy.deepcopy(template) if template is not None else None


def feature_templates_by_id() -> dict[str, dict[str, Any]]:
    """Return feature templates keyed by id after one cached registry scan."""
    return copy.deepcopy(_feature_templates_by_id_cached(*_registry_cache_key()))


def _registry_cache_key() -> tuple[str | None, str | None, str | None]:
    """Describe the configured registry source so configuration changes invalidate the lookup."""
    config = api_config.get_config() or {}
    plugins_dir = config.get("plugins_dir")
    config_path = api_config.get_config_path()
    return (
        str(CONFIGS_DIR) if CONFIGS_DIR is not None else None,
        str(config_path) if config_path is not None else None,
        str(plugins_dir) if plugins_dir is not None else None,
    )


@functools.lru_cache(maxsize=8)
def _feature_templates_by_id_cached(
    configs_dir: str | None,
    config_path: str | None,
    plugins_dir: str | None,
) -> dict[str, dict[str, Any]]:
    """Build the id lookup once for each configured registry source."""
    del configs_dir, config_path, plugins_dir
    return {str(template["id"]): template for template in list_feature_templates()}


def reset_feature_template_caches() -> None:
    """Forget the parsed built-in and plugin feature templates.

    Mirrors `data_registry.services.datasets.reset_template_caches` for the same reason: tests
    that install a fake entry point or patch package data must not leak state between them.
    """
    with _PARSE_LOCK:
        _parse_builtin_feature_templates.cache_clear()
        _parse_entry_point_feature_templates.cache_clear()
        _feature_templates_by_id_cached.cache_clear()


def _load_builtin_feature_templates() -> list[dict[str, Any]]:
    """Built-in feature templates, parsed once per process and deep-copied per caller.

    Same reasoning as `_load_builtin_datasets`: package data cannot change mid-process, so
    parsing and validating once and handing every caller its own copy is what keeps a mutation
    in one caller from poisoning the next.
    """
    with _PARSE_LOCK:
        parsed = _parse_builtin_feature_templates()
    return copy.deepcopy(parsed)


@functools.lru_cache(maxsize=1)
def _parse_builtin_feature_templates() -> list[dict[str, Any]]:
    pkg = importlib.resources.files("open_climate_service") / "plugins" / "features"
    templates: list[dict[str, Any]] = []
    try:
        resources = list(pkg.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return templates
    for resource in resources:
        if not resource.name.endswith((".yaml", ".yml")):
            continue
        try:
            content = resource.read_text(encoding="utf-8")
            file_templates = yaml.safe_load(content)
            if not isinstance(file_templates, list):
                raise ValueError(f"{resource.name} must contain a list of feature templates")
            for template in file_templates:
                _validate_feature_template(template, source=resource.name)
            templates.extend(file_templates)
        except Exception:
            logger.exception("Error loading %s", resource.name)
            raise
    return templates


def _load_entry_point_feature_templates() -> list[tuple[str, dict[str, Any]]]:
    """Feature templates from installed plugin packages, parsed once per process."""
    with _PARSE_LOCK:
        parsed = _parse_entry_point_feature_templates()
    return copy.deepcopy(parsed)


@functools.lru_cache(maxsize=1)
def _parse_entry_point_feature_templates() -> list[tuple[str, dict[str, Any]]]:
    from open_climate_service.plugin_discovery import iter_plugin_subdirs

    results: list[tuple[str, dict[str, Any]]] = []
    for plugin_name, _package, features_res in iter_plugin_subdirs("features"):
        try:
            for resource in features_res.iterdir():
                if not resource.name.endswith((".yaml", ".yml")):
                    continue
                file_templates = yaml.safe_load(resource.read_text(encoding="utf-8"))
                if not isinstance(file_templates, list):
                    raise ValueError(f"{plugin_name} ({resource.name}) must contain a list of feature templates")
                for template in file_templates:
                    _validate_feature_template(template, source=f"plugin '{plugin_name}' ({resource.name})")
                    results.append((plugin_name, template))
        except Exception:
            logger.exception("Error loading feature templates from plugin '%s'", plugin_name)
            raise
    return results


def _load_from_dir(folder: Path) -> list[dict[str, Any]]:
    """Load feature templates from a directory on disk."""
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    templates: list[dict[str, Any]] = []
    for file_path in sorted(folder.glob("*.y*ml")):
        try:
            with open(file_path, encoding="utf-8") as f:
                file_templates = yaml.safe_load(f)
                if not isinstance(file_templates, list):
                    raise ValueError(f"{file_path.name} must contain a list of feature templates")
                for template in file_templates:
                    _validate_feature_template(template, source=str(file_path))
                templates.extend(file_templates)
        except Exception:
            logger.exception("Error loading %s", file_path.name)
            raise
    return templates


def _validate_feature_template(template: object, *, source: str) -> None:
    """Validate the fields a feature template must declare.

    Deliberately a different shape from `_validate_dataset_template`: no `sync.kind`, no
    `period_type` — a static feature collection has no cadence and no upstream to compare
    against, which is exactly the property CLIM-1067 relied on to keep `ArtifactRecord.version`
    and `period_type` optional in the first place.
    """
    if not isinstance(template, dict):
        raise ValueError(f"{source} contains a non-object feature template")

    template_id = template.get("id")
    if not isinstance(template_id, str) or not template_id:
        raise ValueError(f"{source} contains a feature template with a missing or invalid id")
    # The id becomes a path segment in every link this collection gets --
    # `/features/{id}`, `/features/{id}/data.parquet`, `/stac/collections/{id}` -- so an id
    # outside this shape publishes links that do not resolve. Same rule, same reason, as a
    # raster template id (`_validate_dataset_template`) and a stored collection id
    # (`store.validate_collection_id`); checked here too so the error names the template file
    # rather than surfacing only much later, at the first refresh.
    if not is_segment_safe_id(template_id):
        raise ValueError(
            f"Feature template id '{template_id}' in {source} cannot be used in a URL; it must "
            "start with a letter or digit and carry only letters, digits, '.', '_' or '-'"
        )

    name = template.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Feature template '{template_id}' in {source} must define a non-empty 'name'")

    # Required for the same reason `FeatureDetail.id_property` is required inside its own
    # submodel: a feature record cannot exist without one, and the loss is the quiet kind --
    # a missing or duplicated identifier pushes values against the wrong org unit. Validated
    # here too, so the error names the template rather than surfacing as a pydantic failure on
    # a record built much later, at the first refresh.
    id_property = template.get("id_property")
    if not isinstance(id_property, str) or not id_property.strip():
        raise ValueError(f"Feature template '{template_id}' in {source} must define a non-empty 'id_property'")
    if id_property != id_property.strip():
        raise ValueError(
            f"Feature template '{template_id}' in {source} declares id_property {id_property!r} with "
            "leading or trailing whitespace; it is compared exactly against the stored file, so "
            "declare it without padding"
        )

    provider = template.get("provider")
    if provider is not None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"Feature template '{template_id}' in {source} has an invalid 'provider'")
        if provider != provider.strip():
            raise ValueError(
                f"Feature template '{template_id}' in {source} declares provider {provider!r} with "
                "leading or trailing whitespace; declare it without padding"
            )
        # Not resolved against the provider registry here. A provider module can be installed
        # after templates load (or not at all, for a template that is metadata-only until one
        # is), so an unknown name is a fact for `load_features`/refresh time to raise on, not a
        # reason to fail every other template in this file — the same reasoning
        # `_validate_dataset_template` already applies to a raster template's `produced_by`.

    params = template.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"Feature template '{template_id}' in {source} has 'params' that is not a mapping")
    if params and provider is None:
        raise ValueError(
            f"Feature template '{template_id}' in {source} declares 'params' but no 'provider' to pass them to"
        )
