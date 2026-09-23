"""`@feature_provider` decorator and discovery of registered feature providers (CLIM-926).

A provider is a callable that returns a GeoJSON FeatureCollection — the geometry source for a
declared feature collection, the same role a raster template's `ingestion.plugin` fills for a
dataset. Discovery mirrors `openeo/plugin_processes.py`'s `@process` scan: built-in, then
installed package, then instance `plugins_dir`, last wins — kept as its own module rather than
sharing code with that scanner, matching how `exports/registry.py` is also its own independent
discovery path with the same shape.
"""

from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from open_climate_service import config as api_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Mapping[str, Any]])

_OCS_FEATURE_PROVIDER_ATTR = "__ocs_feature_provider__"


def feature_provider(name: object) -> Callable[[F], F]:
    """Register a function as a named feature provider.

    `name` is the stable registry key a template's `provider:` field references, and the
    identity `FeatureDetail.provider` records once the provider has run — so it is required and
    explicit here, unlike `@process`, which infers its id from the function name. A function
    name is free to change in a refactor; the registry name a template and every stored record
    depend on must not, so this makes the two independent from the start.

    Usage::

        @feature_provider("dhis2")
        def load_org_units(*, connection: str, level: int) -> dict:
            '''Return org unit boundaries as a GeoJSON FeatureCollection.'''
            ...

    The decorated function receives a template's declared `params` as keyword arguments and must
    return a GeoJSON FeatureCollection. It does not, and must not, set its own `properties.id`
    identity guarantee beyond what the template's `id_property` names — identity validation
    happens once, uniformly, on the way to disk (`shared.features.validate_feature_ids`, via
    `store.write_feature_collection`), not inside each provider.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("@feature_provider name must be a non-empty string")
    if name != name.strip():
        raise ValueError(
            f"@feature_provider name {name!r} has leading or trailing whitespace; declare it without padding"
        )

    def decorator(fn: F) -> F:
        setattr(fn, _OCS_FEATURE_PROVIDER_ATTR, name)
        return fn

    return decorator


def get_feature_provider_name(obj: Any) -> str | None:
    """Return the name set by `@feature_provider`, or None."""
    return getattr(obj, _OCS_FEATURE_PROVIDER_ATTR, None)


def load_feature_providers() -> dict[str, Callable[..., Mapping[str, Any]]]:
    """Return {name: callable} for every @feature_provider-decorated function.

    Resolution order (last wins), matching `plugin_processes.load_plugin_processes` and
    `data_registry.services.datasets.list_datasets`:

    1. Built-in file plugins — `open_climate_service/plugins/features/`.
    2. Installed plugin packages — `<package>/features/` (entry-point plugins, #118).
    3. Instance plugins — `plugins_dir/features/` (overrides everything).

    A name registered at a later stage silently replaces an earlier one — a `plugins_dir`
    provider named `dhis2` can override the built-in of the same name, matching how a template
    at that stage overrides a built-in template. This is not cached the way built-in datasets
    are: providers are rarely-invoked, explicit actions rather than a per-request path, so the
    modest cost of re-scanning is not worth the staleness risk `reset_template_caches` exists to
    manage for the busier dataset path.
    """
    found: dict[str, Callable[..., Mapping[str, Any]]] = {}
    for func in _scan_builtin_providers():
        name = get_feature_provider_name(func)
        if name:
            found[name] = func
    for func in _scan_plugin_package_providers():
        name = get_feature_provider_name(func)
        if name:
            found[name] = func
    for func in _scan_instance_providers():
        name = get_feature_provider_name(func)
        if name:
            found[name] = func
    return found


def get_feature_provider(name: str) -> Callable[..., Mapping[str, Any]] | None:
    """Return one registered provider by name, or None if nothing is registered under it."""
    return load_feature_providers().get(name)


def _scan_builtin_providers() -> list[Any]:
    pkg = importlib.resources.files("open_climate_service") / "plugins" / "features"
    funcs: list[Any] = []
    try:
        for resource in pkg.iterdir():
            if not resource.name.endswith(".py") or resource.name.startswith("_"):
                continue
            module_name = f"open_climate_service.plugins.features.{resource.name[:-3]}"
            funcs.extend(_load_from_module(module_name))
    except (FileNotFoundError, NotADirectoryError):
        pass
    return funcs


def _scan_plugin_package_providers() -> list[Any]:
    """Scan `features/` in each installed plugin package (#118)."""
    from open_climate_service.plugin_discovery import iter_plugin_subdirs

    funcs: list[Any] = []
    for _name, package, features_res in iter_plugin_subdirs("features"):
        for resource in features_res.iterdir():
            if not resource.name.endswith(".py") or resource.name.startswith("_"):
                continue
            funcs.extend(_load_from_module(f"{package}.features.{resource.name[:-3]}"))
    return funcs


def _scan_instance_providers() -> list[Any]:
    config = api_config.get_config()
    plugins_dir_raw = config.get("plugins_dir") if config else None
    if not plugins_dir_raw:
        return []
    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    features_dir = (base / plugins_dir_raw).resolve() / "features"
    if not features_dir.is_dir():
        return []
    funcs: list[Any] = []
    for path in sorted(features_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        funcs.extend(_load_from_path(path))
    return funcs


def _load_from_path(path: Path) -> list[Any]:
    """Load one instance provider file without changing process-wide import search paths."""
    module_name = f"_ocs_instance_feature_provider_{path.stem}_{abs(hash(path))}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return [obj for obj in vars(module).values() if callable(obj) and get_feature_provider_name(obj) is not None]
    except Exception:
        logger.warning("Failed to load feature providers from %s", path, exc_info=True)
        return []


def _load_from_module(module_name: str) -> list[Any]:
    try:
        module: ModuleType = importlib.import_module(module_name)
        return [obj for obj in vars(module).values() if callable(obj) and get_feature_provider_name(obj) is not None]
    except Exception:
        logger.warning("Failed to load feature providers from %s", module_name, exc_info=True)
        return []
