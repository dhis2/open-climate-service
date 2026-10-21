"""Shared utility for loading and instantiating streaming plugin classes."""

from __future__ import annotations

import inspect
from typing import Any

from open_climate_service.shared.dynamic_import import get_dynamic_function
from open_climate_service.streaming.protocol import IngestionPlugin, StaticIngestionPlugin


def instantiate_plugin(plugin_path: str, default_params: dict[str, Any]) -> IngestionPlugin:
    """Load and instantiate a streaming plugin class from a dotted import path.

    Filters constructor kwargs to only those the class accepts, so templates
    can include extra keys in ``ingestion.default_params`` without breaking
    plugins that do not declare ``**kwargs``.
    """
    instance = _instantiate(plugin_path, default_params)
    if not isinstance(instance, IngestionPlugin):
        raise TypeError(f"'{plugin_path}' does not implement the IngestionPlugin protocol")
    return instance


def instantiate_static_plugin(plugin_path: str, default_params: dict[str, Any]) -> StaticIngestionPlugin:
    """Load and instantiate the plugin of a static dataset, which needs only ``fetch``."""
    instance = _instantiate(plugin_path, default_params)
    if not isinstance(instance, StaticIngestionPlugin):
        raise TypeError(f"'{plugin_path}' does not implement the StaticIngestionPlugin protocol")
    return instance


def _instantiate(plugin_path: str, default_params: dict[str, Any]) -> Any:
    PluginClass = get_dynamic_function(plugin_path)
    sig = inspect.signature(PluginClass)
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    kwargs = default_params if accepts_var_kwargs else {k: v for k, v in default_params.items() if k in sig.parameters}
    return PluginClass(**kwargs)
