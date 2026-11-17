"""Utilities for importing callables from dotted module paths at runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def get_dynamic_function(full_path: str) -> Callable[..., Any]:
    """Import and return a callable given its dotted module path.

    Example: ``get_dynamic_function("mypackage.module.my_func")``
    """
    parts = full_path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(f"Invalid dotted function path '{full_path}'")
    module_path = ".".join(parts[:-1])
    function_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, function_name)  # type: ignore[no-any-return]
