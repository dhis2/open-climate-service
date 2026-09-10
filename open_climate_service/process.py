"""@process decorator for registering Python functions as named openEO processes."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

_OCS_PROCESS_ATTR = "__ocs_process__"

_PYTHON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    """Return a JSON Schema dict for a Python type annotation.

    Handles plain types (str, int, …) and nullable unions (str | None).
    Returns {} for types with no known mapping (e.g. xr.DataArray).

    A nullable annotation keeps its null: `str | None` is `{"type": ["string", "null"]}`,
    matching how the openEO process specs express an optional parameter whose default is
    null (`aggregate_spatial`'s `target_dimension` is exactly this). Unwrapping to a bare
    `"string"` would publish a schema that rejects the documented default, which is worse
    than publishing none — a client validating the graph would refuse a valid call.
    """
    direct = _PYTHON_TYPE_MAP.get(ann)
    if direct:
        return {"type": direct}
    # str | None  →  UnionType (Python 3.10+) or typing.Union
    origin = getattr(ann, "__origin__", None)
    args: tuple[Any, ...] = getattr(ann, "__args__", ())
    if isinstance(ann, types.UnionType) or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _annotation_to_schema(non_none[0])
            if not inner:
                return {}
            if len(non_none) < len(args):
                return {**inner, "type": [inner["type"], "null"]}
            return inner
    return {}


def _schema_types(schema: Any) -> tuple[str, ...]:
    """The JSON Schema `type` of a parameter schema, always as a tuple.

    `type` is either a string or a list of them, and an absent or declared-empty schema has
    none. Callers only ask whether a particular type is admitted, so normalise the shape here
    rather than at each site.
    """
    if not isinstance(schema, dict):
        return ()
    declared = schema.get("type")
    if isinstance(declared, str):
        return (declared,)
    if isinstance(declared, list):
        return tuple(str(item) for item in declared)
    return ()


def _resolved_annotations(fn: Any) -> dict[str, Any]:
    """A function's annotations as objects rather than strings.

    `from __future__ import annotations` stores every annotation as a string, so a raw
    `param.annotation` is `"str"` rather than `str` and the type map below never matches — the
    parameter is then published with an empty schema, and a client building a graph gets an
    untyped field with nothing to validate against.

    Resolution needs the module's namespace and can fail on a type imported only under
    `TYPE_CHECKING`.

    `get_type_hints` resolves the whole mapping atomically, so a single unresolvable name would
    cost *every* parameter its schema — a plugin annotating one exotic type would publish an
    untyped contract for its ordinary `str` and `int` parameters too. So a failure degrades to
    resolving each annotation on its own, and only the ones that genuinely cannot be resolved
    are left out. Omitting a name is what the caller wants: it falls back to the raw string
    annotation, which maps to an empty schema for that parameter alone.
    """
    try:
        return typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 — fall through to per-annotation resolution below
        pass

    # Same operation `get_type_hints` performs internally, against the function's own module
    # globals — the strings come from the plugin's source, so this is no wider a trust boundary.
    globalns = getattr(fn, "__globals__", {})
    resolved: dict[str, Any] = {}
    for name, annotation in getattr(fn, "__annotations__", {}).items():
        if not isinstance(annotation, str):
            resolved[name] = annotation
            continue
        try:
            resolved[name] = eval(annotation, globalns)  # noqa: S307 — see above
        except Exception:  # noqa: BLE001, S112 — one unresolvable name costs only its own schema
            continue
    return resolved


@overload
def process(func: F) -> F: ...


@overload
def process(
    func: None = None,
    *,
    summary: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> Callable[[F], F]: ...


def process(
    func: F | None = None,
    *,
    summary: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> F | Callable[[F], F]:
    """Register a function as a named openEO process plugin.

    Decorated functions are discovered automatically from ``plugins_dir/processes/``
    and appear in ``GET /processes``, callable directly by ``process_id`` in any
    openEO process graph.

    Usage — minimal (summary and parameter descriptions from docstring)::

        @process
        def consecutive_dry_days(pr: xr.DataArray, thresh: str = "1mm/day") -> xr.DataArray:
            '''Maximum consecutive dry days per period.'''
            return xclim.atmos.maximum_consecutive_dry_days(pr, thresh=thresh)

    Usage — explicit metadata::

        @process(
            summary="Maximum consecutive dry days",
            parameters={"thresh": {"description": "Precipitation threshold"}},
        )
        def consecutive_dry_days(pr: xr.DataArray, thresh: str = "1mm/day") -> xr.DataArray: ...
    """

    def decorator(fn: F) -> F:
        doc = inspect.getdoc(fn) or ""
        sig = inspect.signature(fn)
        hints = _resolved_annotations(fn)

        params: list[dict[str, Any]] = []
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            p: dict[str, Any] = {"name": name, "schema": {}}
            ann = hints.get(name, param.annotation)
            if ann is not inspect.Parameter.empty:
                schema = _annotation_to_schema(ann)
                if schema:
                    p["schema"] = schema
            override = parameters.get(name, {}) if parameters else {}
            if override:
                p.update(override)
            # The default is decided after the override, against the schema that actually
            # ships. An explicit schema can narrow a nullable annotation or widen a
            # non-nullable one, and deciding beforehand would contradict either.
            if param.default is not inspect.Parameter.empty:
                p.setdefault("optional", True)
                # A schema carrying "null" admits a None default, so it is published as-is.
                # A non-nullable one would be contradicted by it, so the default is withheld.
                # An explicit default in the override always wins.
                if "default" not in override and (
                    param.default is not None or not p.get("schema") or "null" in _schema_types(p.get("schema"))
                ):
                    p["default"] = param.default
            params.append(p)

        meta: dict[str, Any] = {
            "id": fn.__name__,
            "summary": summary or (doc.splitlines()[0] if doc else fn.__name__),
            "description": description or doc,
            "parameters": params,
            "returns": {"schema": {}},
        }

        setattr(fn, _OCS_PROCESS_ATTR, meta)
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def get_process_metadata(obj: Any) -> dict[str, Any] | None:
    """Return the process metadata set by ``@process``, or None."""
    return getattr(obj, _OCS_PROCESS_ATTR, None)
