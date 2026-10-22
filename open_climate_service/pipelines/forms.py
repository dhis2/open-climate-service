"""Turn workflow parameters and destination settings schemas into form fields, and back.

The pipeline page has no field of its own for any workflow or destination: it renders the
workflow's declared parameters as ``arg.<name>`` fields and the destination plugin's settings
JSON Schema as ``dest.<name>`` fields (``dest.<name>.<row>.<field>`` for a list of objects).
A parameter or setting with no simple form shape is edited as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from open_climate_service.exports.base import PipelineSettingsForm
from open_climate_service.pipelines.workflow import COLLECTION_ID, GEOJSON, TEMPORAL_INTERVAL

_SCALARS = {"string", "integer", "number", "boolean"}
EXTRA_ROWS = 1


def parameter_fields(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One descriptor per workflow parameter: how the form shows and reads it."""
    fields: list[dict[str, Any]] = []
    for parameter in parameters:
        schema = parameter.get("schema") if isinstance(parameter.get("schema"), dict) else {}
        assert isinstance(schema, dict)
        subtype, kind_type = schema.get("subtype"), schema.get("type")
        if subtype == TEMPORAL_INTERVAL:
            kind = "interval"
        elif subtype == COLLECTION_ID:
            kind = "dataset"
        elif subtype == GEOJSON:
            kind = "features"
        elif isinstance(schema.get("enum"), list):
            kind = "enum"
        elif kind_type in _SCALARS:
            kind = str(kind_type)
        else:
            kind = "json"
        fields.append(
            {
                "name": parameter["name"],
                "kind": kind,
                "value_kind": str(kind_type) if kind_type in _SCALARS else "string",
                "description": parameter.get("description", ""),
                "enum": schema.get("enum") or [],
                "default": parameter.get("default"),
                "required": "default" not in parameter and not parameter.get("optional", False),
            }
        )
    return fields


def _deref(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        return resolved if isinstance(resolved, dict) else {}
    return node


def _scalar_type(node: dict[str, Any]) -> str | None:
    """The scalar type of a property, looking through ``anyOf: [X, null]``."""
    if node.get("type") in _SCALARS:
        return str(node["type"])
    options = [option for option in node.get("anyOf", []) if isinstance(option, dict) and option.get("type") != "null"]
    if len(options) == 1 and options[0].get("type") in _SCALARS:
        return str(options[0]["type"])
    return None


def _default_widget(field: dict[str, Any]) -> str:
    if field["enum"]:
        return "select"
    if field["kind"] == "boolean":
        return "checkbox"
    if field["kind"] == "json":
        return "json"
    if field["kind"] == "rows":
        return "rows"
    return "text"


def _widget(field: dict[str, Any], hint: str | None) -> str:
    """Use a hint only when it is safe for the schema shape; otherwise use its default."""
    compatible = {
        "select": bool(field["enum"]),
        "checkbox": field["kind"] == "boolean",
        "textarea": field["kind"] in {"string", "json"},
        "json": field["kind"] == "json",
        "rows": field["kind"] == "rows",
        "text": field["kind"] in {"string", "integer", "number"},
    }
    return hint if hint is not None and compatible.get(hint, False) else _default_widget(field)


def _apply_presentation(field: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    label = metadata.get("label")
    help_text = metadata.get("help_text")
    option_labels = metadata.get("option_labels") or {}
    field["title"] = label if label is not None else field["title"]
    field["description"] = help_text if help_text is not None else field["description"]
    field["widget"] = _widget(field, metadata.get("widget"))
    field["options"] = [
        {"value": option, "label": option_labels.get(str(option), str(option))} for option in field["enum"]
    ]
    return field


def settings_fields(
    schema: dict[str, Any] | None, presentation: PipelineSettingsForm | None = None
) -> list[dict[str, Any]]:
    """One descriptor per destination setting: schema data plus optional presentation."""
    if not schema:
        return []
    metadata = {field.name: asdict(field) for field in presentation.fields} if presentation else {}
    required = set(schema.get("required", []))
    fields: list[dict[str, Any]] = []
    for name, raw in (schema.get("properties") or {}).items():
        node = _deref(schema, raw if isinstance(raw, dict) else {})
        if node.get("deprecated") is True:
            continue
        field: dict[str, Any] = {
            "name": name,
            "title": node.get("title", name),
            "description": node.get("description", ""),
            "required": name in required,
            "enum": node.get("enum") or [],
        }
        scalar = _scalar_type(node)
        items = _deref(schema, node.get("items", {})) if node.get("type") == "array" else {}
        if scalar is not None:
            field["kind"] = scalar
        elif items.get("type") == "object" and all(
            _scalar_type(_deref(schema, sub)) is not None for sub in (items.get("properties") or {}).values()
        ):
            field["kind"] = "rows"
            item_required = set(items.get("required", []))
            field["columns"] = [
                _apply_presentation(
                    {
                        "name": sub_name,
                        "title": sub.get("title", sub_name),
                        "description": sub.get("description", ""),
                        "required": sub_name in item_required,
                        "kind": _scalar_type(_deref(schema, sub)) or "string",
                        "enum": _deref(schema, sub).get("enum") or [],
                    },
                    metadata.get(f"{name}.{sub_name}"),
                )
                for sub_name, sub in (items.get("properties") or {}).items()
                if _deref(schema, sub).get("deprecated") is not True
            ]
        else:
            field["kind"] = "json"
        fields.append(_apply_presentation(field, metadata.get(name)))
    return fields


def settings_form(
    schema: dict[str, Any] | None,
    presentation: PipelineSettingsForm | None,
    *,
    plugin_id: str,
) -> dict[str, Any]:
    """A complete, safe form view; schema-only plugins get one untitled section."""
    fields = settings_fields(schema, presentation)
    by_name = {field["name"]: field for field in fields}
    used: set[str] = set()
    sections: list[dict[str, Any]] = []
    if presentation:
        for section in presentation.sections:
            section_fields = [by_name[name] for name in section.fields if name in by_name and name not in used]
            used.update(field["name"] for field in section_fields)
            if section_fields:
                sections.append({"title": section.title, "description": section.description, "fields": section_fields})
    remaining = [field for field in fields if field["name"] not in used]
    if remaining:
        sections.append(
            {
                "title": "Additional settings" if sections else "",
                "description": "",
                "fields": remaining,
            }
        )
    return {
        "title": presentation.title if presentation else plugin_id,
        "description": presentation.description if presentation else "",
        "sections": sections,
    }


def _parse_scalar(kind: str, raw: str) -> Any:
    if kind == "integer":
        return int(raw)
    if kind == "number":
        return float(raw)
    if kind == "boolean":
        return raw.lower() in {"true", "on", "yes", "1"}
    return raw


def read_arguments(form: dict[str, str], fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Workflow arguments from ``arg.<name>`` form fields; empty fields stay unbound."""
    arguments: dict[str, Any] = {}
    for field in fields:
        raw = form.get(f"arg.{field['name']}", "").strip()
        if not raw:
            continue
        kind = field["kind"]
        try:
            if kind == "features":
                arguments[field["name"]] = {"from_features": raw}
            elif kind in ("json", "interval"):
                arguments[field["name"]] = json.loads(raw)
            elif kind in ("integer", "number", "boolean", "enum"):
                arguments[field["name"]] = _parse_scalar(field.get("value_kind", kind), raw)
            else:
                arguments[field["name"]] = raw
        except ValueError as exc:
            raise ValueError(f"workflow argument '{field['name']}': {exc}") from None
    return arguments


def read_settings(form: dict[str, str], fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Destination settings from ``dest.`` form fields; empty fields and rows are left out."""
    settings: dict[str, Any] = {}
    for field in fields:
        name, kind = field["name"], field["kind"]
        if kind == "rows":
            rows: list[dict[str, Any]] = []
            index = 0
            prefix = f"dest.{name}."
            while any(key.startswith(f"{prefix}{index}.") for key in form):
                try:
                    row: dict[str, Any] = {}
                    has_value = f"{prefix}{index}.__present" in form
                    for column in field["columns"]:
                        key = f"dest.{name}.{index}.{column['name']}"
                        raw = form.get(key, "").strip()
                        if column["widget"] == "checkbox":
                            if raw:
                                has_value = True
                            row[column["name"]] = _parse_scalar("boolean", raw) if raw else False
                        elif raw:
                            has_value = True
                            row[column["name"]] = _parse_scalar(column["kind"], raw)
                except ValueError as exc:
                    raise ValueError(f"destination setting '{name}' row {index + 1}: {exc}") from None
                if has_value:
                    rows.append(row)
                index += 1
            if rows:
                settings[name] = rows
            continue
        if field["widget"] == "checkbox" and f"dest.{name}.__present" in form:
            settings[name] = _parse_scalar("boolean", form.get(f"dest.{name}", ""))
            continue
        raw = form.get(f"dest.{name}", "").strip()
        if not raw:
            continue
        try:
            settings[name] = json.loads(raw) if kind == "json" else _parse_scalar(kind, raw)
        except ValueError as exc:
            raise ValueError(f"destination setting '{name}': {exc}") from None
    return settings


def argument_value(arguments: dict[str, Any], field: dict[str, Any]) -> str:
    """The text a field shows for a bound argument."""
    value = arguments.get(field["name"])
    if value is None:
        return ""
    if field["kind"] == "features" and isinstance(value, dict):
        return str(value.get("from_features", ""))
    if field["kind"] in ("json", "interval") or isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool) and field["kind"] == "boolean":
        return str(value).lower()
    return str(value)


def setting_rows(settings: dict[str, Any], field: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows a list setting shows: the stored ones plus room for one more."""
    stored = settings.get(field["name"])
    rows = [row for row in stored if isinstance(row, dict)] if isinstance(stored, list) else []
    return [
        *({"values": row, "existing": True} for row in rows),
        *({"values": {}, "existing": False} for _ in range(EXTRA_ROWS)),
    ]


def setting_value(settings: dict[str, Any], field: dict[str, Any]) -> str:
    """The text a field shows for a stored setting."""
    value = settings.get(field["name"])
    if value is None:
        return ""
    if field["kind"] == "json":
        return json.dumps(value)
    if isinstance(value, bool) and field["widget"] == "checkbox":
        return str(value).lower()
    return str(value)
