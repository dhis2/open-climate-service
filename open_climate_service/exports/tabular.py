"""Shared tabular serialization used by legacy writers and export plugins."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from open_climate_service.shared.vectors import GEOMETRY_WKT_COORD

_NON_VALUE_FIELDS = frozenset({"geometry", GEOMETRY_WKT_COORD, "spatial_ref", "index", "band", "bands"})
"""Columns that are never a data value once a cube is flattened to a dataframe.

Shared by the tabular exports rather than repeated in each: they identify their value column by
elimination, so a coordinate missing from one of these lists is not a cosmetic slip - it either
becomes a bogus value column or makes the export refuse an otherwise valid cube.
"""


def _build_dhis2_json_payload(df: Any, options: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    import pandas as pd

    data_element_id = _required_str_option(options, "data_element_id")
    org_unit_field = _required_str_option(options, "org_unit_field")
    period_field = _optional_str_option(options, "period_field") or "t"
    period_type = _optional_str_option(options, "period_type")
    category_option_combo = _optional_str_option(options, "category_option_combo")

    frame = pd.DataFrame(df).copy()
    if org_unit_field not in frame.columns:
        raise ValueError(f"Missing org unit field '{org_unit_field}' in aggregated result")
    if period_field not in frame.columns:
        raise ValueError(f"Missing period field '{period_field}' in aggregated result")

    value_field = _select_dhis2_value_field(frame, org_unit_field, period_field)

    data_values: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        value = record.get(value_field)
        if _is_nullish(value):
            continue

        org_unit = record.get(org_unit_field)
        if _is_nullish(org_unit):
            raise ValueError(f"Null org unit value in field '{org_unit_field}'")

        period_value = record.get(period_field)
        if _is_nullish(period_value):
            raise ValueError(f"Null period value in field '{period_field}'")

        item = {
            "dataElement": data_element_id,
            "orgUnit": str(org_unit),
            "period": _to_dhis2_period_string(period_value, period_type),
            "value": _to_dhis2_value_string(value),
        }
        if category_option_combo is not None:
            item["categoryOptionCombo"] = category_option_combo
        data_values.append(item)

    return {"dataValues": data_values}


def _required_str_option(options: dict[str, Any], key: str) -> str:
    value = _optional_str_option(options, key)
    if value is None:
        raise ValueError(f"Missing required export option '{key}'")
    return value


def _optional_str_option(options: dict[str, Any], key: str) -> str | None:
    raw = options.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _select_dhis2_value_field(frame: Any, org_unit_field: str, period_field: str) -> str:
    excluded = {org_unit_field, period_field, *_NON_VALUE_FIELDS}
    candidates = [str(c) for c in frame.columns if c not in excluded and not str(c).startswith("level_")]
    if len(candidates) != 1:
        raise ValueError(
            "DHIS2JSON export requires exactly one value column after excluding "
            f"'{org_unit_field}' and '{period_field}', found {candidates}"
        )
    return candidates[0]


def _is_nullish(value: Any) -> bool:
    import numpy as np
    import pandas as pd

    result = pd.isna(value)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            return bool(result.item())
        raise ValueError("Array-like values are not supported in tabular export cells")
    if hasattr(result, "shape") and getattr(result, "shape", ()) not in [(), None]:
        raise ValueError("Array-like values are not supported in tabular export cells")
    if hasattr(result, "item"):
        return bool(result.item())
    return bool(result)


def _normalise_period_type(period_type: str | None) -> str | None:
    if period_type is None:
        return None
    value = period_type.strip().lower()
    aliases = {
        "day": "daily",
        "daily": "daily",
        "week": "weekly",
        "weekly": "weekly",
        "month": "monthly",
        "monthly": "monthly",
        "quarter": "quarterly",
        "quarterly": "quarterly",
        "year": "yearly",
        "yearly": "yearly",
    }
    kind = aliases.get(value)
    if kind is None:
        raise ValueError(f"Unsupported period_type '{period_type}'")
    return kind


def _direct_dhis2_period_string(value: str) -> str | None:
    patterns = (
        re.compile(r"^\d{8}$"),
        re.compile(r"^\d{6}$"),
        re.compile(r"^\d{4}$"),
        re.compile(r"^\d{4}W\d{2}$"),
        re.compile(r"^\d{4}Q[1-4]$"),
    )
    if any(pattern.fullmatch(value) for pattern in patterns):
        return value
    return None


def _to_dhis2_period_string(value: Any, period_type: str | None = None) -> str:
    import pandas as pd

    if _is_nullish(value):
        raise ValueError("Cannot serialize null period value")

    kind = _normalise_period_type(period_type)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Cannot serialize blank period value")
        direct = _direct_dhis2_period_string(stripped)
        if direct is not None:
            if kind is None:
                return direct
            pattern_map = {
                "daily": re.compile(r"^\d{8}$"),
                "weekly": re.compile(r"^\d{4}W\d{2}$"),
                "monthly": re.compile(r"^\d{6}$"),
                "quarterly": re.compile(r"^\d{4}Q[1-4]$"),
                "yearly": re.compile(r"^\d{4}$"),
            }
            if pattern_map[kind].fullmatch(stripped):
                return direct
            for known_type, pattern in pattern_map.items():
                if known_type != kind and pattern.fullmatch(stripped):
                    raise ValueError(f"Period value appears to be {known_type}, but period_type={kind}")
        if kind is None:
            raise ValueError(
                "Ambiguous period value; provide save_result option 'period_type' "
                "for date-like values that are not already in DHIS2 format"
            )
        try:
            timestamp = pd.Timestamp(stripped)
        except Exception as exc:
            raise ValueError(f"Could not parse period value {stripped!r} for period_type={kind}") from exc
        return _format_dhis2_timestamp(timestamp, kind)

    if kind is None:
        raise ValueError(
            "Ambiguous period value; provide save_result option 'period_type' "
            "for date-like values that are not already in DHIS2 format"
        )

    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Could not parse period value {value!r} for period_type={kind}") from exc
    return _format_dhis2_timestamp(timestamp, kind)


def _format_dhis2_timestamp(timestamp: Any, period_type: str) -> str:
    if period_type == "daily":
        return str(timestamp.strftime("%Y%m%d"))
    if period_type == "weekly":
        iso = timestamp.isocalendar()
        return f"{iso.year}W{iso.week:02d}"
    if period_type == "monthly":
        return str(timestamp.strftime("%Y%m"))
    if period_type == "quarterly":
        return f"{timestamp.year}Q{timestamp.quarter}"
    if period_type == "yearly":
        return str(timestamp.strftime("%Y"))
    raise ValueError(f"Unsupported period_type '{period_type}'")


def _to_dhis2_value_string(value: Any) -> str:
    import numpy as np

    if _is_nullish(value):
        raise ValueError("Cannot serialize null value")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        as_float32 = float(np.float32(as_float))
        if as_float == as_float32:
            return np.format_float_positional(np.float32(as_float), trim="-")
        return np.format_float_positional(as_float, trim="-")
    if isinstance(value, Decimal):
        normalized = format(value, "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"
    return str(value)
