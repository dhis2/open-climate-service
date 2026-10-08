"""Operator-friendly sync-check choices; the stored schedule remains a cron expression."""

from __future__ import annotations

import re
from typing import Any, Mapping

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_CLOCK = re.compile(r"^(\d{2}):(\d{2})$")


def suggested_frequency(period_type: str | None) -> str:
    """Suggest a check interval, not an assumed upstream publication time."""
    if period_type == "yearly":
        return "monthly"
    if period_type in {"monthly", "dekadal"}:
        return "weekly"
    return "daily"


def cron_from_form(fields: Mapping[str, str]) -> str:
    """Turn the simple form into the same five-field cron used by the API and YAML."""
    frequency = fields.get("frequency", "")
    if frequency == "custom":
        cron = fields.get("cron", "").strip()
        if not cron:
            raise ValueError("Enter a five-field cron expression for a custom schedule")
        return cron
    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("Choose daily, weekly, monthly, or custom checks")

    clock = _CLOCK.fullmatch(fields.get("check_time", ""))
    if clock is None:
        raise ValueError("Enter a check time in HH:MM format")
    hour, minute = (int(part) for part in clock.groups())
    if hour > 23 or minute > 59:
        raise ValueError("Check time must be between 00:00 and 23:59")
    if frequency == "daily":
        return f"{minute} {hour} * * *"
    if frequency == "weekly":
        weekday = fields.get("weekday", "")
        if weekday not in _WEEKDAYS:
            raise ValueError("Choose a day of the week")
        return f"{minute} {hour} * * {weekday}"
    try:
        month_day = int(fields.get("month_day", ""))
    except ValueError as exc:
        raise ValueError("Choose a day of the month from 1 to 28") from exc
    if not 1 <= month_day <= 28:
        raise ValueError("Choose a day of the month from 1 to 28 so every month has a check")
    return f"{minute} {hour} {month_day} * *"


def form_values(cron: str | None, period_type: str | None, draft: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Prefill the simple form, retaining custom expressions and refused form input."""
    values = {
        "frequency": suggested_frequency(period_type),
        "check_time": "06:00",
        "weekday": "mon",
        "month_day": "5",
        "cron": cron or "",
    }
    if cron:
        parts = cron.split()
        if len(parts) == 5 and parts[0].isdigit() and parts[1].isdigit():
            minute, hour = int(parts[0]), int(parts[1])
            if 0 <= minute <= 59 and 0 <= hour <= 23:
                values["check_time"] = f"{hour:02d}:{minute:02d}"
                if parts[2:] == ["*", "*", "*"]:
                    values["frequency"] = "daily"
                elif parts[2:4] == ["*", "*"] and parts[4] in _WEEKDAYS:
                    values["frequency"] = "weekly"
                    values["weekday"] = parts[4]
                elif parts[2].isdigit() and parts[3:] == ["*", "*"] and 1 <= int(parts[2]) <= 28:
                    values["frequency"] = "monthly"
                    values["month_day"] = parts[2]
                else:
                    values["frequency"] = "custom"
            else:
                values["frequency"] = "custom"
        else:
            values["frequency"] = "custom"
    if draft:
        for key in ("frequency", "check_time", "weekday", "month_day"):
            if f"_ui_{key}" in draft:
                values[key] = str(draft[f"_ui_{key}"])
        if "_ui_cron" in draft:
            values["cron"] = str(draft["_ui_cron"])
    return values
