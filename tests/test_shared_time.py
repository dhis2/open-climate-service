from datetime import date

import pytest

from open_geo_service.shared.time import (
    daily_period_ids,
    datetime_to_period_string,
    normalize_period_string,
    parse_period_string_to_datetime,
    period_type_to_iso_step,
)


def test_daily_period_ids_enumerates_inclusive_range_from_strings() -> None:
    assert daily_period_ids("2026-01-30", "2026-02-02") == [
        "2026-01-30",
        "2026-01-31",
        "2026-02-01",
        "2026-02-02",
    ]


def test_daily_period_ids_accepts_date_objects_and_single_day() -> None:
    assert daily_period_ids(date(2026, 3, 1), date(2026, 3, 1)) == ["2026-03-01"]


def test_daily_period_ids_returns_empty_when_start_after_end() -> None:
    assert daily_period_ids("2026-05-10", "2026-05-01") == []


def test_daily_period_ids_caps_end_to_cutoff() -> None:
    # cutoff (a source's latest available day) caps end; accepts a date or string
    assert daily_period_ids("2026-01-01", "2026-01-31", cutoff=date(2026, 1, 3)) == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]
    # a cutoff later than end leaves the range unchanged
    assert daily_period_ids("2026-01-01", "2026-01-02", cutoff="2026-06-01") == ["2026-01-01", "2026-01-02"]


@pytest.mark.parametrize(
    ("period_type", "expected"),
    [
        ("hourly", "PT1H"),
        ("daily", "P1D"),
        ("weekly", "P7D"),
        ("monthly", "P1M"),
        ("quarterly", "P3M"),
        ("yearly", "P1Y"),
        ("climatology", None),
        (None, None),
    ],
)
def test_period_type_to_iso_step(period_type: object, expected: str | None) -> None:
    assert period_type_to_iso_step(period_type) == expected


def test_normalize_period_string_raises_targeted_monthly_error() -> None:
    with pytest.raises(ValueError, match="Invalid monthly period '2024-13'; expected YYYY-MM or ISO datetime"):
        normalize_period_string("2024-13", "monthly")


def test_normalize_period_string_accepts_dataset_native_hourly_period() -> None:
    assert normalize_period_string("2026-04-21T13", "hourly") == "2026-04-21T13"


def test_normalize_period_string_converts_aware_hourly_datetime_to_utc_period() -> None:
    assert normalize_period_string("2026-04-21T13:30:00+02:00", "hourly") == "2026-04-21T11"


def test_normalize_period_string_converts_aware_daily_datetime_to_utc_period() -> None:
    assert normalize_period_string("2026-04-21T00:30:00+02:00", "daily") == "2026-04-20"


def test_normalize_period_string_passes_climatology_through() -> None:
    # A day-of-year climatology has ordinal period ids, not dates — nothing to
    # normalize, so the request value passes through instead of raising.
    assert normalize_period_string("2020", "climatology") == "2020"
    assert normalize_period_string("anything", "climatology") == "anything"
    assert normalize_period_string("  2020  ", "climatology") == "2020"  # incidental whitespace trimmed


def test_normalize_period_string_accepts_dataset_native_weekly_period() -> None:
    assert normalize_period_string("2026-W17", "weekly") == "2026-W17"


def test_normalize_period_string_converts_datetime_to_weekly_period() -> None:
    assert normalize_period_string("2026-04-21T13:30:00+00:00", "weekly") == "2026-W17"


def test_datetime_to_period_string_converts_aware_monthly_datetime_to_utc_period() -> None:
    from datetime import datetime

    value = datetime.fromisoformat("2026-05-01T00:30:00+02:00")

    assert datetime_to_period_string(value, "monthly") == "2026-04"


def test_normalize_period_string_rejects_invalid_weekly_period() -> None:
    with pytest.raises(ValueError, match="Invalid weekly period '2026-W54'; expected YYYY-Www or ISO datetime"):
        normalize_period_string("2026-W54", "weekly")


def test_parse_period_string_to_datetime_accepts_dataset_native_hourly_period() -> None:
    parsed = parse_period_string_to_datetime("2026-04-21T13")

    assert parsed.isoformat() == "2026-04-21T13:00:00+00:00"


def test_parse_period_string_to_datetime_accepts_dataset_native_weekly_period() -> None:
    parsed = parse_period_string_to_datetime("2026-W17")

    assert parsed.isoformat() == "2026-04-20T00:00:00+00:00"
