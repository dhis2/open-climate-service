"""Dekadal (10-daily) period support.

A dekad is not a fixed duration — day 1-10, 11-20, then 21 to the end of the month —
so the third dekad of a month runs 8, 9, 10 or 11 days. These tests pin that variability
and the consequences: snapping rather than truncation, stepping by the covered dekad's own
length, and an explicitly irregular STAC step rather than a fabricated duration.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

import numpy as np
import pytest

from open_climate_service.shared.time import (
    SUPPORTED_PERIOD_TYPES,
    Cadence,
    daily_period_ids,
    datetime_to_period_string,
    dekad_bounds,
    dekad_period_ids,
    dekad_start,
    normalize_period_string,
    numpy_datetime_to_period_string,
    parse_period_string_to_datetime,
    period_cadence,
    period_type_to_iso_step,
)


def test_dekadal_is_classified_irregular_and_has_no_iso_step() -> None:
    # The pair that matters: a real cadence with no duration. P10D would be wrong for
    # every third dekad, so the step lookup must yield nothing at all.
    assert period_cadence("dekadal") is Cadence.IRREGULAR
    assert period_type_to_iso_step("dekadal") is None


@pytest.mark.parametrize(
    ("period_type", "expected"),
    [
        ("daily", Cadence.REGULAR),
        ("monthly", Cadence.REGULAR),
        ("dekadal", Cadence.IRREGULAR),
        ("climatology", Cadence.NON_TEMPORAL),
        ("dekad", Cadence.UNKNOWN),  # a plausible typo for the real value
        ("fortnightly", Cadence.UNKNOWN),
        (None, Cadence.UNKNOWN),
        (10, Cadence.UNKNOWN),
    ],
)
def test_period_cadence_classifies_every_kind(period_type: object, expected: Cadence) -> None:
    assert period_cadence(period_type) is expected


def test_irregular_is_distinguishable_from_unknown() -> None:
    """Both lack an ISO step; conflating them would hide a misconfigured template."""
    assert period_type_to_iso_step("dekadal") is period_type_to_iso_step("fortnightly") is None
    assert period_cadence("dekadal") is not period_cadence("fortnightly")


def test_dekadal_is_a_supported_period_type() -> None:
    assert "dekadal" in SUPPORTED_PERIOD_TYPES
    assert "dekad" not in SUPPORTED_PERIOD_TYPES


@pytest.mark.parametrize(
    ("day", "expected_start"),
    [(1, 1), (2, 1), (10, 1), (11, 11), (12, 11), (20, 11), (21, 21), (28, 21), (31, 21)],
)
def test_dekad_start_snaps_to_the_containing_dekad(day: int, expected_start: int) -> None:
    assert dekad_start(date(2026, 1, day)).day == expected_start


def test_dekad_bounds_third_dekad_ends_with_the_month() -> None:
    assert dekad_bounds("2026-01-25") == (date(2026, 1, 21), date(2026, 1, 31))  # 11 days
    assert dekad_bounds("2026-04-25") == (date(2026, 4, 21), date(2026, 4, 30))  # 10 days
    assert dekad_bounds("2026-02-25") == (date(2026, 2, 21), date(2026, 2, 28))  # 8 days
    assert dekad_bounds("2024-02-25") == (date(2024, 2, 21), date(2024, 2, 29))  # 9 days, leap


def test_dekad_bounds_first_two_dekads_are_always_ten_days() -> None:
    for month in range(1, 13):
        for start_day in (1, 11):
            first, last = dekad_bounds(date(2026, month, start_day))
            assert (last - first).days + 1 == 10


def test_a_year_has_thirty_six_dekads_of_three_different_lengths() -> None:
    ids = dekad_period_ids("2026-01-01", "2026-12-31")
    assert len(ids) == 36
    lengths: dict[int, int] = {}
    for period_id in ids:
        first, last = dekad_bounds(period_id)
        lengths[(last - first).days + 1] = lengths.get((last - first).days + 1, 0) + 1
    # Seven 31-day months give an 11-day third dekad, the four 30-day months give a
    # 10-day one, and February (28 days in 2026) gives 8. Plus the first two dekads of
    # every month, always 10 days: 24 + 4 = 28 ten-day dekads.
    assert lengths == {8: 1, 10: 28, 11: 7}
    assert sum(k * v for k, v in lengths.items()) == 365


def test_dekad_period_ids_includes_partially_covered_dekads_at_both_ends() -> None:
    # The dekad is the smallest unit the data has, so overlapping one means needing it.
    assert dekad_period_ids("2026-01-05", "2026-02-15") == [
        "2026-01-01",
        "2026-01-11",
        "2026-01-21",
        "2026-02-01",
        "2026-02-11",
    ]


def test_dekad_period_ids_honours_the_availability_cutoff() -> None:
    assert dekad_period_ids("2026-01-01", "2026-12-31", cutoff="2026-01-15") == ["2026-01-01", "2026-01-11"]


def test_dekad_period_ids_returns_empty_for_a_reversed_range() -> None:
    assert dekad_period_ids("2026-03-01", "2026-01-01") == []


def test_dekad_period_ids_crosses_a_year_boundary() -> None:
    assert dekad_period_ids("2025-12-22", "2026-01-05") == ["2025-12-21", "2026-01-01"]


def test_period_string_snaps_a_mid_dekad_date() -> None:
    assert datetime_to_period_string(datetime(2026, 1, 15), "dekadal") == "2026-01-11"
    assert normalize_period_string("2026-01-15", "dekadal") == "2026-01-11"
    assert normalize_period_string("2026-01-31T12:00:00", "dekadal") == "2026-01-21"


def test_period_string_leaves_an_exact_dekad_start_alone() -> None:
    for period_id in ("2026-01-01", "2026-01-11", "2026-01-21"):
        assert normalize_period_string(period_id, "dekadal") == period_id


def test_invalid_dekadal_period_names_the_expected_format() -> None:
    with pytest.raises(ValueError, match="Invalid dekadal period"):
        normalize_period_string("not-a-date", "dekadal")


def test_numpy_conversion_snaps_rather_than_truncates() -> None:
    """Truncating to 10 characters would only be right for timestamps already on a start."""
    stamps = np.array(["2026-01-05", "2026-01-15", "2026-01-25", "2026-02-28"], dtype="datetime64[ns]")
    assert list(numpy_datetime_to_period_string(stamps, "dekadal")) == [
        "2026-01-01",
        "2026-01-11",
        "2026-01-21",
        "2026-02-21",
    ]


def test_dekad_ids_round_trip_through_the_generic_parser() -> None:
    # Ids are plain dates, so they need no special case in parse_period_string_to_datetime.
    for period_id in dekad_period_ids("2026-01-01", "2026-03-31"):
        parsed = parse_period_string_to_datetime(period_id)
        assert parsed.date().isoformat() == period_id


def test_dekad_ids_sort_chronologically_as_strings() -> None:
    ids = dekad_period_ids("2025-11-01", "2026-02-28")
    assert ids == sorted(ids)


def test_dekad_bounds_tile_the_month_without_gap_or_overlap() -> None:
    for month in range(1, 13):
        last_day = calendar.monthrange(2026, month)[1]
        covered: list[date] = []
        for start_day in (1, 11, 21):
            first, last = dekad_bounds(date(2026, month, start_day))
            day = first
            while day <= last:
                covered.append(day)
                day = date.fromordinal(day.toordinal() + 1)
        assert len(covered) == last_day
        assert len(set(covered)) == last_day


# --- consumers: sync planning, template validation, STAC declaration -------------------


def test_next_period_start_steps_by_the_covered_dekads_own_length() -> None:
    """A fixed +10 days would land mid-month for every 11-day and 8-day third dekad."""
    from open_climate_service.shared.time import next_period_string

    assert next_period_string("2026-01-01", "dekadal") == "2026-01-11"
    assert next_period_string("2026-01-11", "dekadal") == "2026-01-21"
    # 11-day third dekad: 21 Jan + 11 = 1 Feb, not 31 Jan.
    assert next_period_string("2026-01-21", "dekadal") == "2026-02-01"
    # 8-day third dekad in a common February.
    assert next_period_string("2026-02-21", "dekadal") == "2026-03-01"
    # 9-day third dekad in a leap February.
    assert next_period_string("2024-02-21", "dekadal") == "2024-03-01"


def test_next_period_start_never_repeats_or_skips_a_dekad_across_a_year() -> None:
    from open_climate_service.shared.time import next_period_string

    expected = dekad_period_ids("2026-01-01", "2026-12-31")
    walked = [expected[0]]
    while len(walked) < len(expected):
        walked.append(next_period_string(walked[-1], "dekadal"))
    assert walked == expected


def test_sync_and_request_defaults_return_the_current_dekad() -> None:
    from open_climate_service.ingestions.services import _current_request_period
    from open_climate_service.ingestions.sync_engine import _default_target_end
    from open_climate_service.shared.time import utc_today

    today = dekad_start(utc_today()).isoformat()
    assert _default_target_end(period_type="dekadal") == today
    assert _current_request_period("dekadal") == today


def test_template_validation_accepts_dekadal_and_rejects_an_unsupported_cadence() -> None:
    from open_climate_service.data_registry.services.datasets import _validate_dataset_template

    base = {"id": "gpp", "sync": {"kind": "temporal"}, "ingestion": {"plugin": "pkg.Cls"}}
    _validate_dataset_template({**base, "period_type": "dekadal"}, source="t.yaml")

    with pytest.raises(ValueError, match="unsupported period_type 'dekad'"):
        _validate_dataset_template({**base, "period_type": "dekad"}, source="t.yaml")


def test_stac_declares_an_irregular_step_as_null_rather_than_a_duration() -> None:
    from open_climate_service.stac.services import _override_time_step

    collection = {"cube:dimensions": {"t": {"type": "temporal", "extent": ["2026-01-01", "2026-12-31"]}}}
    _override_time_step(collection, None, cadence=Cadence.IRREGULAR)
    # Explicitly null — the datacube extension's encoding for "irregularly spaced steps".
    assert collection["cube:dimensions"]["t"]["step"] is None


def test_stac_leaves_an_xstac_inferred_step_alone_when_the_cadence_is_unknown() -> None:
    from open_climate_service.stac.services import _override_time_step

    collection = {"cube:dimensions": {"t": {"type": "temporal", "step": "P1D"}}}
    _override_time_step(collection, None, cadence=Cadence.UNKNOWN)
    assert collection["cube:dimensions"]["t"]["step"] == "P1D"


def test_stac_still_sets_a_duration_for_a_regular_cadence() -> None:
    from open_climate_service.stac.services import _override_time_step

    collection = {"cube:dimensions": {"t": {"type": "temporal"}}}
    _override_time_step(collection, "P1M", cadence=Cadence.REGULAR)
    assert collection["cube:dimensions"]["t"]["step"] == "P1M"


def test_coverage_is_reported_in_dekad_ids_for_a_dekadal_store() -> None:
    """Coverage runs through numpy_datetime_to_period_string, so it must snap too."""
    import xarray as xr

    from open_climate_service.data_accessor.services.accessor import _coverage_from_dataset

    ids = dekad_period_ids("2026-01-01", "2026-03-31")
    assert len(ids) == 9
    stamps = np.array(ids, dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"gpp": (("t", "y", "x"), np.zeros((len(stamps), 2, 2), "f4"))},
        coords={"t": stamps, "y": [1.0, 0.0], "x": [33.0, 34.0]},
    )

    coverage = _coverage_from_dataset(ds=ds, period_type="dekadal")["coverage"]["temporal"]
    assert coverage == {"start": "2026-01-01", "end": "2026-03-21"}


def test_dhis2_export_refuses_a_dekadal_dataset_rather_than_inventing_a_period() -> None:
    """DHIS2 has no dekad period type, so there is no correct id to emit."""
    import pandas as pd

    from open_climate_service.openeo.jobs import _format_dhis2_timestamp

    with pytest.raises(ValueError, match="Unsupported period_type 'dekadal'"):
        _format_dhis2_timestamp(pd.Timestamp("2026-01-11"), "dekadal")


def test_irregular_cadence_lists_its_timestamps_as_values() -> None:
    """A null step gives a client nothing to extrapolate from, so `values` must carry
    the real timestamps — otherwise a time control collapses to a single position."""
    import xarray as xr

    from open_climate_service.stac.services import _add_temporal_values

    ids = dekad_period_ids("2026-01-01", "2026-03-31")
    stamps = np.array(ids, dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"gpp": (("t", "y", "x"), np.zeros((len(stamps), 2, 2), "f4"))},
        coords={"t": stamps, "y": [1.0, 0.0], "x": [33.0, 34.0]},
    )
    collection: dict[str, Any] = {"cube:dimensions": {"t": {"type": "temporal", "step": None}}}

    _add_temporal_values(collection, ds, "t")

    values: list[str] = collection["cube:dimensions"]["t"]["values"]
    assert len(values) == len(ids) == 9
    assert values[0] == "2026-01-01T00:00:00Z"
    assert [v[:10] for v in values] == ids


def test_temporal_values_are_not_added_to_a_non_temporal_dimension() -> None:
    import xarray as xr

    from open_climate_service.stac.services import _add_temporal_values

    ds = xr.Dataset({"v": ("t", np.zeros(2, "f4"))}, coords={"t": [0, 1]})
    collection = {"cube:dimensions": {"t": {"type": "other"}}}
    _add_temporal_values(collection, ds, "t")
    assert "values" not in collection["cube:dimensions"]["t"]


def test_irregular_step_is_null_even_when_a_resolution_is_declared() -> None:
    """No duration can describe a variable-length cadence, so a declared
    extents.temporal.resolution must not make a dekadal store look regular."""
    from open_climate_service.stac.services import _override_time_step

    collection = {"cube:dimensions": {"t": {"type": "temporal"}}}
    _override_time_step(collection, "P10D", cadence=Cadence.IRREGULAR)
    assert collection["cube:dimensions"]["t"]["step"] is None


def test_quarterly_is_not_registerable_though_it_has_a_stac_step() -> None:
    """The step map serves whatever is already in a store; the registerable set is
    narrower. Quarterly has a P3M step but no period-string or sync implementation."""
    assert period_type_to_iso_step("quarterly") == "P3M"
    assert "quarterly" not in SUPPORTED_PERIOD_TYPES
    with pytest.raises(ValueError, match="Unsupported period_type 'quarterly'"):
        datetime_to_period_string(datetime(2026, 1, 1), "quarterly")


def test_period_type_is_required_unless_the_dataset_is_static() -> None:
    """Sync planning and coverage index period_type unguarded, so an absent one on a
    temporal dataset registers and then raises a KeyError further in. Static is exempt:
    an openEO save_result output legitimately has no cadence."""
    from open_climate_service.data_registry.services.datasets import _validate_dataset_template

    temporal = {"id": "x", "sync": {"kind": "temporal"}, "ingestion": {"plugin": "p.C"}}
    with pytest.raises(ValueError, match="must define period_type"):
        _validate_dataset_template(temporal, source="t.yaml")
    _validate_dataset_template({**temporal, "period_type": "dekadal"}, source="t.yaml")

    # An openEO output: static, and period_type may be underivable.
    _validate_dataset_template({"id": "y", "sync": {"kind": "static"}}, source="t.yaml")


def test_quarterly_template_is_rejected_at_registration() -> None:
    from open_climate_service.data_registry.services.datasets import _validate_dataset_template

    with pytest.raises(ValueError, match="unsupported period_type 'quarterly'"):
        _validate_dataset_template(
            {"id": "q", "sync": {"kind": "temporal"}, "ingestion": {"plugin": "p.C"}, "period_type": "quarterly"},
            source="t.yaml",
        )


def test_dekad_period_ids_matches_daily_on_an_inverted_range() -> None:
    """Snapping the start back into its dekad must not make an empty range non-empty.

    `dekad_period_ids` documents itself as the dekadal counterpart of `daily_period_ids`, and
    an inverted range overlaps no dekad, so both must return nothing.
    """
    assert daily_period_ids("2024-03-10", "2024-03-05") == []
    assert dekad_period_ids("2024-03-10", "2024-03-05") == []


def test_dekad_period_ids_is_empty_when_the_cutoff_precedes_the_start() -> None:
    """The realistic inverted case: a sync asking for periods past the available data."""
    assert dekad_period_ids("2026-08-01", "2026-12-31", cutoff="2026-07-21") == []


def test_dekad_period_ids_still_snaps_a_mid_dekad_start_forward_of_the_cutoff() -> None:
    """The guard must not swallow a legitimately overlapping dekad."""
    assert dekad_period_ids("2026-01-15", "2026-01-25") == ["2026-01-11", "2026-01-21"]
