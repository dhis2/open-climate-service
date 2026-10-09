"""An export's period must be reachable from its source cadence (CLIM-1302), and CLIM-1139's collision.

Three layers are covered here: the reachability table and completeness arithmetic in
`shared.time`; the declaration checked at configuration time and the evidence checked at
execution time in `exports.service`; and the two places a label meets data: the ad-hoc
`save_result` path, and a graph that aggregates explicitly before a named export. The export
never aggregates; it exports what exists and refuses the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo_pg_parser_networkx.process_registry import Process

from open_climate_service import config
from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.exports.dhis2_renderer import Dhis2ExportPlugin
from open_climate_service.exports.service import (
    _drop_periods,
    check_execution_declarations,
    incomplete_periods_to_drop,
    render_named_export,
    resolve_named_export,
    validate_configured_exports,
)
from open_climate_service.jobs import service as job_service_module
from open_climate_service.jobs import store as native_store
from open_climate_service.openeo import execution
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo import workflows as workflow_store
from open_climate_service.openeo.jobs import check_ad_hoc_period_reachability
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.provenance import (
    capture_execution,
    observe_temporal_aggregation,
    record_incomplete_periods,
    record_reduction,
    record_source,
)
from open_climate_service.shared.time import (
    Reachability,
    cadence_of,
    export_period_label,
    incomplete_destination_periods,
    infer_cadence,
    is_dekadal_axis,
    period_label_start,
    period_reachability,
    stamp_cadence,
    utc_now,
)

_DATA_ELEMENT = "BXgDHhPdFVU"
_OU_A = "ImspTQPwCqd"
_OU_B = "O6uvpzGd5pu"


# --- the table ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "destination", "outcome"),
    [
        ("hourly", "daily", Reachability.AGGREGATE),
        ("hourly", "weekly", Reachability.AGGREGATE),
        ("daily", "daily", Reachability.PASS_THROUGH),
        ("daily", "weekly", Reachability.AGGREGATE),
        ("daily", "monthly", Reachability.AGGREGATE),
        ("daily", "quarterly", Reachability.UNREACHABLE),
        ("daily", "yearly", Reachability.AGGREGATE),
        ("dekadal", "monthly", Reachability.AGGREGATE),
        ("dekadal", "weekly", Reachability.UNREACHABLE),
        ("weekly", "daily", Reachability.UNREACHABLE),
        ("weekly", "monthly", Reachability.UNREACHABLE),
        ("weekly", "yearly", Reachability.UNREACHABLE),
        ("monthly", "monthly", Reachability.PASS_THROUGH),
        ("monthly", "weekly", Reachability.UNREACHABLE),
        ("monthly", "daily", Reachability.UNREACHABLE),
        ("monthly", "quarterly", Reachability.UNREACHABLE),
        ("monthly", "yearly", Reachability.AGGREGATE),
        ("quarterly", "yearly", Reachability.AGGREGATE),
        ("yearly", "monthly", Reachability.UNREACHABLE),
        ("yearly", "yearly", Reachability.PASS_THROUGH),
        ("climatology", "monthly", Reachability.UNREACHABLE),
        (None, "monthly", Reachability.UNREACHABLE),
    ],
)
def test_period_reachability_table(source: str | None, destination: str, outcome: Reachability) -> None:
    decided, reason = period_reachability(source, destination)
    assert decided is outcome
    assert reason


def test_finer_destination_says_so() -> None:
    _, reason = period_reachability("monthly", "weekly")
    assert "finer than monthly" in reason


def _days(start: str, end: str) -> np.ndarray:
    return pd.date_range(start, end, freq="D").values


def test_incomplete_periods_name_the_trailing_month_only() -> None:
    assert incomplete_destination_periods(_days("2025-01-01", "2025-03-14"), "daily", "monthly") == ["202503"]


def test_incomplete_periods_follow_iso_weeks() -> None:
    # 1 Jan 2025 is a Wednesday: week 1 lacks Mon-Tue, and 31 Jan is a Friday: week 5 lacks the weekend.
    assert incomplete_destination_periods(_days("2025-01-01", "2025-01-31"), "daily", "weekly") == [
        "2025W01",
        "2025W05",
    ]


def test_incomplete_periods_count_whole_months_in_a_year() -> None:
    months = pd.date_range("2025-01-01", "2025-11-01", freq="MS").values
    assert incomplete_destination_periods(months, "monthly", "yearly") == ["2025"]
    assert (
        incomplete_destination_periods(pd.date_range("2025-01-01", "2025-12-01", freq="MS").values, "monthly", "yearly")
        == []
    )


def test_infer_cadence_recognises_the_store_shapes() -> None:
    assert infer_cadence(_days("2025-01-01", "2025-01-10")) == "daily"
    assert infer_cadence(pd.date_range("2024-01-01", "2025-02-01", freq="MS").values) == "monthly"
    assert infer_cadence(np.array(["2025-01-01", "2025-01-11", "2025-01-21"], dtype="datetime64[ns]")) == "dekadal"
    assert infer_cadence(np.array(["2025-01-01"], dtype="datetime64[ns]")) is None


# --- the mapping -------------------------------------------------------------------------------


def _mapping(**extra: Any) -> dict[str, Any]:
    return {"period_type": "monthly", "series": [{"data_element": _DATA_ELEMENT}], **extra}


def test_mapping_accepts_the_two_new_fields() -> None:
    validated = Dhis2ExportPlugin().validate_mapping(_mapping(temporal_aggregation="sum", incomplete_periods="drop"))
    assert validated["temporal_aggregation"] == "sum"
    assert validated["incomplete_periods"] == "drop"


@pytest.mark.parametrize("extra", [{"temporal_aggregation": "median"}, {"incomplete_periods": "ignore"}])
def test_mapping_rejects_unknown_vocabulary(extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Dhis2ExportPlugin().validate_mapping(_mapping(**extra))


# --- configuration time ------------------------------------------------------------------------


def _configure(monkeypatch: pytest.MonkeyPatch, source: str | None, **extra: Any) -> None:
    export: dict[str, Any] = {
        "id": "rain",
        "plugin": "dhis2",
        "dataset": "src",
        "period_type": "monthly",
        "series": [{"data_element": _DATA_ELEMENT}],
        **extra,
    }
    monkeypatch.setattr(config, "_cache", {"exports": [export]})
    template = {"id": "src", "period_type": source} if source is not None else None
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: template if dataset_id == "src" else None)


def test_daily_dataset_with_monthly_export_must_declare_temporal_aggregation(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "daily")
    with pytest.raises(ValueError, match="must declare temporal_aggregation"):
        resolve_named_export("DHIS2JSON", {"export": "rain"})
    with pytest.raises(ValueError, match="must declare temporal_aggregation"):
        validate_configured_exports()


def test_daily_dataset_with_monthly_export_and_a_declaration_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "daily", temporal_aggregation="sum")
    assert resolve_named_export("DHIS2JSON", {"export": "rain"}).mapping["temporal_aggregation"] == "sum"
    validate_configured_exports()


@pytest.mark.parametrize(
    ("definitions", "message"),
    [(["not-a-mapping"], "must be a mapping"), ([{"plugin": "dhis2"}], "requires an ID")],
)
def test_startup_rejects_malformed_export_definitions(
    monkeypatch: pytest.MonkeyPatch, definitions: list[Any], message: str
) -> None:
    monkeypatch.setattr(config, "_cache", {"exports": definitions})
    with pytest.raises(ValueError, match=message):
        validate_configured_exports()


def test_monthly_dataset_with_monthly_export_refuses_a_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "monthly", temporal_aggregation="mean")
    with pytest.raises(ValueError, match="remove the declaration"):
        resolve_named_export("DHIS2JSON", {"export": "rain"})


@pytest.mark.parametrize(("source", "period_type"), [("weekly", "daily"), ("monthly", "weekly"), ("weekly", "monthly")])
def test_unreachable_periods_are_refused_at_configuration(
    monkeypatch: pytest.MonkeyPatch, source: str, period_type: str
) -> None:
    _configure(monkeypatch, source, period_type=period_type)
    with pytest.raises(ValueError, match="cannot emit"):
        resolve_named_export("DHIS2JSON", {"export": "rain"})


def test_unknown_dataset_defers_to_the_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, None)
    resolve_named_export("DHIS2JSON", {"export": "rain"})


# --- execution time ----------------------------------------------------------------------------


def _provenance(**overrides: Any) -> dict[str, Any]:
    return {"sources": [], "features": [], "spatial_aggregations": [], "temporal_aggregations": [], **overrides}


def test_declared_temporal_aggregation_must_have_run_with_that_method(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "daily", temporal_aggregation="sum")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    with pytest.raises(ValueError, match="no temporal aggregation to that period ran"):
        check_execution_declarations(resolved, _provenance())
    with pytest.raises(ValueError, match="used 'mean'"):
        check_execution_declarations(
            resolved, _provenance(temporal_aggregations=[{"period": "monthly", "method": "mean", "incomplete": []}])
        )
    check_execution_declarations(
        resolved, _provenance(temporal_aggregations=[{"period": "monthly", "method": "sum", "incomplete": []}])
    )


def test_incomplete_periods_are_refused_unless_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _provenance(temporal_aggregations=[{"period": "monthly", "method": "sum", "incomplete": ["202503"]}])
    _configure(monkeypatch, "daily", temporal_aggregation="sum")
    with pytest.raises(ValueError, match="does not fully cover: 202503"):
        check_execution_declarations(resolve_named_export("DHIS2JSON", {"export": "rain"}), observed)
    _configure(monkeypatch, "daily", temporal_aggregation="sum", incomplete_periods="drop")
    check_execution_declarations(resolve_named_export("DHIS2JSON", {"export": "rain"}), observed)


def test_undeclared_export_refuses_an_aggregation_to_another_period(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "monthly")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    with pytest.raises(ValueError, match="aggregated to yearly"):
        check_execution_declarations(
            resolved, _provenance(temporal_aggregations=[{"period": "yearly", "method": "sum", "incomplete": []}])
        )


def test_render_refuses_data_spaced_finer_than_the_export_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """The data-based guard: daily rows labelled monthly would collide (CLIM-1139)."""
    _configure(monkeypatch, None)
    frame = pd.DataFrame({"geometry": [_OU_A] * 3, "t": _days("2025-01-01", "2025-01-03"), "rain": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="spaced daily"):
        render_named_export(frame, "DHIS2JSON", {"export": "rain"})


def test_observe_temporal_aggregation_records_method_and_incomplete_periods() -> None:
    with capture_execution({}) as evidence:
        with observe_temporal_aggregation("monthly"):
            record_reduction("sum")
            record_incomplete_periods(["202503"])
    assert evidence.describe()["temporal_aggregations"] == [
        {"period": "monthly", "method": "sum", "incomplete": ["202503"], "completeness": "checked"}
    ]


def _cube(stamps: np.ndarray, value: float = 1.0) -> xr.DataArray:
    return xr.DataArray(
        np.full((len(stamps), 2, 2), value, dtype="float32"),
        dims=("t", "y", "x"),
        coords={"t": stamps, "y": [1.0, 2.0], "x": [1.0, 2.0]},
        name="tp",
    )


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_aggregate_dekads_leaves_the_same_evidence(method: str) -> None:
    """Coverage evidence is independent of the reducer used for dekads-to-months."""
    from open_climate_service.plugins.processes.aggregate_dekads import aggregate_dekads

    stamps = np.array(["2025-01-01", "2025-01-11", "2025-01-21", "2025-02-01", "2025-02-11"], dtype="datetime64[ns]")
    with capture_execution({}) as evidence:
        monthly = aggregate_dekads(_cube(stamps), period="month", method=method)
    assert monthly.sizes["t"] == 2
    assert evidence.describe()["temporal_aggregations"] == [
        {"period": "monthly", "method": method, "incomplete": ["202502"], "completeness": "checked"}
    ]


# --- the ad-hoc path (CLIM-1139) -----------------------------------------------------------------


def test_ad_hoc_week_label_on_daily_data_is_refused_naming_both() -> None:
    ds = _cube(_days("2025-01-01", "2025-03-31")).to_dataset()
    with pytest.raises(ValueError) as caught:
        check_ad_hoc_period_reachability(ds, {"period_type": "week"})
    message = str(caught.value)
    assert "weekly" in message and "daily" in message
    assert "aggregate_temporal_period(period='week')" in message


def test_ad_hoc_matching_label_passes() -> None:
    check_ad_hoc_period_reachability(_cube(_days("2025-01-01", "2025-03-31")).to_dataset(), {"period_type": "day"})


def test_ad_hoc_finer_label_cannot_be_derived() -> None:
    ds = _cube(pd.date_range("2025-01-01", "2025-03-01", freq="MS").values).to_dataset()
    with pytest.raises(ValueError, match="cannot be derived"):
        check_ad_hoc_period_reachability(ds, {"period_type": "day"})


def test_dataarray_cadence_survives_promotion_to_dataset() -> None:
    promoted = stamp_cadence(_cube(_days("2025-01-01", "2025-01-03")), "daily").to_dataset()
    assert cadence_of(promoted) == "daily"
    check_ad_hoc_period_reachability(promoted, {"period_type": "day"})


def test_dataset_with_conflicting_variable_cadences_has_no_cadence() -> None:
    daily = stamp_cadence(_cube(_days("2025-01-01", "2025-01-03")), "daily")
    monthly = stamp_cadence(daily.copy(), "monthly")
    assert cadence_of(xr.Dataset({"daily": daily, "monthly": monthly})) is None


def test_temporal_aggregation_refuses_to_make_data_finer() -> None:
    array = stamp_cadence(
        _cube(pd.date_range("2025-01-01", "2025-03-01", freq="MS").values),
        "monthly",
    )

    class Cube:
        openeo = type("OpenEOMetadata", (), {"temporal_dims": ["t"]})()

        def __init__(self, data: xr.DataArray) -> None:
            self.data = data
            self.attrs = data.attrs
            self.coords = data.coords

        def sortby(self, dimension: str) -> Cube:
            self.data = self.data.sortby(dimension)
            return self

        def __getitem__(self, key: str) -> xr.DataArray:
            return self.data[key]

    monthly = Cube(array)
    called = False

    def original(**_: Any) -> Cube:
        nonlocal called
        called = True
        return monthly

    with pytest.raises(ValueError, match="cannot derive daily data from monthly data"):
        execution._make_sorted_atp(original)(monthly, reducer=lambda data: data, period="day")
    assert called is False


# --- a graph that aggregates explicitly, end to end ------------------------------------------


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}


_GEOMETRIES = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": _OU_A, "geometry": _box(0.5, 0.5, 1.5, 1.5)},
        {"type": "Feature", "id": _OU_B, "geometry": _box(1.5, 1.5, 2.5, 2.5)},
    ],
}


@pytest.fixture
def daily_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[openeo_jobs.OpenEOJobService]:
    """A daily source behind load_collection, declared daily in the registry."""
    monkeypatch.setattr(openeo_jobs, "_JOBS_DIR", tmp_path / "openeo_jobs")
    monkeypatch.setattr(config, "get_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(native_store, "JOBS_DIR", tmp_path / "data" / "jobs")
    monkeypatch.setattr(native_store, "JOBS_INDEX_PATH", tmp_path / "data" / "jobs" / "jobs.json")
    monkeypatch.setattr(workflow_store, "_load_records", lambda: [])
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: {"id": "rain_daily", "period_type": "daily"})
    end: dict[str, Any] = {"end": "2025-02-28", "step": 1}

    def load_collection(id: str | None = None, temporal_extent: Any = None, **_: Any) -> xr.DataArray:
        record_source(str(id), type("A", (), {"artifact_id": "a1", "source_dataset_id": id, "path": None})())
        return execution.stamp_declared_cadence(_cube(_days("2025-01-01", end["end"])[:: int(end["step"])]), str(id))

    base = execution._build_process_registry()
    mocked = execution._RegistryOverlay(base, {"load_collection": Process(spec={}, implementation=load_collection)})
    monkeypatch.setattr(execution, "_build_process_registry", lambda: mocked)
    service = openeo_jobs.OpenEOJobService()
    service.coverage_end = end  # type: ignore[attr-defined]
    try:
        yield service
    finally:
        service.shutdown()
        job_service_module.reset_job_service()


def _export_config(monkeypatch: pytest.MonkeyPatch, **extra: Any) -> None:
    monkeypatch.setattr(
        config,
        "_cache",
        {
            "exports": [
                {
                    "id": "rain-monthly",
                    "plugin": "dhis2",
                    "dataset": "rain_daily",
                    "period_type": "monthly",
                    "aggregation": "mean",
                    "series": [{"data_element": _DATA_ELEMENT}],
                    **extra,
                }
            ]
        },
    )


def _builtin_workflow() -> dict[str, Any]:
    return {
        "process_graph": {
            "agg": {
                "process_id": "aggregate_to_dhis2_json",
                "arguments": {
                    "dataset_id": "rain_daily",
                    "temporal_extent": ["2025-01-01", "2025-03-31"],
                    "geometries": _GEOMETRIES,
                    "export": "rain-monthly",
                    "method": "mean",
                },
                "result": True,
            }
        }
    }


def _explicit_graph(reducer: str = "sum") -> dict[str, Any]:
    """load -> aggregate_temporal_period(month) -> aggregate_spatial_weighted(mean) -> named export."""
    return {
        "process_graph": {
            "load": {
                "process_id": "load_collection",
                "arguments": {"id": "rain_daily", "temporal_extent": ["2025-01-01", "2025-03-31"]},
            },
            "monthly": {
                "process_id": "aggregate_temporal_period",
                "arguments": {
                    "data": {"from_node": "load"},
                    "period": "month",
                    "reducer": {
                        "process_graph": {
                            "r": {
                                "process_id": reducer,
                                "arguments": {"data": {"from_parameter": "data"}},
                                "result": True,
                            }
                        }
                    },
                },
            },
            "zonal": {
                "process_id": "aggregate_spatial_weighted",
                "arguments": {"data": {"from_node": "monthly"}, "geometries": _GEOMETRIES, "reducer": "mean"},
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {"from_node": "zonal"},
                    "format": "DHIS2JSON",
                    "options": {"export": "rain-monthly"},
                },
                "result": True,
            },
        }
    }


def _run(service: openeo_jobs.OpenEOJobService, job_id: str, process: dict[str, Any]) -> OpenEOJobRecord:
    openeo_jobs.store_create_job(
        OpenEOJobRecord(id=job_id, status=OpenEOJobStatus.QUEUED, created=utc_now(), process=process)
    )
    service._execute(job_id)
    record = openeo_jobs.store_get_job(job_id)
    assert record is not None
    return record


def _values(record: OpenEOJobRecord) -> list[dict[str, Any]]:
    return json.loads(Path(str((record.usage or {})["output_path"])).read_text())["dataValues"]


def test_builtin_workflow_exports_only_what_exists_and_refuses_a_daily_source(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export never aggregates: a daily source behind a monthly export is refused, not resampled."""
    _export_config(monkeypatch, temporal_aggregation="sum")
    record = _run(daily_instance, "builtin-daily", _builtin_workflow())
    assert record.status == OpenEOJobStatus.ERROR
    assert "no temporal aggregation to that period ran" in str(record.error_message)


def test_explicit_aggregation_is_verified_and_exported(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export_config(monkeypatch, temporal_aggregation="sum")
    record = _run(daily_instance, "explicit-sum", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    values = _values(record)
    assert {value["period"] for value in values} == {"202501", "202502"}
    assert {value["orgUnit"] for value in values} == {_OU_A, _OU_B}
    # January holds 31 daily ones, so a sum then a spatial mean reads 31.
    assert {value["value"] for value in values if value["period"] == "202501"} == {"31"}


def test_explicit_aggregation_with_the_wrong_reducer_is_refused(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export_config(monkeypatch, temporal_aggregation="sum")
    record = _run(daily_instance, "explicit-mean", _explicit_graph("mean"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "used 'mean'" in str(record.error_message)


def test_incomplete_trailing_month_is_refused_by_default(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export_config(monkeypatch, temporal_aggregation="sum")
    daily_instance.coverage_end["end"] = "2025-03-14"  # type: ignore[attr-defined]
    record = _run(daily_instance, "incomplete-march", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "does not fully cover: 202503" in str(record.error_message)


def test_incomplete_trailing_month_is_dropped_when_declared(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export_config(monkeypatch, temporal_aggregation="sum", incomplete_periods="drop")
    daily_instance.coverage_end["end"] = "2025-03-14"  # type: ignore[attr-defined]
    record = _run(daily_instance, "dropped-march", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    assert {value["period"] for value in _values(record)} == {"202501", "202502"}


def test_export_without_a_declaration_fails_at_resolution(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export_config(monkeypatch)
    record = _run(daily_instance, "undeclared", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "must declare temporal_aggregation" in str(record.error_message)


# --- review findings: regressions --------------------------------------------------------------


def test_a_single_day_does_not_pass_as_a_whole_month(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One timestamp has no spacing to infer, so the declared dataset cadence must fill in."""
    _export_config(monkeypatch, temporal_aggregation="sum")
    daily_instance.coverage_end["end"] = "2025-01-01"  # type: ignore[attr-defined]
    record = _run(daily_instance, "single-day", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "does not fully cover: 202501" in str(record.error_message)


def _chained_graph() -> dict[str, Any]:
    graph = _explicit_graph("sum")
    graph["process_graph"]["load"]["arguments"]["temporal_extent"] = ["2025-01-01", "2025-12-31"]
    graph["process_graph"]["yearly"] = {
        "process_id": "aggregate_temporal_period",
        "arguments": {
            "data": {"from_node": "monthly"},
            "period": "year",
            "reducer": {
                "process_graph": {
                    "r": {"process_id": "sum", "arguments": {"data": {"from_parameter": "data"}}, "result": True}
                }
            },
        },
    }
    graph["process_graph"]["zonal"]["arguments"]["data"] = {"from_node": "yearly"}
    return graph


def test_chained_aggregation_keeps_an_incomplete_month_as_an_incomplete_year(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily to monthly to yearly: the year sees twelve whole months, but December was short of days."""
    _export_config(monkeypatch, period_type="yearly", temporal_aggregation="sum")
    daily_instance.coverage_end["end"] = "2025-12-14"  # type: ignore[attr-defined]
    record = _run(daily_instance, "chained-year", _chained_graph())
    assert record.status == OpenEOJobStatus.ERROR
    assert "does not fully cover: 2025" in str(record.error_message)


def test_incomplete_periods_are_relabelled_at_the_export_period(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _provenance(
        temporal_aggregations=[
            {"period": "monthly", "method": "sum", "incomplete": ["202512"]},
            {"period": "yearly", "method": "sum", "incomplete": []},
        ]
    )
    _configure(monkeypatch, "daily", period_type="yearly", temporal_aggregation="sum")
    with pytest.raises(ValueError, match="does not fully cover: 2025"):
        check_execution_declarations(resolve_named_export("DHIS2JSON", {"export": "rain"}), observed)
    _configure(monkeypatch, "daily", period_type="yearly", temporal_aggregation="sum", incomplete_periods="drop")
    assert incomplete_periods_to_drop(resolve_named_export("DHIS2JSON", {"export": "rain"}), observed) == ["2025"]


def test_period_label_start_round_trips() -> None:
    for label, cadence in [
        ("2025W01", "weekly"),
        ("2025Q3", "quarterly"),
        ("202512", "monthly"),
        ("20250214", "daily"),
        ("2025", "yearly"),
    ]:
        assert export_period_label(period_label_start(label, cadence), cadence) == label


def test_dropping_periods_indexes_the_dimension_the_coordinate_lies_along(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vector cube carries `t(observation)`; isel must index `observation`, not `t`."""
    _configure(monkeypatch, None, incomplete_periods="drop")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    stamps = pd.date_range("2025-01-01", "2025-03-01", freq="MS").values
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("observation",),
        coords={"t": ("observation", stamps), "geometry": ("observation", [_OU_A] * 3)},
        name="rain",
    )
    kept = _drop_periods(data, resolved, ["202503"])
    assert kept.sizes["observation"] == 2
    assert list(pd.DatetimeIndex(kept["t"].values).strftime("%Y%m")) == ["202501", "202502"]


def test_unexportable_aggregation_periods_do_not_leak_none_into_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "monthly")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    with pytest.raises(ValueError) as caught:
        check_execution_declarations(
            resolved, _provenance(temporal_aggregations=[{"period": None, "method": "mean", "incomplete": []}])
        )
    assert "None" not in str(caught.value)
    assert "another period" in str(caught.value)


def test_malformed_incomplete_periods_are_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "daily", temporal_aggregation="sum")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    check_execution_declarations(
        resolved, _provenance(temporal_aggregations=[{"period": "monthly", "method": "sum", "incomplete": None}])
    )
    with pytest.raises(ValueError, match="malformed incomplete periods"):
        check_execution_declarations(
            resolved,
            _provenance(temporal_aggregations=[{"period": "monthly", "method": "sum", "incomplete": "202503"}]),
        )


def test_dekadal_axis_tolerates_duplicates_and_enforces_the_lower_bound() -> None:
    dekads = np.array(["2025-01-01", "2025-01-01", "2025-01-11", "2025-01-21"], dtype="datetime64[ns]")
    assert is_dekadal_axis(dekads)
    monthly_on_the_first = np.array(["2025-01-01", "2025-02-01", "2025-03-01"], dtype="datetime64[ns]")
    assert not is_dekadal_axis(monthly_on_the_first)
    daily_on_dekad_days = np.array(["2025-01-01", "2025-01-11", "2025-01-12"], dtype="datetime64[ns]")
    assert not is_dekadal_axis(daily_on_dekad_days)


def test_completeness_scales_to_years_of_daily_data() -> None:
    three_years = _days("2023-01-01", "2025-12-31")
    assert incomplete_destination_periods(three_years, "daily", "monthly") == []
    assert incomplete_destination_periods(three_years[:-1], "daily", "monthly") == ["202512"]


def test_sparse_daily_input_is_not_read_as_a_coarser_cadence(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other day missing infers as weekly; the declared daily cadence must win and count the gaps."""
    _export_config(monkeypatch, temporal_aggregation="sum")
    daily_instance.coverage_end["end"] = "2025-02-28"  # type: ignore[attr-defined]
    daily_instance.coverage_end["step"] = 2  # type: ignore[attr-defined]
    record = _run(daily_instance, "sparse-daily", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "does not fully cover: 202501, 202502" in str(record.error_message)


def test_unknown_source_cadence_cannot_pass_as_complete(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing declared and a sparse axis, completeness is unverifiable and the export refuses."""
    _export_config(monkeypatch, temporal_aggregation="sum")
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: None)
    daily_instance.coverage_end["step"] = 2  # type: ignore[attr-defined]
    record = _run(daily_instance, "unknown-cadence", _explicit_graph("sum"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "cannot verify that its monthly periods are complete" in str(record.error_message)


def test_unverifiable_completeness_is_refused_even_with_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "daily", temporal_aggregation="sum", incomplete_periods="drop")
    resolved = resolve_named_export("DHIS2JSON", {"export": "rain"})
    observed = _provenance(
        temporal_aggregations=[
            {"period": "monthly", "method": "sum", "incomplete": [], "completeness": "unknown", "reason": "no axis"}
        ]
    )
    with pytest.raises(ValueError, match="cannot verify that its monthly periods are complete: no axis"):
        check_execution_declarations(resolved, observed)


def test_ad_hoc_check_trusts_the_carried_cadence_over_a_sparse_axis() -> None:
    sparse = stamp_cadence(_cube(_days("2025-01-01", "2025-02-28")[::2]).to_dataset(), "daily")
    check_ad_hoc_period_reachability(sparse, {"period_type": "day"})
    with pytest.raises(ValueError, match="coarser than the result's daily spacing"):
        check_ad_hoc_period_reachability(sparse, {"period_type": "month"})


def test_ad_hoc_check_reads_the_result_cadence_not_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily, aggregated to months: `month` is right and `day` would relabel totals as days."""
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: {"id": dataset_id, "period_type": "daily"})
    daily = execution.stamp_declared_cadence(_cube(_days("2025-01-01", "2025-02-28")), "rain_daily")
    monthly = daily.resample(t="MS").sum()
    assert cadence_of(daily) == "daily"
    stamp_cadence(monthly, "monthly")  # what the aggregate_temporal_period wrapper does to its result
    check_ad_hoc_period_reachability(monthly.to_dataset(), {"period_type": "month"})
    with pytest.raises(ValueError, match="cannot be derived"):
        check_ad_hoc_period_reachability(monthly.to_dataset(), {"period_type": "day"})


def test_load_stamps_the_declared_cadence_and_steps_rewrite_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: {"id": dataset_id, "period_type": "daily"})
    assert cadence_of(execution.stamp_declared_cadence(_cube(_days("2025-01-01", "2025-01-03")), "x")) == "daily"
    monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: None)
    assert cadence_of(execution.stamp_declared_cadence(_cube(_days("2025-01-01", "2025-01-03")), "x")) is None
    from open_climate_service.plugins.processes.aggregate_dekads import aggregate_dekads

    dekads = np.array(["2025-01-01", "2025-01-11", "2025-01-21"], dtype="datetime64[ns]")
    assert cadence_of(aggregate_dekads(stamp_cadence(_cube(dekads), "dekadal"), period="month")) == "monthly"


def test_complete_chained_aggregation_is_exported(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily to monthly to yearly over a whole year: the yearly step must count months, not days."""
    _export_config(monkeypatch, period_type="yearly", temporal_aggregation="sum")
    daily_instance.coverage_end["end"] = "2025-12-31"  # type: ignore[attr-defined]
    record = _run(daily_instance, "chained-complete", _chained_graph())
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    values = _values(record)
    assert {value["period"] for value in values} == {"2025"}
    assert {value["value"] for value in values} == {"365"}


def _ad_hoc_chap_graph(period_type: str) -> dict[str, Any]:
    graph = _explicit_graph("sum")
    graph["process_graph"]["save"]["arguments"] = {
        "data": {"from_node": "zonal"},
        "format": "CHAPCSV",
        "options": {"period_type": period_type},
    }
    return graph


def test_ad_hoc_export_validates_the_result_cadence(
    daily_instance: openeo_jobs.OpenEOJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an explicit monthly aggregation, `month` exports and `day` is refused as not derivable."""
    _export_config(monkeypatch, temporal_aggregation="sum")
    record = _run(daily_instance, "adhoc-month", _ad_hoc_chap_graph("month"))
    assert record.status == OpenEOJobStatus.FINISHED, record.error_message
    rows = Path(str((record.usage or {})["output_path"])).read_text().splitlines()
    assert rows[0] == "time_period,location,tp"
    assert {row.split(",")[0] for row in rows[1:]} == {"202501", "202502"}
    record = _run(daily_instance, "adhoc-day", _ad_hoc_chap_graph("day"))
    assert record.status == OpenEOJobStatus.ERROR
    assert "cannot be derived from monthly data" in str(record.error_message)
