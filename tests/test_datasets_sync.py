from datetime import UTC, date, datetime, tzinfo
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from open_climate_service.ingestions import services, sync_engine
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    DatasetDetailRecord,
    DatasetPublication,
    PublicationStatus,
    SyncAction,
    SyncDetail,
    SyncKind,
    SyncResponse,
)
from open_climate_service.shared.time import next_period_string


def _artifact(
    *,
    artifact_id: str,
    source_dataset_id: str = "chirps3_precipitation_daily",
    managed_dataset_id: str = "chirps3_precipitation_daily_sle",
    created_at: str = "2026-01-10T00:00:00+00:00",
    end: str = "2026-01-10",
    path: str = "/tmp/chirps3_precipitation_daily.icechunk",
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=source_dataset_id,
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        format=ArtifactFormat.ICECHUNK,
        path=path,
        asset_paths=[path],
        variables=["precip"],
        request_scope=ArtifactRequestScope(
            start="2026-01-01",
            end=end,
            bbox=(1.0, 2.0, 3.0, 4.0),
        ),
        coverage=ArtifactCoverage(
            temporal=CoverageTemporal(start="2026-01-01", end=end),
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
        ),
        created_at=datetime.fromisoformat(created_at),
        publication=ArtifactPublication(
            status=PublicationStatus.PUBLISHED,
            collection_id=managed_dataset_id,
        ),
    )


def _dataset_detail(dataset_id: str) -> DatasetDetailRecord:
    return DatasetDetailRecord(
        dataset_id=dataset_id,
        source_dataset_id="chirps3_precipitation_daily",
        dataset_name="CHIRPS3 precipitation",
        short_name="CHIRPS3 precip",
        variable="precip",
        period_type="daily",
        units="mm",
        resolution="5 km x 5 km",
        source="CHIRPS v3",
        source_url="https://example.com/chirps",
        extent=ArtifactCoverage(
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-11"),
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
        ),
        last_updated=datetime(2026, 1, 11, tzinfo=UTC),
        links=[],
        publication=DatasetPublication(
            status=PublicationStatus.PUBLISHED,
            published_at=datetime(2026, 1, 11, tzinfo=UTC),
        ),
        versions=[],
    )


def test_sync_dataset_returns_up_to_date_when_no_new_period_is_due(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    monkeypatch.setattr(
        services,
        "get_latest_artifact_for_dataset_or_404",
        lambda _: _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31"),
    )
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal"},
            "ingestion": {},
        },
    )
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))

    result = services.sync_dataset(dataset_id=dataset_id, end="2026-01-31", publish=True)

    assert result.sync_id is None
    assert result.status == "up_to_date"
    assert result.message == "Managed dataset is already current with upstream source state."
    assert result.dataset is not None
    assert result.dataset.dataset_id == dataset_id
    assert result.sync_detail is not None
    assert result.sync_detail.sync_kind == SyncKind.TEMPORAL
    assert result.sync_detail.action == SyncAction.NO_OP
    assert result.sync_detail.reason == "no_new_period"
    assert (
        result.sync_detail.message
        == "Data already exists through 2026-01-31; target 2026-01-31 does not require a new download."
    )


def test_sync_dataset_creates_new_version_from_next_period(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31")
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
    )

    captured: dict[str, object] = {}

    def fake_create_artifact(**kwargs: object) -> ArtifactRecord:
        captured.update(kwargs)
        return _artifact(artifact_id="a2", managed_dataset_id=dataset_id, end="2026-02-10")

    monkeypatch.setattr(services, "create_artifact", fake_create_artifact)
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))
    result = services.sync_dataset(dataset_id=dataset_id, end="2026-02-10", publish=True)

    assert captured["start"] == "2026-01-01"
    assert captured["end"] == "2026-02-10"
    assert captured["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert captured["country_code"] == "SLE"
    assert result.sync_id == "a2"
    assert result.status == "completed"
    assert result.message == "Managed dataset was rematerialized against the latest planned upstream state."
    assert result.sync_detail is not None
    assert result.sync_detail.sync_kind == SyncKind.TEMPORAL
    assert result.sync_detail.action == SyncAction.REMATERIALIZE
    assert result.sync_detail.reason == "new_periods_available"
    assert "Data exists through 2026-01-31" in result.sync_detail.message
    assert "Sync will rematerialize the dataset through 2026-02-10" in result.sync_detail.message
    assert result.sync_detail.current_start == "2026-01-01"
    assert result.sync_detail.current_end == "2026-01-31"
    assert result.sync_detail.target_end == "2026-02-10"
    assert result.sync_detail.target_end_source == "request"
    assert result.sync_detail.delta_start == "2026-02-01"
    assert result.sync_detail.delta_end == "2026-02-10"


def test_sync_dataset_append_policy_falls_back_to_rematerialize_without_icechunk_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "worldpop_population_yearly.zarr"

    dataset_id = "worldpop_population_yearly_sle"
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="worldpop_population_yearly",
        managed_dataset_id=dataset_id,
        end="2024",
        path=str(zarr_path),
    )
    latest.format = ArtifactFormat.ZARR  # legacy download-path artifact, not Icechunk
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "worldpop_population_yearly",
            "period_type": "yearly",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {},
        },
    )

    captured: dict[str, object] = {}

    def fake_create_artifact(**kwargs: object) -> ArtifactRecord:
        captured.update(kwargs)
        return _artifact(
            artifact_id="a2",
            source_dataset_id="worldpop_population_yearly",
            managed_dataset_id=dataset_id,
            end="2025",
        )

    messages: list[str] = []

    def fake_info(message: str, *args: object) -> None:
        messages.append(message % args if args else message)

    monkeypatch.setattr(services, "create_artifact", fake_create_artifact)
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))
    monkeypatch.setattr(sync_engine.logger, "info", fake_info)

    result = services.sync_dataset(dataset_id=dataset_id, end="2025", publish=True)

    assert "download_start" in captured
    assert captured["download_start"] is None
    assert result.sync_detail is not None
    assert result.sync_detail.action == SyncAction.REMATERIALIZE
    assert any("requires an existing Icechunk artifact" in message for message in messages)


def test_sync_dataset_append_policy_uses_store_based_append_for_plugin_backed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id=dataset_id,
        end="2026-01-31",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
    )

    captured: dict[str, object] = {}

    def fake_create_artifact(**kwargs: object) -> ArtifactRecord:
        captured.update(kwargs)
        return _artifact(artifact_id="a2", managed_dataset_id=dataset_id, end="2026-02-10")

    monkeypatch.setattr(services, "create_artifact", fake_create_artifact)
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))

    result = services.sync_dataset(dataset_id=dataset_id, end="2026-02-10", publish=True)

    assert "download_start" in captured
    assert captured["download_start"] == "2026-02-01"
    assert captured["download_end"] == "2026-02-10"
    assert result.sync_detail is not None
    assert result.sync_detail.action == SyncAction.APPEND
    assert result.sync_detail.reason == "new_periods_available_for_append"
    assert "Data exists through 2026-01-31" in result.sync_detail.message
    assert "Sync will append missing periods 2026-02-01 through 2026-02-10" in result.sync_detail.message
    assert "extend coverage through 2026-02-10" in result.sync_detail.message
    assert result.message is not None
    assert "appending missing periods" in result.message
    assert "committed store" in result.message


def test_plan_sync_for_plugin_backed_icechunk_uses_committed_store_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-31",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK
    latest.coverage.temporal.end = "2026-01-15"

    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/tmp").resolve(),))
    monkeypatch.setattr(sync_engine, "read_committed_period_ids", lambda *args, **kwargs: {"2026-01-31"})

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.NO_OP
    assert result.reason == "no_new_period"
    assert result.current_end == "2026-01-31"


def test_plan_sync_uses_cumulative_coverage_start_not_latest_request_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-03",
    )
    latest.request_scope.start = "2026-01-03"
    latest.coverage.temporal.start = "2026-01-01"
    monkeypatch.setattr(sync_engine, "_query_available_periods", lambda *args, **kwargs: ["2026-01-04"])

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "example.Plugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-04",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_start == "2026-01-01"


def test_plan_sync_marks_static_non_temporal_dataset_not_syncable() -> None:
    # A static, non-temporal dataset (e.g. a day-of-year climatology) has no temporal
    # end. Planning must return NOT_SYNCABLE, not raise/400 by trying to compute one.
    latest = _artifact(artifact_id="c1", managed_dataset_id="era5land_temperature_normal_sle")
    latest.coverage.temporal.start = None
    latest.coverage.temporal.end = None

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "era5land_temperature_daily_normal_1991_2020",
            "period_type": "climatology",
            "sync": {"kind": "static"},
        },
        latest_artifact=latest,
        requested_end=None,
    )

    assert result.action == SyncAction.NOT_SYNCABLE
    assert result.reason == "static_dataset"
    assert result.current_end is None


def test_plan_sync_for_plugin_backed_icechunk_normalizes_committed_period_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-04-21T13",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK
    latest.coverage.temporal.end = "2026-04-21T13"

    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/tmp").resolve(),))
    monkeypatch.setattr(
        sync_engine,
        "read_committed_period_ids",
        lambda *args, **kwargs: {"2026-04-21T13:27:45"},
    )

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "era5land_temperature_hourly",
            "period_type": "hourly",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "example.HourlyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-04-21T13",
    )

    assert result.action == SyncAction.NO_OP
    assert result.reason == "no_new_period"
    assert result.current_end == "2026-04-21T13"


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_without_store_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
    )
    latest.format = ArtifactFormat.ICECHUNK
    latest.path = None
    latest.asset_paths = []

    read_calls: list[tuple[object, ...]] = []

    def fake_read_committed_period_ids(*args: object, **kwargs: object) -> set[str]:
        read_calls.append(args)
        return {"2026-01-31"}

    monkeypatch.setattr(sync_engine, "read_committed_period_ids", fake_read_committed_period_ids)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert read_calls == []


def test_plan_sync_for_plugin_backed_icechunk_skips_non_local_store_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="s3://bucket/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    read_calls: list[tuple[object, ...]] = []
    warnings: list[str] = []

    def fake_read_committed_period_ids(*args: object, **kwargs: object) -> set[str]:
        read_calls.append(args)
        return {"2026-01-31"}

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine, "read_committed_period_ids", fake_read_committed_period_ids)
    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert read_calls == []
    assert any("non-local URI" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_for_windows_drive_letter_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows drive-letter paths (C:\...) are detected explicitly before urlparse
    # can misread the drive letter as a URI scheme. On Linux the path is not
    # absolute, so _resolve_local_artifact_path returns "relative path" and
    # committed-store inspection falls back to artifact metadata.
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="C:\\data\\downloads\\chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    read_calls: list[tuple[object, ...]] = []
    warnings: list[str] = []

    def fake_read_committed_period_ids(*args: object, **kwargs: object) -> set[str]:
        read_calls.append(args)
        return {"2026-01-31"}

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine, "read_committed_period_ids", fake_read_committed_period_ids)
    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert read_calls == []
    assert any("relative path" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_for_file_uri_with_windows_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # file:///C:/... URIs are valid on Windows but not within any trusted storage
    # root on this Linux service. urlparse produces path "/C:/..." which is
    # absolute on Linux but won't match any configured storage root, so
    # _resolve_local_artifact_path returns "untrusted local path" and planning
    # falls back to artifact metadata.
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="file:///C:/data/downloads/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    read_calls: list[tuple[object, ...]] = []
    warnings: list[str] = []

    def fake_read_committed_period_ids(*args: object, **kwargs: object) -> set[str]:
        read_calls.append(args)
        return {"2026-01-31"}

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine, "read_committed_period_ids", fake_read_committed_period_ids)
    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert read_calls == []
    assert any("untrusted local path" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_when_store_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    warnings: list[str] = []
    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/tmp").resolve(),))

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(
        sync_engine,
        "read_committed_period_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert any("falling back to artifact metadata" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_when_committed_set_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/tmp").resolve(),))
    monkeypatch.setattr(sync_engine, "read_committed_period_ids", lambda *args, **kwargs: set())
    # Avoid a real network call from CHIRPS3DailyPlugin._availability_cutoff() —
    # this test is about the committed-set fallback, not plugin availability.
    monkeypatch.setattr(sync_engine, "_query_available_periods", lambda *args, **kwargs: ["2026-01-31"])

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_for_untrusted_local_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    warnings: list[str] = []

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)
    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/srv/app/data/downloads"),))

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert any("untrusted local path" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_for_relative_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="data/downloads/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    warnings: list[str] = []

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert any("relative path" in message for message in warnings)


def test_plan_sync_for_plugin_backed_icechunk_falls_back_to_artifact_end_when_committed_period_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-01-15",
        path="/tmp/chirps3_precipitation_daily.icechunk",
    )
    latest.format = ArtifactFormat.ICECHUNK

    warnings: list[str] = []

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (Path("/tmp").resolve(),))
    monkeypatch.setattr(sync_engine, "read_committed_period_ids", lambda *args, **kwargs: {"not-a-period"})
    monkeypatch.setattr(sync_engine.logger, "warning", fake_warning)

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND
    assert result.current_end == "2026-01-15"
    assert any("malformed committed periods" in message for message in warnings)


def test_sync_dataset_append_policy_falls_back_for_plugin_backed_non_icechunk_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(
        artifact_id="a1",
        managed_dataset_id=dataset_id,
        end="2026-01-31",
        path="/tmp/chirps3_precipitation_daily.zarr",
    )
    latest.format = ArtifactFormat.ZARR
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
    )

    captured: dict[str, object] = {}
    infos: list[str] = []

    def fake_create_artifact(**kwargs: object) -> ArtifactRecord:
        captured.update(kwargs)
        return _artifact(artifact_id="a2", managed_dataset_id=dataset_id, end="2026-02-10")

    def fake_info(message: str, *args: object) -> None:
        infos.append(message % args if args else message)

    monkeypatch.setattr(services, "create_artifact", fake_create_artifact)
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))
    monkeypatch.setattr(sync_engine.logger, "info", fake_info)

    result = services.sync_dataset(dataset_id=dataset_id, end="2026-02-10", publish=True)

    assert "download_start" in captured
    assert captured["download_start"] is None
    assert captured["download_end"] is None
    assert result.sync_detail is not None
    assert result.sync_detail.action == SyncAction.REMATERIALIZE
    assert result.sync_detail.reason == "new_periods_available"
    assert any("requires an existing Icechunk artifact" in message for message in infos)


def test_sync_dataset_release_policy_returns_up_to_date_when_release_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = "worldpop_population_yearly_sle"
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="worldpop_population_yearly",
        managed_dataset_id=dataset_id,
        end="2024",
    )
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "worldpop_population_yearly", "period_type": "yearly", "sync": {"kind": "release"}},
    )
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))

    result = services.sync_dataset(dataset_id=dataset_id, end="2024", publish=True)

    assert result.sync_id is None
    assert result.status == "up_to_date"
    assert result.sync_detail is not None
    assert result.sync_detail.sync_kind == SyncKind.RELEASE
    assert result.sync_detail.action == SyncAction.NO_OP
    assert result.sync_detail.reason == "no_new_release"


def test_default_hourly_target_end_is_utc_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "FixedDateTime":
            return cls(2026, 4, 21, 13, 47, 31, tzinfo=tz if tz is UTC else None)

    monkeypatch.setattr(sync_engine, "utc_now", lambda: FixedDateTime(2026, 4, 21, 13, 47, 31, tzinfo=UTC))

    result = sync_engine._default_target_end(period_type="hourly")

    assert result == "2026-04-21T13"


def test_default_target_end_rejects_unsupported_period_type() -> None:
    with pytest.raises(ValueError, match="Unsupported period_type 'fortnightly' for sync"):
        sync_engine._default_target_end(period_type="fortnightly")


def test_next_period_start_preserves_hourly_period_format() -> None:
    result = next_period_string("2026-04-21T13", "hourly")

    assert result == "2026-04-21T14"


def test_default_weekly_target_end_uses_iso_week_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync_engine, "utc_now", lambda: datetime(2026, 4, 21, 13, 47, 31, tzinfo=UTC))

    result = sync_engine._default_target_end(period_type="weekly")

    assert result == "2026-W17"


def test_next_period_start_preserves_weekly_period_format() -> None:
    result = next_period_string("2026-W17", "weekly")

    assert result == "2026-W18"


def test_next_period_start_rolls_weekly_period_across_iso_year_boundary() -> None:
    result = next_period_string("2020-W53", "weekly")

    assert result == "2021-W01"


def test_sync_dataset_static_policy_returns_not_syncable_without_period_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "static_dataset_sle"
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="static_dataset",
        managed_dataset_id=dataset_id,
        end="static-release",
    )
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "static_dataset", "period_type": "unsupported-static-period", "sync": {"kind": "static"}},
    )
    monkeypatch.setattr(services, "create_artifact", lambda **_: pytest.fail("static sync should not create artifacts"))
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))

    result = services.sync_dataset(dataset_id=dataset_id, end="ignored", publish=True)

    assert result.sync_id is None
    assert result.status == "not_syncable"
    assert result.sync_detail is not None
    assert result.sync_detail.sync_kind == SyncKind.STATIC
    assert result.sync_detail.action == SyncAction.NOT_SYNCABLE
    assert result.sync_detail.reason == "static_dataset"


def test_plan_sync_requires_sync_kind() -> None:
    latest = _artifact(artifact_id="a1", end="2026-01-31")

    with pytest.raises(ValueError, match="must define sync.kind"):
        sync_engine.plan_sync(
            source_dataset={"id": "chirps3_precipitation_daily", "period_type": "daily"},
            latest_artifact=latest,
            requested_end="2026-02-10",
        )


def test_plan_sync_static_policy_ignores_period_normalization() -> None:
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="static_dataset",
        managed_dataset_id="static_dataset_sle",
        end="static-release",
    )

    result = sync_engine.plan_sync(
        source_dataset={"id": "static_dataset", "period_type": "unsupported-static-period", "sync": {"kind": "static"}},
        latest_artifact=latest,
        requested_end=None,
    )

    assert result.sync_kind == SyncKind.STATIC
    assert result.action == SyncAction.NOT_SYNCABLE
    assert result.reason == "static_dataset"


def test_plan_sync_dataset_returns_plan_without_creating_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31")
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
    )
    monkeypatch.setattr(services, "create_artifact", lambda **_: pytest.fail("sync plan should not create artifacts"))

    result = services.plan_sync_dataset(dataset_id=dataset_id, end="2026-02-10")

    assert result.sync_kind == SyncKind.TEMPORAL
    assert result.action == SyncAction.REMATERIALIZE
    assert result.reason == "new_periods_available"
    assert result.current_start == "2026-01-01"
    assert result.current_end == "2026-01-31"
    assert result.target_end == "2026-02-10"
    assert result.target_end_source == "request"
    assert result.delta_start == "2026-02-01"
    assert result.delta_end == "2026-02-10"


def test_sync_plan_route_returns_plan_without_creating_artifact(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31")
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
    )
    monkeypatch.setattr(services, "create_artifact", lambda **_: pytest.fail("sync plan should not create artifacts"))

    response = client.get(f"/sync/{dataset_id}/plan", params={"end": "2026-02-10"})

    assert response.status_code == 200
    assert response.json() == {
        "source_dataset_id": "chirps3_precipitation_daily",
        "sync_kind": "temporal",
        "action": "rematerialize",
        "reason": "new_periods_available",
        "message": "Data exists through 2026-01-31. Sync will rematerialize the dataset through 2026-02-10.",
        "current_start": "2026-01-01",
        "current_end": "2026-01-31",
        "target_end": "2026-02-10",
        "target_end_source": "request",
        "delta_start": "2026-02-01",
        "delta_end": "2026-02-10",
    }


def test_sync_plan_route_returns_400_for_invalid_end_period(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31")
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
    )

    response = client.get(f"/sync/{dataset_id}/plan", params={"end": "not-a-period"})

    assert response.status_code == 400


def test_plan_sync_treats_blank_end_as_default_target(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 4, 20)

    monkeypatch.setattr(sync_engine, "utc_today", lambda: FixedDate(2026, 4, 20))

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {},
        },
        latest_artifact=_artifact(artifact_id="a1", end="2024-02-29"),
        requested_end="",
    )

    assert result.target_end == "2026-04-20"
    assert result.target_end_source == "default_today"


def test_sync_route_executes_rematerialize_and_returns_structured_detail(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "chirps3_precipitation_daily_sle"
    latest = _artifact(artifact_id="a1", managed_dataset_id=dataset_id, end="2026-01-31")
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
    )
    monkeypatch.setattr(
        services,
        "create_artifact",
        lambda **_: _artifact(artifact_id="a2", managed_dataset_id=dataset_id, end="2026-02-10"),
    )
    monkeypatch.setattr(services, "get_dataset_or_404", lambda _: _dataset_detail(dataset_id))

    response = client.post(
        f"/sync/{dataset_id}",
        json={"end": "2026-02-10", "publish": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sync_id"] == "a2"
    assert payload["status"] == "completed"
    assert payload["dataset"]["dataset_id"] == dataset_id
    assert payload["sync_detail"]["sync_kind"] == "temporal"
    assert payload["sync_detail"]["action"] == "rematerialize"
    assert payload["sync_detail"]["target_end"] == "2026-02-10"


def test_plan_sync_marks_default_target_end_source(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 4, 20)

    monkeypatch.setattr(sync_engine, "utc_today", lambda: FixedDate(2026, 4, 20))

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {},
        },
        latest_artifact=_artifact(artifact_id="a1", end="2024-02-29"),
        requested_end=None,
    )

    assert result.target_end == "2026-04-20"
    assert result.target_end_source == "default_today"
    assert result.delta_start == "2024-03-01"
    assert result.delta_end == "2026-04-20"


def test_run_sync_raises_clear_error_when_append_invariants_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    latest_artifact = _artifact(
        artifact_id="a1",
        source_dataset_id="chirps3_precipitation_daily",
        managed_dataset_id="chirps3_precipitation_daily_sle",
        end="2026-02-10",
    )

    broken_plan = SyncDetail(
        source_dataset_id="chirps3_precipitation_daily",
        sync_kind=SyncKind.TEMPORAL,
        action=SyncAction.APPEND,
        reason="new_periods_available_for_append",
        message="broken test plan",
        current_start=None,
        current_end="2026-02-10",
        target_end="2026-02-11",
        target_end_source="request",
        delta_start="2026-02-11",
        delta_end="2026-02-11",
    )
    monkeypatch.setattr(sync_engine, "plan_sync", lambda **_: broken_plan)

    with pytest.raises(ValueError, match="Sync execution requires current_start"):
        sync_engine.run_sync(
            latest_artifact=latest_artifact,
            source_dataset={"id": "chirps3_precipitation_daily", "period_type": "daily", "sync": {"kind": "temporal"}},
            requested_end="2026-02-11",
            country_code=None,
            publish=True,
            create_artifact_fn=lambda **_: pytest.fail("create_artifact should not be called"),
            get_dataset_fn=lambda _: pytest.fail("get_dataset should not be called"),
        )


def test_run_sync_preserves_rematerialize_action_as_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="worldpop_population_yearly",
        managed_dataset_id="worldpop_population_yearly_sle",
        end="2024",
    )
    plan = SyncDetail(
        source_dataset_id="worldpop_population_yearly",
        sync_kind=SyncKind.RELEASE,
        action=SyncAction.REMATERIALIZE,
        reason="new_release_available",
        message="new release",
        current_start="2020",
        current_end="2024",
        target_end="2025",
        target_end_source="request",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(sync_engine, "plan_sync", lambda **kwargs: plan)

    def fake_create_artifact(**kwargs: object) -> ArtifactRecord:
        captured.update(kwargs)
        return _artifact(
            artifact_id="a2",
            source_dataset_id="worldpop_population_yearly",
            managed_dataset_id="worldpop_population_yearly_sle",
            end="2025",
        )

    sync_engine.run_sync(
        latest_artifact=latest,
        source_dataset={"id": "worldpop_population_yearly", "period_type": "yearly", "sync": {"kind": "release"}},
        requested_end="2025",
        country_code="SLE",
        publish=False,
        create_artifact_fn=fake_create_artifact,
        get_dataset_fn=lambda dataset_id: _dataset_detail(dataset_id),
    )

    assert captured["overwrite"] is True
    assert captured["periods"] is None


def test_sync_dataset_forwards_country_code_from_extent(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = "worldpop_population_yearly_sle"
    latest = _artifact(
        artifact_id="a1",
        source_dataset_id="worldpop_population_yearly",
        managed_dataset_id=dataset_id,
        end="2020",
    )
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: latest)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {"id": "worldpop_population_yearly", "period_type": "yearly", "sync": {"kind": "release"}},
    )
    monkeypatch.setattr(
        services,
        "get_extent",
        lambda: {"id": "sle", "bbox": [-13.5, 6.9, -10.1, 10.0], "country_code": "SLE"},
    )

    captured: dict[str, object] = {}

    def fake_run_sync(**kwargs: object) -> SyncResponse:
        captured.update(kwargs)
        return SyncResponse(
            sync_id="a2",
            status="completed",
            message="ok",
            dataset=_dataset_detail(dataset_id),
            sync_detail=SyncDetail(
                source_dataset_id="worldpop_population_yearly",
                sync_kind=SyncKind.RELEASE,
                action=SyncAction.REMATERIALIZE,
                reason="new_release_available",
                message="ok",
                current_start="2020",
                current_end="2020",
                target_end="2021",
                target_end_source="request",
            ),
        )

    monkeypatch.setattr(services, "run_sync", fake_run_sync)

    services.sync_dataset(dataset_id=dataset_id, end="2021", publish=True)

    assert captured["country_code"] == "SLE"


# ---------------------------------------------------------------------------
# Pyramid promotion tests
# ---------------------------------------------------------------------------


def test_maybe_build_pyramid_calls_write_to_icechunk_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import xarray as xr

    from open_climate_service.data_manager.services import downloader
    from open_climate_service.ingestions.services import _maybe_build_pyramid

    icechunk_path = tmp_path / "dataset.icechunk"
    icechunk_path.mkdir()

    ds = xr.Dataset()
    monkeypatch.setattr(
        "open_climate_service.data_accessor.services.accessor.open_icechunk_dataset",
        lambda _: ds,
    )

    written: list[tuple] = []

    def fake_write(ds_arg: xr.Dataset, path: Path, *a: object, **kw: object) -> None:
        written.append((ds_arg, path))
        path.mkdir(parents=True, exist_ok=True)  # a real write leaves a store behind

    monkeypatch.setattr(downloader, "write_to_icechunk_store", fake_write)

    result = _maybe_build_pyramid(icechunk_path, {"id": "ds1", "variable": "precip"})

    assert len(written) == 1
    # The rewrite cannot target the store it is reading from, so it goes to a sibling and is
    # swapped in. What matters to callers is where it ends up, and that nothing is left over.
    assert written[0][1] == icechunk_path.with_name(f"{icechunk_path.name}.rebuild")
    assert icechunk_path.is_dir()
    assert not written[0][1].exists()
    assert not icechunk_path.with_name(f"{icechunk_path.name}.retired").exists()
    assert result.completed is True
    assert result.swapped is True


def test_maybe_build_pyramid_skips_rewrite_for_normalized_flat_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temporal append to an already-normalized flat store needs no read-rewrite."""
    import numpy as np
    import xarray as xr

    from open_climate_service.data_manager.services import downloader
    from open_climate_service.ingestions.services import _maybe_build_pyramid

    icechunk_path = tmp_path / "dataset.icechunk"
    icechunk_path.mkdir()

    # Small flat store that already carries the spatial_ref CRS coordinate, i.e.
    # the output of a prior write_to_icechunk_store normalization.
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((2, 4, 5), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "spatial_ref": 0,
        },
    )
    monkeypatch.setattr(
        "open_climate_service.data_accessor.services.accessor.open_icechunk_dataset",
        lambda _: ds,
    )

    written: list[tuple] = []
    monkeypatch.setattr(downloader, "write_to_icechunk_store", lambda *a, **kw: written.append(a))

    result = _maybe_build_pyramid(icechunk_path, {"id": "ds1", "variable": "precip"})

    assert written == []  # already GeoZarr-normalized and flat → no rewrite
    assert result.completed is True
    assert result.swapped is False


def test_maybe_build_pyramid_falls_back_on_build_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import xarray as xr

    from open_climate_service.data_manager.services import downloader
    from open_climate_service.ingestions.services import _maybe_build_pyramid

    icechunk_path = tmp_path / "dataset.icechunk"
    icechunk_path.mkdir()

    ds = xr.Dataset()
    monkeypatch.setattr(
        "open_climate_service.data_accessor.services.accessor.open_icechunk_dataset",
        lambda _: ds,
    )

    def fake_write_fail(*_a: object, **_k: object) -> None:
        raise RuntimeError("fail")

    monkeypatch.setattr(downloader, "write_to_icechunk_store", fake_write_fail)

    # Must not raise — errors are swallowed so the flat artifact is still registered.
    result = _maybe_build_pyramid(icechunk_path, {"id": "ds1", "variable": "v"})

    assert result.completed is False
    assert result.swapped is False


def test_plan_sync_append_for_icechunk_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    icechunk_path = tmp_path / "dataset.icechunk"
    icechunk_path.mkdir()

    latest = _artifact(artifact_id="a1", managed_dataset_id="chirps3_precipitation_daily_sle", end="2026-01-15")
    latest.format = ArtifactFormat.ICECHUNK
    latest.path = str(icechunk_path)
    latest.asset_paths = [str(icechunk_path)]

    monkeypatch.setattr(sync_engine, "read_committed_period_ids", lambda *_a, **_k: {"2026-01-15"})

    result = sync_engine.plan_sync(
        source_dataset={
            "id": "chirps3_precipitation_daily",
            "period_type": "daily",
            "sync": {"kind": "temporal", "execution": "append"},
            "ingestion": {"plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"},
        },
        latest_artifact=latest,
        requested_end="2026-01-31",
    )

    assert result.action == SyncAction.APPEND


def test_swap_store_replaces_the_target_and_cleans_up(tmp_path: Path) -> None:
    from open_climate_service.ingestions.services import _swap_store

    target = tmp_path / "ds.icechunk"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "ds.icechunk.rebuild"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    _swap_store(staging, target)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (target / "old.txt").exists()
    assert not staging.exists()
    assert not (tmp_path / "ds.icechunk.retired").exists()


def test_swap_store_puts_the_original_back_when_the_swap_fails(tmp_path: Path) -> None:
    """A failed swap must not leave the dataset without a store."""
    from open_climate_service.ingestions.services import _swap_store

    target = tmp_path / "ds.icechunk"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    missing_staging = tmp_path / "ds.icechunk.rebuild"  # never created -> rename raises

    with pytest.raises(OSError):
        _swap_store(missing_staging, target)

    # The original survived, contents intact.
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"


def test_recover_interrupted_swap_restores_a_store_killed_between_renames(tmp_path: Path) -> None:
    """A SIGKILL between the two renames leaves the published path missing and the data at
    `.retired`, which no reader looks for. The exception handler cannot cover that case.
    """
    from open_climate_service.ingestions.services import recover_interrupted_swap

    target = tmp_path / "ds.icechunk"
    retired = tmp_path / "ds.icechunk.retired"
    retired.mkdir()
    (retired / "data").write_text("published", encoding="utf-8")
    assert not target.exists()  # the state a killed swap leaves behind

    assert recover_interrupted_swap(target) is True
    assert (target / "data").read_text(encoding="utf-8") == "published"
    assert not retired.exists()


def test_recover_interrupted_swap_leaves_a_healthy_store_alone(tmp_path: Path) -> None:
    """A `.retired` directory alongside a live store is leftover space, not a pending recovery.

    Restoring over a healthy store would roll back a completed swap.
    """
    from open_climate_service.ingestions.services import recover_interrupted_swap

    target = tmp_path / "ds.icechunk"
    target.mkdir()
    (target / "data").write_text("current", encoding="utf-8")
    retired = tmp_path / "ds.icechunk.retired"
    retired.mkdir()
    (retired / "data").write_text("stale", encoding="utf-8")

    assert recover_interrupted_swap(target) is False
    assert (target / "data").read_text(encoding="utf-8") == "current"


def test_recover_interrupted_swap_is_a_no_op_for_a_brand_new_dataset(tmp_path: Path) -> None:
    from open_climate_service.ingestions.services import recover_interrupted_swap

    assert recover_interrupted_swap(tmp_path / "never-existed.icechunk") is False


def test_recover_interrupted_swap_removes_stale_ingest_rollback_branches(tmp_path: Path) -> None:
    from open_climate_service.ingestions.services import recover_interrupted_swap
    from open_climate_service.streaming.store import open_or_create_repo

    target = tmp_path / "ds.icechunk"
    repo = open_or_create_repo(target)
    snapshot = repo.lookup_branch("main")
    repo.create_branch("ocs-ingest-rollback-first", snapshot)
    repo.create_branch("ocs-ingest-rollback-second", snapshot)

    assert recover_interrupted_swap(target) is True
    assert repo.list_branches() == {"main"}


def test_an_interrupted_swap_is_healed_before_ingest_reads_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery has to happen before anything inspects the store, not after ingest.

    A swap killed between its two renames leaves the published path missing and the whole
    history at `.retired`. If recovery runs after the ingest — as it did when it lived only in
    `_maybe_build_pyramid` — the sequence is:

    1. ingest sees no store, treats it as new, and writes *only* the requested delta;
    2. recovery then finds the target present and does nothing;
    3. the entire history stays stranded in `.retired`, unreferenced.

    Nothing errors, and the dataset silently loses everything before the interruption. So this
    asserts on what the ingest *sees*: the store must already be there, carrying its history.
    """
    from types import SimpleNamespace

    from open_climate_service.data_manager.services import downloader
    from open_climate_service.ingestions import services as ingestion_services
    from open_climate_service.ingestions.schemas import ArtifactRequestScope

    target = tmp_path / "ds.icechunk"
    retired = tmp_path / "ds.icechunk.retired"
    # The state a killed swap leaves: no published store, the history under `.retired`.
    retired.mkdir()
    (retired / "history").write_text("2026-01-01,2026-01-02", encoding="utf-8")

    seen: dict[str, object] = {}

    class FakePlugin:
        async def periods(self, start: str, end: str) -> list[str]:
            assert (start, end) == ("2026-01-01", "2026-01-03")
            return ["2026-01-01", "2026-01-02", "2026-01-03"]

    def fake_ingest(**kwargs: object) -> object:
        store_path = kwargs["store_path"]
        assert isinstance(store_path, Path)
        seen["store_existed"] = store_path.exists()
        history = store_path / "history"
        seen["history"] = history.read_text(encoding="utf-8") if history.exists() else None
        # Append the new period the way a real sync would, onto whatever is there.
        if history.exists():
            history.write_text(history.read_text(encoding="utf-8") + ",2026-01-03", encoding="utf-8")
        else:
            store_path.mkdir(parents=True, exist_ok=True)
            history.write_text("2026-01-03", encoding="utf-8")
        return SimpleNamespace(periods_written=1)

    monkeypatch.setattr(downloader, "get_icechunk_path", lambda _dataset: target)
    monkeypatch.setattr(ingestion_services, "run_streaming_ingest_sync", fake_ingest)
    monkeypatch.setattr(ingestion_services, "_load_streaming_plugin", lambda *a, **k: FakePlugin())
    monkeypatch.setattr(
        ingestion_services,
        "_maybe_build_pyramid",
        lambda *a, **k: ingestion_services._StoreNormalizationResult(completed=True),
    )
    monkeypatch.setattr(
        ingestion_services,
        "get_data_coverage_for_paths",
        lambda *a, **k: {
            "has_data": True,
            "forecast_reference": None,
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-03"},
                "spatial": {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                "spatial_wgs84": None,
            },
        },
    )
    monkeypatch.setattr(ingestion_services, "_upsert_artifact_record", lambda record, **k: record)

    ingestion_services._create_streaming_artifact(
        dataset={
            "id": "ds1",
            "name": "DS1",
            "variable": "precip",
            "period_type": "daily",
            "ingestion": {"plugin": "example.Plugin"},
        },
        plugin_path="example.Plugin",
        start="2026-01-01",
        end="2026-01-03",
        bbox=[0.0, 0.0, 1.0, 1.0],
        country_code=None,
        overwrite=False,
        publish=False,
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-03", bbox=(0.0, 0.0, 1.0, 1.0)),
    )

    # The point of the test: the ingest ran against the recovered store, not a bare path.
    assert seen["store_existed"] is True
    assert seen["history"] == "2026-01-01,2026-01-02"
    # And the delta landed on top of the history rather than replacing it.
    assert (target / "history").read_text(encoding="utf-8") == "2026-01-01,2026-01-02,2026-01-03"
    assert not retired.exists()


@pytest.mark.parametrize("restored", [False, True])
def test_recover_interrupted_rollback_removes_rejected_store(tmp_path: Path, restored: bool) -> None:
    from open_climate_service.ingestions.services import recover_interrupted_swap

    target = tmp_path / "ds.icechunk"
    retired = tmp_path / "ds.icechunk.retired"
    failed = tmp_path / "ds.icechunk.failed"
    original = target if restored else retired
    original.mkdir()
    (original / "data").write_text("original", encoding="utf-8")
    failed.mkdir()
    (failed / "data").write_text("rejected", encoding="utf-8")

    assert recover_interrupted_swap(target) is True
    assert (target / "data").read_text(encoding="utf-8") == "original"
    assert not retired.exists()
    assert not failed.exists()
    assert recover_interrupted_swap(target) is False


def test_interrupted_rollback_preserves_copies_if_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.ingestions.services import recover_interrupted_swap

    target = tmp_path / "ds.icechunk"
    retired = tmp_path / "ds.icechunk.retired"
    failed = tmp_path / "ds.icechunk.failed"
    retired.mkdir()
    failed.mkdir()

    def fail_rename(self: Path, destination: Path) -> Path:
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="rename failed"):
        recover_interrupted_swap(target)
    assert retired.exists()
    assert failed.exists()
    assert not target.exists()


def test_rollback_refuses_to_claim_success_without_retained_store(tmp_path: Path) -> None:
    from open_climate_service.ingestions.services import _rollback_store_swap

    target = tmp_path / "ds.icechunk"
    target.mkdir()
    with pytest.raises(FileNotFoundError, match="retained store .* is missing"):
        _rollback_store_swap(target)
    assert target.exists()


def test_recovery_does_not_publish_a_rejected_store_without_original(tmp_path: Path) -> None:
    from open_climate_service.ingestions.services import recover_interrupted_swap

    target = tmp_path / "ds.icechunk"
    failed = tmp_path / "ds.icechunk.failed"
    failed.mkdir()
    with pytest.raises(RuntimeError, match="only the rejected .failed store remains"):
        recover_interrupted_swap(target)
    assert failed.exists()
    assert not target.exists()
