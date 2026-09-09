from datetime import UTC, datetime
from pathlib import Path

import icechunk
import pytest
import xarray as xr

from open_climate_service.ingestions import services
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
)
from open_climate_service.plugins.datasets.chirps3 import CHIRPS3DailyPlugin
from open_climate_service.publications.services import managed_dataset_id_for


@pytest.fixture(autouse=True)
def materialized_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "_artifact_storage_exists", lambda _: True)


def _artifact(
    *,
    artifact_id: str,
    source_dataset_id: str = "chirps3_precipitation_daily",
    managed_dataset_id: str = "chirps3_precipitation_daily",
    created_at: str = "2026-01-10T00:00:00+00:00",
    end: str = "2026-01-10",
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=source_dataset_id,
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        format=ArtifactFormat.ICECHUNK,
        path="/tmp/chirps3_precipitation_daily.icechunk",
        asset_paths=["/tmp/chirps3_precipitation_daily.icechunk"],
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


def test_list_datasets_groups_artifacts_by_managed_dataset_id(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _artifact(artifact_id="a1", created_at="2026-01-10T00:00:00+00:00", end="2026-01-10"),
        _artifact(artifact_id="a2", created_at="2026-01-11T00:00:00+00:00", end="2026-01-11"),
    ]
    monkeypatch.setattr(services, "_load_records", lambda: records)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "short_name": "CHIRPS3 precip",
            "period_type": "daily",
            "units": "mm",
            "resolution": "5 km x 5 km",
            "source": "CHIRPS v3",
            "source_url": "https://example.com/chirps",
        },
    )

    result = services.list_datasets()

    assert len(result.items) == 1
    dataset = result.items[0]
    assert dataset.dataset_id == "chirps3_precipitation_daily"
    assert dataset.source_dataset_id == "chirps3_precipitation_daily"
    assert dataset.period_type == "daily"
    assert dataset.units == "mm"
    assert dataset.extent.temporal.end == "2026-01-11"
    assert dataset.publication.status == PublicationStatus.PUBLISHED
    assert any(link.href == f"/zarr/{dataset.dataset_id}" for link in dataset.links)
    assert any(link.href == f"/stac/collections/{dataset.dataset_id}" for link in dataset.links)


def test_dataset_links_include_stac_for_published_icechunk() -> None:
    links = services._dataset_links("chirps3_precipitation_daily", _artifact(artifact_id="a1"))

    assert any(link.rel == "stac" and link.href == "/stac/collections/chirps3_precipitation_daily" for link in links)


def test_dataset_links_omit_stac_for_unpublished_or_netcdf() -> None:
    unpublished = _artifact(artifact_id="a1")
    unpublished.publication.status = PublicationStatus.UNPUBLISHED
    netcdf = _artifact(artifact_id="a2")
    netcdf.format = ArtifactFormat.NETCDF

    unpublished_links = services._dataset_links("chirps3_precipitation_daily", unpublished)
    netcdf_links = services._dataset_links("chirps3_precipitation_daily", netcdf)

    assert all(link.rel != "stac" for link in unpublished_links)
    assert all(link.rel != "stac" for link in netcdf_links)


def test_dataset_links_include_zarr_and_stac_for_icechunk() -> None:
    artifact = _artifact(artifact_id="a3")

    links = services._dataset_links("chirps3_precipitation_daily", artifact)

    assert any(link.rel == "zarr" for link in links)
    assert any(link.rel == "stac" for link in links)
    assert all(link.rel != "ogc-collection" for link in links)


def test_get_dataset_zarr_store_file_reads_icechunk_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_path = tmp_path / "chirps3.icechunk"
    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), [[[1.0]]])},
        coords={"t": ["2026-01-01"], "x": [1.0], "y": [2.0]},
        attrs={"proj:code": "EPSG:4326"},
    )
    ds.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("seed metadata")
    ds.close()

    artifact = _artifact(artifact_id="a5")
    artifact.format = ArtifactFormat.ICECHUNK
    artifact.path = str(store_path)
    artifact.asset_paths = [str(store_path)]
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: artifact)

    response = services.get_dataset_zarr_store_file_or_404("chirps3_precipitation_daily", "zarr.json")

    assert isinstance(response, services.JSONResponse)
    assert b'"node_type":"group"' in response.body


def test_zarr_store_root_serves_group_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing slash addresses the store root, and must not look like a missing store.

    No Zarr library requests the root — they treat the URL as a base and append keys — but a
    person pasting the URL does, and FastAPI redirects the slashless form here too, so a 404
    made the root of every store a dead end reachable by the most obvious route.
    """
    store_path = tmp_path / "root.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(store_path)))
    session = repo.writable_session("main")
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), [[[1.0]]])},
        coords={"t": ["2026-01-01"], "x": [1.0], "y": [2.0]},
        attrs={"proj:code": "EPSG:4326"},
    )
    ds.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("seed metadata")
    ds.close()

    artifact = _artifact(artifact_id="a-root")
    artifact.format = ArtifactFormat.ICECHUNK
    artifact.path = str(store_path)
    artifact.asset_paths = [str(store_path)]
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: artifact)

    # "" is what the route passes for both /zarr/{id} (after its redirect) and /zarr/{id}/.
    response = services.get_dataset_zarr_store_file_or_404("chirps3_precipitation_daily", "")

    assert isinstance(response, services.JSONResponse)
    assert b'"node_type":"group"' in response.body


def test_get_dataset_zarr_store_file_rejects_invalid_icechunk_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "chirps3.icechunk"
    storage = icechunk.local_filesystem_storage(str(store_path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), [[[1.0]]])},
        coords={"t": ["2026-01-01"], "x": [1.0], "y": [2.0]},
    )
    ds.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("seed metadata")
    ds.close()

    artifact = _artifact(artifact_id="a5-invalid")
    artifact.format = ArtifactFormat.ICECHUNK
    artifact.path = str(store_path)
    artifact.asset_paths = [str(store_path)]
    monkeypatch.setattr(services, "get_latest_artifact_for_dataset_or_404", lambda _: artifact)

    with pytest.raises(services.HTTPException, match="invalid segments"):
        services.get_dataset_zarr_store_file_or_404("chirps3_precipitation_daily", "../zarr.json")


def test_list_ingestions_returns_most_recent_first(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _artifact(artifact_id="a1", created_at="2026-01-10T00:00:00+00:00", end="2026-01-10"),
        _artifact(artifact_id="a2", created_at="2026-01-11T00:00:00+00:00", end="2026-01-11"),
    ]
    monkeypatch.setattr(services, "_load_records", lambda: records)
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "short_name": "CHIRPS3 precip",
            "period_type": "daily",
            "units": "mm",
            "resolution": "5 km x 5 km",
            "source": "CHIRPS v3",
            "source_url": "https://example.com/chirps",
        },
    )

    result = services.list_ingestions()

    assert result.kind == "IngestionList"
    assert [item.ingestion_id for item in result.items] == ["a2", "a1"]
    assert result.items[0].dataset is not None
    assert result.items[0].dataset.dataset_id == "chirps3_precipitation_daily"


def test_managed_dataset_id_is_derived_from_dataset_id(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _artifact(artifact_id="a1")

    assert managed_dataset_id_for(artifact) == "chirps3_precipitation_daily"


def test_mutate_records_persists_mutation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")

    created = _artifact(artifact_id="mutated")

    def mutate(records: list[ArtifactRecord]) -> ArtifactRecord:
        records.append(created)
        return created

    result = services._mutate_records(mutate)

    assert result.artifact_id == "mutated"
    records = services._load_records()
    assert [record.artifact_id for record in records] == ["mutated"]


def test_find_existing_artifact_ignores_record_with_overwide_coverage() -> None:
    request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end="2026-02-10",
        bbox=(1.0, 2.0, 3.0, 4.0),
    )
    stale_artifact = _artifact(artifact_id="stale", end="2026-02-29")
    stale_artifact.request_scope = request_scope
    valid_artifact = _artifact(artifact_id="valid", end="2026-02-10")
    valid_artifact.request_scope = request_scope

    result = services._find_existing_artifact_in_records(
        records=[stale_artifact, valid_artifact],
        dataset_id="chirps3_precipitation_daily",
        request_scope=request_scope,
    )

    assert result == valid_artifact


def test_find_existing_artifact_does_not_reuse_clamped_icechunk_artifact_for_later_requested_end() -> None:
    request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end="2026-02-10",
        bbox=(1.0, 2.0, 3.0, 4.0),
    )
    clamped = _artifact(artifact_id="icechunk", end="2026-01-31")
    clamped.format = ArtifactFormat.ICECHUNK
    clamped.path = "/tmp/chirps3_precipitation_daily.icechunk"
    clamped.asset_paths = [clamped.path]
    clamped.request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end="2026-01-31",
        bbox=(1.0, 2.0, 3.0, 4.0),
    )

    result = services._find_existing_artifact_in_records(
        records=[clamped],
        dataset_id="chirps3_precipitation_daily",
        request_scope=request_scope,
    )

    assert result is None


def test_find_existing_artifact_ignores_stale_record(monkeypatch: pytest.MonkeyPatch) -> None:
    request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end="2026-02-10",
        bbox=(1.0, 2.0, 3.0, 4.0),
    )
    stale_artifact = _artifact(artifact_id="stale", end="2026-02-10")
    valid_artifact = _artifact(artifact_id="valid", end="2026-02-10")
    monkeypatch.setattr(
        services,
        "_artifact_storage_exists",
        lambda record: record.artifact_id != "stale",
    )

    result = services._find_existing_artifact_in_records(
        records=[stale_artifact, valid_artifact],
        dataset_id="chirps3_precipitation_daily",
        request_scope=request_scope,
    )

    assert result == valid_artifact


def test_list_datasets_ignores_stale_records(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _artifact(artifact_id="stale", created_at="2026-01-10T00:00:00+00:00", end="2026-01-10"),
        _artifact(artifact_id="valid", created_at="2026-01-11T00:00:00+00:00", end="2026-01-11"),
    ]
    monkeypatch.setattr(services, "_load_records", lambda: records)
    monkeypatch.setattr(
        services,
        "_artifact_storage_exists",
        lambda record: record.artifact_id != "stale",
    )
    monkeypatch.setattr(
        services.registry_datasets,
        "get_dataset",
        lambda _: {
            "id": "chirps3_precipitation_daily",
            "short_name": "CHIRPS3 precip",
            "period_type": "daily",
            "units": "mm",
            "resolution": "5 km x 5 km",
            "source": "CHIRPS v3",
            "source_url": "https://example.com/chirps",
        },
    )

    result = services.list_datasets()

    assert len(result.items) == 1
    assert result.items[0].extent.temporal.end == "2026-01-11"


def test_temporal_coverage_matches_request_scope_allows_open_ended_reuse() -> None:
    request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end=None,
        bbox=(1.0, 2.0, 3.0, 4.0),
    )

    assert services._temporal_coverage_matches_request_scope(
        CoverageTemporal(start="2026-01-01", end="2026-02-10"),
        request_scope,
    )


def test_temporal_coverage_matches_streaming_request_scope_requires_exact_start() -> None:
    request_scope = ArtifactRequestScope(
        start="2026-01-01",
        end="2026-02-10",
        bbox=(1.0, 2.0, 3.0, 4.0),
    )

    assert not services._temporal_coverage_matches_streaming_request_scope(
        CoverageTemporal(start="2026-01-03", end="2026-01-31"),
        request_scope,
    )


def test_create_artifact_uses_streaming_plugin_for_direct_ingest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
            "params": {"stage": "final"},
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()

    plugin = object()
    captured: dict[str, object] = {}

    def fake_load_streaming_plugin(plugin_path: str, *, params: dict[str, object]) -> object:
        captured["plugin_path"] = plugin_path
        captured["params"] = params
        return plugin

    monkeypatch.setattr(services, "_load_streaming_plugin", fake_load_streaming_plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)

    def fake_run_streaming_ingest_sync(**kwargs: object) -> object:
        captured["run"] = kwargs
        return type("Result", (), {"periods_written": 1})()

    monkeypatch.setattr(services, "run_streaming_ingest_sync", fake_run_streaming_ingest_sync)

    def fake_get_data_coverage_for_paths(
        dataset_arg: dict[str, object],
        *,
        zarr_path: str | None = None,
        icechunk_path: str | None = None,
        netcdf_paths: list[str] | None = None,
    ) -> dict[str, object]:
        captured["dataset_id"] = dataset_arg["id"]
        captured["zarr_path"] = zarr_path
        captured["icechunk_path"] = icechunk_path
        captured["netcdf_paths"] = netcdf_paths
        return {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-03"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        }

    monkeypatch.setattr(services, "get_data_coverage_for_paths", fake_get_data_coverage_for_paths)
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )

    assert captured["plugin_path"] == "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin"
    assert captured["params"] == {"stage": "final"}
    assert captured["dataset_id"] == "chirps3_precipitation_daily"
    assert captured["zarr_path"] is None
    assert captured["icechunk_path"] == str(store_path.resolve())
    assert captured["netcdf_paths"] is None
    assert captured["run"] == {
        "plugin": plugin,
        "params": {"stage": "final"},
        "dataset": dataset,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "start": "2026-01-01",
        "end": "2026-01-03",
        "store_path": store_path,
        "period_type": "daily",
        "on_progress": None,
        "is_cancel_requested": None,
        "save_cursor": None,
        "periods": None,
    }
    assert artifact.format == ArtifactFormat.ICECHUNK
    assert artifact.path == str(store_path.resolve())
    assert artifact.asset_paths == [str(store_path.resolve())]


def test_create_artifact_uses_streaming_plugin_for_store_based_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
            "params": {"stage": "final"},
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()

    plugin = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)

    def fake_run_streaming_ingest_sync(**kwargs: object) -> object:
        captured["run"] = kwargs
        return type("Result", (), {"periods_written": 2})()

    monkeypatch.setattr(services, "run_streaming_ingest_sync", fake_run_streaming_ingest_sync)
    monkeypatch.setattr(
        services,
        "get_data_coverage_for_paths",
        lambda *args, **kwargs: {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-02-10"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        },
    )
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-02-10",
        download_start="2026-02-01",
        download_end="2026-02-10",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )

    run_kwargs = captured["run"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["plugin"] is plugin
    assert run_kwargs["params"] == {"stage": "final"}
    assert run_kwargs["dataset"] == dataset
    assert run_kwargs["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert run_kwargs["start"] == "2026-01-01"
    assert run_kwargs["end"] == "2026-02-10"
    assert run_kwargs["store_path"] == store_path
    assert run_kwargs["period_type"] == "daily"
    assert artifact.format == ArtifactFormat.ICECHUNK
    assert artifact.coverage.temporal.end == "2026-02-10"


def test_create_artifact_forwards_country_code_to_streaming_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "worldpop_population_yearly",
        "name": "Total population (WorldPop Global2)",
        "variable": "pop_total",
        "period_type": "yearly",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.worldpop.WorldPopYearlyPlugin",
            "params": {"version": "global2"},
        },
    }
    plugin = object()
    store_path = tmp_path / "worldpop_population_yearly.icechunk"
    captured: dict[str, object] = {}

    def fake_load_streaming_plugin(plugin_path: str, *, params: dict[str, object]) -> object:
        captured["plugin_path"] = plugin_path
        captured["params"] = dict(params)
        return plugin

    monkeypatch.setattr(services, "_load_streaming_plugin", fake_load_streaming_plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(
        services,
        "run_streaming_ingest_sync",
        lambda **kwargs: captured.update({"run": kwargs}) or type("Result", (), {"periods_written": 1})(),
    )
    monkeypatch.setattr(
        services,
        "get_data_coverage_for_paths",
        lambda *args, **kwargs: {
            "coverage": {
                "temporal": {"start": "2020", "end": "2022"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        },
    )
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)

    services.create_artifact(
        dataset=dataset,
        start="2020",
        end="2022",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code="SLE",
        overwrite=False,
        publish=False,
    )

    assert captured["plugin_path"] == "open_climate_service.plugins.datasets.worldpop.WorldPopYearlyPlugin"
    assert captured["params"] == {"version": "global2", "country_code": "SLE"}
    run_kwargs = captured["run"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["params"] == {"version": "global2", "country_code": "SLE"}
    assert run_kwargs["dataset"] == dataset


def test_load_streaming_plugin_rejects_non_callable_symbol() -> None:
    with pytest.raises(
        services.HTTPException,
        match="Failed to load ingestion.plugin 'open_climate_service.ingestions.services.logger'",
    ):
        services._load_streaming_plugin(
            "open_climate_service.ingestions.services.logger",
            params={},
        )


def test_load_streaming_plugin_rejects_symbol_outside_plugin_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NotAPlugin:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(services, "_NotAPlugin", _NotAPlugin, raising=False)

    with pytest.raises(
        services.HTTPException,
        match="Failed to load ingestion.plugin 'open_climate_service.ingestions.services._NotAPlugin'",
    ):
        services._load_streaming_plugin(
            "open_climate_service.ingestions.services._NotAPlugin",
            params={"stage": "final"},
        )


def test_load_streaming_plugin_filters_runtime_only_params_for_constructor() -> None:
    plugin = services._load_streaming_plugin(
        "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        params={"stage": "final", "country_code": "SLE"},
    )

    assert isinstance(plugin, CHIRPS3DailyPlugin)
    assert plugin.stage == "final"
    assert plugin.flavor == "rnl"


def test_create_artifact_allows_streaming_coverage_clamped_to_source_availability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(
        services,
        "run_streaming_ingest_sync",
        lambda **kwargs: type("Result", (), {"periods_written": 1})(),
    )
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)

    def fake_get_data_coverage_for_paths(
        dataset_arg: dict[str, object],
        *,
        zarr_path: str | None = None,
        icechunk_path: str | None = None,
        netcdf_paths: list[str] | None = None,
    ) -> dict[str, object]:
        assert dataset_arg["id"] == "chirps3_precipitation_daily"
        assert zarr_path is None
        assert icechunk_path == str(store_path.resolve())
        assert netcdf_paths is None
        return {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-31"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        }

    monkeypatch.setattr(services, "get_data_coverage_for_paths", fake_get_data_coverage_for_paths)

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-02-03",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )

    assert artifact.coverage.temporal.start == "2026-01-01"
    assert artifact.coverage.temporal.end == "2026-01-31"
    assert artifact.request_scope.start == "2026-01-01"
    assert artifact.request_scope.end == "2026-01-31"


def test_create_artifact_rejects_streaming_coverage_with_late_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(
        services,
        "run_streaming_ingest_sync",
        lambda **kwargs: type("Result", (), {"periods_written": 1})(),
    )
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)
    monkeypatch.setattr(
        services,
        "get_data_coverage_for_paths",
        lambda *args, **kwargs: {
            "coverage": {
                "temporal": {"start": "2026-01-03", "end": "2026-01-31"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        },
    )

    with pytest.raises(services.HTTPException, match="coverage does not match the requested scope"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-02-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )


def test_create_artifact_returns_409_when_streaming_plugin_has_no_periods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(
        services,
        "run_streaming_ingest_sync",
        lambda **kwargs: type("Result", (), {"periods_written": 0})(),
    )
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)

    with pytest.raises(services.HTTPException, match="Source has no data for the requested temporal scope"):
        services.create_artifact(
            dataset=dataset,
            start="2026-02-01",
            end="2026-02-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )


def test_create_artifact_overwrite_replaces_existing_icechunk_store_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    replacement_path = store_path.with_name(f"{store_path.name}.replacement")
    store_path.mkdir()
    (store_path / "stale").write_text("old", encoding="utf-8")

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda record, **_: record)

    def fake_run_streaming_ingest_sync(**kwargs: object) -> object:
        replacement = kwargs["store_path"]
        assert isinstance(replacement, Path)
        assert replacement == replacement_path
        assert (store_path / "stale").read_text(encoding="utf-8") == "old"
        replacement.mkdir()
        (replacement / "fresh").write_text("new", encoding="utf-8")
        return type("Result", (), {"periods_written": 1})()

    def fake_coverage(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["icechunk_path"] == str(replacement_path.resolve())
        assert (store_path / "stale").read_text(encoding="utf-8") == "old"
        return {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-03"},
                "spatial": {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0},
            }
        }

    def fake_build_pyramid(path: Path, dataset: dict[str, object]) -> None:
        assert path == replacement_path
        assert (store_path / "stale").read_text(encoding="utf-8") == "old"
        (path / "normalized").write_text("complete", encoding="utf-8")

    monkeypatch.setattr(services, "run_streaming_ingest_sync", fake_run_streaming_ingest_sync)
    monkeypatch.setattr(services, "get_data_coverage_for_paths", fake_coverage)
    monkeypatch.setattr(services, "_maybe_build_pyramid", fake_build_pyramid)

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=True,
        publish=False,
    )

    assert artifact.path == str(store_path.resolve())
    assert not (store_path / "stale").exists()
    assert (store_path / "fresh").read_text(encoding="utf-8") == "new"
    assert (store_path / "normalized").read_text(encoding="utf-8") == "complete"
    assert not replacement_path.exists()
    assert not store_path.with_name(f"{store_path.name}.retired").exists()


def test_create_artifact_overwrite_keeps_existing_store_when_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()
    (store_path / "committed").write_text("old", encoding="utf-8")

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)

    def failing_ingest(**kwargs: object) -> object:
        replacement = kwargs["store_path"]
        assert isinstance(replacement, Path)
        assert (store_path / "committed").read_text(encoding="utf-8") == "old"
        replacement.mkdir()
        (replacement / "partial").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("upstream fetch failed")

    monkeypatch.setattr(services, "run_streaming_ingest_sync", failing_ingest)

    with pytest.raises(RuntimeError, match="upstream fetch failed"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=True,
            publish=False,
        )

    assert (store_path / "committed").read_text(encoding="utf-8") == "old"
    assert not store_path.with_name(f"{store_path.name}.replacement").exists()
    assert not store_path.with_name(f"{store_path.name}.retired").exists()


def test_create_artifact_overwrite_releases_lock_when_replacement_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    store_path.mkdir()

    class TrackingLock:
        released = False

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return True

        def release(self) -> None:
            self.released = True

    lock = TrackingLock()
    cleanup_calls = 0

    def fail_final_cleanup(path: Path) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 2:
            raise PermissionError(f"cannot remove {path}")

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(services, "_acquire_store_lock", lambda _: lock)
    monkeypatch.setattr(services, "_remove_store_path", fail_final_cleanup)
    monkeypatch.setattr(
        services,
        "run_streaming_ingest_sync",
        lambda **_: (_ for _ in ()).throw(RuntimeError("upstream fetch failed")),
    )

    with pytest.raises(RuntimeError, match="upstream fetch failed"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=True,
            publish=False,
        )

    assert cleanup_calls == 2
    assert lock.released is True


def test_create_artifact_overwrite_keeps_existing_store_when_replacement_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {
            "plugin": "open_climate_service.plugins.datasets.chirps3.CHIRPS3DailyPlugin",
        },
    }
    store_path = tmp_path / "chirps3_precipitation_daily.icechunk"
    replacement_path = store_path.with_name(f"{store_path.name}.replacement")
    store_path.mkdir()
    (store_path / "committed").write_text("old", encoding="utf-8")

    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: object())
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "_find_existing_artifact", lambda **_: None)

    def incomplete_ingest(**kwargs: object) -> object:
        replacement = kwargs["store_path"]
        assert replacement == replacement_path
        replacement_path.mkdir()
        return type("Result", (), {"periods_written": 1})()

    monkeypatch.setattr(services, "run_streaming_ingest_sync", incomplete_ingest)
    monkeypatch.setattr(
        services,
        "get_data_coverage_for_paths",
        lambda *args, **kwargs: {"has_data": False},
    )

    with pytest.raises(services.HTTPException, match="Materialized artifact contains no data"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=True,
            publish=False,
        )

    assert (store_path / "committed").read_text(encoding="utf-8") == "old"
    assert not replacement_path.exists()
    assert not store_path.with_name(f"{store_path.name}.retired").exists()


def test_create_artifact_rejects_missing_plugin_definition() -> None:
    dataset: dict[str, object] = {
        "id": "broken_dataset",
        "name": "Broken dataset",
        "variable": "value",
        "period_type": "daily",
        "ingestion": {},
    }

    with pytest.raises(services.HTTPException) as exc_info:
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-10",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Dataset 'broken_dataset' does not define ingestion.plugin"


def test_create_artifact_rejects_partial_download_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
    }

    with pytest.raises(services.HTTPException) as exc_info:
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-02-10",
            download_start=None,
            download_end="2026-02-10",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert exc_info.value.status_code == 400
    assert "download_start and download_end must either both be provided or both be omitted" in str(
        exc_info.value.detail
    )


def test_create_artifact_rejects_download_scope_outside_request_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset: dict[str, object] = {
        "id": "chirps3_precipitation_daily",
        "name": "Total precipitation (CHIRPS3)",
        "variable": "precip",
        "period_type": "daily",
    }

    with pytest.raises(services.HTTPException) as exc_info:
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-02-10",
            download_start="2026-02-01",
            download_end="2026-02-11",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert exc_info.value.status_code == 400
    assert "download_end must be less than or equal to end" in str(exc_info.value.detail)
