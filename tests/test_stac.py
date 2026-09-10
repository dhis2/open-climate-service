import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.openeo import collections as openeo_collections
from open_climate_service.stac import services as stac_services


@pytest.fixture(autouse=True)
def _clear_xstac_collection_cache() -> None:
    stac_services._clear_xstac_collection_cache()


def _minimal_xstac_payload(dataset_id: str = "chirps3_precipitation_daily") -> dict[str, object]:
    """Return a minimal xstac-style collection payload for tests that don't need full metadata."""
    return {
        "type": "Collection",
        "id": dataset_id,
        "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
        "cube:dimensions": {"time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}},
        "cube:variables": {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}},
        "assets": {"zarr": {}},
    }


def _artifact(
    *,
    artifact_id: str,
    dataset_id: str = "chirps3_precipitation_daily",
    dataset_name: str = "CHIRPS3 precipitation",
    variable: str = "precip",
    managed_dataset_id: str = "chirps3_precipitation_daily",
    status: PublicationStatus = PublicationStatus.PUBLISHED,
    format: ArtifactFormat = ArtifactFormat.ICECHUNK,
    path: str | None = "/tmp/chirps3_precipitation_daily.icechunk",
    asset_paths: list[str] | None = None,
    temporal_start: str = "2026-01-01",
    temporal_end: str = "2026-01-10",
    created_at: datetime | None = None,
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        variable=variable,
        format=format,
        path=path,
        asset_paths=[path] if asset_paths is None and path is not None else (asset_paths or []),
        variables=[variable],
        request_scope=ArtifactRequestScope(
            start=temporal_start,
            end=temporal_end,
            bbox=(1.0, 2.0, 3.0, 4.0),
        ),
        coverage=ArtifactCoverage(
            temporal=CoverageTemporal(start=temporal_start, end=temporal_end),
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
        ),
        created_at=created_at or datetime(2026, 1, 10, tzinfo=UTC),
        publication=ArtifactPublication(
            status=status,
            collection_id=managed_dataset_id,
        ),
    )


def test_stac_landing_returns_catalog(client: TestClient) -> None:
    response = client.get("/stac/catalog.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "Catalog"
    assert any(link["rel"] == "self" for link in payload["links"])
    assert any(link["rel"] == "root" for link in payload["links"])


def test_stac_catalog_lists_child_collections(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stac_services,
        "_eligible_artifacts_by_dataset",
        lambda: {"chirps3_precipitation_daily": _artifact(artifact_id="a1")},
    )

    response = client.get("/stac/catalog.json")

    assert response.status_code == 200
    payload = response.json()
    child_links = [link for link in payload["links"] if link["rel"] == "child"]
    assert len(child_links) == 1
    assert child_links[0]["href"].endswith("/stac/collections/chirps3_precipitation_daily")


def test_stac_landing_links_to_collections(client: TestClient) -> None:
    response = client.get("/stac")

    assert response.status_code == 200
    payload = response.json()
    assert any(link["href"].endswith("/stac/catalog.json") for link in payload["links"] if link["rel"] == "root")


def test_catalog_self_link_reflects_request_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )

    response = client.get("/stac")

    assert response.status_code == 200
    payload = response.json()
    assert payload["links"][0]["href"].endswith("/stac")


def test_catalog_excludes_unpublished_and_netcdf(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(
            items=[
                _artifact(artifact_id="a1", status=PublicationStatus.UNPUBLISHED),
                _artifact(artifact_id="a2", format=ArtifactFormat.NETCDF),
            ]
        ),
    )

    response = client.get("/stac/catalog.json")

    assert response.status_code == 200
    payload = response.json()
    assert [link for link in payload["links"] if link["rel"] == "child"] == []


def test_collection_uses_xstac_and_adds_expected_fields(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )
    monkeypatch.setattr(
        stac_services.registry_datasets,
        "get_dataset",
        lambda _: {
            "period_type": "daily",
            "units": "mm",
            "source": "CHIRPS v3",
            "extents": {"temporal": {"resolution": "P1D"}},
        },
    )
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "chirps3_precipitation_daily",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {
                "x": {
                    "type": "spatial",
                    "axis": "x",
                    "extent": [1.0, 3.0],
                    "step": 0.05000000074505806,
                    "reference_system": 4326,
                },
                "y": {
                    "type": "spatial",
                    "axis": "y",
                    "extent": [2.0, 4.0],
                    "step": -0.05000000074505806,
                    "reference_system": 4326,
                },
                "time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]},
            },
            "stac_extensions": ["https://stac-extensions.github.io/projection/v2.0.0/schema.json"],
            "cube:variables": {
                "precip": {
                    "type": "data",
                    "dimensions": ["time", "y", "x"],
                    "attrs": {
                        "long_name": "Precipitation",
                        "units": "mm/day",
                        "standard_name": "lwe_precipitation_rate",
                        "cell_methods": "time: mean",
                        "TIFFTAG_SOFTWARE": "IDL 9.0.0",
                    },
                },
            },
            "assets": {"zarr": {}},
        },
    )
    monkeypatch.setattr(
        stac_services,
        "_zarr_asset_metadata",
        lambda _: {"zarr:consolidated": True, "zarr:zarr_format": 3},
    )
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "Collection"
    assert payload["description"] == "Published GeoZarr dataset for CHIRPS3 precipitation"
    assert payload["assets"]["zarr"]["href"].endswith("/zarr/chirps3_precipitation_daily")
    assert payload["assets"]["zarr"]["xarray:open_kwargs"] == {"consolidated": True}
    assert payload["extent"]["temporal"]["interval"] == [["2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"]]
    assert payload["cube:dimensions"]["x"]["step"] == 0.05
    assert payload["cube:dimensions"]["y"]["step"] == -0.05
    assert payload["cube:dimensions"]["t"]["extent"] == ["2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"]
    assert payload["cube:dimensions"]["t"]["step"] == "P1D"
    assert payload["cube:variables"]["precip"]["unit"] == "mm/day"
    # CF semantics surfaced as cube:variable fields, named per the STAC CF extension
    # (CLIM-828). Prefixed at this level because that is the defined STAC field; the raw CF
    # spellings stay inside `attrs`, which passes the store's own attribute names through.
    assert payload["cube:variables"]["precip"]["cf:standard_name"] == "lwe_precipitation_rate"
    assert payload["cube:variables"]["precip"]["cf:cell_methods"] == "time: mean"
    assert "standard_name" not in payload["cube:variables"]["precip"]
    assert payload["cube:variables"]["precip"]["attrs"] == {
        "long_name": "Precipitation",
        "units": "mm/day",
        "standard_name": "lwe_precipitation_rate",
        "cell_methods": "time: mean",
    }
    assert "https://stac-extensions.github.io/projection/v2.0.0/schema.json" in payload["stac_extensions"]
    # Declared because cf: fields were emitted.
    assert "https://stac-extensions.github.io/cf/v1.0.0/schema.json" in payload["stac_extensions"]


def test_stac_collection_compatibility_route_builds_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stac_services,
        "build_collection",
        lambda dataset_id, request: {"type": "Collection", "id": dataset_id, "links": []},
    )

    response = client.get("/stac/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    assert response.json()["id"] == "chirps3_precipitation_daily"


def test_collections_logs_skipped_dataset_failures(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        stac_services,
        "_eligible_artifacts_by_dataset",
        lambda: {"broken_dataset": _artifact(artifact_id="a1", dataset_id="broken_dataset")},
    )

    def _raise(dataset_id: str, request: object) -> dict[str, object]:
        raise stac_services.HTTPException(status_code=503, detail="store unavailable")

    monkeypatch.setattr(stac_services, "build_collection", _raise)
    monkeypatch.setattr(openeo_collections.logger, "warning", lambda *args: calls.append(args))

    response = client.get("/collections")

    assert response.status_code == 200
    assert response.json()["collections"] == []
    assert calls == [("Skipping collection '%s' from openEO listing: %s", "broken_dataset", "store unavailable")]


def test_collection_uses_configured_base_url(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIMATE_SERVICE_BASE_URL", "https://climate.example.org")
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )
    monkeypatch.setattr(
        stac_services.registry_datasets,
        "get_dataset",
        lambda _: {"period_type": "daily", "units": "mm"},
    )
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "chirps3_precipitation_daily",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {"time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}},
            "cube:variables": {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}},
            "assets": {"zarr": {}},
        },
    )
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {"zarr:consolidated": True})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    payload = response.json()
    assert payload["links"][0]["href"] == "https://climate.example.org/collections/chirps3_precipitation_daily"


def test_collection_sets_hourly_step_to_pt1h(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    hourly_artifact = _artifact(
        artifact_id="a1",
        dataset_id="era5land_temperature_hourly",
        dataset_name="ERA5-Land temperature",
        variable="t2m",
        managed_dataset_id="era5land_temperature_hourly",
        path="/tmp/era5land_temperature_hourly.zarr",
        temporal_start="2026-01-01T00",
        temporal_end="2026-01-01T12",
    )
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[hourly_artifact]),
    )
    monkeypatch.setattr(
        stac_services.registry_datasets,
        "get_dataset",
        lambda _: {
            "period_type": "hourly",
            "source": "ERA5-Land",
            "short_name": "2m temperature",
            "extents": {"temporal": {"resolution": "PT1H"}},
        },
    )
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "era5land_temperature_hourly",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {
                "valid_time": {"type": "temporal", "extent": ["2026-01-01T00", "2026-01-01T12"]},
            },
            "cube:variables": {"t2m": {"type": "data", "dimensions": ["valid_time", "y", "x"]}},
            "assets": {"zarr": {}},
        },
    )
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {"zarr:consolidated": True})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})

    response = client.get("/collections/era5land_temperature_hourly")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cube:dimensions"]["t"]["step"] == "PT1H"


def test_collection_uses_root_href_for_icechunk_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ICECHUNK, path="/tmp/chirps3.icechunk")
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[artifact]),
    )
    monkeypatch.setattr(
        stac_services.registry_datasets,
        "get_dataset",
        lambda _: {"period_type": "daily", "source": "CHIRPS v3", "ingestion": {}},
    )
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "chirps3_precipitation_daily",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {"time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}},
            "cube:variables": {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}},
            "assets": {"zarr": {}},
        },
    )

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"]["zarr"]["href"].endswith("/zarr/chirps3_precipitation_daily")
    assert payload["assets"]["zarr"]["xarray:open_kwargs"] == {"consolidated": True}
    assert payload["assets"]["zarr"]["zarr:zarr_format"] == 3


def test_collection_returns_404_for_unknown_dataset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[]),
    )

    response = client.get("/collections/unknown-dataset")

    assert response.status_code == 404


def test_collections_prefers_latest_artifact_per_managed_dataset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = _artifact(artifact_id="a1", created_at=datetime(2026, 1, 9, tzinfo=UTC))
    newer = _artifact(artifact_id="a2", created_at=datetime(2026, 1, 10, tzinfo=UTC))
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[older, newer]),
    )
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", lambda **_: _minimal_xstac_payload())
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {})

    response = client.get("/collections")

    assert response.status_code == 200
    payload = response.json()
    collections = payload["collections"]
    assert len(collections) == 1
    assert collections[0]["id"] == "chirps3_precipitation_daily"


def test_collections_sorts_by_managed_dataset_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(
            items=[
                _artifact(artifact_id="a1", managed_dataset_id="worldpop_population_yearly"),
                _artifact(
                    artifact_id="a2",
                    dataset_id="aa_dataset",
                    dataset_name="AA Dataset",
                    variable="value",
                    managed_dataset_id="aa_dataset",
                    path="/tmp/aa_dataset.zarr",
                ),
            ]
        ),
    )
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", lambda **_: _minimal_xstac_payload())
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {})

    response = client.get("/collections")

    assert response.status_code == 200
    payload = response.json()
    collection_ids = [col["id"] for col in payload["collections"]]
    assert collection_ids == sorted(collection_ids)


def test_build_collection_with_xstac_normalizes_pystac_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyDataset:
        attrs: dict[str, object] = {}

        def close(self) -> None:
            pass

    artifact = _artifact(artifact_id="a1")
    template = pystac.Collection(
        id="chirps3_precipitation_daily",
        description="template",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[1.0, 2.0, 3.0, 4.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 10, tzinfo=UTC)]]),
        ),
        title="CHIRPS3 precipitation",
        license="proprietary",
    )
    template.add_asset("zarr", pystac.Asset(href="http://example.test/zarr"))
    monkeypatch.setattr(stac_services, "open_icechunk_dataset", lambda _: DummyDataset())
    monkeypatch.setattr(stac_services, "get_x_y_dims", lambda _: ("x", "y"))
    monkeypatch.setattr(stac_services, "get_time_dim", lambda _: "time")
    monkeypatch.setattr(stac_services, "xarray_to_stac", lambda *args, **kwargs: template)

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert isinstance(payload, dict)
    assert payload["type"] == "Collection"
    assert payload["id"] == "chirps3_precipitation_daily"


def test_collection_reuses_cached_xstac_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyDataset:
        attrs: dict[str, object] = {}

        def close(self) -> None:
            pass

    open_count = 0

    def fake_open(_: str) -> DummyDataset:
        nonlocal open_count
        open_count += 1
        return DummyDataset()

    def fake_xarray_to_stac(*args: object, **kwargs: object) -> pystac.Collection:
        template = kwargs["template"] if "template" in kwargs else args[1]
        assert isinstance(template, pystac.Collection)
        template.extra_fields["cube:dimensions"] = {
            "time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}
        }
        template.extra_fields["cube:variables"] = {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}}
        return template

    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "open_icechunk_dataset", fake_open)
    monkeypatch.setattr(stac_services, "get_x_y_dims", lambda _: ("x", "y"))
    monkeypatch.setattr(stac_services, "get_time_dim", lambda _: "time")
    monkeypatch.setattr(stac_services, "xarray_to_stac", fake_xarray_to_stac)
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {"zarr:consolidated": True})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})

    first_response = client.get("/collections/chirps3_precipitation_daily")
    second_response = client.get("/collections/chirps3_precipitation_daily")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()
    assert open_count == 1


def test_zarr_consolidated_flag_detects_v3_and_v2_markers(tmp_path: Path) -> None:
    v3_consolidated = tmp_path / "v3_consolidated.zarr"
    v3_consolidated.mkdir()
    (v3_consolidated / "zarr.json").write_text('{"consolidated_metadata": {}}', encoding="utf-8")

    v3_unconsolidated = tmp_path / "v3_unconsolidated.zarr"
    v3_unconsolidated.mkdir()
    (v3_unconsolidated / "zarr.json").write_text("{}", encoding="utf-8")

    v2_consolidated = tmp_path / "v2_consolidated.zarr"
    v2_consolidated.mkdir()
    (v2_consolidated / ".zmetadata").write_text("{}", encoding="utf-8")

    assert stac_services._zarr_consolidated_flag(str(v3_consolidated)) is True
    assert stac_services._zarr_consolidated_flag(str(v3_unconsolidated)) is False
    assert stac_services._zarr_consolidated_flag(str(v2_consolidated)) is True
    assert stac_services._zarr_consolidated_flag("s3://example-bucket/store.zarr") is None


def test_collection_preserves_template_links_when_xstac_mutates_template(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyDataset:
        attrs: dict[str, object] = {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "open_icechunk_dataset", lambda _: DummyDataset())
    monkeypatch.setattr(stac_services, "get_x_y_dims", lambda _: ("x", "y"))
    monkeypatch.setattr(stac_services, "get_time_dim", lambda _: "time")

    def fake_xarray_to_stac(*args: object, **kwargs: object) -> pystac.Collection:
        template = kwargs["template"] if "template" in kwargs else args[1]
        assert isinstance(template, pystac.Collection)
        template.extra_fields["cube:dimensions"] = {
            "time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-10"]}
        }
        template.extra_fields["cube:variables"] = {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}}
        return template

    monkeypatch.setattr(stac_services, "xarray_to_stac", fake_xarray_to_stac)
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {"zarr:consolidated": True})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 200
    payload = response.json()
    links = {link["rel"]: link["href"] for link in payload["links"]}
    assert links["self"].endswith("/collections/chirps3_precipitation_daily")
    # root/parent links from stac_services point to /stac (rewritten from /stac/catalog.json)
    assert links["root"].endswith("/stac")
    assert links["parent"].endswith("/stac")
    assert links["alternate"].endswith("/datasets/chirps3_precipitation_daily")


def test_collection_returns_500_for_missing_artifact_store_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(artifact_id="a1", path=None, asset_paths=[])
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[artifact]))
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 500
    assert "no readable storage path metadata" in response.json()["detail"]


def test_collection_returns_503_when_zarr_store_cannot_be_opened(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_file_not_found(_: str) -> None:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="a1")]),
    )
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(
        stac_services,
        "open_icechunk_dataset",
        raise_file_not_found,
    )

    response = client.get("/collections/chirps3_precipitation_daily")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_build_collection_with_xstac_reads_normalised_zarr_coordinates(tmp_path: Path) -> None:
    """STAC collection metadata is derived correctly from a normalised longitude/latitude/time store."""
    zarr_path = tmp_path / "chirps3_precipitation_daily.zarr"
    ds = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], np.ones((5, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=5, freq="D"),
            "latitude": [4.0, 3.0, 2.0],
            "longitude": [1.0, 2.0, 3.0],
        },
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))

    template = pystac.Collection(
        id="chirps3_precipitation_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[1.0, 2.0, 3.0, 4.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 5, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["type"] == "Collection"
    cube_dims = payload["cube:dimensions"]
    assert "longitude" in cube_dims, f"expected 'longitude' in cube:dimensions, got {list(cube_dims)}"
    assert "latitude" in cube_dims, f"expected 'latitude' in cube:dimensions, got {list(cube_dims)}"
    assert "time" in cube_dims, f"expected 'time' in cube:dimensions, got {list(cube_dims)}"
    assert cube_dims["longitude"]["axis"] == "x"
    assert cube_dims["latitude"]["axis"] == "y"
    assert cube_dims["time"]["type"] == "temporal"


def test_build_collection_emits_crs_render_hints_for_projected_store(tmp_path: Path) -> None:
    """Projected (non-built-in) stores get proj:wkt2 + proj:projjson so map
    clients can reproject without a runtime epsg.io lookup."""
    zarr_path = tmp_path / "senorge_temperature_daily.zarr"
    ds = xr.Dataset(
        {"tg": (["time", "y", "x"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "y": [7000000.0, 6999000.0, 6998000.0],
            "x": [100000.0, 101000.0, 102000.0],
        },
        attrs={"proj:code": "EPSG:32633"},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="senorge_temperature_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[100000.0, 6998000.0, 102000.0, 7000000.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["proj:code"] == "EPSG:32633"
    # The non-standard proj4 hint is retired (CLIM-833): zarr-layer resolves the code
    # through proj4's own registry, so only standard fields are published now.
    assert "open_climate_service:proj4" not in payload
    assert payload["proj:wkt2"].startswith("PROJCRS")  # WKT2, not WKT1 (PROJCS)
    # proj:projjson — same CRS as a PROJJSON object (EPSG:32633 = UTM zone 33N)
    assert payload["proj:projjson"]["type"] == "ProjectedCRS"
    assert payload["proj:projjson"]["id"] == {"authority": "EPSG", "code": 32633}
    # proj:bbox derived from the x/y coordinate arrays (no spatial:bbox on this store)
    assert payload["proj:bbox"] == [100000.0, 6998000.0, 102000.0, 7000000.0]
    # extent.spatial.bbox is reprojected to WGS84 from the store — not the native metres
    from pyproj import Transformer

    expected_wgs84 = list(
        Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True).transform_bounds(
            100000.0, 6998000.0, 102000.0, 7000000.0
        )
    )
    assert payload["extent"]["spatial"]["bbox"][0] == pytest.approx(expected_wgs84)
    assert payload["extent"]["spatial"]["bbox"][0] != [100000.0, 6998000.0, 102000.0, 7000000.0]


def test_build_collection_omits_crs_render_hints_for_wgs84(tmp_path: Path) -> None:
    """Built-in CRSes (EPSG:4326) resolve on the client from their code, so no hint is emitted."""
    zarr_path = tmp_path / "era5land_temperature_daily.zarr"
    ds = xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "latitude": [4.0, 3.0, 2.0],
            "longitude": [1.0, 2.0, 3.0],
        },
        attrs={"proj:code": "EPSG:4326"},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="era5land_temperature_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[1.0, 2.0, 3.0, 4.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["proj:code"] == "EPSG:4326"
    assert "open_climate_service:proj4" not in payload
    assert "proj:wkt2" not in payload
    assert "proj:bbox" not in payload


def test_build_collection_spatial_extent_derives_from_store_not_coverage(tmp_path: Path) -> None:
    """The collection's spatial extent is read live from the store, so a stale or degenerate
    coverage/template extent (the worldpop/chirps bug) never leaks into STAC."""
    zarr_path = tmp_path / "worldpop_population_yearly.zarr"
    ds = xr.Dataset(
        {"pop": (["time", "latitude", "longitude"], np.ones((1, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2020-01-01", periods=1, freq="YS"),
            "latitude": [60.0, 59.0, 58.0],
            "longitude": [4.0, 5.0, 6.0],
        },
        attrs={"proj:code": "EPSG:4326"},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="worldpop_population_yearly",
        description="test",
        extent=pystac.Extent(
            # A degenerate/stale coverage bbox — must NOT reach the payload.
            spatial=pystac.SpatialExtent([[10.5113, 0.0005, 10.5115, 0.0006]]),
            temporal=pystac.TemporalExtent([[datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["extent"]["spatial"]["bbox"] == [[4.0, 58.0, 6.0, 60.0]]


def test_build_collection_proj_bbox_prefers_store_spatial_bbox_attr(tmp_path: Path) -> None:
    """When the store root carries the GeoZarr ``spatial:bbox`` (native-CRS extent),
    proj:bbox uses it verbatim rather than re-deriving from coords."""
    zarr_path = tmp_path / "senorge_temperature_daily.zarr"
    ds = xr.Dataset(
        {"tg": (["time", "y", "x"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "y": [7000000.0, 6999000.0, 6998000.0],
            "x": [100000.0, 101000.0, 102000.0],
        },
        # Half-pixel-expanded edges differ from the coord centres above.
        attrs={"proj:code": "EPSG:32633", "spatial:bbox": [99500.0, 6997500.0, 102500.0, 7000500.0]},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="senorge_temperature_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[100000.0, 6998000.0, 102000.0, 7000000.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["proj:bbox"] == [99500.0, 6997500.0, 102500.0, 7000500.0]


def test_build_collection_normalizes_crs84_alias_to_epsg4326(tmp_path: Path) -> None:
    """A CRS84 alias in the store is emitted as canonical EPSG:4326 (and treated as built-in),
    so the map client never receives an alias ZarrLayer can't resolve."""
    zarr_path = tmp_path / "chirps3_precipitation_daily.zarr"
    ds = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "latitude": [4.0, 3.0, 2.0],
            "longitude": [1.0, 2.0, 3.0],
        },
        attrs={"proj:code": "OGC:CRS84"},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="chirps3_precipitation_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[1.0, 2.0, 3.0, 4.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert payload["proj:code"] == "EPSG:4326"  # normalized from OGC:CRS84
    assert "open_climate_service:proj4" not in payload  # built-in → no hint
    assert "proj:wkt2" not in payload


@pytest.mark.parametrize(
    "code,expected",
    [
        ("CRS84", "EPSG:4326"),
        ("OGC:CRS84", "EPSG:4326"),
        ("CRS:84", "EPSG:4326"),  # short OGC form (separators stripped)
        ("urn:ogc:def:crs:OGC:1.3:CRS84", "EPSG:4326"),
        ("EPSG:4326", "EPSG:4326"),
        ("EPSG:32633", "EPSG:32633"),  # projected code passes through unchanged
        (4326, "EPSG:4326"),  # bare EPSG int is prefixed so is_builtin_crs(4326) holds
        ("4326", "EPSG:4326"),  # ...and the bare-number string form too
        (32633, "EPSG:32633"),
    ],
)
def test_canonical_crs_code(code: str | int, expected: str) -> None:
    from open_climate_service.shared.crs import canonical_crs_code

    assert canonical_crs_code(code) == expected


def test_build_collection_crs_render_hints_use_store_wkt(tmp_path: Path) -> None:
    """When the store carries a CF grid-mapping WKT (spatial_ref/crs_wkt), the hints derive
    from it (the CRS.from_wkt branch), and proj:wkt2 is emitted as WKT2."""
    from pyproj import CRS as PyCRS

    wkt = PyCRS.from_epsg(32633).to_wkt(version="WKT2_2019")
    zarr_path = tmp_path / "senorge_temperature_daily.zarr"
    ds = xr.Dataset(
        {"tg": (["time", "y", "x"], np.ones((2, 3, 3), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "y": [7000000.0, 6999000.0, 6998000.0],
            "x": [100000.0, 101000.0, 102000.0],
            "spatial_ref": ((), 0, {"crs_wkt": wkt, "spatial_ref": wkt}),
        },
        attrs={"proj:code": "EPSG:32633"},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    artifact = _artifact(artifact_id="a1", format=ArtifactFormat.ZARR, path=str(zarr_path))
    template = pystac.Collection(
        id="senorge_temperature_daily",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[100000.0, 6998000.0, 102000.0, 7000000.0]]),
            temporal=pystac.TemporalExtent([[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]]),
        ),
    )

    payload = stac_services._build_collection_with_xstac(artifact=artifact, template=template)

    assert "open_climate_service:proj4" not in payload
    assert payload["proj:wkt2"].startswith("PROJCRS")


def _stub_collection_build(monkeypatch: pytest.MonkeyPatch, artifact: ArtifactRecord) -> None:
    """Stub out the store-reading parts of collection building, leaving the media type real."""
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[artifact]))
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", lambda **_: _minimal_xstac_payload())
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})


def test_collection_advertises_the_detected_media_type(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever detection concludes must reach the payload — the `profile` parameter is what
    gates rendering in pyramid-only clients (CLIM-853).

    Detection itself is covered against real stores in test_geozarr_media_type.py; this is the
    plumbing from there to the collection's zarr asset.
    """
    artifact = _artifact(artifact_id="pyr1")
    _stub_collection_build(monkeypatch, artifact)
    monkeypatch.setattr(
        stac_services,
        "zarr_media_type",
        lambda *_a, **_k: "application/vnd.zarr; version=3; profile=multiscales",
    )

    payload = client.get("/stac/collections/chirps3_precipitation_daily").json()

    assert payload["assets"]["zarr"]["type"] == "application/vnd.zarr; version=3; profile=multiscales"


def test_collection_keeps_the_plain_media_type_for_a_flat_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claiming a pyramid for a flat store sends a renderer looking for levels that don't exist."""
    artifact = _artifact(artifact_id="flat1")
    _stub_collection_build(monkeypatch, artifact)
    monkeypatch.setattr(stac_services, "zarr_media_type", lambda *_a, **_k: "application/vnd.zarr; version=3")

    payload = client.get("/stac/collections/chirps3_precipitation_daily").json()

    assert payload["assets"]["zarr"]["type"] == "application/vnd.zarr; version=3"


def test_media_type_detection_is_cached_per_request(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection reads the store's root group; that must not happen on every STAC request.

    The media type is set on the collection *template*, which is rebuilt per request — so it
    sits outside the xstac payload cache and needs its own.
    """
    reads = 0

    def counting_media_type(store_path: str, *, icechunk: bool) -> str:
        nonlocal reads
        reads += 1
        return "application/vnd.zarr; version=3"

    artifact = _artifact(artifact_id="cache1")
    _stub_collection_build(monkeypatch, artifact)
    monkeypatch.setattr(stac_services, "zarr_media_type", counting_media_type)

    client.get("/stac/collections/chirps3_precipitation_daily")
    client.get("/stac/collections/chirps3_precipitation_daily")

    assert reads == 1


def test_media_type_cache_is_keyed_by_artifact_not_store_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-ingest reuses the store path and can cross the pyramid threshold as data grows.

    A path-keyed cache would keep serving the old claim; keying on the artifact id cannot, since
    every write produces a new artifact record.
    """
    detections: list[str] = []

    def growing_media_type(store_path: str, *, icechunk: bool) -> str:
        detections.append(store_path)
        return (
            "application/vnd.zarr; version=3; profile=multiscales"
            if len(detections) > 1
            else "application/vnd.zarr; version=3"
        )

    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(stac_services, "_build_collection_with_xstac", lambda **_: _minimal_xstac_payload())
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})
    monkeypatch.setattr(stac_services, "_zarr_open_kwargs", lambda _: {"consolidated": True})
    monkeypatch.setattr(stac_services, "zarr_media_type", growing_media_type)

    same_path = "/tmp/chirps3_precipitation_daily.icechunk"
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="before", path=same_path)]),
    )
    first = client.get("/stac/collections/chirps3_precipitation_daily").json()

    # Same store path, new artifact record — as a re-ingest produces.
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[_artifact(artifact_id="after", path=same_path)]),
    )
    second = client.get("/stac/collections/chirps3_precipitation_daily").json()

    assert first["assets"]["zarr"]["type"] == "application/vnd.zarr; version=3"
    assert second["assets"]["zarr"]["type"] == "application/vnd.zarr; version=3; profile=multiscales"
    assert detections == [same_path, same_path]  # detected twice, despite the identical path


def test_cf_extension_is_not_declared_when_no_cf_attrs_are_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store with no CF attributes emits no cf: field, so the extension is left undeclared."""
    artifact = _artifact(artifact_id="nocf")
    _stub_collection_build(monkeypatch, artifact)

    payload = client.get("/stac/collections/chirps3_precipitation_daily").json()

    assert not any(key.startswith("cf:") for key in payload["cube:variables"]["precip"])
    assert stac_services.CF_EXTENSION not in payload["stac_extensions"]


# --- reference_system (CLIM-1004) -------------------------------------------------------
#
# The defect was a hardcoded `reference_system=4326` at the xstac call site, so a projected
# store published its UTM metres as degrees. Testing `reference_system_for` alone would not
# catch that returning: the constant has to be gone from the *caller*, so the call-site test
# below asserts on what xstac actually receives.


@pytest.mark.parametrize(
    ("store_crs", "expected"),
    [
        (None, 4326),
        ("", 4326),
        ("EPSG:4326", 4326),
        ("EPSG:32633", 32633),
        ("EPSG:3857", 3857),
        ("OGC:CRS84", 4326),  # a CRS84 alias is geographic WGS84
        ("CRS:84", 4326),
        (4326, 4326),  # a bare number, as canonical_crs_code accepts
        # A proj4 string that names a known CRS resolves to its EPSG code rather than being
        # published verbatim: proj4 is neither WKT2 nor PROJJSON, so the string form of the
        # field cannot carry it.
        ("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +type=crs", 32633),
    ],
)
def test_reference_system_for_resolves_the_stores_own_crs(store_crs: str | int | None, expected: int | str) -> None:
    assert stac_services.reference_system_for(store_crs) == expected


def test_reference_system_for_emits_wkt2_when_there_is_no_epsg_code() -> None:
    """A CRS with no EPSG equivalent must be published as WKT2, never as proj4."""
    # A rotated pole grid has no EPSG code, so this exercises the WKT2 branch.
    proj4 = "+proj=ob_tran +o_proj=longlat +o_lat_p=90 +o_lon_p=0 +lon_0=0 +datum=WGS84 +no_defs +type=crs"
    result = stac_services.reference_system_for(proj4)

    assert isinstance(result, str)
    assert not result.startswith("+proj="), "a proj4 string is not a valid reference_system"
    # WKT2 opens with a CRS keyword; good enough to tell WKT from proj4 without parsing it.
    assert result.split("[")[0].endswith("CRS") or result.startswith("GEOGCRS"), result[:60]


def test_reference_system_for_falls_back_to_4326_and_warns_on_an_uninterpretable_crs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing honest can be published for a CRS we cannot parse, so say so in the log."""
    # `startup.py` sets `propagate = False` on the `open_climate_service` logger, so caplog's
    # root handler never sees these records and `caplog.text` stays empty. Attaching its
    # handler to the emitting logger works whatever the propagation policy is.
    emitting = logging.getLogger(stac_services.__name__)
    emitting.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING", logger=emitting.name):
            assert stac_services.reference_system_for("not a coordinate reference system") == 4326
    finally:
        emitting.removeHandler(caplog.handler)

    assert "Could not interpret CRS" in caplog.text


def test_xstac_is_given_the_stores_crs_not_a_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A projected store's axes must be published in that CRS, not as degrees."""
    ds = xr.Dataset(
        {"tg": (("time", "y", "x"), np.zeros((2, 3, 4), dtype="float32"))},
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "y": [7000000.0, 6999000.0, 6998000.0],
            "x": [-74500.0, -73500.0, -72500.0, -71500.0],
        },
        attrs={"proj:code": "EPSG:32633"},
    )

    seen: dict[str, object] = {}

    def _capture(dataset: xr.Dataset, template: pystac.Collection, **kwargs: object) -> pystac.Collection:
        seen.update(kwargs)
        return template

    monkeypatch.setattr(stac_services, "_open_published_store", lambda _: ds)
    monkeypatch.setattr(stac_services, "xarray_to_stac", _capture)
    monkeypatch.setattr(stac_services, "_add_crs_render_hints", lambda **_: None)

    template = pystac.Collection(
        id="senorge_temperature_daily",
        description="test",
        extent=pystac.Extent(
            pystac.SpatialExtent([[0, 0, 0, 0]]),
            pystac.TemporalExtent([[None, None]]),
        ),
    )
    stac_services._build_collection_with_xstac(
        artifact=_artifact(artifact_id="a1", dataset_id="senorge_temperature_daily"),
        template=template,
    )

    assert seen["reference_system"] == 32633, (
        "the xstac call must pass the store's CRS; 4326 here means the constant is back"
    )


def test_both_cube_paths_agree_on_reference_system() -> None:
    """The static (no time axis) path must resolve the CRS the same way as the temporal one."""
    ds = xr.Dataset(
        {"tg": (("y", "x"), np.zeros((3, 4), dtype="float32"))},
        coords={"y": [7000000.0, 6999000.0, 6998000.0], "x": [-74500.0, -73500.0, -72500.0, -71500.0]},
        attrs={"proj:code": "EPSG:32633"},
    )
    dims = stac_services._build_static_cube_dimensions(ds, "x", "y")
    expected = stac_services.reference_system_for("EPSG:32633")
    assert dims["x"]["reference_system"] == expected
    assert dims["y"]["reference_system"] == expected
