"""Static datasets without a time axis (CLIM-1448): ingested whole, stored as (y, x).

A template with `sync: {kind: static}` and no `period_type` has no periods. Its plugin implements
`fetch(bbox)`, and the store has no time dimension.
Driven through the routes: ingest, the dataset listing, STAC, the ingest form and load_collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset
from open_climate_service.data_manager.services import downloader
from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.openeo.execution import _load_collection_impl
from open_climate_service.streaming import BaseDatasetPlugin

_BBOX = [30.0, -2.0, 30.4, -1.6]
_FETCHES: list[float] = []


class ElevationPlugin(BaseDatasetPlugin):
    """A static source without periods: each fetch returns the next value, so a rewrite shows."""

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        value = float(len(_FETCHES) + 1) * 100
        _FETCHES.append(value)
        x = np.arange(bbox[0] + 0.05, bbox[2], 0.1)
        y = np.arange(bbox[3] - 0.05, bbox[1], -0.1)
        data = np.full((y.size, x.size), value, dtype="float32")
        return xr.Dataset({"elevation": (("y", "x"), data)}, coords={"y": y, "x": x})


class SeriesInDisguisePlugin(BaseDatasetPlugin):
    """A static plugin that wrongly returns a one-step time axis."""

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = ElevationPlugin().fetch(bbox)
        return ds.expand_dims({"t": [np.datetime64("2010-01-01")]})


class LandCoverPlugin(BaseDatasetPlugin):
    """Integer classes with 255 as the no-data code, as categorical sources publish them."""

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        ds = ElevationPlugin().fetch(bbox)
        classes = np.full(ds["elevation"].shape, 3, dtype="uint8")
        classes[0, 0] = 255
        landcover = xr.DataArray(classes, dims=("y", "x"), coords=ds.coords)
        landcover.encoding["_FillValue"] = np.uint8(255)
        return landcover.to_dataset(name="elevation")


class StandaloneElevationPlugin:
    """Exactly the documented static contract: `fetch` alone, no base class supplying the rest."""

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        return ElevationPlugin().fetch(bbox)


class LeftoverTimePlugin(BaseDatasetPlugin):
    """A static plugin that selects one step but keeps its scalar time coordinate."""

    def fetch(self, bbox: list[float], **params: Any) -> xr.Dataset:
        return SeriesInDisguisePlugin().fetch(bbox).isel(t=0)


_TEMPLATES = """
- id: {prefix}_elevation
  name: Elevation
  variable: elevation
  sync:
    kind: static
  extents:
    spatial:
      bbox: [-180, -90, 180, 90]
  ingestion:
    plugin: tests.test_static_datasets.ElevationPlugin
  units: m
  display:
    colormap: viridis
    range: [0, 1000]
- id: {prefix}_disguised
  name: Disguised series
  variable: elevation
  sync:
    kind: static
  ingestion:
    plugin: tests.test_static_datasets.SeriesInDisguisePlugin
- id: {prefix}_leftover_time
  name: Leftover time
  variable: elevation
  sync:
    kind: static
  ingestion:
    plugin: tests.test_static_datasets.LeftoverTimePlugin
- id: {prefix}_standalone
  name: Standalone
  variable: elevation
  sync:
    kind: static
  ingestion:
    plugin: tests.test_static_datasets.StandaloneElevationPlugin
- id: {prefix}_classes
  name: Classes
  variable: elevation
  sync:
    kind: static
  ingestion:
    plugin: tests.test_static_datasets.LandCoverPlugin
"""


@pytest.fixture
def static_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest) -> str:
    """Templates for this test only, with ids unique to it, since records persist across the session."""
    prefix = "static_" + request.node.name.replace("test_", "")[:20]
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "static.yaml").write_text(_TEMPLATES.format(prefix=prefix), encoding="utf-8")
    monkeypatch.setattr(registry, "CONFIGS_DIR", templates)
    _FETCHES.clear()
    return prefix


def _ingest(client: TestClient, dataset_id: str, **extra: Any) -> Any:
    return client.post("/ingestions", json={"dataset_id": dataset_id, "bbox": _BBOX, "publish": True, **extra})


def _store(dataset_id: str) -> xr.Dataset:
    return open_icechunk_dataset(downloader.get_icechunk_path({"id": dataset_id}))


def test_a_static_dataset_is_ingested_without_a_time_axis(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_elevation"

    response = _ingest(client, dataset_id)

    assert response.status_code in (200, 201), response.text
    store = _store(dataset_id)
    assert set(store["elevation"].dims) == {"y", "x"} and "t" not in store.coords
    listed = client.get(f"/datasets/{dataset_id}").json()
    assert listed["extent"]["temporal"] == {"start": None, "end": None}
    assert listed.get("period_type") is None


def test_ingesting_again_keeps_the_data_unless_overwrite_is_set(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_elevation"
    assert _ingest(client, dataset_id).status_code in (200, 201)

    kept = _ingest(client, dataset_id)
    assert kept.status_code in (200, 201), kept.text
    assert _FETCHES == [100.0] and float(_store(dataset_id)["elevation"].max()) == 100.0

    replaced = _ingest(client, dataset_id, overwrite=True)
    assert replaced.status_code in (200, 201), replaced.text
    assert float(_store(dataset_id)["elevation"].max()) == 200.0


def test_load_collection_returns_a_spatial_cube(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_elevation"
    assert _ingest(client, dataset_id).status_code in (200, 201)

    cube = _load_collection_impl(dataset_id, temporal_extent=["2020-01-01", "2020-12-31"])

    assert isinstance(cube, xr.DataArray)
    assert set(cube.dims) == {"y", "x"}
    # It combines with a time series, as a weight or a mask does.
    series = xr.DataArray(np.ones((2, *cube.shape)), dims=("t", *cube.dims), coords={"t": [1, 2], **cube.coords})
    assert (series * cube).dims == ("t", *cube.dims)


def test_a_plugin_that_returns_a_time_axis_is_refused(client: TestClient, static_sources: str) -> None:
    """A plugin error, so a server error naming the fix; the test client raises it rather than answering 500."""
    dataset_id = f"{static_sources}_disguised"

    with pytest.raises(RuntimeError, match=r"without a time axis must be fetched as \(y, x\)"):
        _ingest(client, dataset_id)

    assert not downloader.get_icechunk_path({"id": dataset_id}).exists()


def test_the_ingest_form_asks_for_no_dates(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_elevation"

    page = client.get(f"/data-sources/{dataset_id}?f=html").text
    assert "this dataset has no time axis" in page and 'id="start"' not in page
    started = client.post("/manage/ingest", data={"dataset_id": dataset_id}, headers={"Accept": "text/event-stream"})
    assert "Start period is required" not in started.text


def test_an_overwrite_whose_rebuild_fails_keeps_the_stored_data(
    client: TestClient, static_sources: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.ingestions import services

    dataset_id = f"{static_sources}_elevation"
    assert _ingest(client, dataset_id).status_code in (200, 201)
    monkeypatch.setattr(
        services, "_maybe_build_pyramid", lambda *args, **kwargs: services._StoreNormalizationResult(completed=False)
    )

    with pytest.raises(RuntimeError, match="Could not normalize"):
        _ingest(client, dataset_id, overwrite=True)

    assert float(_store(dataset_id)["elevation"].max()) == 100.0


def test_a_plugin_implementing_only_fetch_is_accepted(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_standalone"

    response = _ingest(client, dataset_id)

    assert response.status_code in (200, 201), response.text
    assert set(_store(dataset_id)["elevation"].dims) == {"y", "x"}


def test_a_leftover_time_coordinate_is_refused(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_leftover_time"

    with pytest.raises(RuntimeError, match="must not carry a time coordinate"):
        _ingest(client, dataset_id)

    assert not downloader.get_icechunk_path({"id": dataset_id}).exists()


def test_an_integer_no_data_code_is_kept(client: TestClient, static_sources: str) -> None:
    dataset_id = f"{static_sources}_classes"

    assert _ingest(client, dataset_id).status_code in (200, 201)

    stored = _store(dataset_id)["elevation"].load()
    assert bool(stored.isnull()[0, 0]) and int(stored[1, 1]) == 3


def test_a_cancelled_ingest_writes_nothing(static_sources: str) -> None:
    from open_climate_service.ingestions import services
    from open_climate_service.jobs.models import JobCancelledError

    dataset_id = f"{static_sources}_elevation"
    with pytest.raises(JobCancelledError):
        services.create_artifact(
            dataset=registry.get_dataset(dataset_id) or {},
            start=None,
            end=None,
            bbox=_BBOX,
            country_code=None,
            overwrite=False,
            publish=False,
            is_cancel_requested=lambda: True,
        )

    assert _FETCHES == [] and not downloader.get_icechunk_path({"id": dataset_id}).exists()


def test_a_workflow_result_keeps_its_recorded_cadence() -> None:
    """A static template without a period_type, whose result has a time axis and a recorded cadence."""
    from types import SimpleNamespace

    from open_climate_service.ingestions import services
    from open_climate_service.ingestions.schemas import ArtifactFormat

    latest = SimpleNamespace(format=ArtifactFormat.ICECHUNK, period_type="daily")
    template = {"id": "derived", "sync": {"kind": "static"}}

    assert services._dataset_period_type(latest, template) == "daily"  # type: ignore[arg-type]


def test_a_rebuilt_store_renames_any_time_axis_alias(tmp_path: Path) -> None:
    """The rewrite after ingest finds the axis under every name publication accepts, not only t/time."""
    from open_climate_service.ingestions import services
    from open_climate_service.streaming.store import open_or_create_repo

    store_path = tmp_path / "aliased.icechunk"
    ds = ElevationPlugin().fetch(_BBOX).expand_dims({"date": [np.datetime64("2024-01-01")]})
    session = open_or_create_repo(store_path).writable_session("main")
    ds.to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("aliased")

    result = services._maybe_build_pyramid(store_path, {"id": "aliased"}, retain_previous=False)

    assert result.completed
    assert "t" in _store_at(store_path).dims and "date" not in _store_at(store_path).dims


def _store_at(path: Path) -> xr.Dataset:
    return open_icechunk_dataset(path)
