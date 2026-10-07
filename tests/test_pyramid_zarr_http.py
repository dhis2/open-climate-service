"""A pyramided store opened with xarray over HTTP reaches its data (CLIM-1354).

A pyramided root holds only the time coordinate and ``spatial_ref``; the data sits in level
groups. These tests open the store the way a notebook does — ``xr.open_zarr`` against a real
server — because the failure was silent: an empty dataset, no error, and nothing a unit test of
the metadata builder would notice.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("icechunk")
pytest.importorskip("topozarr")
pytest.importorskip("aiohttp")
uvicorn = pytest.importorskip("uvicorn")

from open_climate_service.data_manager.services.downloader import (  # noqa: E402
    needs_pyramid,
    write_to_icechunk_store,
)
from open_climate_service.ingestions import services as ingestion_services  # noqa: E402
from open_climate_service.ingestions.schemas import (  # noqa: E402
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.main import app  # noqa: E402
from open_climate_service.openeo.jobs import _result_assets  # noqa: E402
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus  # noqa: E402
from open_climate_service.shared.time import utc_now  # noqa: E402
from open_climate_service.stac import services as stac_services  # noqa: E402

DATASET_ID = "chirps3_precipitation_daily"
SIZE = 1100  # just over the 1024 x 1024 pyramid threshold


def _pyramid_artifact(store_path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="pyramid-http",
        dataset_id=DATASET_ID,
        dataset_name="CHIRPS3 precipitation",
        variable="precip",
        format=ArtifactFormat.ICECHUNK,
        path=str(store_path),
        asset_paths=[str(store_path)],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-02", bbox=(28.8, -2.9, 30.9, -1.0)),
        coverage=ArtifactCoverage(
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-02"),
            spatial=CoverageSpatial(xmin=28.8, ymin=-2.9, xmax=30.9, ymax=-1.0),
        ),
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=DATASET_ID),
    )


@pytest.fixture(scope="module")
def pyramid_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ds = xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((2, SIZE, SIZE), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
            "y": np.linspace(-1.0, -2.9, SIZE),
            "x": np.linspace(28.8, 30.9, SIZE),
        },
    )
    assert needs_pyramid(ds)
    store_path = tmp_path_factory.mktemp("pyramid") / "precip.icechunk"
    write_to_icechunk_store(ds, store_path, crs="EPSG:4326")
    return store_path


@pytest.fixture(scope="module")
def live_server() -> Generator[str, None, None]:
    """The app served by uvicorn on a free port: xarray needs a real HTTP origin."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def published(pyramid_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _pyramid_artifact(pyramid_store)
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[artifact]))
    stac_services._clear_xstac_collection_cache()


def _assert_full_resolution(ds: Any) -> None:
    assert list(ds.data_vars) == ["precip"]
    assert ds.sizes["y"] == SIZE and ds.sizes["x"] == SIZE


@pytest.mark.usefixtures("published")
def test_the_full_resolution_level_opens_as_a_store_of_its_own(live_server: str) -> None:
    with xr.open_zarr(f"{live_server}/zarr/{DATASET_ID}/0", zarr_format=3, consolidated=True) as ds:
        _assert_full_resolution(ds)


@pytest.mark.usefixtures("published")
def test_the_root_opens_at_the_full_resolution_level(live_server: str) -> None:
    with xr.open_zarr(f"{live_server}/zarr/{DATASET_ID}", group="0", zarr_format=3, consolidated=True) as ds:
        _assert_full_resolution(ds)


@pytest.mark.usefixtures("published")
def test_a_coarser_level_lists_only_its_own_arrays(live_server: str) -> None:
    """Each level's metadata is its own subtree, keyed relative to the level."""
    meta = httpx.get(f"{live_server}/zarr/{DATASET_ID}/1/zarr.json").json()

    nodes = meta["consolidated_metadata"]["metadata"]
    assert "precip" in nodes
    assert not any(key.startswith(("0/", "1/")) for key in nodes)
    with xr.open_zarr(f"{live_server}/zarr/{DATASET_ID}/1", zarr_format=3, consolidated=True) as ds:
        assert list(ds.data_vars) == ["precip"]
        assert ds.sizes["x"] < SIZE


@pytest.mark.usefixtures("published")
def test_the_stac_asset_opens_at_the_data(live_server: str) -> None:
    """A client following the catalogue — href plus `xarray:open_kwargs` — gets the variable."""
    collection = httpx.get(f"{live_server}/stac/collections/{DATASET_ID}", timeout=60).json()
    asset = collection["assets"]["zarr"]

    assert asset["type"] == "application/vnd.zarr; version=3; profile=multiscales"
    assert asset["xarray:open_kwargs"]["group"] == "0"
    assert collection["assets"]["icechunk"]["xarray:open_kwargs"]["group"] == "0"
    with xr.open_zarr(asset["href"], **asset["xarray:open_kwargs"]) as ds:
        _assert_full_resolution(ds)


@pytest.mark.usefixtures("published")
def test_a_job_result_asset_opens_at_the_data(live_server: str) -> None:
    """A workflow writing to a managed dataset links the same store, with the same arguments."""
    record = OpenEOJobRecord(
        id="job-1",
        status=OpenEOJobStatus.FINISHED,
        created=utc_now(),
        updated=utc_now(),
        usage={"output_path": f"managed://{DATASET_ID}"},
    )
    asset = _result_assets(record)["zarr"]

    assert asset["xarray:open_kwargs"]["group"] == "0"
    with xr.open_zarr(f"{live_server}{asset['href']}", **asset["xarray:open_kwargs"]) as ds:
        _assert_full_resolution(ds)
