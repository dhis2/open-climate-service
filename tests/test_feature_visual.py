"""The simplified copy of a feature collection that the map viewer draws (CLIM-1234).

Driven through `refresh_feature_collection`, the door a provider run uses, so the tests cover
the call site as well as the simplification, and through the HTTP routes a browser uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
import shapely
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.features import store, visual
from open_climate_service.ingestions import services as ingestion_services

TEMPLATE = {"id": "districts", "name": "District boundaries", "id_property": "code"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep collections, records and thumbnails out of the developer's data directory."""
    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "vectors")
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    monkeypatch.setattr(api_config, "get_crs", lambda: "EPSG:4326")


def _zigzag(x: float, south: float, north: float, *, steps: int = 200, amplitude: float = 0.0002) -> list[list[float]]:
    """A north-going border at *x* that wiggles by *amplitude* degrees, about 20 m at the equator."""
    return [[x + (amplitude if i % 2 else -amplitude), south + (north - south) * i / steps] for i in range(steps + 1)]


def _neighbours() -> list[dict[str, Any]]:
    """Two districts sharing a detailed border, with names, codes and a property the copy drops."""
    border = _zigzag(0.1, 0.0, 0.1)
    west = [[0.0, 0.0], *border, [0.0, 0.1], [0.0, 0.0]]
    east = [*border, [0.2, 0.1], [0.2, 0.0], border[0]]
    return [
        {
            "type": "Feature",
            "properties": {"code": "W", "name": "West", "population": 100},
            "geometry": {"type": "Polygon", "coordinates": [west]},
        },
        {
            "type": "Feature",
            "properties": {"code": "E", "name": "East", "population": 200},
            "geometry": {"type": "Polygon", "coordinates": [list(reversed(east))]},
        },
    ]


def _refresh(features: list[dict[str, Any]] | None = None, *, store_crs: str = "EPSG:4326", **template: Any) -> Path:
    feature_services.refresh_feature_collection(
        template={**TEMPLATE, **template},
        features={"type": "FeatureCollection", "features": features or _neighbours()},
        store_crs=store_crs,
    )
    record = feature_services.registered_collections()["districts"]
    return Path(str(record.path))


def _epsg(frame: gpd.GeoDataFrame) -> int | None:
    assert frame.crs is not None
    return frame.crs.to_epsg()


def _vertices(frame: gpd.GeoDataFrame) -> int:
    return int(shapely.get_num_coordinates(frame.geometry.values).sum())


def test_a_refresh_writes_a_simplified_copy_beside_the_collection() -> None:
    stored = _refresh()
    copy_path = store.visual_path(stored)

    assert copy_path.is_file()
    assert copy_path.parent == stored.parent
    full = gpd.read_parquet(stored)
    copy = gpd.read_parquet(copy_path)
    assert list(copy["code"]) == ["W", "E"]
    assert list(copy["name"]) == ["West", "East"]
    assert "population" not in copy.columns
    assert _epsg(copy) == 4326
    assert _vertices(copy) < _vertices(full) / 10


def test_a_border_two_districts_share_stays_shared() -> None:
    # Simplifying each polygon on its own moves the two sides of the border differently and
    # opens slivers between neighbours; simplified as a coverage they still tile exactly.
    copy = gpd.read_parquet(store.visual_path(_refresh()))

    assert bool(shapely.coverage_is_valid(copy.to_crs(copy.estimate_utm_crs()).geometry.values))
    assert copy.geometry.is_valid.all()


def test_the_copy_is_in_wgs84_when_the_collection_is_stored_projected() -> None:
    stored = _refresh(store_crs="EPSG:32631")

    assert _epsg(gpd.read_parquet(stored)) == 32631
    copy = gpd.read_parquet(store.visual_path(stored))
    assert _epsg(copy) == 4326
    xmin, ymin, xmax, ymax = copy.total_bounds
    assert (xmin, ymin) == pytest.approx((0.0, 0.0), abs=0.01)
    assert (xmax, ymax) == pytest.approx((0.2, 0.1), abs=0.01)


def test_a_template_sets_the_tolerance() -> None:
    coarse = gpd.read_parquet(store.visual_path(_refresh(display={"simplify_tolerance": 1000})))
    fine = gpd.read_parquet(store.visual_path(_refresh(display={"simplify_tolerance": 1})))

    assert _vertices(coarse) < _vertices(fine)


@pytest.mark.parametrize("declared", [0, -5, "250", True])
def test_a_tolerance_that_is_not_a_positive_number_falls_back_to_the_default(declared: object) -> None:
    template = {"id": "districts", "display": {"simplify_tolerance": declared}}

    assert visual.simplify_tolerance(template) == visual.DEFAULT_SIMPLIFY_TOLERANCE_METRES


def test_points_are_kept_and_lines_simplified() -> None:
    point = {"type": "Point", "coordinates": [0.05, 0.05]}
    line = {"type": "LineString", "coordinates": [[0.0, y / 100] for y in range(101)]}
    stored = _refresh(
        [
            {"type": "Feature", "properties": {"code": "P"}, "geometry": point},
            {"type": "Feature", "properties": {"code": "L"}, "geometry": line},
        ]
    )

    copy = gpd.read_parquet(store.visual_path(stored)).set_index("code")
    # Through UTM and back, so equal to within floating-point noise rather than bit for bit.
    assert copy.geometry["P"].equals_exact(shapely.Point(0.05, 0.05), tolerance=1e-9)
    assert shapely.get_num_coordinates(copy.geometry["L"]) == 2


def test_a_failed_copy_does_not_fail_the_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simplifier broke")

    monkeypatch.setattr(visual, "simplify_for_display", broken)
    stored = _refresh()

    assert stored.is_file()
    assert not store.visual_path(stored).exists()
    assert not list(stored.parent.glob("*.writing"))


def test_a_superseded_version_takes_its_copy_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    first = _refresh()
    _refresh()  # marks the first version as superseded
    latest = _refresh()  # and this one removes it

    assert not first.exists()
    assert not store.visual_path(first).exists()
    assert store.visual_path(latest).is_file()


def test_a_copy_left_without_its_collection_file_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = _refresh()
    orphan = store.visual_path(stored.with_name(f"districts.{'0' * 32}.parquet"))
    orphan.write_bytes(b"left behind")

    store.prune_superseded_files("districts", keep=stored)

    assert not orphan.exists()
    assert store.visual_path(stored).is_file()


def test_the_copy_is_served_and_advertised_as_the_visual_asset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _refresh()
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    served = client.get("/features/districts/visual.parquet")
    assert served.status_code == 200
    assert served.headers["content-type"] == "application/x-parquet"
    downloaded = tmp_path / "visual.parquet"
    downloaded.write_bytes(served.content)
    assert list(gpd.read_parquet(downloaded)["code"]) == ["W", "E"]

    asset = client.get("/stac/collections/districts").json()["assets"]["visual"]
    assert asset["href"] == "http://testserver/features/districts/visual.parquet"
    assert asset["type"] == "application/x-parquet"
    assert asset["roles"] == ["visual"]


def test_a_collection_without_a_copy_does_not_advertise_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _refresh()
    store.visual_path(stored).unlink()
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert client.get("/features/districts/visual.parquet").status_code == 404
    assert "visual" not in client.get("/stac/collections/districts").json()["assets"]


def test_an_unpublished_collection_serves_no_copy(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    feature_services.refresh_feature_collection(
        template=TEMPLATE, features={"type": "FeatureCollection", "features": _neighbours()}, publish=False
    )
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert client.get("/features/districts/visual.parquet").status_code == 404


def test_the_map_viewer_lists_vector_datasets_and_draws_the_visual_copy(client: TestClient) -> None:
    body = client.get("/map").text

    assert 'import { parquetReadObjects } from "https://esm.sh/hyparquet@' in body
    assert 'item.itemType === "feature" && item.publication?.status === "published"' in body
    # The simplified copy when there is one, the stored file otherwise.
    assert "collection.assets?.visual ?? collection.assets?.data" in body
    assert 'storedCrs !== "EPSG:4326"' in body
