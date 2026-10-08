"""Feature collection thumbnails: drawn from the stored geometry at each refresh.

Driven through `refresh_feature_collection`, the door a provider run uses, rather than through
the renderer alone, so the tests cover the call site as well as the drawing. The image is
checked for content — filled inside a polygon, transparent outside it, the right proportions —
because a thumbnail that exists but shows the wrong shape is the failure worth catching.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.shared import thumbnails
from open_climate_service.shared.thumbnails import THUMBNAIL_LONG_SIDE_PIXELS, thumbnail_path

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


def _feature(code: str, geometry: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "properties": {"code": code}, "geometry": geometry}


def _box(code: str, west: float, south: float, east: float, north: float) -> dict[str, Any]:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return _feature(code, {"type": "Polygon", "coordinates": [ring]})


def _refresh(*features: dict[str, Any], collection_id: str = "districts") -> None:
    feature_services.refresh_feature_collection(
        template={**TEMPLATE, "id": collection_id},
        features={"type": "FeatureCollection", "features": list(features)},
    )


def _image(collection_id: str = "districts") -> Image.Image:
    return Image.open(io.BytesIO(thumbnail_path(collection_id).read_bytes())).convert("RGBA")


def _broken_renderer(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("renderer broke")


def _alpha_at(image: Image.Image, fx: float, fy: float) -> int:
    """Alpha at a fractional position, with fy measured from the top."""
    width, height = image.size
    x, y = min(width - 1, int(fx * width)), min(height - 1, int(fy * height))
    return image.getchannel("A").tobytes()[y * width + x]


def test_a_refresh_draws_the_collection_to_its_thumbnail() -> None:
    # Two boxes side by side at the equator, with a gap between them.
    _refresh(_box("W", 0.0, 0.0, 1.0, 1.0), _box("E", 2.0, 0.0, 3.0, 1.0))

    image = _image()
    assert max(image.size) == THUMBNAIL_LONG_SIDE_PIXELS
    assert _alpha_at(image, 0.15, 0.5) == 255  # inside the western box
    assert _alpha_at(image, 0.85, 0.5) == 255  # inside the eastern box
    assert _alpha_at(image, 0.5, 0.5) == 0  # the gap between them


def test_a_geographic_instance_draws_in_degrees() -> None:
    # Drawn in the instance CRS as it is: a one-degree square is square in the image.
    _refresh(_box("N", 10.0, 59.5, 11.0, 60.5))

    width, height = _image().size
    assert height == THUMBNAIL_LONG_SIDE_PIXELS
    assert width == pytest.approx(THUMBNAIL_LONG_SIDE_PIXELS, abs=3)


def test_a_projected_instance_draws_in_its_own_crs(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyproj import Transformer

    monkeypatch.setattr(api_config, "get_crs", lambda: "EPSG:32633")
    _refresh(_box("N", 10.0, 59.5, 11.0, 60.5))

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    xs, ys = to_utm.transform([10.0, 11.0, 11.0, 10.0], [59.5, 59.5, 60.5, 60.5])
    expected = (max(xs) - min(xs)) / (max(ys) - min(ys))
    width, height = _image().size
    assert width / height == pytest.approx(expected, abs=0.02)


def test_points_and_lines_are_drawn_as_well_as_polygons() -> None:
    _refresh(
        _feature("P1", {"type": "Point", "coordinates": [0.0, 0.0]}),
        _feature("P2", {"type": "Point", "coordinates": [1.0, 1.0]}),
        _feature("L1", {"type": "LineString", "coordinates": [[0.0, 1.0], [1.0, 0.0]]}),
    )

    alphas = list(_image().getchannel("A").tobytes())
    assert any(alpha > 0 for alpha in alphas)
    assert sum(alpha > 0 for alpha in alphas) < len(alphas) / 2  # drawn marks, not a filled frame


def test_a_single_point_collection_still_gets_a_thumbnail() -> None:
    _refresh(_feature("P1", {"type": "Point", "coordinates": [5.0, 5.0]}))

    image = _image()
    assert max(image.size) == THUMBNAIL_LONG_SIDE_PIXELS
    assert _alpha_at(image, 0.5, 0.5) > 0


def test_a_geometry_collection_is_drawn_as_its_parts() -> None:
    # A polygon in the lower left and a point in the upper right, in one GeometryCollection.
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    collection = {
        "type": "GeometryCollection",
        "geometries": [{"type": "Polygon", "coordinates": [ring]}, {"type": "Point", "coordinates": [3.0, 3.0]}],
    }
    _refresh(_feature("GC", collection))

    image = _image()
    assert _alpha_at(image, 0.15, 0.85) == 255  # inside the polygon
    assert _alpha_at(image, 0.6, 0.4) == 0  # between the polygon and the point


def test_a_failed_draw_keeps_the_previous_thumbnail_and_the_refresh_still_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refresh(_box("W", 0.0, 0.0, 1.0, 1.0))
    before = thumbnail_path("districts").read_bytes()

    monkeypatch.setattr(thumbnails, "render_features_png", _broken_renderer)
    _refresh(_box("W", 0.0, 0.0, 2.0, 2.0))

    assert thumbnail_path("districts").read_bytes() == before
    assert not list(thumbnail_path("districts").parent.glob(".*.tmp.png"))


def test_the_thumbnail_is_served_and_advertised_in_stac(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _refresh(_box("W", 0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    served = client.get("/datasets/districts/thumbnail.png")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"

    asset = client.get("/stac/collections/districts").json()["assets"]["thumbnail"]
    assert asset["href"] == "http://testserver/datasets/districts/thumbnail.png"
    assert asset["roles"] == ["thumbnail"]


def test_a_collection_without_a_thumbnail_does_not_advertise_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(thumbnails, "render_features_png", _broken_renderer)
    _refresh(_box("W", 0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)

    assert not thumbnail_path("districts").exists()
    assert "thumbnail" not in client.get("/stac/collections/districts").json()["assets"]
