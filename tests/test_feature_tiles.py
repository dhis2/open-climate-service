"""Vector tiles for feature collections (CLIM-1177): built on refresh, served by range, in STAC.

Driven through `refresh_feature_collection` and the HTTP routes rather than through the builder
alone, so the call sites are covered as well as the build.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.features import store, tiles
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactRecord

TEMPLATE = {"id": "districts", "name": "District boundaries", "id_property": "code"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep collections, records and thumbnails out of the developer's data directory."""
    root = tmp_path / "vectors"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path / "data")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    monkeypatch.setattr(ingestion_services, "_artifact_storage_exists", lambda _: True)
    monkeypatch.setattr(api_config, "get_crs", lambda: "EPSG:4326")
    return root


def _box(code: str, name: str, west: float, south: float, east: float, north: float) -> dict[str, Any]:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {
        "type": "Feature",
        "properties": {"code": code, "name": name},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _refresh(**kwargs: Any) -> ArtifactRecord:
    # Two districts in Sierra Leone, side by side.
    return feature_services.refresh_feature_collection(
        template=TEMPLATE,
        features={
            "type": "FeatureCollection",
            "features": [
                _box("W", "Western", -13.3, 8.0, -12.5, 8.8),
                _box("E", "Eastern", -11.5, 7.5, -10.5, 8.5),
            ],
        },
        **kwargs,
    )


def _header(archive: Path) -> dict[str, Any]:
    """The PMTiles v3 header and JSON metadata: what a browser reads before any tile."""
    data = archive.read_bytes()
    assert data[:7] == b"PMTiles" and data[7] == 3
    metadata_offset, metadata_length = struct.unpack_from("<QQ", data, 24)
    raw = data[metadata_offset : metadata_offset + metadata_length]
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    min_lon, min_lat, max_lon, max_lat = (v / 1e7 for v in struct.unpack_from("<iiii", data, 102))
    return {
        "max_zoom": data[101],
        "bounds": (min_lon, min_lat, max_lon, max_lat),
        "metadata": json.loads(raw),
    }


def test_a_refresh_builds_vector_tiles_beside_the_geoparquet() -> None:
    record = _refresh()

    archive = tiles.archive_path(Path(str(record.path)))
    assert archive.is_file()
    header = _header(archive)
    # `auto` would stop at a coarse zoom for two large boxes; boundaries need at least this.
    assert header["max_zoom"] == tiles.MIN_MAX_ZOOM
    (layer,) = header["metadata"]["vector_layers"]
    assert layer["id"] == tiles.TILE_LAYER
    # The id and the name the viewer's popup shows travel with the geometry.
    assert {"code", "name"} <= set(layer["fields"])
    west, south, east, north = header["bounds"]
    assert west == pytest.approx(-13.3, abs=0.01) and east == pytest.approx(-10.5, abs=0.01)
    assert south == pytest.approx(7.5, abs=0.01) and north == pytest.approx(8.8, abs=0.01)


def test_a_projected_collection_is_tiled_in_longitude_and_latitude() -> None:
    """tylertoo reads WGS 84 or Web Mercator only, so a UTM store is reprojected for the build."""
    record = _refresh(store_crs="EPSG:32628")

    west, _south, east, _north = _header(tiles.archive_path(Path(str(record.path))))["bounds"]
    assert west == pytest.approx(-13.3, abs=0.01) and east == pytest.approx(-10.5, abs=0.01)


def test_a_failed_tile_build_does_not_fail_the_refresh(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, _isolated: Path
) -> None:
    import tylertoo

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tiler broke")

    monkeypatch.setattr(tylertoo, "overview", broken)

    record = _refresh()

    assert Path(str(record.path)).is_file()
    assert not tiles.archive_path(Path(str(record.path))).exists()
    assert not [p for p in _isolated.iterdir() if p.name.endswith(".writing")]
    # Nothing advertises or serves an archive that is not there.
    assert client.get("/features/districts/data.pmtiles").status_code == 404
    collection = client.get("/stac/collections/districts").json()
    assert "tiles" not in collection["assets"] and "style" not in collection["assets"]
    assert "data" in collection["assets"]
    assert not [link for link in collection["links"] if link["rel"] == "pmtiles"]
    assert client.get("/features/districts/styles/default.json").status_code == 404


def test_the_archive_is_served_by_range_and_advertised_in_stac(client: TestClient) -> None:
    record = _refresh()
    archive = tiles.archive_path(Path(str(record.path)))

    asset = client.get("/stac/collections/districts").json()["assets"]["tiles"]
    assert asset["href"] == "http://testserver/features/districts/data.pmtiles"
    assert asset["type"] == tiles.PMTILES_MEDIA_TYPE

    # A PMTiles reader starts with the 127-byte header, never the whole archive.
    response = client.get("/features/districts/data.pmtiles", headers={"Range": "bytes=0-126"})
    assert response.status_code == 206
    assert response.headers["content-type"] == tiles.PMTILES_MEDIA_TYPE
    assert response.content == archive.read_bytes()[:127]

    head = client.head("/features/districts/data.pmtiles")
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == archive.stat().st_size


def test_the_archive_is_linked_for_web_map_clients(client: TestClient) -> None:
    """The web-map-links `pmtiles` link names the archive and the layer to show."""
    _refresh()

    collection = client.get("/stac/collections/districts").json()

    (link,) = [link for link in collection["links"] if link["rel"] == "pmtiles"]
    assert link["href"] == "http://testserver/features/districts/data.pmtiles"
    assert link["type"] == tiles.PMTILES_MEDIA_TYPE
    assert link["pmtiles:layers"] == [tiles.TILE_LAYER]
    assert "https://stac-extensions.github.io/web-map-links/v1.3.0/schema.json" in collection["stac_extensions"]


def test_the_style_draws_the_tiles_with_the_collections_attribution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = {**TEMPLATE, "providers": [{"name": "Overture Maps"}]}
    monkeypatch.setattr(feature_services.feature_templates, "get_feature_template", lambda _id: template)
    feature_services.refresh_feature_collection(
        template=template,
        features={"type": "FeatureCollection", "features": [_box("W", "Western", -13.3, 8.0, -12.5, 8.8)]},
    )

    asset = client.get("/stac/collections/districts").json()["assets"]["style"]
    assert asset["href"] == "http://testserver/features/districts/styles/default.json"
    assert asset["type"] == tiles.STYLE_MEDIA_TYPE and asset["roles"] == ["style"]

    response = client.get("/features/districts/styles/default.json")
    assert response.headers["content-type"] == tiles.STYLE_MEDIA_TYPE
    style = response.json()
    assert style["version"] == 8 and style["name"] == "District boundaries"
    assert style["sources"] == {
        "features": {
            "type": "vector",
            "url": "pmtiles://http://testserver/features/districts/data.pmtiles",
            "attribution": "Overture Maps",
        }
    }
    # The viewer's click handlers and clean-up are registered for these layer ids.
    assert [layer["id"] for layer in style["layers"]] == ["features-fill", "features-line", "features-point"]
    assert {layer["source-layer"] for layer in style["layers"]} == {tiles.TILE_LAYER}


def test_an_unpublished_collection_serves_no_tiles(client: TestClient) -> None:
    """The same gate as its GeoParquet: STAC withholds it, so these routes do too."""
    _refresh(publish=False)

    assert client.get("/features/districts/data.pmtiles").status_code == 404
    assert client.get("/features/districts/styles/default.json").status_code == 404


def test_a_superseded_file_takes_its_archive_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    first = Path(str(_refresh().path))
    _refresh()  # marks the first file superseded
    current = Path(str(_refresh().path))  # removes it, past a zero grace period

    assert not first.exists()
    assert not tiles.archive_path(first).exists()
    assert tiles.archive_path(current).is_file()


def test_an_archive_that_cannot_be_removed_is_tried_again_later(monkeypatch: pytest.MonkeyPatch) -> None:
    """The archive goes before its Parquet file, so a failure leaves the version discoverable."""
    monkeypatch.setattr(store, "SUPERSEDED_FILE_GRACE_SECONDS", 0)
    first = Path(str(_refresh().path))
    archive = tiles.archive_path(first)
    _refresh()  # marks the first file superseded
    archive.unlink()
    archive.mkdir()  # an archive that cannot be unlinked
    (archive / "held").touch()

    _refresh()  # tries to remove the first version, and fails on its archive

    assert first.exists() and archive.exists()
    (archive / "held").unlink()
    archive.rmdir()
    archive.touch()
    _refresh()  # the next prune finds the first version again and removes both
    assert not first.exists() and not archive.exists()


def test_the_viewer_draws_with_the_collections_style(client: TestClient) -> None:
    page = client.get("/map").text

    assert 'maplibregl.addProtocol("pmtiles"' in page
    assert "const styleHref = collection.assets?.style?.href;" in page
    assert "for (const layer of style.layers) map.addLayer(layer);" in page
    assert 'const FEATURE_LAYERS = ["features-fill", "features-line", "features-point"];' in page
