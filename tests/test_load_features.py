"""`load_features` — read a registered feature collection back as WGS 84 GeoJSON (CLIM-926)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.features import templates as feature_templates
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.plugins.processes import load_features as load_features_module
from open_climate_service.plugins.processes.load_features import load_features

DISTRICTS_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}

# A real, projected CRS (UTM zone 29N) whose axes point somewhere other than lon/lat, so a bug
# that fails to reproject shows up as coordinates in the hundreds of thousands, not merely a
# wrong CRS *label*.
UTM_29N = "EPSG:32629"

# Sierra Leone-ish, safely inside UTM zone 29N's area of use.
_WEST, _SOUTH, _EAST, _NORTH = -12.6, 7.9, -12.4, 8.1


def _box(code: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[_WEST, _SOUTH], [_EAST, _SOUTH], [_EAST, _NORTH], [_WEST, _NORTH], [_WEST, _SOUTH]]],
        },
    }


def _collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features) or [_box("SL-01")]}


@pytest.fixture(autouse=True)
def feature_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the store and record index at a temporary directory, isolated from other tests."""
    root = tmp_path / "features"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    return root


@pytest.fixture(autouse=True)
def _districts_template_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feature-template lookup must resolve `districts`, as it would in a deployment."""
    monkeypatch.setattr(feature_templates, "_load_builtin_feature_templates", lambda: [DISTRICTS_TEMPLATE])
    feature_templates.reset_feature_template_caches()


def _register(*, store_crs: str = "EPSG:4326", features: dict[str, Any] | None = None) -> None:
    feature_services.refresh_feature_collection(
        template=DISTRICTS_TEMPLATE, features=features or _collection(), store_crs=store_crs
    )


# --- refusal paths --------------------------------------------------------------------------


def test_an_unknown_id_is_refused() -> None:
    with pytest.raises(ValueError, match="does not name a known feature collection"):
        load_features("no-such-id")


def test_a_declared_but_never_ingested_template_is_refused() -> None:
    """`districts` is a known template (the fixture above declares it) but nothing has
    registered it yet -- distinct from an id nothing declares at all."""
    with pytest.raises(ValueError, match="never been ingested"):
        load_features("districts")


def test_a_matching_version_pins_the_registered_collection() -> None:
    _register()
    record = feature_services.registered_collections()["districts"]

    result = load_features("districts", version=record.created_at.isoformat())

    assert result["features"][0]["id"] == "SL-01"


def test_a_stale_version_is_refused_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    _register()
    called = False

    def read(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("stale bindings must fail before opening the collection")

    monkeypatch.setattr(load_features_module.store, "read_feature_collection", read)

    with pytest.raises(ValueError, match="changed after this job was submitted"):
        load_features("districts", version="2026-01-01T00:00:00+00:00")

    assert called is False


# --- the CRS-reprojection contract --------------------------------------------------------------


def test_a_collection_stored_in_wgs84_round_trips_its_coordinates() -> None:
    _register(store_crs="EPSG:4326")

    result = load_features("districts")

    coords = result["features"][0]["geometry"]["coordinates"][0]
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    assert min(xs) == pytest.approx(_WEST, abs=1e-6)
    assert max(xs) == pytest.approx(_EAST, abs=1e-6)
    assert min(ys) == pytest.approx(_SOUTH, abs=1e-6)
    assert max(ys) == pytest.approx(_NORTH, abs=1e-6)


def test_a_collection_stored_in_a_projected_crs_is_reprojected_to_wgs84() -> None:
    """The genuinely-projected case: bytes on disk are UTM easting/northing (hundreds of
    thousands of metres), and load_features must hand back lon/lat degrees, not raw UTM."""
    _register(store_crs=UTM_29N)

    result = load_features("districts")

    coords = result["features"][0]["geometry"]["coordinates"][0]
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    # A UTM easting/northing pair would be far outside these ranges (hundreds of thousands).
    assert all(-180 <= x <= 180 for x in xs)
    assert all(-90 <= y <= 90 for y in ys)
    assert min(xs) == pytest.approx(_WEST, abs=1e-3)
    assert max(xs) == pytest.approx(_EAST, abs=1e-3)
    assert min(ys) == pytest.approx(_SOUTH, abs=1e-3)
    assert max(ys) == pytest.approx(_NORTH, abs=1e-3)


def test_a_wgs84_collection_is_not_reprojected_unnecessarily(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-op reprojection is still a real geopandas call; skip it when the store is already
    WGS 84, rather than reprojecting a CRS onto itself on every read."""
    import geopandas as gpd

    _register(store_crs="EPSG:4326")
    original_to_crs = gpd.GeoDataFrame.to_crs
    calls: list[Any] = []

    def _spy_to_crs(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original_to_crs(self, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", _spy_to_crs)

    load_features("districts")

    assert calls == []


# --- identifier re-stamping ------------------------------------------------------------------


def test_id_property_is_restamped_onto_the_top_level_id() -> None:
    """`aggregate_spatial` reads its geometry labels from the top-level `id`, not `properties`;
    load_features must re-stamp it there so a loaded collection feeds straight into it with
    meaningful labels rather than sequential integers."""
    _register(features=_collection(_box("SL-01")))

    result = load_features("districts")

    feature = result["features"][0]
    assert feature["id"] == "SL-01"
    assert feature["properties"]["orgUnitCode"] == "SL-01"


def test_every_feature_of_a_multi_feature_collection_is_restamped() -> None:
    second = _box("SL-02")
    second["geometry"]["coordinates"] = [[[-11.6, 6.9], [-11.4, 6.9], [-11.4, 7.1], [-11.6, 7.1], [-11.6, 6.9]]]
    _register(features=_collection(_box("SL-01"), second))

    result = load_features("districts")

    ids = {feature["id"] for feature in result["features"]}
    assert ids == {"SL-01", "SL-02"}


# --- spatial_extent filtering ------------------------------------------------------------------


def test_spatial_extent_narrows_the_read_to_matching_features() -> None:
    far = _box("SL-FAR")
    far["geometry"]["coordinates"] = [[[10.0, 40.0], [10.2, 40.0], [10.2, 40.2], [10.0, 40.2], [10.0, 40.0]]]
    _register(features=_collection(_box("SL-01"), far))

    result = load_features("districts", spatial_extent={"west": -13.0, "south": 7.0, "east": -12.0, "north": 9.0})

    ids = {feature["id"] for feature in result["features"]}
    assert ids == {"SL-01"}


def test_a_non_object_spatial_extent_is_refused() -> None:
    _register()
    with pytest.raises(ValueError, match="spatial_extent must be an object"):
        load_features("districts", spatial_extent="not-an-object")


def test_a_spatial_extent_missing_a_required_key_is_refused() -> None:
    _register()
    with pytest.raises(ValueError, match="west/south/east/north"):
        load_features("districts", spatial_extent={"west": -13.0, "south": 7.0})


def test_a_spatial_extent_with_non_finite_coordinates_is_refused() -> None:
    _register()
    with pytest.raises(ValueError, match="finite"):
        load_features(
            "districts",
            spatial_extent={"west": float("nan"), "south": 7.0, "east": -12.0, "north": 9.0},
        )


def test_a_spatial_extent_with_reversed_bounds_is_refused() -> None:
    _register()
    with pytest.raises(ValueError, match="west < east"):
        load_features(
            "districts",
            spatial_extent={"west": -12.0, "south": 7.0, "east": -13.0, "north": 9.0},
        )


def test_a_spatial_extent_with_an_invalid_crs_is_refused() -> None:
    _register()
    with pytest.raises(ValueError, match="invalid CRS"):
        load_features(
            "districts",
            spatial_extent={"west": -13.0, "south": 7.0, "east": -12.0, "north": 9.0, "crs": "nope"},
        )


# --- the unqualified-read guard is deliberately disabled -------------------------------------


def test_load_features_disables_the_unqualified_read_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graph author naming a collection by id means the whole thing, however large -- the
    same way `load_collection` loads a whole datacube unless a filter narrows it. Verified as a
    spy rather than by constructing a collection past the real 5000-feature limit."""
    _register(features=_collection(_box("SL-01")))
    original_read = load_features_module.store.read_feature_collection
    calls: list[dict[str, Any]] = []

    def _spy_read(record: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_read(record, **kwargs)

    monkeypatch.setattr(load_features_module.store, "read_feature_collection", _spy_read)

    load_features("districts")

    assert calls == [{"bbox": None, "bbox_crs": "EPSG:4326", "max_unqualified_read": None}]
