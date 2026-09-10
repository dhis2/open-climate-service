"""Store-level CRS/extent attributes written by ``write_geozarr_attrs``.

The store conveys its CRS through the CF grid-mapping (``crs_wkt``) and the GeoZarr
``proj:`` convention, and its native-CRS extent through the GeoZarr ``spatial:bbox`` root
attribute. No non-standard ``proj4`` / ``bounds`` attrs are written — those duplicated the
above and nothing consumed them (GDAL/QGIS read ``crs_wkt``).

The ``proj:`` set has to be enough on its own. ``create_geozarr_attrs`` reduces an EPSG
input to ``proj:code``, and a code is only as good as the reader's lookup table: proj4,
which the map viewer resolves through, ships WGS84, Web Mercator and the UTM zones and
nothing else. So ``store_crs_attrs`` adds the full definition for anything outside that
set, and the tests below pin which codes get one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from open_climate_service.shared.crs import is_builtin_crs
from open_climate_service.streaming.protocol import GridSpec
from open_climate_service.streaming.store import write_geozarr_attrs


def _init_group(path: Path) -> str:
    store = str(path / "s.zarr")
    zarr.open_group(store, mode="w", zarr_format=3)
    return store


def test_write_geozarr_attrs_writes_standard_extent_not_custom_attrs(tmp_path: Path) -> None:
    """A projected store carries proj:code + the GeoZarr spatial:bbox, and no bespoke
    proj4/bounds attrs (which nothing reads and which duplicate crs_wkt / spatial:bbox)."""
    store = _init_group(tmp_path)
    spec = GridSpec(shape=(3, 3), crs=32633, dtype=np.dtype("float32"), x_dim="x", y_dim="y")
    bbox = [-74500.0, 6450500.0, 1119500.0, 7999500.0]  # native UTM33 metres

    write_geozarr_attrs(store, spec=spec, bbox=bbox)

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs["proj:code"] == "EPSG:32633"
    assert attrs["spatial:bbox"] == bbox  # native-CRS extent, GeoZarr convention
    assert "proj4" not in attrs
    assert "bounds" not in attrs


def test_write_geozarr_attrs_wgs84_extent(tmp_path: Path) -> None:
    store = _init_group(tmp_path)
    bbox = [-13.5, 6.9, -10.1, 10.0]
    spec = GridSpec(shape=(3, 3), crs=4326, dtype=np.dtype("float32"), x_dim="x", y_dim="y")

    write_geozarr_attrs(store, spec=spec, bbox=bbox)

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs["proj:code"] == "EPSG:4326"
    assert attrs["spatial:bbox"] == bbox
    assert "proj4" not in attrs
    assert "bounds" not in attrs


@pytest.mark.parametrize(
    "code,is_builtin",
    [
        ("EPSG:4326", True),
        ("EPSG:3857", True),
        ("CRS84", True),
        ("CRS:84", True),
        ("EPSG:32633", False),
        (4326, True),  # bare EPSG int normalizes to EPSG:4326
        (32633, False),
    ],
)
def test_is_builtin_crs(code: str | int, is_builtin: bool) -> None:
    assert is_builtin_crs(code) is is_builtin


def test_write_geozarr_attrs_carries_a_definition_for_a_code_proj4_does_not_ship(
    tmp_path: Path,
) -> None:
    """A store on a national grid must describe its CRS in full, not just name it.

    proj4 resolves WGS84, Web Mercator and the UTM zones from a code and nothing else, and
    a client handed a code it cannot resolve deliberately leaves the CRS unresolved rather
    than inferring another one. EPSG:27700 is the boundary case: with ``proj:code`` alone
    such a store ingests cleanly and then renders in the wrong place, so the definition has
    to travel with it (CLIM-833).

    Both fields are asserted by parsing them back, not by matching text: what matters is
    that a reader recovers *this* CRS from them, and pyproj's exact WKT rendering is free
    to change between versions.
    """
    from pyproj import CRS

    store = _init_group(tmp_path)
    spec = GridSpec(shape=(3, 3), crs=27700, dtype=np.dtype("float32"), x_dim="x", y_dim="y")

    write_geozarr_attrs(store, spec=spec, bbox=[100000.0, 100000.0, 600000.0, 700000.0])

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    wkt2, projjson = attrs["proj:wkt2"], attrs["proj:projjson"]
    assert isinstance(wkt2, str) and isinstance(projjson, dict)
    assert attrs["proj:code"] == "EPSG:27700"
    assert CRS.from_wkt(wkt2) == CRS.from_epsg(27700)
    assert CRS.from_json_dict(projjson) == CRS.from_epsg(27700)


def test_write_geozarr_attrs_leaves_a_builtin_crs_as_the_code_alone(tmp_path: Path) -> None:
    """EPSG:4326 needs no definition: it is what clients reproject *to*.

    Nearly every store we hold is WGS84, so writing a kilobyte of redundant WKT2 and
    PROJJSON into each one's root metadata buys nothing.
    """
    store = _init_group(tmp_path)
    spec = GridSpec(shape=(3, 3), crs=4326, dtype=np.dtype("float32"), x_dim="x", y_dim="y")

    write_geozarr_attrs(store, spec=spec, bbox=[-13.5, 6.9, -10.1, 10.0])

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs["proj:code"] == "EPSG:4326"
    assert "proj:wkt2" not in attrs
    assert "proj:projjson" not in attrs
