"""GeoJSON geometries for zonal statistics: parsing, supported types, and the cube's CRS."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
POINT_TYPES = frozenset({"Point"})


def parse_geometries(geometries: Any) -> tuple[list[Any], list[str]]:
    """Extract Shapely geometries and stable string labels from GeoJSON input.

    Labels use the feature ``id`` when present, otherwise a sequential integer.
    """
    from shapely.geometry import shape

    from open_climate_service.shared.provenance import record_features

    record_features(geometries)

    if isinstance(geometries, dict):
        gtype = geometries.get("type", "")
        if gtype == "FeatureCollection":
            features = geometries.get("features", [])
            geoms = [shape(f["geometry"]) for f in features]
            labels = [str(f.get("id", i)) for i, f in enumerate(features)]
        elif gtype == "Feature":
            geoms, labels = [shape(geometries["geometry"])], [str(geometries.get("id", 0))]
        else:
            geoms, labels = [shape(geometries)], ["0"]
    else:
        geoms = [shape(g) if isinstance(g, dict) else g for g in geometries]
        labels = [str(i) for i in range(len(geoms))]
    _require_supported_geometry_types(geoms, labels)
    return geoms, labels


def _require_supported_geometry_types(geoms: list[Any], labels: list[str]) -> None:
    """Refuse a geometry type this process has no defined sampling rule for.

    Polygons are area-weighted and points are interpolated. A line or a multipoint would need
    a rule of its own (length weighting, or one value per member), so it is refused by name
    rather than routed through a path that would return a number with no defined meaning.
    """
    for geom, label in zip(geoms, labels, strict=True):
        if geom.geom_type not in POLYGON_TYPES | POINT_TYPES:
            raise ValueError(
                f"aggregate_spatial: geometry '{label}' is a {geom.geom_type}, which is not "
                "supported; only Polygon, MultiPolygon and Point are"
            )


def to_cube_crs(geoms: list[Any], data: xr.Dataset) -> tuple[list[Any], list[Any]]:
    """The geometries in the cube's CRS, and the same geometries in WGS 84.

    The first set is what is aggregated; the second is what the result carries, since the
    vector writers read it as WGS 84.

    GeoJSON coordinates are WGS 84 by definition (RFC 7946), and ``load_features`` always
    returns them that way, but a cube keeps its native grid: seNorge over Norway is UTM 33 in
    metres. Without this every Norwegian kommune fell outside the grid and a job finished with
    nothing but NaN. Geometries whose coordinates are not all within longitude and latitude
    ranges are taken as already being in the cube's CRS and left alone, as is a cube that
    declares no CRS or a geographic one.
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    try:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers .rio

        cube_crs = data.rio.crs
    except Exception:
        cube_crs = None
    if cube_crs is None:
        return geoms, geoms
    crs = CRS.from_user_input(cube_crs.to_wkt())
    if crs.is_geographic:
        return geoms, geoms
    bounds = np.array([g.bounds for g in geoms if not g.is_empty])
    if not bounds.size or not (
        (bounds[:, [0, 2]] >= -180).all()
        and (bounds[:, [0, 2]] <= 180).all()
        and (bounds[:, [1, 3]] >= -90).all()
        and (bounds[:, [1, 3]] <= 90).all()
    ):
        # Already in the cube's CRS: aggregated as given, carried out in WGS 84.
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        return geoms, [transform(to_wgs84, g) for g in geoms]
    to_cube = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    return [transform(to_cube, g) for g in geoms], geoms
