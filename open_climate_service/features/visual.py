"""The simplified copy of a feature collection that a browser draws (CLIM-1234).

The stored GeoParquet keeps every vertex the provider delivered, which is what aggregation
needs and far more than a map at country scale can show: Norway's 357 municipalities are
11.6 MB at full detail and 1.1 MB simplified to 250 m, with nothing visible lost at that scale.
So each refresh also writes a *visual* copy next to the collection file, and the map viewer
reads that instead.

It is GeoParquet rather than GeoJSON so that it stays a cloud-native file a client reads
without a server in between, which is the shape Portolan asks of a collection's `visual`
asset. In WGS 84 whatever CRS the collection is stored in, because that is what a web map
draws, and with only the id and name columns, which is all the viewer shows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from open_climate_service import config as api_config
from open_climate_service.features import store

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

DEFAULT_SIMPLIFY_TOLERANCE_METRES = 250.0
"""How far a simplified boundary may move from the stored one.

250 m is under a pixel at the zoom a country is viewed at, and it is where the measured sizes
level off: Norway's municipalities are 2.2 MB at 50 m, 1.1 MB at 250 m and 0.6 MB at 1 km.
A template overrides it with `display.simplify_tolerance`.
"""

_LABEL_COLUMN = "name"


def simplify_tolerance(template: Mapping[str, Any]) -> float:
    """The template's `display.simplify_tolerance` in metres, or the default.

    A value that is not a positive number is logged and replaced by the default rather than
    refused: the visual copy is a convenience, and a typo in it should not fail a refresh.
    """
    display = template.get("display")
    raw = display.get("simplify_tolerance") if isinstance(display, Mapping) else None
    if raw is None:
        return DEFAULT_SIMPLIFY_TOLERANCE_METRES
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        logger.warning(
            "Feature template '%s' declares display.simplify_tolerance %r, which is not a positive "
            "number of metres; using %s",
            template.get("id"),
            raw,
            DEFAULT_SIMPLIFY_TOLERANCE_METRES,
        )
        return DEFAULT_SIMPLIFY_TOLERANCE_METRES
    return float(raw)


def simplify_for_display(frame: gpd.GeoDataFrame, *, tolerance: float) -> gpd.GeoDataFrame:
    """Return *frame* simplified by *tolerance* metres, in WGS 84.

    Simplified in metres, so in a projected CRS: the instance's when it has one, since that is
    the projection its data is drawn in, otherwise the UTM zone the collection falls in.

    Polygons that tile without overlapping — an administrative level — are simplified as a
    coverage, so a border two units share is simplified once and stays shared. Simplifying
    each polygon on its own moves the two sides of a border differently and opens slivers
    between neighbours. Polygons that overlap are not a coverage and are simplified one by one,
    keeping each valid. Lines are simplified, and points are left as they are.
    """
    import shapely
    from pyproj import CRS

    instance_crs = api_config.get_crs()
    working = frame.to_crs(instance_crs if CRS.from_user_input(instance_crs).is_projected else frame.estimate_utm_crs())
    geometry = working.geometry.values
    types = shapely.get_type_id(geometry)
    area_mask = (types == 3) | (types == 6)  # Polygon, MultiPolygon
    line_mask = (types == 1) | (types == 5)  # LineString, MultiLineString
    simplified = geometry.copy()
    if area_mask.any():
        areas = geometry[area_mask]
        if bool(shapely.coverage_is_valid(areas)):
            simplified[area_mask] = shapely.coverage_simplify(areas, tolerance=tolerance)
        else:
            simplified[area_mask] = shapely.simplify(areas, tolerance, preserve_topology=True)
    if line_mask.any():
        simplified[line_mask] = shapely.simplify(geometry[line_mask], tolerance, preserve_topology=True)
    working = working.set_geometry(simplified, crs=working.crs)
    return working.to_crs(store.WGS84)


def write_visual_copy(collection_path: Path, *, template: Mapping[str, Any], id_property: str) -> Path | None:
    """Write the visual copy of the collection stored at *collection_path*, returning its path.

    Called once per refresh, after the collection's record is durable. Written beside the
    target and moved into place, so a reader sees no partial file; and beside *this version*
    of the collection, so it is removed with it (`store.prune_superseded_files`).

    **Never raises.** A collection is not less registered for lacking a simplified copy, and
    the viewer falls back to the stored file.
    """
    target = store.visual_path(collection_path)
    staging = target.with_name(f"{target.name}.{uuid4().hex}.writing")
    try:
        import geopandas as gpd

        frame = gpd.read_parquet(collection_path)
        columns = [column for column in (id_property, _LABEL_COLUMN) if column in frame.columns]
        frame = frame[[*dict.fromkeys(columns), frame.geometry.name]]
        visual = simplify_for_display(frame, tolerance=simplify_tolerance(template))
        visual.to_parquet(staging, write_covering_bbox=True, schema_version="1.1.0")
        staging.replace(target)
        return target
    except Exception:
        logger.warning(
            "Could not write a simplified copy of feature collection '%s'; the map viewer will draw the stored file",
            template.get("id"),
            exc_info=True,
        )
        return None
    finally:
        staging.unlink(missing_ok=True)
