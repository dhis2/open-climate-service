"""Overture Maps feature provider: one bbox extract per theme (CLIM-893).

Overture publishes its themes as GeoParquet on S3, partitioned and carrying a per-row `bbox`
struct, so a country-sized window is a predicate pushdown on plain numeric fields rather than a
download-and-clip. `divisions` is the theme this exists for: administrative boundaries for a
country with no usable DHIS2 hierarchy of its own, and the polygons an aggregation runs over.

Written generic over themes because the theme is a template parameter, not a plugin per theme
(CLIM-836, build step 8). It returns a FeatureCollection, the single form the provider seam
defines -- which is what divisions being *small* buys us: measured against the live bucket,
Sierra Leone's whole extent is 201 features across every level, not the millions a buildings
extract would be. Themes at that scale need a streaming-to-file contract this deliberately
does not have; see "Out of scope" in CLIM-836.

**`filters` is load-bearing, not a convenience.** A bbox window returns every level that
overlaps it. For Sierra Leone that measured as country 6, region 18, county 44, localadmin 15,
locality 43, macrohood 24, neighborhood 49, dependency 2 -- and aggregating over a mixture of
administrative levels is rarely what anyone means. `{"subtype": "county"}` is the usual fix.

**A bbox is a rectangle, not a border**, so a window over one country also returns its
neighbours': a Sierra Leone extent measured 44 counties, of which Coyah and Conakry are in
Guinea. The extract is therefore confined to the instance's own country by default, matching
Overture's `country` column. That column is ISO 3166-1 **alpha-2** while an instance declares
`extent.country_code` as **alpha-3** (`NPL`), so the two are reconciled here rather than asking
a template to restate a code the instance already holds.

**The release id is the version.** Overture publishes monthly, so a release *is* a version and
"is there a newer one than we hold" is answerable without comparing data. It is passed
explicitly rather than resolved to `latest`, so an instance upgrades deliberately and a
re-extract is reproducible.

**Licence.** Divisions incorporate OpenStreetMap, so the theme is ODbL: attribution plus
share-alike on a derived database. The template declaring this provider carries `license` and
`attribution`; serving an extract without surfacing them is a licence breach, not an oversight.
CLIM-1010 covers where that metadata is published.

Ships built-in but is written against public names only (`features.providers.feature_provider`,
`extents.services.get_extent`), so the same file would work unchanged as an installed-package or
`plugins_dir` plugin.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from open_climate_service.extents.services import get_extent
from open_climate_service.features.providers import feature_provider

logger = logging.getLogger(__name__)

DEFAULT_THEME = "divisions"

# A theme is a family of types, not one type: `divisions` publishes `division` (points),
# `division_area` (polygons) and `division_boundary` (lines). Only the polygons are aggregation
# zones, so the theme a template names resolves to the one type worth extracting. A template
# wanting another may still name `type` outright.
_THEME_DEFAULT_TYPE = {
    "divisions": "division_area",
    "buildings": "building",
    "places": "place",
    "transportation": "segment",
    "addresses": "address",
    "base": "land_cover",
}

# Carried through to every feature's properties unless a template names `columns` itself.
# `names` is a struct -- its `primary` is flattened to a plain `name`, see `_properties`.
_DEFAULT_COLUMNS = ("id", "subtype", "class", "names", "country", "region", "admin_level")

_GEOMETRY_COLUMN = "geometry"
_COUNTRY_COLUMN = "country"

# What a template writes to keep a neighbouring country's divisions, where the bbox is the whole
# of the intended selection and a border is not.
ANY_COUNTRY = "any"


@feature_provider("overture")
def overture_features(
    *,
    release: str,
    bbox: Sequence[float] | None = None,
    theme: str = DEFAULT_THEME,
    type: str | None = None,  # noqa: A002 -- Overture's own field name, and a template writes it
    country: str | None = None,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    stac: bool = False,
) -> dict[str, Any]:
    """Extract one Overture theme for a bounding box as a GeoJSON FeatureCollection.

    `bbox` is `[west, south, east, north]` in lon/lat. Omitted, it defaults to the instance
    extent, which is what a template almost always wants and keeps the shipped template free of
    a per-instance coordinate list.

    `country` confines the extract to one country, given as ISO 3166-1 alpha-2 or alpha-3.
    Omitted, it defaults to the instance's own `extent.country_code`, because a bbox crosses
    borders and a neighbour's divisions are almost never wanted. Pass `"any"` to keep them,
    for an instance whose extent is deliberately transboundary.

    `filters` keeps only rows whose column matches, as `{column: value}` or
    `{column: [value, ...]}` -- see the module docstring on why divisions need one. A `country`
    named here wins over the parameter above, so a template can always say exactly what it means.

    `stac` selects Overture's STAC catalogue for partition pruning. It is **off by default**
    because it does not work for every type: `division_area` returns nothing at all through it,
    while the ordinary path answers the same query in about two seconds, divisions being a small
    theme. It is worth turning on for a large theme like buildings, where pruning is the
    difference between seconds and tens of minutes.
    """
    window = _resolve_bbox(bbox)
    overture_type = _resolve_type(theme=theme, type_=type)
    keep = tuple(columns) if columns is not None else _DEFAULT_COLUMNS
    selection = _with_country(filters, country=country)

    reader = _record_batch_reader(overture_type, bbox=window, release=release, stac=stac)
    if reader is None:
        # The client returns None rather than raising, and prints its own "No data found" to
        # stdout, so without this a misconfigured extract looks like an empty answer.
        raise ValueError(
            f"Overture returned no data for type {overture_type!r} in release {release!r} for bbox "
            f"{list(window)}"
            + (" (stac=True prunes partitions and does not cover every type; try stac=False)" if stac else "")
        )

    _require_columns(reader.schema.names, keep=keep, filters=selection, overture_type=overture_type)

    features: list[dict[str, Any]] = []
    for batch in reader:
        for row in batch.to_pylist():
            if not _matches(row, selection):
                continue
            geometry = _to_geojson_geometry(row.get(_GEOMETRY_COLUMN))
            if geometry is None:
                continue
            features.append({"type": "Feature", "geometry": geometry, "properties": _properties(row, keep)})

    if not features:
        raise ValueError(
            f"Overture {overture_type!r} returned no rows for bbox {list(window)} in release {release!r}"
            + (f" matching {dict(selection)}" if selection else "")
        )

    logger.info(
        "Extracted %d Overture %s features (release %s, type %s) for bbox %s",
        len(features),
        theme,
        release,
        overture_type,
        list(window),
    )
    return {"type": "FeatureCollection", "features": features}


def _record_batch_reader(overture_type: str, *, bbox: tuple[float, float, float, float], release: str, stac: bool):  # type: ignore[no-untyped-def]
    """The one network call, isolated so a test can substitute a reader without a live bucket."""
    from overturemaps import core

    return core.record_batch_reader(overture_type, bbox=bbox, release=release, stac=stac)


def _resolve_bbox(bbox: Sequence[float] | None) -> tuple[float, float, float, float]:
    """Validate an explicit window, or fall back to the instance extent."""
    if bbox is None:
        extent = get_extent()
        if not extent or not extent.get("bbox"):
            raise ValueError(
                "Overture extract needs a bbox: none was given and this instance declares no extent to fall back on"
            )
        bbox = extent["bbox"]
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"Overture bbox must be [west, south, east, north], got {values!r}")
    west, south, east, north = (float(value) for value in values)
    if west >= east or south >= north:
        raise ValueError(f"Overture bbox is empty or inverted: {values!r}")
    return west, south, east, north


def _with_country(filters: Mapping[str, Any] | None, *, country: str | None) -> dict[str, Any]:
    """Add a country filter to the template's own, unless it already names one or opts out.

    An explicit `filters: {country: ...}` wins: a template that says exactly what it means is
    never second-guessed. Otherwise the instance's declared country confines the window, which
    is what makes the shipped template correct on an instance it knows nothing about.
    """
    selection = dict(filters or {})
    if _COUNTRY_COLUMN in selection:
        return selection
    if country is not None and country.strip().lower() == ANY_COUNTRY:
        return selection
    code = _alpha_2(country) if country is not None else _instance_alpha_2()
    if code is not None:
        selection[_COUNTRY_COLUMN] = code
    return selection


def _instance_alpha_2() -> str | None:
    """The instance's own country as alpha-2, or None if it declares no usable code.

    No code is not an error: an instance may legitimately have a transboundary extent, and the
    bbox alone is then the whole of the selection.
    """
    extent = get_extent() or {}
    declared = extent.get("country_code")
    if not isinstance(declared, str) or not declared.strip():
        return None
    code = _alpha_2(declared)
    if code is None:
        logger.warning(
            "Instance extent declares country_code %r, which is not an ISO 3166-1 code; "
            "the Overture extract will not be confined to one country",
            declared,
        )
    return code


def _alpha_2(code: str) -> str | None:
    """Normalise an ISO 3166-1 alpha-2 or alpha-3 code to the alpha-2 Overture stores.

    An instance declares `extent.country_code` as alpha-3 (`NPL`, what WorldPop needs), while
    Overture's `country` column is alpha-2 (`NP`). Looked up rather than mapped by hand so the
    register cannot go stale silently -- the same reasoning that put `license_expression` in
    this project rather than a vendored SPDX list. `iso3166` is that register and nothing else,
    at about 40 KB; `pycountry` answers the same question but carries ~21 MB of translations
    this has no use for.
    """
    from iso3166 import countries

    try:
        return str(countries.get(code.strip()).alpha2)
    except KeyError:
        return None


def _resolve_type(*, theme: str, type_: str | None) -> str:
    """Pick the Overture type to extract, preferring an explicit one over the theme's default."""
    if type_ is not None:
        return type_
    try:
        return _THEME_DEFAULT_TYPE[theme]
    except KeyError:
        raise ValueError(
            f"No default Overture type for theme {theme!r}. Known themes: "
            f"{', '.join(sorted(_THEME_DEFAULT_TYPE))}. Name `type` explicitly for any other."
        ) from None


def _require_columns(
    available: Sequence[str], *, keep: Sequence[str], filters: Mapping[str, Any] | None, overture_type: str
) -> None:
    """Refuse a projection or filter naming a column this type does not have.

    A filter on an absent column would otherwise match nothing and surface as "returned no rows",
    which reads as "this bbox is empty" rather than "this template has a typo".
    """
    known = set(available)
    missing = [name for name in keep if name not in known]
    if missing:
        raise ValueError(f"Overture {overture_type!r} has no column(s): {', '.join(missing)}")
    unknown = [name for name in (filters or {}) if name not in known]
    if unknown:
        raise ValueError(f"Overture {overture_type!r} has no column(s) to filter on: {', '.join(unknown)}")


def _matches(row: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Whether a row satisfies every `{column: value-or-values}` entry."""
    if not filters:
        return True
    for name, wanted in filters.items():
        values = wanted if isinstance(wanted, (list, tuple, set)) else (wanted,)
        if row.get(name) not in values:
            return False
    return True


def _to_geojson_geometry(geometry: Any) -> dict[str, Any] | None:
    """Decode Overture's WKB geometry to a GeoJSON mapping, skipping a row that has none.

    Overture carries geometry as WKB in a plain binary column. A row without one is not an error
    -- the store's identity contract cares about ids, and a boundary with no shape is nothing to
    aggregate over -- so it is dropped, and the emptiness check upstream catches the case where
    that leaves nothing at all.
    """
    if not geometry:
        return None
    from shapely import wkb
    from shapely.geometry import mapping

    return dict(mapping(wkb.loads(bytes(geometry))))


def _properties(row: Mapping[str, Any], keep: Sequence[str]) -> dict[str, Any]:
    """Project a row to the declared columns, flattening Overture's `names` struct to `name`.

    `names` is `{primary, common, rules}`; `primary` is the label a map or an export wants, and
    the nested struct would otherwise reach GeoJSON as a dict nobody reads. The struct is kept
    alongside only if a template asked for it by name *and* nothing else -- the flattened `name`
    is what the default projection is for.
    """
    properties: dict[str, Any] = {}
    for name in keep:
        value = row.get(name)
        if name == "names":
            properties["name"] = value.get("primary") if isinstance(value, Mapping) else None
            continue
        properties[name] = value
    return properties
