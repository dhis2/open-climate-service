"""Read a dataset's own CRS, and describe it to a reader.

Data is stored in its **native** CRS — the CRS the source provides it in — never
the instance-wide config CRS. These helpers recover that native CRS from a
dataset so that ingestion writes it, and serving (STAC, coverage) reports it,
without ever consulting ``api_config.get_crs()``.

:func:`store_crs_attrs` is the other direction: what a *store* has to say about its CRS
for a client to resolve it without a lookup table of its own.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


# CRSes that map clients (proj4js, GDAL) resolve from the authority code alone; any
# other CRS needs a full definition (proj4 / WKT) surfaced to the client. CRS84 aliases
# are normalized to EPSG:4326 before this check (see canonical_crs_code).
_BUILTIN_CRS_CODES = frozenset({"EPSG:4326", "EPSG:3857"})


def canonical_crs_code(code: str | int) -> str:
    """Collapse CRS84 geographic aliases to the canonical ``EPSG:4326``.

    A CRS84 alias (``CRS84``, ``OGC:CRS84``, the short OGC form ``CRS:84``, or a CRS84
    URI) is geographic WGS84, but map reprojectors resolve only ``EPSG:4326`` /
    ``EPSG:3857`` from a code — so mapping the alias to ``EPSG:4326`` lets every consumer
    work with one canonical code. Separators are stripped before matching so ``CRS:84`` is
    caught too. A bare EPSG number (the int ``4326`` or the string ``"4326"``) is prefixed
    to ``EPSG:4326`` so downstream checks like :func:`is_builtin_crs` see a full code; any
    other input is returned unchanged (as a string).
    """
    s = str(code).strip()
    if s.isdigit():
        return f"EPSG:{s}"
    return "EPSG:4326" if re.sub(r"[^A-Z0-9]", "", s.upper()).endswith("CRS84") else s


def is_builtin_crs(code: str | int) -> bool:
    """True for a CRS a client resolves from the code alone (EPSG:4326 / EPSG:3857).

    CRS84 aliases are normalized to EPSG:4326 first.
    """
    return canonical_crs_code(code).upper() in _BUILTIN_CRS_CODES


def dataset_crs(ds: "xr.Dataset", default: str = "EPSG:4326") -> str:
    """Return *ds*'s own CRS as an ``EPSG:xxxx`` string.

    Prefers the GeoZarr ``proj:code`` / ``proj:epsg`` root attribute written at
    ingest, then the rioxarray-detected CRS, falling back to *default*
    (WGS84). The instance config CRS is deliberately never consulted — every
    dataset keeps the CRS its source delivered it in.
    """
    code = ds.attrs.get("proj:code") or ds.attrs.get("proj:epsg")
    if code:
        return str(code)
    try:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]

        rio_crs = ds.rio.crs
        if rio_crs is not None:
            epsg = rio_crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
    except Exception:
        pass
    return default


def store_crs_attrs(code: str | int) -> dict[str, Any]:
    """The ``proj:`` attributes a store must carry to describe *code* by itself.

    ``proj:code`` identifies a CRS only to a reader that can look the code up, and map
    clients cannot. proj4 — which zarr-layer, and so the viewer, resolves through — ships
    definitions for WGS84, Web Mercator and every UTM zone, and knows nothing else; given a
    code outside that set it deliberately leaves the CRS unresolved rather than inferring a
    different one. A store on a national grid (EPSG:27700, EPSG:2154) would then ingest
    cleanly and render in the wrong place, with only a console warning to say so.

    ``proj:wkt2`` and ``proj:projjson`` carry the definition in full, and proj4 parses
    either without a lookup. Writing them beside the code is what makes the store
    self-describing to *any* client, and what makes dropping the epsg.io fallback safe for
    an arbitrary CRS rather than only the ones proj4 happens to ship (CLIM-833).

    The same two fields are published on the STAC collection by ``stac/services.py``, for
    clients that read the catalogue; these are for the ones that only ever open the Zarr.

    A built-in code gets the code alone. Nothing has to resolve those — they *are* what a
    client reprojects to — and they cover the large majority of our stores, so spending a
    kilobyte of root metadata each on the redundant case is not worth it. A CRS pyproj cannot interpret gets the
    code alone too, with a warning: an unusable definition helps no one, and the code at
    least records what the store claimed.
    """
    attrs: dict[str, Any] = {"proj:code": canonical_crs_code(code)}
    if is_builtin_crs(code):
        return attrs
    try:
        from pyproj import CRS

        crs = CRS.from_user_input(attrs["proj:code"])
        # WKT2 forced explicitly: pyproj's to_wkt() default varies by version and can emit
        # WKT1 (PROJCS), while both the STAC Projection extension and the GeoZarr ``proj:``
        # convention define this field as WKT2.
        attrs["proj:wkt2"] = crs.to_wkt(version="WKT2_2019")
        attrs["proj:projjson"] = crs.to_json_dict()
    except Exception:
        logger.warning(
            "Could not derive a full CRS definition for %r; writing proj:code alone. A "
            "client that cannot resolve the code will not render this store correctly.",
            attrs["proj:code"],
            exc_info=True,
        )
    return attrs
