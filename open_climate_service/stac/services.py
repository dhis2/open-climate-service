"""STAC catalogue builders backed by published artifact records."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from fastapi import HTTPException, Request
from xstac import xarray_to_stac

from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset, open_zarr_dataset
from open_climate_service.data_manager.services.utils import get_time_dim, get_x_y_dims
from open_climate_service.data_registry.services import datasets as registry_datasets
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord
from open_climate_service.shared.crs import canonical_crs_code, is_builtin_crs
from open_climate_service.shared.time import (
    Cadence,
    parse_period_string_to_datetime,
    period_cadence,
    period_type_to_iso_step,
    resolve_iso_period_step,
)
from open_climate_service.shared.urls import absolute_url, self_url
from open_climate_service.stac.media_types import ZARR_V3_MEDIA_TYPE, zarr_media_type

CATALOG_TITLE = "Open Climate Service"
CATALOG_DESCRIPTION = "Published Open Climate Service GeoZarr datasets"
STAC_VERSION = "1.1.0"
CF_EXTENSION = "https://stac-extensions.github.io/cf/v1.0.0/schema.json"
DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.3.0/schema.json"
PROJECTION_EXTENSION = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
RENDER_EXTENSION = "https://stac-extensions.github.io/render/v2.0.0/schema.json"
ZARR_EXTENSION = "https://stac-extensions.github.io/zarr/v1.1.0/schema.json"
DEFAULT_STAC_LICENSE = "various"
# CF attributes surfaced from the store onto cube:variables, as `cf:`-prefixed STAC fields.
# `cell_methods` is passed through as the CF string ("time: mean"); the CF extension also
# defines a per-dimension array form, but permits the plain string for methods that span axes.
_CF_VARIABLE_ATTRS = ("standard_name", "cell_methods")
# CF section 2.6.2 "Description of file contents". Of the six attributes there, these four are
# the ones CF permits on a variable: "We wish to allow the newly defined attributes, i.e.,
# institution, source, references, and comment, to be either global or assigned to individual
# variables." (`title` and `history` are global-only, so they have no place on cube:variables.)
# Checked against both CF 1.10 and current, which agree.
#
# CF is formally the *netCDF* Climate and Forecast Metadata Conventions, and GeoZarr does not
# require it — "CF in Zarr" is listed there as a convention still under consideration. So this
# is not inherited: it is OCS choosing CF and applying it consistently, which it already does
# via `shared/cf.py` and by emitting `cf:standard_name` / `cf:cell_methods` through the STAC CF
# extension. xarray writes the same attribute names to Zarr as to netCDF, so CF is the de facto
# vocabulary here regardless of which container the bytes land in. Given that choice, 2.6.2 is
# the right authority for which of these are legitimate at variable scope rather than global.
#
# Passed through UNPREFIXED, unlike `_CF_VARIABLE_ATTRS` above. The STAC CF extension v1.0.0
# defines only `cf:standard_name` and `cf:cell_methods` — checked against the published schema
# — so emitting `cf:comment` or `cf:references` would invent fields and then declare
# conformance to an extension that does not define them, and a validating client would reject
# the collection. `attrs` is documented as a passthrough of the store's own CF attribute names,
# which is precisely what these are.
#
# This is an allowlist, not a passthrough of everything: WorldPop rasters arrive carrying
# TIFFTAG_*, STATISTICS_* and AREA_OR_POINT, none of which belongs in a catalogue.
#
# `comment` is the caveat about what the values mean, and the per-variable counterpart to the
# collection description (CLIM-973). `references` is the standard home for attribution, so a
# plugin should use it rather than inventing one. Licensing is deliberately absent: CF has no
# licence attribute, and CLIM-946 is designing proper fields for it.
_CF_CONTENT_ATTRS = ("comment", "references", "institution", "source")
SPATIAL_STEP_DECIMALS = 8
ARTIFACT_CACHE_MAXSIZE = 128
logger = logging.getLogger(__name__)
_xstac_collection_cache: dict[str, dict[str, Any]] = {}
# Media type per artifact id — see _zarr_media_type for why this needs its own cache.
_zarr_media_type_cache: dict[str, str] = {}
_V = TypeVar("_V")


def _get_catalog_id() -> str:
    from open_climate_service import config as api_config

    return api_config.get_id()


def build_catalog(request: Request) -> dict[str, object]:
    """Build the STAC catalog document."""
    self_href = self_url(request)
    catalog_href = absolute_url(request, "/stac/catalog.json")
    links = [
        {"rel": "self", "href": self_href, "type": "application/json"},
        {"rel": "root", "href": catalog_href, "type": "application/json"},
    ]
    for dataset_id, artifact in _eligible_artifacts_by_dataset().items():
        links.append(
            {
                "rel": "child",
                "href": absolute_url(request, f"/stac/collections/{dataset_id}"),
                "title": artifact.dataset_name,
                "type": "application/json",
            }
        )
    return {
        "stac_version": STAC_VERSION,
        "type": "Catalog",
        "id": _get_catalog_id(),
        "title": CATALOG_TITLE,
        "description": CATALOG_DESCRIPTION,
        "links": links,
    }


def build_collection(dataset_id: str, request: Request) -> dict[str, object]:
    """Build one STAC collection document."""
    artifact = _eligible_artifacts_by_dataset().get(dataset_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"STAC collection '{dataset_id}' not found")

    source_dataset = registry_datasets.get_dataset(artifact.dataset_id) or {}
    collection_href = absolute_url(request, f"/stac/collections/{dataset_id}")
    catalog_href = absolute_url(request, "/stac/catalog.json")
    dataset_href = absolute_url(request, f"/datasets/{dataset_id}")
    zarr_href = absolute_url(request, f"/zarr/{dataset_id}")

    template = _build_collection_template(
        dataset_id=dataset_id,
        artifact=artifact,
        collection_href=collection_href,
        catalog_href=catalog_href,
        dataset_href=dataset_href,
        zarr_href=zarr_href,
        source_dataset=source_dataset,
        description=_collection_description(dataset_id, artifact, source_dataset),
    )
    template_links = [_link_to_dict(link) for link in template.links]
    period_type = source_dataset.get("period_type")

    collection_payload = _build_collection_with_xstac(artifact=artifact, template=template, period_type=period_type)
    collection_payload["id"] = dataset_id
    collection_payload["type"] = "Collection"
    collection_payload["stac_version"] = STAC_VERSION
    collection_payload["description"] = template.description
    collection_payload["title"] = template.title
    renders = _build_renders(artifact, source_dataset)
    extensions = {DATACUBE_EXTENSION, ZARR_EXTENSION}
    if renders is not None:
        collection_payload["renders"] = renders
        extensions.add(RENDER_EXTENSION)
    existing_extensions = collection_payload.get("stac_extensions", [])
    if isinstance(existing_extensions, list):
        collection_payload["stac_extensions"] = sorted({*existing_extensions, *extensions})
    else:
        collection_payload["stac_extensions"] = sorted(extensions)
    collection_payload["links"] = template_links
    assets = collection_payload.setdefault("assets", {})
    zarr_from_xstac = assets.get("zarr", {}) if isinstance(assets, dict) else {}
    template_asset = _asset_to_dict(_required_zarr_asset(template))
    xarray_open_kwargs = _zarr_open_kwargs(artifact)
    collection_payload["assets"]["zarr"] = {
        **zarr_from_xstac,
        **_zarr_asset_metadata(artifact),
        "href": template_asset["href"],
        "type": template_asset.get("type"),
        "title": template_asset.get("title"),
        "roles": template_asset.get("roles"),
        "xarray:open_kwargs": xarray_open_kwargs,
    }
    if artifact.format == ArtifactFormat.ICECHUNK:
        collection_payload["assets"]["icechunk"] = {
            "href": absolute_url(request, f"/icechunk/{dataset_id}"),
            "type": "application/octet-stream",
            "title": "Icechunk store (native SDK access)",
            "roles": ["data"],
            "xarray:open_kwargs": {"zarr_format": 3, "consolidated": False},
        }
    collection_payload["license"] = template.license
    _remove_helper_variables(collection_payload)
    _round_spatial_steps(collection_payload)
    # Prefer an explicit extents.temporal.resolution; fall back to the dataset's
    # period_type so openEO save_result outputs (which omit the extents block) still
    # get a temporal step — the map viewer needs it to build the time slider.
    _override_time_step(
        collection_payload,
        resolve_iso_period_step(source_dataset) or period_type_to_iso_step(period_type),
        cadence=period_cadence(period_type),
    )
    # Spatial extent comes from the live store (set in _build_collection_with_xstac),
    # not the artifact coverage — see _wgs84_extent_from_store. Temporal still tracks the
    # artifact's materialized coverage.
    _override_temporal_extent_from_artifact(collection_payload, artifact)
    _sanitize_variable_attrs(collection_payload)
    # Last, because the cf: fields are only known after the variables are built and sanitized.
    _declare_cf_extension_if_used(collection_payload)
    return collection_payload


def _collection_description(dataset_id: str, artifact: ArtifactRecord, source_dataset: dict[str, Any]) -> str:
    """The collection description: the template's own text when it has one (CLIM-973).

    A dataset template can carry a `description`, and until this it had no reader anywhere —
    the collection always published a generated sentence, so a template author wrote a
    description, saw it accepted, and it went nowhere. That matters most for datasets whose
    values mislead without a caveat: Meta RWI ranks micro-regions *within one country*, MODIS
    LST is surface rather than 2 m air temperature, CHIRPS3 monthly is a mean daily rate and
    not a monthly total. Each is a trap that produces plausible-looking output when missed, and
    the published collection is the only place an API consumer would look.

    The generated fallback stays for templates without one, and notably for openEO
    `save_result` outputs, which have no template block at all.
    """
    description = source_dataset.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    logger.debug("Dataset %s declares no description; using the generated one", dataset_id)
    return f"Published GeoZarr dataset for {artifact.dataset_name}"


def _eligible_artifacts_by_dataset() -> dict[str, ArtifactRecord]:
    return ingestion_services.latest_published_zarr_artifacts_by_dataset()


def _build_collection_template(
    *,
    dataset_id: str,
    artifact: ArtifactRecord,
    collection_href: str,
    catalog_href: str,
    dataset_href: str,
    zarr_href: str,
    source_dataset: dict[str, Any],
    description: str,
) -> pystac.Collection:
    spatial = artifact.coverage.spatial_wgs84 or artifact.coverage.spatial
    temporal = artifact.coverage.temporal
    template = pystac.Collection(
        id=dataset_id,
        description=description,
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[spatial.xmin, spatial.ymin, spatial.xmax, spatial.ymax]]),
            temporal=pystac.TemporalExtent(
                [
                    [
                        parse_period_string_to_datetime(temporal.start) if temporal.start else None,
                        parse_period_string_to_datetime(temporal.end) if temporal.end else None,
                    ]
                ]
            ),
        ),
        title=artifact.dataset_name,
        stac_extensions=[
            DATACUBE_EXTENSION,
            PROJECTION_EXTENSION,
            ZARR_EXTENSION,
        ],
        license=DEFAULT_STAC_LICENSE,
    )
    # WGS84 default; the store's native CRS (read from its proj:code attr in
    # _build_collection_with_xstac) overrides this. The instance config CRS is
    # never used — every dataset reports the CRS it was stored in.
    template.extra_fields["keywords"] = _keywords(artifact, source_dataset)
    template.extra_fields["proj:code"] = "EPSG:4326"
    template.clear_links()
    template.add_link(pystac.Link(rel="self", target=collection_href, media_type="application/json"))
    template.add_link(pystac.Link(rel="root", target=catalog_href, media_type="application/json"))
    template.add_link(pystac.Link(rel="parent", target=catalog_href, media_type="application/json"))
    template.add_link(
        pystac.Link(rel="alternate", target=dataset_href, media_type="application/json", title="Dataset detail")
    )
    template.add_asset(
        "zarr",
        pystac.Asset(
            href=zarr_href,
            media_type=_zarr_media_type(artifact),
            title="Zarr store",
            roles=["data"],
        ),
    )
    return template


def _add_crs_render_hints(*, template: pystac.Collection, ds: xr.Dataset, store_crs: str) -> None:
    """Surface CRS render hints on the collection for projected (non-built-in) stores.

    Map clients reproject Zarr on the fly with proj4js, which resolves only the
    built-in ``EPSG:4326`` / ``EPSG:3857`` from their code — any other CRS (e.g.
    seNorge's UTM33, ``EPSG:32633``) needs a full definition, or the client has to
    fetch one at render time (an external epsg.io lookup). Publishing the definition
    in STAC removes that runtime dependency.

    Four fields are emitted:

    * ``proj:wkt2`` — the STAC Projection-extension standard, lossless CRS
      representation. It is the same information GeoZarr already carries in the CF
      ``spatial_ref`` grid-mapping's ``crs_wkt`` attribute.
    * ``proj:projjson`` — the same CRS as PROJJSON. Also a STAC Projection-extension
      standard (and the GeoZarr ``proj:`` convention), but a JSON object a JS/STAC
      client can consume directly instead of parsing WKT2 — the convention-aligned
      sibling of the namespaced proj4 hint below.
    * ``open_climate_service:proj4`` — a proj4 string for direct consumption by
      proj4js. proj4 is intentionally *not* a STAC-standard field (PROJ treats proj4
      strings as lossy), so it lives under our namespace rather than ``proj:``.
      zarr-layer only accepts a built-in ``crs`` code or a ``proj4`` string today; once
      carbonplan/zarr-layer#61 lands (auto-resolving any EPSG code) this proj4 hint
      becomes unnecessary and only ``proj:wkt2`` need remain, for other STAC clients.
    * ``proj:bbox`` — the data extent in the store's native CRS. Without it, a client
      reprojecting the Zarr must fetch the x/y coordinate arrays at render time to
      derive bounds (zarr-layer's "proj4 provided without explicit bounds" warning);
      publishing it lets the client pass explicit bounds and skip that round-trip.
    """
    if is_builtin_crs(store_crs):
        return
    try:
        import warnings

        from pyproj import CRS

        # Prefer the CRS the data was actually written with (the CF grid-mapping WKT)
        # over re-deriving from the code, when the store carries it.
        wkt: str | None = None
        spatial_ref = ds.get("spatial_ref")
        if spatial_ref is not None:
            wkt = spatial_ref.attrs.get("crs_wkt") or spatial_ref.attrs.get("spatial_ref")
        crs = CRS.from_wkt(wkt) if wkt else CRS.from_user_input(store_crs)

        # Force WKT2 explicitly — pyproj's to_wkt() default can vary by version and could
        # emit WKT1 (PROJCS); the STAC Projection extension defines proj:wkt2 as WKT2.
        template.extra_fields["proj:wkt2"] = crs.to_wkt(version="WKT2_2019")
        # PROJJSON alongside it — the same CRS as a JSON object, cheaper for JS/STAC
        # clients than parsing WKT2. Cheap to emit next to the WKT2 above.
        template.extra_fields["proj:projjson"] = crs.to_json_dict()
        with warnings.catch_warnings():
            # to_proj4() warns that proj4 is lossy; that's acceptable for a render hint.
            warnings.simplefilter("ignore")
            proj4 = crs.to_proj4()
        if proj4:
            template.extra_fields["open_climate_service:proj4"] = proj4.strip()
    except Exception:
        logger.warning("Could not derive CRS render hints for '%s'", store_crs, exc_info=True)

    # proj:bbox — the native-CRS extent. Prefer the GeoZarr ``spatial:bbox`` the store
    # root already carries; fall back to the x/y coordinate arrays for stores that lack it.
    try:
        bbox = ds.attrs.get("spatial:bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            x_dim, y_dim = get_x_y_dims(ds)
            xs, ys = ds[x_dim].values, ds[y_dim].values
            # Coordinates are cell centres, so this bbox sits a half-pixel inside the
            # true data edge — acceptable for a render/placement hint.
            bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        template.extra_fields["proj:bbox"] = [float(v) for v in bbox]
    except Exception:
        logger.warning("Could not derive proj:bbox for '%s'", store_crs, exc_info=True)


def _wgs84_extent_from_store(ds: xr.Dataset, store_crs: str, x_dim: str, y_dim: str) -> list[float] | None:
    """Return the store's spatial extent as WGS84 ``[west, south, east, north]``.

    STAC collections must report their spatial extent in WGS84. Deriving it live from the
    published store (the same coordinates ``cube:dimensions`` reads) keeps the two
    consistent and avoids the cached ``coverage`` record, which can go stale or hold a
    native-CRS (metre) bbox for projected stores.
    """
    try:
        xmin, xmax = float(ds[x_dim].min()), float(ds[x_dim].max())
        ymin, ymax = float(ds[y_dim].min()), float(ds[y_dim].max())
        if canonical_crs_code(store_crs) == "EPSG:4326":
            return [xmin, ymin, xmax, ymax]
        from pyproj import Transformer

        transformer = Transformer.from_crs(store_crs, "EPSG:4326", always_xy=True)
        west, south, east, north = transformer.transform_bounds(xmin, ymin, xmax, ymax)
        return [west, south, east, north]
    except Exception:
        logger.warning("Could not derive WGS84 extent from store for '%s'", store_crs, exc_info=True)
        return None


def _build_collection_with_xstac(
    *, artifact: ArtifactRecord, template: pystac.Collection, period_type: Any = None
) -> dict[str, Any]:
    # Keyed on period_type as well as the artifact: correcting a template's cadence
    # changes the payload (an irregular one gains `values`) without producing a new artifact.
    cache_key = f"{artifact.artifact_id}:{period_type}"
    cached_payload = _xstac_collection_cache.get(cache_key)
    if cached_payload is not None:
        return deepcopy(cached_payload)

    try:
        ds = _open_published_store(artifact)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to open published Zarr store for artifact '%s'", artifact.artifact_id)
        raise HTTPException(
            status_code=503,
            detail=f"Published Zarr store for artifact '{artifact.artifact_id}' is temporarily unavailable",
        ) from exc
    try:
        # The store's native CRS (written by the ingest orchestrator) takes precedence
        # over the instance-wide CRS set in _build_collection_template. Plugins that
        # store data in WGS84 (e.g. CHIRPS3) should not inherit the instance UTM CRS.
        store_crs = ds.attrs.get("proj:code")
        if store_crs:
            store_crs = canonical_crs_code(store_crs)
            template.extra_fields["proj:code"] = store_crs
            _add_crs_render_hints(template=template, ds=ds, store_crs=store_crs)
        x_dimension, y_dimension = get_x_y_dims(ds)
        try:
            time_dimension: str | None = get_time_dim(ds)
        except ValueError:
            time_dimension = None  # static dataset with no time dimension
        if time_dimension is not None:
            result = xarray_to_stac(
                ds,
                template,
                temporal_dimension=time_dimension,
                x_dimension=x_dimension,
                y_dimension=y_dimension,
                reference_system=4326,
                # Schema validation can trigger outbound fetches for STAC extension schemas.
                validate=False,
            )
        else:
            # xarray_to_stac raises KeyError when temporal_dimension is None and the
            # dataset has no CF time axis (e.g. after reduce_dimension removes it).
            # Fall back to the template and build cube metadata manually so the
            # map viewer can identify the correct variable and dimensions.
            result = template
        # build_collection replaces links from the template after xstac runs, so
        # clear xstac/pystac-owned links before serialization to avoid root-link
        # resolution attempts during to_dict().
        result.clear_links()
        payload: dict[str, Any] = result.to_dict(include_self_link=False)
        if time_dimension is None:
            # xstac was skipped — inject minimal cube:dimensions and cube:variables
            # so the map viewer can resolve the variable name and spatial axes.
            payload.setdefault("cube:dimensions", _build_static_cube_dimensions(ds, x_dimension, y_dimension))
            payload.setdefault("cube:variables", _build_cube_variables(ds))
        # Declare ordinal (non-spatial, non-temporal) axes — e.g. a day-of-year
        # climatology — that xstac/the static path would otherwise drop, so the map
        # viewer can offer a slider over them.
        ordinal_dims = _build_ordinal_dimensions(ds, x_dimension, y_dimension, time_dimension)
        if ordinal_dims:
            payload.setdefault("cube:dimensions", {}).update(ordinal_dims)
        # WGS84 spatial extent from the live store — consistent with cube:dimensions and
        # immune to a stale/native-CRS cached coverage record (see _wgs84_extent_from_store).
        extent_bbox = _wgs84_extent_from_store(ds, store_crs or "EPSG:4326", x_dimension, y_dimension)
        if extent_bbox is not None:
            payload.setdefault("extent", {}).setdefault("spatial", {})["bbox"] = [extent_bbox]
        if time_dimension is not None and period_cadence(period_type) is Cadence.IRREGULAR:
            _add_temporal_values(payload, ds, time_dimension)
        _cache_xstac_collection_payload(cache_key, payload)
        return deepcopy(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to derive STAC metadata from artifact '%s'", artifact.artifact_id)
        raise HTTPException(
            status_code=503,
            detail=f"Published Zarr store for artifact '{artifact.artifact_id}' is temporarily unavailable",
        ) from exc
    finally:
        ds.close()


def _zarr_media_type(artifact: ArtifactRecord) -> str:
    """Media type for the artifact's Zarr asset, advertising a pyramid only when there is one.

    Lets pyramid-aware clients (STAC Browser via ``ol/source/GeoZarr``) tell a multiscales store
    from a flat one, which the plain Zarr media type cannot express. Falls back to the flat type
    whenever the store can't be inspected.

    Cached because detection reads the store's root group, and this is called from the collection
    *template*, which is rebuilt on every request — so it sits outside the xstac payload cache.
    See :func:`_cache_by_artifact` for why the key is the artifact id.
    """
    cached = _zarr_media_type_cache.get(artifact.artifact_id)
    if cached is not None:
        return cached
    try:
        store_path = _artifact_store_path(artifact)
    except HTTPException:
        return ZARR_V3_MEDIA_TYPE
    media_type = zarr_media_type(store_path, icechunk=artifact.format == ArtifactFormat.ICECHUNK)
    _cache_by_artifact(_zarr_media_type_cache, artifact.artifact_id, media_type)
    return media_type


def _cache_by_artifact(cache: dict[str, _V], artifact_id: str, value: _V) -> None:
    """Store *value* under *artifact_id*, evicting the oldest entry past the size bound.

    Both per-artifact caches here are keyed on the artifact id rather than the store path: a
    re-ingest reuses the path, so a path-keyed entry would go stale, while every write produces a
    new artifact record. Refreshing an existing key keeps its insertion position, so a
    frequently-rebuilt collection is not treated as newest and does not evict others.
    """
    if artifact_id in cache:
        cache[artifact_id] = value
        return
    if len(cache) >= ARTIFACT_CACHE_MAXSIZE:
        cache.pop(next(iter(cache)), None)
    cache[artifact_id] = value


def _cache_xstac_collection_payload(artifact_id: str, payload: dict[str, Any]) -> None:
    _cache_by_artifact(_xstac_collection_cache, artifact_id, deepcopy(payload))


def _clear_xstac_collection_cache() -> None:
    _xstac_collection_cache.clear()
    _zarr_media_type_cache.clear()


def _link_to_dict(link: pystac.Link) -> dict[str, Any]:
    target = link.target
    href = target if isinstance(target, str) else link.href
    payload = {"rel": link.rel, "href": str(href)}
    if link.media_type is not None:
        payload["type"] = link.media_type
    if link.title is not None:
        payload["title"] = link.title
    return payload


def _asset_to_dict(asset: pystac.Asset) -> dict[str, Any]:
    payload: dict[str, Any] = {"href": asset.href}
    if asset.media_type is not None:
        payload["type"] = asset.media_type
    if asset.title is not None:
        payload["title"] = asset.title
    if asset.roles is not None:
        payload["roles"] = asset.roles
    payload.update(asset.extra_fields)
    return payload


def _required_zarr_asset(template: pystac.Collection) -> pystac.Asset:
    asset = template.assets.get("zarr")
    if asset is None:
        raise HTTPException(status_code=500, detail="STAC template is missing the required zarr asset")
    return asset


def _artifact_store_path(artifact: ArtifactRecord) -> str:
    if artifact.path:
        return artifact.path
    if artifact.asset_paths:
        return artifact.asset_paths[0]
    raise HTTPException(
        status_code=500,
        detail=f"Published artifact '{artifact.artifact_id}' has no readable storage path metadata",
    )


def _override_time_step(collection: dict[str, Any], step: str | None, *, cadence: Cadence) -> None:
    """Set the temporal dimension's ``step`` to the duration, or to an explicit null.

    A regular cadence gets its ISO 8601 duration. An **irregular** one gets ``null``,
    which the datacube extension defines as "irregularly spaced steps" — the only
    truthful answer for a cadence whose periods differ in length, and better than
    leaving whatever xstac inferred, which would imply a regular spacing the data does
    not have.

    A null step alone is not enough for a client, though: without a duration there is
    nothing to extrapolate from, so the accompanying ``values`` must carry the real
    timestamps. See ``_add_temporal_values``, which supplies them.
    """
    if cadence is Cadence.IRREGULAR:
        # No duration can describe a variable-length cadence, so a declared
        # extents.temporal.resolution must not win here: a dekadal template that happens to
        # declare P10D would otherwise publish it and look regular.
        step = None
    elif step is None:
        return
    dimensions = collection.setdefault("cube:dimensions", {})
    for key, value in dimensions.items():
        if isinstance(value, dict) and value.get("type") == "temporal":
            value["step"] = step
            dimensions[key] = value
            return


def _add_temporal_values(collection: dict[str, Any], ds: xr.Dataset, time_dimension: str) -> None:
    """List the temporal dimension's actual timestamps as ``values``.

    Required for an irregular cadence, not decorative. A client builds its time control
    either from explicit ``values`` or by stepping ``extent`` by ``step``; with an
    irregular cadence there is no step to walk, so omitting ``values`` leaves a consumer
    with a single position and no slider. This mirrors ``_build_ordinal_dimensions``,
    which lists values for the same reason on a day-of-year axis.

    Only emitted for irregular cadences. A regular one is fully described by extent plus
    duration, and listing every timestamp would add one ISO string per period to a cached
    response for no gain — thousands of entries for a multi-year daily store.
    """
    dimensions = collection.get("cube:dimensions") or {}
    dim = dimensions.get(time_dimension)
    if not isinstance(dim, dict) or dim.get("type") != "temporal":
        return
    stamps = pd.DatetimeIndex(np.asarray(ds[time_dimension].values, dtype="datetime64[ns]"))
    dim["values"] = [f"{s.isoformat()}Z" for s in stamps.tz_localize(None)]


def _build_static_cube_dimensions(ds: xr.Dataset, x_dim: str, y_dim: str) -> dict[str, Any]:
    """Build minimal cube:dimensions for a dataset with no time axis."""
    crs = ds.attrs.get("proj:code", "EPSG:4326")
    x_vals = ds[x_dim].values
    y_vals = ds[y_dim].values
    x_step = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else None
    y_step = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else None
    return {
        x_dim: {
            "type": "spatial",
            "axis": "x",
            "extent": [float(x_vals.min()), float(x_vals.max())],
            **({"step": x_step} if x_step is not None else {}),
            "reference_system": crs,
        },
        y_dim: {
            "type": "spatial",
            "axis": "y",
            "extent": [float(y_vals.min()), float(y_vals.max())],
            **({"step": y_step} if y_step is not None else {}),
            "reference_system": crs,
        },
    }


def _build_ordinal_dimensions(ds: xr.Dataset, x_dim: str, y_dim: str, time_dim: str | None) -> dict[str, Any]:
    """cube:dimensions entries for non-spatial, non-temporal axes (e.g. ``dayofyear``).

    The datacube/STAC machinery only declares spatial and temporal axes, so an ordinal
    coordinate like a day-of-year climatology axis would otherwise be dropped — and the
    map viewer could not slider it. Declared here with explicit ``values`` (and a ``step``
    for evenly spaced integer axes) so the viewer can step over them.
    """
    import numpy as np

    dims = getattr(ds, "dims", None)
    if not dims:
        return {}
    skip = {x_dim, y_dim} | ({time_dim} if time_dim else set())
    out: dict[str, Any] = {}
    for name in dims:
        if name in skip or name not in ds.coords:
            continue
        coord = ds[name]
        if np.issubdtype(coord.dtype, np.datetime64):
            continue  # a temporal axis — handled by xstac
        values = coord.values.tolist()
        entry: dict[str, Any] = {"type": "other", "values": values}
        if len(values) > 1 and all(isinstance(v, int) for v in values):
            # Only publish a step for a genuinely evenly spaced axis: every
            # consecutive delta must match. Deriving it from the first delta
            # alone would emit a wrong step for irregular integer coordinates.
            deltas = {b - a for a, b in zip(values, values[1:])}
            if len(deltas) == 1:
                (step,) = deltas
                if step:
                    entry["step"] = step
        out[str(name)] = entry
    return out


def _build_cube_variables(ds: xr.Dataset) -> dict[str, Any]:
    """Build cube:variables from an xr.Dataset's data variables."""
    result: dict[str, Any] = {}
    for name in ds.data_vars:
        var = ds[name]
        entry: dict[str, Any] = {
            "type": "data",
            "dimensions": [str(d) for d in var.dims],
            "unit": var.attrs.get("units"),
        }
        # See `_CF_CONTENT_ATTRS`. Both builders must agree or these appear on one path only.
        for key in _CF_CONTENT_ATTRS:
            value = var.attrs.get(key)
            if isinstance(value, str):
                entry.setdefault("attrs", {})[key] = value
        # Surface the CF semantics stamped onto the store so catalog clients can identify the
        # quantity, not just its unit (CLIM-828). Named per the STAC CF extension, which lists
        # `cube:variables` among the places its fields may be used — an unprefixed
        # `standard_name` here is not a field any STAC client is defined to understand.
        for cf_key in _CF_VARIABLE_ATTRS:
            value = var.attrs.get(cf_key)
            if isinstance(value, str):
                entry[f"cf:{cf_key}"] = value
        result[str(name)] = entry
    return result


def _round_spatial_steps(collection: dict[str, Any]) -> None:
    dimensions = collection.get("cube:dimensions")
    if not isinstance(dimensions, dict):
        return
    for key, value in dimensions.items():
        if not isinstance(value, dict) or value.get("type") != "spatial":
            continue
        step = value.get("step")
        if isinstance(step, int | float):
            value["step"] = round(float(step), SPATIAL_STEP_DECIMALS)
            dimensions[key] = value


def _override_temporal_extent_from_artifact(collection: dict[str, Any], artifact: ArtifactRecord) -> None:
    temporal = artifact.coverage.temporal

    def _fmt(v: str | None) -> str | None:
        return parse_period_string_to_datetime(v).isoformat().replace("+00:00", "Z") if v else None

    start = _fmt(temporal.start)
    end = _fmt(temporal.end)
    collection["extent"]["temporal"]["interval"] = [[start, end]]
    dimensions = collection.setdefault("cube:dimensions", {})
    for key, value in dimensions.items():
        if isinstance(value, dict) and value.get("type") == "temporal":
            value["extent"] = [start, end]
            dimensions[key] = value
            return


def _declare_cf_extension_if_used(collection: dict[str, Any]) -> None:
    """Add the CF extension to ``stac_extensions`` when a ``cf:`` field was actually emitted.

    Declared conditionally rather than always: a dataset whose store carries no CF attributes
    emits no ``cf:`` field, and advertising an extension none of whose fields are present is
    noise a client has to fetch and check for nothing.
    """
    variables = collection.get("cube:variables")
    if not isinstance(variables, dict):
        return
    used = any(
        isinstance(variable, dict) and any(key.startswith("cf:") for key in variable) for variable in variables.values()
    )
    if not used:
        return
    existing = collection.get("stac_extensions")
    extensions = set(existing) if isinstance(existing, list) else set()
    collection["stac_extensions"] = sorted({*extensions, CF_EXTENSION})


def _sanitize_variable_attrs(collection: dict[str, Any]) -> None:
    variables = collection.get("cube:variables")
    if not isinstance(variables, dict):
        return
    for _, variable in variables.items():
        if not isinstance(variable, dict):
            continue
        attrs = variable.get("attrs")
        if not isinstance(attrs, dict):
            continue
        kept_attrs: dict[str, str] = {}
        long_name = attrs.get("long_name")
        units = attrs.get("units")
        if isinstance(long_name, str):
            kept_attrs["long_name"] = long_name
        if isinstance(units, str):
            kept_attrs["units"] = units
            variable["unit"] = units
        for key in _CF_CONTENT_ATTRS:
            value = attrs.get(key)
            if isinstance(value, str):
                kept_attrs[key] = value
        # Same CF semantics as _build_cube_variables, for the xstac-produced path. Prefixed at
        # the cube:variable level (a defined STAC CF extension field); unprefixed inside
        # `attrs`, which is a passthrough of the store's own CF attribute names.
        for cf_key in _CF_VARIABLE_ATTRS:
            value = attrs.get(cf_key)
            if isinstance(value, str):
                kept_attrs[cf_key] = value
                variable[f"cf:{cf_key}"] = value
        variable["attrs"] = kept_attrs


def _remove_helper_variables(collection: dict[str, Any]) -> None:
    variables = collection.get("cube:variables")
    if not isinstance(variables, dict):
        return
    for key in list(variables):
        variable = variables.get(key)
        if not isinstance(variable, dict):
            continue
        dimensions = variable.get("dimensions")
        # xstac can emit scalar CRS/grid-mapping helper variables with no dimensions.
        if isinstance(dimensions, list) and len(dimensions) == 0:
            variables.pop(key, None)


def _keywords(artifact: ArtifactRecord, source_dataset: dict[str, Any]) -> list[str]:
    keywords = [artifact.dataset_id, artifact.variable, "zarr", "stac"]
    for key in ("source", "short_name"):
        value = source_dataset.get(key)
        if isinstance(value, str) and value:
            keywords.append(value)
    return keywords


def _zarr_asset_metadata(artifact: ArtifactRecord) -> dict[str, object]:
    metadata: dict[str, object] = {"zarr:node_type": "group"}
    if artifact.format == ArtifactFormat.ICECHUNK:
        metadata["zarr:zarr_format"] = 3
        return metadata
    artifact_path = _artifact_store_path(artifact)
    consolidated = _zarr_consolidated_flag(artifact_path)
    if consolidated is not None:
        metadata["zarr:consolidated"] = consolidated
    if "://" in artifact_path:
        return metadata
    store_root = Path(artifact_path)
    zarr_json = store_root / "zarr.json"
    if zarr_json.exists():
        metadata["zarr:zarr_format"] = 3
    else:
        zgroup = store_root / ".zgroup"
        if zgroup.exists():
            metadata["zarr:zarr_format"] = 2
    return metadata


def _zarr_open_kwargs(artifact: ArtifactRecord) -> dict[str, bool | None]:
    if artifact.format == ArtifactFormat.ICECHUNK:
        return {"consolidated": True}
    return {"consolidated": _zarr_consolidated_flag(_artifact_store_path(artifact))}


def _open_published_store(artifact: ArtifactRecord) -> xr.Dataset:
    if artifact.format == ArtifactFormat.ICECHUNK:
        return open_icechunk_dataset(_artifact_store_path(artifact))
    return open_zarr_dataset(_artifact_store_path(artifact))


def _build_renders(artifact: ArtifactRecord, source_dataset: dict[str, Any]) -> dict[str, Any] | None:
    display = source_dataset.get("display")
    if not isinstance(display, dict):
        return None
    colormap_name = display.get("colormap")
    value_range = display.get("range")
    if not isinstance(colormap_name, str) or not isinstance(value_range, list) or len(value_range) != 2:
        return None
    render: dict[str, Any] = {
        "title": artifact.dataset_name,
        "assets": ["zarr"],
        "rescale": [[float(value_range[0]), float(value_range[1])]],
        "colormap_name": colormap_name,
        "open_climate_service:variable": artifact.variable,
    }
    nodata = display.get("nodata")
    if nodata is not None:
        render["nodata"] = float(nodata)
    # Units aren't duplicated here: they're published on the datacube-standard
    # ``cube:variables[<var>].unit``, which clients read instead.
    return {"default": render}


def _zarr_consolidated_flag(artifact_path: str) -> bool | None:
    if "://" in artifact_path:
        return None

    store_root = Path(artifact_path)
    zarr_json = store_root / "zarr.json"
    if zarr_json.exists():
        try:
            payload = json.loads(zarr_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return "consolidated_metadata" in payload

    if (store_root / ".zmetadata").exists():
        return True
    if (store_root / ".zgroup").exists():
        return False
    return None
