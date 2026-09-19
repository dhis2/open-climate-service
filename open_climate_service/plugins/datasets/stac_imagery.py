"""Ingest a public STAC imagery release as a true-colour GeoZarr cube.

A generic replacement for per-release plugins: a STAC catalogue of georeferenced image assets
becomes a dataset *template* rather than code (CLIM-957). Verified against three structurally
different sources, which between them exercise every branch here:

- Planet crisis-response (Source Cooperative) — a static nested tree, root -> pre/post-event
  -> one collection per sensor and date -> items, with a 4-band `visual` carrying alpha.
- Vantor open data (S3) — a static flat collection of items, with a 3-band `visual` that
  declares no mask of any kind, so validity falls back to an all-zero heuristic.
- Sentinel-2 L2A (Earth Search) — a STAC *API*, queried rather than walked, whose TCI asset
  declares `nodata=0`.

Those three cover the three mask strategies, both catalogue shapes, and both a projected and
a geographic source CRS. That is the point of the module: adding a fourth release should be a
template, not a code change.

Licensing differs and matters. Planet and Vantor are CC-BY-NC-4.0; Sentinel-2 is openly
licensed. OCS has no licence field on a dataset template (CLIM-946), so a `source_license`
param is written as an attribute on the stored variable and the published STAC collection
reports `various` for all of them. That attribute is not a STAC licence and no client looks
for it — carry the restriction in the template's `source` string too.

What is stored is a colour composite with dims ``(t, band, y, x)`` and a string ``band``
coordinate, the layout OCS renders when a template declares ``display.bands`` (CLIM-947).
These are display values: 8-bit and colour-corrected per scene, so differencing two dates
partly measures the provider's balancing rather than the ground. The variable is named by
the template; prefer something like `true_colour` over `reflectance`, because the variable
name is most of what a downstream consumer sees.
"""

from __future__ import annotations

import asyncio
import logging
import math
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, cast

import httpx
import numpy as np
import rioxarray  # noqa: F401  # registers the .rio accessor
import xarray as xr
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.warp import transform_bounds
from rioxarray.exceptions import NoDataInBounds

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

# Upper bound on the target grid, in cells. Sized so a legitimate event-scoped extent never
# trips it but an unscoped one fails loudly: `fetch_period` is handed the *instance* extent,
# which for a country at sub-metre resolution is a terabyte-scale allocation. See `clip_bbox`.
_MAX_GRID_CELLS = 60_000_000

# COG-over-HTTP settings, measured against the Planet release: opening one scene took 6.4 s
# with GDAL's defaults and 1.0 s with these. The cost was GDAL probing for sidecar files
# (.aux.xml, .ovr, ...) on every open, each probe a round trip to a slow host.
_GDAL_ENV: dict[str, Any] = {
    # rasterio.Env is strict about types: booleans and integers must be passed as such, not
    # as the strings GDAL's own documentation uses.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": True,
    "VSI_CACHE": True,
    "GDAL_CACHEMAX": 512,
}

_DEFAULT_BANDS = ("red", "green", "blue")

# STAC API item search. The page size is the spec default most servers cap at anyway; the
# page cap is a runaway guard — a clip and a date range should return far less than this, and
# hitting it means the request was wider than intended.
_SEARCH_PAGE_SIZE = 100
_SEARCH_MAX_PAGES = 50

# How a scene declares which pixels carry data, in the order they are preferred. Chosen by
# probing the two known releases rather than assumed: Planet's `visual` declares colorinterp
# ['red','green','blue','alpha'], while Vantor's declares ['red','green','blue'] with nodata
# None and every band `all_valid` — no mask information whatsoever.
_MASK_ALPHA = "alpha"
_MASK_NODATA = "nodata"
_MASK_NONZERO = "nonzero"


def _get_json(url: str) -> dict[str, Any]:
    """Fetch one STAC document.

    httpx rather than urllib: Source Cooperative 403s the default Python user agent, the same
    reason `worldpop.py` reaches for httpx.
    """
    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    return dict(response.json())


def _links(doc: dict[str, Any], rel: str, base: str) -> list[str]:
    return [urllib.parse.urljoin(base, link["href"]) for link in doc.get("links", []) if link.get("rel") == rel]


def _open_scene(href: str, overview_level: int | None) -> xr.DataArray:
    """Open one scene lazily, as a DataArray.

    ``open_rasterio`` is typed as returning a Dataset, a DataArray or a list of Datasets; a
    single-subdataset raster always yields a DataArray, and anything else here means the
    source is not the plain image asset the plugin assumes.
    """
    opened = rioxarray.open_rasterio(
        href,
        chunks={"x": 2048, "y": 2048},
        overview_level=overview_level,
    )
    if not isinstance(opened, xr.DataArray):
        raise TypeError(f"Expected a DataArray from {href}, got {type(opened).__name__}")
    return opened


class _Scene(NamedTuple):
    href: str
    bbox: tuple[float, float, float, float]  # EPSG:4326, straight from the STAC item
    item_id: str


class _SceneMeta(NamedTuple):
    band_indices: tuple[int, ...]  # 1-based, in requested band order
    mask_kind: str
    mask_ref: Any  # alpha band index, nodata value, or None
    overview_level: int | None


class _Grid(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    bounds: tuple[float, float, float, float]  # left, bottom, right, top in the target CRS


def _utm_epsg(lon: float, lat: float) -> str:
    """EPSG code of the UTM zone containing a point.

    Used when no `target_crs` is given, so `resolution_m` is always metres regardless of
    whether the source is projected (Planet, UTM 45N) or geographic (Vantor, EPSG:4326).
    """
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _resolve_bands(
    colorinterp: tuple[Any, ...],
    descriptions: tuple[Any, ...],
    bands: tuple[str, ...],
) -> tuple[int, ...]:
    """Map requested band names to 1-based raster indices.

    Three sources, in order: GDAL's colour interpretation (Planet and Vantor both declare it),
    then band descriptions (Vantor has 'red'/'green'/'blue', Planet has None), then position.
    Positional is the last resort and only when the count matches, so a source that changes
    its band layout fails loudly rather than silently swapping channels.
    """
    by_interp = {c.name.lower(): i + 1 for i, c in enumerate(colorinterp) if c is not None}
    if all(name in by_interp for name in bands):
        return tuple(by_interp[name] for name in bands)

    by_desc = {str(d).lower(): i + 1 for i, d in enumerate(descriptions) if d}
    if all(name in by_desc for name in bands):
        return tuple(by_desc[name] for name in bands)

    if len(colorinterp) == len(bands):
        logger.warning(
            "Falling back to positional band order for %s: colorinterp=%s descriptions=%s",
            bands,
            [c.name if c else None for c in colorinterp],
            descriptions,
        )
        return tuple(range(1, len(bands) + 1))

    raise ValueError(
        f"Cannot resolve bands {bands} in a {len(colorinterp)}-band asset: "
        f"colorinterp={[c.name if c else None for c in colorinterp]}, "
        f"descriptions={descriptions}. Set `bands` in the template to names the asset "
        f"declares, or supply an asset whose band count matches."
    )


def _resolve_mask(src: Any) -> tuple[str, Any]:
    """Pick how to tell data pixels from padding, preferring the most explicit signal.

    This is the one place the known sources genuinely differ, and getting it wrong produces a
    subtly wrong mosaic rather than an error: without a mask the black padding around a
    scene's footprint wins the first-non-null test and punches holes in the result.
    """
    for index, interp in enumerate(src.colorinterp):
        if interp == ColorInterp.alpha:
            return _MASK_ALPHA, index + 1
    if src.nodata is not None:
        return _MASK_NODATA, src.nodata
    if any(MaskFlags.all_valid not in flags for flags in src.mask_flag_enums):
        # A per-dataset mask band with no alpha and no nodata. Not read directly — GDAL does
        # not expose it through rioxarray — so it is approximated by the all-zero test below.
        # Warned about because that approximation is the plugin's, not the source's.
        logger.warning(
            "%s declares an internal mask that cannot be read through rioxarray; "
            "approximating it by treating all-zero pixels as padding",
            getattr(src, "name", "scene"),
        )
    # No usable mask information (Vantor). Treating all-zero pixels as padding is a heuristic:
    # a genuinely black pixel is dropped. In practice colour-corrected visual imagery over
    # land does not produce true 0,0,0, and the alternative — trusting the padding — is worse.
    return _MASK_NONZERO, None


class StacImageryPlugin(BaseDatasetPlugin):
    """True-colour imagery from any static STAC catalogue, mosaicked onto a fixed grid.

    Template params:

    - `catalog_url` — catalogue or collection root; `child` and `item` links are walked
      recursively, so a nested tree and a flat collection both work
    - `asset` — asset key to read (default `visual`)
    - `bands` — output band names (default red, green, blue)
    - `resolution_m` — target grid resolution in metres
    - `clip_bbox` — the ingest footprint; see the note in `__init__`
    - `target_crs` — default: the UTM zone containing the clip centre, so the grid is metric
    - `variable` — stored variable name
    - `collection_filter` — substring a leaf collection's URL must contain to be walked
      (static catalogues only)
    - `collections` — STAC API collection ids to search (item-search APIs only)
    - `dates` — keep only these acquisition dates, for a non-contiguous selection
    - `max_cloud_cover` — drop items above this `eo:cloud_cover`
    - `property_filters` — mapping of STAC item property to required value(s)
    - `source_license`, `long_name` — written onto the stored variable
    """

    max_concurrency = 1
    commit_batch_size = 2

    def __init__(
        self,
        catalog_url: str,
        resolution_m: float,
        variable: str = "true_colour",
        asset: str = "visual",
        bands: list[str] | None = None,
        clip_bbox: list[float] | None = None,
        target_crs: str | None = None,
        collection_filter: str | None = None,
        collections: list[str] | None = None,
        dates: list[str] | None = None,
        max_cloud_cover: float | None = None,
        property_filters: dict[str, Any] | None = None,
        source_license: str | None = None,
        long_name: str | None = None,
        **_: Any,
    ) -> None:
        self.catalog_url = str(catalog_url)
        # Validated because this reaches both a division and the step of an `np.arange`. Zero
        # raises there with a message about arange rather than about configuration; a negative
        # or NaN value silently yields an empty grid and fails much later inside a raster call.
        self.resolution_m = float(resolution_m)
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be a finite positive number, got {resolution_m!r}")
        self.variable = str(variable)
        self.asset = str(asset)
        # `is None` rather than falsy: an explicitly empty list is a configuration mistake and
        # must reach the guard below, not silently become the RGB default.
        self.bands = tuple(str(b).lower() for b in (_DEFAULT_BANDS if bands is None else bands))
        if not self.bands:
            raise ValueError("bands must not be empty")
        # OCS hands `fetch_period` the *instance* extent, not the template's
        # `extents.spatial.bbox` — that field documents where the source has data and is never
        # read by the ingest (only `extents.temporal` is). For a country-sized instance at
        # sub-metre resolution that is a multi-hundred-gigabyte allocation: the process is
        # SIGKILLed by the kernel with no Python traceback. So an event-scoped dataset has to
        # carry its own clip. Deliberately not named `bbox`: the orchestrator calls
        # `fetch_period(period_id, bbox, **params)`, so a `bbox` key here would collide with
        # the positional argument and raise TypeError.
        self.clip_bbox = [float(v) for v in clip_bbox] if clip_bbox else None
        self.target_crs = str(target_crs) if target_crs else None
        self.collection_filter = collection_filter
        # STAC API collection ids to search. Distinct from `collection_filter`, which is a
        # substring match on a *static* catalogue's child URLs.
        self.collections = [str(c) for c in collections] if collections else []
        # Explicit acquisition dates. A range cannot express "these two and not the five
        # between", which is the common case for a before/after pair drawn from a repeat-pass
        # source. Empty means "every date in the requested range".
        self.dates = sorted({str(d)[:10] for d in dates}) if dates else []
        self.max_cloud_cover = float(max_cloud_cover) if max_cloud_cover is not None else None
        self.property_filters = dict(property_filters or {})
        self.source_license = source_license
        self.long_name = long_name
        self._index: dict[str, list[_Scene]] | None = None
        # "" means "checked, not an item-search API"; None means "not yet checked".
        self._api_search_url: str | None = None
        self._meta_cache: dict[tuple[str, str, float], _SceneMeta] = {}

    # -- discovery ---------------------------------------------------------------------

    def _keep_item(self, item: dict[str, Any]) -> bool:
        properties = item.get("properties", {})
        if self.max_cloud_cover is not None:
            cloud = properties.get("eo:cloud_cover")
            if cloud is not None and float(cloud) > self.max_cloud_cover:
                return False
        for key, wanted in self.property_filters.items():
            actual = properties.get(key)
            allowed = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            if actual not in allowed:
                return False
        return True

    def _add_item(self, by_date: dict[str, list[_Scene]], item: dict[str, Any], base_url: str) -> bool:
        """Turn one STAC item into a scene, or explain why it was not usable."""
        assets = item.get("assets", {})
        if self.asset not in assets:
            logger.warning(
                "Item %s has no %r asset; found %s",
                item.get("id", base_url.rsplit("/", 1)[-1]),
                self.asset,
                sorted(assets),
            )
            return False
        date = str(item["properties"]["datetime"])[:10]
        # The item's own bbox lets a non-overlapping scene be skipped without opening the file
        # at all — the difference between 9 opens and 5.
        by_date.setdefault(date, []).append(
            _Scene(
                urllib.parse.urljoin(base_url, assets[self.asset]["href"]),
                tuple(item["bbox"][:4]),
                str(item.get("id", base_url.rsplit("/", 1)[-1])),
            )
        )
        return True

    def _is_item_search_api(self) -> bool:
        """Whether `catalog_url` is a STAC API root that supports item search.

        Both signals are required: a `search` link alone appears on some static catalogues that
        merely link to a search UI, and the conformance class alone does not give the endpoint.
        """
        if self._api_search_url is not None:
            return bool(self._api_search_url)
        root = _get_json(self.catalog_url)
        conforms = root.get("conformsTo", [])
        supports = any("item-search" in str(c) for c in conforms)
        search = _links(root, "search", self.catalog_url)
        self._api_search_url = search[0] if (supports and search) else ""
        if self._api_search_url:
            logger.info("%s is a STAC API; using item search", self.catalog_url)
        return bool(self._api_search_url)

    def _discover_via_search(self, start: str | None, end: str | None) -> dict[str, list[_Scene]]:
        """Query a STAC API instead of walking it.

        A global archive cannot be walked — Earth Search's `sentinel-2-l2a` alone holds tens of
        millions of items — so the clip and the requested date range become search parameters.
        That inverts the cost: the static path fetches every item and discards most, while this
        fetches only what the extent and range already imply.
        """
        by_date: dict[str, list[_Scene]] = {}
        skipped = 0
        body: dict[str, Any] = {"limit": _SEARCH_PAGE_SIZE}
        if self.collections:
            body["collections"] = list(self.collections)
        if self.clip_bbox:
            body["bbox"] = list(self.clip_bbox)
        if start and end:
            body["datetime"] = f"{start[:10]}T00:00:00Z/{end[:10]}T23:59:59Z"

        url: str | None = self._api_search_url or None
        pages = 0
        total = 0
        while url:
            response = httpx.post(url, json=body, timeout=120, follow_redirects=True)
            response.raise_for_status()
            page = response.json()
            for item in page.get("features", []):
                total += 1
                if not self._keep_item(item):
                    skipped += 1
                    continue
                # Asset hrefs in a search response are normally absolute, but resolve them
                # against the item's own `self` link when present rather than against whichever
                # link happens to be first, which need not be a URL base at all.
                self_links = [
                    link.get("href") for link in item.get("links", []) if link.get("rel") == "self" and link.get("href")
                ]
                self._add_item(by_date, item, self_links[0] if self_links else url)
            pages += 1
            if pages >= _SEARCH_MAX_PAGES:
                # A bounded sweep beats an unbounded one, but a silent cap would read as
                # "that is everything". Say what was left.
                logger.warning(
                    "Stopped after %d search pages (%d items); narrow clip_bbox or the date range if dates are missing",
                    pages,
                    total,
                )
                break
            nxt = [link for link in page.get("links", []) if link.get("rel") == "next"]
            # Paging carries the cursor in the body for POST search, in the href for GET.
            if nxt and nxt[0].get("body"):
                body = {**body, **nxt[0]["body"]}
                url = nxt[0].get("href", url)
            else:
                url = nxt[0]["href"] if nxt else None
                body = {}
        if skipped:
            logger.info("%d item(s) dropped by cloud/property filters", skipped)
        if not by_date:
            raise ValueError(
                f"STAC API search over {self.catalog_url} returned no items with a "
                f"{self.asset!r} asset for bbox={self.clip_bbox} range={start}..{end}"
            )
        return by_date

    def _discover(self, start: str | None = None, end: str | None = None) -> dict[str, list[_Scene]]:
        """Walk the catalogue once, returning ``{acquisition date: [scenes]}``.

        Handles both known shapes: a document may carry `child` links (a catalogue), `item`
        links (a collection), or both. A flat collection is simply the case where the root has
        no children. A STAC API root is queried rather than walked.
        """
        if self._is_item_search_api():
            return self._discover_via_search(start, end)

        by_date: dict[str, list[_Scene]] = {}
        seen: set[str] = set()
        skipped = 0

        def walk(url: str) -> None:
            nonlocal skipped
            if url in seen:
                return
            seen.add(url)
            doc = _get_json(url)

            item_urls = _links(doc, "item", url)
            if item_urls:
                # Concurrently: these hosts answer in roughly a second per request, so
                # fetching a collection's items in series dominated start-up.
                with ThreadPoolExecutor(max_workers=8) as pool:
                    items = list(pool.map(_get_json, item_urls))
                found = 0
                filtered = 0
                for item_url, item in zip(item_urls, items, strict=True):
                    if not self._keep_item(item):
                        filtered += 1
                        continue
                    if self._add_item(by_date, item, item_url):
                        found += 1
                skipped += filtered
                if not found and not filtered:
                    # Loudly, not silently: a collection contributing nothing means a date
                    # vanishes from `periods()`, and a missing period is invisible once
                    # ingested. Filters legitimately empty a collection, hence the guard.
                    raise ValueError(f"STAC collection {url} yielded no usable {self.asset!r} assets")

            for child_url in _links(doc, "child", url):
                if self.collection_filter and self.collection_filter not in child_url:
                    # Applied only at the leaf: an intermediate node (Planet's pre/post-event
                    # split) does not carry the sensor name, so filtering it out here would
                    # prune the whole subtree. Children that have children are still followed.
                    child = _get_json(child_url)
                    if not _links(child, "child", child_url):
                        continue
                walk(child_url)

        walk(self.catalog_url)
        if skipped:
            logger.info("%d item(s) dropped by cloud/property filters", skipped)
        if not by_date:
            raise ValueError(f"No items with a {self.asset!r} asset found under {self.catalog_url}")
        return by_date

    def _scenes(self, start: str | None = None, end: str | None = None) -> dict[str, list[_Scene]]:
        if self._index is None:
            self._index = self._discover(start, end)
            logger.info(
                "Discovered %d date(s): %s",
                len(self._index),
                ", ".join(f"{d} ({len(v)})" for d, v in sorted(self._index.items())),
            )
        return self._index

    @staticmethod
    def _overlapping(scenes: list[_Scene], box: tuple[float, float, float, float]) -> list[_Scene]:
        w, s, e, n = box
        return [sc for sc in scenes if not (sc.bbox[2] <= w or sc.bbox[0] >= e or sc.bbox[3] <= s or sc.bbox[1] >= n)]

    async def periods(self, start: str, end: str) -> list[str]:
        # The range reaches discovery so a STAC API search is bounded by it; the static walk
        # ignores it and filters below.
        index = await asyncio.to_thread(self._scenes, start, end)
        dates = sorted(d for d in index if start[:10] <= d <= end[:10])
        if self.clip_bbox is None:
            return self._restrict_to_requested_dates(dates, start, end)
        # A date whose scenes all miss the configured footprint is not an available period.
        # Without this the ingest requests it and `fetch_period` raises, so a release that
        # images several valleys fails the whole run on the first irrelevant date.
        box = (self.clip_bbox[0], self.clip_bbox[1], self.clip_bbox[2], self.clip_bbox[3])
        usable = [d for d in dates if self._overlapping(index[d], box)]
        for dropped in sorted(set(dates) - set(usable)):
            logger.info("%s: no scene over the configured extent, skipping", dropped)
        return self._restrict_to_requested_dates(usable, start, end)

    def _restrict_to_requested_dates(self, available: list[str], start: str, end: str) -> list[str]:
        """Narrow to an explicit `dates` allow-list, refusing silently-missing entries.

        A requested date that falls inside the range but is not on offer is a configuration
        error, not an empty result: ingesting fewer periods than asked for is invisible once
        stored. Dates outside the range are ignored rather than refused, since the range is
        the narrower statement of intent.
        """
        if not self.dates:
            return available
        in_range = [d for d in self.dates if start[:10] <= d <= end[:10]]
        missing = sorted(set(in_range) - set(available))
        if missing:
            raise ValueError(
                f"Requested date(s) {missing} are within {start[:10]}..{end[:10]} but no scene "
                f"covers the configured extent then. Available: {available}. Check `dates` in "
                f"the template, and `max_cloud_cover` if set — filtering can remove a date."
            )
        return [d for d in available if d in set(in_range)]

    # -- fetching ----------------------------------------------------------------------

    def _resolved_crs(self, box: tuple[float, float, float, float]) -> str:
        """The grid CRS, derived once per period from the effective extent.

        Deliberately not derived per scene: two scenes in different UTM zones would otherwise
        build different grids for the same period, and OCS infers the grid from the first
        period and appends along time.
        """
        if self.target_crs:
            return self.target_crs
        return _utm_epsg((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _scene_meta(self, href: str, crs: str, target_res: float) -> _SceneMeta:
        """Read band layout, mask strategy and a suitable overview level in one open.

        Cached per (scene, grid, resolution): the same file is reopened for each period that
        uses it, and the metadata read is a round trip to a slow host.

        Band descriptions must come from the full-resolution dataset — GDAL does not carry
        them on overview levels, so reading them from an overview yields an empty tuple. The
        overview chosen is the coarsest still at or finer than the target grid.
        """
        key = (href, crs, target_res)
        cached = self._meta_cache.get(key)
        if cached is not None:
            return cached
        import rasterio

        with rasterio.open(href) as src:
            indices = _resolve_bands(tuple(src.colorinterp), tuple(src.descriptions), self.bands)
            mask_kind, mask_ref = _resolve_mask(src)
            # Native resolution measured in the *target* CRS: comparing a degree-valued
            # resolution against a metre-valued target would pick the finest overview every
            # time for a geographic source like Vantor, reading far more data than the grid
            # can use.
            left, bottom, right, top = transform_bounds(src.crs, crs, *src.bounds)
            native = max(
                abs(right - left) / max(src.width, 1),
                abs(top - bottom) / max(src.height, 1),
            )
            level: int | None = None
            for index, factor in enumerate(src.overviews(1)):
                if native * factor <= target_res:
                    level = index
            meta = _SceneMeta(indices, mask_kind, mask_ref, level)
        logger.debug(
            "%s: bands %s, mask %s, native %.3g m, overview %s",
            href.rsplit("/", 1)[-1],
            meta.band_indices,
            meta.mask_kind,
            native,
            meta.overview_level,
        )
        self._meta_cache[key] = meta
        return meta

    def _build_grid(self, box: tuple[float, float, float, float], crs: str) -> _Grid:
        w, s, e, n = box
        left, bottom, right, top = transform_bounds("EPSG:4326", crs, w, s, e, n)
        res = self.resolution_m
        # Snap the grid to the resolution so every period lands on identical cells — OCS
        # infers the grid from the first period and appends along time, so it cannot drift.
        left, bottom = np.floor(left / res) * res, np.floor(bottom / res) * res
        right, top = np.ceil(right / res) * res, np.ceil(top / res) * res
        xs = np.arange(left + res / 2, right, res)
        ys = np.arange(top - res / 2, bottom, -res)
        return _Grid(xs, ys, (left, bottom, right, top))

    def _valid_mask(self, window: xr.DataArray, meta: _SceneMeta) -> xr.DataArray:
        """Pixels carrying real data, by whichever strategy the scene supports."""
        if meta.mask_kind == _MASK_ALPHA:
            return window.sel(band=meta.mask_ref).load() > 0
        colour = window.sel(band=list(meta.band_indices)).load()
        reference = meta.mask_ref if meta.mask_kind == _MASK_NODATA else 0
        return cast(xr.DataArray, (colour != reference).any(dim="band"))

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        scenes = self._scenes().get(period_id, [])
        if not scenes:
            raise ValueError(f"No scenes for {period_id!r}")

        w, s, e, n = (float(v) for v in bbox)
        if self.clip_bbox is not None:
            cw, cs, ce, cn = self.clip_bbox
            w, s, e, n = max(w, cw), max(s, cs), min(e, ce), min(n, cn)
            if w >= e or s >= n:
                raise ValueError(f"clip_bbox {self.clip_bbox} does not intersect the instance extent {bbox}")
        scenes = self._overlapping(scenes, (w, s, e, n))
        if not scenes:
            raise ValueError(f"No scene overlaps the configured extent for {period_id}")

        crs = self._resolved_crs((w, s, e, n))
        grid = self._build_grid((w, s, e, n), crs)
        left, bottom, right, top = grid.bounds

        # Refuse loudly rather than allocate. Without this the template below is a silent
        # `np.full` of whatever the extent implies; the kernel kills the process and the
        # operator sees exit 137 with no traceback and no clue why.
        cells = int(grid.ys.size) * int(grid.xs.size)
        if cells > _MAX_GRID_CELLS:
            # A *lower bound*, not the memory the build needs: mosaicking allocates several
            # such arrays (the template, each reprojected scene, and the `xr.where` result),
            # so peak RSS runs well above it — a 14.6M-cell grid is 0.18 GB by this formula
            # and peaked at 4.5 GB in practice. Stated as a floor so an operator sizing a box
            # is not misled by a number that looks complete.
            raise ValueError(
                f"{period_id}: grid {grid.ys.size} x {grid.xs.size} = {cells / 1e6:.0f}M cells "
                f"at {self.resolution_m} m needs at least "
                f"{cells * len(self.bands) * 4 / 1e9:.1f} GB for a single float32 array, and "
                f"several times that to mosaic. The ingest was given the extent "
                f"{[w, s, e, n]}; set `clip_bbox` in the template's ingestion params to the "
                f"event footprint, or raise resolution_m."
            )

        # The target grid carries the band axis, so every scene is reprojected onto identical
        # cells *and* identical channels; reproject_match then needs no per-band bookkeeping.
        template = xr.DataArray(
            np.full((len(self.bands), grid.ys.size, grid.xs.size), np.nan, dtype="float32"),
            coords={"band": list(self.bands), "y": grid.ys, "x": grid.xs},
            dims=("band", "y", "x"),
        ).rio.write_crs(crs)

        import rasterio

        mosaic = template.copy()
        used = 0
        with rasterio.Env(**_GDAL_ENV):
            for scene_ref in scenes:
                meta = self._scene_meta(scene_ref.href, crs, self.resolution_m)
                opened = _open_scene(scene_ref.href, meta.overview_level)
                # The clip box must be expressed in the scene's own CRS, which is not always
                # the grid's: Vantor publishes EPSG:4326 while the grid is UTM.
                scene_box = transform_bounds(crs, opened.rio.crs, left, bottom, right, top)
                try:
                    window = opened.rio.clip_box(*scene_box)
                except NoDataInBounds:
                    continue  # projected footprint misses the snapped grid despite a bbox overlap
                used += 1

                colour = window.sel(band=list(meta.band_indices)).astype("float32").load()
                colour = colour.where(self._valid_mask(window, meta))
                colour = colour.assign_coords(band=list(self.bands)).rio.write_crs(opened.rio.crs)
                resampled = colour.rio.reproject_match(template)
                mosaic = xr.where(mosaic.isnull(), resampled, mosaic).rio.write_crs(crs)

        if not used or bool(mosaic.isnull().all()):
            raise ValueError(f"No scene overlaps the configured extent for {period_id}")
        logger.info(
            "%s: %d of %d overlapping scenes used, grid %d x %d at %g m (%s)",
            period_id,
            used,
            len(scenes),
            grid.ys.size,
            grid.xs.size,
            self.resolution_m,
            crs,
        )

        # Back to uint8 for storage: the source is 8-bit and float32 would quadruple the store
        # for no added information. Masked cells become 0, which the template declares as
        # nodata so they render transparent rather than black.
        stored = mosaic.fillna(0).round().clip(0, 255).astype("uint8")
        cube = stored.expand_dims(t=[np.datetime64(period_id, "ns")]).to_dataset(name=self.variable)
        # Replace the attributes rather than adding to them. `xr.where` propagates attrs from
        # whichever scene it last merged, so the mosaic arrives carrying that one scene's TIFF
        # tags — ACQUISITION_TIME, COLLECT_IDENTIFIER, VEHICLE_NAME. On a cube spanning several
        # dates and several scenes per date those describe one arbitrary contributor while
        # appearing to describe the whole variable, which is worse than having no provenance.
        cube[self.variable].attrs.clear()
        attrs: dict[str, Any] = {
            "long_name": self.long_name or f"{'/'.join(self.bands)} composite (display values, not calibrated)",
            "units": "1",
        }
        if self.source_license:
            # A custom attribute, so the licence does travel with the array. It is not a STAC
            # licence: OCS has no licence field on a dataset template, so the published
            # collection still reports `various` (CLIM-946). Machine-readable on the variable,
            # absent from where a client would actually look.
            attrs["source_license"] = self.source_license
        cube[self.variable].attrs.update(attrs)
        return cast(xr.Dataset, cube)
