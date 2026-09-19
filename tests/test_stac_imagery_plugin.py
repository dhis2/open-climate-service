"""Unit tests for the generic STAC imagery plugin (CLIM-957).

The behaviours worth pinning down are the ones that differ between the two real releases the
plugin was built against, because those are where a wrong answer is silent rather than loud:
band resolution, mask strategy, catalogue shape, and the grid guard.
"""

import asyncio
import logging
from collections.abc import Callable, Generator, Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers the .rio accessor
import xarray as xr
from rasterio.enums import ColorInterp, MaskFlags

from open_climate_service.plugins.datasets.stac_imagery import (
    _MASK_ALPHA,
    _MASK_NODATA,
    _MASK_NONZERO,
    StacImageryPlugin,
    _resolve_bands,
    _resolve_mask,
    _Scene,
    _utm_epsg,
)

_CI = ColorInterp
_RGB = ("red", "green", "blue")
_LOGGER = "open_climate_service.plugins.datasets.stac_imagery"


@pytest.fixture
def warnings_log(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture, None, None]:
    """Capture the plugin's warnings.

    `open_climate_service` sets ``propagate = False`` (startup.py), so caplog's root handler
    never sees these records — the handler has to be attached to the module logger, as
    ``test_plugins_diagnostics.py`` does.
    """
    logger = logging.getLogger(_LOGGER)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _src(
    colorinterp: Sequence[ColorInterp],
    *,
    nodata: float | None = None,
    mask_flags: Sequence[tuple[MaskFlags, ...]] | None = None,
    descriptions: Sequence[str | None] | None = None,
) -> SimpleNamespace:
    """A stand-in for an open rasterio dataset, carrying only what the resolvers read."""
    count = len(colorinterp)
    return SimpleNamespace(
        colorinterp=tuple(colorinterp),
        nodata=nodata,
        mask_flag_enums=tuple(mask_flags or [(MaskFlags.all_valid,)] * count),
        descriptions=tuple(descriptions or [None] * count),
        name="scene.tif",
    )


# -- band resolution -------------------------------------------------------------------


def test_bands_resolved_from_colorinterp_skipping_alpha() -> None:
    """Planet's layout: RGBA, so blue is band 3 and the alpha band is not a colour."""
    assert _resolve_bands((_CI.red, _CI.green, _CI.blue, _CI.alpha), (None,) * 4, _RGB) == (1, 2, 3)


def test_bands_resolved_when_channel_order_is_not_rgb() -> None:
    """Order comes from the declaration, not from position — a BGR asset must not be swapped."""
    assert _resolve_bands((_CI.blue, _CI.green, _CI.red), (None,) * 3, _RGB) == (3, 2, 1)


def test_bands_fall_back_to_descriptions() -> None:
    """Vantor names its bands but an undefined colorinterp would otherwise lose them."""
    src = (_CI.undefined, _CI.undefined, _CI.undefined)
    assert _resolve_bands(src, ("red", "green", "blue"), _RGB) == (1, 2, 3)


def test_bands_fall_back_to_position_only_when_count_matches(warnings_log: pytest.LogCaptureFixture) -> None:
    assert _resolve_bands((_CI.undefined,) * 3, (None,) * 3, _RGB) == (1, 2, 3)
    assert "positional band order" in warnings_log.text


def test_bands_raise_when_unresolvable() -> None:
    """A 5-band asset with no usable names must fail loudly rather than guess channels."""
    with pytest.raises(ValueError, match="Cannot resolve bands"):
        _resolve_bands((_CI.undefined,) * 5, (None,) * 5, _RGB)


# -- mask strategy ---------------------------------------------------------------------


def test_mask_prefers_alpha_band() -> None:
    kind, ref = _resolve_mask(_src((_CI.red, _CI.green, _CI.blue, _CI.alpha)))
    assert (kind, ref) == (_MASK_ALPHA, 4)


def test_mask_uses_nodata_when_there_is_no_alpha() -> None:
    kind, ref = _resolve_mask(_src((_CI.red, _CI.green, _CI.blue), nodata=255.0))
    assert (kind, ref) == (_MASK_NODATA, 255.0)


def test_mask_falls_back_to_nonzero_for_a_bare_rgb_asset() -> None:
    """Vantor's layout: three bands, no alpha, no nodata, every band all_valid."""
    kind, ref = _resolve_mask(_src((_CI.red, _CI.green, _CI.blue)))
    assert (kind, ref) == (_MASK_NONZERO, None)


def test_mask_warns_when_an_unreadable_internal_mask_is_approximated(warnings_log: pytest.LogCaptureFixture) -> None:
    src = _src(
        (_CI.red, _CI.green, _CI.blue),
        mask_flags=[(MaskFlags.per_dataset,)] * 3,
    )
    kind, _ = _resolve_mask(src)
    assert kind == _MASK_NONZERO
    assert "approximating" in warnings_log.text


# -- grid ------------------------------------------------------------------------------


def test_utm_zone_is_picked_from_the_clip_centre() -> None:
    assert _utm_epsg(85.37, 28.21) == "EPSG:32645"  # Nepal, UTM 45N
    assert _utm_epsg(85.37, -28.21) == "EPSG:32745"  # southern hemisphere


def _plugin(**kwargs: Any) -> StacImageryPlugin:
    params: dict[str, Any] = {
        "catalog_url": "https://example.invalid/catalog.json",
        "resolution_m": 10.0,
        "clip_bbox": [85.30, 28.12, 85.40, 28.22],
    }
    params.update(kwargs)
    return StacImageryPlugin(**params)


def test_grid_is_snapped_so_periods_cannot_drift() -> None:
    """OCS infers the grid from the first period and appends along time, so every period must
    land on identical cells regardless of the extent it was asked for."""
    plugin = _plugin()
    crs = plugin._resolved_crs((85.30, 28.12, 85.40, 28.22))
    grid = plugin._build_grid((85.30, 28.12, 85.40, 28.22), crs)
    left, bottom, right, top = grid.bounds
    for edge in (left, bottom, right, top):
        assert edge % plugin.resolution_m == pytest.approx(0.0, abs=1e-6)
    # Half-cell offsets, so coordinates are cell centres rather than edges.
    assert grid.xs[0] == pytest.approx(left + plugin.resolution_m / 2)
    assert grid.ys[0] == pytest.approx(top - plugin.resolution_m / 2)
    assert grid.ys[1] < grid.ys[0]  # y descends


def test_resolution_must_be_finite_and_positive() -> None:
    for bad in (0, -3, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="resolution_m"):
            _plugin(resolution_m=bad)


def test_oversized_grid_is_refused_with_actionable_advice() -> None:
    """The failure this replaces is a kernel SIGKILL with no traceback, so the message has to
    name both the extent it was given and the two ways out."""
    plugin = StacImageryPlugin(catalog_url="https://example.invalid/catalog.json", resolution_m=0.34)
    plugin._index = {"2026-08-27": [_Scene("s.tif", (80.0, 26.0, 88.0, 30.0), "s")]}
    with pytest.raises(ValueError) as excinfo:
        plugin.fetch_period("2026-08-27", [80.0, 26.0, 88.0, 30.0])
    message = str(excinfo.value)
    assert "clip_bbox" in message and "resolution_m" in message
    assert "M cells" in message and "GB" in message


def test_clip_that_misses_the_instance_extent_is_refused() -> None:
    plugin = _plugin(clip_bbox=[10.0, 10.0, 11.0, 11.0])
    plugin._index = {"2026-08-27": [_Scene("s.tif", (10.0, 10.0, 11.0, 11.0), "s")]}
    with pytest.raises(ValueError, match="does not intersect"):
        plugin.fetch_period("2026-08-27", [85.0, 28.0, 86.0, 29.0])


# -- discovery and periods -------------------------------------------------------------


def _item(
    date: str,
    bbox: Iterable[float],
    *,
    cloud: float | None = None,
    props: dict[str, Any] | None = None,
    asset: str = "visual",
) -> dict[str, Any]:
    properties: dict[str, Any] = {"datetime": f"{date}T05:00:00Z"}
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    properties.update(props or {})
    return {
        "id": f"item-{date}",
        "bbox": list(bbox),
        "properties": properties,
        "assets": {asset: {"href": f"{date}.tif"}},
    }


def _catalog(docs: dict[str, dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    """Patch the module's fetcher with an in-memory catalogue keyed by URL."""
    return lambda url: docs[url]


def test_flat_collection_is_walked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vantor's shape: a collection whose items hang directly off it."""
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/collection.json"
    docs = {
        root: {"links": [{"rel": "item", "href": "a.json"}, {"rel": "item", "href": "b.json"}]},
        "https://example.invalid/a.json": _item("2026-08-27", (85.3, 28.1, 85.4, 28.3)),
        "https://example.invalid/b.json": _item("2021-10-16", (85.3, 28.1, 85.4, 28.3)),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))
    plugin = _plugin(catalog_url=root)
    assert sorted(plugin._scenes()) == ["2021-10-16", "2026-08-27"]


def test_nested_catalog_is_walked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planet's shape: root -> phase -> per-sensor collection -> items."""
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/catalog.json"
    docs = {
        root: {"links": [{"rel": "child", "href": "post/catalog.json"}]},
        "https://example.invalid/post/catalog.json": {
            "links": [{"rel": "child", "href": "planetscope-2026-08-26/collection.json"}]
        },
        "https://example.invalid/post/planetscope-2026-08-26/collection.json": {
            "links": [{"rel": "item", "href": "x.json"}]
        },
        "https://example.invalid/post/planetscope-2026-08-26/x.json": _item("2026-08-26", (85.3, 28.1, 85.4, 28.3)),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))
    assert sorted(_plugin(catalog_url=root)._scenes()) == ["2026-08-26"]


def test_collection_filter_prunes_leaves_but_not_intermediate_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sensor name is on the leaf collection, not on Planet's pre/post-event split, so a
    filter applied to intermediate nodes would prune the whole tree."""
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/catalog.json"
    docs = {
        root: {"links": [{"rel": "child", "href": "post-event/catalog.json"}]},
        "https://example.invalid/post-event/catalog.json": {
            "links": [
                {"rel": "child", "href": "planetscope-2026-08-26/collection.json"},
                {"rel": "child", "href": "skysat-2026-08-26/collection.json"},
            ]
        },
        "https://example.invalid/post-event/planetscope-2026-08-26/collection.json": {
            "links": [{"rel": "item", "href": "p.json"}]
        },
        "https://example.invalid/post-event/planetscope-2026-08-26/p.json": _item(
            "2026-08-26", (85.3, 28.1, 85.4, 28.3)
        ),
        "https://example.invalid/post-event/skysat-2026-08-26/collection.json": {
            "links": [{"rel": "item", "href": "s.json"}]
        },
        "https://example.invalid/post-event/skysat-2026-08-26/s.json": _item("2026-08-26", (85.3, 28.1, 85.4, 28.3)),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))
    scenes = _plugin(catalog_url=root, collection_filter="planetscope")._scenes()
    assert [s.item_id for s in scenes["2026-08-26"]] == ["item-2026-08-26"]
    assert len(scenes["2026-08-26"]) == 1


def test_cloud_and_property_filters_drop_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/collection.json"
    docs = {
        root: {
            "links": [
                {"rel": "item", "href": "a.json"},
                {"rel": "item", "href": "b.json"},
                {"rel": "item", "href": "c.json"},
            ]
        },
        "https://example.invalid/a.json": _item(
            "2026-08-27", (85.3, 28.1, 85.4, 28.3), cloud=10, props={"phase": "post"}
        ),
        "https://example.invalid/b.json": _item(
            "2026-08-28", (85.3, 28.1, 85.4, 28.3), cloud=90, props={"phase": "post"}
        ),
        "https://example.invalid/c.json": _item(
            "2021-10-16", (85.3, 28.1, 85.4, 28.3), cloud=5, props={"phase": "pre"}
        ),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))
    plugin = _plugin(catalog_url=root, max_cloud_cover=50, property_filters={"phase": "post"})
    assert sorted(plugin._scenes()) == ["2026-08-27"]


def test_periods_drop_dates_with_no_scene_over_the_clip() -> None:
    """A release covering several valleys otherwise fails the whole run on the first date whose
    scenes lie elsewhere — the Vantor 2026-02-05 case."""
    plugin = _plugin(clip_bbox=[85.30, 28.12, 85.40, 28.22])
    plugin._index = {
        "2026-02-05": [_Scene("elsewhere.tif", (85.06, 27.81, 85.17, 28.04), "e")],
        "2026-08-27": [_Scene("here.tif", (85.28, 28.11, 85.44, 28.37), "h")],
    }
    assert asyncio.run(plugin.periods("2000-01-01", "2026-12-31")) == ["2026-08-27"]


def test_periods_are_bounded_by_the_requested_range() -> None:
    plugin = _plugin(clip_bbox=None)
    plugin._index = {
        "2021-10-16": [_Scene("a.tif", (85.3, 28.1, 85.4, 28.3), "a")],
        "2023-09-17": [_Scene("b.tif", (85.3, 28.1, 85.4, 28.3), "b")],
        "2026-08-27": [_Scene("c.tif", (85.3, 28.1, 85.4, 28.3), "c")],
    }
    assert asyncio.run(plugin.periods("2022-01-01", "2026-08-27")) == [
        "2023-09-17",
        "2026-08-27",
    ]


def test_empty_catalogue_raises_rather_than_ingesting_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/collection.json"
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog({root: {"links": []}}))
    with pytest.raises(ValueError, match="No items with a 'visual' asset"):
        _plugin(catalog_url=root)._scenes()


def test_missing_asset_key_is_reported_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, warnings_log: pytest.LogCaptureFixture
) -> None:
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/collection.json"
    docs = {
        root: {"links": [{"rel": "item", "href": "a.json"}]},
        "https://example.invalid/a.json": _item("2026-08-27", (85.3, 28.1, 85.4, 28.3), asset="thumbnail"),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))
    with pytest.raises(ValueError):
        _plugin(catalog_url=root)._scenes()
    assert "has no 'visual' asset" in warnings_log.text


def test_overlap_test_excludes_scenes_that_only_touch_the_edge() -> None:
    scenes = [
        _Scene("touching.tif", (85.40, 28.12, 85.50, 28.22), "t"),
        _Scene("inside.tif", (85.31, 28.13, 85.35, 28.20), "i"),
    ]
    kept = StacImageryPlugin._overlapping(scenes, (85.30, 28.12, 85.40, 28.22))
    assert [s.item_id for s in kept] == ["i"]


def test_band_names_reach_the_stored_variable() -> None:
    """`display.bands` renders by name, so the band coordinate is part of the contract."""
    plugin = _plugin(bands=["red", "green", "blue"], variable="true_colour")
    assert plugin.bands == ("red", "green", "blue")
    assert plugin.variable == "true_colour"


def test_bands_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="bands must not be empty"):
        _plugin(bands=[])


def test_utm_grid_is_stable_across_scenes_in_different_source_crs() -> None:
    """The CRS is derived once from the extent, not per scene: two scenes in different UTM
    zones must not build two different grids for the same period."""
    plugin = _plugin()
    box = (85.30, 28.12, 85.40, 28.22)
    assert plugin._resolved_crs(box) == plugin._resolved_crs(box) == "EPSG:32645"


def test_explicit_target_crs_wins_over_the_derived_zone() -> None:
    plugin = _plugin(target_crs="EPSG:3857")
    assert plugin._resolved_crs((85.30, 28.12, 85.40, 28.22)) == "EPSG:3857"


def test_mask_reference_is_applied_per_strategy() -> None:
    """The nodata strategy must compare against the declared value, not against zero."""
    import xarray as xr

    from open_climate_service.plugins.datasets.stac_imagery import _SceneMeta

    window = xr.DataArray(
        np.array([[[0, 255], [7, 7]], [[0, 255], [7, 7]], [[0, 255], [7, 7]]], dtype="uint8"),
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": [1, 0], "x": [0, 1]},
    )
    plugin = _plugin()
    nodata = plugin._valid_mask(window, _SceneMeta((1, 2, 3), _MASK_NODATA, 255, None))
    assert nodata.values.tolist() == [[True, False], [True, True]]
    nonzero = plugin._valid_mask(window, _SceneMeta((1, 2, 3), _MASK_NONZERO, None, None))
    assert nonzero.values.tolist() == [[False, True], [True, True]]


def _fake_scene_array(bounds: tuple[float, float, float, float]) -> xr.DataArray:
    """A small RGB scene in UTM 45N carrying the TIFF tags a real Maxar asset does."""

    left, bottom, right, top = bounds
    n = 64
    xs = np.linspace(left, right, n)
    ys = np.linspace(top, bottom, n)
    data = np.full((3, n, n), 120, dtype="uint8")
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": ys, "x": xs},
        attrs={
            "ACQUISITION_TIME": "2021-10-16T05:22:48Z",
            "COLLECT_IDENTIFIER": "10300100C86CED00",
            "VEHICLE_NAME": "WV02",
        },
    ).rio.write_crs("EPSG:32645")


def test_scene_tiff_tags_do_not_leak_onto_the_stored_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`xr.where` propagates attrs from whichever scene it last merged, so the mosaic arrives
    carrying one contributor's ACQUISITION_TIME / COLLECT_IDENTIFIER / VEHICLE_NAME. On a cube
    spanning several dates and several scenes per date those describe one arbitrary scene while
    appearing to describe the whole variable."""
    from rasterio.warp import transform_bounds

    from open_climate_service.plugins.datasets import stac_imagery
    from open_climate_service.plugins.datasets.stac_imagery import _SceneMeta

    clip = [85.3474, 28.2087, 85.3608, 28.2207]
    plugin = _plugin(
        resolution_m=10.0,
        clip_bbox=clip,
        source_license="CC-BY-NC-4.0",
        long_name="Maxar true-colour composite",
    )
    plugin._index = {"2026-08-27": [_Scene("s.tif", (85.34, 28.20, 85.37, 28.23), "s")]}
    utm = transform_bounds("EPSG:4326", "EPSG:32645", 85.34, 28.20, 85.37, 28.23)

    monkeypatch.setattr(stac_imagery, "_open_scene", lambda href, level: _fake_scene_array(utm))
    monkeypatch.setattr(
        StacImageryPlugin,
        "_scene_meta",
        lambda self, href, crs, res: _SceneMeta((1, 2, 3), _MASK_NONZERO, None, None),
    )

    cube = plugin.fetch_period("2026-08-27", clip)
    attrs = cube["true_colour"].attrs
    for leaked in ("ACQUISITION_TIME", "COLLECT_IDENTIFIER", "VEHICLE_NAME"):
        assert leaked not in attrs, f"{leaked} leaked from a source scene onto the variable"
    assert attrs["source_license"] == "CC-BY-NC-4.0"
    assert attrs["long_name"] == "Maxar true-colour composite"
    assert isinstance(cube, xr.Dataset)
    assert cube["true_colour"].dtype == "uint8"
    assert list(np.asarray(cube["true_colour"]["band"].values)) == ["red", "green", "blue"]


# -- STAC API item search --------------------------------------------------------------


def _api_root(search_href: str = "https://api.invalid/search") -> dict[str, Any]:
    return {
        "conformsTo": ["https://api.stacspec.org/v1.0.0/item-search"],
        "links": [{"rel": "search", "href": search_href}],
    }


class _FakePost:
    """Stands in for httpx.post, serving canned search pages and recording the bodies sent."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def __call__(self, url: str, json: dict[str, Any], **_: Any) -> Any:
        self.urls.append(url)
        self.bodies.append(json)
        page = self.pages[min(len(self.bodies) - 1, len(self.pages) - 1)]
        return SimpleNamespace(json=lambda: page, raise_for_status=lambda: None)


def _search_page(items: list[dict[str, Any]], next_body: dict[str, Any] | None = None) -> dict[str, Any]:
    links = []
    if next_body is not None:
        links.append({"rel": "next", "href": "https://api.invalid/search", "body": next_body})
    return {"features": items, "links": links}


def test_stac_api_is_detected_and_searched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A global archive cannot be walked, so an item-search API must be queried instead."""
    from open_climate_service.plugins.datasets import stac_imagery

    monkeypatch.setattr(stac_imagery, "_get_json", lambda url: _api_root())
    post = _FakePost([_search_page([_item("2026-08-12", (85.3, 28.1, 85.4, 28.3))])])
    monkeypatch.setattr(stac_imagery.httpx, "post", post)

    plugin = _plugin(catalog_url="https://api.invalid/v1", collections=["sentinel-2-l2a"])
    assert asyncio.run(plugin.periods("2026-08-01", "2026-08-31")) == ["2026-08-12"]
    body = post.bodies[0]
    assert body["collections"] == ["sentinel-2-l2a"]
    # The clip and the range become search parameters — that is the whole point.
    assert body["bbox"] == [85.30, 28.12, 85.40, 28.22]
    assert body["datetime"] == "2026-08-01T00:00:00Z/2026-08-31T23:59:59Z"


def test_a_static_catalogue_is_not_mistaken_for_an_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `search` link alone is not enough: some static catalogues link to a search UI."""
    from open_climate_service.plugins.datasets import stac_imagery

    root = "https://example.invalid/collection.json"
    docs = {
        root: {
            "links": [
                {"rel": "search", "href": "https://example.invalid/ui"},
                {"rel": "item", "href": "a.json"},
            ]
        },
        "https://example.invalid/a.json": _item("2026-08-27", (85.3, 28.1, 85.4, 28.3)),
    }
    monkeypatch.setattr(stac_imagery, "_get_json", _catalog(docs))

    def _explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("static catalogue must not be POSTed to")

    monkeypatch.setattr(stac_imagery.httpx, "post", _explode)
    assert sorted(_plugin(catalog_url=root)._scenes()) == ["2026-08-27"]


def test_search_follows_next_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.datasets import stac_imagery

    monkeypatch.setattr(stac_imagery, "_get_json", lambda url: _api_root())
    post = _FakePost(
        [
            _search_page([_item("2026-08-12", (85.3, 28.1, 85.4, 28.3))], next_body={"page": 2}),
            _search_page([_item("2026-08-27", (85.3, 28.1, 85.4, 28.3))]),
        ]
    )
    monkeypatch.setattr(stac_imagery.httpx, "post", post)
    plugin = _plugin(catalog_url="https://api.invalid/v1", collections=["s2"])
    assert sorted(plugin._scenes()) == ["2026-08-12", "2026-08-27"]
    assert post.bodies[1]["page"] == 2  # cursor carried forward


def test_search_page_cap_warns_rather_than_silently_truncating(
    monkeypatch: pytest.MonkeyPatch, warnings_log: pytest.LogCaptureFixture
) -> None:
    """Bounding the sweep is right; letting it read as 'that was everything' is not."""
    from open_climate_service.plugins.datasets import stac_imagery

    monkeypatch.setattr(stac_imagery, "_get_json", lambda url: _api_root())
    # Every page advertises another, so only the cap stops it.
    endless = _FakePost([_search_page([_item("2026-08-12", (85.3, 28.1, 85.4, 28.3))], next_body={"p": 1})])
    monkeypatch.setattr(stac_imagery.httpx, "post", endless)
    monkeypatch.setattr(stac_imagery, "_SEARCH_MAX_PAGES", 3)
    _plugin(catalog_url="https://api.invalid/v1", collections=["s2"])._scenes()
    assert len(endless.bodies) == 3
    assert "Stopped after 3 search pages" in warnings_log.text


def test_search_results_with_no_usable_asset_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.plugins.datasets import stac_imagery

    monkeypatch.setattr(stac_imagery, "_get_json", lambda url: _api_root())
    monkeypatch.setattr(stac_imagery.httpx, "post", _FakePost([_search_page([])]))
    with pytest.raises(ValueError, match="STAC API search"):
        _plugin(catalog_url="https://api.invalid/v1", collections=["s2"])._scenes()


def test_asset_href_resolves_against_the_item_self_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative asset href in a search response must resolve against the item, not against
    whichever link happens to be first."""
    from open_climate_service.plugins.datasets import stac_imagery

    item = _item("2026-08-12", (85.3, 28.1, 85.4, 28.3))
    item["assets"]["visual"]["href"] = "TCI.tif"
    item["links"] = [
        {"rel": "collection", "href": "https://api.invalid/collections/s2"},
        {"rel": "self", "href": "https://data.invalid/tiles/45RUM/item.json"},
    ]
    monkeypatch.setattr(stac_imagery, "_get_json", lambda url: _api_root())
    monkeypatch.setattr(stac_imagery.httpx, "post", _FakePost([_search_page([item])]))
    scenes = _plugin(catalog_url="https://api.invalid/v1", collections=["s2"])._scenes()
    assert scenes["2026-08-12"][0].href == "https://data.invalid/tiles/45RUM/TCI.tif"


# -- explicit date selection -----------------------------------------------------------


def _indexed(plugin: StacImageryPlugin, dates: list[str]) -> StacImageryPlugin:
    plugin._index = {d: [_Scene(f"{d}.tif", (85.3, 28.1, 85.4, 28.3), d)] for d in dates}
    return plugin


def test_dates_allow_list_keeps_a_non_contiguous_selection() -> None:
    """A range cannot express "these two and not the five between", which is the common case
    for a before/after pair drawn from a repeat-pass source."""
    plugin = _indexed(
        _plugin(dates=["2026-08-12", "2026-08-27"]),
        ["2026-08-12", "2026-08-14", "2026-08-19", "2026-08-24", "2026-08-27"],
    )
    assert asyncio.run(plugin.periods("2026-08-01", "2026-08-31")) == [
        "2026-08-12",
        "2026-08-27",
    ]


def test_dates_allow_list_applies_without_a_clip() -> None:
    plugin = _indexed(_plugin(clip_bbox=None, dates=["2026-08-12"]), ["2026-08-12", "2026-08-14"])
    assert asyncio.run(plugin.periods("2026-08-01", "2026-08-31")) == ["2026-08-12"]


def test_a_requested_date_that_is_not_on_offer_is_refused() -> None:
    """Ingesting fewer periods than asked for is invisible once stored, so a date inside the
    range that no scene covers must fail rather than quietly shrink the result."""
    plugin = _indexed(_plugin(dates=["2026-08-12", "2026-08-13"]), ["2026-08-12"])
    with pytest.raises(ValueError, match="2026-08-13"):
        asyncio.run(plugin.periods("2026-08-01", "2026-08-31"))


def test_requested_dates_outside_the_range_are_ignored_not_refused() -> None:
    """The range is the narrower statement of intent, so it wins without complaint."""
    plugin = _indexed(_plugin(dates=["2026-08-12", "2026-09-30"]), ["2026-08-12"])
    assert asyncio.run(plugin.periods("2026-08-01", "2026-08-31")) == ["2026-08-12"]


def test_no_dates_means_every_date_in_range() -> None:
    plugin = _indexed(_plugin(), ["2026-08-12", "2026-08-14"])
    assert asyncio.run(plugin.periods("2026-08-01", "2026-08-31")) == [
        "2026-08-12",
        "2026-08-14",
    ]
