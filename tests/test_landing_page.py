"""The landing page's areas, and the split of dataset templates between them (CLIM-940).

A template is shown as a data source when it can be ingested, and as a workflow output when a
workflow produces it. The split is decided by `_landing_catalogue`, so the unit tests below
drive it directly; the rendering tests go through `GET /`, where the page is actually served.
"""

from __future__ import annotations

import math
import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.system import templates as landing

AREAS = ["overview", "datasets", "data-sources", "workflows", "processes"]


def _template(template_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": template_id, "name": template_id.replace("_", " "), "sync": {"kind": "static"}, **fields}


def _ingestable(template_id: str, **fields: Any) -> dict[str, Any]:
    return _template(template_id, ingestion={"plugin": "some.Plugin"}, **fields)


def _workflow(workflow_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=workflow_id, summary=f"{workflow_id} summary")


class _VisibleText(HTMLParser):
    """Collects the text a reader sees: no markup, scripts, styles or comments."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return " ".join(parser.parts)


def _area_ids(html: str) -> list[str]:
    return re.findall(r'data-area-link="([^"]+)"', html)


# --- the split ---------------------------------------------------------------------------


def test_ingestable_templates_are_data_sources_and_nothing_else() -> None:
    catalogue = landing._landing_catalogue(
        [_ingestable("chirps"), _template("normal", produced_by="climate_normal")],
        [_workflow("climate_normal")],
    )

    assert [source["id"] for source in catalogue["sources"]] == ["chirps"]
    assert [output["id"] for output in catalogue["workflows"][0]["outputs"]] == ["normal"]
    assert catalogue["unattributed_outputs"] == []


def test_every_template_appears_in_exactly_one_place() -> None:
    templates = [
        _ingestable("a"),
        _template("b", produced_by="w1"),
        _template("c", produced_by="w2"),
        _template("d", produced_by="unregistered"),
        _template("e"),
    ]
    catalogue = landing._landing_catalogue(templates, [_workflow("w1"), _workflow("w2")])

    placed = [s["id"] for s in catalogue["sources"]]
    placed += [o["id"] for w in catalogue["workflows"] for o in w["outputs"]]
    placed += [o["id"] for o in catalogue["unattributed_outputs"]]
    assert sorted(placed) == ["a", "b", "c", "d", "e"]


def test_an_output_of_an_unknown_workflow_is_listed_not_dropped() -> None:
    """A workflow can be registered at runtime, so the template may name one not yet known."""
    catalogue = landing._landing_catalogue(
        [_template("orphan", produced_by="not_registered"), _template("undeclared")],
        [],
    )

    assert {o["id"] for o in catalogue["unattributed_outputs"]} == {"orphan", "undeclared"}


def test_data_sources_are_ordered_by_provider_then_name() -> None:
    catalogue = landing._landing_catalogue(
        [
            _ingestable("z", name="Zeta", source="B provider"),
            _ingestable("y", name="Beta", source="A provider"),
            _ingestable("x", name="Alpha", source="B provider"),
        ],
        [],
    )

    assert [s["id"] for s in catalogue["sources"]] == ["y", "x", "z"]


@pytest.mark.parametrize(
    ("licence", "label"),
    [
        ("CC-BY-4.0", "CC-BY-4.0"),
        ({"name": "Copernicus licence", "url": "https://example.org"}, "Copernicus licence"),
        (None, None),
    ],
)
def test_licence_label(licence: object, label: str | None) -> None:
    template = _ingestable("t") if licence is None else _ingestable("t", license=licence)
    assert landing._landing_catalogue([template], [])["sources"][0]["licence"] == label


# --- the rendered page -------------------------------------------------------------------


def test_the_overview_links_to_every_area_as_a_page(client: TestClient) -> None:
    """The areas are pages now, so the root counts them and links to them.

    A fragment could not be linked to by anything that does not run JavaScript; each of these
    can be bookmarked, shared and crawled.
    """
    html = client.get("/", headers={"Accept": "text/html"}).text

    for path in ("/datasets", "/dataset-templates", "/workflows", "/processes"):
        assert f'href="{path}"' in html
    assert "data-area-link" not in html, "nothing switches areas in place any more"
    # The stat cards are links too, and pointed at fragments this page no longer has.
    assert 'class="panel stat" href="/datasets"' in html
    assert 'href="#datasets"' not in html and 'href="#data-sources"' not in html


def test_the_list_script_carries_nothing_for_areas(client: TestClient) -> None:
    """The switcher went with the areas.

    It ships to all four list pages, so leaving it in meant every one of them ran a
    `querySelectorAll` for elements no template contains.
    """
    body = client.get("/datasets", headers={"Accept": BROWSER_ACCEPT}).text

    assert "initList" in body, "the list behaviour is still there"
    assert "data-area" not in body
    assert "scroll-margin-top: 100vh" not in body, "the rule existed only for the switcher"


def test_read_only_is_stated_on_the_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    """Said once, where a visitor will see it, since no page then offers ingest or sync."""
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    html = landing.render_landing("0.0.0", "")

    assert "read-only" in _visible_text(html)


def test_access_and_version_are_stated_without_a_configured_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They describe the instance, not its extent, so an unconfigured one still reports them.

    Both sat inside the extent block, so a fresh instance — the one an operator is in the
    middle of setting up, and the most likely to be read — showed neither.
    """
    monkeypatch.setattr(landing, "_load_extent", lambda: None)

    visible = _visible_text(landing.render_landing("9.9.9", ""))

    assert "No extent configured" in visible
    assert "Access" in visible and "Read and write" in visible
    assert "Version" in visible and "9.9.9" in visible


def test_the_interface_calls_them_dataset_templates(client: TestClient) -> None:
    """The stat card and the rail agree, and neither says "data source"."""
    visible = _visible_text(client.get("/", headers={"Accept": "text/html"}).text)

    assert visible.lower().count("dataset templates") == 2
    assert "data source" not in visible.lower()


def test_colours_come_from_the_token_block(client: TestClient) -> None:
    """Hex literals may appear only where the DHIS2 tokens are declared."""
    html = client.get("/", headers={"Accept": "text/html"}).text
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    rules = style.split("--header-height", 1)[1]

    assert "@dhis2/ui-constants 10.17.0" in style
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b", rules) == []


def test_the_json_representation_is_unchanged(client: TestClient) -> None:
    response = client.get("/", params={"f": "json"})

    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert {"api_version", "backend_version", "endpoints", "links"} <= body.keys()


# --- datasets: tiles and list ------------------------------------------------------------


def _record(dataset_id: str = "chirps_monthly", **fields: Any) -> Any:
    from open_climate_service.ingestions.schemas import DatasetDetailRecord

    values: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_dataset_id": dataset_id,
        "dataset_name": "Precipitation (CHIRPS, monthly)",
        "description": "First line\nwraps here.\n\nSecond paragraph.",
        "variable": "precip",
        "period_type": "monthly",
        "units": "mm/d",
        "source": "CHIRPS v3",
        "source_url": "https://example.org/chirps",
        "extent": {
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": "2020-01", "end": "2026-07"},
        },
        "last_updated": "2026-09-01T07:58:03Z",
        "links": [
            {"href": f"/datasets/{dataset_id}", "rel": "self", "title": "Dataset detail"},
            {"href": f"/zarr/{dataset_id}", "rel": "zarr", "title": "Zarr store"},
        ],
        # Required since GeoParquet feature collections landed: a listing has to tell a raster
        # from a feature without a request per row.
        "itemType": "coverage",
        "publication": {"status": "published", "published_at": "2026-09-01T07:58:03Z"},
        "versions": [],
        **fields,
    }
    return DatasetDetailRecord.model_validate(values)


BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


# --- the overview panel -------------------------------------------------------------------


def test_the_overview_counts_what_the_instance_holds(client: TestClient) -> None:
    """Published is not a headline figure — how much data is stored is."""
    section = client.get("/", headers={"Accept": "text/html"}).text.split('id="overview"', 1)[1]
    section = section.split("</section>", 1)[0]

    assert "Data stored" in section
    assert "Published" not in section


@pytest.mark.parametrize(
    ("total", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (150_000, "150 KB"), (2_400_000_000, "2.4 GB")],
)
def test_a_stored_size_reads_at_a_glance(total: int, expected: str) -> None:
    assert landing._format_bytes(total) == expected


def test_the_stored_size_sums_each_store_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two artifacts appended to one store must not be counted twice."""
    store = tmp_path / "chirps.icechunk"
    (store / "chunks").mkdir(parents=True)
    (store / "chunks" / "0").write_bytes(b"x" * 1000)

    monkeypatch.setattr(
        landing,
        "_stored_bytes_cache",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        ingestion_services,
        "list_artifacts",
        lambda: SimpleNamespace(items=[SimpleNamespace(path=str(store)), SimpleNamespace(path=str(store))]),
    )

    assert landing._stored_bytes() == 1000
    landing._stored_bytes_cache = None


# --- processes in the landing page --------------------------------------------------------

# --- processes ---------------------------------------------------------------------------


def test_processes_are_tagged_with_their_origin() -> None:
    processes = {p["id"]: p for p in landing._load_processes()}

    assert processes["load_collection"]["origin"] == "core"
    assert processes["add"]["origin"] == "core"
    assert processes["spi"]["origin"] == "ocs"
    assert {"xclim", "earthkit"} <= {p["origin"] for p in processes.values()}
    # OCS first, openEO core last, so the default view leads with what is specific to OCS.
    origins = [p["origin"] for p in processes.values()]
    assert origins[0] == "ocs" and origins[-1] == "core"


def test_a_plugin_overriding_an_indicator_counts_as_ocs(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import earthkit_processes, plugin_processes, processes, xclim_processes

    def indicator() -> None: ...

    def override() -> None: ...

    monkeypatch.setattr(processes, "list_openeo_processes", lambda: [{"id": "tg_mean"}, {"id": "abs"}])
    monkeypatch.setattr(xclim_processes, "scan", lambda: [indicator])
    monkeypatch.setattr(earthkit_processes, "scan", lambda: [])
    monkeypatch.setattr(plugin_processes, "load_plugin_processes", lambda: [("tg_mean", override)])

    assert {p["id"]: p["origin"] for p in landing._load_processes()} == {"tg_mean": "ocs", "abs": "core"}


# --- the extent globe ---------------------------------------------------------------------

# --- the extent globe --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bbox", "zoom"),
    [
        ([-180, -90, 180, 90], 1.85),  # a global extent shows as much as the frame can hold
        ([-25, 34, 45, 72], 1.85),  # Europe is wide enough to sit at the same bound
        ([80.05, 26.35, 88.2, 30.45], 2.2),  # Nepal, at the upper bound
        ([-13.5, 6.9, -10.1, 10.0], 2.2),  # a small country keeps its continent around it
    ],
)
def test_the_globe_zooms_to_the_size_of_the_extent(bbox: list[float], zoom: float) -> None:
    width, height = abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])

    assert landing._globe_zoom(width, height, (bbox[1] + bbox[3]) / 2) == pytest.approx(zoom, abs=0.01)


def test_the_globe_is_centred_on_the_extent() -> None:
    """The extent's centre projects to the middle of the globe, wherever on Earth it is."""
    for bbox in ([-13.5, 6.9, -10.1, 10.0], [80.05, 26.35, 88.2, 30.45], [-70, -40, -60, -30]):
        lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        point = landing._project(lon0, lat0, lon0, lat0, landing._GLOBE_RADIUS)
        assert point == (landing._GLOBE_WIDTH / 2, landing._GLOBE_HEIGHT / 2)


def test_the_far_side_of_the_world_is_pushed_out_to_the_limb() -> None:
    """Orthographic shows one hemisphere; the antipode must not fold onto the front.

    Pushed out rather than dropped. Dropping it broke the ring it belonged to, and the visible
    run then closed with a straight chord — a hard diagonal across Europe on a globe centred on
    Nepal. On the limb it keeps the ring whole, and the limb is outside the frame.
    """
    scale = 48.0
    centre = (landing._GLOBE_WIDTH / 2, landing._GLOBE_HEIGHT / 2)

    far = landing._project(0.0, 0.0, 180.0, 0.0, scale)
    near = landing._project(1.0, 1.0, 0.0, 0.0, scale)

    assert math.hypot(far[0] - centre[0], far[1] - centre[1]) == pytest.approx(scale)
    assert math.hypot(near[0] - centre[0], near[1] - centre[1]) < scale
    assert landing._on_far_side(0.0, 0.0, 180.0, 0.0)
    assert not landing._on_far_side(1.0, 1.0, 0.0, 0.0)


def test_a_ring_cut_by_the_horizon_is_never_split_into_chords() -> None:
    """The bug this guards: a ring with anything visible draws as exactly one subpath.

    A ring split at the horizon leaves each visible run to close itself, and a run closed
    across the width of a hemisphere is a straight line through everything between its ends.
    Counting subpaths is the structural form of "no chord": as long as no ring is cut, none
    can be closed short. A ring with nothing visible draws nothing at all.
    """
    # Centred on Nepal, which puts Eurasia across the horizon — the ring that showed the chord.
    bbox = (80.05, 26.35, 88.2, 30.45)
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    showing = sum(
        any(not landing._on_far_side(lon, lat, lon0, lat0) for lon, lat in ring) for ring in landing._world_rings()
    )

    globe = landing._globe(bbox)

    assert 0 < showing < len(landing._world_rings()), "this extent hides some rings and shows others"
    assert globe["land"].count("M") == showing


def test_the_limb_step_is_fine_enough_that_its_chords_stay_off_the_frame() -> None:
    """Putting a hidden vertex on the limb does not put the segment to the next one there.

    SVG joins two limb points with a chord, which dips to `scale * cos(gap / 2)`. The frame's
    own corner is the radius that matters: anything beyond it is outside the rectangle. At
    `_MIN_ZOOM` — where Brazil sits — the limb clears that corner by about a unit, so the gap
    has to stay small or the chord lands back on the map.
    """
    scale = landing._GLOBE_RADIUS * landing._MIN_ZOOM
    corner = math.hypot(landing._GLOBE_WIDTH / 2, landing._GLOBE_HEIGHT / 2)

    assert scale * math.cos(landing._LIMB_STEP / 2) > corner


@pytest.mark.parametrize(
    "bbox",
    [
        # Brazil is where this was first found: three chords, all in one fully hidden ring.
        # New Zealand is the harder case and the reason the limb arc is what carries the
        # property — its offender is a ring that *is* partly visible, so dropping hidden
        # rings does not reach it, and only walking the limb keeps the chord off the frame.
        (-74.0, -34.0, -34.0, 5.3),
        (166.0, -47.0, 179.0, -34.0),
        (80.05, 26.35, 88.2, 30.45),
        (4.5, 58.0, 31.0, 71.2),
        (-180.0, -85.0, 180.0, 85.0),
    ],
)
def test_the_hidden_side_of_the_world_never_paints_into_the_frame(
    bbox: tuple[float, float, float, float],
) -> None:
    """End to end: walk every drawn segment that lies on the limb and check it stays off."""
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    scale = landing._GLOBE_RADIUS * landing._globe_zoom(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]), lat0)
    centre = (landing._GLOBE_WIDTH / 2, landing._GLOBE_HEIGHT / 2)

    def on_limb(point: tuple[float, float]) -> bool:
        return math.hypot(point[0] - centre[0], point[1] - centre[1]) >= scale - 0.5

    intruders = []
    for ring in landing._world_rings():
        drawn = landing._land_path(ring, lon0, lat0, scale)
        if not drawn:
            continue
        values = [float(v) for v in re.findall(r"-?\d+\.?\d*", drawn)]
        points = list(zip(values[0::2], values[1::2], strict=True))
        for start, end in zip(points, points[1:] + points[:1], strict=True):
            if not (on_limb(start) and on_limb(end)):
                continue
            # Sampled along its length, not at its midpoint: a chord this shallow enters the
            # frame through a corner and leaves again, with its deepest point still outside.
            if any(
                0 <= start[0] + (end[0] - start[0]) * step / 64 <= landing._GLOBE_WIDTH
                and 0 <= start[1] + (end[1] - start[1]) * step / 64 <= landing._GLOBE_HEIGHT
                for step in range(65)
            ):
                intruders.append((start, end))

    assert intruders == []


def test_an_extent_wider_than_the_visible_face_draws_no_outline() -> None:
    """Its real edge is the limb, which the zoom keeps off-frame, so a box would be a lie."""
    world = landing._globe((-180.0, -85.0, 180.0, 85.0))
    nepal = landing._globe((80.05, 26.35, 88.2, 30.45))

    assert world["extent"] == ""
    assert world["land"].count("M") > 10, "the land is still drawn"
    assert nepal["extent"].startswith("M"), "an extent that fits still gets its outline"


def test_the_map_never_shows_the_sphere_s_edge() -> None:
    """Cropped to a rectangle rather than drawn as a ball: the sphere must overflow the corners."""
    half_diagonal = ((landing._GLOBE_WIDTH / 2) ** 2 + (landing._GLOBE_HEIGHT / 2) ** 2) ** 0.5

    assert landing._GLOBE_RADIUS * landing._MIN_ZOOM > half_diagonal


def test_the_globe_draws_land_and_the_extent() -> None:
    globe = landing._extent_globe({"bbox": [-13.5, 6.9, -10.1, 10.0]})

    assert globe is not None
    assert globe["land"].count("M") > 10, "the visible hemisphere's coastlines"
    assert globe["extent"].startswith("M") and globe["extent"].endswith("Z")
    # Every drawn point is inside the viewBox's circle-ish area, so nothing escapes the clip.
    assert "-" not in globe["extent"], "the extent stays on the front of the globe"


@pytest.mark.parametrize(
    "extent",
    [None, {}, {"bbox": "world"}, {"bbox": [1, 2, 3]}, {"bbox": [1, 2, 3, "x"]}, {"bbox": [10, 10, 5, 5]}],
)
def test_no_globe_without_a_usable_extent(extent: dict[str, Any] | None) -> None:
    assert landing._extent_globe(extent) is None


def test_the_overview_draws_the_globe_without_fetching_anything(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    globe = re.search(r'<svg[^>]*class="globe".*?</svg>', html, re.S)

    assert globe is not None
    svg = globe.group(0)
    assert 'class="land"' in svg and 'class="extent"' in svg
    assert "http" not in svg, "no tiles, no external image: the page works offline"
