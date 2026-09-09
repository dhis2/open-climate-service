"""Dataset thumbnails: which slice is rendered, when, and how it reaches STAC (CLIM-1076).

A thumbnail is what makes a bad ingest visible — a flipped grid, a wrong extent, a unit
error all look identical in metadata. So the tests here assert the *content* of the image
(which slice it shows) rather than only that a file appeared, by rendering the expected
slice independently and comparing bytes.

The two timing tests are a pair, and neither is sufficient alone: one asserts the renderer
is never reached during a multi-period sync, the other that one ingest produces exactly one
render. Together they pin "once per run, at the end" rather than "per commit", which is a
distinction invisible in the published result — only the final image survives either way.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.data_manager.services import downloader
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.shared import thumbnails
from open_climate_service.shared.thumbnails import (
    THUMBNAIL_LONG_SIDE_PIXELS,
    declared_midpoint,
    render_png,
    representative_slice,
    resolve_colormap,
    stretch_range,
    thumbnail_path,
    write_dataset_thumbnail,
)

# A date the tests pin so "nearest to now" is a fixed answer rather than a moving one.
GENERATED_AT = datetime(2026, 3, 1, tzinfo=UTC)

DATASET = {"id": "thumb_dataset", "variable": "precip", "display": {"colormap": "blues", "range": [0.0, 10.0]}}


@pytest.fixture(autouse=True)
def _data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the data root at a temp dir so thumbnails never land in the developer's own."""
    root = tmp_path / "data"
    monkeypatch.setattr(thumbnails.api_config, "get_data_root", lambda: root)
    return root


def _store(tmp_path: Path, ds: xr.Dataset, *, t_dim: str | None = "t") -> Path:
    """Write *ds* to a real Icechunk store, so the read path under test is the real one."""
    path = tmp_path / "thumb.icechunk"
    downloader.write_to_icechunk_store(ds, path, t_dim=t_dim, commit_message="test")
    return path


# One distinct *pattern* per step, not one distinct constant. The thumbnail scales to the
# slice it renders, so two constant slices produce the same image whatever their values —
# only a different arrangement survives the normalisation and identifies which step was used.
_PATTERNS = [
    [[0.0, 1.0], [2.0, 3.0]],
    [[3.0, 2.0], [1.0, 0.0]],
    [[0.0, 3.0], [1.0, 2.0]],
    [[2.0, 0.0], [3.0, 1.0]],
]


def _daily_cube(steps: int, start: str = "2026-01-01") -> xr.Dataset:
    times = pd.date_range(start, periods=steps, freq="D")
    data = np.array(_PATTERNS[:steps], dtype="float32")
    return xr.Dataset(
        {"precip": (("t", "y", "x"), data)},
        coords={"t": times, "y": [1.5, 0.5], "x": [10.5, 11.5]},
    )


def _rendered_reference(arr: Any, tmp_path: Path) -> bytes:
    """The bytes the thumbnail should have, rendered from the slice we expect it to show.

    Uses the same per-slice stretch as the code under test, because these tests are about
    *which slice* was chosen; the stretch itself is covered separately below.
    """
    reference = render_png(
        arr,
        tmp_path / "reference.png",
        colormap="blues",
        clim=stretch_range(arr.values),
        long_side=THUMBNAIL_LONG_SIDE_PIXELS,
    )
    return reference.read_bytes()


def _rendered_declared_range(arr: Any, tmp_path: Path) -> bytes:
    """What the thumbnail would have been against the template's declared display range."""
    return render_png(
        arr,
        tmp_path / "declared.png",
        colormap="blues",
        clim=(0.0, 20.0),
        long_side=THUMBNAIL_LONG_SIDE_PIXELS,
    ).read_bytes()


# -- which slice -----------------------------------------------------------------------


def test_a_datetime_store_renders_the_step_nearest_the_generation_date(tmp_path: Path) -> None:
    """Nearest to now, not the last step. The store here runs past the generation date, the
    way a forecast does, so the two answers differ and the last step would be wrong."""
    # Steps at 60, 30 and 1 days before, and 30 days after, the pinned generation date.
    cube = _daily_cube(4, start="2025-12-31")
    cube = cube.assign_coords(
        t=pd.to_datetime(["2025-12-31", "2026-01-30", "2026-02-28", "2026-03-31"]),
    )
    store = _store(tmp_path, cube)

    written = write_dataset_thumbnail(store, DATASET, now=GENERATED_AT)

    assert written is not None
    # 2026-02-28 is one day before the generation date; 2026-03-31 is thirty after.
    assert written.read_bytes() == _rendered_reference(cube["precip"].isel(t=2), tmp_path)


def test_a_climatology_renders_its_first_slice(tmp_path: Path) -> None:
    """A climatology's axis is an ordinal dayofyear, not a datetime, so there is no "nearest
    to today" without inventing a mapping. Index 0 is the answer, and it does not drift."""
    doy = xr.Dataset(
        {"precip": (("dayofyear", "y", "x"), np.array(_PATTERNS[:3], dtype="float32"))},
        coords={"dayofyear": [1, 2, 3], "y": [1.5, 0.5], "x": [10.5, 11.5]},
    )
    store = _store(tmp_path, doy, t_dim=None)

    written = write_dataset_thumbnail(store, DATASET, now=GENERATED_AT)

    assert written is not None
    assert written.read_bytes() == _rendered_reference(doy["precip"].isel(dayofyear=0), tmp_path)


def test_a_store_with_no_non_spatial_axis_renders_its_grid(tmp_path: Path) -> None:
    flat = xr.Dataset(
        {"precip": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )
    store = _store(tmp_path, flat, t_dim=None)

    written = write_dataset_thumbnail(store, DATASET, now=GENERATED_AT)

    assert written is not None
    assert written.read_bytes() == _rendered_reference(flat["precip"], tmp_path)


def test_the_nearest_step_is_chosen_by_dtype_not_by_a_declared_period_type(tmp_path: Path) -> None:
    """An ordinal axis of plain integers resolves to index 0 even when the numbers look like
    years. The coordinate's dtype cannot disagree with the data; a declared period_type can."""
    ordinal = xr.DataArray(
        np.array([[[1.0]], [[2.0]]], dtype="float32"),
        dims=("year", "y", "x"),
        coords={"year": [2020, 2026], "y": [0.5], "x": [1.5]},
    )

    assert representative_slice(ordinal, now=GENERATED_AT).item() == 1.0


# -- size --------------------------------------------------------------------------------


def _rendered_size(path: Path) -> tuple[int, int]:
    from matplotlib import image as mpimg

    height, width = mpimg.imread(path).shape[:2]
    return height, width


def test_a_store_larger_than_the_target_is_scaled_down(tmp_path: Path) -> None:
    """STAC best practice for the `thumbnail` role is under 600x600, and a store is routinely
    far larger, so the size has to be applied rather than assumed."""
    big = xr.Dataset(
        {"precip": (("y", "x"), np.random.default_rng(0).random((1200, 800), dtype="float32"))},
        coords={"y": np.linspace(10.0, 0.0, 1200), "x": np.linspace(0.0, 8.0, 800)},
    )

    written = write_dataset_thumbnail(_store(tmp_path, big, t_dim=None), DATASET, now=GENERATED_AT)

    assert written is not None
    # The aspect ratio survives: 1200x800 scaled by 512/1200 is 512x341.
    assert _rendered_size(written) == (THUMBNAIL_LONG_SIDE_PIXELS, 341)


def test_a_store_coarser_than_the_target_is_scaled_up(tmp_path: Path) -> None:
    """Both directions. A 32x16 forecast grid left at its own size is a postage stamp in any
    client that does not scale it, and mush in any client that does — we control the CSS in
    neither STAC Browser nor a DHIS2 app, so the resolution is decided here instead."""
    coarse = xr.Dataset(
        {"precip": (("y", "x"), np.random.default_rng(0).random((16, 32), dtype="float32"))},
        coords={"y": np.linspace(10.0, 0.0, 16), "x": np.linspace(0.0, 20.0, 32)},
    )

    written = write_dataset_thumbnail(_store(tmp_path, coarse, t_dim=None), DATASET, now=GENERATED_AT)

    assert written is not None
    # 32x16 enlarged by 512/32 is 512x256, and the aspect ratio is unchanged.
    assert _rendered_size(written) == (256, THUMBNAIL_LONG_SIDE_PIXELS)


def test_an_enlarged_thumbnail_keeps_its_cell_boundaries(tmp_path: Path) -> None:
    """Nearest-neighbour, not a smooth blow-up: a coarse dataset should look coarse. A 2x2
    field enlarged 256x either has four flat quadrants or it has been interpolated."""
    tiny = xr.Dataset(
        {"precip": (("y", "x"), np.array([[0.0, 1.0], [2.0, 3.0]], dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )

    written = write_dataset_thumbnail(_store(tmp_path, tiny, t_dim=None), DATASET, now=GENERATED_AT)

    assert written is not None
    from matplotlib import image as mpimg

    image = mpimg.imread(written)
    half = THUMBNAIL_LONG_SIDE_PIXELS // 2
    # Each quadrant is one source cell, so it is a single colour throughout.
    for row, col in ((0, 0), (0, half), (half, 0), (half, half)):
        quadrant = image[row + 10 : row + half - 10, col + 10 : col + half - 10]
        assert np.allclose(quadrant, quadrant[0, 0]), "an enlarged cell was interpolated"


# -- failure is not an ingest failure -----------------------------------------------------


def test_a_failing_render_does_not_fail_the_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset is not less ingested for being unrecognisable. The callers rely on this
    function never raising rather than each wrapping it, so the guarantee is tested here."""
    store = _store(tmp_path, _daily_cube(2))

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no renderer today")

    monkeypatch.setattr(thumbnails, "render_png", explode)

    assert write_dataset_thumbnail(store, DATASET, now=GENERATED_AT) is None
    assert not thumbnail_path("thumb_dataset").exists()


def test_an_unreadable_store_does_not_fail_the_ingest(tmp_path: Path) -> None:
    assert write_dataset_thumbnail(tmp_path / "not-a-store.icechunk", DATASET) is None


# -- colormaps ----------------------------------------------------------------------------


def test_every_built_in_template_colormap_resolves_to_itself() -> None:
    """The whole-catalogue guard. Template colormap names are written for the viewer, which
    matches case-insensitively; matplotlib does not, so `blues`, `rdbu_r` and `reds` all
    raise when passed straight to it — 18 of the 27 built-in templates. Falling back to the
    default would render every one of them in the wrong colours while still "working"."""
    from open_climate_service.data_registry.services import datasets as registry

    declared = {
        str(template["display"]["colormap"])
        for template in registry.list_datasets()
        if isinstance(template.get("display"), dict) and template["display"].get("colormap")
    }
    assert declared, "no template declares a colormap"

    unresolved = {name for name in declared if resolve_colormap(name).name.lower() != name.lower()}
    assert not unresolved, f"colormaps that fell back to the default instead of resolving: {unresolved}"


def test_an_unknown_colormap_falls_back_instead_of_raising() -> None:
    assert resolve_colormap("not-a-colormap").name == "viridis"
    assert resolve_colormap(None).name == "viridis"


def test_a_low_signal_slice_still_shows_its_structure(tmp_path: Path) -> None:
    """The case that decided per-slice scaling. CHIRPS daily declares a 0-20 mm display range,
    and 31 January 2025 over Nepal peaks at 0.408 mm — 2% of it — so against the declared range
    the whole frame renders as the palest end of the colormap and shows nothing at all."""
    faint = xr.Dataset(
        {"precip": (("y", "x"), np.array([[0.0, 0.05], [0.2, 0.408]], dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )
    store = _store(tmp_path, faint, t_dim=None)

    written = write_dataset_thumbnail(store, DATASET, now=GENERATED_AT)

    assert written is not None
    # Against the declared 0-20 range every cell would land in the same colour; scaled to the
    # slice they separate.
    assert written.read_bytes() != _rendered_declared_range(faint["precip"], tmp_path)


def test_a_constant_slice_renders_rather_than_dividing_by_zero(tmp_path: Path) -> None:
    """A flat field has no range to stretch. It should come out one colour, not raise."""
    flat = xr.Dataset(
        {"precip": (("y", "x"), np.full((2, 2), 3.0, dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )
    low, high = stretch_range(flat["precip"].values) or (0.0, 0.0)
    assert low < high

    assert write_dataset_thumbnail(_store(tmp_path, flat, t_dim=None), DATASET) is not None


def test_an_all_missing_slice_publishes_without_a_thumbnail(tmp_path: Path) -> None:
    """Nothing to show is not the same as a failure, and neither is an ingest problem."""
    empty = xr.Dataset(
        {"precip": (("y", "x"), np.full((2, 2), np.nan, dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )

    assert stretch_range(empty["precip"].values) is None
    assert write_dataset_thumbnail(_store(tmp_path, empty, t_dim=None), DATASET) is None


def test_the_stretch_resists_a_single_outlier() -> None:
    """Raw min/max would hand the whole scale to one storm cell and flatten the rest, which is
    the problem a per-slice stretch exists to avoid — hence percentiles rather than extremes.

    The field has real structure under the outlier, which is what makes the difference
    visible: a field that is 99% one value has nothing for either rule to preserve, and there
    the percentiles collapse and the extremes are used deliberately.
    """
    field = np.linspace(0.0, 10.0, 100).astype("float32")
    field[0] = 500.0

    low, high = stretch_range(field) or (0.0, 0.0)

    assert high < 11.0, f"one outlier took the whole range: {(low, high)}"
    assert (low, high) != (float(field.min()), float(field.max()))


@pytest.mark.parametrize(
    "declared,midpoint",
    [
        ([-5.0, 5.0], 0.0),  # temperature anomaly
        ([-30.0, 30.0], 0.0),  # temperature in Celsius, where zero is freezing
        ([-3, 3], 0.0),  # SPI
        ([0.0, 20.0], None),  # precipitation
        ([0, 4000], None),  # elevation, which pairs Spectral_r with a one-sided range
        ([-0.1, 1.0], None),  # NDVI: negative, but not symmetric
        ([-1.0, 1.5], None),
        (None, None),
        ("nonsense", None),
    ],
)
def test_only_a_zero_symmetric_declared_range_names_a_midpoint(declared: Any, midpoint: float | None) -> None:
    """The signal for "this quantity diverges about zero" is the declared range, not the
    colormap: `copernicus_dem_elevation` uses the diverging Spectral_r over [0, 4000] and must
    not be centred, and across the shipped templates the split by range is exact."""
    assert declared_midpoint(declared) == midpoint


def test_a_diverging_scale_keeps_zero_at_its_midpoint(tmp_path: Path) -> None:
    """Blue means below normal. Stretched to its own extremes, an anomaly slice that happens
    to be entirely positive would still render half blue and invert that meaning."""
    all_positive = np.linspace(0.5, 4.0, 100).astype("float32")

    low, high = stretch_range(all_positive, midpoint=0.0) or (0.0, 0.0)

    assert low == -high, "the range is not symmetric about zero"
    assert low < 0.0 < high
    # Every value sits in the upper half, so nothing renders on the "below" side of the scale.
    assert float(all_positive.min()) > 0.0


def test_a_sequential_scale_is_not_centred(tmp_path: Path) -> None:
    """Centring a one-sided quantity would throw away half the colour scale on values that
    cannot occur — no rainfall is below zero."""
    rain = np.linspace(0.0, 4.0, 100).astype("float32")

    assert stretch_range(rain, midpoint=None) != stretch_range(rain, midpoint=0.0)
    low, _ = stretch_range(rain, midpoint=None) or (0.0, 0.0)
    assert low >= 0.0


def test_an_anomaly_store_renders_centred_on_zero(tmp_path: Path) -> None:
    """Through the entry point, since the midpoint has to be read off the template's declared
    range and reach the render — the wiring is the part that can silently not happen."""
    anomaly = xr.Dataset(
        {"precip": (("y", "x"), np.array([[0.5, 1.0], [2.0, 4.0]], dtype="float32"))},
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5]},
    )
    store = _store(tmp_path, anomaly, t_dim=None)
    diverging = {**DATASET, "display": {"colormap": "rdbu_r", "range": [-5.0, 5.0]}}

    written = write_dataset_thumbnail(store, diverging, now=GENERATED_AT)

    assert written is not None
    expected = render_png(
        anomaly["precip"],
        tmp_path / "centred.png",
        colormap="rdbu_r",
        clim=stretch_range(anomaly["precip"].values, midpoint=0.0),
        long_side=THUMBNAIL_LONG_SIDE_PIXELS,
    ).read_bytes()
    assert written.read_bytes() == expected


# -- when it is generated ------------------------------------------------------------------


def test_a_multi_period_sync_never_reaches_the_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Half of "once per run, at the end". A streaming sync commits per period, so a renderer
    reached from the commit path would run once per period — a few hundred times on a
    historical backfill, all but the last discarded."""
    from open_climate_service.streaming import orchestrator as streaming_orchestrator

    renders: list[Any] = []
    monkeypatch.setattr(thumbnails, "render_png", lambda *a, **k: renders.append(a))

    class _Plugin:
        max_concurrency = 1
        commit_batch_size = 1

        async def periods(self, start: str, end: str) -> list[str]:
            _ = start, end
            return ["2026-01-01", "2026-01-02", "2026-01-03"]

        async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
            _ = bbox, params
            return xr.Dataset(
                {"precip": (("t", "y", "x"), np.array([[[float(period_id[-2:])]]], dtype="float32"))},
                coords={"t": [np.datetime64(period_id, "D")], "y": [0.0], "x": [1.0]},
            )

    store_path = tmp_path / "streaming.zarr"
    monkeypatch.setattr(streaming_orchestrator, "is_store_empty", lambda path: not path.exists())

    result = streaming_orchestrator.run_streaming_ingest_sync(
        plugin=_Plugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-03",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 3
    assert renders == [], "the renderer was reached from the per-commit path"


def test_one_ingest_produces_exactly_one_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half. Driven through `create_artifact`, the entry point, so it covers the
    wiring rather than the helper: a render that never gets called would pass a test of
    `write_dataset_thumbnail` alone."""
    store_path = tmp_path / "thumb_dataset.icechunk"
    dataset: dict[str, object] = {
        "id": "thumb_dataset",
        "name": "Thumb dataset",
        "variable": "precip",
        "period_type": "daily",
        "display": {"colormap": "blues", "range": [0.0, 10.0]},
        "ingestion": {"plugin": "example.Plugin", "params": {}},
    }

    def fake_sync(**kwargs: object) -> object:
        # Stand in for a three-period sync: the store the finalisation sees is the finished one.
        downloader.write_to_icechunk_store(_daily_cube(3), store_path, commit_message="test")
        return SimpleNamespace(periods_written=3)

    renders: list[Any] = []
    real_render = thumbnails.render_png

    def counting_render(*args: Any, **kwargs: Any) -> Any:
        renders.append(args)
        return real_render(*args, **kwargs)

    monkeypatch.setattr(thumbnails, "render_png", counting_render)
    monkeypatch.setattr(ingestion_services, "_load_streaming_plugin", lambda path, *, params: object())
    monkeypatch.setattr(ingestion_services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(ingestion_services, "run_streaming_ingest_sync", fake_sync)
    monkeypatch.setattr(ingestion_services, "_find_existing_artifact", lambda **_: None)
    monkeypatch.setattr(ingestion_services, "_upsert_artifact_record", lambda record, **_: record)
    monkeypatch.setattr(
        ingestion_services,
        "get_data_coverage_for_paths",
        lambda dataset_arg, **_: {
            "coverage": {
                "temporal": {"start": "2026-01-01", "end": "2026-01-03"},
                "spatial": {"xmin": 10.0, "ymin": 0.0, "xmax": 12.0, "ymax": 2.0},
            }
        },
    )

    ingestion_services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[10.0, 0.0, 12.0, 2.0],
        country_code=None,
        overwrite=True,
        publish=False,
    )

    assert len(renders) == 1
    assert thumbnail_path("thumb_dataset").is_file()


# -- how it reaches a client ----------------------------------------------------------------


def _published_artifact() -> Any:
    from open_climate_service.ingestions.schemas import (
        ArtifactCoverage,
        ArtifactFormat,
        ArtifactPublication,
        ArtifactRecord,
        ArtifactRequestScope,
        CoverageSpatial,
        CoverageTemporal,
        PublicationStatus,
    )

    return ArtifactRecord(
        artifact_id="a1",
        dataset_id="thumb_dataset",
        dataset_name="Thumb dataset",
        variable="precip",
        period_type="daily",
        format=ArtifactFormat.ICECHUNK,
        path="/tmp/thumb_dataset.icechunk",
        asset_paths=["/tmp/thumb_dataset.icechunk"],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-03"),
        coverage=ArtifactCoverage(
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-03"),
            spatial=CoverageSpatial(xmin=10.0, ymin=0.0, xmax=12.0, ymax=2.0),
        ),
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        publication=ArtifactPublication(
            status=PublicationStatus.PUBLISHED,
            collection_id="thumb_dataset",
            published_at=datetime(2026, 1, 3, tzinfo=UTC),
        ),
    )


@pytest.fixture
def _published(monkeypatch: pytest.MonkeyPatch) -> None:
    """One published artifact, with the store-reading parts of the build stubbed.

    The assertions here are about the assets a collection advertises, not about the cube
    metadata xstac derives, so opening a real Icechunk store would only add a fixture to
    maintain. `_minimal_collection` mirrors what xstac returns.
    """
    from open_climate_service.stac import services as stac_services

    stac_services._clear_xstac_collection_cache()
    monkeypatch.setattr(ingestion_services, "list_artifacts", lambda: SimpleNamespace(items=[_published_artifact()]))
    monkeypatch.setattr(stac_services.registry_datasets, "get_dataset", lambda _: {"period_type": "daily"})
    monkeypatch.setattr(
        stac_services,
        "_build_collection_with_xstac",
        lambda **_: {
            "type": "Collection",
            "id": "thumb_dataset",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "cube:dimensions": {"time": {"type": "temporal", "extent": ["2026-01-01", "2026-01-03"]}},
            "cube:variables": {"precip": {"type": "data", "dimensions": ["time", "y", "x"]}},
            "assets": {"zarr": {}},
        },
    )
    monkeypatch.setattr(stac_services, "_zarr_asset_metadata", lambda _: {})


def _write_a_thumbnail() -> Path:
    path = thumbnail_path("thumb_dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake but on disk")
    return path


def test_the_collection_advertises_the_thumbnail_as_a_stac_role(client: TestClient, _published: None) -> None:
    """`thumbnail` is a standardised STAC asset role on a Collection, so no extension is
    needed and a STAC client picks the image up unaided."""
    _write_a_thumbnail()

    payload = client.get("/stac/collections/thumb_dataset").json()

    asset = payload["assets"]["thumbnail"]
    assert asset["roles"] == ["thumbnail"]
    assert asset["type"] == "image/png"
    assert asset["href"].endswith("/datasets/thumb_dataset/thumbnail.png")
    # The href leaves the process, so it names an origin rather than being a bare path.
    assert asset["href"].startswith("http")


def test_the_collection_omits_the_thumbnail_when_there_is_none(client: TestClient, _published: None) -> None:
    """Advertising an asset a client then 404s on is worse than advertising none, and absence
    is normal: thumbnails are not backfilled, so a store never rewritten never gains one."""
    payload = client.get("/stac/collections/thumb_dataset").json()

    assert "thumbnail" not in payload["assets"]
    assert "zarr" in payload["assets"]


def test_the_route_serves_the_thumbnail(client: TestClient) -> None:
    written = _write_a_thumbnail()

    response = client.get("/datasets/thumb_dataset/thumbnail.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == written.read_bytes()


def test_the_route_404s_when_a_dataset_has_no_thumbnail(client: TestClient) -> None:
    assert client.get("/datasets/thumb_dataset/thumbnail.png").status_code == 404
