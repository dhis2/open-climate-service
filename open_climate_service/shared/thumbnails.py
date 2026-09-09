"""Render a dataset as a small PNG, for STAC ``thumbnail`` assets and openEO PNG results.

A STAC collection is metadata and a Zarr href, which is nothing to recognise a layer by:
an ingest that produced a flipped grid, a wrong extent or a unit error looks exactly like
one that did not. A thumbnail is what makes the difference visible, in a STAC browser, on
the landing page and during ingest QA.

Both callers render the same way and differ only in what they are for, so the difference is
one parameter rather than two renderers:

* an openEO ``PNG`` result is a *data product* — it keeps the cube's own pixel dimensions;
* a STAC thumbnail is an *icon* — its longest side is rendered at
  :data:`THUMBNAIL_LONG_SIDE_PIXELS` whatever the store's resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from open_climate_service import config as api_config

# The longest side is rendered at this size and the other follows, so the aspect ratio is
# never distorted. Every thumbnail therefore has the same longest side whatever the store's
# resolution, which is what makes a gallery of them line up.
#
# Both directions, not just down. Pixel dimensions do not limit how large a client displays
# the file — CSS does — but we do not control the CSS in STAC Browser or a DHIS2 app, and
# their default smoothing turns a coarse grid into mush when it is blown up from 32x16.
# Upscaling here with nearest-neighbour keeps the cell boundaries crisp, which is also the
# honest picture of a coarse dataset, and costs a couple of KB: flat blocks compress well.
#
# 512 rather than 600: STAC best practice for the role is "less than 600x600 pixels", and a
# square store rendered at an exact 600 long side would sit on that boundary rather than
# inside it.
THUMBNAIL_LONG_SIDE_PIXELS = 512

_DEFAULT_COLORMAP = "viridis"


def thumbnails_dir() -> Path:
    """Directory holding published thumbnails, one per dataset id."""
    return api_config.get_data_root() / "thumbnails"


def thumbnail_path(dataset_id: str) -> Path:
    """On-disk location of *dataset_id*'s thumbnail, whether or not it exists yet."""
    return thumbnails_dir() / f"{dataset_id}.png"


def resolve_colormap(name: str | None) -> Any:
    """A matplotlib colormap for a template's ``display.colormap``, or the default.

    Template colormap names are written for the map viewer, which resolves them through
    chroma-js — ColorBrewer/D3 names, matched **case-insensitively**, with matplotlib's
    ``_r`` reversed suffix. matplotlib is case-*sensitive*, so passing those names straight
    to it raises: of the 27 built-in templates, only the 9 declaring ``RdBu`` would render,
    while ``blues`` (9), ``rdbu_r`` (7) and ``reds`` (2) all fail. Matching the viewer's
    case-insensitive rule is what makes one declared colormap mean the same thing in the
    viewer and in the image.

    An unresolvable name falls back to the default rather than raising. A thumbnail in the
    wrong colours is worth more than no thumbnail, and this must never be able to fail a
    render on its own.
    """
    from matplotlib import colormaps

    requested = (name or "").strip()
    if requested:
        try:
            return colormaps[requested]
        except KeyError:
            pass
        matched = {registered.lower(): registered for registered in colormaps}.get(requested.lower())
        if matched is not None:
            return colormaps[matched]
    return colormaps[_DEFAULT_COLORMAP]


def representative_index(values: Any, *, now: datetime | None = None) -> int:
    """Index of the slice a thumbnail should show along one non-spatial axis.

    A **datetime** axis resolves to the step nearest *now*. Not the last step: for an
    observed store those coincide, but a forecast store runs into the future, and its final
    step is a lead time nobody recognises the dataset by while the step nearest now is the
    one a person is looking at.

    Any **other** axis resolves to the first step. A climatology's axis is an ordinal
    ``dayofyear`` (1..366) or ``month`` (1..12) rather than a datetime, so a nearest-to-today
    rule there would need a date-to-ordinal mapping, a leap-day decision for 366 and an
    answer to whose timezone "today" is in. Index 0 needs none of that, and the thumbnail is
    then stable rather than drifting through the year. The same answer is right for the
    genuinely categorical axes (sex, age band) for want of anything better.

    The choice keys off the coordinate's **dtype** rather than the dataset's declared
    ``period_type``, because the dtype cannot disagree with the data it describes.
    """
    import numpy as np

    array = np.asarray(getattr(values, "values", values))
    if array.size == 0 or not np.issubdtype(array.dtype, np.datetime64):
        return 0
    reference = now if now is not None else datetime.now(UTC)
    # numpy rejects a tz-aware datetime; published stores hold naive UTC stamps.
    target = np.datetime64(reference.replace(tzinfo=None), "ns")
    return int(np.abs(array.astype("datetime64[ns]") - target).argmin())


def representative_slice(arr: Any, *, now: datetime | None = None) -> Any:
    """Reduce a DataArray to the 2-D slice a thumbnail should show.

    Published stores are ``(non-spatial..., y, x)``, so the leading dimensions are the ones
    to pin; each is resolved by :func:`representative_index`. A store with no non-spatial
    axis at all is returned unchanged.
    """
    while arr.ndim > 2:
        dim = str(arr.dims[0])
        arr = arr.isel({dim: representative_index(arr.coords.get(dim), now=now)})
    return arr


def render_png(
    arr: Any,
    path: str | Path,
    *,
    colormap: str | None = None,
    clim: tuple[float, float] | None = None,
    long_side: int | None = None,
) -> Path:
    """Render a 2-D DataArray to a styled PNG at *path*, and return that path.

    ``clim`` is the value range the colours span; without one the slice's own min/max are
    used, which is right for a one-off render and wrong for comparing two of them, so
    callers that have a declared display range should pass it.

    ``long_side`` renders the longest side at exactly that many pixels, scaling up as well as
    down, with the other side following so the aspect ratio holds. Left as None, the figure is
    sized from the data at 150 dpi with a minimum of 4x3 inches and a tight bounding box — the
    openEO ``PNG`` result behaviour, kept as it was.
    """
    import matplotlib

    matplotlib.use("agg")  # non-interactive backend — safe on worker threads
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    data = np.asarray(arr.values).astype(float)
    if data.ndim != 2:
        raise ValueError(f"A thumbnail needs a 2-D slice, got {data.ndim} dimensions {arr.dims!r}")

    # `origin="upper"` puts array row 0 at the top, which is right because published stores
    # guarantee y descending (row 0 = north) — see shared/raster_contract. A cube that
    # reaches here south-up (an in-flight openEO result, not a published store) is flipped
    # first, so the image is never upside down.
    y_name = next((str(d) for d in arr.dims if str(d) in ("y", "lat", "latitude")), None)
    if y_name is not None and y_name in arr.coords and arr.sizes.get(y_name, 0) >= 2:
        y_values = arr[y_name].values
        if float(y_values[1]) > float(y_values[0]):
            data = data[::-1]

    cmap = resolve_colormap(colormap).copy()
    cmap.set_bad(alpha=0)  # NaN → transparent
    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

    height, width = data.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if long_side is None:
        dpi = 150
        fig, ax = plt.subplots(figsize=(max(4, width / dpi), max(3, height / dpi)), dpi=dpi)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    else:
        # Exact output dimensions: the axes fill the figure, and no tight bounding box is
        # applied, because trimming would make the final pixel size unpredictable and the
        # size is a promise about the published file rather than about the figure. The scale
        # is not clamped to 1, so a store coarser than the target is enlarged rather than left
        # tiny; `interpolation="nearest"` below keeps the enlargement blocky rather than blurred.
        dpi = 100
        scale = long_side / max(height, width)
        fig, ax = plt.subplots(
            figsize=(max(1, round(width * scale)) / dpi, max(1, round(height * scale)) / dpi), dpi=dpi
        )
        ax.set_position((0.0, 0.0, 1.0, 1.0))
        save_kwargs = {}

    try:
        fig.patch.set_alpha(0)
        ax.imshow(data, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        ax.axis("off")
        if long_side is None:
            fig.tight_layout(pad=0)
        fig.savefig(path, dpi=dpi, transparent=True, pad_inches=0, **save_kwargs)
    finally:
        # Figures are not garbage collected while pyplot holds them, so a render that raises
        # after the figure exists would leak one per ingest.
        plt.close(fig)
    return path


# A percentile rather than the raw extremes, because geophysical fields are skewed: one storm
# cell at 80 mm over a field that is otherwise under 1 mm would push everything else into a
# single colour, which is the very problem a per-slice stretch is meant to solve.
_STRETCH_PERCENTILES = (2.0, 98.0)


def declared_midpoint(declared: Any) -> float | None:
    """Zero when *declared* is a display range symmetric about it, else None.

    A template that declares ``[-5, 5]`` or ``[-30, 30]`` is saying the quantity diverges
    about zero — an anomaly, or a temperature in Celsius where zero is freezing — and pairs
    that range with a diverging colormap. One that declares ``[0, 20]`` or ``[0, 4000]`` is
    saying the opposite. Across the Nepal and Norway instances that split is exact, so the
    declaration is a better signal than a hardcoded list of diverging colormap names: it also
    gets ``copernicus_dem_elevation`` right, which pairs the diverging ``Spectral_r`` with a
    ``[0, 4000]`` range and must *not* be centred.
    """
    if not isinstance(declared, (list, tuple)) or len(declared) != 2:
        return None
    try:
        low, high = float(declared[0]), float(declared[1])
    except (TypeError, ValueError):
        return None
    return 0.0 if low < 0.0 and high == -low else None


def stretch_range(data: Any, *, midpoint: float | None = None) -> tuple[float, float] | None:
    """The value range a thumbnail should span for *data*, or None if there is nothing to show.

    ``midpoint`` keeps a diverging scale honest. Stretched to its own extremes, a slice of an
    anomaly field that happens to be entirely positive would still render half blue, so blue
    would no longer mean "below normal" — the one thing the colour is there to say. Given a
    midpoint the range is made symmetric about it instead, so the neutral colour always lands
    on the neutral value and only the *span* varies with the slice.

    Scaled to the slice rather than to the template's declared ``display.range``. The declared
    range is chosen so a dataset's *layers* are comparable with each other in the viewer, and
    a thumbnail has the opposite job: it has to make one slice recognisable on its own. CHIRPS
    daily precipitation is the case that decided it — 31 January 2025 over Nepal peaks at
    0.408 mm against a declared 0-20 mm range, so 2% of the scale, and the whole frame renders
    as the palest end of the colormap.

    The cost is that two thumbnails no longer share a scale, and neither matches the viewer's
    rendering of the same layer. For an icon whose job is recognition that is the better
    trade; for a value read off the image it would not be, and the image is not for that.

    Degenerate slices are handled rather than left to raise: an all-NaN slice returns None
    (nothing to render), and a constant one is widened so the normalisation cannot divide by
    zero — it renders as a single flat colour, which is what a constant field looks like.
    """
    import numpy as np

    finite = np.asarray(data)[np.isfinite(np.asarray(data))]
    if finite.size == 0:
        return None
    low, high = (float(value) for value in np.percentile(finite, _STRETCH_PERCENTILES))
    if high <= low:
        # The percentiles collapse when most of the field shares one value; the extremes still
        # separate it.
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        low, high = low - 0.5, low + 0.5
    if midpoint is not None:
        half = max(abs(low - midpoint), abs(high - midpoint)) or 0.5
        return midpoint - half, midpoint + half
    return low, high


def write_dataset_thumbnail(
    store_path: str | Path,
    dataset: dict[str, Any],
    *,
    now: datetime | None = None,
) -> Path | None:
    """Render *dataset*'s published store to its thumbnail, returning the path, or None.

    Called once per sync run from the end-of-sync finalisation, not per commit: a streaming
    ingest commits one period at a time, so rendering per commit would produce a few hundred
    PNGs during a historical backfill and keep the last. Rendering here also means rendering
    from the finished store rather than one still being appended to.

    **Never raises.** A store that cannot be previewed publishes without a thumbnail; a
    dataset is not less ingested for being unrecognisable, and the alternative is a render
    bug taking down an ingest that otherwise succeeded. The failure is logged with a
    traceback so it is diagnosable rather than silent.
    """
    import logging

    logger = logging.getLogger(__name__)
    dataset_id = str(dataset.get("id") or "")
    if not dataset_id:
        return None
    try:
        from open_climate_service.data_accessor.services.accessor import open_icechunk_dataset

        ds = open_icechunk_dataset(store_path)
        variable = str(dataset.get("variable") or "")
        if variable not in ds.data_vars:
            variable = str(next(iter(ds.data_vars), ""))
        if not variable:
            logger.warning("No data variable to render a thumbnail from for '%s'", dataset_id)
            return None

        display = dataset.get("display")
        display = display if isinstance(display, dict) else {}
        chosen = representative_slice(ds[variable], now=now)
        clim = stretch_range(chosen.values, midpoint=declared_midpoint(display.get("range")))
        if clim is None:
            logger.warning("Every value in the slice chosen for '%s' is missing; no thumbnail", dataset_id)
            return None
        return render_png(
            chosen,
            thumbnail_path(dataset_id),
            colormap=display.get("colormap"),
            clim=clim,
            long_side=THUMBNAIL_LONG_SIDE_PIXELS,
        )
    except Exception:
        logger.warning("Could not render a thumbnail for '%s'; publishing without one", dataset_id, exc_info=True)
        return None
