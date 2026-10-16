"""The layout contract every published store must satisfy, enforced at the write boundary.

Dimension naming, axis order and CRS consistency are all *ingest-normalisation invariants*:
properties of a published GeoZarr, not of any one source. Enforced per-plugin they get
forgotten one plugin at a time — that is how datasets have shipped on a `time` dim the map
viewer could not find, and with a projected CRS stamped over geographic coordinates. So they
are enforced here instead, once, and plugins may return data in whatever orientation their
source delivers (see CLIM-821).

The contract:

1. the temporal dimension is named ``t`` — or, for a forecast cube, the pair
   ``reference_time``/``lead_time`` stands in for it (see :mod:`open_climate_service.shared.forecast`)
2. spatial dims are named ``y`` / ``x`` and ordered ``(…, y, x)``
3. ``y`` descends — array row 0 is the northernmost row
4. for a geographic CRS, longitudes run −180…180 rather than 0…360
5. the declared CRS matters and matches the coordinates it describes

Invariant 3 exists because real consumers assume a direction and never check. Two of them:

* ``openeo/jobs.py`` renders result thumbnails with ``imshow(..., origin="upper")`` straight off
  the array — row 0 is north, unconditionally.
* OpenLayers' ``ol/source/GeoZarr``, the renderer STAC Browser uses, derives its tile grid
  ``origin`` as ``[bbox[0], bbox[3]]`` (top-left) and computes
  ``minRow = (origin[1] - tileExtent[3]) / resolution``, feeding that straight into the array
  slice. There is no ``reverse``/``flip`` and no read of ``spatial:transform``'s y step anywhere
  in that file, so an ascending store renders vertically mirrored with no signal it could use.

Descending rather than ascending: OpenLayers requires it and cannot detect otherwise, GDAL's
``GeoTransform`` convention for north-up is a negative y step, most sources are north-up already
so the normalisation is usually a no-op — and carbonplan/zarr-layer, the one consumer that *does*
detect the direction from the coordinate array, is satisfied either way and so does not constrain
the choice.

Axis order here is *array* order, ``(…, y, x)`` — NumPy row-major, CF's ``T,Z,Y,X``, and what
xarray/rioxarray/GDAL/GeoZarr all assume. It does not conflict with GeoJSON's ``[x, y]``
coordinate pairs and ``[west, south, east, north]`` bboxes, which stay as RFC 7946 requires:
those describe coordinate tuples, not an N-D array. See docs/conventions.md.

Invariants 3 and 4 reorder the array. Applying either to a period being appended to a store
whose axes were written the other way round would put rows or columns under coordinates that no
longer describe them — silent mirrored data, which is worse than stale metadata. So an append
must first check the store it is appending to; see
:func:`spatial_coords_match` and the orchestrator's use of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# Source spellings for the temporal dimension, mapped onto the canonical ``t``. The map
# viewer looks for `t`; a store that ships `time` renders without a time control.
_TIME_ALIASES = ("time", "valid_time", "date", "time_counter")

# Beyond this longitude a geographic x axis is on the 0…360 frame rather than −180…180.
_LON_ROLL_THRESHOLD = 180.0


@dataclass(frozen=True)
class PreparedCube:
    """A cube normalised for publication, with the CRS that describes it."""

    dataset: "xr.Dataset"
    crs: str


def prepare_for_publication(
    ds: "xr.Dataset",
    *,
    fallback_crs: str | int | None = None,
    time_dim: str = "t",
    x_dim: str = "x",
    y_dim: str = "y",
) -> PreparedCube:
    """Bring *ds* up to the published-store contract, and resolve the CRS that describes it.

    **The entry point every write path should use.** The individual ``normalize_*`` steps below
    are exported for focused testing, but composing them by hand at each call site is how the
    two write paths came to run them in different orders with different CRS inputs — the drift
    this module exists to prevent. The order here is not arbitrary:

    * naming and axis order first, because everything after addresses axes by name;
    * then the CRS, because whether the x axis is a longitude decides whether rolling it is
      correct or destructive — an easting of 500000 rolled as a longitude becomes −40;
    * then the coordinate reorderings, which need that answer.

    ``fallback_crs`` is a *declaration* about a source that carries no CRS of its own (a
    plugin's ``crs`` attribute, or an explicit argument); the data's own CRS still wins. See
    :func:`resolve_store_crs`.
    """
    ds = normalize_dim_layout(ds, time_dim=time_dim, x_dim=x_dim, y_dim=y_dim)
    crs = resolve_store_crs(ds, fallback_crs, x_dim=x_dim, y_dim=y_dim)
    ds = normalize_longitudes(ds, x_dim=x_dim, crs=crs)
    ds = normalize_y_direction(ds, y_dim=y_dim)

    # Normalisation should leave nothing behind. If it did, something upstream is producing a
    # shape this module does not understand, and the store is about to be published that way —
    # so say so in the ingest's own logs rather than waiting for a client to render it oddly.
    ds.attrs.setdefault("proj:code", crs)
    remaining = published_contract_violations(ds, time_dim=time_dim, x_dim=x_dim, y_dim=y_dim)
    if remaining:
        logger.warning(
            "Publishing a store that still violates the layout contract: %s",
            "; ".join(str(violation) for violation in remaining),
        )
    return PreparedCube(dataset=ds, crs=crs)


def normalize_dim_layout(
    ds: "xr.Dataset",
    *,
    time_dim: str = "t",
    x_dim: str = "x",
    y_dim: str = "y",
) -> "xr.Dataset":
    """Rename the temporal dimension to ``time_dim`` and order spatial dims ``(…, y, x)``.

    Renames only coordinate/dimension *names* and axis order — no coordinate value changes —
    so this is safe to apply to a single period being appended to an existing store as well
    as to a whole-store rewrite. Longitude rolling, which does change values, is separate
    (:func:`normalize_longitudes`).

    A no-op when the dataset already conforms, so the common case costs one dims comparison
    rather than a copy of the data.
    """
    if time_dim not in ds.dims:
        for alias in _TIME_ALIASES:
            if alias in ds.dims or alias in ds.coords:
                ds = ds.rename({alias: time_dim})
                logger.info("Renamed temporal dimension %r to %r", alias, time_dim)
                break

    if x_dim not in ds.dims or y_dim not in ds.dims:
        # Nothing to order — the caller (orchestrator or downloader) raises a clearer error
        # about the missing spatial dims than a transpose would.
        return ds

    for name, var in ds.data_vars.items():
        dims = tuple(str(d) for d in var.dims)
        if y_dim not in dims or x_dim not in dims:
            continue  # a non-spatial variable, e.g. a scalar CRS holder
        if dims[-2:] == (y_dim, x_dim):
            continue
        leading = [d for d in dims if d not in (y_dim, x_dim)]
        ds[name] = var.transpose(*leading, y_dim, x_dim)
        logger.info(
            "Transposed %r from %s to (…, %s, %s) — array order is row-major",
            str(name),
            dims,
            y_dim,
            x_dim,
        )
    return ds


def normalize_y_direction(ds: "xr.Dataset", *, y_dim: str = "y") -> "xr.Dataset":
    """Reverse the rows of a south-up dataset so ``y`` descends (row 0 = north).

    Consumers assume this and do not check — see the module docstring for the two that would
    otherwise render mirrored. Reverses the coordinate *and* the data together, so the value at
    a given latitude does not move; only its row index does.

    A no-op for the north-up rasters most sources deliver, so this usually costs one comparison.
    """
    if y_dim not in ds.coords or ds.sizes.get(y_dim, 0) < 2:
        return ds
    y = ds[y_dim].values
    try:
        already_descending = bool(y[1] < y[0])
    except (IndexError, TypeError, ValueError):
        return ds
    if already_descending:
        return ds
    logger.info("Reversing %r so row 0 is the northernmost row (was south-up)", y_dim)
    return ds.isel({y_dim: slice(None, None, -1)})


def normalize_longitudes(ds: "xr.Dataset", *, x_dim: str = "x", crs: str | None = None) -> "xr.Dataset":
    """Roll a geographic x axis from 0…360 to −180…180 and sort it ascending.

    Only for a geographic CRS — an easting in a projected CRS is legitimately larger than 180
    and must not be touched.

    A no-op when no longitude exceeds 180, which is the case for every source that already
    publishes on the −180…180 frame.
    """
    if x_dim not in ds.coords:
        return ds
    if crs is not None and _axis_units(crs) == "linear":
        return ds

    x = ds[x_dim]
    try:
        needs_roll = bool((x > _LON_ROLL_THRESHOLD).any())
    except (TypeError, ValueError):
        return ds
    if not needs_roll:
        return ds

    logger.info("Rolling %r from the 0…360 frame to −180…180", x_dim)
    rolled = ds.assign_coords({x_dim: ((ds[x_dim] + 180.0) % 360.0) - 180.0})
    return rolled.sortby(x_dim)


def _axis_units(crs: str | int) -> str | None:
    """``"degrees"`` for a lon/lat CRS, ``"linear"`` for a projected one, ``None`` if unknown.

    The three-way answer matters: a CRS that pyproj cannot resolve — a polluted host PROJ
    database is a common developer-machine failure, see ``startup._ensure_proj_database`` — must
    come back as *unknown* rather than being lumped in with projected. Treating an unresolvable
    CRS as projected would let a perfectly good geographic one (ETRS89, say) be overridden with
    EPSG:4326 purely because the database was unreadable.
    """
    from open_climate_service.shared.crs import canonical_crs_code

    code = canonical_crs_code(crs)
    if code == "EPSG:4326":
        return "degrees"
    try:
        from pyproj import CRS

        return "degrees" if CRS.from_user_input(code).is_geographic else "linear"
    except Exception:  # noqa: BLE001 — unresolvable: report unknown, never guess "projected"
        logger.debug("Could not resolve CRS %s to decide its axis units", code, exc_info=True)
        return None


def _looks_geographic(ds: "xr.Dataset", *, x_dim: str, y_dim: str) -> bool:
    """True when the coordinate magnitudes can only be degrees.

    Latitude carries the argument: it fits ±90, where a projected northing runs to millions
    within a few hundred metres of the origin. Longitude is allowed up to 360 rather than 180
    because a grid still on the 0…360 frame has not been rolled yet — testing it against 180
    would make this return False for exactly the datasets that need rolling, and the roll needs
    this answer to know the axis is a longitude at all.

    Deliberately one-directional: this answers "these cannot be metres", never "these cannot be
    degrees". A projected grid confined to a few hundred metres either side of its own origin
    would pass, but no real dataset extent is that small.
    """
    try:
        x = ds[x_dim].values
        y = ds[y_dim].values
        return bool(x.min() >= -180.0 and x.max() <= 360.0 and abs(y).max() <= 90.0)
    except (KeyError, ValueError, TypeError):
        return False


def resolve_store_crs(
    ds: "xr.Dataset",
    fallback: str | int | None = None,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
) -> str:
    """The CRS to write for *ds*, in one decision.

    Precedence, highest first:

    1. the dataset's own CRS — its ``proj:code`` attribute, else what rioxarray detects from
       the CF grid mapping
    2. ``fallback`` — how a plugin declares the projection of a source that carries none
    3. ``EPSG:4326``

    The data wins over a declaration because a declaration is a *guess about* the data, and
    getting that precedence backwards is how a projected code ends up over lon/lat coordinates.

    Whichever wins is then checked against the coordinates: a CRS in linear units over
    coordinates that can only be degrees is wrong however it was arrived at, and is replaced by
    plain WGS84 with the evidence logged. A stale ``proj:code`` is copied forward on every
    rewrite, so *both* sources can be wrong at once — Norway's WorldPop store has EPSG:32633 at
    the root and EPSG:4326 on level 0.

    The instance config CRS is never consulted, at any step. Using it as a fallback is what
    created the contradiction this function exists to catch.
    """
    from open_climate_service.shared.crs import canonical_crs_code, dataset_crs

    default = canonical_crs_code(fallback) if fallback is not None else "EPSG:4326"
    code = canonical_crs_code(dataset_crs(ds, default=default))

    # Override only on a definite contradiction: the CRS is known to be in linear units and the
    # coordinates can only be degrees. An unresolvable CRS is left alone rather than guessed at.
    if _axis_units(code) != "linear" or not _looks_geographic(ds, x_dim=x_dim, y_dim=y_dim):
        return code

    logger.warning(
        "CRS %s is projected but the coordinates are within ±180/±90 (x %.4f…%.4f, y %.4f…%.4f) "
        "— they can only be degrees. Using EPSG:4326 instead; a projected code over geographic "
        "coordinates puts the store at the projection's origin rather than on the map.",
        code,
        float(ds[x_dim].min()),
        float(ds[x_dim].max()),
        float(ds[y_dim].min()),
        float(ds[y_dim].max()),
    )
    return "EPSG:4326"


def spatial_coords_match(
    stored: "Any",
    incoming: "Any",
    *,
    axis: str,
    tolerance: float = 1e-9,
) -> bool:
    """True when an incoming period's axis coordinates match the ones already in the store.

    An append writes along the time axis only: the spatial coordinate arrays already committed
    stay as they are. So if normalisation reorders a period's rows or columns relative to how the
    store was written — a store created before invariant 3 or 4, or by a plugin that has since
    changed orientation — the new data lands under coordinates that no longer describe it. That
    is silently mirrored data, which is why this is checked rather than assumed.

    Compares elementwise with a tolerance, since a coordinate can survive a float round trip
    through Zarr with a last-bit difference.
    """
    import numpy as np

    try:
        a = np.asarray(getattr(stored, "values", stored), dtype="float64")
        b = np.asarray(getattr(incoming, "values", incoming), dtype="float64")
    except (TypeError, ValueError):
        logger.debug("Could not compare %s coordinates; skipping the check", axis)
        return True
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True))


class ContractViolation(StrEnum):
    """A way a dataset fails the published-store contract."""

    TEMPORAL_DIM_NAME = "temporal-dim-name"
    MISSING_SPATIAL_DIM = "missing-spatial-dim"
    AXIS_ORDER = "axis-order"
    Y_ASCENDS = "y-ascends"
    CRS_CONTRADICTS_COORDS = "crs-contradicts-coords"
    LONGITUDE_FRAME = "longitude-frame"


@dataclass(frozen=True)
class Violation:
    """One contract failure: a machine-checkable kind plus the detail behind it."""

    kind: ContractViolation
    detail: str

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.detail}"


def published_contract_violations(
    ds: "xr.Dataset",
    *,
    time_dim: str = "t",
    x_dim: str = "x",
    y_dim: str = "y",
) -> list[Violation]:
    """Ways *ds* fails the published-store contract; empty means conformant.

    Called at the write boundary so a regression shows up in the logs of a real ingest rather
    than only in a test run, and available to anyone debugging a store that renders oddly.
    Each violation carries a ``kind`` so callers can match on it without depending on wording.
    """
    from . import forecast

    problems: list[Violation] = []
    dims = set(str(d) for d in ds.dims)
    # A forecast cube satisfies invariant 1 with two axes rather than one: it has no `t` by
    # design, because no single axis means "the period this value describes" (see shared/forecast).
    if forecast.is_forecast_cube(ds):
        time_dim = forecast.REFERENCE_DIM
    if time_dim not in dims:
        found = [a for a in _TIME_ALIASES if a in dims]
        problems.append(
            Violation(
                ContractViolation.TEMPORAL_DIM_NAME,
                f"temporal dimension is not named {time_dim!r}" + (f" (found {found[0]!r})" if found else ""),
            )
        )
    for expected in (y_dim, x_dim):
        if expected not in dims:
            problems.append(
                Violation(ContractViolation.MISSING_SPATIAL_DIM, f"spatial dimension {expected!r} is missing")
            )
    for name, var in ds.data_vars.items():
        var_dims = tuple(str(d) for d in var.dims)
        if y_dim in var_dims and x_dim in var_dims and var_dims[-2:] != (y_dim, x_dim):
            problems.append(
                Violation(
                    ContractViolation.AXIS_ORDER,
                    f"variable {str(name)!r} has dims {var_dims}, not (…, {y_dim}, {x_dim})",
                )
            )
    if y_dim in ds.coords and ds.sizes.get(y_dim, 0) >= 2:
        y = ds[y_dim].values
        try:
            if bool(y[1] > y[0]):
                problems.append(
                    Violation(
                        ContractViolation.Y_ASCENDS,
                        f"{y_dim!r} ascends — row 0 must be the northernmost row",
                    )
                )
        except (IndexError, TypeError, ValueError):
            pass
    declared: Any = ds.attrs.get("proj:code")
    units = _axis_units(str(declared)) if declared else None
    if declared and units == "linear" and _looks_geographic(ds, x_dim=x_dim, y_dim=y_dim):
        problems.append(
            Violation(
                ContractViolation.CRS_CONTRADICTS_COORDS,
                f"declared CRS {declared} is projected but the coordinates are degrees",
            )
        )
    if declared and units == "degrees" and x_dim in ds.coords:
        try:
            if bool((ds[x_dim] > _LON_ROLL_THRESHOLD).any()):
                problems.append(
                    Violation(
                        ContractViolation.LONGITUDE_FRAME,
                        f"geographic {x_dim!r} exceeds 180 — still on the 0…360 frame",
                    )
                )
        except (TypeError, ValueError):
            pass
    return problems
