"""Built-in openEO process: climate anomaly (observed − climatological normal).

Delegates the day-of-year/month alignment and subtraction to
``earthkit.transforms.climatology.anomaly``, which indexes the normal's ordinal axis
(``dayofyear`` 1..366 or ``month`` 1..12, auto-detected from the climatology) onto the
observed cube's datetime axis and combines them — so anomalies can be computed from an
already-published observed dataset and a published normal (see the ``climate_anomaly``
workflow). ``relative=True`` yields percent-of-normal.

The subtraction, the calendar indexing and the ``relative`` percent form are all earthkit's.
What this module adds is deliberately limited to two kinds of thing, kept separate so the
first shrinks over time and the second does not (see CLIM-859):

**Tracked upstream gaps** — delete once earthkit covers them:

- ``_match_spatial_grid`` works around exact-equality coordinate alignment.

**OCS policy** — ours to own, because earthkit is a general library and should not be
deciding our domain rules:

- refusing ``relative`` for temperature, an interval scale where percent-of-normal is
  meaningless;
- refusing an observed/normal temporal-resolution mismatch, which earthkit would otherwise
  silently resample.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import xarray as xr
from earthkit.transforms import climatology as ek_climatology

from open_climate_service.data_manager.services.utils import get_x_y_dims
from open_climate_service.process import process
from open_climate_service.shared.cf import is_temperature_like

_ORDINALS = ("dayofyear", "month")
_METHODS = ("absolute", "relative")

# Day count separating a sub-daily/daily observed cube from a monthly one: a day-of-year
# normal expects the former, a month normal the latter.
_MONTHLY_STEP_DAYS = 20
_MONTHLY_STEP_DAYS_MAX = 32
"""Upper bound on what counts as monthly. A one-sided `>= 20` test accepted bimonthly (~61),
quarterly (~91) and yearly (~365) series as monthly, indexing a 12-bin normal by each
timestep's calendar month and returning a plausible anomaly for a pairing nobody intended.
Calendar months span 28-31 days, so 32 leaves room for a median over a leap year.
"""


def _observed_step_days(observed: xr.DataArray, t_dim: str) -> float | None:
    """Median spacing of the observed time axis in days (None for a single timestep)."""
    times = observed[t_dim].values
    if times.size < 2:
        return None
    diffs = np.abs(np.diff(times).astype("timedelta64[h]").astype("float64")) / 24.0
    return float(np.median(diffs))


_COORD_NOISE_RTOL = 1e-6
"""How far a normal's coordinate may sit from the observed one, as a fraction of the grid step.

Sized for floating-point noise and nothing else. The noise this absorbs is ~1e-11 absolute on
coordinates of order 1–360, so a millionth of a cell is already five orders of magnitude of
headroom, while still being five orders below any offset that could represent a different cell.
"""


def _declared_crs(da: xr.DataArray) -> Any:
    """The cube's CRS if it declares one, else None. Never raises."""
    try:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates .rio

        return da.rio.crs
    except Exception:  # noqa: BLE001 — a cube without a grid mapping simply has no CRS
        return None


def _crs_label(crs: Any) -> str:
    """A short name for a CRS: its EPSG code when resolvable, else a trimmed description.

    ``to_string()`` returns the full WKT when the authority cannot be looked up (a broken or
    mismatched PROJ database will do that), which is unreadable in an error message.
    """
    try:
        code = crs.to_epsg()
    except Exception:  # noqa: BLE001 — falls through to the textual form
        code = None
    if code:
        return f"EPSG:{code}"
    text = str(getattr(crs, "name", "") or crs.to_string())
    return text if len(text) <= 60 else f"{text[:57]}..."


def _require_matching_crs(normal: xr.DataArray, observed: xr.DataArray) -> xr.DataArray:
    """Refuse an observed/normal pair that declares two different projections.

    `load_collection` attaches each store's CRS, so this is usually known. Where only one side
    declares one there is nothing to compare and the numeric checks stand alone.
    """
    observed_crs = _declared_crs(observed)
    normal_crs = _declared_crs(normal)
    if observed_crs is None or normal_crs is None or observed_crs == normal_crs:
        return normal
    raise ValueError(
        f"observed is in {_crs_label(observed_crs)} but the normal is in {_crs_label(normal_crs)}; "
        "their coordinates are not comparable, so reproject the normal onto the observed grid before "
        "computing an anomaly"
    )


def _rename_spatial_axes_onto(normal: xr.DataArray, *, x_dim: str, y_dim: str) -> xr.DataArray:
    """Rename the normal's recognised spatial axes onto the observed cube's names."""
    try:
        normal_x, normal_y = get_x_y_dims(normal)
    except ValueError:
        return normal  # no recognisable pair; a spatially constant normal broadcasts legitimately
    if (normal_x, normal_y) == (x_dim, y_dim):
        return normal
    return normal.rename({normal_x: x_dim, normal_y: y_dim})


def _reject_broadcasting_dimensions(normal: xr.DataArray, observed: xr.DataArray, t_dim: str) -> None:
    """Refuse dimensions on the normal that the observed cube lacks.

    xarray would multiply them out rather than align them, so the result grows by a factor of
    their length while meaning nothing. The climatology's own ordinal axis is exempt — indexing
    it onto the observed time axis is precisely what earthkit does.
    """
    extra = [str(d) for d in normal.dims if d not in observed.dims and str(d) not in _ORDINALS and d != t_dim]
    if extra:
        raise ValueError(
            f"the normal has dimension(s) {extra} that the observed cube does not, so subtracting would "
            "broadcast rather than align them; regrid or reduce the normal onto the observed cube's axes "
            "before computing an anomaly"
        )


def _align_units(normal: xr.DataArray, observed: xr.DataArray) -> xr.DataArray:
    """Convert the normal onto the observed cube's units, or refuse the pairing.

    earthkit subtracts (or divides, for ``relative``) with plain xarray arithmetic, which
    ignores units entirely. An observed cube in ``degC`` against a normal in ``K`` therefore
    yields an anomaly near −273 — and the result still carries the observed cube's units, so
    the number is plausible and nothing downstream can tell. Rate units are the same trap
    (``mm/d`` against ``m/s``), and ``relative`` is worse again: a ratio of two different units
    is meaningless rather than merely offset.

    Both cubes are usually CF-stamped from their dataset templates, so the units are known and
    the fix is a conversion. Where a unit is absent or unparseable there is nothing to check
    against, so alignment is left to earthkit as before — this guard adds a refusal, it does
    not add a requirement that every cube be labelled.
    """
    observed_units = str((getattr(observed, "attrs", None) or {}).get("units", "")).strip()
    normal_units = str((getattr(normal, "attrs", None) or {}).get("units", "")).strip()
    if not observed_units or not normal_units or observed_units == normal_units:
        return normal

    from open_climate_service.shared.import_lock import IMPORT_LOCK

    try:
        with IMPORT_LOCK:  # see shared/import_lock (CLIM-1549)
            from xclim.core.units import convert_units_to, units2pint
    except ImportError:  # pragma: no cover - client-only install
        return normal

    try:
        same_unit = units2pint(observed_units) == units2pint(normal_units)
        compatible = units2pint(observed_units).dimensionality == units2pint(normal_units).dimensionality
    except Exception:  # noqa: BLE001 — an unparseable unit is not ours to adjudicate
        return normal

    if same_unit:
        return normal  # a spelling difference such as `mm/d` vs `mm/day`
    if not compatible:
        raise ValueError(
            f"observed is in '{observed_units}' but the normal is in '{normal_units}', which measures a "
            "different quantity; an anomaly between them is not defined. Pair the observed dataset with a "
            "normal computed from it."
        )
    # `convert_units_to` is typed as accepting/returning scalars too; ours is always a cube.
    converted = cast(xr.DataArray, convert_units_to(normal, observed_units))
    converted.attrs = {**normal.attrs, "units": observed_units}
    return converted


def _match_spatial_grid(normal: xr.DataArray, observed: xr.DataArray, t_dim: str) -> xr.DataArray:
    """Relabel the normal's spatial coordinates with the observed cube's, or raise.

    Observed and normal cubes can be produced independently (e.g. a CDS-derived observed vs
    an EDH normal whose longitudes were remapped from [0, 360)), so nominally equal grids
    may still differ by floating-point noise (~1e-11). earthkit's anomaly subtracts the
    climatology with plain xarray arithmetic, which aligns coordinates by **exact
    equality**, and it does not warn when that fails.

    Measured against earthkit-transforms 1.0.0: with 1e-11 of noise on a single row, the
    result keeps its full shape — a closing ``broadcast_like`` restores the dimensions — but
    every cell on the mismatched row comes back NaN. So the output *looks* structurally
    correct while silently losing data, and when the whole axis is uniformly offset (the
    realistic remapped-longitude case) that is every cell in the cube.

    This is therefore **coordinate normalisation, not resampling**: the axes must already
    describe the same cells (same length, same spacing, same positions to within
    :data:`_COORD_NOISE_RTOL` of a step), and then the observed coordinates are copied over
    so the arithmetic aligns. Anything larger raises and asks for an explicit regridding step.

    An earlier version reindexed by nearest neighbour within *half a grid step*, which is
    nine orders of magnitude more slack than float noise needs, and paired different physical
    cells without saying so: on a 0.1° grid, a normal offset by a full 0.1° came back shifted
    one cell with only a single NaN, so the anomaly was computed against the neighbouring
    cell for 9 of 10 cells while looking clean. A cell-centre vs cell-edge convention
    mismatch between two products is exactly that case.

    Upstream gap, not a preference: earthkit could either accept a tolerance or — cheaper and
    arguably better — detect the failed alignment and raise instead of returning a quietly
    NaN cube. Filed as `ecmwf/earthkit-transforms#126
    <https://github.com/ecmwf/earthkit-transforms/issues/126>`_; delete this helper once it lands.

    A *regular* grid is assumed (true for the lat/lon and UTM grids this serves); the spacing
    check below is what makes that assumption explicit rather than silent.

    Only the x/y axes are considered. "Every non-temporal shared dimension" would also catch
    `bands`, whose labels are strings — the spacing check would raise
    ``could not convert string to float`` before earthkit ever ran — and axes like `quantile`,
    where numeric closeness has no grid meaning. Those align by ordinary xarray rules.

    Three things are settled before the numeric comparison, because each would otherwise make
    it meaningless:

    * **The CRS.** Coordinates are only comparable within one projection. Two UTM cubes an
      hour's drive apart carry overlapping eastings and northings, so identical numbers can
      describe entirely different places — UTM 45N against 44N passes every numeric check.
    * **Axis naming.** A normal on the same grid but naming its axes `latitude`/`longitude`
      shares no dimension name with an `x`/`y` observed cube, so nothing was compared and
      xarray broadcast the subtraction across all four axes: a `(10, 3, 3)` cube came back
      `(10, 3, 3, 3, 3)`, which on a country grid is thousands of times the memory for a
      meaningless result. Recognised spellings are renamed onto the observed cube's.
    * **Leftover dimensions.** Anything still on the normal that the observed cube lacks —
      and that is not the climatology's own ordinal axis — would broadcast the same way, so it
      is refused rather than silently multiplied out.
    """
    normal = _require_matching_crs(normal, observed)

    try:
        x_dim, y_dim = get_x_y_dims(observed)
    except ValueError:
        # No axes to rename onto, but the guard still applies. Returning ahead of it let an
        # observed `(t, northing, easting)` cube reach xarray's arithmetic unchecked and
        # broadcast against a `(dayofyear, y, x)` normal to five dimensions — the blow-up the
        # guard exists to stop. It compares dimension names, so it needs no axis of its own.
        _reject_broadcasting_dimensions(normal, observed, t_dim)
        return normal

    # After the rename, not before: a normal spelling its axes `lat`/`lon` is renameable onto the
    # observed grid, and checking first would refuse it as an extra dimension.
    normal = _rename_spatial_axes_onto(normal, x_dim=x_dim, y_dim=y_dim)
    _reject_broadcasting_dimensions(normal, observed, t_dim)

    spatial_dims = [d for d in (x_dim, y_dim) if d in normal.dims and d in observed.coords]
    if not spatial_dims:
        return normal

    replacements: dict[str, Any] = {}
    for dim in spatial_dims:
        obs_coord = observed[dim].values
        nrm_coord = normal[dim].values

        if obs_coord.shape != nrm_coord.shape:
            raise ValueError(
                f"observed and normal disagree on the size of '{dim}' "
                f"({obs_coord.size} vs {nrm_coord.size}); regrid the normal onto the observed "
                "grid before computing an anomaly"
            )
        if obs_coord.size < 2:
            # A single cell has no spacing to compare; fall through to the position check,
            # which needs a step — use the coordinate magnitude to scale the tolerance.
            step = abs(float(obs_coord[0])) or 1.0
        else:
            obs_steps = np.diff(obs_coord.astype(float))
            nrm_steps = np.diff(nrm_coord.astype(float))
            step = abs(float(obs_steps[0]))
            if not np.allclose(nrm_steps, obs_steps, rtol=_COORD_NOISE_RTOL, atol=0.0):
                raise ValueError(
                    f"observed and normal have different '{dim}' spacing "
                    f"({obs_steps[0]:g} vs {nrm_steps[0]:g}); regrid the normal onto the "
                    "observed grid before computing an anomaly"
                )

        offset = np.abs(nrm_coord.astype(float) - obs_coord.astype(float))
        worst = float(offset.max())
        if worst > _COORD_NOISE_RTOL * abs(step):
            raise ValueError(
                f"observed and normal '{dim}' coordinates differ by up to {worst:g} "
                f"({worst / abs(step):.3g} of the {abs(step):g} grid step), which is more than "
                "floating-point noise — they describe different cells. Regrid the normal onto "
                "the observed grid before computing an anomaly."
            )
        replacements[dim] = observed[dim]

    # assign_coords, not reindex: a pure relabel cannot reorder, drop or pair cells.
    return normal.assign_coords(replacements)


@process(
    summary="Climate anomaly (observed − climatological normal)",
    parameters={
        "observed": {"description": "Observed cube with a datetime time axis (e.g. era5land_temperature_daily)."},
        "normal": {"description": "Climatological normal with a `dayofyear` or `month` ordinal axis."},
        "method": {
            "description": (
                "'absolute' (observed − normal, default) or 'relative' (percent: "
                "100·(observed − normal)/normal). 'relative' is only meaningful for a "
                "ratio-scale variable such as precipitation, not temperature. "
                "'standardised' (z-score) needs a standard-deviation normal — not yet "
                "supported."
            )
        },
    },
)
def compute_anomaly(observed: xr.DataArray, normal: xr.DataArray, method: str = "absolute") -> xr.DataArray:
    """Compute observed − climatological normal, aligning the normal by day-of-year/month.

    earthkit indexes the normal's ordinal axis (``dayofyear`` or ``month``) by each observed
    timestep's calendar value and combines per ``method``; the result keeps the observed
    time axis and stays lazy/dask-backed. The observed temporal resolution must match the
    normal's ordinal axis (daily observed ↔ ``dayofyear`` normal, monthly observed ↔
    ``month`` normal); a mismatch is rejected rather than silently resampled.
    """
    if method not in _METHODS:
        if method == "standardised":
            # CLIM-887 tracks standard-deviation normals.
            raise ValueError("method 'standardised' (z-score) needs a standard-deviation normal, not yet supported")
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    # earthkit 1.0 auto-detects the time dimension, so passing time_dim below is belt and
    # braces. The detection is kept for the *error*: given a cube with no datetime axis,
    # earthkit fails with "Invalid frequency 'month' - see xarray documentation", which
    # names neither the real problem nor the cube. Checking here says what is actually wrong.
    t_dim = next(
        (d for d in observed.dims if d in observed.coords and np.issubdtype(observed[d].dtype, np.datetime64)),
        None,
    )
    if t_dim is None:
        raise ValueError("compute_anomaly requires a datetime temporal dimension on the observed cube")
    t_dim = str(t_dim)

    ordinal = next((d for d in _ORDINALS if d in normal.dims), None)
    if ordinal is None:
        raise ValueError(f"normal must have one of {_ORDINALS} as a dimension, got dims {tuple(normal.dims)}")

    # OCS policy, not an earthkit gap: earthkit offers `relative` for any variable, which is
    # correct for a general library. Deciding it is invalid for our temperature datasets is
    # our call, so this guard stays regardless of what upstream does.
    if method == "relative" and is_temperature_like(observed, normal):
        raise ValueError(
            "method 'relative' is not meaningful for temperature (an interval scale): dividing by the "
            "normal flips sign for a negative normal and diverges near 0 °C. Use 'absolute', or 'relative' "
            "only for ratio-scale variables such as precipitation."
        )

    # OCS policy, as above. earthkit would silently resample the observed cube to the
    # normal's frequency (daily observed against a month normal → monthly means), returning
    # a plausible result for a pairing the caller almost certainly did not intend. Upstream
    # could reasonably warn; refusing outright is our decision, so this guard also stays.
    step = _observed_step_days(observed, t_dim)
    if step is not None:
        if ordinal == "month" and not _MONTHLY_STEP_DAYS <= step <= _MONTHLY_STEP_DAYS_MAX:
            raise ValueError(
                f"a 'month' normal expects a monthly observed dataset, but the observed steps ~{step:.0f} "
                "day(s); pair it with a monthly observed, or use a 'dayofyear' normal"
            )
        if ordinal == "dayofyear" and step >= _MONTHLY_STEP_DAYS:
            raise ValueError(
                f"a 'dayofyear' normal expects a daily observed dataset, but the observed steps ~{step:.0f} "
                "day(s); pair it with a daily observed, or use a 'month' normal"
            )

    # Guard the float-noise grid mismatch before earthkit's xarray subtraction (see helper).
    normal = _match_spatial_grid(normal, observed, t_dim)
    normal = _align_units(normal, observed)
    result = cast(
        xr.DataArray,
        ek_climatology.anomaly(observed, climatology=normal, time_dim=t_dim, relative=method == "relative"),
    )
    if method == "relative":
        result = _mask_undefined_ratios(result)
    return _restore_scalar_coords(result, observed, ordinal)


def _mask_undefined_ratios(result: xr.DataArray) -> xr.DataArray:
    """Blank the cells where dividing by the normal has no answer.

    A relative anomaly is `100 * (observed - normal) / normal`, and a zero normal is ordinary
    rather than exceptional: CHIRPS January over Sierra Leone averages 0.4 mm/d, with individual
    cells at exactly zero. The division then yields ±inf, or NaN for 0/0, and published straight
    into the store those read as real values — an infinite percentage on a map, or a number that
    silently poisons any spatial mean taken over it.

    Undefined is the honest answer, and NaN is how the rest of the store spells it. Stays lazy.
    """
    return result.where(np.isfinite(result))


def _restore_scalar_coords(result: xr.DataArray, observed: xr.DataArray, ordinal: str) -> xr.DataArray:
    """Undo the coordinate broadcasting earthkit's per-timestep indexing leaves behind.

    Indexing the normal by each observed timestep carries the normal's non-dimension coords
    along, so a scalar ``spatial_ref`` returns with a length-N time dimension. Publishing that
    fails at ``xproj.assign_crs`` — "can only create a CRSIndex from one scalar variable" — and
    a per-timestep CRS would be meaningless even if it wrote. The normal's ordinal coord
    (``month``/``dayofyear``) rides along too and is not part of the anomaly's own geometry.

    Only coords that are scalar on the observed cube are restored, so nothing that legitimately
    varies is touched.
    """
    if ordinal in result.coords and ordinal not in result.dims:
        result = result.drop_vars(ordinal, errors="ignore")
    for name, coord in observed.coords.items():
        if coord.dims:
            continue
        if name in result.coords and result[name].dims:
            result = result.assign_coords({name: coord})
    return result
