"""Protocol and shared types for per-period streaming ingest.

This package is intentionally internal.

`open_climate_service.ingestions` owns the application-facing ingestion lifecycle
(request handling, artifact records, publication state, and compatibility with
the rest of the API surface). `open_climate_service.streaming` owns the execution
mechanics for one ingestion strategy: enumerate periods, fetch them one at a time,
and append them to an Icechunk-backed Zarr v3 store (the store grid is inferred
from the first fetched period).

The protocol defined here is deliberately narrow so plugins stay source-focused.
They should not manage jobs, artifact persistence, sync policy, or store
mutation outside writing one fetched period as an `xarray.Dataset`.

Most plugins should subclass `BaseDatasetPlugin` (see ``base.py``) rather than
implement this protocol from scratch: the base class supplies the concurrency
defaults, leaving the plugin to implement just ``periods()`` and ``fetch_period()``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def to_epsg_int(crs: int | str) -> int:
    """Normalize a CRS given as an int or string to an EPSG integer code.

    Accepts ``4326``, ``"EPSG:4326"``, ``"epsg:4326"``, ``"4326"``, or the OGC
    ``CRS84`` aliases (treated as EPSG:4326).
    """
    if isinstance(crs, int):
        return crs
    token = str(crs).strip()
    if token.upper() in {"CRS84", "OGC:CRS84"} or token.upper().endswith("/CRS84"):
        return 4326
    if ":" in token:
        token = token.rsplit(":", 1)[-1]
    return int(token)


@dataclass
class GridSpec:
    """Store-shaping metadata for one streamed dataset.

    Internal type: the orchestrator infers it from the first fetched period
    (shape and dtype from the array, nodata from the source ``_FillValue``, CRS
    from the data or the plugin's ``crs`` attribute). Plugins no longer construct
    it by hand.

    Fields:
    - spatial shape of one fetched period
    - native CRS (accepts an int EPSG code or a string like ``"EPSG:4326"``;
      normalized to an int on construction)
    - value dtype / nodata
    - canonical time/x/y dimension names and any root attrs the writer should keep

    `shape` is ordered as `(y, x)`, matching array axis order:
    `(len(y_dim), len(x_dim))` — the size of `y_dim` then `x_dim`.
    """

    shape: tuple[int, int]  # (y, x) == (len(y_dim), len(x_dim))
    crs: int | str
    dtype: np.dtype[Any]
    nodata: float | None = None
    time_dim: str = "t"
    x_dim: str = "x"
    y_dim: str = "y"
    attrs: dict[str, Any] = field(default_factory=dict)
    extra_dims: dict[str, int] = field(default_factory=dict)
    """Optional non-spatial, non-time dimensions, e.g. ``{"age_group": 20}``.

    The orchestrator does not use this field; it exists for plugin authors who
    need to document multidimensional stores and for future orchestrator
    extensions.
    """

    def __post_init__(self) -> None:
        self.crs = to_epsg_int(self.crs)


@runtime_checkable
class IngestionPlugin(Protocol):
    """Minimal source contract for the streaming ingest engine.

    Only `periods(...)` and `fetch_period(...)` are required. Plugins may
    additionally:

    - accept template-defined `ingestion.params` through their constructor
    - declare `max_concurrency` / `commit_batch_size` (the orchestrator defaults
      both to 1 when absent), `time_dim` / `x_dim` / `y_dim` (default
      `t` / `x` / `y`), and `crs` (the CRS fallback for grid inference, default 4326)

    The store grid is inferred from the first fetched period.
    Planning may run `periods(...)` on a different event loop from asynchronous
    `fetch_period(...)` calls. Do not share loop-bound resources between them;
    create and close those resources within the call that uses them.
    `ingestion.params` are forwarded to `fetch_period(...)` as `**params` for
    sources that prefer per-call configuration rather than constructor state.

    The engine remains responsible for concurrency, resume, cancellation,
    cursor persistence, and store writes. Most plugins should subclass
    `BaseDatasetPlugin` rather than implement this protocol directly.
    """

    async def periods(self, start: str, end: str) -> list[str]:
        """Return ordered available period identifiers for the requested range."""
        ...

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> "xr.Dataset | Awaitable[xr.Dataset]":
        """Fetch one period as a dataset (normalized to t, y, x).

        May be a regular method (run in a worker thread) or ``async def`` for
        natively-async sources; the orchestrator awaits or threads it accordingly.
        """
        ...
