"""The media type a STAC client is told for a published Zarr store.

A *read* concern about an already-written store, and a *serving* one: it decides what the STAC
collection and the openEO job result advertise, which is what gates whether a pyramid-only client
will render the store at all. It lives here rather than beside the writer (``shared/geozarr.py``)
because it runs per request against a store that may have been written long ago, and needs
filesystem and Icechunk access the writer's pure geometry helpers do not.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Media type: telling a STAC client whether the store is pyramided ---------------------#
# STAC has no way to distinguish a flat store from a pyramided one through the plain Zarr
# media type, and clients that only render pyramided stores therefore cannot decide whether an
# asset is safe to open. The `profile` parameter closes that gap. It is a
# STAC Zarr best-practices recommendation, which notes it is not (yet) part of the official
# Zarr media type registration:
#
#     https://github.com/radiantearth/stac-best-practices/blob/main/best-practices-zarr.md
#
# Consumers match it as a literal, not by parsing parameters. stac-js holds
# `wozMediaTypes = ['application/vnd.zarr; version=3; profile=multiscales']` and compares with
# `allowedTypes.includes(type.toLowerCase())` — so case is forgiven but parameter order and
# spacing are not. WEB_OPTIMIZED_ZARR_MEDIA_TYPE must stay byte-identical.

ZARR_V3_MEDIA_TYPE = "application/vnd.zarr; version=3"
"""Media type for a published Zarr v3 store with no resolution pyramid."""

WEB_OPTIMIZED_ZARR_MEDIA_TYPE = f"{ZARR_V3_MEDIA_TYPE}; profile=multiscales"
"""Media type for a pyramided ("web-optimized") Zarr v3 store.

Byte-identical to the string web clients match on; do not reformat or reorder the parameters.
"""

FULL_RESOLUTION_GROUP = "0"
"""The pyramid level holding the data at native resolution, the one analysis reads."""

MULTISCALES_CONVENTION_UUID = "d35379db-88df-4056-af3a-620245f8e347"
"""UUID of the Zarr multiscales convention (https://github.com/zarr-conventions/multiscales)."""


def is_multiscales_convention(convention: Any) -> bool:
    """True for one ``zarr_conventions`` entry declaring the multiscales convention.

    Shared with the streaming writer, which has to carry this entry across an append rather
    than let a flat store's convention list replace it.
    """
    return isinstance(convention, Mapping) and (
        convention.get("uuid") == MULTISCALES_CONVENTION_UUID or convention.get("name") == "multiscales"
    )


def attributes_declare_multiscales(attributes: Mapping[str, Any]) -> bool:
    """True when root group *attributes* describe a real multiscales pyramid.

    Requires both the convention declared in ``zarr_conventions`` *and* a non-empty
    ``multiscales.layout``. Those are the same two conditions pyramid-aware renderers check, so
    the claim matches what a renderer can actually use. A store that declares the convention but
    lists no levels is treated as flat — there is nothing extra for a client to read.
    """
    conventions = attributes.get("zarr_conventions")
    if not isinstance(conventions, list):
        return False
    if not any(is_multiscales_convention(convention) for convention in conventions):
        return False
    multiscales = attributes.get("multiscales")
    if not isinstance(multiscales, Mapping):
        return False
    layout = multiscales.get("layout")
    return isinstance(layout, list) and len(layout) > 0


def read_root_attributes(store_path: str, *, icechunk: bool) -> Mapping[str, Any]:
    """Root group attributes of a published store; empty mapping when unreadable.

    Reads group metadata only, never chunk data. Deliberately the *root* group: the store
    accessors descend into pyramid level ``0``, whose attributes do not carry the multiscales
    layout, so this needs its own path.

    Remote stores return empty rather than paying a network round trip — the caller degrades to
    the flat media type.
    """
    if icechunk:
        return _read_icechunk_root_attributes(store_path)
    if "://" in store_path:
        return {}
    zarr_json = Path(store_path) / "zarr.json"
    try:
        payload = json.loads(zarr_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    attributes = payload.get("attributes")
    return attributes if isinstance(attributes, Mapping) else {}


def _read_icechunk_root_attributes(store_path: str) -> Mapping[str, Any]:
    import icechunk
    import zarr

    if not Path(store_path).exists():
        return {}
    storage = icechunk.local_filesystem_storage(store_path)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    group = zarr.open_group(session.store, mode="r")
    return dict(group.attrs)


def zarr_media_type(store_path: str, *, icechunk: bool) -> str:
    """The media type to advertise for the store at *store_path*.

    Falls back to the flat :data:`ZARR_V3_MEDIA_TYPE` whenever the pyramid cannot be confirmed.
    The asymmetry is deliberate: understating is harmless — a client opens the store the ordinary
    way — while overstating sends a renderer looking for levels that do not exist. A locked or
    corrupt store must never take the STAC collection down with it.
    """
    try:
        attributes = read_root_attributes(store_path, icechunk=icechunk)
    except Exception:  # noqa: BLE001 — any read failure degrades to the flat type
        logger.debug("Could not read root attributes of store %r", store_path, exc_info=True)
        return ZARR_V3_MEDIA_TYPE
    if attributes_declare_multiscales(attributes):
        return WEB_OPTIMIZED_ZARR_MEDIA_TYPE
    return ZARR_V3_MEDIA_TYPE


def data_group_open_kwargs(media_type: str) -> dict[str, str]:
    """The ``xarray:open_kwargs`` that reach the data variables of a store with *media_type*.

    A pyramided root holds only the time coordinate and ``spatial_ref``, so a client opening it
    as-is gets a dataset with no variables and no error. Naming the full-resolution level sends
    it to the data. Empty for a flat store, whose root is the data.
    """
    if media_type == WEB_OPTIMIZED_ZARR_MEDIA_TYPE:
        return {"group": FULL_RESOLUTION_GROUP}
    return {}
