"""Shared conventions for vector datacubes."""

from __future__ import annotations

GEOMETRY_WKT_COORD = "geometry_wkt"
"""Companion coordinate on a vector cube's geometry dimension, holding each feature's WKT.

Defined here because both sides need it and neither may import the other: `aggregate_spatial`
is a discovered plugin process that writes it, and the openEO job writers read it. The
dimension itself carries feature *labels* (ids) — which the DHIS2 and CHAP exports use as
their location column — so the shapes ride alongside rather than replacing them.
"""

RESAMPLING_ATTR = "ocs:resampling"
"""Cube attribute carrying the source dataset's ``ingestion.resampling``.

Set by ``load_collection`` and read by ``aggregate_spatial``, which aggregates a categorical
cube (``mode`` or ``max``) by majority rather than by mean. Here for the same reason as
``GEOMETRY_WKT_COORD``: the writer and the reader may not import each other.
"""
