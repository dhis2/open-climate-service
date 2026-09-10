"""Shared conventions for vector datacubes."""

from __future__ import annotations

GEOMETRY_WKT_COORD = "geometry_wkt"
"""Companion coordinate on a vector cube's geometry dimension, holding each feature's WKT.

Defined here because both sides need it and neither may import the other: `aggregate_spatial`
is a discovered plugin process that writes it, and the openEO job writers read it. The
dimension itself carries feature *labels* (ids) — which the DHIS2 and CHAP exports use as
their location column — so the shapes ride alongside rather than replacing them.
"""
