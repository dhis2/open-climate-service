"""Shared conventions for vector datacubes."""

from __future__ import annotations

GEOMETRY_WKT_COORD = "geometry_wkt"
"""Coordinate containing each feature's geometry as WKT.

The vector cube's geometry dimension carries feature labels/IDs; this coordinate
stores the corresponding geometry without replacing those labels.
"""
