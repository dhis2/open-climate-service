"""Zonal statistics over raster cubes: what ``aggregate_spatial`` computes.

``plugins/processes/aggregate_spatial.py`` is the process; this package holds how it works.

* ``geometries`` parses GeoJSON input and brings it into the cube's CRS.
* ``reducers`` tells a reducer the weighted path can compute from one it must run as given.
* ``grid`` describes the cube's cells, each polygon's coverage of them, and blocked reads.
* ``weighting`` computes area-weighted statistics over polygons.
* ``categorical`` handles class data: majority, per-class fractions, and which variables are categorical.
* ``pixel_centre`` is the openEO specification's rule, for reducers that cannot be weighted.
"""
