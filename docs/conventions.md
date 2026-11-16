# Raster and coordinate conventions

Two different standards apply to two different things, and confusing them is how rasters end
up transposed or mislocated. This page settles which applies where, and what the framework
guarantees so that plugin authors do not have to.

## The two orders, and why they don't conflict

| Concern                             | Order                      | Standard                                     |
| ----------------------------------- | -------------------------- | -------------------------------------------- |
| Raster array / data cube dimensions | `(…, y, x)` — x last       | NumPy row-major, CF `T,Z,Y,X`, GeoZarr, GDAL |
| GeoJSON geometry coordinate pairs   | `[x, y]` = `[lon, lat]`    | RFC 7946                                     |
| Bounding boxes                      | `[xmin, ymin, xmax, ymax]` | RFC 7946 (`[west, south, east, north]`)      |

These are not in tension — they describe different objects. `(…, y, x)` is the shape of an
N-dimensional array, where the row index comes before the column index because that is how
row-major memory is laid out. `[x, y]` is a coordinate _tuple_, where longitude comes first
because RFC 7946 says so. A single codebase uses both, in their own places.

So: **array dimensions are `(…, y, x)`; coordinate variables are named `x` / `y`; GeoJSON
pairs and bboxes are `[x, y]`-ordered.**

openEO, which the processing layer is built on, is deliberately neutral here — it addresses
dimensions by _name_ (`reduce_dimension(dimension="t")`) and spatial extents by _field_
(`west`/`east`/`south`/`north`), so the question never arises at the API level. But the
reference implementation runs on xarray, which is NumPy/CF-ordered, so every cube flowing
through the engine is physically `(…, y, x)` regardless.

## What the framework guarantees

These are properties of a _published store_, not of any one source, so they are enforced once
at the write boundary — see [`shared/raster_contract.py`](https://github.com/dhis2/open-geo-service/blob/main/open_geo_service/shared/raster_contract.py).
A plugin may return data in whatever orientation and naming its source delivers.

1. **The temporal dimension is named `t`.** Sources spelling it `time`, `valid_time`, `date`
   or `time_counter` are renamed. The map viewer looks for `t`; a store that ships `time`
   renders with no time control.
2. **Spatial dimensions are named `y` / `x` and ordered `(…, y, x)`.** Variables arriving in
   another order are transposed.
3. **`y` descends — array row 0 is the northernmost row.** South-up sources are reversed,
   coordinate and data together, so the value at a given latitude does not move; only its row
   index does. See below for why this is guaranteed rather than left to the reader.
4. **A geographic x axis runs −180…180.** Sources on the 0…360 frame (ERA5 among them) are
   rolled and re-sorted. Projected eastings are left alone — one legitimately exceeds 180.
5. **The declared CRS matches the coordinates it describes.** Data is stored in its _native_
   CRS, whatever the source delivered; nothing is reprojected on ingest. The invariant is
   consistency, not a fixed CRS. The instance-wide `crs:` setting is never used as a fallback
   for an untagged cube — doing so stamped `EPSG:32633` onto degree coordinates and put the
   store at the projection's origin instead of on the map.

## What this means if you write a plugin

Nothing. Return data in whatever orientation, dimension naming and longitude frame your source
delivers, and declare a `crs` class attribute only if the source carries no CRS of its own. The
framework normalises the rest at the write boundary; a plugin that hand-rolls a y-flip or a
longitude roll is doing work that will simply be re-done.
