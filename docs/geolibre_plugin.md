# GeoLibre plugin

[GeoLibre](https://geolibre.app/) is a free, open-source, lightweight GIS that runs in the browser, on the desktop, and in Jupyter notebooks. The **Open Geo Service plugin** connects GeoLibre to a running OCS instance: it browses the instance's STAC catalog and renders its published GeoZarr datasets on an interactive map — taking the CRS, time axis, variable, colormap, and value range straight from the dataset's metadata, so nothing has to be entered by hand.

![The Open Geo Service plugin rendering a daily mean temperature anomaly over Norway in GeoLibre](assets/geolibre-plugin.png)

It's a good way to explore an instance's data interactively without writing any code — pick a dataset, step through time, and click the map to read values.

## What you can do

- **Browse** any OCS instance's published datasets.
- **Render** a dataset on the map with its own default styling — projected national grids (e.g. seNorge on `EPSG:32633`) land in the right place automatically.
- **Step through time** with GeoLibre's time slider, at the dataset's own period type.
- **Read values** by clicking the map (click-to-value).

## Install

Two options:

- **From the GeoLibre plugin marketplace** — in GeoLibre go to **Settings → Manage Plugins**, find *Open Geo Service*, and install it.
- **From a downloaded zip** — grab the latest `open-geo-service-<version>.zip` from the plugin's
  [Releases](https://github.com/dhis2/open-geo-service-geolibre-plugin/releases) page, then in GeoLibre go to
  **Settings → Manage Plugins → Install from file** and select the zip.

## Use

1. Activate **Open Geo Service** from GeoLibre's **Plugins** menu.
2. In the panel, enter your OCS instance URL (e.g. `http://localhost:8002`) and click **Connect** — the map frames the instance's region.
3. Pick a dataset to draw it as a layer.
4. Step through periods with the **time slider** at the bottom of the map, adjust the **colormap** and **value range** in the panel, and **click the map** to read the value at a point.

## Notes

- The plugin needs a running Open Geo Service instance to connect to; it depends only on OCS's public HTTP contract (`/stac` and `/zarr/{id}/`), so it works against any instance.
- Serving a browser build of GeoLibre from a different origin than the OCS instance needs CORS headers on the OCS `/zarr/` and `/stac` routers; the desktop app fetches cross-origin without that.

## Links

- **Plugin repository:** <https://github.com/dhis2/open-geo-service-geolibre-plugin>
- **GeoLibre:** <https://geolibre.app/>
