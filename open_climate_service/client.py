"""Lightweight client for a running Open Climate Service instance.

The base install needs only ``httpx`` and ``pystac``. With it you can discover
published datasets, list the available processes and workflows, and run process
graphs over HTTP — everything that talks to a remote instance via REST::

    pip install open-climate-service             # talk to an instance (REST only)
    pip install open-climate-service[xarray]     # + open datasets as xarray
    pip install open-climate-service[server]     # run your own instance

Opening a published dataset as an in-memory :class:`xarray.Dataset` additionally
requires the optional ``xarray`` extra; :meth:`ClimateService.open_dataset` raises a
clear error pointing to it when the dependency is missing.

Example:
    >>> from open_climate_service import ClimateService
    >>> service = ClimateService("https://my-instance.example.org")
    >>> service.datasets()  # discover published collections
    >>> ds = service.open_dataset("chirps3_precipitation_daily")  # needs [xarray]
    >>> service.workflows()  # list reusable workflows (UDPs)
    >>> service.execute(
    ...     {  # run a process graph synchronously
    ...         "agg": {
    ...             "process_id": "aggregate_to_dhis2_json",
    ...             "arguments": {...},
    ...             "result": True,
    ...         }
    ...     }
    ... )
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    import xarray as xr

_FALLBACK_BASE_URL = "http://127.0.0.1:9000"
_DEFAULT_TIMEOUT = 30.0
# Synchronous process graphs (POST /result) can run for minutes — allow more headroom.
_RESULT_TIMEOUT = 300.0

_XARRAY_INSTALL_HINT = (
    "Opening datasets as xarray needs extra dependencies that are not part of the "
    "base client. Install them with: pip install 'open-climate-service[xarray]'"
)


def _default_base_url() -> str:
    return os.environ.get("CLIMATE_SERVICE_BASE_URL", _FALLBACK_BASE_URL)


def _id_from_href(href: str) -> str:
    """Extract the dataset id from a STAC child href by reading the last URL path segment."""
    return urlparse(href).path.rstrip("/").rsplit("/", 1)[-1]


def _child_links(catalog: Any, source: str) -> list[dict]:
    """Extract STAC child links (one per published dataset) from a catalog response."""
    raw_links = catalog.get("links") if isinstance(catalog, dict) else None
    if not isinstance(raw_links, list):
        raise ValueError(f"Invalid STAC catalog response from {source}: missing or non-list 'links' field")
    links = []
    for link in raw_links:
        if isinstance(link, dict) and link.get("rel") == "child":
            href = link.get("href")
            if not isinstance(href, str) or not href:
                raise ValueError(f"STAC child link from {source} has a missing or invalid href")
            links.append({**link, "id": _id_from_href(href)})
    return links


def _zarr_asset(collection: Any, dataset_id: str, source: str) -> tuple[str, dict]:
    """Return the Zarr asset ``(href, open_kwargs)`` from a STAC collection response."""
    assets = collection.get("assets") if isinstance(collection, dict) else None
    if not isinstance(assets, dict):
        raise ValueError(f"STAC collection for '{dataset_id}' from {source} has a missing or invalid 'assets' field")
    asset = assets.get("zarr")
    if not isinstance(asset, dict):
        raise ValueError(f"Dataset '{dataset_id}' has no Zarr asset in the STAC collection")
    href = asset.get("href")
    if not isinstance(href, str) or not href:
        raise ValueError(f"Zarr asset for '{dataset_id}' has a missing or invalid href")
    open_kwargs = asset.get("xarray:open_kwargs", {})
    if not isinstance(open_kwargs, dict):
        raise ValueError(f"Zarr asset for '{dataset_id}' has a malformed xarray:open_kwargs field")
    return href, open_kwargs


def _open_zarr(href: str, open_kwargs: dict) -> xr.Dataset:
    """Open a (possibly remote) Zarr store as an xarray Dataset, lazy-importing xarray."""
    if importlib.util.find_spec("xarray") is None:
        raise ModuleNotFoundError(_XARRAY_INSTALL_HINT)
    import xarray as xr

    ds: xr.Dataset = xr.open_zarr(href, **open_kwargs)
    # Pyramided (multiscale) stores expose data under the highest-resolution group "0". The
    # asset names that group itself; this covers a server that does not yet.
    if not ds.data_vars:
        ds.close()
        without_group = {key: value for key, value in open_kwargs.items() if key != "group"}
        ds = xr.open_zarr(href, group="0", **without_group)
    return ds


def _process_graph_body(process_graph: dict[str, Any]) -> dict[str, Any]:
    """Wrap a process-graph nodes mapping into the POST /result request body.

    Accepts a bare nodes mapping (``{node_id: {...}}``, e.g. an openEO client's
    ``flat_graph()``) or a ``{"process_graph": {...}}`` wrapper.
    """
    pg = process_graph["process_graph"] if set(process_graph) == {"process_graph"} else process_graph
    return {"process": {"process_graph": pg}}


class ClimateService:
    """Client for an Open Climate Service instance.

    Maintains a persistent HTTP connection for efficient repeated calls. Use as a
    context manager to ensure the connection is closed::

        with ClimateService("http://localhost:9000") as service:
            datasets = service.datasets()

    Args:
        base_url: Base URL of the running Open Climate Service (e.g. ``http://localhost:9000``).
        timeout: Default HTTP request timeout in seconds (default: 30).
    """

    def __init__(self, base_url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._http.close()

    def __enter__(self) -> ClimateService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def datasets(self) -> list[dict]:
        """Return all published datasets from the STAC catalog.

        Each entry is a STAC child link dict with at least ``id``, ``title``, and ``href``.
        """
        response = self._http.get(f"{self.base_url}/stac/catalog.json")
        response.raise_for_status()
        return _child_links(response.json(), self.base_url)

    def open_dataset(self, dataset_id: str) -> xr.Dataset:
        """Open a published dataset as an xarray Dataset.

        Fetches the STAC collection for ``dataset_id``, reads the Zarr asset metadata,
        and returns the opened (lazy) dataset. Requires the ``xarray`` extra.
        """
        response = self._http.get(f"{self.base_url}/stac/collections/{dataset_id}")
        response.raise_for_status()
        href, open_kwargs = _zarr_asset(response.json(), dataset_id, self.base_url)
        return _open_zarr(href, open_kwargs)

    def processes(self) -> list[dict]:
        """Return all openEO processes available on the instance (``GET /processes``)."""
        response = self._http.get(f"{self.base_url}/processes")
        response.raise_for_status()
        processes: list[dict] = response.json().get("processes", [])
        return processes

    def workflows(self) -> list[dict]:
        """Return the reusable workflows (stored process graphs / UDPs) (``GET /process_graphs``)."""
        response = self._http.get(f"{self.base_url}/process_graphs")
        response.raise_for_status()
        workflows: list[dict] = response.json().get("processes", [])
        return workflows

    def execute(
        self,
        process_graph: dict[str, Any],
        *,
        timeout: float | None = _RESULT_TIMEOUT,
        path: str | Path | None = None,
    ) -> Any:
        """Run a process graph synchronously and return its result (``POST /result``).

        ``process_graph`` is a process-graph nodes mapping (e.g. an openEO client's
        ``flat_graph()``); one node must carry ``"result": true``.

        Returns a parsed object for JSON results (e.g. a DHIS2 ``dataValueSet`` from
        ``save_result`` ``format: "DHIS2JSON"``). For file results (CSV, GeoJSON,
        GeoTIFF, …) it returns the raw ``bytes``, or — when ``path`` is given — writes
        them there and returns the :class:`~pathlib.Path`.
        """
        response = self._http.post(
            f"{self.base_url}/result",
            json=_process_graph_body(process_graph),
            timeout=timeout,
        )
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        if path is not None:
            Path(path).write_bytes(response.content)
            return Path(path)
        return response.content


def list_datasets(base_url: str | None = None) -> list[dict]:
    """Return all published datasets from the STAC catalog (one-shot, no persistent connection).

    Each entry is a STAC child link dict with at least ``id``, ``title``, and ``href``.
    ``base_url`` defaults to the ``CLIMATE_SERVICE_BASE_URL`` environment variable,
    falling back to ``http://127.0.0.1:9000``.
    """
    url = (base_url or _default_base_url()).rstrip("/")
    response = httpx.get(f"{url}/stac/catalog.json", timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return _child_links(response.json(), url)


def open_dataset(dataset_id: str, *, base_url: str | None = None) -> xr.Dataset:
    """Open a published dataset as an xarray Dataset (one-shot, no persistent connection).

    Fetches the STAC collection for ``dataset_id``, reads the Zarr asset metadata, and
    returns the opened dataset. Requires the ``xarray`` extra. ``base_url`` defaults to
    the ``CLIMATE_SERVICE_BASE_URL`` environment variable, falling back to
    ``http://127.0.0.1:9000``.
    """
    url = (base_url or _default_base_url()).rstrip("/")
    response = httpx.get(f"{url}/stac/collections/{dataset_id}", timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    href, open_kwargs = _zarr_asset(response.json(), dataset_id, url)
    return _open_zarr(href, open_kwargs)
