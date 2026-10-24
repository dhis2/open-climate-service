"""Opt-in live interoperability smoke tests for downstream STAC clients.

These checks are skipped by default and currently expect a separate environment
for `openeo` because the available `openeo` releases are incompatible with this
project's pinned xarray stack through `dhis2eo`.

The smoke test uses ``openeo.rest.datacube.DataCube.load_stac(...)`` directly
instead of an openEO backend connection so it can validate client-side STAC
parsing against a live Open Climate Service instance without requiring a running openEO
backend.

Manual setup example:

1. ``python3.12 -m venv /tmp/stac-interop``
2. ``source /tmp/stac-interop/bin/activate``
3. ``pip install pytest httpx openeo``
4. start the Open Climate Service separately, e.g. ``make run``
5. run:
   ``RUN_STAC_INTEROP=1 STAC_BASE_URL=http://127.0.0.1:9000 pytest --noconftest tests/integration/test_stac_interop.py``
"""

import os
import socket

import httpx
import pytest


def test_openeo_can_load_stac_collection() -> None:
    if os.getenv("RUN_STAC_INTEROP") != "1":
        pytest.skip("Set RUN_STAC_INTEROP=1 to enable live STAC interoperability checks")

    openeo = pytest.importorskip("openeo")
    base_url = os.getenv("STAC_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
    dataset_id = os.getenv("STAC_DATASET_ID")

    catalog_url = f"{base_url}/stac/catalog.json"
    with httpx.Client(timeout=10.0) as client:
        catalog_response = client.get(catalog_url)
        catalog_response.raise_for_status()
        catalog = catalog_response.json()
        assert isinstance(catalog, dict)

        child_links = [
            link for link in catalog.get("links", []) if isinstance(link, dict) and link.get("rel") == "child"
        ]
        assert child_links, "Catalog returned no child collection links"

        if dataset_id is None:
            collection_href = child_links[0].get("href")
            assert isinstance(collection_href, str)
        else:
            suffix = f"/stac/{dataset_id}/collection.json"
            collection_href = next(
                (
                    link.get("href")
                    for link in child_links
                    if isinstance(link.get("href"), str) and str(link.get("href")).endswith(suffix)
                ),
                None,
            )
            assert isinstance(collection_href, str), f"Dataset '{dataset_id}' not found in catalog child links"

        collection_response = client.get(collection_href)
        collection_response.raise_for_status()
        collection = collection_response.json()
        assert isinstance(collection, dict)
        assert collection.get("type") == "Collection"
        assert "cube:dimensions" in collection
        assert "assets" in collection

    original_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(15.0)
        cube = openeo.DataCube.load_stac(collection_href)
    finally:
        socket.setdefaulttimeout(original_timeout)

    assert cube is not None
    assert cube.metadata is not None
    assert cube.metadata.has_temporal_dimension()
    assert cube.metadata.temporal_dimension is not None
    temporal_dimension_name = cube.metadata.temporal_dimension.name
    dimension_names = cube.metadata.dimension_names()
    assert dimension_names
    assert temporal_dimension_name in dimension_names
