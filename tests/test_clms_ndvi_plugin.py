"""Tests for the CLMS NDVI dekadal plugin, with the STAC catalogue faked and a local GeoTIFF."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NoReturn

import httpx
import numpy as np
import pytest

from open_climate_service.plugins.rasters.clms_ndvi import CLMSNDVIDekadalPlugin

_BBOX = [29.0, -2.0, 30.0, -1.0]


def _write_dekad(path: Path) -> str:
    """Write a 10 x 10 GeoTIFF encoded like the CLMS files: uint8, scale 0.004, offset -0.08.

    The first row holds the edges of the valid range (0 and 250) and every value above it
    (251-255); the rest is 200, an NDVI of 0.72.
    """
    import rasterio
    from rasterio.transform import from_origin

    values = np.full((10, 10), 200, dtype="uint8")
    values[0, :7] = [0, 250, 251, 252, 253, 254, 255]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(29.0, -1.0, 0.1, 0.1),
        nodata=255,
    ) as dst:
        dst.write(values, 1)
        dst.scales = (0.004,)
        dst.offsets = (-0.08,)
    return str(path)


def _item(dekad: str, href: str) -> dict[str, Any]:
    return {"properties": {"start_datetime": f"{dekad}T00:00:00.000000Z"}, "assets": {"ndvi": {"href": href}}}


def _fake_catalogue(monkeypatch: pytest.MonkeyPatch, pages: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Answer catalogue searches with ``pages``, linked by ``next``; return the requests made."""
    requests: list[dict[str, Any]] = []

    def get(url: str, params: dict[str, Any] | None = None, **_: Any) -> httpx.Response:
        requests.append({"url": url, "params": params})
        page = int(url.rsplit("page=", 1)[1]) if "page=" in url else 0
        links = [{"rel": "next", "href": f"{url.split('?')[0]}?page={page + 1}"}] if page + 1 < len(pages) else []
        body = {"features": pages[page], "links": links}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    return requests


def test_periods_stop_at_the_newest_published_dekad(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _fake_catalogue(monkeypatch, [[_item("2026-10-11", "unused")]])

    periods = asyncio.run(CLMSNDVIDekadalPlugin().periods("2026-09-15", "2026-12-31"))

    assert periods == ["2026-09-11", "2026-09-21", "2026-10-01", "2026-10-11"]
    assert requests[0]["params"]["sortby"] == "-datetime"


def test_fetch_period_masks_flags_and_scales_to_ndvi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_catalogue(monkeypatch, [[_item("2024-01-11", _write_dekad(tmp_path / "ndvi.tif"))]])

    ds = CLMSNDVIDekadalPlugin().fetch_period("2024-01-11", [29.0, -1.45, 29.65, -1.0])
    ndvi = ds["ndvi"]

    # Clipped to the bbox, one time step stamped with the dekad's first day.
    assert dict(ds.sizes) == {"t": 1, "y": 5, "x": 7}
    assert str(ds["t"].values[0])[:10] == "2024-01-11"
    # The edges of the valid range scale to -0.08 and 0.92; every value above 250 is empty.
    first_row = ndvi.isel(t=0, y=0).values
    assert first_row[:2] == pytest.approx([-0.08, 0.92], abs=1e-6)
    assert np.isnan(first_row[2:]).all()
    assert float(ndvi.isel(t=0, y=1, x=0)) == pytest.approx(0.72, abs=1e-6)
    # The raw file's encoding attributes describe bytes the result no longer holds.
    assert ndvi.dtype == np.float32
    assert ndvi.attrs == {"units": "1", "long_name": "NDVI"}
    assert ds.rio.crs.to_epsg() == 4326


def test_parallel_fetches_share_one_paged_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    href = _write_dekad(tmp_path / "ndvi.tif")
    dekads = ["2024-01-01", "2024-01-11", "2024-01-21", "2024-02-01"]
    pages = [[_item(dekad, href) for dekad in dekads[:2]], [_item(dekad, href) for dekad in dekads[2:]]]
    requests = _fake_catalogue(monkeypatch, pages)
    plugin = CLMSNDVIDekadalPlugin()

    with ThreadPoolExecutor(plugin.max_concurrency) as pool:
        fetched = list(pool.map(lambda dekad: plugin.fetch_period(dekad, _BBOX), dekads))

    assert [str(ds["t"].values[0])[:10] for ds in fetched] == dekads
    # One search over two pages; the first request carries the bbox and the trimmed fields.
    assert len(requests) == 2
    assert requests[0]["params"]["bbox"] == "29.0,-2.0,30.0,-1.0"
    assert requests[0]["params"]["fields"] == "id,properties.start_datetime,assets.ndvi.href"


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "No CLMS NDVI composite published"),
        ([_item("2024-01-11", "a.tif"), _item("2024-01-11", "b.tif")], "does not mosaic tiles"),
    ],
)
def test_fetch_period_refuses_a_dekad_without_exactly_one_item(
    monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]], message: str
) -> None:
    _fake_catalogue(monkeypatch, [items])

    with pytest.raises(RuntimeError, match=message):
        CLMSNDVIDekadalPlugin().fetch_period("2024-01-11", _BBOX)


def test_refused_s3_read_names_the_cdse_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import rioxarray
    from rasterio.errors import RasterioIOError

    def refuse(*_: Any, **__: Any) -> NoReturn:
        raise RasterioIOError("HTTP response code: 403 - AccessDenied")

    _fake_catalogue(monkeypatch, [[_item("2024-01-11", "s3://eodata/ndvi.tiff")]])
    monkeypatch.setattr(rioxarray, "open_rasterio", refuse)

    with pytest.raises(PermissionError, match=r"\[cdse\] profile"):
        CLMSNDVIDekadalPlugin().fetch_period("2024-01-11", _BBOX)
