"""NASA Earthdata helpers shared by the NASA-hosted raster plugins.

Granules are found through CMR (public, no login). Downloads go through Earthdata Login (EDL):
the data host redirects to ``urs.earthdata.nasa.gov``, which needs the user's credentials,
then back to the data host with a session cookie. Credentials are read from ``~/.netrc``
(``machine urs.earthdata.nasa.gov login ... password ...``). An Earthdata user token in the
``EARTHDATA_TOKEN`` environment variable, if set, is sent as a bearer token instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

_CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
_EDL_HOST = "urs.earthdata.nasa.gov"
_TOKEN_ENV = "EARTHDATA_TOKEN"


def earthdata_session() -> requests.Session:
    """A session that authenticates through Earthdata Login using ``~/.netrc``.

    Plain ``requests`` redirect handling is what EDL needs: on each cross-host redirect it
    drops the previous host's credentials and applies the ``~/.netrc`` entry for the new host,
    so the password goes only to ``urs.earthdata.nasa.gov`` and the data host gets the
    session cookie on the way back.
    """
    session = requests.Session()
    session.trust_env = True
    token = os.getenv(_TOKEN_ENV)
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def cmr_granules(
    short_name: str, temporal: tuple[str, str], bbox: list[float] | None = None, **params: Any
) -> list[dict]:
    """Return CMR granule entries for a collection, oldest first, following pagination."""
    query: dict[str, Any] = {
        "short_name": short_name,
        "temporal": f"{temporal[0]},{temporal[1]}",
        "sort_key": "start_date",
        "page_size": 200,
        **params,
    }
    if bbox is not None:
        query["bounding_box"] = ",".join(str(float(v)) for v in bbox)
    entries: list[dict] = []
    headers: dict[str, str] = {}
    while True:
        response = requests.get(_CMR_GRANULES, params=query, headers=headers, timeout=60)
        response.raise_for_status()
        page = response.json()["feed"]["entry"]
        entries.extend(page)
        search_after = response.headers.get("CMR-Search-After")
        if not page or not search_after:
            return entries
        headers["CMR-Search-After"] = search_after


def data_links(granule: dict, suffix: str) -> list[str]:
    """HTTPS data links of a CMR granule entry ending in ``suffix``."""
    return [
        link["href"]
        for link in granule.get("links", [])
        if link.get("href", "").startswith("https://")
        and link["href"].endswith(suffix)
        and "opendap" not in link["href"]
    ]


def download(url: str, target: Path, session: requests.Session | None = None) -> Path:
    """Stream ``url`` to ``target`` through Earthdata Login, failing clearly on an auth error."""
    session = session or earthdata_session()
    with session.get(url, stream=True, timeout=300) as response:
        if response.status_code in (401, 403):
            raise PermissionError(
                f"Earthdata Login refused {url} ({response.status_code}). Put your credentials in ~/.netrc as "
                f"'machine {_EDL_HOST} login <user> password <pass>' (or set {_TOKEN_ENV}), and approve the "
                "data provider's application in your Earthdata profile."
            )
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise PermissionError(f"Earthdata Login returned a login page for {url}; check the ~/.netrc credentials")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return target
