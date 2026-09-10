"""The dataset-template detail route (CLIM-897).

`GET /dataset-templates/{id}` returned 500 for every template with nothing ingested, because
it derived coverage unconditionally and the last of the three data sources raised
`OSError: no files to open` on an empty glob.

The route had **no tests at all**, which is why it shipped: it works on a developer machine,
where you have ingested the thing you are working on, and fails only for templates you have
not — which is precisely the state of a fresh or demo instance. So the coverage here starts
with a whole-catalogue check rather than a single case.
"""

import pathlib
from typing import Any

import pytest
import xarray as xr
from fastapi.testclient import TestClient

from open_climate_service.data_accessor.services import accessor
from open_climate_service.data_manager.services import downloader


@pytest.fixture(autouse=True)
def _empty_store_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Point the data directory at an empty temp dir, so "fresh instance" is what it says.

    `DOWNLOAD_DIR` is resolved at module import, and `conftest.py` imports the app before its
    session fixture installs a temporary config — so without this the tests read the developer's
    real XDG stores. That made the whole-catalogue check host-dependent: slow, and passing or
    failing on which datasets happen to be ingested locally, with a corrupt local store failing
    it for a reason unrelated to the code.

    Redirecting the directory rather than stubbing the absent-data result keeps the real
    three-source fall-through in play — icechunk miss, zarr miss, empty glob — so the code under
    test still runs. Stubbing it would assert that a stub returns stubs.
    """
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", tmp_path)


def _cube() -> xr.Dataset:
    """A small real cube, so the actual coverage code runs rather than a stubbed answer."""
    import numpy as np
    import pandas as pd

    times = pd.date_range("2026-01-01", periods=3, freq="D")
    return xr.Dataset(
        {"precip": (("t", "y", "x"), np.ones((3, 2, 2), dtype="float32"))},
        coords={"t": times, "y": [1.5, 0.5], "x": [10.5, 11.5]},
    )


# -- the whole catalogue, which is the regression that matters ------------------------------


def test_every_listed_template_is_individually_fetchable(client: TestClient) -> None:
    """The guard the bug needed. Sixteen of seventeen templates on the Nepal demo returned 500
    while the list endpoint returned all of them, so a per-template check that happened to pick
    an ingested one would have passed."""
    listed = client.get("/dataset-templates/").json()
    assert listed, "no templates to check"

    responses = {t["id"]: client.get(f"/dataset-templates/{t['id']}") for t in listed}

    failures = {name: r.status_code for name, r in responses.items() if r.status_code != 200}
    assert not failures, f"templates not individually fetchable: {failures}"
    # Every one of them is un-ingested here, which is the state the bug was specific to, so the
    # answer is uniform rather than dependent on what this machine happens to hold.
    assert all(r.json()["has_data"] is False for r in responses.values())


# -- the three states of one template ------------------------------------------------------


def test_a_template_with_nothing_ingested_is_returned_with_null_coverage(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template is a declaration: it exists whether or not anyone ingested it. 404 would be
    wrong for the same reason 500 is."""

    def nothing_ingested(dataset: dict[str, Any], *args: object, **kwargs: object) -> xr.Dataset:
        raise accessor.DatasetDataUnavailable(f"{dataset['id']} has no ingested data")

    monkeypatch.setattr(accessor, "get_data", nothing_ingested)

    response = client.get("/dataset-templates/chirps3_precipitation_daily")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chirps3_precipitation_daily"
    assert body["has_data"] is False
    # Null rather than absent, so a client can tell "not ingested" from "not reported".
    assert body["coverage"]["temporal"] == {"start": None, "end": None}
    assert body["coverage"]["spatial_wgs84"] is None
    # The declaration itself still has to survive.
    assert body["units"] == "mm/d"


def test_a_template_with_data_still_reports_derived_coverage(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix must not turn coverage off for everyone — the failure mode of a too-broad
    `except`."""
    monkeypatch.setattr(accessor, "get_data", lambda *args, **kwargs: _cube())

    body = client.get("/dataset-templates/chirps3_precipitation_daily").json()

    assert body["has_data"] is True
    assert body["coverage"]["temporal"]["start"] is not None
    assert body["coverage"]["spatial"]["xmin"] is not None


def test_an_id_that_is_not_a_template_is_still_404(client: TestClient) -> None:
    assert client.get("/dataset-templates/definitely_not_a_template").status_code == 404


# -- the accessor contract underneath -------------------------------------------------------


def test_get_data_raises_a_typed_error_when_no_source_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drives the real three-source fall-through rather than the exception the route catches.

    Typed rather than the bare `OSError: no files to open` that `open_mfdataset` produced: that
    message named no dataset and could not be told apart from a genuinely unreadable store, so
    a caller had nothing safe to catch.
    """
    monkeypatch.setattr(accessor, "get_icechunk_path", lambda _: __import__("pathlib").Path("/nonexistent/store"))
    monkeypatch.setattr(accessor, "get_zarr_path", lambda _: None)
    monkeypatch.setattr(accessor, "get_cache_files", lambda _: [])

    with pytest.raises(accessor.DatasetDataUnavailable, match="has no ingested data"):
        accessor.get_data({"id": "never_ingested", "period_type": "daily"})


def test_reading_data_still_fails_while_describing_it_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """The asymmetry is the point. Serving data that does not exist must error; describing a
    template that has not been ingested must not."""
    monkeypatch.setattr(accessor, "get_icechunk_path", lambda _: __import__("pathlib").Path("/nonexistent/store"))
    monkeypatch.setattr(accessor, "get_zarr_path", lambda _: None)
    monkeypatch.setattr(accessor, "get_cache_files", lambda _: [])
    dataset = {"id": "never_ingested", "period_type": "daily"}

    with pytest.raises(accessor.DatasetDataUnavailable):
        accessor.get_data(dataset)

    assert accessor.get_data_coverage(dataset)["has_data"] is False


def test_an_unreadable_store_is_not_reported_as_absent_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store that exists but cannot be opened is a fault, not an empty catalogue entry.
    Catching every OSError here would have hidden it behind `has_data: False`."""

    def corrupt(*_args: object, **_kwargs: object) -> xr.Dataset:
        raise OSError("store is corrupt")

    monkeypatch.setattr(accessor, "get_data", corrupt)

    with pytest.raises(OSError, match="corrupt"):
        accessor.get_data_coverage({"id": "broken", "period_type": "daily"})
