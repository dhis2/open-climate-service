"""A template with no ingestion plugin says so, and refuses politely (CLIM-912).

Half the shipped catalogue is *produced* rather than fetched: anomalies, normals and change
rasters are written by a workflow through `save_result` and registered as static templates.
They appeared in `GET /dataset-templates/` beside ingestable ones with nothing to tell them
apart, so an operator found out by picking one and getting

    500: Dataset 'worldpop_population_change' does not define ingestion.plugin

A 500 says the server broke and invites a retry. Nothing was broken: the template is valid and
the request was well formed, it just named a dataset with no upstream.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.ingestions import services as ingestion_services


def _template(**overrides: Any) -> dict[str, Any]:
    return {"id": "t", "period_type": "daily", **overrides}


# -- which templates can be ingested --------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        pytest.param(_template(ingestion={"plugin": "pkg.mod"}), True, id="declares-a-plugin"),
        pytest.param(_template(), False, id="no-ingestion-block"),
        pytest.param(_template(ingestion={}), False, id="empty-ingestion-block"),
        pytest.param(_template(ingestion={"plugin": ""}), False, id="blank-plugin"),
        pytest.param(_template(ingestion={"plugin": "   "}), False, id="whitespace-plugin"),
        pytest.param(_template(ingestion="pkg.mod"), False, id="ingestion-not-a-mapping"),
    ],
)
def test_ingestability_is_the_presence_of_a_plugin(template: dict[str, Any], expected: bool) -> None:
    assert registry.is_ingestable(template) is expected


def test_a_static_template_with_a_plugin_is_still_ingestable() -> None:
    """`sync.kind` looks like the same question and is not. Keying on it would refuse
    `era5land_temperature_daily_normal_1991_2020`, which is static and ingests perfectly well —
    so the flag has to follow the plugin, which is what `create_artifact` requires."""
    assert registry.is_ingestable(_template(sync={"kind": "static"}, ingestion={"plugin": "pkg.mod"})) is True
    assert registry.is_ingestable(_template(sync={"kind": "temporal"})) is False


def test_the_shipped_catalogue_splits_both_ways() -> None:
    """Guards the guard: if every built-in became ingestable the tests above would still pass
    while the listing had nothing left to distinguish."""
    flags = {registry.is_ingestable(t) for t in registry.list_datasets()}
    assert flags == {True, False}


# -- the listing says so --------------------------------------------------------------------


def test_every_listed_template_reports_ingestability(client: TestClient) -> None:
    payload = client.get("/dataset-templates/").json()
    assert payload, "no templates listed"
    assert all("ingestable" in t for t in payload)
    assert all(t["ingestable"] == registry.is_ingestable(t) for t in payload)


def test_a_single_template_reports_it_too(client: TestClient) -> None:
    """The detail route is where an operator looks before ingesting one."""
    listed = client.get("/dataset-templates/").json()
    derived = next(t["id"] for t in listed if not t["ingestable"])
    assert client.get(f"/dataset-templates/{derived}").json()["ingestable"] is False


# -- and asking anyway is a 4xx, on both request shapes ---------------------------------------


def test_the_async_ingest_refuses_before_it_enqueues(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`Prefer: respond-async` returns 202 and reports everything afterwards through the job
    record. A check that lived only in `create_artifact` therefore accepted the request, ran in
    the worker, and failed the job — logging a traceback for a client mistake and putting the
    400 somewhere the caller was not looking."""
    from open_climate_service.extents import services as extent_services
    from open_climate_service.jobs import service as job_service

    monkeypatch.setattr(
        extent_services, "get_extent_or_404", lambda: {"bbox": [-13.5, 6.9, -10.1, 10.0], "country_code": "SLE"}
    )
    submitted: list[object] = []
    monkeypatch.setattr(
        job_service.JobService,
        "submit_callable_job",
        lambda self, **kw: submitted.append(kw) or (_ for _ in ()).throw(AssertionError("enqueued")),
    )

    response = client.post(
        "/ingestions",
        json={"dataset_id": "worldpop_population_change"},
        headers={"Prefer": "respond-async"},
    )

    assert response.status_code == 400, response.text
    assert "cannot be ingested" in response.json()["detail"]
    assert submitted == [], "the job was enqueued before the template was checked"


def test_the_async_ingest_still_accepts_an_ingestable_template(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not close the door on the ordinary case."""
    from open_climate_service.extents import services as extent_services

    monkeypatch.setattr(
        extent_services, "get_extent_or_404", lambda: {"bbox": [-13.5, 6.9, -10.1, 10.0], "country_code": "SLE"}
    )
    response = client.post(
        "/ingestions",
        json={"dataset_id": "chirps3_precipitation_daily"},
        headers={"Prefer": "respond-async"},
    )
    assert response.status_code == 202, response.text


# -- and asking anyway is a 4xx --------------------------------------------------------------


def test_ingesting_a_derived_template_is_a_client_error_not_a_server_error() -> None:
    """The whole point: nothing here is a fault, so it must not be reported as one."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        ingestion_services.create_artifact(
            dataset=_template(id="worldpop_population_change", sync={"kind": "static"}),
            start=None,
            end=None,
            bbox=None,
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert exc_info.value.status_code == 400
    detail = str(exc_info.value.detail)
    assert "worldpop_population_change" in detail
    assert "ingestion.plugin" in detail
    assert "save_result" in detail, "the message should say what does produce it"
