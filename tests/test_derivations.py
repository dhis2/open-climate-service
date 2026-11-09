"""Derived dataset recipes: what a template may declare, and refreshing after the input updates."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from open_climate_service.data_registry import recipes
from open_climate_service.data_registry.recipes import Recipe
from open_climate_service.derivations import service as derivations
from open_climate_service.jobs import dispatch
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo import jobs as openeo_jobs
from open_climate_service.openeo import workflows
from open_climate_service.openeo.schemas import OpenEOJobRecord, OpenEOJobStatus
from tests.test_managed_publish_units import managed_instance  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tests.test_openeo_publication_events import (
    _DATASET,
    _job,
    service,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

_ANOMALY = Recipe(
    output="rain_anomaly",
    workflow="climate_anomaly",
    arguments={"observed_dataset_id": "rain", "normal_dataset_id": "rain_normal", "variable": "tp"},
    refresh_on="observed_dataset_id",
)


def test_both_forms_name_their_workflow_and_only_a_recipe_runs() -> None:
    named = {"id": "rain_normal", "produced_by": "climate_normal"}
    recipe = {"id": "rain_anomaly", "produced_by": {"workflow": "climate_anomaly", **_recipe_body()}}

    assert [recipes.producing_workflow(t) for t in (named, recipe)] == ["climate_normal", "climate_anomaly"]
    assert recipes.recipe(named) is None
    assert recipes.recipe(recipe) == _ANOMALY
    assert _ANOMALY.input == "rain"
    changed = replace(_ANOMALY, arguments={**_ANOMALY.arguments, "variable": "t2m"})
    assert changed.digest != _ANOMALY.digest


def _recipe_body() -> dict[str, Any]:
    return {"arguments": dict(_ANOMALY.arguments), "refresh_on": "observed_dataset_id"}


def test_startup_refuses_recipes_that_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    derivations.validate_recipes()  # the built-in recipes run

    broken = [
        Recipe(output="a", workflow="no_such_workflow", arguments={}),
        Recipe(output="b", workflow="aggregate_by_features", arguments={}),
        Recipe(output="c", workflow="climate_anomaly", arguments={**_ANOMALY.arguments, "unknown": 1}),
        Recipe(output="d", workflow="climate_anomaly", arguments={"observed_dataset_id": "rain"}),
        Recipe(output="e", workflow="climate_anomaly", arguments=dict(_ANOMALY.arguments)),
        Recipe(
            output="f",
            workflow="climate_anomaly",
            arguments={**_ANOMALY.arguments, "temporal_extent": ["2020-01", "2020-12"]},
            refresh_on="observed_dataset_id",
        ),
        Recipe(
            output="g",
            workflow="climate_anomaly",
            arguments={**_ANOMALY.arguments, "observed_dataset_id": "h"},
            refresh_on="observed_dataset_id",
        ),
        Recipe(
            output="h",
            workflow="climate_anomaly",
            arguments={**_ANOMALY.arguments, "observed_dataset_id": "g"},
            refresh_on="observed_dataset_id",
        ),
    ]
    monkeypatch.setattr(derivations, "_recipes", lambda: broken)

    with pytest.raises(ValueError) as refused:
        derivations.validate_recipes()

    message = str(refused.value)
    for expected in (
        "'a': 'no_such_workflow' is not a registered workflow",
        "'b': workflow 'aggregate_by_features' does not publish a dataset named by output_dataset_id",
        "'c': 'unknown' is not a parameter of workflow 'climate_anomaly'",
        "'d': missing argument(s) normal_dataset_id, variable",
        "'e': set 'temporal_extent': a recipe without refresh_on runs over a fixed range",
        "'f': leave 'temporal_extent' out",
        "'g': its refresh_on chain leads back to itself",
        "'h': its refresh_on chain leads back to itself",
    ):
        assert expected in message


class _FakeOpenEO:
    """Records the jobs refreshes create; nothing runs until a test says how it ended."""

    def __init__(self) -> None:
        self.jobs: dict[str, OpenEOJobRecord] = {}

    def create_triggered_job(
        self, body: Any, *, source_event_id: str, trigger_id: str, max_attempts: int = 1
    ) -> tuple[OpenEOJobRecord, bool]:
        job_id = openeo_jobs.triggered_job_id(source_event_id, trigger_id)
        if job_id in self.jobs:
            return self.jobs[job_id], False
        self.jobs[job_id] = OpenEOJobRecord(
            id=job_id,
            title=body.title,
            description=body.description,
            process=body.process,
            status=OpenEOJobStatus.CREATED,
            created=datetime.now(UTC),
            trigger_id=trigger_id,
            source_event_id=source_event_id,
            max_attempts=max_attempts,
        )
        return self.jobs[job_id], True

    def start_triggered_job(self, job_id: str) -> bool:
        self.set(job_id, OpenEOJobStatus.QUEUED)
        return True

    def set(self, job_id: str, status: OpenEOJobStatus) -> OpenEOJobRecord:
        self.jobs[job_id] = self.jobs[job_id].model_copy(update={"status": status})
        return self.jobs[job_id]


def _update(number: int, dataset_id: str = "rain") -> JobEvent:
    return JobEvent(
        event_id=f"sync-{number}:0",
        time=datetime(2026, 11, 1, tzinfo=UTC) + timedelta(days=number),
        type=DATASET_UPDATED_EVENT_TYPE,
        source=f"/datasets/{dataset_id}",
        data={"dataset_id": dataset_id, "current_start": "1991-01", "current_end": f"2026-0{number}"},
    )


class _Instance:
    """Recipes, the persisted event log and the derived datasets that exist, as a test sets them."""

    def __init__(self) -> None:
        self.openeo = _FakeOpenEO()
        self.log: list[JobEvent] = []
        self.created: set[str] = set()
        self.recipes = [_ANOMALY]


@pytest.fixture
def instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Instance:
    from open_climate_service.jobs import events as job_events

    state = _Instance()
    monkeypatch.setattr(derivations.api_config, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(openeo_jobs, "get_openeo_job_service", lambda: state.openeo)
    monkeypatch.setattr(openeo_jobs, "store_list_jobs", lambda: list(state.openeo.jobs.values()))
    monkeypatch.setattr(job_events, "persisted_events", lambda: list(state.log))
    monkeypatch.setattr(derivations, "_recipes", lambda: state.recipes)
    monkeypatch.setattr(derivations, "_exists", lambda dataset_id: dataset_id in state.created)
    return state


def _graph(end: str, recipe: Recipe = _ANOMALY) -> dict[str, Any]:
    """The frozen graph a refresh of ``recipe`` over 1991-01 to ``end`` submits."""
    arguments = {**recipe.arguments, "output_dataset_id": recipe.output, "temporal_extent": ["1991-01", end]}
    return workflows.freeze_process_graph(
        {"derive": {"process_id": recipe.workflow, "arguments": arguments, "result": True}}
    )


def test_a_derived_dataset_is_refreshed_once_per_latest_update_of_its_input(
    instance: _Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, log, created, current = instance.openeo, instance.log, instance.created, instance.recipes

    def updated(number: int) -> None:
        log.append(_update(number))
        derivations.consume([log[-1]])

    updated(1)
    assert fake.jobs == {}  # never created, so nothing to keep current
    created.add("rain_anomaly")
    derivations.reconcile()  # at startup, the update it missed
    [first] = fake.jobs.values()
    assert first.status == OpenEOJobStatus.QUEUED and first.trigger_id == "derive:rain_anomaly"
    assert first.process["process_graph"] == _graph("2026-01")
    derivations.consume([log[-1]])
    updated(2)
    updated(3)
    assert list(fake.jobs) == [first.id]  # one refresh at a time
    fake.set(first.id, OpenEOJobStatus.FINISHED)
    derivations.on_job_finished(fake.jobs[first.id])
    [_, latest] = fake.jobs.values()
    assert latest.process["process_graph"] == _graph("2026-03")
    fake.set(latest.id, OpenEOJobStatus.ERROR)
    derivations.reconcile()
    assert len(fake.jobs) == 2  # a failed refresh is left for an operator
    current[0] = replace(_ANOMALY, arguments={**_ANOMALY.arguments, "variable": "t2m"})
    derivations.reconcile()
    assert len(fake.jobs) == 3  # a changed recipe is another refresh of the same update
    fake.set(list(fake.jobs)[-1], OpenEOJobStatus.FINISHED)
    registered = workflows.get_workflow
    edited = registered("climate_anomaly").model_copy(  # type: ignore[union-attr]
        update={"description": "edited", "process_graph": {**registered("climate_anomaly").process_graph, "note": {}}}  # type: ignore[union-attr]
    )
    monkeypatch.setattr(
        workflows, "get_workflow", lambda name: edited if name == "climate_anomaly" else registered(name)
    )
    derivations.reconcile()
    assert len(fake.jobs) == 4  # so is a changed workflow
    monkeypatch.setattr(derivations.api_config, "is_read_only", lambda: True)
    updated(4)
    assert len(fake.jobs) == 4


def test_updates_handled_at_once_start_one_refresh(instance: _Instance, monkeypatch: pytest.MonkeyPatch) -> None:
    instance.created.add("rain_anomaly")
    creating = threading.Event()
    create = instance.openeo.create_triggered_job

    def slow_create(*args: Any, **kwargs: Any) -> tuple[OpenEOJobRecord, bool]:
        creating.set()
        time.sleep(0.2)  # long enough for the second update to look while the first creates
        return create(*args, **kwargs)

    monkeypatch.setattr(instance.openeo, "create_triggered_job", slow_create)
    instance.log.append(_update(1))
    first = threading.Thread(target=derivations.consume, args=([instance.log[-1]],))
    first.start()
    assert creating.wait(5)
    instance.log.append(_update(2))
    derivations.consume([instance.log[-1]])
    first.join(5)

    assert not first.is_alive()
    assert len(instance.openeo.jobs) == 1


def test_one_failing_consumer_does_not_keep_events_from_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[str] = []

    def fail(_: list[JobEvent]) -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(
        dispatch,
        "_consumers",
        lambda: [("first", fail), ("second", lambda events: received.extend(e.event_id for e in events))],
    )

    dispatch.consume([_update(1)])

    assert received == ["sync-1:0"]


def test_an_input_update_refreshes_the_derived_dataset_and_its_pipeline_hears_of_it_across_a_restart(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real job service and publication; only the graph's computed result is stubbed.

    Stops where the pipeline runner receives the derived dataset's update: that an update starts
    one run, and that a run delivers once, are `test_a_dataset_update_runs_the_same_path_once`
    and `test_run_once_stores_its_intent_first_and_delivers_once` in test_pipelines.py.
    """
    first_process = request.getfixturevalue("service")
    from open_climate_service.jobs import store as native_store
    from open_climate_service.jobs.models import JobRecord, JobStatus
    from open_climate_service.pipelines import runner
    from open_climate_service.shared.time import utc_now

    recipe = replace(_ANOMALY, output=_DATASET)
    monkeypatch.setattr(derivations, "_recipes", lambda: [recipe])
    running = [first_process]
    monkeypatch.setattr(openeo_jobs, "get_openeo_job_service", lambda: running[0])
    heard: list[str] = []
    monkeypatch.setattr(
        runner,
        "consume",
        lambda events: heard.extend(e.data["dataset_id"] for e in events if e.type == "dataset.updated"),
    )

    def started(openeo: Any) -> None:
        openeo.set_event_consumer(dispatch.consume)
        openeo.set_finished_listener(derivations.on_job_finished)

    def synced(number: int) -> JobEvent:
        """The input's update, persisted with its sync job."""
        event = _update(number)
        native_store.create_job_record(
            JobRecord(
                job_id=f"sync-{number}",
                process_id="sync",
                status=JobStatus.SUCCESSFUL,
                created_at=utc_now(),
                events=[event],
            )
        )
        return event

    def refreshes(expected: int) -> list[OpenEOJobRecord]:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            done = [
                job
                for job in openeo_jobs.store_list_jobs()
                if job.trigger_id == f"derive:{_DATASET}" and job.status == OpenEOJobStatus.FINISHED
            ]
            if len(done) >= expected and heard.count(_DATASET) >= expected:
                return done
            time.sleep(0.05)
        pytest.fail(f"expected {expected} finished refreshes of {_DATASET}")

    started(first_process)
    _job("created-by-hand", 1.0, monkeypatch)  # the operator creates the derived dataset once
    first_process._execute("created-by-hand")
    heard.clear()

    dispatch.consume([synced(1)])
    [first] = refreshes(1)
    assert first.process["process_graph"] == workflows.freeze_process_graph(
        {
            "derive": {
                "process_id": "climate_anomaly",
                "arguments": {
                    **recipe.arguments,
                    "output_dataset_id": _DATASET,
                    "temporal_extent": ["1991-01", "2026-01"],
                },
                "result": True,
            }
        }
    )

    first_process.shutdown()
    synced(2)  # while no process runs
    restarted = openeo_jobs.OpenEOJobService()
    running[0] = restarted
    try:
        started(restarted)
        derivations.reconcile()  # at startup
        assert len(refreshes(2)) == 2
        # The pipelines hear the input's update, then each refresh's publication.
        assert heard == ["rain", _DATASET, _DATASET]
    finally:
        restarted.shutdown()
