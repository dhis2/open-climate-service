"""Run a derived dataset's recipe again when the dataset it refreshes on updates.

Dataset maintenance owns derivation; pipelines own delivery. A recipe with ``refresh_on``
(`data_registry.recipes`) is computed again, over the full coverage of its driving dataset,
after that dataset's ``dataset.updated`` event. The run is an openEO batch job of the recipe's
workflow, and the workflow publishes the derived dataset, which emits its own update through
the publication path, so a pipeline on the derived dataset runs next.

Only a derived dataset that exists is kept current: one an operator created once. A built-in
recipe therefore costs nothing on an instance that never asked for its dataset. Once it exists,
the driving dataset's latest update is refreshed if no refresh ran for it, also one from before
the dataset was created: a dataset created outside OCS costs one more full run.

Each refresh is one job whose id is derived from the output, the driving update, the recipe's
digest and the digest of the workflow definition it runs, frozen with its nested workflows. The
job carries that frozen graph, so it runs what was current when it was submitted, submitting it
again finds the same job, and a changed recipe or workflow is a new job.

Whether the latest update has its job is the whole test of being current. A time comparison,
or the input snapshot the output recorded, would take a refresh that started before a newer
update as covering it, though it computed only the older range. Only the latest update
matters, since every run recomputes the full range: while a refresh runs, later updates wait,
and the refresh's completion reconciles against the newest one. Reconciliation runs at startup
and on the maintenance interval too, so an update missed by a stopped process is caught up. A
refresh whose job failed or was cancelled is not submitted again by itself; the next update of
its input starts a new one.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from open_climate_service import config as api_config
from open_climate_service.data_registry import recipes
from open_climate_service.data_registry.recipes import OUTPUT_PARAMETER, Recipe
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo.schemas import OpenEOJobCreate, OpenEOJobRecord, OpenEOJobStatus, WorkflowRecord

logger = logging.getLogger(__name__)

TRIGGER_PREFIX = "derive:"
MAX_ATTEMPTS = 3
_ACTIVE = frozenset({OpenEOJobStatus.QUEUED, OpenEOJobStatus.RUNNING})


def trigger_id(output: str) -> str:
    """The trigger id every refresh job of ``output`` carries, which is how its jobs are found."""
    return f"{TRIGGER_PREFIX}{output}"


def _recipes() -> list[Recipe]:
    from open_climate_service.data_registry.services import datasets

    return [found for template in datasets.list_datasets() if (found := recipes.recipe(template)) is not None]


# --- validation --------------------------------------------------------------------------------


def _publishes_output(record: WorkflowRecord) -> bool:
    """Whether the workflow's result saves a managed Zarr dataset named by ``output_dataset_id``."""
    for node in record.process_graph.values():
        if not isinstance(node, dict) or not node.get("result") or node.get("process_id") != "save_result":
            continue
        arguments = node.get("arguments") or {}
        options = arguments.get("options") or {}
        return str(arguments.get("format", "")).lower() == "zarr" and options.get("dataset_id") == {
            "from_parameter": OUTPUT_PARAMETER
        }
    return False


def recipe_problems(recipe: Recipe) -> list[str]:
    """Why the recipe cannot run against its registered workflow; empty when it can."""
    from open_climate_service.openeo import workflows as workflow_registry
    from open_climate_service.pipelines import workflow as binding

    record = workflow_registry.get_workflow(recipe.workflow)
    if record is None:
        return [f"'{recipe.workflow}' is not a registered workflow"]
    problems: list[str] = []
    declared = {str(p["name"]): p for p in binding.parameters(record)}
    if OUTPUT_PARAMETER not in declared or not _publishes_output(record):
        problems.append(f"workflow '{recipe.workflow}' does not publish a dataset named by {OUTPUT_PARAMETER}")
    for name, value in recipe.arguments.items():
        if name not in declared:
            problems.append(f"'{name}' is not a parameter of workflow '{recipe.workflow}'")
        elif (problem := binding._type_problem(name, binding._schema(declared[name]), value)) is not None:
            problems.append(problem)
    intervals = binding.interval_parameters(record)
    open_intervals = [name for name in intervals if name not in recipe.arguments]
    if len(intervals) != 1:
        problems.append(f"workflow '{recipe.workflow}' must declare exactly one temporal-interval parameter")
    elif recipe.refresh_on is not None and not open_intervals:
        problems.append(f"leave '{intervals[0]}' out: each refresh fills it with the input's full coverage")
    elif recipe.refresh_on is None and open_intervals:
        problems.append(f"set '{intervals[0]}': a recipe without refresh_on runs over a fixed range")
    supplied = {*recipe.arguments, OUTPUT_PARAMETER, *open_intervals}
    missing = sorted(name for name, p in declared.items() if binding._required(p) and name not in supplied)
    if missing:
        problems.append(f"missing argument(s) {', '.join(missing)}")
    return problems


def _cycles(found: list[Recipe]) -> list[str]:
    """The derived datasets whose ``refresh_on`` chain leads back to themselves."""
    drives = {recipe.output: recipe.input for recipe in found if recipe.input is not None}
    looped: set[str] = set()
    for start in drives:
        seen: list[str] = []
        current = start
        while current in drives and current not in seen:
            seen.append(current)
            current = drives[current]
        if current in seen:
            looped.update(seen[seen.index(current) :])
    return sorted(looped)


def _trigger_outputs() -> set[str]:
    """Datasets a workflow trigger writes, which no recipe may also produce. Goes with CLIM-1426."""
    from open_climate_service.automation.config import get_automation_config
    from open_climate_service.automation.service import _resolved_output_dataset
    from open_climate_service.openeo import workflows as workflow_registry

    outputs: set[str] = set()
    for trigger in get_automation_config().workflow_triggers:
        workflow = workflow_registry.get_workflow(trigger.workflow_id)
        if workflow is not None and (output := _resolved_output_dataset(trigger, workflow)) is not None:
            outputs.add(output)
    return outputs


def validate_recipes() -> None:
    """Refuse startup with every recipe that cannot run, loops, or competes with a trigger."""
    found = _recipes()
    problems = [f"'{recipe.output}': {problem}" for recipe in found for problem in recipe_problems(recipe)]
    problems.extend(f"'{output}': its refresh_on chain leads back to itself" for output in _cycles(found))
    contested = _trigger_outputs()
    problems.extend(
        f"'{recipe.output}': a workflow trigger also writes it; remove one of the two"
        for recipe in found
        if recipe.output in contested
    )
    if problems:
        raise ValueError("Derived dataset recipes cannot run:\n- " + "\n- ".join(problems))


# --- refreshing -------------------------------------------------------------------------------


def _latest_updates() -> dict[str, JobEvent]:
    """Each dataset's latest persisted update."""
    from open_climate_service.jobs.events import persisted_events

    latest: dict[str, JobEvent] = {}
    for event in persisted_events():
        if event.type == DATASET_UPDATED_EVENT_TYPE and isinstance(dataset_id := event.data.get("dataset_id"), str):
            latest[dataset_id] = event
    return latest


def _exists(dataset_id: str) -> bool:
    from open_climate_service.ingestions.services import get_latest_artifact_for_dataset_or_404

    try:
        get_latest_artifact_for_dataset_or_404(dataset_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return False
        raise
    return True


def _refresh(recipe: Recipe, update: JobEvent | None) -> OpenEOJobRecord | None:
    """Submit the refresh ``update``, the driving dataset's latest, calls for, if one is due; return it.

    Checking the jobs and creating the next one happen under one lock per derived dataset,
    shared by every process on the data directory: two updates handled at once would otherwise
    each find no refresh running and start two. An update persisted after the caller read the
    log is handled by its own dispatch, or by the next reconciliation.
    """
    from open_climate_service.shared.persistence import index_lock

    if update is None:
        return None
    with index_lock(api_config.get_data_root() / "derivations" / recipe.output):
        return _refresh_locked(recipe, update)


def _refresh_locked(recipe: Recipe, update: JobEvent) -> OpenEOJobRecord | None:
    from open_climate_service.openeo import workflows as workflow_registry
    from open_climate_service.openeo.jobs import get_openeo_job_service, store_list_jobs, triggered_job_id
    from open_climate_service.pipelines import workflow as binding

    if not _exists(recipe.output):
        return None
    record = workflow_registry.get_workflow(recipe.workflow)
    if record is None:
        raise LookupError(f"workflow '{recipe.workflow}' is not registered")
    # One snapshot of the definition, nested workflows inlined, for the identity, the interval
    # and the submitted graph alike: the job runs what its id names, whatever is edited later,
    # and a changed workflow is another refresh of the same update.
    snapshot = binding.frozen(record)
    workflow_digest = binding.definition_digest(snapshot)
    source = f"{update.event_id}:{recipe.digest}:{workflow_digest}"
    service = get_openeo_job_service()
    job_id = triggered_job_id(source, trigger_id(recipe.output))
    jobs = [job for job in store_list_jobs() if job.trigger_id == trigger_id(recipe.output)]
    existing = next((job for job in jobs if job.id == job_id), None)
    if existing is not None:
        if existing.status == OpenEOJobStatus.CREATED:
            service.start_triggered_job(existing.id)  # created by a process that stopped before starting it
        return None  # running, done, or failed and left for an operator
    if any(job.status in _ACTIVE for job in jobs):
        return None  # its completion reconciles against the newest update
    open_intervals = [name for name in binding.interval_parameters(snapshot) if name not in recipe.arguments]
    if len(open_intervals) != 1:
        # Checked at startup, but a workflow can be edited while OCS runs.
        raise ValueError(
            f"workflow '{recipe.workflow}' must leave exactly one temporal-interval parameter open, "
            f"not {open_intervals}"
        )
    [interval] = open_intervals
    arguments: dict[str, Any] = {
        **recipe.arguments,
        OUTPUT_PARAMETER: recipe.output,
        interval: [update.data.get("current_start"), update.data.get("current_end")],
    }
    call = {"derive": {"process_id": recipe.workflow, "arguments": arguments, "result": True}}
    job, _ = service.create_triggered_job(
        OpenEOJobCreate(
            title=f"Refresh {recipe.output}",
            description=(
                f"Refreshes derived dataset '{recipe.output}' with workflow '{recipe.workflow}' after update "
                f"{update.event_id} of '{recipe.input}' (recipe {recipe.digest[:12]}, "
                f"workflow {workflow_digest[:12]})"
            ),
            process={
                "process_graph": workflow_registry.freeze_process_graph(
                    call, resolve=lambda name: snapshot if name == recipe.workflow else None
                )
            },
        ),
        source_event_id=source,
        trigger_id=trigger_id(recipe.output),
        max_attempts=MAX_ATTEMPTS,
    )
    service.start_triggered_job(job.id)
    logger.info("Refreshing derived dataset %s as job %s after update %s", recipe.output, job.id, update.event_id)
    return job


def _reconcile(only: set[str] | None = None) -> None:
    if api_config.is_read_only():
        return
    found = [recipe for recipe in _recipes() if recipe.input is not None]
    if only is not None:
        found = [recipe for recipe in found if recipe.input in only or recipe.output in only]
    if not found:
        return
    updates = _latest_updates()  # one read of the log for every recipe
    for recipe in found:
        assert recipe.input is not None
        try:
            _refresh(recipe, updates.get(recipe.input))
        except Exception:
            logger.exception("Derived dataset %s could not be refreshed", recipe.output)


def consume(events: list[JobEvent]) -> None:
    """Refresh the derived datasets whose driving dataset these events updated."""
    updated = {
        dataset_id
        for event in events
        if event.type == DATASET_UPDATED_EVENT_TYPE and isinstance(dataset_id := event.data.get("dataset_id"), str)
    }
    if updated:
        _reconcile(updated)


def reconcile() -> None:
    """Refresh every derived dataset whose driving dataset updated since it was last computed."""
    _reconcile()


def on_job_finished(job: OpenEOJobRecord) -> None:
    """Reconcile a derived dataset once its refresh ends, so an update that waited is taken up. Never raises."""
    trigger = job.trigger_id
    if not isinstance(trigger, str) or not trigger.startswith(TRIGGER_PREFIX):
        return
    try:
        _reconcile({trigger.removeprefix(TRIGGER_PREFIX)})
    except Exception:
        logger.exception("Derived dataset %s could not be reconciled", trigger.removeprefix(TRIGGER_PREFIX))
