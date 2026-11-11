"""Run a derived dataset's recipe again when the dataset it refreshes on updates.

Dataset maintenance owns derivation; pipelines own delivery. A recipe with ``refresh_on``
(`data_registry.recipes`) is computed again, over the full coverage of its driving dataset,
after that dataset's ``dataset.updated`` event. The run is an openEO batch job of the recipe's
workflow, and the workflow publishes the derived dataset, which emits its own update through
the publication path, so a pipeline on the derived dataset runs next.

Only a derived dataset that exists is kept current: one an operator created once. A built-in
recipe therefore costs nothing on an instance that never asked for its dataset. The run that
created it counts as current for the driving dataset's updates it published after, so creating
it does not start a second full run at once; a dataset that predates the latest update is
refreshed.

Each refresh is one job whose id is derived from the output, the driving update, the recipe's
digest and the digest of the workflow definition it runs, frozen with its nested workflows. The
job carries that frozen graph, so it runs what was current when it was submitted, submitting it
again finds the same job, and a changed recipe or workflow is a new job.

Whether the latest update has its job is the whole test of being current. A time comparison,
or the input snapshot the output recorded, would take a refresh that started before a newer
update as covering it, though it computed only the older range. Only the latest update
matters, since every run recomputes the full range: while a refresh runs, later updates wait,
and a finished refresh reconciles against the newest one; after a failed one, the next
reconciliation does. Reconciliation runs at startup
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


def _templates() -> list[dict[str, Any]]:
    from open_climate_service.data_registry.services import datasets

    return datasets.list_datasets()


def _recipes() -> list[Recipe]:
    return [found for template in _templates() if (found := recipes.recipe(template)) is not None]


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
        elif (problem := binding.type_problem(name, binding.parameter_schema(declared[name]), value)) is not None:
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
    missing = sorted(name for name, p in declared.items() if binding.is_required(p) and name not in supplied)
    if missing:
        problems.append(f"missing argument(s) {', '.join(missing)}")
    return problems


def _cycles(found: list[Recipe], triggers: set[tuple[str, str]]) -> list[str]:
    """The recipe outputs whose updates lead back to themselves, through recipes or triggers.

    ``triggers`` holds each workflow trigger's (input, output): the trigger side refuses its
    own cycles, so only a loop that passes through a recipe is left to find here.
    """
    feeds: dict[str, set[str]] = {}
    for source, output in {*triggers, *((r.input, r.output) for r in found if r.input is not None)}:
        feeds.setdefault(source, set()).add(output)

    def returns_to(start: str) -> bool:
        seen: set[str] = set()
        pending = list(feeds.get(start, ()))
        while pending:
            current = pending.pop()
            if current == start:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(feeds.get(current, ()))
        return False

    return sorted({recipe.output for recipe in found if recipe.input is not None and returns_to(recipe.output)})


def _trigger_edges() -> set[tuple[str, str]]:
    """Each workflow trigger's (input, output) where the output is known. Goes with CLIM-1426."""
    from open_climate_service.automation.config import get_automation_config
    from open_climate_service.automation.service import _resolved_output_dataset
    from open_climate_service.openeo import workflows as workflow_registry

    edges: set[tuple[str, str]] = set()
    for trigger in get_automation_config().workflow_triggers:
        workflow = workflow_registry.get_workflow(trigger.workflow_id)
        if workflow is not None and (output := _resolved_output_dataset(trigger, workflow)) is not None:
            edges.add((trigger.on_update_of, output))
    return edges


def validate_recipes() -> None:
    """Refuse startup with every recipe that cannot run, loops, or competes with a trigger."""
    found = _recipes()
    problems = [f"'{recipe.output}': {problem}" for recipe in found for problem in recipe_problems(recipe)]
    triggers = _trigger_edges()
    known = {str(template["id"]) for template in _templates()} | {output for _, output in triggers}
    problems.extend(
        f"'{recipe.output}': refresh_on names '{recipe.input}', which is not a dataset template"
        for recipe in found
        if recipe.input is not None and recipe.input not in known
    )
    problems.extend(
        f"'{output}': its updates lead back to itself through recipes or workflow triggers"
        for output in _cycles(found, triggers)
    )
    contested = {output for _, output in triggers}
    problems.extend(
        f"'{recipe.output}': a workflow trigger also writes it; remove one of the two"
        for recipe in found
        if recipe.output in contested
    )
    if problems:
        raise ValueError("Derived dataset recipes cannot run:\n- " + "\n- ".join(problems))


# --- refreshing -------------------------------------------------------------------------------


def _latest_updates() -> dict[str, tuple[int, JobEvent]]:
    """Each dataset's latest persisted update, with its position in the log."""
    from open_climate_service.jobs.events import persisted_events

    latest: dict[str, tuple[int, JobEvent]] = {}
    for position, event in enumerate(persisted_events()):
        if event.type == DATASET_UPDATED_EVENT_TYPE and isinstance(dataset_id := event.data.get("dataset_id"), str):
            latest[dataset_id] = (position, event)
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


def _refresh(recipe: Recipe) -> OpenEOJobRecord | None:
    """Submit the refresh the driving dataset's latest update calls for, if one is due; return it.

    Reading the latest update, checking the jobs and creating the next one happen under one
    lock per derived dataset, shared by every process on the data directory. Two updates
    handled at once would otherwise each find no refresh running and start two, and an update
    read before the lock could be older than one another caller already refreshed.
    """
    from open_climate_service.shared.persistence import index_lock

    with index_lock(api_config.get_data_root() / "derivations" / recipe.output):
        return _refresh_locked(recipe)


def _refresh_locked(recipe: Recipe) -> OpenEOJobRecord | None:
    from open_climate_service.openeo import workflows as workflow_registry
    from open_climate_service.openeo.jobs import get_openeo_job_service, store_list_jobs, triggered_job_id
    from open_climate_service.pipelines import workflow as binding

    if recipe.input is None or not _exists(recipe.output):
        return None
    updates = _latest_updates()
    if recipe.input not in updates:
        return None
    position, update = updates[recipe.input]
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
    created = updates.get(recipe.output)
    if not jobs and created is not None and created[0] > position:
        return None  # the run that created it published after this update
    if any(job.status in _ACTIVE for job in jobs):
        return None  # its completion reconciles against the newest update
    existing = next((job for job in jobs if job.id == job_id), None)
    if existing is not None:
        if existing.status == OpenEOJobStatus.CREATED:
            service.start_triggered_job(existing.id)  # created by a process that stopped before starting it
        return None  # done, or failed and left for an operator
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
    for recipe in found:
        try:
            _refresh(recipe)
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
