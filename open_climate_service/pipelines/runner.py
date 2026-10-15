"""The one way a pipeline runs: `run_pipeline`, whoever asks.

An operator's run, a retry and a source dataset update all resolve to a `RunContext` with an
explicit interval and go through `run_pipeline`. It checks the pipeline may run, stores the
run with its deterministic job id *before* the job exists, then submits the workflow as an
openEO batch job. When the job finishes, `on_job_finished` hands the saved result to the
export delivery framework, the same path as ``POST /exports/{id}``, under the run's
idempotency key.

Recovery. `recover_runs` finishes what an interruption or a transient failure left open:
it resubmits pending runs, hands off finished jobs whose completion was missed, records
jobs that ended without a result, retries transient hand-off failures, and releases held
deliveries that are eligible again. Transient failures are retried a bounded number of
times; permanent ones are not retried at all. It runs at startup and periodically.

Dataset updates. Each pipeline keeps a durable checkpoint of the updates it has decided on
(`EventCheckpoint`), recorded in the same write as the run an update starts. `consume`
decides newly persisted updates; `reconcile_events`, at startup, decides any persisted
update that was missed and advances the checkpoint only as far as no update still to be
persisted can precede.

Pause semantics: a paused pipeline starts no run by itself and sends nothing by itself.
Jobs already running are not cancelled; they finish, their results are kept, and the
delivery they owe is held. Resuming releases the held deliveries that are eligible: dry-run
ones on any resume, live ones once the pipeline is live again with the bindings they were
computed for. An operator can always deliver a held dry run by hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException

from open_climate_service import config as api_config
from open_climate_service.pipelines import service, store
from open_climate_service.pipelines import workflow as workflow_binding
from open_climate_service.pipelines.schemas import PipelineRecord, PipelineRun, RunTrigger
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

_RUNS_KEPT = 50
TRIGGER_PREFIX = "pipeline:"
# Attempts per run job, the first included; the openEO job service retries transient failures.
MAX_ATTEMPTS = 3
# Attempts to submit a run's job, and to hand a finished job to delivery, before giving up.
MAX_SUBMISSION_ATTEMPTS = 5
MAX_DELIVERY_ATTEMPTS = 5
# Attempts to decide a dataset update whose run could not be started for a transient reason.
MAX_EVENT_ATTEMPTS = 5
# Skipped updates kept on the record for an operator.
_SKIPPED_KEPT = 50
# HTTP statuses that say "not now" rather than "never".
_RETRYABLE_STATUSES = frozenset({408, 425, 429})
# How far behind the safe bound the event checkpoint stays: covers the moment between an
# update's time being taken and its job record being written.
_EVENT_MARGIN = timedelta(minutes=15)


class RunRefused(ValueError):
    """The pipeline may not run, or deliver, in its current state."""


@dataclass(frozen=True)
class RunContext:
    """Why a run happens and over which interval. Every run has an explicit interval."""

    start: str
    end: str
    trigger: RunTrigger = "manual"
    mode: Literal["dry_run", "live"] | None = None
    event_id: str | None = None
    event_time: str | None = None
    retry_of: str | None = None
    source: dict[str, Any] = field(default_factory=dict)


def trigger_id(pipeline_id: str) -> str:
    """The openEO trigger id a pipeline's jobs carry."""
    return f"{TRIGGER_PREFIX}{pipeline_id}"


def _source_event(run: PipelineRun) -> str:
    """The durable event a run's job is deterministic on: the dataset update, or the run itself."""
    return run.event_id or f"pipeline-run:{run.run_id}"


def _message(exc: BaseException) -> str:
    detail = getattr(exc, "detail", None)
    return str(detail) if detail else str(exc) or type(exc).__name__


def _is_permanent(exc: BaseException) -> bool:
    """Whether repeating the same call can only fail the same way.

    A refusal (an HTTP 4xx: a changed export, a manifest mismatch, an invalid graph, a reused
    key) or an invalid value is permanent. A 408, 425 or 429, and a conflict that only says a
    result or an export is busy (`ResourceBusy`), say "not now" and are retried, as is
    anything else: the store, a lock, the network.
    """
    from open_climate_service.exports.retention import ResourceBusy

    if isinstance(exc, ResourceBusy):
        return False
    if isinstance(exc, HTTPException):
        return 400 <= exc.status_code < 500 and exc.status_code not in _RETRYABLE_STATUSES
    return isinstance(exc, (ValueError, TypeError, KeyError, LookupError))


def _refusal(record: PipelineRecord, context: RunContext, mode: str) -> str | None:
    """Why the pipeline may not start this run now, or None."""
    spec = record.spec
    if record.validation is None or not record.validation.valid:
        return "the pipeline must validate before it can run; re-check it"
    if record.validation.revision != record.revision:
        return "the pipeline changed since it was last validated; re-check it"
    if context.trigger == "dataset_update" and not store.automates(spec):
        return f"the pipeline does not run on updates (mode {spec.delivery.mode}, policy {spec.delivery.policy})"
    if mode == "live":
        if spec.delivery.mode != "live":
            return f"a live run needs the pipeline to be live; it is {spec.delivery.mode.replace('_', ' ')}"
        if not service.approved(record):
            return "a live run needs a dry run that passed for the current bindings"
    return None


def run_pipeline(pipeline_id: str, context: RunContext) -> PipelineRun:
    """Run the pipeline's workflow over ``context``'s interval and deliver it when it finishes.

    Returns the run, or the existing one when the same event or request already produced it.
    Raises LookupError for an unknown pipeline and RunRefused when it may not run.
    """
    from open_climate_service.exports.service import resolve_named_export
    from open_climate_service.openeo.jobs import triggered_job_id

    record = store.get_record(pipeline_id)
    if record is None:
        raise LookupError(f"Unknown pipeline '{pipeline_id}'")
    spec = record.spec
    mode: Literal["dry_run", "live"] = context.mode or ("live" if spec.delivery.mode == "live" else "dry_run")
    refusal = _refusal(record, context, mode)
    if refusal is not None:
        raise RunRefused(refusal)
    bound = workflow_binding.bind(spec.workflow)
    digest = service.bindings_digest(spec, bound)
    assert record.validation is not None  # checked by _refusal
    if record.validation.bindings != digest:
        raise RunRefused("the workflow or bindings changed since the pipeline was validated; re-check it")
    plugin = service.destination_plugin(spec)
    # The export must resolve by name, because the job saves and the delivery sends through it.
    resolve_named_export(plugin.format, {"export": spec.id})

    run_id = uuid4().hex
    event = context.event_id or f"pipeline-run:{run_id}"
    process, resolved = workflow_binding.build_process(
        spec.workflow, bound, start=context.start, end=context.end, save_format=plugin.format, export_id=spec.id
    )
    run = PipelineRun(
        run_id=run_id,
        trigger=context.trigger,
        event_id=context.event_id,
        retry_of=context.retry_of,
        revision=record.revision,
        bindings=digest,
        workflow_id=spec.workflow.id,
        arguments=resolved,
        process=process,
        source={"datasets": list(bound.dataset_ids), **context.source},
        job_id=triggered_job_id(event, trigger_id(spec.id)),
        mode=mode,
        start=context.start,
        end=context.end,
        submitted_at=utc_now().isoformat(),
        idempotency_key=f"pipeline:{spec.id}:{event}:{mode}",
        submission="pending",
    )

    def _record_intent(current: PipelineRecord) -> PipelineRun:
        # One run per dataset update, whatever the mode: a replayed event after a switch to
        # live must not deliver the earlier dry-run result live.
        existing = next(
            (
                item
                for item in current.runs
                if item.idempotency_key == run.idempotency_key
                or (run.event_id is not None and item.event_id == run.event_id)
            ),
            None,
        )
        if existing is not None:
            return existing
        if run.event_id is not None and _decided(current, run.event_id, context.event_time):
            raise RunRefused(f"update {run.event_id} was already decided for this pipeline")
        if current.revision != record.revision:
            raise RunRefused("the pipeline changed while the run was being prepared; run it again")
        current.runs.insert(0, run)
        if run.event_id is not None and context.event_time is not None:
            # Decided in the same write as the run it starts.
            current.events.handled[run.event_id] = context.event_time
        _prune(current)
        return run

    _, stored = store.update_record(pipeline_id, _record_intent)
    if stored is not run:
        return stored
    return _submit(pipeline_id, run.run_id)


def _submit(pipeline_id: str, run_id: str) -> PipelineRun:
    """Create and start a pending run's deterministic job, then record the outcome. Safe to repeat.

    A transient failure leaves the run pending for `recover_runs`, until the attempts run out;
    a permanent one fails it. Once the job holds the graph, the run no longer keeps a copy.
    """
    from open_climate_service.openeo.jobs import get_openeo_job_service
    from open_climate_service.openeo.schemas import OpenEOJobCreate

    record = store.get_record(pipeline_id)
    if record is None:
        raise LookupError(f"Unknown pipeline '{pipeline_id}'")
    run = _find(record, run_id)
    if run.submission != "pending":
        return run
    error: BaseException | None = None
    try:
        if run.process is None:
            raise ValueError("the run has no process graph to submit")
        job_service = get_openeo_job_service()
        job, _ = job_service.create_triggered_job(
            OpenEOJobCreate(
                process=run.process,
                title=f"{pipeline_id}: {run.start} to {run.end} ({run.mode.replace('_', ' ')})",
                description=(
                    f"Pipeline '{pipeline_id}' run {run.run_id} ({run.trigger}) of workflow '{run.workflow_id}', "
                    f"{run.start} to {run.end}, delivered as {run.mode.replace('_', ' ')}"
                ),
            ),
            source_event_id=_source_event(run),
            trigger_id=trigger_id(pipeline_id),
            max_attempts=MAX_ATTEMPTS,
        )
        job_service.start_triggered_job(job.id)
    except Exception as exc:
        error = exc
        logger.exception("Pipeline '%s' could not submit run %s", pipeline_id, run.run_id)

    def _outcome(current: PipelineRecord) -> PipelineRun:
        stored = _find(current, run.run_id)
        if stored.submission != "pending":
            return stored
        stored.submission_attempts += 1
        if error is None:
            stored.submission = "submitted"
            stored.error = None
            stored.process = None  # the openEO job record holds it now
        else:
            stored.error = _message(error)
            if _is_permanent(error) or stored.submission_attempts >= MAX_SUBMISSION_ATTEMPTS:
                stored.submission = "failed"
        return stored

    try:
        _, stored = store.update_record(pipeline_id, _outcome)
    except LookupError:
        return run
    return stored


def _settled(run: PipelineRun, superseded: set[str]) -> bool:
    """Whether a run may leave the history: confirmed delivered, or replaced by a run of its interval.

    Handing a result to the delivery framework is not delivering it: a run is kept until its
    delivery job confirms the destination accepted every value (or validated them, for a dry
    run). A run that is pending, running, owes a delivery, ended without a result, or whose
    delivery failed, conflicted or has no known outcome is kept whatever its age, because its
    recovery or the operator's action needs it.
    """
    return run.delivery_result in ("delivered", "validated", "acknowledged") or run.run_id in superseded


def _prune(record: PipelineRecord) -> None:
    """Keep the newest runs, and every older run that is not settled."""
    if len(record.runs) <= _RUNS_KEPT:
        return
    superseded = {run.retry_of for run in record.runs if run.retry_of is not None}
    kept = record.runs[:_RUNS_KEPT]
    kept.extend(run for run in record.runs[_RUNS_KEPT:] if not _settled(run, superseded))
    record.runs = kept


def _find(record: PipelineRecord, run_id: str) -> PipelineRun:
    run = next((item for item in record.runs if item.run_id == run_id or item.job_id == run_id), None)
    if run is None:
        raise LookupError(f"pipeline '{record.spec.id}' has no run '{run_id}'")
    return run


# --- delivery --------------------------------------------------------------------------------


def _delivery_key(run: PipelineRun) -> str:
    """The idempotency key of the run's current delivery; a redelivery gets a new one."""
    return run.idempotency_key if run.delivery_round == 0 else f"{run.idempotency_key}:round-{run.delivery_round}"


def _record_job_outcome(pipeline_id: str, run: PipelineRun, job: Any) -> str | None:
    """Record a job that ended without a result; return its outcome, or None while it may finish."""
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    outcome: Literal["error", "canceled", "missing"] | None
    if job is None:
        outcome = "missing"
    elif job.status == OpenEOJobStatus.ERROR:
        outcome = "error"
    elif job.status == OpenEOJobStatus.CANCELED:
        outcome = "canceled"
    else:
        return None
    if run.job_outcome != outcome:
        _update_run(pipeline_id, run.run_id, job_outcome=outcome)
    return outcome


def deliver_run(pipeline_id: str, run_id: str, *, operator: bool, redeliver: bool = False) -> PipelineRun:
    """Submit the delivery a finished run owes, through the export delivery framework.

    ``operator`` is True for the page's actions and False for the automatic hand-off. A
    repeat returns the delivery already submitted, under the same idempotency key. With
    ``redeliver``, an operator sends a run again whose delivery job failed at the destination:
    that is a new delivery with a new key, refused while the earlier one has not failed.
    """
    from open_climate_service.exports.delivery import submit_delivery
    from open_climate_service.openeo.jobs import store_get_job
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    record = store.get_record(pipeline_id)
    if record is None:
        raise LookupError(f"Unknown pipeline '{pipeline_id}'")
    run = _find(record, run_id)
    key = _delivery_key(run)
    if redeliver:
        if not operator:
            raise RunRefused("only an operator can deliver a run again")
        run = _check_redelivery(pipeline_id, record, run)
        key = f"{run.idempotency_key}:round-{run.delivery_round + 1}"
    elif run.delivery_job_id is not None:
        return run
    job = store_get_job(run.job_id)
    outcome = _record_job_outcome(pipeline_id, run, job)
    if outcome == "missing":
        raise LookupError(f"job '{run.job_id}' no longer exists; run the interval again")
    if job is None or job.status != OpenEOJobStatus.FINISHED:
        status = job.status.value if job is not None else "missing"
        raise ValueError(f"job '{run.job_id}' is {status}, not finished")
    hold = _hold_reason(record, run, operator=operator)
    if hold is not None:
        if operator:
            raise RunRefused(hold)
        return _update_run(pipeline_id, run.run_id, delivery="held", error=None)
    try:
        delivery_job_id, _ = submit_delivery(pipeline_id, run.job_id, run.mode == "dry_run", key)
    except Exception as exc:
        if redeliver:
            # The earlier round stays as it was. If the framework reserved the new key before
            # failing, sending again reuses it and finds that reservation: still one delivery.
            _update_run(pipeline_id, run.run_id, error=f"sending again failed: {_message(exc)}")
            raise
        attempts = run.delivery_attempts + 1
        give_up = _is_permanent(exc) or attempts >= MAX_DELIVERY_ATTEMPTS
        _update_run(
            pipeline_id,
            run.run_id,
            delivery="failed" if give_up else "awaiting",
            delivery_attempts=attempts,
            error=_message(exc),
        )
        raise
    if redeliver:
        # The new round exists only now, with the delivery that opened it.
        return _update_run(
            pipeline_id,
            run.run_id,
            delivery_round=run.delivery_round + 1,
            delivery="submitted",
            delivery_result=None,
            delivery_attempts=1,
            error=None,
            delivery_job_id=delivery_job_id,
            delivery_status_url=f"/exports/{pipeline_id}/jobs/{delivery_job_id}",
            delivered_at=utc_now().isoformat(),
        )
    return _update_run(
        pipeline_id,
        run.run_id,
        delivery="submitted",
        delivery_attempts=run.delivery_attempts + 1,
        error=None,
        delivery_job_id=delivery_job_id,
        delivery_status_url=f"/exports/{pipeline_id}/jobs/{delivery_job_id}",
        delivered_at=utc_now().isoformat(),
    )


def delivery_result(delivery: Any) -> str | None:
    """The confirmed outcome of a delivery job, or None while it is still running."""
    from open_climate_service.jobs.models import JobStatus

    if delivery.status == JobStatus.FAILED:
        return "failed"
    if delivery.status == JobStatus.CANCELLED:
        return "cancelled"
    if delivery.status != JobStatus.SUCCESSFUL:
        return None
    report = delivery.result if isinstance(delivery.result, dict) else {}
    outcome = report.get("outcome")
    if outcome == "success":
        return "partial" if report.get("conflicts") else "delivered"
    if outcome == "dry_run":
        return "partial" if report.get("conflicts") else "validated"
    return {"partial": "partial", "rejected": "rejected", "cancelled": "cancelled"}.get(str(outcome), "unknown")


def _refresh_delivery_result(pipeline_id: str, run: PipelineRun) -> PipelineRun:
    """Record the delivery job's outcome on the run once the job has ended.

    A delivery job whose record is gone (deleted, or its store lost) has no outcome anyone can
    confirm: the run records ``unknown``, so it stops being polled and waits for an operator.
    Nothing is sent again by itself.
    """
    from open_climate_service.jobs.service import get_job_service

    if run.delivery_job_id is None or run.delivery_result is not None:
        return run
    try:
        delivery = get_job_service().get_job_or_404(run.delivery_job_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return _update_run(
            pipeline_id,
            run.run_id,
            delivery_result="unknown",
            error=f"delivery job '{run.delivery_job_id}' no longer exists; its outcome cannot be confirmed",
        )
    result = delivery_result(delivery)
    return run if result is None else _update_run(pipeline_id, run.run_id, delivery_result=result)


def _check_redelivery(pipeline_id: str, record: PipelineRecord, run: PipelineRun) -> PipelineRun:
    """Refuse sending a run again unless its delivery ended unconfirmed and a repeat is safe.

    A failed, rejected, conflicting or unanswered delivery may have reached the destination in
    part; sending the payload again is safe only where a repeat cannot add anything, which the
    destination plugin declares (`redelivery_safe`). Writes nothing but the delivery job's
    observed outcome: the new round is recorded only with the delivery that opens it, so a
    refusal here, or the job and hold checks after it, leave the earlier round as it was.
    """
    if run.delivery_job_id is None:
        raise RunRefused("the run has no delivery to send again; deliver it")
    run = _refresh_delivery_result(pipeline_id, run)
    if run.delivery_result is None:
        raise RunRefused("its delivery is still running; only an ended, unconfirmed delivery is sent again")
    if run.delivery_result in ("delivered", "validated", "acknowledged"):
        raise RunRefused(f"its delivery is settled ({run.delivery_result}); there is nothing to send again")
    if not service.destination_plugin(record.spec).redelivery_safe:
        raise RunRefused(
            "this destination does not declare that a repeated delivery is harmless; check what arrived there, "
            "then acknowledge the delivery or run the interval again"
        )
    return run


def acknowledge_delivery(pipeline_id: str, run_id: str) -> PipelineRun:
    """Record that an operator checked the destination and settled an unconfirmed delivery.

    The way to resolve a delivery OCS cannot confirm where sending it again is not safe, or not
    wanted. Nothing is sent; the run is settled, so it may leave the history.
    """
    record = store.get_record(pipeline_id)
    if record is None:
        raise LookupError(f"Unknown pipeline '{pipeline_id}'")
    run = _refresh_delivery_result(pipeline_id, _find(record, run_id))
    if run.delivery_job_id is None or run.delivery_result is None:
        raise RunRefused("only a delivery that ended unconfirmed can be acknowledged")
    if run.delivery_result in ("delivered", "validated", "acknowledged"):
        return run
    return _update_run(pipeline_id, run.run_id, delivery_result="acknowledged", error=None)


def _hold_reason(record: PipelineRecord, run: PipelineRun, *, operator: bool) -> str | None:
    """Why a finished run's delivery must wait, or None."""
    mode = record.spec.delivery.mode
    if run.mode == "live":
        if mode != "live":
            return f"the pipeline is {mode.replace('_', ' ')}; make it live to release this live delivery"
        validation = record.validation
        if validation is None or not validation.valid or run.bindings != validation.bindings:
            return "the run was computed for bindings the pipeline no longer validates; run it again"
        if not service.approved(record):
            return "live delivery needs a dry run that passed for the current bindings"
    elif mode == "paused" and not operator:
        return "the pipeline is paused"
    return None


def _update_run(pipeline_id: str, run_id: str, **changes: Any) -> PipelineRun:
    def _apply(current: PipelineRecord) -> PipelineRun:
        stored = _find(current, run_id)
        for key, value in changes.items():
            setattr(stored, key, value)
        return stored

    _, stored = store.update_record(pipeline_id, _apply)
    return stored


def on_job_finished(job: Any) -> None:
    """Hand off the delivery a pipeline run owes for a job that just finished. Never raises."""
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    trigger = getattr(job, "trigger_id", None)
    if getattr(job, "status", None) != OpenEOJobStatus.FINISHED or not isinstance(trigger, str):
        return
    if not trigger.startswith(TRIGGER_PREFIX):
        return
    pipeline_id = trigger[len(TRIGGER_PREFIX) :]
    try:
        record = store.get_record(pipeline_id)
        run = next((item for item in record.runs if item.job_id == job.id), None) if record else None
        if run is not None and run.delivery == "awaiting":
            deliver_run(pipeline_id, run.run_id, operator=False)
    except Exception:
        logger.exception("Pipeline '%s' could not deliver job %s", pipeline_id, getattr(job, "id", "?"))


def release_held(pipeline_id: str) -> list[str]:
    """Deliver every held run whose hold no longer applies; return their run ids."""
    record = store.get_record(pipeline_id)
    if record is None:
        return []
    released: list[str] = []
    for run in record.runs:
        if run.delivery != "held" or _hold_reason(record, run, operator=False) is not None:
            continue
        try:
            _update_run(pipeline_id, run.run_id, delivery="awaiting")
            deliver_run(pipeline_id, run.run_id, operator=False)
            released.append(run.run_id)
        except Exception:
            logger.exception("Pipeline '%s' could not release held run %s", pipeline_id, run.run_id)
    return released


def recover_runs() -> None:
    """Finish what an interruption or a transient failure left open, for every pipeline.

    Idempotent: a deterministic job id makes resubmission create nothing new, and a run's
    idempotency key makes a repeated hand-off return the delivery already submitted.
    """
    from open_climate_service.openeo.jobs import store_get_job
    from open_climate_service.openeo.schemas import OpenEOJobStatus

    if api_config.is_read_only():
        return
    for record in store.list_records():
        pipeline_id = record.spec.id
        for run in list(record.runs):
            try:
                if run.submission == "pending":
                    _submit(pipeline_id, run.run_id)
                elif run.submission == "submitted" and run.job_outcome is None and run.delivery == "awaiting":
                    job = store_get_job(run.job_id)
                    if _record_job_outcome(pipeline_id, run, job) is None and job is not None:
                        if job.status == OpenEOJobStatus.FINISHED:
                            deliver_run(pipeline_id, run.run_id, operator=False)
                elif run.delivery == "submitted" and run.delivery_result is None:
                    _refresh_delivery_result(pipeline_id, run)
            except Exception:
                logger.exception("Pipeline '%s' could not recover run %s", pipeline_id, run.run_id)
        if any(run.delivery == "held" for run in record.runs):
            release_held(pipeline_id)


# --- dataset updates -------------------------------------------------------------------------


def _parse(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _decided(record: PipelineRecord, event_id: str, event_time: str | None) -> bool:
    """Whether this pipeline already decided on an update: run it, or deliberately not."""
    if event_id in record.events.handled:
        return True
    through, when = _parse(record.events.through), _parse(event_time)
    return through is not None and when is not None and when < through


def update_interval(data: dict[str, Any]) -> tuple[str, str, Literal["appended", "rewritten"]] | None:
    """The interval a dataset update affects, from the update's own description of itself.

    An append (``previous_end`` set) added the periods after ``previous_end`` and left the
    earlier ones as they were: the run covers ``previous_end`` to ``current_end``, re-sending
    the last period already stored, so an overwrite at the destination completes it. A
    rewrite (``previous_end`` null: a new store or a rematerialization) may have changed any
    period, so the run covers the whole coverage after it. Updates do not describe a
    correction of individual stored periods; there is none for a pipeline to follow.
    """
    end = data.get("current_end")
    previous = data.get("previous_end")
    if not isinstance(end, str):
        return None
    if isinstance(previous, str):
        return previous, end, "appended"
    start = data.get("current_start")
    return (start, end, "rewritten") if isinstance(start, str) else None


def _event_context(event: Any) -> RunContext | None:
    interval = update_interval(event.data)
    if interval is None:
        return None
    start, end, scope = interval
    data = event.data
    source = {key: data[key] for key in ("dataset_id", "artifact_id", "action") if data.get(key) is not None}
    return RunContext(
        start=start,
        end=end,
        trigger="dataset_update",
        event_id=event.event_id,
        event_time=_utc(event.time).isoformat(),
        source={**source, "update": scope},
    )


def _skip(pipeline_id: str, event: Any, reason: str, *, decide: bool = True) -> None:
    """Record that an update is not run, and why, for an operator to run its interval by hand.

    With ``decide`` it is also marked decided, so it is not reconsidered. An update seen while
    the pipeline is paused is only listed: the checkpoint starts afresh when it resumes.
    """
    from open_climate_service.pipelines.schemas import SkippedUpdate

    interval = update_interval(event.data)
    entry = SkippedUpdate(
        event_id=event.event_id,
        time=_utc(event.time).isoformat(),
        dataset_id=event.data.get("dataset_id"),
        start=interval[0] if interval else None,
        end=interval[1] if interval else None,
        reason=reason,
    )

    def _apply(current: PipelineRecord) -> None:
        if decide:
            current.events.handled.setdefault(event.event_id, entry.time)
            current.events.retries.pop(event.event_id, None)
        if all(item.event_id != entry.event_id for item in current.events.skipped):
            current.events.skipped.insert(0, entry)
            discarded = current.events.skipped[_SKIPPED_KEPT:]
            if discarded:
                # Bounded history, not an audit trail: say how much was dropped, and up to when.
                del current.events.skipped[_SKIPPED_KEPT:]
                current.events.skipped_discarded += len(discarded)
                newest = max(item.time for item in discarded)
                if (
                    current.events.skipped_discarded_through is None
                    or newest > current.events.skipped_discarded_through
                ):
                    current.events.skipped_discarded_through = newest
                logger.warning(
                    "Pipeline '%s' discarded %d skipped-update entr%s from its bounded list",
                    pipeline_id,
                    len(discarded),
                    "y" if len(discarded) == 1 else "ies",
                )

    try:
        store.update_record(pipeline_id, _apply)
    except LookupError:
        return
    logger.info("Pipeline '%s' does not run for update %s: %s", pipeline_id, event.event_id, reason)


def _defer(pipeline_id: str, event: Any, exc: BaseException) -> bool:
    """Count a transient failure to start a run for an update; False once the attempts run out."""

    def _apply(current: PipelineRecord) -> int:
        current.events.retries[event.event_id] = current.events.retries.get(event.event_id, 0) + 1
        return current.events.retries[event.event_id]

    try:
        _, attempts = store.update_record(pipeline_id, _apply)
    except LookupError:
        return False
    if attempts >= MAX_EVENT_ATTEMPTS:
        _skip(pipeline_id, event, f"could not be started after {attempts} attempts: {_message(exc)}")
        return False
    return True


def _source_of(record: PipelineRecord, event: Any) -> bool:
    """Whether the update is for a dataset the pipeline's workflow reads."""
    try:
        bound = workflow_binding.bind(record.spec.workflow)
    except ValueError:
        return False
    return event.data.get("dataset_id") in bound.dataset_ids


def _dispatch(record: PipelineRecord, event: Any, *, live: bool = False) -> datetime | None:
    """Decide one update for one pipeline. Returns the update's time when it must be decided again later.

    ``live`` is True when the update was just persisted, False when a reconciliation reads it
    back from the job store. Outcomes:

    - not this pipeline's source, or older than the pipeline's activation: ignored, the same
      way every time;
    - run: decided in the same write as the run;
    - refused (the pipeline is invalid, changed since it was validated, or the update names no
      interval) or a permanent error: decided and listed as skipped, never run later;
    - a transient error: left undecided and retried, up to `MAX_EVENT_ATTEMPTS`, then skipped;
    - while paused, a live update is listed as skipped but not decided; the checkpoint starts
      afresh on resume, so updates from the pause are never run by themselves;
    - a live update older than the checkpoint and not decided arrived too late to be counted:
      it is listed as skipped rather than lost silently.
    """
    spec = record.spec
    if not store.automates(spec):
        if (
            live
            and spec.delivery.policy == "on_update"
            and spec.delivery.mode == "paused"
            and _source_of(record, event)
        ):
            _skip(spec.id, event, "the pipeline was paused", decide=False)
        return None
    since = _parse(record.automation_since)
    event_time = _utc(event.time)
    if since is None or event_time < since:
        return None
    if event.event_id in record.events.handled:
        return None
    if _decided(record, event.event_id, event_time.isoformat()):
        if live and _source_of(record, event):
            _skip(spec.id, event, "it arrived after the checkpoint had passed its time")
        return None
    try:
        bound = workflow_binding.bind(spec.workflow)
    except ValueError as exc:
        _skip(spec.id, event, f"the workflow binding is invalid: {exc}")
        return None
    if event.data.get("dataset_id") not in bound.dataset_ids:
        return None  # not this pipeline's source: decided the same way every time
    context = _event_context(event)
    if context is None:
        _skip(spec.id, event, "the update names no interval")
        return None
    try:
        run_pipeline(spec.id, context)
    except RunRefused as exc:
        _skip(spec.id, event, str(exc))
    except Exception as exc:
        if _is_permanent(exc):
            _skip(spec.id, event, _message(exc))
            return None
        logger.exception("Pipeline '%s' failed to run for update %s", spec.id, event.event_id)
        # Left undecided, bounded: the next reconciliation tries again, and the checkpoint waits.
        return event_time if _defer(spec.id, event, exc) else None
    return None


def consume(events: list[Any]) -> None:
    """Decide the dataset updates just persisted, for every pipeline."""
    from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE

    if api_config.is_read_only():
        return
    updates = [event for event in events if event.type == DATASET_UPDATED_EVENT_TYPE]
    if not updates:
        return
    for record in store.list_records():
        for event in updates:
            _dispatch(record, event, live=True)


def reconcile_events() -> None:
    """Decide every persisted update a pipeline missed, then advance its checkpoint safely.

    Updates are read from the shared event log (`jobs.events`): each update is stored with the
    job that made it, a native ingestion or sync, or an openEO job that published a managed
    dataset. There is no cursor over it, so this reads it once (as workflow automation does at
    startup) and skips updates before each pipeline's checkpoint without looking further.
    The checkpoint then moves to the earliest of: the start of this pass, the creation of
    any native job still unfinished (its update, if any, will be later; an openEO job stamps
    its update inside the write that persists it, so it needs no such bound), and any update
    that failed to be decided; less a margin. Updates past the checkpoint stay listed as
    handled until the checkpoint passes them.
    """
    from open_climate_service.jobs.events import read_event_log
    from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE

    if api_config.is_read_only():
        return
    scan_start = utc_now()
    automating = [record for record in store.list_records() if store.automates(record.spec)]
    if not automating:
        return
    log = read_event_log()
    bound = scan_start if log.open_since is None else min(scan_start, log.open_since)
    updates = [event for event in log.events if event.type == DATASET_UPDATED_EVENT_TYPE]
    for record in automating:
        retry_from: list[datetime] = []
        for event in updates:
            deferred = _dispatch(record, event)
            if deferred is not None:
                retry_from.append(deferred)
        _advance_checkpoint(record, min([bound, *retry_from]) - _EVENT_MARGIN)


def _advance_checkpoint(record: PipelineRecord, safe: datetime) -> None:
    """Move the checkpoint forward to ``safe``, never back, and forget what it now covers."""

    def _apply(current: PipelineRecord) -> None:
        if current.automation_since != record.automation_since:
            return  # automation restarted meanwhile; its checkpoint is new
        through = _parse(current.events.through)
        if through is not None and safe <= through:
            return
        current.events.through = safe.isoformat()
        current.events.handled = {
            event_id: when
            for event_id, when in current.events.handled.items()
            if (parsed := _parse(when)) is None or parsed >= safe
        }

    try:
        store.update_record(record.spec.id, _apply)
    except LookupError:
        return


def reconcile() -> None:
    """Startup recovery: open runs first, then missed dataset updates."""
    recover_runs()
    reconcile_events()


def run_views(record: PipelineRecord) -> list[dict[str, Any]]:
    """Runs with their job and delivery states, for the page."""
    from open_climate_service.jobs.service import get_job_service
    from open_climate_service.openeo.jobs import store_get_job

    views: list[dict[str, Any]] = []
    for run in record.runs:
        job = store_get_job(run.job_id)
        view: dict[str, Any] = {
            **run.model_dump(mode="json", exclude={"process"}),
            "job_status": job.status.value if job else ("not created" if run.submission != "submitted" else "gone"),
            "job_error": job.error_message if job else None,
        }
        if run.delivery_job_id:
            try:
                delivery = get_job_service().get_job_or_404(run.delivery_job_id)
                result = delivery.result if isinstance(delivery.result, dict) else {}
                view["delivery_status"] = str(getattr(delivery.status, "value", delivery.status))
                view["delivery_outcome"] = result.get("outcome")
                view["delivery_submitted"] = result.get("submitted")
                view["delivery_conflicts"] = len(result.get("conflicts") or [])
            except Exception:
                view["delivery_status"] = "unknown"
        views.append(view)
    return views
