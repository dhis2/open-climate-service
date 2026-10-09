"""Consume durable dataset updates and submit configured openEO workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from open_climate_service import config as api_config
from open_climate_service.automation.config import (
    AutomationConfig,
    TriggerDelivery,
    WorkflowTrigger,
    get_automation_config,
)
from open_climate_service.jobs import store as job_store
from open_climate_service.jobs.models import TASK_EVENT_TYPES, JobEvent
from open_climate_service.openeo import workflows
from open_climate_service.openeo.jobs import (
    OpenEOJobService,
    get_openeo_job_service,
    store_list_jobs,
    store_update_job,
)
from open_climate_service.openeo.schemas import OpenEOJobCreate, OpenEOJobRecord, OpenEOJobStatus
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

_EVENT_VALUES = {
    "$event.dataset_id": "dataset_id",
    "$event.artifact_id": "artifact_id",
    "$event.action": "action",
    "$event.previous_end": "previous_end",
    "$event.current_start": "current_start",
    "$event.current_end": "current_end",
    "$event.collection_id": "collection_id",
}

MAX_CHAIN_DEPTH = 5
"""How many workflow runs one change may set off in a row (CLIM-1243).

A sync starts at depth 0; each workflow that publishes a dataset passes its depth on, one higher,
in the event it emits. A chain longer than this is a cycle or a mistake, so it stops there.
"""


def _resolve_event_values(value: Any, event: JobEvent) -> Any:
    """Replace exact event references recursively while preserving literal values."""
    if isinstance(value, str) and value in _EVENT_VALUES:
        return event.data.get(_EVENT_VALUES[value])
    if isinstance(value, list):
        return [_resolve_event_values(item, event) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_event_values(item, event) for key, item in value.items()}
    return value


def _delivery_mode(dry_run: bool) -> str:
    return "dry-run" if dry_run else "live"


def _load_delivery_activations() -> dict[str, dict[str, str]]:
    """Return ``{trigger_id: {"export", "mode", "activated_at"}}`` from the operational database.

    An incomplete entry is dropped and re-stamped at the next start, which delivers nothing
    finished before it: the safe reading, since pushing history is the failure this boundary
    prevents.
    """
    from open_climate_service.state import db

    fields = ("export", "mode", "activated_at")
    return {
        key: {field: value[field] for field in fields}
        for key, value in db.load_activations("delivery").items()
        if isinstance(value, dict)
        and all(isinstance(value.get(field), str) for field in fields)
        and _parse_time(value["activated_at"]) is not None
    }


def _parse_time(value: str) -> datetime | None:
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _save_delivery_activations(activations: dict[str, dict[str, str]]) -> None:
    from open_climate_service.state import db

    db.save_activations("delivery", activations)


def _sync_delivery_activations(config: AutomationConfig) -> dict[str, dict[str, str]]:
    """Stamp new delivery steps, forget removed ones, and return the boundaries now in force.

    A trigger gains a boundary when it first delivers, or changes its export or its mode.
    The mode matters as much as the export: a job that finished under ``dry_run: true`` and
    was never delivered must not be imported live after switching to ``dry_run: false``.
    Removing ``deliver`` drops the boundary, so adding it back later does not deliver the
    jobs that finished in between.
    """
    current = _load_delivery_activations()
    now = utc_now().isoformat()
    wanted: dict[str, dict[str, str]] = {}
    for trigger in config.workflow_triggers:
        if trigger.deliver is None:
            continue
        binding = {"export": trigger.deliver.export, "mode": _delivery_mode(trigger.deliver.dry_run)}
        existing = current.get(trigger.id)
        if existing is not None and all(existing[key] == value for key, value in binding.items()):
            wanted[trigger.id] = existing
        else:
            wanted[trigger.id] = {**binding, "activated_at": now}
    if wanted != current:
        _save_delivery_activations(wanted)
    return wanted


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


@dataclass(frozen=True)
class _DeliveryStep:
    """A trigger's delivery settings and the boundary before which nothing is delivered."""

    delivery: TriggerDelivery
    activated_at: datetime


def _delivery_steps(config: AutomationConfig, activations: dict[str, dict[str, str]]) -> dict[str, _DeliveryStep]:
    """Index each delivering trigger by ID, parsing its activation boundary once.

    A trigger whose boundary is missing or unreadable is left out, so it delivers nothing:
    pushing history is the failure the boundary exists to prevent.
    """
    steps: dict[str, _DeliveryStep] = {}
    for trigger in config.workflow_triggers:
        activation = activations.get(trigger.id)
        if (
            trigger.deliver is None
            or activation is None
            or activation["export"] != trigger.deliver.export
            or activation["mode"] != _delivery_mode(trigger.deliver.dry_run)
        ):
            continue
        activated_at = _parse_time(activation["activated_at"])
        if activated_at is None:
            continue
        steps[trigger.id] = _DeliveryStep(trigger.deliver, activated_at)
    return steps


def _finished_after(record: OpenEOJobRecord, activated_at: datetime) -> bool:
    """True when a job finished at or after a delivery activation boundary.

    Jobs without a recorded finish time finished before this boundary existed.
    """
    return record.finished_at is not None and _as_utc(record.finished_at) >= activated_at


def delivery_idempotency_key(job_id: str, export_id: str, dry_run: bool) -> str:
    """Deterministic key: one delivery per triggered job, export, and mode.

    The mode is part of the key because a key reused with a different ``dry_run`` is a
    conflict, and switching a trigger from dry run to live must not collide with earlier
    dry-run deliveries.
    """
    return f"auto:{job_id}:{export_id}:{_delivery_mode(dry_run)}"


def _set_delivery_error(job_id: str, export_id: str, message: str | None) -> None:
    """Expose the latest automatic-delivery submission error on the source job."""
    now = utc_now()

    def update(record: OpenEOJobRecord) -> OpenEOJobRecord:
        usage = dict(record.usage or {})
        if message is None:
            usage.pop("delivery_error", None)
        else:
            usage["delivery_error"] = {
                "export_id": export_id,
                "message": message,
                "failed_at": now.isoformat(),
            }
        return record.model_copy(update={"usage": usage, "updated": now})

    try:
        store_update_job(job_id, update)
    except Exception:
        # Error visibility must not turn a completed workflow into a failed one.
        logger.exception("Could not update delivery error state for openEO job %s", job_id)


def _validate_deliveries(config: AutomationConfig, exports: dict[str, dict[str, Any]] | None = None) -> None:
    """Refuse a delivery that could only fail per job, naming its task.

    The export is resolved in its own plugin's format and the plugin decides what its deliveries
    need (CLIM-1289): no plugin id or format is named here.
    """
    from open_climate_service.exports.service import resolve_definition, resolve_export_by_id

    deliveries = [(trigger, trigger.deliver) for trigger in config.workflow_triggers if trigger.deliver is not None]
    for trigger, delivery in deliveries:
        export_id = delivery.export
        prefix = f"Workflow task {trigger.id!r} delivers export {export_id!r}"
        workflow_export = trigger.arguments.get("export")
        if isinstance(workflow_export, str) and workflow_export != export_id:
            raise ValueError(
                f"{prefix}, but arguments.export is {workflow_export!r}; "
                "the workflow and delivery must use the same named export"
            )
        try:
            if exports is None:
                resolved = resolve_export_by_id(export_id)
            elif export_id in exports:
                resolved = resolve_definition(exports[export_id])
            else:
                raise ValueError(f"Unknown export '{export_id}'")
        except ValueError as exc:
            raise ValueError(f"{prefix}, which is invalid: {exc}") from None
        try:
            resolved.plugin.check_delivery_target(resolved.references)
        except ValueError as exc:
            raise ValueError(f"{prefix}, {exc}") from None


def _load_activations() -> dict[str, str]:
    """Return each trigger's activation time from the operational database.

    One transaction writes them, so a crash leaves the previous set or the new one, never a
    truncated file read back as empty (which replayed every historical event, CLIM-927).
    """
    from open_climate_service.state import db

    return {key: value for key, value in db.load_activations("trigger").items() if isinstance(value, str)}


def _save_activations(activations: dict[str, str]) -> None:
    from open_climate_service.state import db

    db.save_activations("trigger", activations)


def _is_before_activation(event: JobEvent, activation_iso: str | None) -> bool:
    """True when an event predates a trigger's activation boundary.

    A missing or unreadable activation counts as *before* activation: ``start()`` always stamps
    one, so its absence is an anomaly, and skipping is the safe reading of the
    ``replay_existing: false`` guarantee — replaying all history is the failure mode this guard
    exists to prevent.
    """
    if not isinstance(activation_iso, str) or not activation_iso:
        return True
    try:
        activation = datetime.fromisoformat(activation_iso)
    except ValueError:
        return True
    event_time = event.time
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    if activation.tzinfo is None:
        activation = activation.replace(tzinfo=timezone.utc)
    return event_time < activation


_MANAGED_OUTPUT_PARAMETER = "output_dataset_id"


def _resolved_output_dataset(trigger: WorkflowTrigger, workflow: Any) -> str | None:
    """Return a trigger's statically resolvable managed output, or None.

    Only workflows that declare an ``output_dataset_id`` parameter produce a
    managed dataset. File-producing workflows and ``$event`` references cannot
    be resolved statically and are left to the per-store lock at run time.
    """
    parameters = getattr(workflow, "parameters", None)
    if not isinstance(parameters, list):
        return None
    names = {param.get("name") for param in parameters if isinstance(param, dict)}
    if _MANAGED_OUTPUT_PARAMETER not in names:
        return None
    value = trigger.arguments.get(_MANAGED_OUTPUT_PARAMETER)
    if isinstance(value, str) and value and not value.startswith("$event"):
        return value
    return None


def _validate_output_ownership(config: AutomationConfig) -> None:
    """Refuse triggers on one source that would concurrently write one output."""
    bindings: dict[tuple[str, str], list[str]] = {}
    for trigger in config.workflow_triggers:
        workflow = workflows.get_workflow(trigger.workflow_id)
        output = _resolved_output_dataset(trigger, workflow) if workflow is not None else None
        if output is None:
            continue
        bindings.setdefault((trigger.on_update_of or "", output), []).append(trigger.id)
    duplicates = [(key, ids) for key, ids in bindings.items() if len(ids) > 1]
    if duplicates:
        details = "; ".join(
            f"source {source!r}, output {output!r}: {', '.join(ids)}" for (source, output), ids in duplicates
        )
        raise ValueError(
            f"workflow triggers would concurrently write the same managed output: {details}. "
            "Give each trigger a distinct output_dataset_id."
        )


def _iter_strings(value: Any) -> Any:
    """Yield every string in a nested mapping/list, for event-reference validation."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)


def _validate_event_references(config: AutomationConfig) -> None:
    """Refuse arguments that look like event references but are not the known tokens.

    ``_resolve_event_values`` substitutes only exact whole-string matches, so a typo such as
    ``$event.datasetid`` would otherwise be submitted as a literal and fail only at run time.
    """
    for trigger in config.workflow_triggers:
        for value in _iter_strings(trigger.arguments):
            if "$event" in value and value not in _EVENT_VALUES:
                raise ValueError(
                    f"Workflow trigger {trigger.id!r} argument {value!r} looks like an event reference "
                    f"but is not one of {sorted(_EVENT_VALUES)}"
                )


FROM_FEATURES = "from_features"
"""Trigger-argument key naming a declared feature collection instead of carrying its geometry."""


def _feature_node_name(feature_id: str) -> str:
    """The graph node that resolves one feature collection. Stable, so two references share one node."""
    return f"features_{feature_id}"


def _is_inline_geojson(value: Any) -> bool:
    """Return whether a mapping is an inline GeoJSON feature or collection."""
    return isinstance(value, dict) and value.get("type") in {"Feature", "FeatureCollection"}


def _iter_feature_references(value: Any) -> Any:
    """Yield valid feature ids and reject malformed `from_features` marker objects."""
    if _is_inline_geojson(value):
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_feature_references(item)
    elif isinstance(value, dict):
        if FROM_FEATURES in value:
            feature_id = value.get(FROM_FEATURES)
            if set(value) != {FROM_FEATURES} or not isinstance(feature_id, str) or not feature_id.strip():
                raise ValueError(
                    f"{FROM_FEATURES!r} must be the only key in its object and name a non-empty feature id"
                )
            if feature_id != feature_id.strip():
                raise ValueError(f"{FROM_FEATURES!r} feature id must not have leading or trailing whitespace")
            yield feature_id
            return
        for item in value.values():
            yield from _iter_feature_references(item)


def _resolve_feature_references(
    value: Any,
    nodes: dict[str, Any] | None = None,
    versions: dict[str, str] | None = None,
) -> Any:
    """Rewrite a feature marker into a pinned reference to a `load_features` node.

    Geometry is resolved during execution and never copied into the persisted graph. The
    submission-time record timestamp is passed as `version`, so a queued job either reads the
    exact collection it declares or fails if a refresh has replaced that version.
    """
    if _is_inline_geojson(value):
        return value
    if isinstance(value, list):
        return [_resolve_feature_references(item, nodes, versions) for item in value]
    if not isinstance(value, dict):
        return value
    if FROM_FEATURES in value:
        references = list(_iter_feature_references(value))
        feature_id = references[0]
        name = _feature_node_name(feature_id)
        if nodes is not None:
            arguments = {"id": feature_id}
            if versions is not None:
                arguments["version"] = versions[feature_id]
            nodes[name] = {"process_id": "load_features", "arguments": arguments}
        return {"from_node": name}
    return {key: _resolve_feature_references(item, nodes, versions) for key, item in value.items()}


def _feature_versions(arguments: Any) -> dict[str, str]:
    """Resolve and pin every referenced collection to its current registered record."""
    from open_climate_service.features import services as feature_services

    feature_ids = sorted(set(_iter_feature_references(arguments)))
    if not feature_ids:
        return {}
    records = feature_services.registered_collections()
    versions: dict[str, str] = {}
    for feature_id in feature_ids:
        record = records.get(feature_id)
        if record is None:
            raise ValueError(
                f"Feature collection {feature_id!r} is declared but unregistered; refresh it before submitting "
                "a workflow that references it"
            )
        versions[feature_id] = record.created_at.isoformat()
    return versions


def _feature_provenance(versions: dict[str, str]) -> str:
    """Describe the pinned feature collection versions carried by a submitted process graph."""
    parts = [f"{feature_id}@{version}" for feature_id, version in sorted(versions.items())]
    return f" against features {', '.join(parts)}" if parts else ""


def _validate_feature_references(config: AutomationConfig) -> None:
    """Refuse malformed references and ids that no feature template declares at startup."""
    from open_climate_service.features.templates import list_feature_templates

    references_by_trigger: list[tuple[WorkflowTrigger, list[str]]] = []
    for trigger in config.workflow_triggers:
        try:
            references = list(_iter_feature_references(trigger.arguments))
        except ValueError as exc:
            raise ValueError(f"Workflow trigger {trigger.id!r} has an invalid feature reference: {exc}") from exc
        if references:
            references_by_trigger.append((trigger, references))
    if not references_by_trigger:
        return

    declared = {str(template["id"]) for template in list_feature_templates()}
    for trigger, references in references_by_trigger:
        for feature_id in references:
            if feature_id not in declared:
                available = ", ".join(sorted(declared)) or "none"
                raise ValueError(
                    f"Workflow trigger {trigger.id!r} has an invalid feature reference: references feature "
                    f"{feature_id!r}, which does not name a declared feature template. Declared: {available}"
                )


class WorkflowAutomationService:
    """Dispatch configured workflows once for each matching durable event."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], AutomationConfig] = get_automation_config,
        openeo_service: OpenEOJobService | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._openeo_service = openeo_service
        self._config: AutomationConfig | None = None
        # Built by start(), the only writer of delivery boundaries in this process, so the
        # finish hook and reconciliation never re-read the activation file per job.
        self._delivery_steps: dict[str, _DeliveryStep] = {}

    def start(self) -> None:
        """Load configuration, validate triggers, and record activation boundaries.

        Validation runs even on a read-only instance so a configuration error surfaces at startup
        rather than only once the instance is made writable.
        """
        self._delivery_steps = {}
        self._config = self._config_loader()
        if not self._config.workflow_triggers:
            if not api_config.is_read_only():
                _sync_delivery_activations(self._config)
            return
        validate_automation(self._config)
        if api_config.is_read_only():
            return
        self._delivery_steps = _delivery_steps(self._config, _sync_delivery_activations(self._config))
        activations = _load_activations()
        now = utc_now().isoformat()
        changed = False
        for trigger in self._config.workflow_triggers:
            if trigger.id not in activations:
                activations[trigger.id] = now
                changed = True
        if changed:
            _save_activations(activations)

    def reload(self) -> None:
        """Apply a change to the tasks store without a restart (CLIM-1378).

        Re-reads and re-validates the triggers, stamps an activation boundary for each new one,
        and re-derives the delivery steps. A task that cannot be loaded leaves the previous
        triggers in force: the store's write path validates first, so this only happens when the
        file was edited by hand.
        """
        previous = (self._config, self._delivery_steps)
        try:
            self.start()
        except Exception:
            self._config, self._delivery_steps = previous
            logger.exception("Workflow tasks could not be reloaded; keeping the previous ones")

    def run_now(self, trigger_id: str, cause: str) -> str:
        """Run one workflow task now, on its cron or by hand, and return the openEO job id.

        ``cause`` names the run (a cron fire time, or a unique id for a run by hand); it makes
        the job id deterministic, so a cron fire that is repeated after a restart creates no
        second job. ``$event.*`` references resolve to None: no event started this run.
        """
        config = self._config or self._config_loader()
        trigger = next((item for item in config.workflow_triggers if item.id == trigger_id), None)
        if trigger is None:
            raise ValueError(f"No enabled workflow task {trigger_id!r}")
        if api_config.is_read_only():
            raise ValueError("This instance is read-only; workflow tasks do not run")
        event = JobEvent(
            event_id=f"{cause}:{trigger_id}",
            type="task.started",
            source=f"/tasks/{trigger_id}",
            time=utc_now(),
            data={},
        )
        service = self._openeo_service or get_openeo_job_service()
        return self._submit(trigger, event, service)[0]

    def replay(self) -> None:
        """Consume persisted events, honouring each trigger's activation boundary."""
        config = self._config or self._config_loader()
        if api_config.is_read_only() or not config.workflow_triggers:
            return
        service = self._openeo_service or get_openeo_job_service()
        activations = _load_activations()
        for record in job_store.list_job_records():
            for event in record.events:
                if event.type not in TASK_EVENT_TYPES:
                    continue
                for trigger in config.workflow_triggers:
                    if not trigger.matches(event.type, event.data):
                        continue
                    if not trigger.replay_existing and _is_before_activation(event, activations.get(trigger.id)):
                        continue
                    self._submit_safely(trigger, event, service)

    def consume(self, events: list[JobEvent]) -> None:
        """Submit workflows for newly persisted successful-job events."""
        config = self._config or self._config_loader()
        if api_config.is_read_only() or not config.workflow_triggers:
            return
        service = self._openeo_service or get_openeo_job_service()
        for event in events:
            if event.type not in TASK_EVENT_TYPES:
                continue
            for trigger in config.workflow_triggers:
                if not trigger.matches(event.type, event.data):
                    continue
                self._submit_safely(trigger, event, service)

    def delivery_due_for(self, record: OpenEOJobRecord) -> dict[str, str] | None:
        """Return the delivery a job finishing now owes, to be stored with its FINISHED state.

        Called while the job is being marked finished, so the obligation is recorded in the
        same write. A job finishing while its trigger has no active delivery step (``deliver``
        absent, an older boundary, or a read-only instance) records nothing and is never
        delivered later, even if the same step is configured again.
        """
        step = self._active_step(record)
        if step is None:
            return None
        return {"export": step.delivery.export, "mode": _delivery_mode(step.delivery.dry_run)}

    def on_job_finished(self, record: OpenEOJobRecord) -> None:
        """Submit the delivery for a triggered job that just finished, when its trigger asks."""
        step = self._due_step(record)
        if step is not None:
            self._deliver_safely(str(record.trigger_id), step.delivery, record)

    def reconcile_deliveries(self) -> None:
        """Submit deliveries missed while the process was down.

        Covers a process killed after a job finished and before its delivery was submitted.
        A job whose source record already lists a delivery for the export is skipped, so
        earlier dry-run deliveries are not repeated live after ``dry_run`` is switched off.
        The deterministic idempotency key makes a double submission harmless.
        """
        if not self._delivery_steps or api_config.is_read_only():
            return
        for record in store_list_jobs():
            step = self._due_step(record)
            if step is not None:
                self._deliver_safely(str(record.trigger_id), step.delivery, record)

    def _active_step(self, record: OpenEOJobRecord) -> _DeliveryStep | None:
        """Return the step configured for a finished triggered job's trigger, or None."""
        # Steps exist only after start() validated and stamped them on a writable instance.
        if record.status != OpenEOJobStatus.FINISHED or record.trigger_id is None or api_config.is_read_only():
            return None
        step = self._delivery_steps.get(record.trigger_id)
        if step is None or not _finished_after(record, step.activated_at):
            return None
        return step

    def _due_step(self, record: OpenEOJobRecord) -> _DeliveryStep | None:
        """Return the step a finished job still owes, or None.

        The job must have recorded this exact export and mode when it finished, and must not
        already list a delivery for the export. The second check covers a re-run: its delivery
        links survive, so a job delivered as a dry run is not imported live after the switch.
        """
        step = self._active_step(record)
        if step is None:
            return None
        expected = {"export": step.delivery.export, "mode": _delivery_mode(step.delivery.dry_run)}
        if record.delivery_due != expected or _lists_delivery(record, step.delivery.export):
            return None
        return step

    def _deliver_safely(self, trigger_id: str, delivery: TriggerDelivery, record: OpenEOJobRecord) -> None:
        """Submit one delivery, logging rather than raising so siblings still run."""
        from fastapi import HTTPException

        from open_climate_service.exports.delivery import submit_delivery

        export_id = delivery.export
        dry_run = delivery.dry_run
        try:
            delivery_job_id, reused = submit_delivery(
                export_id,
                record.id,
                dry_run,
                delivery_idempotency_key(record.id, export_id, dry_run),
            )
        except HTTPException as exc:
            _set_delivery_error(record.id, export_id, str(exc.detail))
            _record_delivery_refusal(trigger_id, delivery, record.id, str(exc.detail))
            # Verification refusals (changed export, missing manifest, re-run source) are
            # operator-actionable configuration states, not crashes.
            logger.warning(
                "Workflow trigger %s could not deliver job %s to export %s: %s",
                trigger_id,
                record.id,
                export_id,
                exc.detail,
            )
            return
        except Exception as exc:
            _set_delivery_error(record.id, export_id, f"{type(exc).__name__}: {exc}")
            _record_delivery_refusal(trigger_id, delivery, record.id, f"{type(exc).__name__}: {exc}")
            logger.exception(
                "Workflow trigger %s failed to deliver job %s to export %s", trigger_id, record.id, export_id
            )
            return
        _set_delivery_error(record.id, export_id, None)
        if not reused:
            from open_climate_service.runs.service import record_run, run_for_job

            parent = run_for_job(record.id)
            record_run(
                task_id=delivery.task_id or f"{trigger_id}:deliver",
                kind="deliver",
                cause="event",
                cause_ref=record.id,
                parent_run_id=parent.id if parent else None,
                outcome="submitted",
                message=f"{'Dry-run' if dry_run else 'Live'} delivery to {export_id} submitted",
                job_kind="native",
                job_id=delivery_job_id,
            )
            logger.info(
                "Submitted %s delivery %s of job %s to export %s",
                "dry-run" if dry_run else "live",
                delivery_job_id,
                record.id,
                export_id,
            )

    def _submit_safely(self, trigger: WorkflowTrigger, event: JobEvent, service: OpenEOJobService) -> None:
        """Submit one trigger/event pair without letting a failure skip its siblings."""
        from open_climate_service.runs.service import record_run, run_for_job

        parent = run_for_job(event.data.get("producing_job_id") or event.event_id.rsplit(":", 1)[0])
        try:
            job_id, created = self._submit(trigger, event, service)
        except Exception as exc:
            record_run(
                task_id=trigger.id,
                kind="workflow",
                cause="event",
                cause_ref=event.event_id,
                parent_run_id=parent.id if parent else None,
                outcome="error",
                message=f"{type(exc).__name__}: {exc}",
            )
            logger.exception(
                "Workflow trigger %s failed for event %s (dataset %s)",
                trigger.id,
                event.event_id,
                event.data.get("dataset_id"),
            )
            return
        if created:
            record_run(
                task_id=trigger.id,
                kind="workflow",
                cause="event",
                cause_ref=event.event_id,
                parent_run_id=parent.id if parent else None,
                outcome="submitted",
                message=f"Workflow {trigger.workflow_id} submitted",
                job_kind="openeo",
                job_id=job_id,
            )

    def _submit(self, trigger: WorkflowTrigger, event: JobEvent, service: OpenEOJobService) -> tuple[str, bool]:
        """Create and start the deterministic job for one trigger/event pair.

        Returns its id and whether this call created it: a replayed event finds the job it
        already made, and is not a new run.
        """
        depth = int(event.data.get("chain_depth") or 0)
        if depth >= MAX_CHAIN_DEPTH:
            raise ValueError(
                f"Event {event.event_id} is {depth} workflow runs from the change that started it; "
                f"chains stop at {MAX_CHAIN_DEPTH} (a cycle between tasks?)"
            )
        dataset_id = event.data.get("dataset_id") or event.data.get("collection_id")
        resolved = _resolve_event_values(trigger.arguments, event)
        versions = _feature_versions(resolved)
        provenance = _feature_provenance(versions)
        feature_nodes: dict[str, Any] = {}
        arguments = _resolve_feature_references(resolved, feature_nodes, versions)
        body = OpenEOJobCreate(
            title=(
                f"{trigger.workflow_id} after {dataset_id} update"
                if dataset_id
                else f"{trigger.workflow_id} ({trigger.id})"
            ),
            description=f"Triggered by {event.event_id} using automation rule {trigger.id}{provenance}",
            process={
                "process_graph": {
                    **feature_nodes,
                    "workflow": {
                        "process_id": trigger.workflow_id,
                        "arguments": arguments,
                        "result": True,
                    },
                }
            },
        )
        job, created = service.create_triggered_job(
            body,
            source_event_id=event.event_id,
            trigger_id=trigger.id,
            max_attempts=trigger.max_attempts,
            chain_depth=depth + 1,
        )
        service.start_triggered_job(job.id)
        if created:
            logger.info(
                "Submitted workflow %s as job %s for event %s",
                trigger.workflow_id,
                job.id,
                event.event_id,
            )
        return job.id, created


def validate_automation(config: AutomationConfig, exports: dict[str, dict[str, Any]] | None = None) -> None:
    """Refuse a set of triggers that could only fail at run time, naming the task.

    Run when the tasks store is written, so a bad task is refused before it is saved, and again
    when the service starts or reloads.
    """
    for trigger in config.workflow_triggers:
        if workflows.get_workflow(trigger.workflow_id) is None:
            raise ValueError(f"Workflow task {trigger.id!r} references unknown workflow {trigger.workflow_id!r}")
    _validate_self_trigger(config)
    _validate_output_ownership(config)
    _validate_event_references(config)
    _validate_feature_references(config)
    _validate_deliveries(config, exports)


def _validate_self_trigger(config: AutomationConfig) -> None:
    """Refuse a task whose workflow writes the dataset it waits for: a loop of length one."""
    for trigger in config.workflow_triggers:
        workflow = workflows.get_workflow(trigger.workflow_id)
        output = _resolved_output_dataset(trigger, workflow) if workflow is not None else None
        if output is not None and output == trigger.on_update_of and trigger.event == "dataset.updated":
            raise ValueError(
                f"Workflow task {trigger.id!r} runs after {output!r} and writes {output!r}; it would start itself"
            )


def _record_delivery_refusal(trigger_id: str, delivery: TriggerDelivery, source_job_id: str, message: str) -> None:
    """A delivery that could not even be submitted is a failed run of its deliver task."""
    from open_climate_service.runs.service import record_run, run_for_job

    parent = run_for_job(source_job_id)
    record_run(
        task_id=delivery.task_id or f"{trigger_id}:deliver",
        kind="deliver",
        cause="event",
        cause_ref=source_job_id,
        parent_run_id=parent.id if parent else None,
        outcome="error",
        message=message,
    )


def _lists_delivery(record: OpenEOJobRecord, export_id: str) -> bool:
    deliveries = (record.usage or {}).get("deliveries")
    return isinstance(deliveries, list) and any(
        isinstance(item, dict) and item.get("export_id") == export_id for item in deliveries
    )


_service: WorkflowAutomationService | None = None


def get_workflow_automation_service() -> WorkflowAutomationService:
    """Return the process-local automation service singleton."""
    global _service
    if _service is None:
        _service = WorkflowAutomationService()
    return _service
