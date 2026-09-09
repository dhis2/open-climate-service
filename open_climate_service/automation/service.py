"""Consume durable dataset updates and submit configured openEO workflows."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_climate_service import config as api_config
from open_climate_service.automation.config import AutomationConfig, WorkflowTrigger, get_automation_config
from open_climate_service.jobs import store as job_store
from open_climate_service.jobs.models import DATASET_UPDATED_EVENT_TYPE, JobEvent
from open_climate_service.openeo import workflows
from open_climate_service.openeo.jobs import OpenEOJobService, get_openeo_job_service
from open_climate_service.openeo.schemas import OpenEOJobCreate
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

_EVENT_VALUES = {
    "$event.dataset_id": "dataset_id",
    "$event.artifact_id": "artifact_id",
    "$event.action": "action",
    "$event.previous_end": "previous_end",
    "$event.current_end": "current_end",
}


def _resolve_event_values(value: Any, event: JobEvent) -> Any:
    """Replace exact event references recursively while preserving literal values."""
    if isinstance(value, str) and value in _EVENT_VALUES:
        return event.data.get(_EVENT_VALUES[value])
    if isinstance(value, list):
        return [_resolve_event_values(item, event) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_event_values(item, event) for key, item in value.items()}
    return value


def _activation_path() -> Path:
    """Return the file recording each trigger's activation boundary."""
    data_dir = api_config.get_data_dir()
    if data_dir is not None:
        base = data_dir
    else:
        xdg_data = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        base = xdg_data / "climate-service"
    return base / "automation" / "activation.json"


def _load_activations() -> dict[str, str]:
    """Return persisted trigger activation times, tolerating a missing or corrupt file."""
    path = _activation_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read automation activation file %s; triggers will not replay history", path)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Automation activation file %s is not a mapping; triggers will not replay history", path)
        return {}
    return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, str)}


def _save_activations(activations: dict[str, str]) -> None:
    """Persist trigger activation times atomically so a crash cannot leave a truncated file."""
    path = _activation_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(activations, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


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
        bindings.setdefault((trigger.on_update_of, output), []).append(trigger.id)
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

    def start(self) -> None:
        """Load configuration, validate triggers, and record activation boundaries.

        Validation runs even on a read-only instance so a configuration error surfaces at startup
        rather than only once the instance is made writable.
        """
        self._config = self._config_loader()
        if not self._config.workflow_triggers:
            return
        for trigger in self._config.workflow_triggers:
            if workflows.get_workflow(trigger.workflow_id) is None:
                raise ValueError(f"Workflow trigger {trigger.id!r} references unknown workflow {trigger.workflow_id!r}")
        _validate_output_ownership(self._config)
        _validate_event_references(self._config)
        if api_config.is_read_only():
            return
        activations = _load_activations()
        now = utc_now().isoformat()
        changed = False
        for trigger in self._config.workflow_triggers:
            if trigger.id not in activations:
                activations[trigger.id] = now
                changed = True
        if changed:
            _save_activations(activations)

    def replay(self) -> None:
        """Consume persisted events, honouring each trigger's activation boundary."""
        config = self._config or self._config_loader()
        if api_config.is_read_only() or not config.workflow_triggers:
            return
        service = self._openeo_service or get_openeo_job_service()
        activations = _load_activations()
        for record in job_store.list_job_records():
            for event in record.events:
                if event.type != DATASET_UPDATED_EVENT_TYPE:
                    continue
                for trigger in config.workflow_triggers:
                    if trigger.on_update_of != event.data.get("dataset_id"):
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
            if event.type != DATASET_UPDATED_EVENT_TYPE:
                continue
            for trigger in config.workflow_triggers:
                if trigger.on_update_of != event.data.get("dataset_id"):
                    continue
                self._submit_safely(trigger, event, service)

    def _submit_safely(self, trigger: WorkflowTrigger, event: JobEvent, service: OpenEOJobService) -> None:
        """Submit one trigger/event pair without letting a failure skip its siblings."""
        try:
            self._submit(trigger, event, service)
        except Exception:
            logger.exception(
                "Workflow trigger %s failed for event %s (dataset %s)",
                trigger.id,
                event.event_id,
                event.data.get("dataset_id"),
            )

    def _submit(self, trigger: WorkflowTrigger, event: JobEvent, service: OpenEOJobService) -> None:
        """Create and start the deterministic job for one trigger/event pair."""
        dataset_id = event.data.get("dataset_id")
        arguments = _resolve_event_values(trigger.arguments, event)
        body = OpenEOJobCreate(
            title=f"{trigger.workflow_id} after {dataset_id} update",
            description=f"Triggered by {event.event_id} using automation rule {trigger.id}",
            process={
                "process_graph": {
                    "workflow": {
                        "process_id": trigger.workflow_id,
                        "arguments": arguments,
                        "result": True,
                    }
                }
            },
        )
        job, created = service.create_triggered_job(
            body,
            source_event_id=event.event_id,
            trigger_id=trigger.id,
        )
        service.start_triggered_job(job.id)
        if created:
            logger.info(
                "Submitted workflow %s as job %s for event %s",
                trigger.workflow_id,
                job.id,
                event.event_id,
            )


_service: WorkflowAutomationService | None = None


def get_workflow_automation_service() -> WorkflowAutomationService:
    """Return the process-local automation service singleton."""
    global _service
    if _service is None:
        _service = WorkflowAutomationService()
    return _service
