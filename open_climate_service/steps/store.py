"""The one store for every step, in ``<data_dir>/steps.json`` (CLIM-1378).

Sync schedules (CLIM-1242), workflow triggers and their deliveries all live here, so an instance
has one place to edit what runs and one rule for when a change takes effect: at once. The file
is written under the cross-process lock the other indexes use and replaced atomically.

This is a whole-file JSON document like the other stores, deliberately: moving operational
state to SQLite is CLIM-927, and this module is the single place that would change.
"""

from __future__ import annotations

import json
from hashlib import blake2b
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.shared.persistence import atomic_json, index_lock
from open_climate_service.shared.time import utc_now
from open_climate_service.steps.models import Step, StepKind


class StepStoreUnreadable(Exception):
    """The steps file exists but cannot be parsed; nothing stored is trusted until fixed."""


def steps_path() -> Path:
    """Where steps live: ``<data_dir>/steps.json``."""
    return api_config.get_data_root() / "steps.json"


def store_stamp() -> str | None:
    """A content token for the store, or None when the file is absent."""
    try:
        contents = steps_path().read_bytes()
    except FileNotFoundError:
        return None
    return blake2b(contents, digest_size=16).hexdigest()


def _read(path: Path) -> dict[str, Step]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StepStoreUnreadable(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise StepStoreUnreadable(f"{path} must hold a mapping of step id to step")
    steps: dict[str, Step] = {}
    for step_id, value in raw.items():
        try:
            step = Step.model_validate(value)
        except ValidationError as exc:
            raise StepStoreUnreadable(f"{path}: step {step_id!r} is invalid: {exc}") from exc
        if step.id != step_id:
            raise StepStoreUnreadable(f"{path}: step stored under {step_id!r} has id {step.id!r}")
        steps[step_id] = step
    return steps


def _write(steps: dict[str, Step], path: Path) -> None:
    atomic_json(path, {key: steps[key].model_dump(mode="json") for key in sorted(steps)})


def list_steps(kind: StepKind | None = None) -> list[Step]:
    """Every stored step, by id, optionally of one kind. Raises when the file is unreadable."""
    steps = _read(steps_path())
    return [steps[key] for key in sorted(steps) if kind is None or steps[key].kind == kind]


def get_step(step_id: str) -> Step | None:
    """One stored step, or None."""
    return _read(steps_path()).get(step_id)


def save_step(step: Step, *, create: bool, check: Any = None) -> Step:
    """Create or replace one step under the store lock.

    ``check`` is called with the complete list of steps as it would be after this write, still
    under the lock, so a rule that spans steps (one sync step per dataset, a deliver step naming
    an existing workflow step) cannot be broken by two writers at once. It raises ValueError to
    refuse.
    """
    path = steps_path()
    with index_lock(path):
        steps = _read(path)
        existing = steps.get(step.id)
        if create and existing is not None:
            raise ValueError(f"A step with id '{step.id}' already exists")
        if not create and existing is None:
            raise ValueError(f"No step with id '{step.id}'")
        stamped = step.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else step.created_at,
                "updated_at": utc_now(),
            }
        )
        candidate = {**steps, step.id: stamped}
        if check is not None:
            check([candidate[key] for key in sorted(candidate)])
        _write(candidate, path)
    return stamped


def set_enabled(step_id: str, enabled: bool) -> Step:
    """Pause or resume one step."""
    path = steps_path()
    with index_lock(path):
        steps = _read(path)
        existing = steps.get(step_id)
        if existing is None:
            raise ValueError(f"No step with id '{step_id}'")
        if existing.enabled == enabled:
            return existing
        updated = existing.model_copy(update={"enabled": enabled, "updated_at": utc_now()})
        steps[step_id] = updated
        _write(steps, path)
    return updated


def delete_step(step_id: str, *, check: Any = None) -> bool:
    """Remove one step; True when something was removed. ``check`` as for ``save_step``."""
    path = steps_path()
    with index_lock(path):
        steps = _read(path)
        if step_id not in steps:
            return False
        remaining = {key: value for key, value in steps.items() if key != step_id}
        if check is not None:
            check([remaining[key] for key in sorted(remaining)])
        _write(remaining, path)
    return True
