"""The one store for every task: the ``tasks`` table of the operational database (CLIM-1378).

Sync schedules (CLIM-1242), workflow tasks and their deliveries all live here, so an instance has
one place to edit what runs and one rule for when a change takes effect: at once. A write runs
in one database transaction, and a rule that spans tasks is checked inside it, so two writers
cannot together break it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from open_climate_service.shared.time import utc_now
from open_climate_service.state import db
from open_climate_service.state.db import StateUnreadable
from open_climate_service.tasks.models import Task, TaskKind

TaskStoreUnreadable = StateUnreadable
"""The database, or a task in it, cannot be read; nothing stored is trusted until fixed."""

Check = Callable[[list[Task]], None]


def tasks_path() -> Path:
    """The file tasks are stored in: the operational database."""
    return db.database_path()


def store_stamp() -> str | None:
    """A token that changes with every write to operational state, or None before any."""
    return db.revision()


def _parse(documents: dict[str, dict[str, object]]) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for task_id, body in documents.items():
        try:
            task = Task.model_validate(body)
        except ValidationError as exc:
            raise TaskStoreUnreadable(f"task {task_id!r} is invalid: {exc}") from exc
        if task.id != task_id:
            raise TaskStoreUnreadable(f"task stored under {task_id!r} has id {task.id!r}")
        tasks[task_id] = task
    return tasks


def _dump(task: Task) -> dict[str, object]:
    return task.model_dump(mode="json")


def list_tasks(kind: TaskKind | None = None) -> list[Task]:
    """Every stored task, by id, optionally of one kind."""
    tasks = _parse(db.list_documents("tasks"))
    return [tasks[key] for key in sorted(tasks) if kind is None or tasks[key].kind == kind]


def get_task(task_id: str) -> Task | None:
    """One stored task, or None."""
    return _parse(db.list_documents("tasks")).get(task_id)


def save_task(task: Task, *, create: bool, check: Check | None = None) -> Task:
    """Create or replace one task.

    ``check`` is called with the complete list of tasks as it would be after this write, inside
    the write transaction, so a rule that spans tasks (one sync task per dataset, a deliver task
    after an existing workflow task) holds whoever writes. It raises ValueError to refuse.
    """
    with db.write() as connection:
        tasks = _parse(db.list_documents("tasks", connection))
        existing = tasks.get(task.id)
        if create and existing is not None:
            raise ValueError(f"A task with id '{task.id}' already exists")
        if not create and existing is None:
            raise ValueError(f"No task with id '{task.id}'")
        stamped = task.model_copy(
            update={
                "created_at": existing.created_at if existing is not None else task.created_at,
                "updated_at": utc_now(),
            }
        )
        candidate = {**tasks, task.id: stamped}
        if check is not None:
            check([candidate[key] for key in sorted(candidate)])
        db.put_document(connection, "tasks", task.id, _dump(stamped))
    return stamped


def set_enabled(task_id: str, enabled: bool) -> Task:
    """Pause or resume one task."""
    with db.write() as connection:
        existing = _parse(db.list_documents("tasks", connection)).get(task_id)
        if existing is None:
            raise ValueError(f"No task with id '{task_id}'")
        if existing.enabled == enabled:
            return existing
        updated = existing.model_copy(update={"enabled": enabled, "updated_at": utc_now()})
        db.put_document(connection, "tasks", task_id, _dump(updated))
    return updated


def delete_task(task_id: str, *, check: Check | None = None) -> bool:
    """Remove one task; True when something was removed. ``check`` as for ``save_task``."""
    with db.write() as connection:
        tasks = _parse(db.list_documents("tasks", connection))
        if task_id not in tasks:
            return False
        remaining = {key: value for key, value in tasks.items() if key != task_id}
        if check is not None:
            check([remaining[key] for key in sorted(remaining)])
        db.delete_document(connection, "tasks", task_id)
    return True


def replace_tasks(tasks: list[Task], *, check: Check | None = None) -> None:
    """Replace every task at once, for importing an operational configuration document."""
    with db.write() as connection:
        if check is not None:
            check(sorted(tasks, key=lambda task: task.id))
        db.replace_documents(connection, "tasks", {task.id: _dump(task) for task in tasks})
