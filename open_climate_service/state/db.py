"""The operational database: ``<data_dir>/ocs.db``, SQLite in WAL mode (CLIM-927 direction).

What an operator edits and what the instance records about its own runs live here: tasks,
named exports, run records and the automation's activation boundaries. One writer at a time and
any number of readers across processes, crash-safe writes, and no whole-file rewrite that a full
disk can truncate into a behaviour change.

Every access goes through this module's small repository functions, never SQL at the call sites,
so moving to another store later is a change here only. Artifact and job records are still the
JSON files they were; moving those is the rest of CLIM-927 and out of this reference
implementation's scope.

``revision`` increases with every write, so a process can tell that another one changed
something without reading everything back (the clock's store watch uses it).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from open_climate_service import config as api_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS exports (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS activations (scope TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL,
    PRIMARY KEY (scope, key));
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    cause TEXT NOT NULL,
    cause_ref TEXT,
    parent_run_id TEXT,
    job_kind TEXT,
    job_id TEXT,
    outcome TEXT NOT NULL,
    message TEXT NOT NULL,
    started_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_task ON runs (task_id, started_at);
CREATE INDEX IF NOT EXISTS runs_by_job ON runs (job_id);
CREATE TABLE IF NOT EXISTS leases (name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL);
"""


class StateUnreadable(Exception):
    """The operational database exists but cannot be opened or read."""


def database_path() -> Path:
    """Where operational state lives: ``<data_dir>/ocs.db``."""
    return api_config.get_data_root() / "ocs.db"


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.executescript(_SCHEMA)
    except sqlite3.DatabaseError as exc:
        raise StateUnreadable(f"{path} cannot be opened: {exc}") from exc
    return connection


@contextmanager
def read() -> Generator[sqlite3.Connection]:
    """A connection for reading; sees the last committed state."""
    connection = _connect()
    try:
        yield connection
    except sqlite3.DatabaseError as exc:
        raise StateUnreadable(f"{database_path()} cannot be read: {exc}") from exc
    finally:
        connection.close()


@contextmanager
def write(*, configuration: bool = False) -> Generator[sqlite3.Connection]:
    """One write transaction, taken immediately so two writers queue rather than conflict.

    ``configuration`` marks a change to what the instance runs (tasks, exports): the revision
    moves on commit, so the clock and automation in every process reload. Bookkeeping such as
    runs, activations and lease renewals leaves it alone.
    """
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        if configuration:
            connection.execute(
                "INSERT INTO meta (key, value) VALUES ('revision', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
            )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def revision() -> str | None:
    """A token that changes with every write, or None before the first one."""
    if not database_path().exists():
        return None
    try:
        with read() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
    except StateUnreadable as exc:
        # A distinct token, so a watcher reloads and the reload reports why the store is unusable.
        return f"unreadable: {exc}"
    return str(row["value"]) if row is not None else None


# --- documents keyed by id: tasks and exports ------------------------------------------------------

_DOCUMENT_TABLES = {"tasks", "exports"}


def _table(name: str) -> str:
    if name not in _DOCUMENT_TABLES:
        raise ValueError(f"unknown document table {name!r}")
    return name


def list_documents(table: str, connection: sqlite3.Connection | None = None) -> dict[str, dict[str, Any]]:
    """Every document in ``table``, by id."""
    query = f"SELECT id, body FROM {_table(table)} ORDER BY id"  # noqa: S608  # table name is checked

    def rows(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        return {row["id"]: json.loads(row["body"]) for row in conn.execute(query)}

    if connection is not None:
        return rows(connection)
    with read() as conn:
        return rows(conn)


def put_document(connection: sqlite3.Connection, table: str, document_id: str, body: dict[str, Any]) -> None:
    """Insert or replace one document inside a write transaction."""
    connection.execute(
        f"INSERT INTO {_table(table)} (id, body) VALUES (?, ?) "  # noqa: S608
        "ON CONFLICT(id) DO UPDATE SET body = excluded.body",
        (document_id, json.dumps(body, sort_keys=True)),
    )


def delete_document(connection: sqlite3.Connection, table: str, document_id: str) -> bool:
    """Remove one document inside a write transaction; True when it existed."""
    cursor = connection.execute(f"DELETE FROM {_table(table)} WHERE id = ?", (document_id,))  # noqa: S608
    return cursor.rowcount > 0


def replace_documents(connection: sqlite3.Connection, table: str, documents: dict[str, dict[str, Any]]) -> None:
    """Replace every document in ``table`` inside a write transaction."""
    connection.execute(f"DELETE FROM {_table(table)}")  # noqa: S608
    for document_id, body in documents.items():
        put_document(connection, table, document_id, body)


# --- activation boundaries ---------------------------------------------------------------------------


def load_activations(scope: str) -> dict[str, Any]:
    """The automation's activation boundaries for ``scope``, by trigger id."""
    with read() as connection:
        return {
            row["key"]: json.loads(row["body"])
            for row in connection.execute("SELECT key, body FROM activations WHERE scope = ?", (scope,))
        }


def save_activations(scope: str, values: dict[str, Any]) -> None:
    """Replace ``scope``'s boundaries in one transaction: a crash leaves the old set or the new one."""
    with write() as connection:
        connection.execute("DELETE FROM activations WHERE scope = ?", (scope,))
        for key, value in values.items():
            connection.execute(
                "INSERT INTO activations (scope, key, body) VALUES (?, ?, ?)", (scope, key, json.dumps(value))
            )


# --- leases -----------------------------------------------------------------------------------------


def acquire_lease(name: str, holder: str, ttl_seconds: float, now: float | None = None) -> bool:
    """Take or renew the lease ``name`` for ``holder``; False while another holder's is unexpired.

    One transaction reads and writes it, so two processes asking at once cannot both win. A
    holder that stops renewing loses the lease when it expires, and the next asker takes it,
    with no operator action (CLIM-997).
    """
    import time

    moment = time.time() if now is None else now
    with write() as connection:
        row = connection.execute("SELECT holder, expires_at FROM leases WHERE name = ?", (name,)).fetchone()
        if row is not None and row["holder"] != holder and row["expires_at"] > moment:
            return False
        connection.execute(
            "INSERT INTO leases (name, holder, expires_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at",
            (name, holder, moment + ttl_seconds),
        )
    return True


def release_lease(name: str, holder: str) -> None:
    """Give the lease up at once, so another process need not wait for it to expire."""
    with write() as connection:
        connection.execute("DELETE FROM leases WHERE name = ? AND holder = ?", (name, holder))


def lease_holder(name: str, now: float | None = None) -> str | None:
    """Who holds ``name`` now, or None when nobody does."""
    import time

    moment = time.time() if now is None else now
    if not database_path().exists():
        return None
    with read() as connection:
        row = connection.execute("SELECT holder, expires_at FROM leases WHERE name = ?", (name,)).fetchone()
    return str(row["holder"]) if row is not None and row["expires_at"] > moment else None
