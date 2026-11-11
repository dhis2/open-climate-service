"""Post a webhook when a scheduled sync or a pipeline starts failing, and when that ends.

Reads the automation events (`automation.failed`, `automation.succeeded`, `automation.resolved`)
that native jobs and pipeline runs persist with their outcomes. Each event's source is one
automation: a schedule or a pipeline. Its health is kept in ``<data_dir>/notifications.json``, so
a schedule that keeps failing posts once, on its first failure, and once more when it ends:
``recovered`` when it next succeeds, ``resolved`` when an operator acknowledged the failure.

A notification is queued in the same file before it is posted, and stays queued until the
receiver answers 2xx, so a crash or an unreachable receiver delays it rather than losing it.
`consume` decides the events a process has just persisted; `replay`, at startup and on the
maintenance interval, decides any it missed and retries what is queued. Each automation's
notifications are posted in order, so a recovery never arrives before its failure, while one
automation's refused notification does not hold up another's. A failure given up on after
`_MAX_ATTEMPTS` takes its recovery with it, so a receiver never gets a recovery alone. Events
older than the file, which is created the first time a webhook is configured, are history and
are not posted.

Each post carries the event id as ``Idempotency-Key``: a post whose answer was lost is sent
again, and the receiver can drop the repeat.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from open_climate_service import config as api_config
from open_climate_service.jobs.models import (
    AUTOMATION_FAILED_EVENT_TYPE,
    AUTOMATION_RESOLVED_EVENT_TYPE,
    AUTOMATION_SUCCEEDED_EVENT_TYPE,
    JobEvent,
)
from open_climate_service.shared import urls
from open_climate_service.shared.persistence import atomic_json, index_lock
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

# What a receiver is told when an automation's health changes on an event of each type.
_NOTIFICATION_TYPES = {
    AUTOMATION_FAILED_EVENT_TYPE: "failed",
    AUTOMATION_SUCCEEDED_EVENT_TYPE: "recovered",
    AUTOMATION_RESOLVED_EVENT_TYPE: "resolved",
}
_OUTCOME_TYPES = frozenset(_NOTIFICATION_TYPES)
_TIMEOUT_S = 10.0
# Attempts per notification before it is dropped: about five hours of retries.
_MAX_ATTEMPTS = 10
# How long a process may hold queued notifications while it posts them.
_CLAIM = timedelta(minutes=2)


class NotificationsConfig(BaseModel):
    """The ``notifications`` block of climate-service.yaml."""

    model_config = ConfigDict(extra="forbid")

    webhook_url: HttpUrl | None = None


def webhook_url() -> str | None:
    """The configured webhook URL, or None when notifications are off. Raises on a bad block."""
    raw = api_config.get_config().get("notifications") or {}
    try:
        settings = NotificationsConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"notifications in CLIMATE_SERVICE_CONFIG is invalid: {exc}") from exc
    return str(settings.webhook_url) if settings.webhook_url is not None else None


class _Health(BaseModel):
    failed: bool
    """Whether the last notification queued for the automation says it is failing."""
    event_id: str
    time: datetime


class _Queued(BaseModel):
    event_id: str
    source: str
    payload: dict[str, Any]
    attempts: int = 0
    next_attempt_at: datetime
    claimed_until: datetime | None = None
    claim_id: str | None = None


class _State(BaseModel):
    since: datetime
    health: dict[str, _Health] = Field(default_factory=dict)
    queue: list[_Queued] = Field(default_factory=list)
    decided: set[str] = Field(default_factory=set)
    abandoned: dict[str, str] = Field(default_factory=dict)
    """Per automation, the failure given up on, whose ending must not be posted either."""


def _path() -> Path:
    return api_config.get_data_root() / "notifications.json"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _read(since: datetime) -> _State:
    """The stored state; a new one counting events from ``since`` when there is none yet."""
    try:
        return _State.model_validate_json(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _State(since=since)


def _enabled() -> str | None:
    return None if api_config.is_read_only() else webhook_url()


def consume(events: list[JobEvent]) -> None:
    """Reconcile durable outcomes after a live callback, then post in the background."""
    outcomes = [event for event in events if event.type in _OUTCOME_TYPES]
    if not outcomes or _enabled() is None:
        return
    from open_climate_service.jobs.events import persisted_events

    # Callbacks from concurrent jobs can arrive out of order. Read the durable log so a later
    # callback also decides an earlier event whose own callback was delayed or missed. Keep the
    # supplied events too: a test double or a newly added event owner may not be in the log yet.
    durable = {(event.source, event.event_id): event for event in persisted_events() if event.type in _OUTCOME_TYPES}
    durable.update({(event.source, event.event_id): event for event in outcomes})
    # A first event can be stamped just before the state is created, at its job's completion.
    _decide(list(durable.values()), since=min([utc_now(), *(_utc(event.time) for event in outcomes)]))
    _kick()


def replay() -> None:
    """Decide every persisted automation event not decided yet, then post in the background."""
    from open_climate_service.jobs.events import persisted_events

    if _enabled() is None:
        return
    _decide([event for event in persisted_events() if event.type in _OUTCOME_TYPES], since=utc_now())
    _kick()


def _decide(events: list[JobEvent], *, since: datetime) -> None:
    with index_lock(_path()):
        state = _read(since)
        before = state.model_dump_json()
        _reconcile(state, events)
        if state.model_dump_json() != before or not _path().exists():
            atomic_json(_path(), state.model_dump(mode="json"))


def _event_key(event: JobEvent) -> str:
    return f"{event.source}\n{event.event_id}"


def _reconcile(state: _State, events: list[JobEvent]) -> None:
    """Rebuild health chronologically, queueing each transition not decided before.

    Rebuilding matters when a failure is persisted before a success but its live callback
    arrives later: advancing health from callback order would discard that failure forever.
    ``decided`` keeps transitions idempotent after their notification leaves the queue.
    """
    health: dict[str, bool] = {}
    opened: dict[str, str] = {}  # per automation, the failure that started its current episode
    latest: dict[str, _Health] = {}
    for event in sorted(events, key=lambda item: (_utc(item.time), item.event_id)):
        time = _utc(event.time)
        if time < state.since:
            continue
        failed = event.type == AUTOMATION_FAILED_EVENT_TYPE
        was_failing = health.get(event.source, False)
        health[event.source] = failed
        latest[event.source] = _Health(failed=failed, event_id=event.event_id, time=time)
        if failed == was_failing:
            continue
        key = _event_key(event)
        if failed:
            opened[event.source] = event.event_id
        elif state.abandoned.get(event.source) == opened.get(event.source):
            # The receiver never got this episode's failure, so it must not get its ending. The
            # ending of an earlier, posted episode is replayed too, and must not clear it.
            del state.abandoned[event.source]
            state.decided.add(key)
            continue
        if key in state.decided:
            continue
        state.decided.add(key)
        state.queue.append(
            _Queued(
                event_id=event.event_id,
                source=event.source,
                payload=_payload(event),
                next_attempt_at=utc_now(),
            )
        )
    state.health.update(latest)


def _absolute(path: str) -> str:
    """The path under the public service root: a URL a receiver can open, when the origin is configured."""
    return urls.configured_public_root() + path


def _payload(event: JobEvent) -> dict[str, Any]:
    kind, _, name = event.source.strip("/").rpartition("/")
    automation = "pipeline" if kind == "pipelines" else "schedule"
    subject = f"Pipeline '{name}'" if automation == "pipeline" else f"Scheduled sync of '{name}'"
    link = _absolute(event.source)
    data = event.data
    notification = _NOTIFICATION_TYPES[event.type]
    failed = notification == "failed"
    if failed:
        stage = data.get("stage")
        text = f"{subject} failed{f' at {stage}' if automation == 'pipeline' and stage else ''}: {data.get('message')}"
    elif notification == "resolved":
        text = f"{subject} failure was acknowledged by an operator"
    else:
        text = f"{subject} succeeded again"
    payload: dict[str, Any] = {
        "status": notification,
        "event": event.event_id,
        "time": _utc(event.time).isoformat(),
        "automation": automation,
        "id": name,
        "link": link,
        "run": data.get("run"),
        "job": data.get("job"),
        "details": _absolute(data["href"]) if data.get("href") else None,
        "stage": data.get("stage") if failed else None,
        "message": data.get("message") if failed else None,
        "text": f"{text} {link}",
    }
    return {key: value for key, value in payload.items() if value is not None}


def _kick() -> None:
    threading.Thread(target=_send_quietly, name="notification-webhook", daemon=True).start()


def _send_quietly() -> None:
    try:
        send_due()
    except Exception:
        logger.exception("Posting queued notifications failed; retrying at the next interval")


def send_due() -> None:
    """Post the queued notifications that are due, each automation's in order.

    Rounds repeat while one posts something: a notification queued while another sender held
    its automation's earlier one is sent once that sender is done, not at the next interval.
    """
    url = _enabled()
    if url is None or not _path().exists():
        return
    while _send_round(url):
        pass


def _send_round(url: str) -> bool:
    """Claim and post one due notification; return whether any work was claimed."""
    now = utc_now()
    with index_lock(_path()):
        state = _read(now)
        claimed: _Queued | None = None
        waiting: set[str] = set()
        for item in state.queue:
            if item.source in waiting:
                continue  # behind an earlier notification of the same automation
            if item.next_attempt_at > now or (item.claimed_until is not None and item.claimed_until > now):
                waiting.add(item.source)
                continue
            item.claim_id = str(uuid4())
            item.claimed_until = now + _CLAIM
            claimed = item.model_copy(deep=True)
            break
        if claimed is None:
            return False
        atomic_json(_path(), state.model_dump(mode="json"))
    accepted = _post(url, claimed)
    with index_lock(_path()):
        _settle(_read(now), claimed, accepted, now)
    return True


def _settle(state: _State, claimed: _Queued, accepted: bool, now: datetime) -> None:
    """Settle one claim only if this sender still owns it."""
    kept: list[_Queued] = []
    dropped: _Queued | None = None
    for item in state.queue:
        owns = item.event_id == claimed.event_id and item.source == claimed.source and item.claim_id == claimed.claim_id
        if not owns:
            kept.append(item)
            continue
        if accepted:
            continue
        item.claimed_until = None
        item.claim_id = None
        item.attempts += 1
        if item.attempts >= _MAX_ATTEMPTS:
            logger.error("Dropping notification %s after %d attempts", item.event_id, item.attempts)
            if item.payload.get("status") == "failed":
                dropped = item
            continue
        item.next_attempt_at = now + timedelta(minutes=min(60, 2**item.attempts))
        kept.append(item)
    if dropped is not None:
        # Transitions alternate, so the automation's next queued notification is this failure's
        # ending. Drop only that one: a newer episode behind it is still posted.
        ending = next((item for item in kept if item.source == dropped.source), None)
        if ending is None:
            state.abandoned[dropped.source] = dropped.event_id
        else:
            kept.remove(ending)
    state.queue = kept
    atomic_json(_path(), state.model_dump(mode="json"))


def _post(url: str, item: _Queued) -> bool:
    try:
        response = httpx.post(url, json=item.payload, headers={"Idempotency-Key": item.event_id}, timeout=_TIMEOUT_S)
    except httpx.HTTPError as exc:
        logger.warning("Notification %s could not be posted: %s", item.event_id, exc)
        return False
    if not response.is_success:
        logger.warning("Notification %s was refused: HTTP %d", item.event_id, response.status_code)
        return False
    return True
