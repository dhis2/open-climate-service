"""Post a webhook when a scheduled sync or a pipeline starts failing, and when it recovers.

Reads the automation events (`automation.failed`, `automation.succeeded`) that native jobs and
pipeline runs persist with their final outcome. Each event's source is one automation: a
schedule or a pipeline. Its health is kept in ``<data_dir>/notifications.json``, so a schedule
that keeps failing posts once, on its first failure, and once more when it next succeeds.

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

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from open_climate_service import config as api_config
from open_climate_service.jobs.models import AUTOMATION_FAILED_EVENT_TYPE, AUTOMATION_SUCCEEDED_EVENT_TYPE, JobEvent
from open_climate_service.shared import urls
from open_climate_service.shared.persistence import atomic_json, index_lock
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)

_OUTCOME_TYPES = frozenset({AUTOMATION_FAILED_EVENT_TYPE, AUTOMATION_SUCCEEDED_EVENT_TYPE})
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


class _State(BaseModel):
    since: datetime
    health: dict[str, _Health] = Field(default_factory=dict)
    queue: list[_Queued] = Field(default_factory=list)


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
    """Decide the automation events this process has just persisted, then post in the background."""
    outcomes = [event for event in events if event.type in _OUTCOME_TYPES]
    if not outcomes or _enabled() is None:
        return
    # A first event can be stamped just before the state is created, at its job's completion.
    _decide(outcomes, since=min([utc_now(), *(_utc(event.time) for event in outcomes)]))
    _kick()


def replay() -> None:
    """Decide every persisted automation event not decided yet, then post what is due."""
    from open_climate_service.jobs.events import persisted_events

    if _enabled() is None:
        return
    _decide([event for event in persisted_events() if event.type in _OUTCOME_TYPES], since=utc_now())
    send_due()


def _decide(events: list[JobEvent], *, since: datetime) -> None:
    with index_lock(_path()):
        state = _read(since)
        before = state.model_dump_json()
        for event in sorted(events, key=lambda item: (_utc(item.time), item.event_id)):
            _apply(state, event)
        if state.model_dump_json() != before or not _path().exists():
            atomic_json(_path(), state.model_dump(mode="json"))


def _apply(state: _State, event: JobEvent) -> None:
    """Move one automation's health to the event's outcome, queueing a notification on a change."""
    time = _utc(event.time)
    if time < state.since:
        return
    current = state.health.get(event.source)
    if current is not None and (time, event.event_id) <= (current.time, current.event_id):
        return  # already decided, or older than the outcome already known
    failed = event.type == AUTOMATION_FAILED_EVENT_TYPE
    state.health[event.source] = _Health(failed=failed, event_id=event.event_id, time=time)
    was_failing = current is not None and current.failed
    if failed != was_failing:
        state.queue.append(
            _Queued(
                event_id=event.event_id,
                source=event.source,
                payload=_payload(event, failed),
                next_attempt_at=utc_now(),
            )
        )


def _absolute(path: str) -> str:
    """The path under the public service root: a URL a receiver can open, when the origin is configured."""
    return urls.configured_public_root() + path


def _payload(event: JobEvent, failed: bool) -> dict[str, Any]:
    kind, _, name = event.source.strip("/").rpartition("/")
    automation = "pipeline" if kind == "pipelines" else "schedule"
    subject = f"Pipeline '{name}'" if automation == "pipeline" else f"Scheduled sync of '{name}'"
    link = _absolute(event.source)
    data = event.data
    if failed:
        stage = data.get("stage")
        text = f"{subject} failed{f' at {stage}' if automation == 'pipeline' and stage else ''}: {data.get('message')}"
    else:
        text = f"{subject} succeeded again"
    payload: dict[str, Any] = {
        "type": "failed" if failed else "recovered",
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
    """Claim and post each automation's due notifications; return whether any was accepted."""
    now = utc_now()
    with index_lock(_path()):
        state = _read(now)
        claimed: list[_Queued] = []
        waiting: set[str] = set()
        for item in state.queue:
            if item.source in waiting:
                continue  # behind an earlier notification of the same automation
            if item.next_attempt_at > now or (item.claimed_until is not None and item.claimed_until > now):
                waiting.add(item.source)
                continue
            item.claimed_until = now + _CLAIM
            claimed.append(item)
        if not claimed:
            return False
        atomic_json(_path(), state.model_dump(mode="json"))
    sent: set[str] = set()
    refused: set[str] = set()
    for item in claimed:
        if item.source in refused:
            continue
        if _post(url, item):
            sent.add(item.event_id)
        else:
            refused.add(item.source)
    with index_lock(_path()):
        _settle(_read(now), claimed, sent, refused, now)
    return bool(sent)


def _settle(state: _State, claimed: list[_Queued], sent: set[str], refused: set[str], now: datetime) -> None:
    """Record a round: drop what was sent, release the rest, and back off the refused."""
    mine = {item.event_id for item in claimed}
    abandoned: set[str] = set()  # automations whose failure was given up on
    kept: list[_Queued] = []
    for item in state.queue:
        if item.event_id in sent:
            continue
        if item.source in abandoned and item.payload.get("type") == "recovered":
            abandoned.discard(item.source)  # the receiver never got the failure it recovers from
            continue
        if item.event_id in mine:
            item.claimed_until = None
            if item.source in refused and item.event_id not in sent:
                item.attempts += 1
                refused.discard(item.source)  # only the first refused one was posted
                if item.attempts >= _MAX_ATTEMPTS:
                    logger.error("Dropping notification %s after %d attempts", item.event_id, item.attempts)
                    if item.payload.get("type") == "failed":
                        abandoned.add(item.source)
                    continue
                item.next_attempt_at = now + timedelta(minutes=min(60, 2**item.attempts))
        kept.append(item)
    for source in abandoned:
        # No recovery was queued yet: the next success must not post one either.
        health = state.health.get(source)
        if health is not None:
            health.failed = False
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
