"""The durable domain-event log: every persisted event, whichever kind of job owns it.

A domain event (such as `dataset.updated`) is persisted atomically with the job whose work it
describes. Two kinds of job own events: a native job (an ingestion or a sync), which writes its
events when it succeeds, and an openEO job, which writes the event of a managed dataset it
publishes together with that publication's outcome. Consumers that replay events read them
here, never from one job store, so an event is found whichever job owns it.

There is no cursor over this log; `read_event_log` reads both stores once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from open_climate_service.jobs.models import JobEvent, JobStatus

_UNFINISHED_NATIVE = frozenset({JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.RETRYING})


@dataclass(frozen=True)
class EventLog:
    """Every persisted event, in a stable order, and the earliest time an unseen one can carry.

    ``open_since`` bounds the time of an event persisted after this read. It is the creation
    time of the oldest unfinished native job, because a native job stamps its events with its
    completion time before writing them, so one that completes just as the log is read can
    persist an event stamped earlier than the read. An openEO job needs no bound: it stamps its
    event inside the same store write that persists it, so an event this read missed carries a
    later time than the read. None when no native job is unfinished.
    """

    events: list[JobEvent]
    open_since: datetime | None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def read_event_log() -> EventLog:
    """Read every persisted event from the native and the openEO job stores.

    Ordered by time, then id, so events stamped with the same time replay in the same order.
    """
    from open_climate_service.jobs import store as native_store
    from open_climate_service.openeo.jobs import read_job_records

    events: list[JobEvent] = []
    open_since: list[datetime] = []
    for record in native_store.list_job_records():
        events.extend(record.events)
        if record.status in _UNFINISHED_NATIVE:
            open_since.append(_utc(record.created_at))
    for job in read_job_records():
        events.extend(job.events)
    events.sort(key=lambda event: (_utc(event.time), event.event_id))
    return EventLog(events=events, open_since=min(open_since) if open_since else None)


def persisted_events() -> list[JobEvent]:
    """Every persisted event, ordered by time, then id."""
    return read_event_log().events
