"""The webhook dispatcher's queue and links, fed the automation events jobs and runs persist."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from open_climate_service import config as api_config
from open_climate_service.jobs.models import AUTOMATION_FAILED_EVENT_TYPE, AUTOMATION_SUCCEEDED_EVENT_TYPE, JobEvent
from open_climate_service.notifications import webhook as notifications

from .conftest import WebhookReceiver


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 6, 6, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(api_config, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(notifications, "utc_now", clock)
    return clock


def _outcome(pipeline_id: str, number: int, clock: _Clock, *, failed: bool) -> JobEvent:
    run = f"{pipeline_id}-run-{number}"
    data = {"run": run, "href": f"/pipelines/{pipeline_id}/runs/{run}"}
    if failed:
        data.update(stage="delivery", message="DHIS2 refused the payload")
    return JobEvent(
        event_id=f"{run}:0",
        time=clock.now,
        type=AUTOMATION_FAILED_EVENT_TYPE if failed else AUTOMATION_SUCCEEDED_EVENT_TYPE,
        source=f"/pipelines/{pipeline_id}",
        data=data,
    )


def test_replay_starts_delivery_without_posting_in_the_startup_thread(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.jobs import events as job_events

    kicked: list[bool] = []
    monkeypatch.setattr(notifications, "webhook_url", lambda: "http://hooks.test/ocs")
    monkeypatch.setattr(job_events, "persisted_events", lambda: [])
    monkeypatch.setattr(notifications, "_kick", lambda: kicked.append(True))
    monkeypatch.setattr(notifications, "send_due", lambda: pytest.fail("replay posted synchronously"))

    notifications.replay()

    assert kicked == [True]


def test_replay_reconstructs_transitions_when_an_older_failure_appears_late(
    clock: _Clock, webhook: WebhookReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_climate_service.jobs import events as job_events

    # Startup established the notification boundary before either outcome.
    notifications._decide([], since=clock.now)
    clock.now += timedelta(minutes=1)
    failure = _outcome("rain", 1, clock, failed=True)
    clock.now += timedelta(minutes=1)
    success = _outcome("rain", 2, clock, failed=False)

    # The success callback arrives while the older failure is not visible in the log yet.
    monkeypatch.setattr(job_events, "persisted_events", lambda: [success])
    notifications.consume([success])
    assert webhook.posts == []

    # A later replay sees the durable chronology and reports both transitions in order.
    monkeypatch.setattr(job_events, "persisted_events", lambda: [failure, success])
    notifications.replay()

    assert [(body["status"], key) for _, key, body in webhook.posts] == [
        ("failed", failure.event_id),
        ("recovered", success.event_id),
    ]


def test_a_stale_sender_cannot_settle_another_senders_claim(clock: _Clock) -> None:
    source = "/pipelines/rain"
    current = notifications._Queued(
        event_id="rain-run-1:0",
        source=source,
        payload={"status": "failed"},
        next_attempt_at=clock.now,
        claimed_until=clock.now + timedelta(minutes=2),
        claim_id="new-sender",
    )
    notifications.atomic_json(
        notifications._path(),
        notifications._State(since=clock.now, queue=[current]).model_dump(mode="json"),
    )
    stale = current.model_copy(update={"claim_id": "stale-sender"})

    notifications._settle(notifications._read(clock.now), stale, True, clock.now)

    [kept] = notifications._read(clock.now).queue
    assert kept.claim_id == "new-sender"


def test_a_refused_automation_holds_up_only_itself_and_a_dropped_failure_takes_its_recovery(
    clock: _Clock, webhook: WebhookReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every decision replays the whole log, so an earlier episode's posted recovery is seen
    again after the next failure is dropped, and must not count as that failure's ending."""
    from open_climate_service.jobs import events as job_events

    log: list[JobEvent] = []
    monkeypatch.setattr(job_events, "persisted_events", lambda: list(log))

    def outcome(automation: str, number: int, *, failed: bool) -> None:
        log.append(_outcome(automation, number, clock, failed=failed))
        notifications.consume([log[-1]])

    outcome("rain", 1, failed=True)
    outcome("rain", 2, failed=False)  # an earlier episode, posted in full
    webhook.refused = {"rain"}
    outcome("rain", 3, failed=True)
    outcome("heat", 1, failed=True)
    for _ in range(notifications._MAX_ATTEMPTS - 1):
        clock.now += timedelta(hours=2)
        notifications.replay()
    webhook.refused = set()
    outcome("rain", 4, failed=False)  # never told it failed
    outcome("rain", 5, failed=True)  # a new episode
    outcome("rain", 6, failed=False)

    posted = [(status, body["id"], body["status"]) for status, _, body in webhook.posts]
    assert posted == [
        (200, "rain", "failed"),
        (200, "rain", "recovered"),
        (503, "rain", "failed"),
        (200, "heat", "failed"),
        *[(503, "rain", "failed")] * (notifications._MAX_ATTEMPTS - 1),
        (200, "rain", "failed"),
        (200, "rain", "recovered"),
    ]
    assert [key for _, key, _ in webhook.posts[-2:]] == ["rain-run-5:0", "rain-run-6:0"]
    assert notifications._read(clock.now).queue == []


@pytest.mark.parametrize(
    ("base_url", "root_path", "link"),
    [
        ("https://ocs.example.org", "/ocs", "https://ocs.example.org/ocs/pipelines/rain"),
        ("https://example.org/climate", "/ocs", "https://example.org/climate/pipelines/rain"),
        ("", "/ocs", "/ocs/pipelines/rain"),
        ("", "", "/pipelines/rain"),
    ],
)
def test_links_name_the_public_root_and_the_deployment_prefix(
    clock: _Clock, webhook: WebhookReceiver, monkeypatch: pytest.MonkeyPatch, base_url: str, root_path: str, link: str
) -> None:
    monkeypatch.setenv("CLIMATE_SERVICE_BASE_URL", base_url)
    monkeypatch.setenv("ROOT_PATH", root_path)
    notifications.consume([_outcome("rain", 1, clock, failed=True)])

    [(_, _, body)] = webhook.posts
    assert body["link"] == link
    assert body["details"] == f"{link}/runs/rain-run-1"
    assert body["text"].endswith(f" {link}")
