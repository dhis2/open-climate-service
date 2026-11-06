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


def test_a_refused_automation_holds_up_only_itself_and_a_dropped_failure_takes_its_recovery(
    clock: _Clock, webhook: WebhookReceiver
) -> None:
    webhook.refused = {"rain"}
    notifications.consume([_outcome("rain", 1, clock, failed=True)])
    notifications.consume([_outcome("heat", 1, clock, failed=True)])
    for _ in range(notifications._MAX_ATTEMPTS - 1):
        clock.now += timedelta(hours=2)
        notifications.replay()
    webhook.refused = set()
    notifications.consume([_outcome("rain", 2, clock, failed=False)])  # never told it failed
    notifications.consume([_outcome("rain", 3, clock, failed=True)])  # a new episode

    posted = [(status, body["id"], body["type"]) for status, _, body in webhook.posts]
    assert posted == [
        (503, "rain", "failed"),
        (200, "heat", "failed"),
        *[(503, "rain", "failed")] * (notifications._MAX_ATTEMPTS - 1),
        (200, "rain", "failed"),
    ]
    assert webhook.posts[-1][1] == "rain-run-3:0"
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
