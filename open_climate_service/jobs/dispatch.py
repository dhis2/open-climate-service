"""Hand newly persisted events to each owner of what they mean.

Derivation keeps derived datasets current, pipelines deliver datasets to their destinations,
and the failure webhook posts changes in an automation's health. Each consumer is isolated: one that
fails is logged and does not keep the events from the others, and every consumer recovers what
it missed from the persisted log at startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from open_climate_service.jobs.models import JobEvent

logger = logging.getLogger(__name__)


def _consumers() -> list[tuple[str, Callable[[list[JobEvent]], None]]]:
    from open_climate_service.derivations import service as derivations
    from open_climate_service.notifications import webhook
    from open_climate_service.pipelines import runner

    return [
        ("Derived dataset refresh", derivations.consume),
        ("Pipeline dispatch", runner.consume),
        ("Failure notification", webhook.consume),
    ]


def consume(events: list[JobEvent]) -> None:
    """Give the events to every consumer, whatever any one of them does with them."""
    for name, consumer in _consumers():
        try:
            consumer(events)
        except Exception:
            logger.exception("%s failed for events %s", name, [event.event_id for event in events])
