"""Sync check results and the clock's heartbeat, shared through ``<data_dir>/schedule_results.json``.

Only the process that runs the clock checks schedules, but every process sharing the data
directory reports their status: a public read-only container beside a private worker, or the
clock's own process after a restart. Results kept in that process's memory left the others
showing no runs at all (CLIM-1490), so the clock records each result here and every process
reads it back. The clock also records a heartbeat, which tells the others that a clock is
running somewhere, and so that the next check computed from a schedule's cron will happen.

The file is advisory: one that cannot be read reports no results rather than failing the
status, and a result that cannot be written is logged and dropped rather than failing the check.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler.dispatcher import CheckResult
from open_climate_service.shared.persistence import atomic_json, index_lock
from open_climate_service.shared.time import utc_now

logger = logging.getLogger(__name__)


class ClockBeat(BaseModel):
    """Which process runs the clock, the timezone its crons fire in, and when it last said so."""

    pid: int
    host: str
    # The clock's own ``scheduler.timezone``: another process may be configured with a different
    # one, and a cron read in the wrong zone gives a shifted next check. None in a beat written
    # before it was recorded.
    timezone: str | None = None
    beat_at: datetime = Field(default_factory=utc_now)

    def is_live(self, within: timedelta) -> bool:
        """Whether the beat is recent enough to trust that the clock still runs."""
        return utc_now() - self.beat_at <= within

    def is_this_process(self) -> bool:
        return self.pid == os.getpid() and self.host == socket.gethostname()


class SharedResults(BaseModel):
    """The last check of each schedule, by schedule id, and the clock's heartbeat."""

    results: dict[str, CheckResult] = Field(default_factory=dict)
    clock: ClockBeat | None = None


class SharedResultsStore:
    """Read and update the shared results file.

    ``path`` defaults to ``<data_dir>/schedule_results.json``, resolved at each call so it
    follows the configured data directory.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else api_config.get_data_root() / "schedule_results.json"

    def read(self) -> SharedResults:
        path = self.path
        try:
            return SharedResults.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return SharedResults()
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("Schedule results in %s could not be read; reporting none: %s", path, exc)
            return SharedResults()

    def _write(self, shared: SharedResults) -> None:
        atomic_json(self.path, shared.model_dump(mode="json"))

    def record(self, result: CheckResult) -> None:
        """Keep one check's result as its schedule's last."""
        with index_lock(self.path):
            shared = self.read()
            self._write(shared.model_copy(update={"results": {**shared.results, result.schedule_id: result}}))

    def keep_only(self, schedule_ids: set[str]) -> None:
        """Forget the results of schedules that no longer exist."""
        with index_lock(self.path):
            shared = self.read()
            kept = {key: value for key, value in shared.results.items() if key in schedule_ids}
            if kept != shared.results:
                self._write(shared.model_copy(update={"results": kept}))

    def beat(self, timezone: str) -> None:
        """Record that this process runs the clock, in ``timezone``, now."""
        with index_lock(self.path):
            shared = self.read()
            beat = ClockBeat(pid=os.getpid(), host=socket.gethostname(), timezone=timezone)
            self._write(shared.model_copy(update={"clock": beat}))

    def stop_beating(self) -> None:
        """Clear the heartbeat if it is this process's, so others stop reporting a clock at once."""
        with index_lock(self.path):
            shared = self.read()
            if shared.clock is not None and shared.clock.is_this_process():
                self._write(shared.model_copy(update={"clock": None}))
