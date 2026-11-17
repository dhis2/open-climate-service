"""First-write detection on a store that was created but never written to.

Ingest creates the Icechunk repository before it fetches anything, so any failure before the
first period commits leaves a repo whose branch tip holds only the initial snapshot and no zarr
hierarchy at all. ``is_store_empty`` has to call that empty: the orchestrator's first-write guard
refuses to touch a store it cannot classify, and an operator has no supported way to delete one
(CLIM-898), so a false "not empty" here wedges the dataset until someone removes the directory on
the host.

The negative cases matter as much as the positive ones: "nothing committed" must not widen into
"cannot inspect", because the value returned here decides whether ingest may write with
``mode="w"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")
zarr_errors = pytest.importorskip("zarr.errors")
pytest.importorskip("icechunk")  # required by the store helpers, not referenced here

from open_geo_service.streaming.orchestrator import run_streaming_ingest_sync  # noqa: E402
from open_geo_service.streaming.store import is_store_empty, open_or_create_repo  # noqa: E402


class _OnePeriodPlugin:
    """Minimal plugin, enough to drive a first write."""

    max_concurrency = 1
    commit_batch_size = 1

    async def periods(self, start: str, end: str) -> list[str]:
        _ = start, end
        return ["2026-01-01"]

    async def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> Any:
        _ = bbox, params
        return xr.Dataset(
            {"precip": (("t", "y", "x"), np.array([[[1.0]]], dtype=np.float32))},
            coords={"t": [np.datetime64(period_id, "D")], "y": [0.0], "x": [1.0]},
        )


def test_repo_created_but_never_written_is_empty(tmp_path: Path) -> None:
    """The state a failed first ingest leaves behind: initial snapshot, no zarr group."""
    store_path = tmp_path / "never-written.icechunk"
    open_or_create_repo(store_path)

    assert store_path.exists(), "repo directory should exist even with nothing committed"
    with pytest.raises(zarr_errors.GroupNotFoundError):
        # The condition being classified: the branch tip has no hierarchy to open.
        repo = open_or_create_repo(store_path)
        zarr.open_group(repo.readonly_session("main").store, mode="r")

    assert is_store_empty(store_path) is True


def test_repo_with_committed_data_is_not_empty(tmp_path: Path) -> None:
    """The guard still protects real data — otherwise the fix above enables a destructive write."""
    store_path = tmp_path / "has-data.icechunk"
    repo = open_or_create_repo(store_path)
    session = repo.writable_session("main")
    xr.Dataset(
        {"precip": (("t", "y", "x"), np.zeros((1, 2, 2), dtype="float32"))},
        coords={
            "t": np.array(["2026-01-01"], dtype="datetime64[ns]"),
            "y": [1.0, 0.0],
            "x": [0.0, 1.0],
        },
    ).to_zarr(session.store, mode="w", zarr_format=3)
    session.commit("one period")

    assert is_store_empty(store_path) is False


def test_ingest_recovers_from_a_repo_left_behind_by_a_failed_first_attempt(tmp_path: Path) -> None:
    """The whole point: a retry after a failed first ingest succeeds, with no manual cleanup.

    Nothing is monkeypatched here — this is the path that raised
    ``RuntimeError: ... is not empty, but committed periods could not be determined safely``
    on every attempt until the directory was deleted on the host.
    """
    store_path = tmp_path / "wedged.icechunk"
    open_or_create_repo(store_path)  # the failed attempt's leftovers

    result = run_streaming_ingest_sync(
        plugin=_OnePeriodPlugin(),
        params={},
        bbox=[0.0, 0.0, 1.0, 1.0],
        start="2026-01-01",
        end="2026-01-01",
        store_path=store_path,
        period_type="daily",
    )

    assert result.periods_written == 1
    assert is_store_empty(store_path) is False


def test_repo_with_only_root_attrs_is_not_empty(tmp_path: Path) -> None:
    """Attrs without arrays still count as content, as they did before."""
    store_path = tmp_path / "attrs-only.icechunk"
    repo = open_or_create_repo(store_path)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")
    root.attrs["title"] = "written by something"
    session.commit("root attrs only")

    assert is_store_empty(store_path) is False


def test_never_written_repo_is_empty_even_if_group_probing_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decision comes from Icechunk history, not from how zarr reports a missing group.

    How ``zarr.open_group`` signals "no hierarchy here" varies by version, so a fix that
    depended on catching one exception type would be one upgrade away from wedging again.
    """

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("some other zarr version's idea of a missing group")

    store_path = tmp_path / "never-written.icechunk"
    open_or_create_repo(store_path)
    monkeypatch.setattr(zarr, "open_group", explode)

    assert is_store_empty(store_path) is True
