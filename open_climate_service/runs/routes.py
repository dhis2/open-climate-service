"""The runs API: what each task did, with live status, and what one run set off (CLIM-1378)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from open_climate_service.runs import service
from open_climate_service.runs.service import RunView

router = APIRouter()


class RunList(BaseModel):
    """Runs, newest first."""

    runs: list[RunView]


class RunChain(BaseModel):
    """One run, the run that caused it, and the runs it caused, down the chain."""

    run: RunView
    parent: RunView | None = None
    caused: list[RunChain] = []


def _chain(run: service.Run, depth: int = 0) -> RunChain:
    children = [] if depth >= 6 else service.list_runs(parent_run_id=run.id, limit=20)
    return RunChain(run=service.view(run), caused=[_chain(child, depth + 1) for child in children])


@router.get("", response_model=RunList)
def list_runs(
    task: str | None = Query(default=None, description="Only the runs of this task."),
    limit: int = Query(default=50, ge=1, le=500),
) -> RunList:
    """Every task run, newest first, each with its job's current status."""
    return RunList(runs=[service.view(run) for run in service.list_runs(task_id=task, limit=limit)])


@router.get("/{run_id}", response_model=RunChain)
def get_run(run_id: str) -> RunChain:
    """One run with the run that caused it and everything it set off: a sync followed to DHIS2."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run with id '{run_id}'")
    chain = _chain(run)
    parent = service.get_run(run.parent_run_id) if run.parent_run_id else None
    chain.parent = service.view(parent) if parent is not None else None
    return chain
