"""The Automation section: how schedules, workflow triggers and pipelines connect.

``GET /automation`` opens the section's first tab, the Pipelines page, and ``GET /automation/graph``
returns the graph a pipeline's diagram is drawn from, as JSON. The Pipelines and Schedules pages
stay at their own paths and show the section tabs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from open_climate_service.automation.graph import AutomationGraph, build_graph
from open_climate_service.shared.urls import mount_prefix

router = APIRouter()


@router.get("", include_in_schema=False)
def automation_section(request: Request) -> RedirectResponse:
    """The section has no page of its own; it opens on Pipelines."""
    return RedirectResponse(f"{mount_prefix(request)}/pipelines")


@router.get("/graph", response_model=AutomationGraph)
def automation_graph() -> AutomationGraph:
    """Every configured path from a data source through schedules, triggers and pipelines, with latest status."""
    return build_graph()
