"""The Automation section: how schedules, workflow triggers and pipelines connect.

``GET /automation`` is the Overview page (or the graph as JSON for an API client), and
``GET /automation/graph`` is the graph as JSON. ``GET /automation/triggers`` lists the workflow
triggers set in ``climate-service.yaml``, read-only. The Schedules and Pipelines pages stay at
their own paths and show the same section tabs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from open_climate_service.automation.graph import AutomationGraph, build_graph
from open_climate_service.shared.urls import mount_prefix

router = APIRouter()


@router.get("", response_model=AutomationGraph, response_class=Response)
def automation_overview(request: Request) -> Response:
    """The Automation overview page, or the automation graph as JSON."""
    from open_climate_service.system.templates import render_automation_page, wants_json

    if wants_json(request):
        return JSONResponse(build_graph().model_dump(mode="json"))
    return HTMLResponse(render_automation_page(mount_prefix(request), tab="overview"))


@router.get("/graph", response_model=AutomationGraph)
def automation_graph() -> AutomationGraph:
    """Every configured path from a data source through schedules, triggers and pipelines, with latest status."""
    return build_graph()


@router.get("/triggers", response_class=HTMLResponse)
def automation_triggers(request: Request) -> HTMLResponse:
    """The workflow triggers set in the instance configuration, read-only."""
    from open_climate_service.system.templates import render_automation_page

    return HTMLResponse(render_automation_page(mount_prefix(request), tab="triggers"))
