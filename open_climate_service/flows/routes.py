"""The flows API and page (CLIM-1377)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from open_climate_service.flows.service import FlowGraph, current_graph

router = APIRouter()


@router.get("", response_model=FlowGraph)
def get_flows(request: Request) -> Any:
    """Every configured path from a source to a destination, as nodes and edges, or the Flows page."""
    from open_climate_service.shared.urls import mount_prefix
    from open_climate_service.system.templates import render_flows_page, wants_json

    graph = current_graph()
    if request.query_params.get("format") == "json" or wants_json(request):
        return graph
    return HTMLResponse(render_flows_page(graph, mount_prefix(request)))
