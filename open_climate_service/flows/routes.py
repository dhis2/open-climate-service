"""The flows API (CLIM-1377): the configured graph as JSON. The Automation page draws it."""

from __future__ import annotations

from fastapi import APIRouter

from open_climate_service.flows.service import FlowGraph, current_graph

router = APIRouter()


@router.get("", response_model=FlowGraph)
def get_flows() -> FlowGraph:
    """Every configured path from a source to a destination, as nodes and edges."""
    return current_graph()
