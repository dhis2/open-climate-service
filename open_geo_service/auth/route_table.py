"""The one table that gives every method and path the application serves a tier.

A route the application serves that the table does not list gets the operator tier, and
`tests/test_authentication.py` fails until it is listed. A path no route serves is left to the
application, which answers 404. HEAD is decided as GET.
"""

from __future__ import annotations

import re

from starlette.routing import compile_path

from open_geo_service.auth.tokens import Tier

ROUTE_TABLE: dict[Tier, frozenset[str]] = {
    Tier.PUBLIC: frozenset(
        {
            # HTML pages and the API reference
            "GET /",
            "GET /api",
            "GET /automation",
            "GET /automation/graph",
            "GET /docs",
            "GET /docs/oauth2-redirect",
            "GET /map",
            "GET /openapi.json",
            "GET /openeo",
            "GET /redoc",
            "GET /workflows",
            "GET /workflows/{workflow_id}",
            # Instance and openEO discovery
            "GET /.well-known/openeo",
            "GET /credentials/basic",
            "GET /credentials/oidc",
            "GET /file_formats",
            "GET /health",
            "GET /info",
            "GET /me",
            "GET /service_types",
            # Web interface sign-in; the middleware checks the posts come from this instance's pages
            "GET /sign-in",
            "POST /sign-in",
            "POST /sign-out",
            # Catalogue and data reads
            "GET /collections",
            "GET /collections/{collection_id}",
            "GET /data-sources",
            "GET /data-sources/{dataset_id}",
            "GET /datasets",
            "GET /datasets/{dataset_id}",
            "GET /datasets/{dataset_id}/download",
            "GET /datasets/{dataset_id}/thumbnail.png",
            "GET /extent",
            "GET /features",
            "GET /features/{collection_id}",
            "GET /features/{collection_id}/data.parquet",
            "GET /features/{collection_id}/data.pmtiles",
            "GET /features/{collection_id}/styles/default.json",
            "GET /icechunk/{dataset_id}/{file_path:path}",
            "GET /stac/AGENTS.md",
            "GET /stac/README.md",
            "GET /stac/catalog.json",
            "GET /stac/{dataset_id}/AGENTS.md",
            "GET /stac/{dataset_id}/README.md",
            "GET /stac/{dataset_id}/collection.json",
            "GET /tables/{dataset_id}/data.parquet",
            "GET /tables/{dataset_id}/rows",
            "GET /zarr/{dataset_id}/{relative_path:path}",
            "GET /processes",
            "GET /processes/{process_id}",
            "GET /process_graphs",
            "GET /process_graphs/{process_graph_id}",
            # Status of operator work, readable as the pages show it
            "GET /ingestions",
            "GET /ingestions/{ingestion_id}",
            "GET /ingestions/jobs/{job_id}",
            "GET /pipelines",
            "GET /pipelines/new",
            "GET /pipelines/{pipeline_id}",
            "GET /pipelines/{pipeline_id}/edit",
            "GET /pipelines/{pipeline_id}/runs",
            "GET /pipelines/{pipeline_id}/runs/{run_id}",
            "GET /schedules",
            "GET /schedules/sync/{dataset_id}",
            "GET /schedules/sync/{dataset_id}/delete",
            "GET /sync/{dataset_id}/plan",
        }
    ),
    Tier.AUTHENTICATED: frozenset(
        {
            "POST /result",
            "GET /jobs",
            "POST /jobs",
            "GET /jobs/{job_id}",
            "PATCH /jobs/{job_id}",
            "DELETE /jobs/{job_id}",
            "GET /jobs/{job_id}/results",
            "POST /jobs/{job_id}/results",
            "DELETE /jobs/{job_id}/results",
            "GET /jobs/{job_id}/results/{filename}",
            "GET /jobs/{job_id}/results/result.geojson",
            "GET /jobs/{job_id}/results/result.zarr/{zarr_path:path}",
            "PUT /process_graphs/{process_graph_id}",
            "DELETE /process_graphs/{process_graph_id}",
        }
    ),
    Tier.OPERATOR: frozenset(
        {
            "POST /ingestions",
            "DELETE /ingestions/jobs/{job_id}",
            "POST /sync/{dataset_id}",
            "POST /features/{collection_id}/refresh",
            "POST /manage/features/refresh",
            "POST /manage/ingest",
            "POST /manage/sync",
            "POST /exports/{export_id}",
            "GET /exports/{export_id}/jobs/{delivery_job_id}",
            "POST /schedules/sync",
            "PUT /schedules/sync/{dataset_id}",
            "POST /schedules/sync/{dataset_id}",
            "DELETE /schedules/sync/{dataset_id}",
            "POST /schedules/sync/{dataset_id}/delete",
            "POST /schedules/sync/{dataset_id}/pause",
            "POST /schedules/sync/{dataset_id}/resume",
            "POST /pipelines",
            "POST /pipelines/new",
            "PUT /pipelines/{pipeline_id}",
            "DELETE /pipelines/{pipeline_id}",
            "POST /pipelines/{pipeline_id}/edit",
            "POST /pipelines/{pipeline_id}/delete",
            "POST /pipelines/{pipeline_id}/validate",
            "POST /pipelines/{pipeline_id}/dry-run",
            "POST /pipelines/{pipeline_id}/mode/{mode}",
            "POST /pipelines/{pipeline_id}/runs",
            "POST /pipelines/{pipeline_id}/runs/{run_id}/acknowledge",
            "POST /pipelines/{pipeline_id}/runs/{run_id}/deliver",
            "POST /pipelines/{pipeline_id}/runs/{run_id}/redeliver",
            "POST /pipelines/{pipeline_id}/runs/{run_id}/retry",
            "POST /pipelines/lookups/{plugin_id}/{lookup}",
        }
    ),
}


def _compile() -> list[tuple[str, re.Pattern[str], Tier]]:
    entries: list[tuple[str, re.Pattern[str], Tier]] = []
    for tier, routes in ROUTE_TABLE.items():
        for route in routes:
            method, path = route.split(" ", 1)
            entries.append((method, compile_path(path)[0], tier))
    return entries


_ENTRIES = _compile()


def listed_tier(method: str, path: str) -> Tier | None:
    """The tier the table gives a method and a path as the app's routes see it; None when unlisted.

    Where two templates match, as `/pipelines/new` and `/pipelines/{pipeline_id}` do, the higher
    tier wins.
    """
    method = "GET" if method.upper() == "HEAD" else method.upper()
    path = path.rstrip("/") or "/"
    tiers = [tier for entry_method, pattern, tier in _ENTRIES if entry_method == method and pattern.match(path)]
    return max(tiers, default=None)


def route_tier(method: str, path: str) -> Tier:
    """The tier a request needs: the listed one, or the operator tier for an unlisted route."""
    tier = listed_tier(method, path)
    return Tier.OPERATOR if tier is None else tier
