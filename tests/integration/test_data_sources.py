"""Fetch one period from every built-in data source, with the parameters its template ships with.

Unit tests fake the network, so they do not show that a source still works against the real
service: that its key is accepted and its endpoint has not moved. Each test reads one period over
a small window of the test instance's extent (Sierra Leone) and checks the shape of what comes
back. Run with `make test-integration`, with every source's key set up.
"""

import asyncio
import inspect
from datetime import date
from typing import Any

import pytest
import xarray as xr

from open_climate_service.data_registry.services import datasets
from open_climate_service.features import providers as feature_providers
from open_climate_service.features.templates import list_feature_templates
from open_climate_service.shared.plugin_loader import instantiate_plugin, instantiate_static_plugin
from open_climate_service.streaming.protocol import IngestionPlugin

_BBOX = [-11.75, 8.25, -11.25, 8.75]
_COUNTRY_CODE = "SLE"

# How far back to look for the newest period: far enough for a source that publishes late or has
# stopped, such as the land cover, whose last year is 2022.
_LOOKBACK_YEARS = 5

# Normals over one reference year instead of thirty: the check needs the source, not the climate.
_LIGHTER_PARAMS = {"era5_land_normals": {"period": [2020, 2020]}}


def _module(template: dict[str, Any]) -> str:
    return str(template["ingestion"]["plugin"]).rsplit(".", 2)[-2]


async def _fetch_period(plugin: IngestionPlugin, period_id: str, params: dict[str, Any]) -> xr.Dataset:
    # A plugin's fetch_period may be blocking or a coroutine, as the orchestrator allows.
    result = plugin.fetch_period(period_id, _BBOX, **params)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.integration
@pytest.mark.parametrize(
    "template",
    [template for template in datasets.list_datasets() if datasets.is_ingestable(template)],
    ids=lambda template: template["id"],
)
def test_raster_source_returns_one_period(template: dict[str, Any]) -> None:
    params = {
        **(template["ingestion"].get("params") or {}),
        **_LIGHTER_PARAMS.get(_module(template), {}),
        "country_code": _COUNTRY_CODE,
    }
    if datasets.is_timeless(template):
        ds = instantiate_static_plugin(template["ingestion"]["plugin"], dict(params)).fetch(_BBOX, **params)
        time_dim = None
    else:
        plugin = instantiate_plugin(template["ingestion"]["plugin"], dict(params))
        # The newest period, as a sync fetches it, so a moved cutoff or a broken listing shows here too.
        today = date.today()
        start = today.replace(year=today.year - _LOOKBACK_YEARS)
        periods = asyncio.run(plugin.periods(start.isoformat(), today.isoformat()))
        assert periods, f"the source lists no period since {start}"
        ds = asyncio.run(_fetch_period(plugin, periods[-1], params))
        time_dim = getattr(plugin, "time_dim", "t")

    assert isinstance(ds, xr.Dataset)
    values = ds[template["variable"]]
    assert values.sizes["y"] > 1 and values.sizes["x"] > 1
    if time_dim is not None:
        assert values.sizes[time_dim] == 1
    assert bool(values.notnull().any()), "every value is missing"


@pytest.mark.integration
@pytest.mark.parametrize("template", list_feature_templates(), ids=lambda template: template["id"])
def test_feature_source_returns_features(template: dict[str, Any]) -> None:
    provider = feature_providers.get_feature_provider(template["provider"])
    assert provider is not None

    result = provider(**template.get("params", {}))

    collection = result[0] if isinstance(result, tuple) else result
    assert collection["features"]
