"""Internal streaming ingest engine.

The public ingestion API continues to live under `open_climate_service.ingestions`.
This package contains the internal execution pieces for the new per-period
streaming path introduced for issue #64 / CLIM-715.

This package is also the single import surface for writing a dataset plugin —
`BaseDatasetPlugin` plus the helpers a plugin reaches for (`normalize_period`,
`daily_period_ids`, `monthly_period_ids`) — so contributors don't have to hunt across
modules. The period-id helpers are re-exported from `open_climate_service.shared.time`,
their canonical home.
"""

from open_climate_service.shared.time import daily_period_ids, monthly_period_ids
from open_climate_service.streaming.base import BaseDatasetPlugin
from open_climate_service.streaming.helpers import bbox_slice, bbox_slices, cell_pad, normalize_period
from open_climate_service.streaming.orchestrator import StreamingIngestResult, run_streaming_ingest_sync
from open_climate_service.streaming.protocol import IngestionPlugin

__all__ = [
    "BaseDatasetPlugin",
    "IngestionPlugin",
    "StreamingIngestResult",
    "bbox_slice",
    "bbox_slices",
    "cell_pad",
    "daily_period_ids",
    "monthly_period_ids",
    "normalize_period",
    "run_streaming_ingest_sync",
]
