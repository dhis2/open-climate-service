"""Stored pipelines as a source of named exports, beside the instance configuration.

A pipeline that validated compiles to exactly one named export under its own id. Making that
export resolvable is what lets a batch job save through it and a delivery job deliver it,
before and without the operator merging the generated YAML. The configuration file wins on
a clash; the pipeline's own validation refuses an id configured differently.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pipeline_export_definitions() -> list[dict[str, Any]]:
    """The export definition of every stored pipeline whose last validation passed."""
    from open_climate_service.pipelines import store
    from open_climate_service.pipelines.service import compile_export

    definitions: list[dict[str, Any]] = []
    try:
        records = store.list_records()
    except Exception:
        logger.exception("Could not list pipelines for export resolution")
        return definitions
    for record in records:
        validation = record.validation
        if validation is None or not validation.valid or validation.period_type is None:
            continue
        definitions.append(compile_export(record.spec, validation.period_type))
    return definitions
