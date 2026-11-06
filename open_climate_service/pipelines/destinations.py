"""Stored pipelines as a source of named destinations, beside the instance configuration.

A pipeline owns exactly one named destination, under its own id: the definition its destination
plugin compiled when the pipeline last validated. Making it resolvable by name is what lets
a run's batch job save through it and its delivery job deliver it. Ids are unique across the
configuration file and the pipelines: a clash is refused when a pipeline is validated, and
a destination claimed by both is refused at resolution rather than resolved by precedence.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pipeline_destination_definitions() -> list[dict[str, Any]]:
    """The destination definition of every stored pipeline whose last validation passed."""
    from open_climate_service.pipelines import store

    definitions: list[dict[str, Any]] = []
    try:
        records = store.list_records()
    except Exception:
        logger.exception("Could not list pipelines for destination resolution")
        return definitions
    for record in records:
        validation = record.validation
        # Only a verdict on the current revision publishes: never a destination for an older spec.
        if (
            validation is None
            or not validation.valid
            or validation.destination_definition is None
            or validation.revision != record.revision
        ):
            continue
        definitions.append(dict(validation.destination_definition))
    return definitions
