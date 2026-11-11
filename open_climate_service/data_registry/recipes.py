"""A derived dataset's recipe: the workflow that produces it, and when it is computed again.

A dataset template's ``produced_by`` is either a workflow id, which only records the workflow
that makes the dataset, or a recipe that OCS can run::

    produced_by:
      workflow: climate_anomaly
      arguments:
        observed_dataset_id: era5land_precipitation_monthly
        normal_dataset_id: era5land_precipitation_monthly_normal_1991_2020
        variable: tp
      refresh_on: observed_dataset_id

``arguments`` are the fixed parameters that define the product; changing one makes another
dataset, with another id. OCS supplies ``output_dataset_id`` from the template's id, so it may
not be given. ``refresh_on`` names the argument holding the dataset that drives this one: when
that dataset updates, this one is computed again over its full coverage, which fills the
workflow's one open temporal-interval parameter. Updates of the other datasets it reads, such as
the normal an anomaly is measured against, do not refresh it. Without ``refresh_on`` the recipe
runs only when an operator asks, and its temporal range is one of its arguments.

This module checks what can be checked from the template alone. Whether the workflow exists and
accepts the recipe is checked once workflows are registered (`derivations.validate_recipes`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from open_climate_service.shared.provenance import json_digest
from open_climate_service.shared.urls import is_segment_safe_id

OUTPUT_PARAMETER = "output_dataset_id"
"""The parameter through which a derived-dataset workflow names the dataset it publishes."""

_RECIPE_KEYS = frozenset({"workflow", "arguments", "refresh_on"})


@dataclass(frozen=True)
class Recipe:
    """An executable ``produced_by``: what to run to produce the dataset ``output``."""

    output: str
    workflow: str
    arguments: dict[str, Any]
    refresh_on: str | None = None

    @property
    def input(self) -> str | None:
        """The dataset whose updates refresh this one, or None when only an operator runs it."""
        return str(self.arguments[self.refresh_on]) if self.refresh_on is not None else None

    @property
    def digest(self) -> str:
        """Identifies what the recipe computes, so a changed recipe is never taken as already run."""
        return json_digest({"workflow": self.workflow, "arguments": self.arguments, "refresh_on": self.refresh_on})


def parse(produced_by: object, *, dataset_id: str) -> tuple[str, Recipe | None]:
    """The producing workflow, and the recipe when ``produced_by`` is one. Raises on a malformed value."""
    if isinstance(produced_by, str):
        if not produced_by.strip() or produced_by != produced_by.strip():
            raise ValueError(f"invalid produced_by {produced_by!r}; it must be a workflow id or a recipe")
        return produced_by, None
    if not isinstance(produced_by, dict):
        raise ValueError(f"invalid produced_by {produced_by!r}; it must be a workflow id or a recipe")
    unknown = sorted(str(key) for key in produced_by if key not in _RECIPE_KEYS)
    if unknown:
        raise ValueError(f"produced_by has unknown keys {unknown}; a recipe has workflow, arguments and refresh_on")
    workflow = produced_by.get("workflow")
    if not isinstance(workflow, str) or not workflow.strip() or workflow != workflow.strip():
        raise ValueError(f"produced_by.workflow must be a workflow id, not {workflow!r}")
    arguments = produced_by.get("arguments", {})
    if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
        raise ValueError("produced_by.arguments must be a mapping of parameter names to values")
    if OUTPUT_PARAMETER in arguments:
        raise ValueError(f"produced_by.arguments may not set {OUTPUT_PARAMETER}: it is always the template's own id")
    refresh_on = produced_by.get("refresh_on")
    if refresh_on is not None:
        if not isinstance(refresh_on, str) or refresh_on not in arguments:
            raise ValueError(f"produced_by.refresh_on must name one of its arguments, not {refresh_on!r}")
        driver = arguments[refresh_on]
        if not isinstance(driver, str) or not is_segment_safe_id(driver):
            raise ValueError(f"produced_by.refresh_on names '{refresh_on}', which must hold a dataset id")
        if driver == dataset_id:
            raise ValueError("produced_by.refresh_on names the dataset itself; a dataset cannot refresh itself")
    return workflow, Recipe(output=dataset_id, workflow=workflow, arguments=dict(arguments), refresh_on=refresh_on)


def producing_workflow(template: dict[str, Any]) -> str | None:
    """The workflow a template says produces it, whichever form ``produced_by`` takes."""
    produced_by = template.get("produced_by")
    if isinstance(produced_by, str):
        return produced_by
    if isinstance(produced_by, dict):
        workflow = produced_by.get("workflow")
        return workflow if isinstance(workflow, str) else None
    return None


def recipe(template: dict[str, Any]) -> Recipe | None:
    """The template's recipe, or None when ``produced_by`` is absent or only names a workflow.

    Templates are validated when they load, so a malformed ``produced_by`` never reaches here.
    """
    produced_by = template.get("produced_by")
    if not isinstance(produced_by, dict):
        return None
    return parse(produced_by, dataset_id=str(template["id"]))[1]
