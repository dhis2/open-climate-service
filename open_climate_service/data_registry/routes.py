"""FastAPI router exposing dataset template endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from .services import datasets

router = APIRouter()


def _with_ingestability(dataset: dict[str, Any]) -> dict[str, Any]:
    """A template plus the derived `ingestable` flag.

    Derived rather than declared, so a template author cannot get it wrong: it is read from
    the same `ingestion.plugin` that `create_artifact` requires. Without it the listing gives
    an operator no way to tell an ingestable template from a workflow output, and the answer
    arrives as a failed ingest (CLIM-912).
    """
    return {**dataset, "ingestable": datasets.is_ingestable(dataset)}


@router.get("/")
def list_dataset_templates() -> list[dict[str, Any]]:
    """Return the available dataset templates from the registry."""
    return [_with_ingestability(dataset) for dataset in datasets.list_datasets()]


def _get_dataset_or_404(dataset_id: str) -> dict[str, Any]:
    """Look up a dataset template by ID or raise 404."""
    dataset = datasets.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset


@router.get("/{dataset_id}", response_model=dict)
def get_dataset_template(dataset_id: str) -> dict[str, Any]:
    """Get a single dataset template by ID with derived coverage metadata."""
    # Note: have to import inside function to avoid circular import
    from ..data_accessor.services.accessor import get_data_coverage

    dataset = _get_dataset_or_404(dataset_id)
    coverage = get_data_coverage(dataset)
    dataset.update(coverage)
    return _with_ingestability(dataset)
