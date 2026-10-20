"""Generated STAC documents validate against core STAC and every extension they declare (CLIM-1484).

A document that declares an extension promises its fields follow that extension's schema, and a
validating client rejects the collection when they do not. Nothing checked this before, which is
how raster collections came to declare the CF extension while emitting `cf:cell_methods` in a
form its schema refuses.

The schemas are stored under `tests/data/stac_schemas/`, one file per schema URL with the URL's
host and path as its location, so the test needs no network. A schema the documents reference
but that is not stored fails the test, naming the URL, rather than being fetched. To add one,
save it at that path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

from open_climate_service.ingestions.schemas import (
    ArtifactCoverage,
    ArtifactFormat,
    ArtifactPublication,
    ArtifactRecord,
    ArtifactRequestScope,
    CoverageSpatial,
    CoverageTemporal,
    PublicationStatus,
)
from open_climate_service.stac import services as stac_services
from tests.test_stac_feature_collections import (
    districts,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    isolated_store,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

SCHEMAS = Path(__file__).parent / "data" / "stac_schemas"
CORE = "https://schemas.stacspec.org/v1.1.0"
CATALOG_SCHEMA = f"{CORE}/catalog-spec/json-schema/catalog.json"
COLLECTION_SCHEMA = f"{CORE}/collection-spec/json-schema/collection.json"


def _stored(uri: str) -> Resource[Any]:
    parts = urlsplit(uri)
    path = SCHEMAS / parts.netloc / parts.path.lstrip("/")
    if not path.is_file():
        raise NoSuchResource(ref=uri)
    return Resource.from_contents(json.loads(path.read_text(encoding="utf-8")))


_REGISTRY: Registry[Any] = Registry(retrieve=_stored)


def _errors(document: dict[str, Any], schema_uri: str) -> list[str]:
    """Every validation error for `document` against the stored schema at `schema_uri`."""
    schema = _stored(schema_uri).contents
    validator_class = validator_for(schema)
    validator = validator_class(schema, registry=_REGISTRY)
    return [
        f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in validator.iter_errors(document)
    ]


def _assert_valid(document: dict[str, Any], core_schema: str) -> None:
    """Valid against `core_schema` and each schema in the document's `stac_extensions`."""
    failures = {uri: _errors(document, uri) for uri in [core_schema, *document.get("stac_extensions", [])]}
    failures = {uri: errors for uri, errors in failures.items() if errors}
    assert not failures, json.dumps(failures, indent=2)


def _publish_raster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_id: str,
    values: np.ndarray,
    attrs: dict[str, str],
    template: dict[str, Any] | None = None,
) -> str:
    """Publish `values` as a real Zarr store, optionally with the dataset template it comes from."""
    zarr_path = tmp_path / f"{dataset_id}.zarr"
    ds = xr.Dataset(
        {"precip": (["time", "latitude", "longitude"], values)},
        coords={
            "time": pd.date_range("2026-01-01", periods=values.shape[0], freq="D"),
            "latitude": [4.0, 3.0, 2.0],
            "longitude": [1.0, 2.0, 3.0],
        },
    )
    ds["precip"].attrs = attrs
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)
    artifact = ArtifactRecord(
        artifact_id=f"{dataset_id}-1",
        dataset_id=dataset_id,
        dataset_name=dataset_id,
        variable="precip",
        period_type="daily",
        format=ArtifactFormat.ZARR,
        path=str(zarr_path),
        asset_paths=[str(zarr_path)],
        variables=["precip"],
        request_scope=ArtifactRequestScope(start="2026-01-01", end="2026-01-05"),
        coverage=ArtifactCoverage(
            spatial=CoverageSpatial(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
            temporal=CoverageTemporal(start="2026-01-01", end="2026-01-05"),
        ),
        created_at=datetime(2026, 1, 5, tzinfo=UTC),
        publication=ArtifactPublication(status=PublicationStatus.PUBLISHED, collection_id=dataset_id),
    )
    eligible = stac_services._eligible_artifacts_by_dataset  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(stac_services, "_eligible_artifacts_by_dataset", lambda: {**eligible(), dataset_id: artifact})
    if template is not None:
        get_dataset = stac_services.registry_datasets.get_dataset
        monkeypatch.setattr(
            stac_services.registry_datasets,
            "get_dataset",
            lambda requested: template if requested == dataset_id else get_dataset(requested),
        )
    return dataset_id


@pytest.fixture
def raster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A published continuous raster with CF attributes."""
    return _publish_raster(
        tmp_path,
        monkeypatch,
        dataset_id="chirps3_precipitation_daily",
        values=np.ones((5, 3, 3), dtype="float32"),
        attrs={
            "units": "mm/day",
            "long_name": "Precipitation",
            "standard_name": "lwe_precipitation_rate",
            "cell_methods": "time: mean",
        },
    )


def test_a_raster_collection_validates_against_every_schema_it_declares(client: TestClient, raster: str) -> None:
    document = client.get(f"/stac/collections/{raster}").json()

    # The case CLIM-1484 was about: CF fields are emitted, so the CF extension is declared.
    assert document["cube:variables"]["precip"]["cf:cell_methods"] == ["time: mean"]
    assert "https://stac-extensions.github.io/cf/v1.0.0/schema.json" in document["stac_extensions"]
    _assert_valid(document, COLLECTION_SCHEMA)


def test_a_categorical_raster_collection_validates_against_every_schema_it_declares(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared classes add the classification extension to the data assets."""
    dataset_id = _publish_raster(
        tmp_path,
        monkeypatch,
        dataset_id="land_cover",
        values=np.full((5, 3, 3), 2, dtype="uint8"),
        attrs={"long_name": "Land cover"},
        template={
            "id": "land_cover",
            "classes": [
                {"value": 1, "label": "Forest", "color": "#1a9850"},
                {"value": 2, "label": "Cropland", "color": "#fee08b"},
            ],
        },
    )

    document = client.get(f"/stac/collections/{dataset_id}").json()

    assert "https://stac-extensions.github.io/classification/v2.0.0/schema.json" in document["stac_extensions"]
    _assert_valid(document, COLLECTION_SCHEMA)


def test_a_feature_collection_validates_against_every_schema_it_declares(
    client: TestClient,
    districts: ArtifactRecord,  # noqa: F811
) -> None:
    document = client.get("/stac/collections/districts").json()

    _assert_valid(document, COLLECTION_SCHEMA)


def test_the_catalog_validates(client: TestClient, raster: str, districts: ArtifactRecord) -> None:  # noqa: F811
    document = client.get("/stac").json()

    assert {link["href"].rsplit("/", 1)[-1] for link in document["links"] if link["rel"] == "child"} >= {
        raster,
        "districts",
    }
    _assert_valid(document, CATALOG_SCHEMA)


def test_a_schema_that_is_not_stored_fails_instead_of_being_fetched() -> None:
    with pytest.raises(NoSuchResource):
        _stored("https://stac-extensions.github.io/not-stored/v1.0.0/schema.json")
