"""Provider ownership: `FeatureDetail.provider` and `refresh_feature_collection_from_provider`
(CLIM-926).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from open_climate_service import config as api_config
from open_climate_service.features import services as feature_services
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import FeatureDetail

DISTRICTS_TEMPLATE: dict[str, object] = {
    "id": "districts",
    "name": "District boundaries",
    "id_property": "orgUnitCode",
}


def _box(code: str, west: float, south: float, east: float, north: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"orgUnitCode": code},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
    }


WEST = _box("SL-W", -13.5, 6.9, -12.0, 8.0)
EAST = _box("SL-E", -11.0, 8.5, -10.1, 10.0)


def _collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features or (WEST, EAST))}


@pytest.fixture(autouse=True)
def feature_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the store at a temporary directory, and keep records out of the real index."""
    root = tmp_path / "features"
    monkeypatch.setattr(api_config, "get_features_root", lambda: root)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts_dir / "records.json")
    return root


def _provider(payload: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """A stand-in `@feature_provider` callable: returns a fixed collection, ignoring params."""
    return payload if payload is not None else _collection()


def _current_provider(dataset_id: str = "districts") -> str | None:
    records = [
        record
        for record in ingestion_services._load_records()
        if record.dataset_id == dataset_id and record.features is not None
    ]
    assert records, f"no registered record for '{dataset_id}'"
    latest = max(records, key=lambda record: record.created_at)
    assert latest.features is not None
    return latest.features.provider


# --- FeatureDetail.provider — the field itself -----------------------------------------------


def test_provider_defaults_to_none_for_a_hand_registered_collection() -> None:
    """A collection nothing declares a provider for is unowned, not owned by an empty string."""
    record = feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    assert record.features is not None
    assert record.features.provider is None


@pytest.mark.parametrize("bad", ["", "   ", " dhis2", "dhis2 ", "dhis2\n"])
def test_provider_rejects_blank_or_padded_names(bad: str) -> None:
    """Compared exactly against the name a later refresh presents, so padding must be rejected
    rather than silently stripped — a stripped mismatch would look like a changed owner."""
    with pytest.raises(ValueError, match="provider"):
        FeatureDetail(
            id_property="orgUnitCode", feature_count=1, primary_geometry="geometry", crs="EPSG:4326", provider=bad
        )


def test_provider_name_survives_serialization_and_reload() -> None:
    """The ownership fact has to persist across a process restart, not just live in memory."""
    feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    reloaded = ingestion_services._load_records()
    assert len(reloaded) == 1
    assert reloaded[0].features is not None
    assert reloaded[0].features.provider == "dhis2"


# --- the ownership table, exactly ---------------------------------------------------------


def test_first_materialization_succeeds_with_no_prior_record() -> None:
    """No record exists yet, so nothing can conflict: any registered provider may create one."""
    record = feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    assert record.features is not None
    assert record.features.provider == "dhis2"


def test_the_same_provider_refreshes_its_own_entry_in_place() -> None:
    first = feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    grown = _collection(WEST, EAST, _box("SL-N", -12.5, 8.5, -11.5, 9.5))
    second = feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=lambda **_: grown
    )

    assert second.artifact_id == first.artifact_id
    assert second.features is not None
    assert second.features.provider == "dhis2"
    assert second.features.feature_count == 3


def test_a_different_provider_is_refused_before_its_callable_runs() -> None:
    """Refusal happens before the provider is ever invoked, not after it has done the work."""
    feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    called = []

    def overture(**_: Any) -> dict[str, Any]:
        called.append(True)
        return _collection()

    with pytest.raises(ValueError, match="owned by provider 'dhis2'.*provider 'overture' cannot overwrite it"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="overture", provider=overture
        )

    assert called == [], "the conflicting provider must never be invoked"
    assert _current_provider() == "dhis2", "the existing record and its ownership are untouched"


def test_an_unowned_record_is_refused_for_every_provider_before_its_callable_runs() -> None:
    """A hand-registered collection is not fair game for any provider, not just a mismatched one."""
    feature_services.refresh_feature_collection(template=DISTRICTS_TEMPLATE, features=_collection())

    called = []

    def dhis2(**_: Any) -> dict[str, Any]:
        called.append(True)
        return _collection()

    with pytest.raises(ValueError, match="not provider-owned and cannot be overwritten by provider 'dhis2'"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=dhis2
        )

    assert called == []
    assert _current_provider() is None


def test_a_changed_template_cannot_transfer_ownership_implicitly() -> None:
    """Ownership is read from the stored record, never from what the template currently says.

    An operator editing `provider:` in a template (or a provider being renamed) must not be
    enough to adopt a collection someone else wrote; only an explicit adoption step should.
    """
    feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )
    # The template itself now claims a different provider -- as if an operator edited the YAML.
    relabelled_template = {**DISTRICTS_TEMPLATE, "provider": "overture"}

    with pytest.raises(ValueError, match="owned by provider 'dhis2'"):
        feature_services._refresh_feature_collection_from_provider(
            template=relabelled_template, provider_name="overture", provider=_provider
        )

    assert _current_provider() == "dhis2"


def test_two_competing_providers_cannot_both_pass_ownership_validation() -> None:
    """The lock spans check-then-write: nothing can observe 'unowned' twice and both proceed.

    Not a timing race (this store's lock already forces refreshes of one collection to be
    strictly sequential -- see test_the_collection_lock_is_still_held_while_the_record_is_written
    in test_feature_store.py for why a threaded test would not exercise anything a sequential
    one doesn't). This is the structural guarantee instead: the second call, run after the
    first has completed, must see the first's ownership and be refused by it.
    """
    feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    with pytest.raises(ValueError, match="owned by provider 'dhis2'"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="overture", provider=_provider
        )

    assert _current_provider() == "dhis2"


def test_provider_failure_leaves_the_previous_record_and_ownership_unchanged() -> None:
    """A provider that raises must not disturb the collection it failed to refresh."""
    first = feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=_provider
    )

    def failing(**_: Any) -> dict[str, Any]:
        raise RuntimeError("upstream unavailable")

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=failing
        )

    reloaded = ingestion_services._load_records()
    assert len(reloaded) == 1
    assert reloaded[0].artifact_id == first.artifact_id
    assert reloaded[0].features is not None
    assert reloaded[0].features.provider == "dhis2"
    assert reloaded[0].features.feature_count == 2


def test_the_provider_receives_its_templates_declared_params() -> None:
    """`params` is how a template configures its provider — a connection name, a level, ..."""
    received: dict[str, Any] = {}

    def dhis2(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return _collection()

    template = {**DISTRICTS_TEMPLATE, "params": {"connection": "national-hmis", "level": 2}}
    feature_services._refresh_feature_collection_from_provider(template=template, provider_name="dhis2", provider=dhis2)

    assert received == {"connection": "national-hmis", "level": 2}


def test_a_provider_returning_broken_identity_fails_before_anything_is_written() -> None:
    """The identity contract (CLIM-1068) applies through the provider path too.

    A duplicate identifier is not a dropped feature: two features would push against one org
    unit, and the failure must happen before either the file or the record exists -- not
    register a collection the identity contract already knows is broken.
    """
    duplicated = _collection(WEST, _box("SL-W", -11.0, 8.5, -10.1, 10.0))

    with pytest.raises(ValueError, match="repeats"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=lambda **_: duplicated
        )

    assert ingestion_services._load_records() == []


def test_a_read_only_instance_never_invokes_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuses before the lock and before the callable, and leaves nothing behind.

    A read-only instance serves what it already holds and never fetches or writes -- and a
    provider call is squarely a fetch, so this is refused unconditionally, whether or not a
    record already exists and regardless of who would own the result.
    """
    original = api_config.get_config
    monkeypatch.setattr(api_config, "get_config", lambda: {**original(), "read_only": True})

    called = []

    def dhis2(**_: Any) -> dict[str, Any]:
        called.append(True)
        return _collection()

    with pytest.raises(ValueError, match="read-only"):
        feature_services._refresh_feature_collection_from_provider(
            template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=dhis2
        )

    assert called == []
    assert ingestion_services._load_records() == []


def test_a_template_with_no_params_calls_the_provider_with_none() -> None:
    received = {"called": False}

    def dhis2(**kwargs: Any) -> dict[str, Any]:
        received["called"] = True
        assert kwargs == {}
        return _collection()

    feature_services._refresh_feature_collection_from_provider(
        template=DISTRICTS_TEMPLATE, provider_name="dhis2", provider=dhis2
    )

    assert received["called"] is True


def test_public_refresh_resolves_template_and_provider_together(monkeypatch: pytest.MonkeyPatch) -> None:
    template = {**DISTRICTS_TEMPLATE, "provider": "dhis2", "params": {"level": 2}}
    received: list[dict[str, Any]] = []

    def dhis2(**params: Any) -> dict[str, Any]:
        received.append(params)
        return _collection()

    monkeypatch.setattr(feature_services.feature_templates, "get_feature_template", lambda _: template)
    monkeypatch.setattr(feature_services.feature_providers, "get_feature_provider", lambda _: dhis2)

    record = feature_services.refresh_feature_collection_from_provider("districts")

    assert received == [{"level": 2}]
    assert record.features is not None
    assert record.features.provider == "dhis2"


def test_public_refresh_refuses_an_unknown_declared_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    template = {**DISTRICTS_TEMPLATE, "provider": "missing"}
    monkeypatch.setattr(feature_services.feature_templates, "get_feature_template", lambda _: template)
    monkeypatch.setattr(feature_services.feature_providers, "get_feature_provider", lambda _: None)

    with pytest.raises(ValueError, match="declares unknown provider 'missing'"):
        feature_services.refresh_feature_collection_from_provider("districts")
