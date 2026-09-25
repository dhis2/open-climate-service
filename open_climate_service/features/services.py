"""Materializing, registering and reading feature collections.

`refresh_feature_collection` and `refresh_feature_collection_from_provider` are the two doors a
collection is written and registered through; `registered_collections` and the `GET /features`
builders below them are how it is read back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from open_climate_service import config as api_config
from open_climate_service.features import providers as feature_providers
from open_climate_service.features import store
from open_climate_service.features import templates as feature_templates
from open_climate_service.features.schemas import FeatureCollectionListResponse, FeatureCollectionRecord
from open_climate_service.ingestions import services as ingestion_services
from open_climate_service.ingestions.schemas import ArtifactFormat, ArtifactRecord, ArtifactVersion
from open_climate_service.publications.services import managed_dataset_id_for
from open_climate_service.shared.licences import parse_licence


def refresh_feature_collection(
    *,
    template: dict[str, Any],
    features: Mapping[str, Any],
    store_crs: str = store.WGS84,
    bbox: Sequence[float] | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """Write a collection and register it as one operation, or leave the previous one in place.

    The public door for a caller that already has a FeatureCollection in hand — a hand-made
    registration, an admin action, a test. A provider run goes through
    `refresh_feature_collection_from_provider` instead, which needs the collection lock held
    across an ownership check *and* this refresh, and a plain second call to this function
    would try to acquire that lock again on the same thread and deadlock. `_locked` below is
    the shared body both doors call into; this one is the safe, ordinary entry point that
    acquires the lock itself, for a caller with no ownership question to ask first.
    """
    dataset_id = _validated_collection_id(template)
    with store.collection_lock(dataset_id):
        return _refresh_feature_collection_locked(
            dataset_id=dataset_id,
            template=template,
            features=features,
            store_crs=store_crs,
            bbox=bbox,
            publish=publish,
        )


def _validated_collection_id(template: dict[str, Any]) -> str:
    """Return the template's collection id, checked as a URL-and-filename-safe segment."""
    raw_id = str(template.get("id", ""))
    if not raw_id.strip():
        raise ValueError("feature collection template must declare a non-empty 'id'")
    # Unstripped on purpose: `validate_collection_id` rejects surrounding whitespace, and an id
    # is compared exactly everywhere else, so silently trimming here would accept a template
    # whose declaration does not say what its author meant.
    return store.validate_collection_id(raw_id)


def _refresh_feature_collection_locked(
    *,
    dataset_id: str,
    template: dict[str, Any],
    features: Mapping[str, Any],
    store_crs: str = store.WGS84,
    bbox: Sequence[float] | None = None,
    provider: str | None = None,
    version: ArtifactVersion | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """The write-then-register body of a refresh. Callable only with `dataset_id`'s lock held.

    Split out of `refresh_feature_collection` so `refresh_feature_collection_from_provider` can
    run its ownership check and this refresh under one lock acquisition (CLIM-926): writing and
    registering are two operations on one collection, and apart they fail badly, and the order
    matters more than it looks:

    * a refresh that replaced one stable file and then failed to register would leave the *old*
      record describing the *new* bytes, so `feature_count`, `extent` and `crs` all describe a
      file that no longer exists — and nothing raises, because the record is valid and a file is
      there. A crash in that window made the mismatch permanent;
    * two refreshes interleave, and whichever registers last stamps its numbers onto whichever
      file was replaced last.

    Both come from letting the bytes move while the record stands still. So the new version is
    written to its *own* path and the record is switched to it: at every instant, and from either
    record, resolving record then path yields bytes that record actually describes. Nothing has
    to be undone on failure — the new file is simply unreferenced.

    The older files are not removed here and now, even once the new record is durable. A reader
    can resolve the *old* record and only then open its file — two steps, not one — so a reader
    caught in that gap when this refresh lands must still find the old file exactly as it was.
    `prune_superseded_files` is what actually removes a superseded file, and it does so only
    once that file has already survived one full refresh cycle plus a grace period; see its
    docstring for why immediate deletion is not safe here.
    """
    id_property = template.get("id_property")
    if not isinstance(id_property, str) or not id_property:
        raise ValueError(f"feature collection template '{dataset_id}' must declare a non-empty 'id_property'")
    if id_property != id_property.strip():
        raise ValueError(
            f"feature collection template '{dataset_id}' declares id_property {id_property!r} with "
            "leading or trailing whitespace"
        )

    prior_records = [
        record
        for record in ingestion_services._load_records()
        if record.dataset_id == dataset_id and record.format == ArtifactFormat.GEOPARQUET
    ]
    # Refuse before writing anything, so a doomed refresh leaves no file to clean up.
    store.validate_features_for_write(dataset_id=dataset_id, features=features, id_property=id_property)
    written, _count, geometry = store.write_feature_collection(
        dataset_id=dataset_id,
        features=features,
        id_property=id_property,
        store_crs=store_crs,
    )
    try:
        record = ingestion_services.create_feature_artifact(
            template=template,
            features=features,
            store_path=written,
            crs=store_crs,
            primary_geometry=geometry,
            bbox=bbox,
            provider=provider,
            version=version,
            publish=publish,
        )
    except BaseException:
        # Records first, then the file. `create_feature_artifact` stores the record before it
        # publishes, so a publication failure leaves a *stored* record pointing at `written` —
        # and durably so: a caller can resolve that record (GET /features does not filter on
        # publication) before this handler ever runs. Restoring the record first removes that
        # exposure going forward; the file cleanup below must not reopen it going backward.
        #
        # That is why `written` is routed through `prune_superseded_files` rather than
        # unlinked directly. A caller that resolved the pre-rollback record a moment before
        # this handler ran is holding a path to `written`, exactly the gap the grace period
        # exists to protect — deleting it on the spot here would be the same race this whole
        # scheme was built to close, just reached from the failure path instead of a normal
        # refresh. Marking it instead defers the actual delete to a later prune call, once
        # `written` has aged past `SUPERSEDED_FILE_GRACE_SECONDS`.
        _restore_records(dataset_id, prior_records)
        restored = max(prior_records, key=lambda record: record.created_at) if prior_records else None
        restored_path = Path(str(restored.path)) if restored is not None else None
        store.prune_superseded_files(dataset_id, keep=restored_path)
        raise
    # Only now, with the record durable and pointing at `written`, can the files it
    # superseded go. A failure here leaves an orphan, not a wrong answer.
    store.prune_superseded_files(dataset_id, keep=written)
    return record


def _unpack_provider_result(result: Any, *, provider_name: str) -> tuple[Mapping[str, Any], ArtifactVersion | None]:
    """Accept either return form a provider may use: a collection, or one with its version.

    A provider that knows which upstream release it just fetched is the only thing that knows
    it truthfully. A template can declare `sync.version`, but that is a *claim* sitting beside
    the parameter that selects the release, and the two drift the moment one is edited — so a
    reported version wins over a declared one, and a provider with nothing to report says so by
    returning the collection alone.

    `authority` is the provider's own registry name, not something read out of the return
    value: a release identity is the pair, and a provider naming its own authority could claim
    another's scheme. That mirrors why `provider_name` is the caller's selected key everywhere
    else in this module.
    """
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                f"Feature provider {provider_name!r} returned a {len(result)}-tuple; it must return either a "
                "FeatureCollection or a (FeatureCollection, version) pair"
            )
        collection, declared = result
        if not isinstance(collection, Mapping):
            raise ValueError(
                f"Feature provider {provider_name!r} returned a pair whose first element is "
                f"{type(collection).__name__}, not a FeatureCollection"
            )
        if declared is None:
            return collection, None
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(
                f"Feature provider {provider_name!r} reported version {declared!r}; a version must be a "
                "non-empty string identifying the upstream release"
            )
        return collection, ArtifactVersion(value=declared.strip(), authority=provider_name)
    if not isinstance(result, Mapping):
        raise ValueError(
            f"Feature provider {provider_name!r} returned {type(result).__name__}; it must return either a "
            "FeatureCollection or a (FeatureCollection, version) pair"
        )
    return result, None


def refresh_feature_collection_from_provider(
    collection_id: str,
    *,
    store_crs: str = store.WGS84,
    bbox: Sequence[float] | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """Resolve a collection's declared provider and materialize it.

    The template and callable are resolved together from their registries so callers cannot
    associate an arbitrary callable with a trusted provider name. Ownership is still checked
    under the collection lock by the lower-level refresh operation.
    """
    template = feature_templates.get_feature_template(collection_id)
    if template is None:
        raise ValueError(f"Unknown feature collection template '{collection_id}'")
    provider_name = template.get("provider")
    if not isinstance(provider_name, str):
        raise ValueError(f"Feature collection template '{collection_id}' does not declare a provider")
    provider = feature_providers.get_feature_provider(provider_name)
    if provider is None:
        raise ValueError(f"Feature collection template '{collection_id}' declares unknown provider '{provider_name}'")
    return _refresh_feature_collection_from_provider(
        template=template,
        provider_name=provider_name,
        provider=provider,
        store_crs=store_crs,
        bbox=bbox,
        publish=publish,
    )


def _refresh_feature_collection_from_provider(
    *,
    template: dict[str, Any],
    provider_name: str,
    provider: Callable[..., feature_providers.ProviderResult],
    store_crs: str = store.WGS84,
    bbox: Sequence[float] | None = None,
    publish: bool = True,
) -> ArtifactRecord:
    """Materialize a collection by calling its provider, refusing to overwrite what it doesn't own.

    A separate, deliberate action from `load_features` (CLIM-926), which only ever reads what is
    already registered — a record is what makes a collection exist, and a graph author calling
    `load_features` expects a fast local read, not a mid-graph fetch of a whole hierarchy over
    the network. Something else (an operator action today; a scheduled trigger once CLIM-926's
    automation step lands) decides *when* a provider runs; this is what runs it, and the one
    place that has a question `refresh_feature_collection` does not ask: does this provider
    actually own the collection it is about to overwrite? Checked, and the provider invoked, and
    the refresh performed, all under one acquisition of the collection's lock — not
    check-then-release-then-run, which would let two different providers each pass the check
    before either has written anything.

    Refuses outright on a read-only instance, before the lock and before the provider is ever
    invoked: read-only instances serve what they already hold and never fetch or write, and a
    provider call is squarely a fetch.

    Ownership rules, read from `FeatureDetail.provider` on the *current* record:

    | current record | requested provider | result |
    | --- | --- | --- |
    | none | any registered provider | allowed — first materialization |
    | `provider=None` | any provider | refused — not provider-owned |
    | `provider="dhis2"` | `"dhis2"` | allowed — refreshes its own entry |
    | `provider="dhis2"` | `"overture"` | refused — owned by a different provider |

    `provider_name` is the caller's own selected registry name — the key it looked `provider`
    up under — never something read back out of `provider`'s return value; a provider does not
    get to declare its own identity. A changed template cannot transfer ownership either: the
    check reads what actually produced the *stored* record, not what the template currently
    says, so editing `provider:` in a template does not adopt a collection someone else (or no
    one) wrote. Adopting one is a deliberate, separate action this function does not perform.
    """
    if api_config.is_read_only():
        raise ValueError(
            f"This instance is read-only: provider '{provider_name}' cannot be run to refresh "
            f"feature collection {template.get('id')!r}. A read-only instance serves what it "
            "already holds and never fetches or writes."
        )
    dataset_id = _validated_collection_id(template)
    with store.collection_lock(dataset_id):
        _require_provider_ownership(dataset_id, requested_provider=provider_name)
        raw_params = template.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        features, version = _unpack_provider_result(provider(**params), provider_name=provider_name)
        return _refresh_feature_collection_locked(
            dataset_id=dataset_id,
            template=template,
            features=features,
            store_crs=store_crs,
            bbox=bbox,
            provider=provider_name,
            version=version,
            publish=publish,
        )


def _require_provider_ownership(dataset_id: str, *, requested_provider: str) -> None:
    """Refuse a provider run that would overwrite a collection it does not own.

    No record at all is the one case that is always allowed: nothing exists yet for a provider
    to be conflicting with. Once a record exists, only the provider already named on it — read
    from `FeatureDetail.provider`, never inferred from the current template — may refresh it;
    `None` there means no provider owns it (a hand-registered file, or one from some other
    origin) and refuses every provider equally.
    """
    current = registered_collections().get(dataset_id)
    if current is None:
        return
    owner = current.features.provider if current.features is not None else None
    if owner is None:
        raise ValueError(
            f"Feature collection '{dataset_id}' is not provider-owned and cannot be overwritten "
            f"by provider '{requested_provider}'."
        )
    if owner != requested_provider:
        raise ValueError(
            f"Feature collection '{dataset_id}' is owned by provider '{owner}'; provider "
            f"'{requested_provider}' cannot overwrite it."
        )


def _restore_records(dataset_id: str, prior_records: list[ArtifactRecord]) -> None:
    """Undo a registration that succeeded before a later publication step failed."""

    def restore(records: list[ArtifactRecord]) -> None:
        records[:] = [
            record
            for record in records
            if record.dataset_id != dataset_id or record.format != ArtifactFormat.GEOPARQUET
        ]
        records.extend(prior_records)

    ingestion_services._mutate_records(restore)


def registered_collections() -> dict[str, ArtifactRecord]:
    """Return the latest record for each registered feature collection, by collection id.

    Reads records, never the filesystem. A GeoParquet file sitting in the store directory that
    nothing registered is not a collection and does not appear here — there is no discovery step
    and so no state in which disk and index disagree. `_materialized_records` upstream already
    drops a record whose file has gone, so what is listed here is registered *and* present.

    Publication is not a filter. `/features` is the operator-facing inventory of what this
    instance holds, and an unpublished collection is still held; publication decides what the
    catalogues advertise, which is a different question asked in `stac_eligible_artifacts_by_dataset`.
    """
    collections: dict[str, ArtifactRecord] = {}
    for record in ingestion_services.list_artifacts().items:
        if record.format != ArtifactFormat.GEOPARQUET or record.features is None:
            continue
        collection_id = managed_dataset_id_for(record)
        current = collections.get(collection_id)
        if current is None or record.created_at > current.created_at:
            collections[collection_id] = record
    return dict(sorted(collections.items()))


def list_feature_collections() -> FeatureCollectionListResponse:
    """Return every registered feature collection.

    Templates are loaded once for the whole listing, avoiding one registry scan per collection.
    """
    collections = registered_collections()
    templates = _templates_by_id() if collections else {}
    return FeatureCollectionListResponse(
        items=[
            _build_record(collection_id, record, templates.get(record.dataset_id, {}))
            for collection_id, record in collections.items()
        ]
    )


def get_feature_collection_or_404(collection_id: str) -> FeatureCollectionRecord:
    """Return one registered feature collection, or raise 404."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return _build_record(collection_id, record, feature_templates.get_feature_template(record.dataset_id) or {})


def _templates_by_id() -> dict[str, dict[str, Any]]:
    """Return every declared template keyed by id, in one registry scan."""
    return feature_templates.feature_templates_by_id()


def get_collection_record_or_404(collection_id: str) -> ArtifactRecord:
    """Return the artifact record behind a registered collection, for callers that read it."""
    record = registered_collections().get(collection_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    return record


def published_collection_file_or_404(collection_id: str) -> Path:
    """Return the GeoParquet file a *published* collection is registered at, or raise 404.

    Resolved through the record, never by looking in the store directory. That is the same rule
    the listing follows, and it is what stops this route becoming a way to read any file that
    happens to be under the store root: a caller can only reach bytes some record already points
    at, and the record is the only thing that puts a file there.

    Publication is required here, unlike `/features`. The listing is the operator's inventory of
    what this instance holds; this is the asset a STAC collection advertises, and STAC only
    advertises published collections — so serving an unpublished one would hand out data the
    catalogue deliberately withholds. Resolved through the catalogue's own gate rather than a
    publication test of our own, so the two cannot disagree about which artifact that is.
    """
    # The STAC gate, not `registered_collections`. That one answers "what does this instance
    # hold" and returns the newest record; STAC advertises the newest *published* one. With an
    # older published collection and a newer unpublished one — the state `_dataset_links` and
    # the catalogue already handle — taking the newest first and then testing publication would
    # 404 the very asset the catalogue is advertising.
    record = ingestion_services.stac_eligible_artifacts_by_dataset().get(collection_id)
    if record is None or record.format != ArtifactFormat.GEOPARQUET:
        raise HTTPException(status_code=404, detail=f"Feature collection '{collection_id}' not found")
    raw = record.path or (record.asset_paths[0] if record.asset_paths else None)
    if raw is None:
        raise HTTPException(status_code=409, detail=f"Feature collection '{collection_id}' has no stored path")
    path = Path(raw)
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Feature collection '{collection_id}' is registered but its file is missing"
        )
    return path


def _build_record(collection_id: str, record: ArtifactRecord, template: dict[str, Any]) -> FeatureCollectionRecord:
    """Build one response row from a record and its already-resolved template.

    The template is passed in rather than looked up, so a listing resolves feature templates
    once. Keeping this lookup type-specific prevents a same-id raster template from supplying
    metadata for a GeoParquet collection.
    """
    detail = record.features
    if detail is None:  # pragma: no cover - registered_collections filters these out
        raise HTTPException(status_code=500, detail=f"Feature collection '{collection_id}' has no feature detail")
    licence = parse_licence(template.get("license"))
    return FeatureCollectionRecord(
        id=collection_id,
        name=record.dataset_name,
        description=_as_text(template.get("description")),
        license=licence.stac_license,
        license_url=licence.url,
        attribution=_feature_attribution(template),
        id_property=detail.id_property,
        feature_count=detail.feature_count,
        geometry_types=store.stored_geometry_types(record),
        primary_geometry=detail.primary_geometry,
        crs=detail.crs,
        version=record.version,
        extent=record.coverage,
        last_updated=record.created_at,
    )


def _as_text(value: Any) -> str | None:
    """Return a non-blank prose field, normalised the way the dataset path normalises one."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _feature_attribution(template: dict[str, Any]) -> str | None:
    """Return attribution, preferring explicit prose over derived provider names.

    STAC represents attribution as structured providers while the feature API has one prose
    field. Older templates may still declare attribution; keep that as the explicit override.
    Otherwise join valid provider names in declaration order, so the same metadata reaches both
    surfaces without flattening provider roles and URLs into prose.
    """
    explicit = _as_text(template.get("attribution"))
    if explicit is not None:
        return explicit
    declared = template.get("providers")
    if not isinstance(declared, list):
        return None
    names: list[str] = []
    for provider in declared:
        if not isinstance(provider, dict):
            continue
        name = _as_text(provider.get("name"))
        if name is not None and name not in names:
            names.append(name)
    return "; ".join(names) or None
