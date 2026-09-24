"""DHIS2 organisation-unit feature provider (CLIM-1009).

Fetches an organisation-unit hierarchy from a configured DHIS2 connection (CLIM-840) and
returns it as a GeoJSON FeatureCollection -- the geometry source for a `provider: dhis2`
feature template.

Ships built-in -- a DHIS2 provider inside a DHIS2 climate service is core -- but is written
against public names only (`open_climate_service.exports.dhis2.get_connection`,
`open_climate_service.features.providers.feature_provider`), so the same file would work
unchanged as an installed-package or `plugins_dir` plugin. That constraint is what keeps the
provider seam honest: if this provider needed a new hook, an external provider would have
needed it too.
"""

from __future__ import annotations

import logging
from contextlib import closing
from typing import Any, Protocol

from open_climate_service.exports.dhis2 import get_connection
from open_climate_service.features.providers import feature_provider

logger = logging.getLogger(__name__)

_GEOJSON_FIELDS = "id,displayName,geometry"
_SAMPLE_SIZE = 5


class _OrgUnitClient(Protocol):
    """The small part of the DHIS2 client used by this provider."""

    def get_org_units_geojson(self, **params: Any) -> dict[str, Any]: ...

    def get_organisation_units(self, **params: Any) -> Any: ...


@feature_provider("dhis2")
def dhis2_org_units(connection: str, level: int | None = None, parent: str | None = None) -> dict[str, Any]:
    """Return an organisation-unit hierarchy as a GeoJSON FeatureCollection.

    `connection` names a `dhis2_connections` entry (CLIM-840); `level` and/or `parent` narrow
    the selection the same way DHIS2's own organisation unit queries do. Each feature's UID
    lands at `properties.id`, so a template using this provider declares `id_property: id`.

    Org units with no geometry are DHIS2's own doing -- the `.geojson` endpoint omits them
    before this ever sees them -- so they are counted by a second, geometry-free request rather
    than silently unaccounted for; see `_fetch_org_units` for that accounting.
    """
    with closing(get_connection(connection)) as client:
        return _fetch_org_units(client, level=level, parent=parent)


def _fetch_org_units(client: _OrgUnitClient, *, level: int | None, parent: str | None) -> dict[str, Any]:
    """The injectable body of `dhis2_org_units`, for testing without a live connection.

    Not reachable through a template's `params` -- YAML cannot carry a live client object --
    so a test calls this directly with a fake one, while a separate test proves the public
    function resolves and closes a real connection through `exports.dhis2.get_connection`.
    """
    selection = {key: value for key, value in {"level": level, "parent": parent}.items() if value is not None}

    collection: dict[str, Any] = client.get_org_units_geojson(fields=_GEOJSON_FIELDS, **selection)
    features = collection.get("features", [])
    _require_unique_ids(features)

    # The GeoJSON endpoint has its own selection vocabulary: `level` defaults to 1 and
    # `parent` means a subtree boundary. The ordinary metadata endpoint understands level,
    # but not that boundary parameter, so translate it to the equivalent path filter rather
    # than forwarding the same-looking query and comparing two different populations.
    metadata_selection: dict[str, Any] = {"level": level if level is not None else 1}
    if parent is not None:
        metadata_selection["filter"] = f"path:like:/{parent}/"
    all_units = {
        str(unit["id"]): unit.get("displayName")
        for unit in client.get_organisation_units(fields="id,displayName", **metadata_selection)
    }
    returned_ids = {feature["properties"]["id"] for feature in features}
    missing = {uid: name for uid, name in all_units.items() if uid not in returned_ids}
    if missing:
        sample = ", ".join(f"{uid} ({name})" for uid, name in list(missing.items())[:_SAMPLE_SIZE])
        more = f", and {len(missing) - _SAMPLE_SIZE} more" if len(missing) > _SAMPLE_SIZE else ""
        logger.warning(
            "DHIS2 organisation units (level=%r, parent=%r): %d of %d have no geometry and were skipped: %s%s",
            level,
            parent,
            len(missing),
            len(all_units),
            sample,
            more,
        )

    if not features:
        raise ValueError(
            f"no DHIS2 organisation units with geometry found for level={level!r}, parent={parent!r}; "
            f"{len(all_units)} matched the selection but none had geometry"
        )
    return collection


def _require_unique_ids(features: list[Any]) -> None:
    """Refuse a hierarchy with a missing or duplicate UID, naming the offending unit.

    The feature store's own `validate_feature_ids` catches this too -- the final,
    provider-agnostic guard every provider's output goes through -- but it names only a GeoJSON
    array index, which is fine for an arbitrary provider and unhelpful for one that already
    knows which organisation unit misbehaved. This runs first so the operator sees a DHIS2 UID
    and display name, not "feature 37".
    """
    seen: dict[str, Any] = {}
    for feature in features:
        properties = feature.get("properties") if isinstance(feature, dict) else None
        uid = properties.get("id") if isinstance(properties, dict) else None
        name = properties.get("displayName") if isinstance(properties, dict) else None
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"DHIS2 organisation unit {name!r} has no usable id; got {uid!r}")
        if uid in seen:
            raise ValueError(
                f"DHIS2 returned organisation unit '{uid}' ({name!r}) more than once; already seen as {seen[uid]!r}"
            )
        seen[uid] = name
