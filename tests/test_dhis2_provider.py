"""The DHIS2 organisation-unit feature provider (CLIM-1009).

`_fetch_org_units` is the injectable body `dhis2_org_units` wraps -- tested here against a fake
client so none of this needs the optional `dhis2-client` package or a live server. A separate,
small set of tests covers the public function's own plumbing: that it resolves a named
connection through `exports.dhis2.get_connection` and closes what it opens.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from open_climate_service.features.providers import get_feature_provider
from open_climate_service.plugins.features import dhis2 as provider
from open_climate_service.plugins.features.dhis2 import _fetch_org_units, _require_unique_ids, dhis2_org_units


def _org_unit(uid: str, name: str) -> dict[str, Any]:
    return {"id": uid, "displayName": name}


def _feature(uid: str, name: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"id": uid, "displayName": name},
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    }


class _FakeClient:
    """Duck-types the two `DHIS2Client` methods `_fetch_org_units` calls."""

    def __init__(self, org_units: list[dict[str, Any]], features: list[dict[str, Any]]) -> None:
        self._org_units = org_units
        self._features = features
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get_organisation_units(self, **params: Any) -> Any:
        self.calls.append(("list", params))
        return iter(self._org_units)  # the real client returns a lazy, auto-paginating generator

    def get_org_units_geojson(self, **params: Any) -> dict[str, Any]:
        self.calls.append(("geojson", params))
        return {"type": "FeatureCollection", "features": self._features}

    def close(self) -> None:
        self.closed = True


# --- discovery -------------------------------------------------------------------------------


def test_dhis2_provider_is_discovered_built_in() -> None:
    assert get_feature_provider("dhis2") is dhis2_org_units


# --- _fetch_org_units: the happy path and parameter forwarding --------------------------------


def test_fetch_org_units_returns_the_geojson_collection() -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [_feature("ou1", "A")])

    result = _fetch_org_units(client, level=2, parent=None)

    assert result == {"type": "FeatureCollection", "features": [_feature("ou1", "A")]}


def test_fetch_org_units_translates_the_geojson_parent_for_the_metadata_audit() -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [_feature("ou1", "A")])

    _fetch_org_units(client, level=2, parent="root")

    assert client.calls == [
        ("geojson", {"fields": "id,displayName,geometry", "level": 2, "parent": "root"}),
        ("list", {"fields": "id,displayName", "level": 2, "filter": "path:like:/root/"}),
    ]


def test_fetch_org_units_mirrors_the_geojson_default_level_in_the_metadata_audit() -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [_feature("ou1", "A")])

    _fetch_org_units(client, level=None, parent=None)

    assert client.calls[0] == ("geojson", {"fields": "id,displayName,geometry"})
    assert client.calls[1] == ("list", {"fields": "id,displayName", "level": 1})


# --- missing geometry: counted and logged, not silently dropped -------------------------------


def test_fetch_org_units_logs_a_warning_for_units_missing_geometry(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(
        [_org_unit("ou1", "Has geometry"), _org_unit("ou2", "No geometry")],
        [_feature("ou1", "Has geometry")],
    )

    with caplog.at_level(logging.WARNING, logger=provider.__name__):
        result = _fetch_org_units(client, level=2, parent=None)

    assert result["features"] == [_feature("ou1", "Has geometry")]
    assert "1 of 2 have no geometry" in caplog.text
    assert "ou2" in caplog.text
    assert "No geometry" in caplog.text


def test_fetch_org_units_fails_when_none_have_geometry() -> None:
    client = _FakeClient([_org_unit("ou1", "A"), _org_unit("ou2", "B")], [])

    with pytest.raises(ValueError, match="no DHIS2 organisation units with geometry.*2 matched"):
        _fetch_org_units(client, level=4, parent=None)


def test_fetch_org_units_does_not_warn_when_nothing_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [_feature("ou1", "A")])

    with caplog.at_level(logging.WARNING, logger=provider.__name__):
        _fetch_org_units(client, level=None, parent=None)

    assert caplog.text == ""


# --- identity: a DHIS2-specific error naming the offending unit -------------------------------


def test_require_unique_ids_accepts_distinct_units() -> None:
    _require_unique_ids([_feature("ou1", "A"), _feature("ou2", "B")])  # must not raise


def test_require_unique_ids_rejects_a_missing_id() -> None:
    bad = {
        "type": "Feature",
        "properties": {"displayName": "Nameless"},
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    }
    with pytest.raises(ValueError, match="'Nameless' has no usable id"):
        _require_unique_ids([bad])


def test_require_unique_ids_rejects_a_duplicate_id() -> None:
    with pytest.raises(ValueError, match="'ou1'.*more than once"):
        _require_unique_ids([_feature("ou1", "A"), _feature("ou1", "A again")])


def test_fetch_org_units_surfaces_a_duplicate_before_the_missing_geometry_audit() -> None:
    """The DHIS2-specific identity check runs first, so a bad payload fails fast."""
    client = _FakeClient([], [_feature("ou1", "A"), _feature("ou1", "A again")])

    with pytest.raises(ValueError, match="more than once"):
        _fetch_org_units(client, level=None, parent=None)

    assert client.calls == [("geojson", {"fields": "id,displayName,geometry"})]  # never reached the list call


# --- the public function: resolves and closes a named connection ------------------------------


def test_dhis2_org_units_resolves_and_closes_the_named_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [_feature("ou1", "A")])
    get_connection = MagicMock(return_value=client)
    monkeypatch.setattr(provider, "get_connection", get_connection)

    result = dhis2_org_units("national-hmis", level=2)

    get_connection.assert_called_once_with("national-hmis")
    assert result["features"] == [_feature("ou1", "A")]
    assert client.closed is True


def test_dhis2_org_units_closes_the_connection_even_when_fetching_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient([_org_unit("ou1", "A")], [])  # no geometry at all -> _fetch_org_units raises
    monkeypatch.setattr(provider, "get_connection", MagicMock(return_value=client))

    with pytest.raises(ValueError, match="no DHIS2 organisation units with geometry"):
        dhis2_org_units("national-hmis")

    assert client.closed is True


def test_dhis2_org_units_params_are_not_reachable_through_a_fake_client_kwarg() -> None:
    """`client` is not part of the provider's public signature -- a template's `params` (plain
    YAML values) cannot supply one, only `connection`/`level`/`parent` can."""
    import inspect

    assert set(inspect.signature(dhis2_org_units).parameters) == {"connection", "level", "parent"}


def test_the_provider_module_imports_nothing_private() -> None:
    """Written against public names only, so the same file would work unchanged as an
    installed-package or `plugins_dir` plugin -- an external provider has no way to import a
    leading-underscore name from another module either."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(provider))
    imported_names = [
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    private = [name for name in imported_names if name.startswith("_") and name != "_"]
    assert private == []
