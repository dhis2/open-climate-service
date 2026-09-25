"""Overture Maps divisions provider (CLIM-893, CLIM-836 build step 8).

No test here reaches the network. `_record_batch_reader` is the single seam the live call sits
behind, so every test below substitutes a reader built from Arrow batches shaped like the real
ones: geometry as WKB in a plain binary column, `names` as a `{primary, common, rules}` struct.
Those shapes are taken from a live `division_area` extract, not invented.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from shapely.geometry import Polygon

from open_climate_service.extents import services as extent_services
from open_climate_service.features import providers as feature_providers
from open_climate_service.features import services as feature_services
from open_climate_service.plugins.features import overture

SIERRA_LEONE = [-13.5, 6.9, -10.1, 10.0]


def _wkb(offset: float = 0.0) -> bytes:
    return Polygon([(offset, 0), (offset + 1, 0), (offset + 1, 1), (offset, 1)]).wkb


def _row(uid: str, subtype: str = "county", name: str = "Western Area Rural", **extra: Any) -> dict[str, Any]:
    return {
        "id": uid,
        "geometry": _wkb(),
        "subtype": subtype,
        "class": None,
        "names": {"primary": name, "common": None, "rules": None},
        "country": "SL",
        "region": "SL-W",
        "admin_level": 4,
        **extra,
    }


def _reader(rows: list[dict[str, Any]]) -> pa.RecordBatchReader:
    """A RecordBatchReader over `rows`, with the real extract's column set."""
    names_type = pa.struct([("primary", pa.string()), ("common", pa.string()), ("rules", pa.string())])
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("subtype", pa.string()),
            ("class", pa.string()),
            ("names", names_type),
            ("country", pa.string()),
            ("region", pa.string()),
            ("admin_level", pa.int32()),
        ]
    )
    batch = pa.RecordBatch.from_pylist(rows, schema=schema)
    return pa.RecordBatchReader.from_batches(schema, [batch])


@pytest.fixture
def stub_reader(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Substitute the network seam, recording the arguments it was called with."""
    calls: dict[str, Any] = {}

    def _install(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
        def _fake(overture_type: str, *, bbox: Any, release: str, stac: bool) -> Any:
            calls.update(type=overture_type, bbox=bbox, release=release, stac=stac)
            return None if rows is None else _reader(rows)

        monkeypatch.setattr(overture, "_record_batch_reader", _fake)
        return calls

    return _install


# --- the extract itself -----------------------------------------------------------------------


def test_divisions_become_a_geojson_feature_collection(stub_reader: Any) -> None:
    stub_reader([_row("a"), _row("b", name="Bo")])

    collection, _release = overture.overture_features(release="2026-09-23.0", bbox=SIERRA_LEONE)

    assert collection["type"] == "FeatureCollection"
    assert [f["properties"]["id"] for f in collection["features"]] == ["a", "b"]
    assert collection["features"][0]["geometry"]["type"] == "Polygon"


def test_the_divisions_theme_resolves_to_the_polygon_type(stub_reader: Any) -> None:
    """`divisions` publishes points, lines and polygons; only the areas are aggregation zones."""
    calls = stub_reader([_row("a")])

    overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert calls["type"] == "division_area"


def test_an_explicit_type_overrides_the_theme_default(stub_reader: Any) -> None:
    calls = stub_reader([_row("a")])

    overture.overture_features(release="r", bbox=SIERRA_LEONE, theme="divisions", type="division")

    assert calls["type"] == "division"


def test_an_unknown_theme_names_the_ones_it_knows(stub_reader: Any) -> None:
    stub_reader([_row("a")])

    with pytest.raises(ValueError, match="No default Overture type for theme 'weather'"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE, theme="weather")


def test_the_names_struct_is_flattened_to_a_plain_name(stub_reader: Any) -> None:
    """`{primary, common, rules}` would otherwise reach GeoJSON as a dict nobody reads."""
    stub_reader([_row("a", name="Bombali")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert collection["features"][0]["properties"]["name"] == "Bombali"
    assert "names" not in collection["features"][0]["properties"]


def test_a_column_projection_keeps_only_what_it_names(stub_reader: Any) -> None:
    stub_reader([_row("a")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE, columns=["id", "subtype"])

    assert collection["features"][0]["properties"] == {"id": "a", "subtype": "county"}


# --- filters: load-bearing for divisions, not a convenience ------------------------------------


def test_a_subtype_filter_keeps_one_administrative_level(stub_reader: Any) -> None:
    """The whole point: a bbox window spans country through neighbourhood."""
    stub_reader([_row("a", subtype="county"), _row("b", subtype="neighborhood"), _row("c", subtype="country")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE, filters={"subtype": "county"})

    assert [f["properties"]["id"] for f in collection["features"]] == ["a"]


def test_a_filter_accepts_several_values(stub_reader: Any) -> None:
    stub_reader([_row("a", subtype="county"), _row("b", subtype="region"), _row("c", subtype="locality")])

    collection, _release = overture.overture_features(
        release="r", bbox=SIERRA_LEONE, filters={"subtype": ["county", "region"]}
    )

    assert [f["properties"]["id"] for f in collection["features"]] == ["a", "b"]


def test_a_filter_on_an_absent_column_is_refused_not_silently_empty(stub_reader: Any) -> None:
    """Otherwise a typo reads as 'this bbox is empty' rather than 'this template is wrong'."""
    stub_reader([_row("a")])

    with pytest.raises(ValueError, match="no column\\(s\\) to filter on: admin_lvl"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE, filters={"admin_lvl": 4})


def test_a_projection_naming_an_absent_column_is_refused(stub_reader: Any) -> None:
    stub_reader([_row("a")])

    with pytest.raises(ValueError, match="no column\\(s\\): population"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE, columns=["id", "population"])


def test_a_filter_matching_nothing_says_so_with_the_filter_in_the_message(stub_reader: Any) -> None:
    stub_reader([_row("a", subtype="neighborhood")])

    with pytest.raises(ValueError, match="returned no rows.*matching"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE, filters={"subtype": "county"})


# --- confining the window to one country -------------------------------------------------------


def _sl_extent(monkeypatch: pytest.MonkeyPatch, country_code: str | None = "SLE") -> None:
    extent: dict[str, Any] = {"bbox": SIERRA_LEONE}
    if country_code is not None:
        extent["country_code"] = country_code
    monkeypatch.setattr(overture, "get_extent", lambda: extent)


def test_a_neighbouring_countrys_divisions_are_dropped_by_default(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured: a Sierra Leone bbox returns Coyah and Conakry, which are in Guinea."""
    _sl_extent(monkeypatch)
    stub_reader(
        [
            _row("a", name="Western Area Rural", country="SL"),
            _row("b", name="Conakry", country="GN"),
            _row("c", name="Coyah", country="GN"),
        ]
    )

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert [f["properties"]["name"] for f in collection["features"]] == ["Western Area Rural"]


def test_the_instance_alpha_3_code_is_matched_against_overtures_alpha_2(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extent.country_code` is alpha-3 (`NPL`); Overture's `country` column is alpha-2 (`NP`)."""
    monkeypatch.setattr(overture, "get_extent", lambda: {"bbox": SIERRA_LEONE, "country_code": "NPL"})
    stub_reader([_row("a", country="NP"), _row("b", country="IN")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert [f["properties"]["id"] for f in collection["features"]] == ["a"]


def test_an_explicit_country_may_be_given_in_either_form(stub_reader: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _sl_extent(monkeypatch, country_code=None)
    stub_reader([_row("a", country="SL"), _row("b", country="GN")])

    for given in ("SL", "sle", "SLE"):
        collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE, country=given)
        assert [f["properties"]["id"] for f in collection["features"]] == ["a"], given


def test_any_country_keeps_the_neighbours(stub_reader: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """For an extent that is deliberately transboundary."""
    _sl_extent(monkeypatch)
    stub_reader([_row("a", country="SL"), _row("b", country="GN")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE, country=overture.ANY_COUNTRY)

    assert [f["properties"]["id"] for f in collection["features"]] == ["a", "b"]


def test_an_explicit_country_filter_wins_over_the_instance_code(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template that says exactly what it means is not second-guessed."""
    _sl_extent(monkeypatch)
    stub_reader([_row("a", country="SL"), _row("b", country="GN")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE, filters={"country": "GN"})

    assert [f["properties"]["id"] for f in collection["features"]] == ["b"]


def test_no_declared_country_code_leaves_the_window_unconfined(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transboundary instance is legitimate; the bbox is then the whole selection."""
    _sl_extent(monkeypatch, country_code=None)
    stub_reader([_row("a", country="SL"), _row("b", country="GN")])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert [f["properties"]["id"] for f in collection["features"]] == ["a", "b"]


def test_an_unrecognised_instance_country_code_warns_and_does_not_confine(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Better an unconfined extract than silently dropping every row."""
    _sl_extent(monkeypatch, country_code="XXX")
    stub_reader([_row("a", country="SL"), _row("b", country="GN")])

    with caplog.at_level("WARNING"):
        collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert len(collection["features"]) == 2
    assert "not an ISO 3166-1 code" in caplog.text


# --- the bbox ----------------------------------------------------------------------------------


def test_bbox_defaults_to_the_instance_extent(stub_reader: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """So the shipped template carries no per-instance coordinate list."""
    calls = stub_reader([_row("a")])
    monkeypatch.setattr(extent_services, "get_extent", lambda: {"bbox": SIERRA_LEONE})
    monkeypatch.setattr(overture, "get_extent", lambda: {"bbox": SIERRA_LEONE})

    overture.overture_features(release="r")

    assert calls["bbox"] == (-13.5, 6.9, -10.1, 10.0)


def test_no_bbox_and_no_extent_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, stub_reader: Any) -> None:
    stub_reader([_row("a")])
    monkeypatch.setattr(overture, "get_extent", lambda: None)

    with pytest.raises(ValueError, match="declares no extent to fall back on"):
        overture.overture_features(release="r")


@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param([-13.5, 6.9, -10.1], id="three values"),
        pytest.param([-10.1, 6.9, -13.5, 10.0], id="inverted longitude"),
        pytest.param([-13.5, 10.0, -10.1, 6.9], id="inverted latitude"),
        pytest.param([-13.5, 6.9, -13.5, 10.0], id="zero width"),
    ],
)
def test_a_malformed_bbox_is_refused(bbox: list[float], stub_reader: Any) -> None:
    stub_reader([_row("a")])

    with pytest.raises(ValueError, match="Overture bbox"):
        overture.overture_features(release="r", bbox=bbox)


# --- the None the client returns instead of raising --------------------------------------------


def test_a_reader_of_none_is_raised_not_returned_as_empty(stub_reader: Any) -> None:
    """`record_batch_reader` returns None and prints its own message; that must not look empty."""
    stub_reader(None)

    with pytest.raises(ValueError, match="Overture returned no data for type 'division_area'"):
        overture.overture_features(release="2026-09-23.0", bbox=SIERRA_LEONE)


def test_the_none_from_a_stac_query_points_at_stac(stub_reader: Any) -> None:
    """Measured: `stac=True` returns nothing for division_area while the ordinary path works."""
    stub_reader(None)

    with pytest.raises(ValueError, match="does not cover every type; try stac=False"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE, stac=True)


def test_stac_is_off_by_default(stub_reader: Any) -> None:
    calls = stub_reader([_row("a")])

    overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert calls["stac"] is False


# --- geometry ----------------------------------------------------------------------------------


def test_a_row_without_geometry_is_dropped_not_emitted(stub_reader: Any) -> None:
    """A boundary with no shape is nothing to aggregate over."""
    stub_reader([_row("a"), _row("b", geometry=None)])

    collection, _release = overture.overture_features(release="r", bbox=SIERRA_LEONE)

    assert [f["properties"]["id"] for f in collection["features"]] == ["a"]


def test_rows_that_are_all_geometryless_fail_rather_than_returning_empty(stub_reader: Any) -> None:
    stub_reader([_row("a", geometry=None)])

    with pytest.raises(ValueError, match="returned no rows"):
        overture.overture_features(release="r", bbox=SIERRA_LEONE)


# --- registration and the plugin seam ----------------------------------------------------------


def test_the_provider_is_discovered_built_in() -> None:
    assert feature_providers.get_feature_provider("overture") is not None


def test_the_release_is_passed_through_as_given(stub_reader: Any) -> None:
    """The release is the collection's version, so it must not be rewritten or defaulted."""
    calls = stub_reader([_row("a")])

    overture.overture_features(release="2026-08-19.0", bbox=SIERRA_LEONE)

    assert calls["release"] == "2026-08-19.0"


def test_the_provider_module_imports_nothing_private() -> None:
    """Written against public names only, so the same file works as an installed-package plugin."""
    source = (overture.__file__ or "").replace(".pyc", ".py")
    with open(source, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip().startswith(("import ", "from "))]
    private = [
        line for line in lines if "open_climate_service" in line and any(part.startswith("_") for part in line.split())
    ]
    assert private == [], private


# --- the shipped template, and params reaching the provider from YAML --------------------------


def test_the_shipped_template_is_declared_and_names_this_provider() -> None:
    """The template ships built-in, so a `GET /dataset-templates` lists it with no config."""
    from open_climate_service.features import templates as feature_templates

    declared = {str(t["id"]): t for t in feature_templates.list_feature_templates()}

    template = declared["overture_divisions"]
    assert template["provider"] == "overture"
    # The identity contract: `id_property` must name a column the default projection keeps.
    assert template["id_property"] == "id"
    assert template["license"] == "ODbL-1.0"
    assert "OpenStreetMap" in template["attribution"]


def test_template_params_reach_the_provider_as_keyword_arguments(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What makes the YAML vocabulary real: an instance narrows the extract with no code change."""
    from open_climate_service.features import templates as feature_templates

    calls = stub_reader([_row("a", subtype="region", country="SL"), _row("b", subtype="county", country="SL")])
    monkeypatch.setattr(overture, "get_extent", lambda: {"bbox": SIERRA_LEONE, "country_code": "SLE"})
    template = {str(t["id"]): t for t in feature_templates.list_feature_templates()}["overture_divisions"]
    params = {**template["params"], "columns": ["id", "subtype"], "filters": {"subtype": "region"}}

    provider = feature_providers.get_feature_provider(str(template["provider"]))
    assert provider is not None, "the shipped template names a provider that is not registered"
    # Unpacked the way the refresh path does it, rather than assuming which return form this
    # provider happens to use — that is the seam's job, and a provider may use either.
    collection, _version = feature_services._unpack_provider_result(provider(**params), provider_name="overture")

    assert calls["release"] == template["params"]["release"]
    assert [f["properties"] for f in collection["features"]] == [{"id": "a", "subtype": "region"}]


# --- the release reaches the record as its version ---------------------------------------------


def test_the_provider_reports_the_release_it_read(stub_reader: Any) -> None:
    """The second half of the pair: what makes a monthly-release source answerable."""
    stub_reader([_row("a")])

    _collection, release = overture.overture_features(release="2026-09-23.0", bbox=SIERRA_LEONE)

    assert release == "2026-09-23.0"


def test_the_reported_release_becomes_the_collection_version(
    stub_reader: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """End to end through the seam: provider -> record.version, authority naming the provider."""
    from open_climate_service import config as api_config
    from open_climate_service.features import services as feature_services
    from open_climate_service.features import templates as feature_templates
    from open_climate_service.ingestions import services as ingestion_services

    monkeypatch.setattr(api_config, "get_features_root", lambda: tmp_path / "features")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(ingestion_services, "ARTIFACTS_INDEX_PATH", artifacts / "records.json")
    monkeypatch.setattr(overture, "get_extent", lambda: {"bbox": SIERRA_LEONE, "country_code": "SLE"})
    stub_reader([_row("a", country="SL"), _row("b", country="SL")])
    # `get_feature_template` reads an lru_cache that `list_feature_templates` bypasses, so a
    # stubbed template list from an earlier test outlives its monkeypatch here.
    feature_templates.reset_feature_template_caches()

    template = {str(t["id"]): t for t in feature_templates.list_feature_templates()}["overture_divisions"]
    record = feature_services.refresh_feature_collection_from_provider(str(template["id"]))

    assert record.version is not None, "the release the provider read was not recorded"
    assert record.version.value == template["params"]["release"]
    # The authority is the registry name the caller looked the provider up under, never
    # something the provider declared about itself.
    assert record.version.authority == "overture"
    assert record.features is not None and record.features.feature_count == 2
