"""Propagating a dataset licence to a derived product (CLIM-946 follow-up).

A derived product is a derivative work: a water mask computed from CC-BY-NC imagery inherits
the restriction, and publishing it as though it carried none is licence laundering by accident
— on the default path, with nothing to catch it.

The first attempt at this decided the licence in two places, one failing open and one failing
closed, and took four review rounds during which two of its own fixes came to contradict each
other. So the tests here are arranged around that failure:

* They drive the real publish paths — `_derive_managed_dataset_template` and
  `_enforce_licence_on_loaded_template` — not the comparison predicate. Every defect that got
  through the first attempt was invisible to a test that called the predicate directly.
* One test asserts the property that makes two paths safe: the decision is stable when fed
  back into itself, so the reload path cannot refuse what the synthesise path wrote.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from open_climate_service.shared.licences import UNDECLARED, parse_licence

if TYPE_CHECKING:
    import xarray as xr


def _cube(variable: str) -> xr.Dataset:
    """A real one-variable cube. The derived-template builder computes a display range from the
    values, so stubbing far enough to satisfy it costs more than handing it a small array."""
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {variable: (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )


def _derive(options: dict[str, Any], source_template: dict[str, Any] | None) -> dict[str, Any]:
    """Run the real derived-template builder, not a reimplementation of its licence rule."""
    from open_climate_service.openeo import jobs

    return jobs._derive_managed_dataset_template(_cube(str(options["variable"])), options, source_template, None)


_NC_SOURCE = {
    "id": "vantor_flood_rgb_2026",
    "name": "Flood true colour",
    "variable": "true_colour",
    "license": "CC-BY-NC-4.0",
    "providers": [{"name": "Vantor", "roles": ["producer"]}],
}
_ND_SOURCE = {"id": "nd_imagery", "variable": "rgb", "license": "CC-BY-ND-4.0"}
_VENDOR_SOURCE = {"id": "vendor_src", "variable": "v", "license": "Some Vendor Terms"}
_SA_SOURCE = {"id": "sa_src", "variable": "v", "license": "CC-BY-SA-4.0"}


# -- one decision, two paths ----------------------------------------------------------------


_CASES = [
    (None, None),
    (_NC_SOURCE, None),
    (_NC_SOURCE, "CC-BY-NC-4.0"),
    (_NC_SOURCE, "CC0-1.0"),
    (_NC_SOURCE, "CC-BY-NC-SA-4.0"),
    (_ND_SOURCE, None),
    (_ND_SOURCE, "CC-BY-ND-4.0"),
    (_VENDOR_SOURCE, None),
    (_VENDOR_SOURCE, "Some Vendor Terms"),
    (_VENDOR_SOURCE, "CC0-1.0"),
    (_SA_SOURCE, "CC-BY-4.0"),
    (_SA_SOURCE, "CC-BY-SA-4.0"),
    ({"id": "unreviewed", "variable": "v", "license": "BSD-3-Clause"}, None),
    ({"id": "unlabelled", "variable": "v"}, "CC0-1.0"),
]


@pytest.mark.parametrize(("source", "declared"), _CASES, ids=lambda v: str(v)[:40])
def test_the_decision_is_stable_when_fed_back(source: dict[str, Any] | None, declared: str | None) -> None:
    """The invariant that lets two code paths share one rule.

    Whatever licence the decision produces, asking again with that licence as the output's own
    claim must give the same verdict. Without this the synthesise path can write a template the
    reload path refuses — which is exactly what happened: an unparseable licence was inherited
    by design and then rejected as incomparable, so nothing derived from such a source could
    publish, and no unit test noticed because none drove both paths.
    """
    from open_climate_service.openeo import jobs

    licence, refusal = jobs._decide_derived_licence(source, declared)
    again_licence, again_refusal = jobs._decide_derived_licence(source, licence)

    assert again_refusal == refusal
    assert again_licence == licence


def test_a_verbatim_inherited_licence_survives_the_reload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that ended the first attempt, driven end to end rather than per-path.

    `_apply_derived_licence` copies an unparseable source licence forward, because carrying a
    licence forward cannot launder it. The reload check then saw a licence it could not parse
    on both sides and refused it, so no derived product from such a source could publish.
    """
    from open_climate_service.data_registry.services import datasets as reg
    from open_climate_service.openeo import jobs

    monkeypatch.setattr(reg, "CONFIGS_DIR", tmp_path)
    synthesised = _derive({"dataset_id": "derived", "variable": "v", "source_dataset_id": "vendor_src"}, _VENDOR_SOURCE)
    assert synthesised["license"] == "Some Vendor Terms"

    # Exactly what the publish path does next: persist, reload, check.
    reg.write_dataset_template(synthesised, overwrite=True)
    reloaded = reg.get_dataset("derived")
    assert reloaded is not None
    checked = jobs._enforce_licence_on_loaded_template("derived", reloaded, _VENDOR_SOURCE)
    assert checked["license"] == "Some Vendor Terms"


# -- the rule itself, through the real builder -----------------------------------------------


def test_a_product_derived_from_a_non_commercial_source_inherits_the_restriction() -> None:
    """The acceptance case: a water mask computed from CC-BY-NC imagery."""
    template = _derive(
        {"dataset_id": "flood_water_mask", "variable": "water", "source_dataset_id": "vantor_flood_rgb_2026"},
        _NC_SOURCE,
    )
    assert parse_licence(template["license"]).commercial_use is False
    # Attribution is a licence condition under CC-BY-NC, so it has to travel too.
    assert [p["name"] for p in template["providers"]] == ["Vantor"]


def test_publishing_a_derived_product_as_permissive_is_refused() -> None:
    with pytest.raises(ValueError, match="drops"):
        _derive(
            {
                "dataset_id": "flood_water_mask",
                "variable": "water",
                "source_dataset_id": "vantor_flood_rgb_2026",
                "license": "CC0-1.0",
            },
            _NC_SOURCE,
        )


def test_a_derived_product_may_declare_a_stricter_licence() -> None:
    template = _derive(
        {
            "dataset_id": "derived",
            "variable": "value",
            "source_dataset_id": "chirps3_precipitation_monthly",
            "license": "CC-BY-NC-4.0",
        },
        {"id": "chirps3_precipitation_monthly", "variable": "precip", "license": "CC0-1.0"},
    )
    assert template["license"] == "CC-BY-NC-4.0"


def test_deriving_from_an_unlabelled_source_leaves_the_output_unlabelled() -> None:
    """Not permissive by omission: an unlabelled input yields an unlabelled output, which
    publishes as `other` rather than as something reassuring."""
    template = _derive(
        {"dataset_id": "derived", "variable": "value", "source_dataset_id": "mystery"},
        {"id": "mystery", "variable": "value"},
    )
    assert parse_licence(template.get("license")) is UNDECLARED


def test_nothing_may_be_derived_from_a_no_derivatives_source() -> None:
    """ND forbids distributing adapted material at all, so there is no licence under which the
    derived product may be published — including the source's own."""
    with pytest.raises(ValueError, match="prohibits derivative works"):
        _derive({"dataset_id": "mask", "variable": "water", "source_dataset_id": "nd_imagery"}, _ND_SOURCE)


def test_saying_nothing_does_not_publish_what_naming_the_licence_would_refuse() -> None:
    """Declaring `CC-BY-ND-4.0` was refused while omitting `license` inherited the same licence
    and published, so the check was reachable only by naming what it would reject."""
    with pytest.raises(ValueError, match="prohibits derivative works"):
        _derive(
            {"dataset_id": "mask", "variable": "water", "source_dataset_id": "nd_imagery", "license": "CC-BY-ND-4.0"},
            _ND_SOURCE,
        )


def test_another_share_alike_licence_is_not_an_acceptable_substitute() -> None:
    """Share-alike is licence identity, not an obligation flag."""
    with pytest.raises(ValueError, match="share-alike"):
        _derive(
            {"dataset_id": "d", "variable": "v", "source_dataset_id": "sa_src", "license": "ODbL-1.0"},
            _SA_SOURCE,
        )


def test_deriving_from_an_unreviewed_spdx_licence_by_declaring_something_else_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not understand|cannot be compared"):
        _derive(
            {"dataset_id": "d", "variable": "v", "source_dataset_id": "unreviewed", "license": "CC0-1.0"},
            {"id": "unreviewed", "variable": "v", "license": "BSD-3-Clause"},
        )


# -- providers ------------------------------------------------------------------------------


def test_derived_providers_keep_the_source_licensor() -> None:
    """An explicit list adds to the source's, never replaces it."""
    template = _derive(
        {
            "dataset_id": "d",
            "variable": "water",
            "source_dataset_id": "vantor_flood_rgb_2026",
            "providers": [{"name": "DHIS2"}],
        },
        _NC_SOURCE,
    )
    assert [p["name"] for p in template["providers"]] == ["Vantor", "DHIS2"]


def test_an_empty_providers_option_cannot_erase_the_source_licensor() -> None:
    template = _derive(
        {"dataset_id": "d", "variable": "water", "source_dataset_id": "vantor_flood_rgb_2026", "providers": []},
        _NC_SOURCE,
    )
    assert [p["name"] for p in template["providers"]] == ["Vantor"]


def test_duplicate_providers_are_not_doubled() -> None:
    template = _derive(
        {
            "dataset_id": "d",
            "variable": "water",
            "source_dataset_id": "vantor_flood_rgb_2026",
            "providers": [{"name": "vantor"}],
        },
        _NC_SOURCE,
    )
    assert len(template["providers"]) == 1


# -- an already-registered output template ---------------------------------------------------


def test_a_preregistered_permissive_template_cannot_receive_restricted_data() -> None:
    """The path CLIM-946 recommends. The rule ran only while synthesising a template, so
    pre-registering one — which the ticket advises, to control the display — skipped it."""
    from open_climate_service.openeo import jobs

    with pytest.raises(ValueError, match="cannot receive data derived from"):
        jobs._enforce_licence_on_loaded_template(
            "flood_water_mask", {"id": "flood_water_mask", "license": "CC0-1.0"}, _NC_SOURCE
        )


def test_a_preregistered_undeclared_template_inherits_rather_than_staying_undeclared() -> None:
    from open_climate_service.openeo import jobs

    updated = jobs._enforce_licence_on_loaded_template("flood_water_mask", {"id": "flood_water_mask"}, _NC_SOURCE)
    assert updated["license"] == "CC-BY-NC-4.0"
    assert [p["name"] for p in updated["providers"]] == ["Vantor"]


def test_a_matching_preregistered_licence_still_has_to_credit_the_licensor() -> None:
    """A compatible identifier is not the whole licence: attribution is a condition of it, and
    this template names nobody."""
    from open_climate_service.openeo import jobs

    updated = jobs._enforce_licence_on_loaded_template(
        "flood_water_mask", {"id": "flood_water_mask", "license": "CC-BY-NC-4.0"}, _NC_SOURCE
    )
    assert [p["name"] for p in updated["providers"]] == ["Vantor"]


def test_a_template_needing_no_change_is_returned_untouched() -> None:
    """No write and no copy when the registered metadata is already right."""
    from open_climate_service.openeo import jobs

    template = {
        "id": "flood_water_mask",
        "license": "CC-BY-NC-4.0",
        "providers": [{"name": "Vantor", "roles": ["producer"]}],
    }
    assert jobs._enforce_licence_on_loaded_template("flood_water_mask", template, _NC_SOURCE) is template


def test_no_source_means_a_preregistered_template_is_untouched() -> None:
    from open_climate_service.openeo import jobs

    template = {"id": "standalone", "license": "CC0-1.0"}
    assert jobs._enforce_licence_on_loaded_template("standalone", template, None) is template


def test_the_enforced_licence_reaches_the_registry_not_just_this_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """STAC builds the collection from the on-disk template (`build_collection` reads the
    registry by dataset id), so an in-memory fix publishes the same wrong licence."""
    from open_climate_service.data_registry.services import datasets as reg
    from open_climate_service.openeo import jobs

    monkeypatch.setattr(reg, "CONFIGS_DIR", tmp_path)
    jobs._enforce_licence_on_loaded_template(
        "flood_water_mask",
        {"id": "flood_water_mask", "variable": "water", "sync": {"kind": "static"}},
        _NC_SOURCE,
    )

    written = yaml.safe_load((tmp_path / "flood_water_mask.yaml").read_text())[0]
    assert written["license"] == "CC-BY-NC-4.0"
    assert [p["name"] for p in written["providers"]] == ["Vantor"]


# -- lineage that does not resolve ------------------------------------------------------------


def test_an_unresolvable_source_dataset_id_is_an_error_not_absent_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning None for a typo would silently mean "not derived from anything", which
    disables propagation entirely — so a typo would publish a permissive output from a
    restrictive source with no error anywhere."""
    from open_climate_service.data_registry.services import datasets as reg
    from open_climate_service.openeo import jobs

    monkeypatch.setattr(reg, "get_dataset", lambda _: None)
    with pytest.raises(ValueError, match="not a registered dataset template"):
        jobs._resolve_source_template({"source_dataset_id": "chirps3_typooo"})


# -- the shipped derived templates ------------------------------------------------------------


def test_no_derived_builtin_declares_away_its_sources_obligations() -> None:
    """The normals and anomalies are derivative works of CHIRPS3 and ERA5-Land. Checked through
    the real predicate here, since propagation is what this module owns."""
    import glob
    import re

    from open_climate_service.shared.licences import refuses_publication

    licences = {}
    for path in sorted(glob.glob("open_climate_service/plugins/datasets/*.yaml")):
        for template in yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")):
            if isinstance(template, dict) and template.get("id"):
                licences[template["id"]] = parse_licence(template.get("license"))

    pairs = [
        (dataset_id, base)
        for dataset_id in sorted(licences)
        if (base := re.sub(r"_(relative_)?(normal|anomaly)_\d{4}_\d{4}$", "", dataset_id)) != dataset_id
        and base in licences
    ]
    assert len(pairs) >= 12, "the suffix pattern stopped matching, so this checked nothing"
    for derived_id, source_id in pairs:
        assert refuses_publication(licences[derived_id], [licences[source_id]]) is None, derived_id
