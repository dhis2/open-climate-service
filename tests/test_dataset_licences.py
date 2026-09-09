"""Dataset licences: declaration and publication (CLIM-946).

The failure this guards against is silent. OCS published the constant `various` on every
collection, so a CC-BY-NC source was published as though it carried no restriction.

Propagation to derived products is deliberately not here. It is a separate change with a
single enforcement point, because splitting the decision across the synthesise and reload
paths produced four rounds of contradictory fixes.
"""

import glob
import itertools
import logging
import pathlib
import re

import pytest
import yaml

from open_climate_service.shared.licences import (
    ATTRIBUTION,
    NON_COMMERCIAL,
    STAC_LICENSE_OTHER,
    UNDECLARED,
    parse_licence,
)

_BUILTIN_TEMPLATES = sorted(glob.glob("open_climate_service/plugins/datasets/*.yaml"))


# -- parsing ----------------------------------------------------------------------------


def test_spdx_identifier_is_recognised() -> None:
    licence = parse_licence("CC-BY-4.0")
    assert licence.identifier == "CC-BY-4.0"
    assert licence.obligations == frozenset({ATTRIBUTION})
    assert licence.commercial_use is True
    assert licence.stac_license == "CC-BY-4.0"


def test_spdx_matching_is_case_insensitive_but_output_is_canonical() -> None:
    """`cc-by-4.0` is easy to write in YAML; the published identifier must still be canonical,
    because `license` is a constrained STAC field."""
    assert parse_licence("cc-by-4.0").stac_license == "CC-BY-4.0"


def test_an_unrecognised_string_is_a_name_not_an_spdx_identifier() -> None:
    """Emitting a vendor string verbatim would produce a collection that does not validate.

    It is kept as a name — so nothing is lost — and published as `other`.
    """
    licence = parse_licence("SomeVendorLicence-2.0")
    assert licence.identifier is None
    assert licence.name == "SomeVendorLicence-2.0"
    assert licence.stac_license == STAC_LICENSE_OTHER
    assert licence.known is False


@pytest.mark.parametrize("identifier", ["CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-4.0", "cc-by-nc-3.0"])
def test_every_non_commercial_cc_licence_is_caught(identifier: str) -> None:
    assert NON_COMMERCIAL in parse_licence(identifier).obligations
    assert parse_licence(identifier).commercial_use is False


def test_a_named_licence_without_spdx_is_accepted() -> None:
    """The Copernicus licence has no SPDX identifier. Forcing it into a near-miss one would be
    a false statement about what a user may do."""
    licence = parse_licence(
        {
            "name": "Licence to Use Copernicus Products",
            "url": "https://apps.ecmwf.int/datasets/licences/copernicus/",
            "commercial_use": True,
        }
    )
    assert licence.identifier is None
    assert licence.name == "Licence to Use Copernicus Products"
    assert licence.commercial_use is True
    assert licence.obligations == frozenset({ATTRIBUTION})
    # STAC 1.1 has a defined value for "not SPDX", and it is not `various`.
    assert licence.stac_license == STAC_LICENSE_OTHER


def test_an_unknown_identifier_is_unknown_not_permissive() -> None:
    """The safe answer for a licence nobody has checked is "we do not know"."""
    assert parse_licence("SomeVendorLicence-2.0").commercial_use is None


def test_explicit_obligations_describe_a_licence_in_neither_table() -> None:
    licence = parse_licence(
        {"name": "Vendor Terms", "url": "https://x", "obligations": ["attribution", "non-commercial"]}
    )
    assert licence.known is True
    assert licence.commercial_use is False


@pytest.mark.parametrize("bad", [None, "", "   ", 42, [], {}, {"url": "https://x"}])
def test_unusable_declarations_degrade_to_undeclared(bad: object) -> None:
    """A malformed licence must not take down the catalogue; the validator warns separately."""
    assert parse_licence(bad) is UNDECLARED


def test_undeclared_publishes_as_other() -> None:
    assert UNDECLARED.stac_license == STAC_LICENSE_OTHER
    assert UNDECLARED.commercial_use is None
    assert UNDECLARED.known is False


# -- every built-in declares one ------------------------------------------------------------


@pytest.mark.parametrize("path", _BUILTIN_TEMPLATES, ids=lambda p: pathlib.Path(p).name)
def test_every_builtin_template_declares_a_licence(path: str) -> None:
    """No built-in may rely on the default. A user seeing a layer should be able to see what
    they may do with it, and `other` is what OCS says when nobody has said anything."""
    undeclared = [
        t.get("id")
        for t in yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
        if isinstance(t, dict) and t.get("license") is None
    ]
    assert not undeclared, f"templates without a licence: {undeclared}"


@pytest.mark.parametrize("path", _BUILTIN_TEMPLATES, ids=lambda p: pathlib.Path(p).name)
def test_every_builtin_licence_parses(path: str) -> None:
    """A declared licence that does not parse is worse than none: it looks handled."""
    for template in yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")):
        if not isinstance(template, dict):
            continue
        assert parse_licence(template.get("license")) is not UNDECLARED, (
            f"{template.get('id')} declares an unparseable licence {template.get('license')!r}"
        )


# The obligations each source's own terms impose, read off the source's page and recorded here
# rather than taken from the template — a test that reads the declaration it is checking
# asserts nothing. CHIRPS3 is why this table exists: CHC's page says CHIRPS3 "is in the public
# domain" and that CHC "waived all copyright and related or neighboring rights", wording that
# reads as CC0, while the same sentence names the instrument as CC BY 4.0. The first
# declaration shipped CC0-1.0, which drops attribution from CHIRPS3 and everything derived
# from it. Nothing in the checks above catches that: CC0 parses, and every derived product
# dropped attribution consistently.
_REVIEWED_SOURCE_TERMS: dict[str, frozenset[str]] = {
    "chirps3.yaml": frozenset({ATTRIBUTION}),  # CC BY 4.0
    "era5_land.yaml": frozenset({ATTRIBUTION}),  # Licence to Use Copernicus Products
    "worldpop.yaml": frozenset({ATTRIBUTION}),  # CC BY 4.0
}


def test_every_shipped_template_has_reviewed_source_terms() -> None:
    """Guards the guard: a new or renamed template file would otherwise go unchecked, which is
    the point at which its licence has not been read by anyone."""
    assert {pathlib.Path(path).name for path in _BUILTIN_TEMPLATES} == set(_REVIEWED_SOURCE_TERMS)


@pytest.mark.parametrize("path", _BUILTIN_TEMPLATES, ids=lambda p: pathlib.Path(p).name)
def test_no_builtin_declares_away_an_obligation_its_source_imposes(path: str) -> None:
    required = _REVIEWED_SOURCE_TERMS[pathlib.Path(path).name]
    for template in yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")):
        if not isinstance(template, dict) or not template.get("id"):
            continue
        obligations = parse_licence(template.get("license")).obligations
        assert required <= obligations, (
            f"{template['id']} declares {template.get('license')!r}, dropping "
            f"{sorted(required - obligations)} required by its source"
        )


def _builtin_licences() -> dict:
    licences = {}
    for path in _BUILTIN_TEMPLATES:
        for template in yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")):
            if isinstance(template, dict) and template.get("id"):
                licences[template["id"]] = parse_licence(template.get("license"))
    return licences


def _derived_source_pairs() -> list[tuple[str, str]]:
    """Every derived built-in paired with the dataset it is computed from.

    Derived from the ids rather than listed by hand: an earlier version enumerated four pairs
    and left twelve unchecked, so a licence change in any of those twelve could have loosened
    silently. Suffix-stripping means a new normal or anomaly is covered the day it is added.
    """
    ids = set(_builtin_licences())
    pairs = []
    for dataset_id in sorted(ids):
        base = re.sub(r"_(relative_)?(normal|anomaly)_\d{4}_\d{4}$", "", dataset_id)
        if base != dataset_id and base in ids:
            pairs.append((dataset_id, base))
    return pairs


def test_the_derived_pairs_are_actually_found() -> None:
    """Guards the guard: a suffix change would silently empty the parametrisation below and
    the whole check would pass by testing nothing."""
    assert len(_derived_source_pairs()) >= 12


@pytest.mark.parametrize(("derived_id", "source_id"), _derived_source_pairs(), ids=lambda v: v)
def test_no_derived_builtin_declares_away_its_sources_obligations(derived_id: str, source_id: str) -> None:
    """The normals and anomalies are derivative works of CHIRPS3 and ERA5-Land.

    Stated as a set comparison rather than by calling the production predicate: a test that
    asks the code under test whether the code under test is right passes when both are wrong.

    They carry no `source_url`, so a first pass that anchored insertion on that field skipped
    every one of them.
    """
    licences = _builtin_licences()
    derived, source = licences[derived_id], licences[source_id]
    assert derived.known and source.known, "a shipped licence OCS cannot parse"
    assert source.obligations <= derived.obligations, (
        f"{derived_id} drops {sorted(source.obligations - derived.obligations)} required by {source_id}"
    )


# -- the SPDX table is authoritative ------------------------------------------------------


def test_explicit_obligations_cannot_override_a_recognised_spdx_identifier() -> None:
    """`{id: CC-BY-NC-4.0, obligations: []}` would otherwise publish as CC-BY-NC while every
    compatibility check saw no restrictions — laundering by configuration."""
    licence = parse_licence({"id": "CC-BY-NC-4.0", "obligations": []})
    assert licence.stac_license == "CC-BY-NC-4.0"
    assert licence.commercial_use is False
    assert NON_COMMERCIAL in licence.obligations


def test_explicit_obligations_cannot_override_a_recognised_name() -> None:
    licence = parse_licence({"name": "Licence to Use Copernicus Products", "url": "https://x", "obligations": []})
    assert licence.obligations == frozenset({ATTRIBUTION})


def test_explicit_obligations_are_used_when_nothing_else_supplies_them() -> None:
    """The legitimate case: a licence in neither table, whose terms someone has read."""
    licence = parse_licence({"name": "Vendor Terms", "url": "https://x", "obligations": ["attribution"]})
    assert licence.known is True
    assert licence.obligations == frozenset({ATTRIBUTION})


# -- valid SPDX identifiers OCS has not reviewed -------------------------------------------


def test_a_valid_spdx_identifier_is_preserved_even_when_unreviewed() -> None:
    """Downgrading BSD-3-Clause to a free-form name and publishing `other` threw away
    information the catalogue already had. The identifier is kept; only the propagation
    semantics are marked unknown."""
    licence = parse_licence("BSD-3-Clause")
    assert licence.stac_license == "BSD-3-Clause"
    assert licence.known is False


# -- a name must be the licence, not merely start like it ----------------------------------


def test_a_qualified_name_does_not_inherit_a_known_licences_terms() -> None:
    """`startswith` matching classified this as NASA's unrestricted terms, so a
    non-commercial variant read as free for commercial use — the exact laundering the
    unknown-is-not-permissive rule exists to prevent."""
    licence = parse_licence("NASA Earth Science Data - Non-Commercial Terms")
    assert licence.known is False
    assert licence.commercial_use is None


def test_an_exact_known_name_still_resolves() -> None:
    assert parse_licence("NASA Earth Science Data").obligations == frozenset()
    assert parse_licence("Licence to Use Copernicus Products").obligations == frozenset({ATTRIBUTION})


@pytest.mark.parametrize("name", ["Licence to Use Copernicus Products v1.2", "Licence to Use Copernicus Products 2"])
def test_a_trailing_version_is_still_the_same_licence(name: str) -> None:
    """A version number cannot change what a licence requires, so these stay recognised."""
    assert parse_licence(name).obligations == frozenset({ATTRIBUTION})


# -- an identifier and a name that disagree -------------------------------------------------


def test_an_identifier_contradicted_by_a_known_name_is_undeclared() -> None:
    """`license` is read far more often than the licence link is fetched, so publishing
    `CC0-1.0` beside a link to terms requiring attribution is worse than publishing nothing.
    There is no basis for choosing between them."""
    licence = parse_licence({"id": "CC0-1.0", "name": "Licence to Use Copernicus Products", "url": "https://x"})
    assert licence is UNDECLARED


def test_an_identifier_with_a_consistent_name_and_url_is_kept() -> None:
    """The ordinary spelling of one licence in all three fields has to keep working — the URL
    is what the `rel: license` link needs."""
    licence = parse_licence(
        {
            "id": "CC-BY-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        }
    )
    assert licence.stac_license == "CC-BY-4.0"
    assert licence.url == "https://creativecommons.org/licenses/by/4.0/"


# -- parsing is silent; the validator does the complaining ---------------------------------


def test_parsing_a_contradictory_licence_logs_nothing() -> None:
    """`parse_licence` runs on every STAC and /datasets request, and instance templates are
    re-read each time, so a warning here would log for as long as the template existed — the
    behaviour `_warn_once` exists to stop (CLIM-904). Asserted by call count, because a
    "warns once" claim about a per-request function is only true until the second request."""
    from open_climate_service.shared import licences

    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collect()
    licences.logger.addHandler(handler)
    try:
        for _ in range(3):
            parse_licence({"id": "CC0-1.0", "name": "Licence to Use Copernicus Products", "url": "https://x"})
            parse_licence({"id": "CC-BY-NC-4.0", "obligations": []})
    finally:
        licences.logger.removeHandler(handler)

    assert records == []


def test_the_validator_names_the_contradiction_rather_than_calling_it_unreadable() -> None:
    """ "Unreadable" sends the author hunting for a typo when the declaration parses fine and
    simply says two different things."""
    from open_climate_service.shared.licences import licence_declaration_problem

    problem = licence_declaration_problem(
        {"id": "CC0-1.0", "name": "Licence to Use Copernicus Products", "url": "https://x"}
    )
    assert problem is not None
    assert "CC0-1.0" in problem
    assert "attribution" in problem


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param({"id": "CC-BY-NC-4.0", "obligations": []}, id="spdx-identifier"),
        pytest.param(
            {"name": "Licence to Use Copernicus Products", "url": "https://x", "obligations": ["non-commercial"]},
            id="recognised-name",
        ),
    ],
)
def test_ignored_obligations_are_reported_whatever_supplied_the_terms(declared: dict) -> None:
    """Parametrised over both, because the two were checked in separate places and drifted.

    `parse_licence` ignores an explicit list whenever the identifier *or* the name is one OCS
    knows, but the validator reported only the identifier case. So Copernicus plus
    `obligations: [non-commercial]` was read as commercially usable and said nothing — the
    operator asserted a restriction and OCS silently dropped it, which is the laundering
    direction.
    """
    from open_climate_service.shared.licences import licence_declaration_problem

    problem = licence_declaration_problem(declared)
    assert problem is not None
    assert "ignored" in problem


def test_the_report_names_both_what_was_asked_for_and_what_applies() -> None:
    """An operator who wrote non-commercial needs to see that attribution is what took effect,
    not just that something was ignored."""
    from open_climate_service.shared.licences import licence_declaration_problem

    problem = licence_declaration_problem(
        {"name": "Licence to Use Copernicus Products", "url": "https://x", "obligations": ["non-commercial"]}
    )
    assert problem is not None
    assert "non-commercial" in problem
    assert "attribution" in problem


def test_a_recognised_name_still_outranks_a_contradictory_obligations_list() -> None:
    """The behaviour is unchanged — the researched terms win. Only the silence is fixed."""
    licence = parse_licence(
        {"name": "Licence to Use Copernicus Products", "url": "https://x", "obligations": ["non-commercial"]}
    )
    assert licence.obligations == frozenset({ATTRIBUTION})
    assert licence.commercial_use is True


def test_a_sound_declaration_has_nothing_to_report() -> None:
    from open_climate_service.shared.licences import licence_declaration_problem

    assert licence_declaration_problem("CC-BY-4.0") is None
    assert licence_declaration_problem({"id": "CC-BY-4.0", "url": "https://x"}) is None
    assert licence_declaration_problem({"name": "Vendor Terms", "url": "https://x"}) is None


def test_conflicting_identifier_aliases_are_named_not_merely_undeclared() -> None:
    """`id` and `spdx` are aliases, and `declared.get("id") or declared.get("spdx")` picked the
    first silently: `{id: CC0-1.0, spdx: CC-BY-NC-4.0}` published as CC0 with commercial use
    allowed.

    Asserted on the *report*, not on the parse result. Resolving a conflict to "no identifier"
    already reaches UNDECLARED through the no-licence-identity path, so a test that only checked
    the parse outcome passes with the fix removed — which the matrix row above does, and which
    is why this exists separately.
    """
    from open_climate_service.shared.licences import licence_declaration_problem

    problem = licence_declaration_problem({"id": "CC0-1.0", "spdx": "CC-BY-NC-4.0"})
    assert problem is not None
    assert "CC0-1.0" in problem
    assert "CC-BY-NC-4.0" in problem
    assert "aliases" in problem


def test_the_same_licence_under_both_aliases_is_not_a_conflict() -> None:
    """Case and spelling differences resolve to one canonical identifier."""
    assert parse_licence({"id": "CC-BY-4.0", "spdx": "cc-by-4.0"}).stac_license == "CC-BY-4.0"


# -- the declaration surface -----------------------------------------------------------------
#
# Four review rounds each found one more way for two parts of a `license` mapping to disagree
# while OCS silently picked one: a prefix-matched name, an identifier contradicted by a name,
# an obligations list beside a recognised name, and an identifier beside a conflicting alias.
# One defect found four times. Finding them individually does not terminate, so the surface is
# enumerated instead.
#
# The dimensions are semantic, not raw keys. What decides the outcome is the *kind* of value in
# each of three positions; `url` is carried through and never affects it, which
# `test_the_url_never_changes_the_outcome` pins rather than assumes.

_IDENTIFIER_CASES: dict[str, dict] = {
    "absent": {},
    "recognised": {"id": "CC-BY-4.0"},
    "unrecognised": {"id": "NotAnSpdx"},
    "aliases-agree": {"id": "CC-BY-4.0", "spdx": "cc-by-4.0"},
    "aliases-conflict": {"id": "CC0-1.0", "spdx": "CC-BY-NC-4.0"},
}
# "agrees"/"conflicts" are relative to CC-BY-4.0's {attribution}: the Copernicus licence
# requires the same, NASA's requires nothing.
_NAME_CASES: dict[str, dict] = {
    "absent": {},
    "known-agrees": {"name": "Licence to Use Copernicus Products"},
    "known-conflicts": {"name": "NASA Earth Science Data"},
    "unknown": {"name": "Vendor Terms"},
}
_OBLIGATION_CASES: dict[str, dict] = {"absent": {}, "present": {"obligations": ["non-commercial"]}}

_ACCEPTED = "accepted"  # a usable licence, nothing to report
_IGNORED = "ignored"  # usable, but a field was discarded and must be reported
_UNDECLARED = "undeclared"  # not usable; publishes as `other`

# One entry per (identifier, name, obligations) shape. A shape with no entry fails the
# completeness guard below, so a new key or value kind cannot be added without deciding what
# every combination of it does.
_EXPECTED: dict[tuple[str, str, str], str] = {
    # No identifier: the name governs, and an explicit list is used only when nothing else
    # supplies the terms.
    ("absent", "absent", "absent"): _UNDECLARED,
    ("absent", "absent", "present"): _UNDECLARED,  # obligations without a licence identity
    ("absent", "known-agrees", "absent"): _ACCEPTED,
    ("absent", "known-agrees", "present"): _IGNORED,
    ("absent", "known-conflicts", "absent"): _ACCEPTED,
    ("absent", "known-conflicts", "present"): _IGNORED,
    ("absent", "unknown", "absent"): _ACCEPTED,
    ("absent", "unknown", "present"): _ACCEPTED,  # the legitimate use of an explicit list
    # A recognised identifier is authoritative, and a known name that disagrees with it is a
    # contradiction rather than a redundancy.
    ("recognised", "absent", "absent"): _ACCEPTED,
    ("recognised", "absent", "present"): _IGNORED,
    ("recognised", "known-agrees", "absent"): _ACCEPTED,
    ("recognised", "known-agrees", "present"): _IGNORED,
    ("recognised", "known-conflicts", "absent"): _UNDECLARED,
    ("recognised", "known-conflicts", "present"): _UNDECLARED,
    ("recognised", "unknown", "absent"): _ACCEPTED,
    ("recognised", "unknown", "present"): _IGNORED,
    # An unrecognised identifier supplies nothing, so the name decides. The string itself is
    # dropped rather than kept as a name, unlike the plain-string form — a known asymmetry, and
    # the safe direction, since the result is `other` rather than a guess.
    ("unrecognised", "absent", "absent"): _UNDECLARED,
    ("unrecognised", "absent", "present"): _UNDECLARED,
    ("unrecognised", "known-agrees", "absent"): _ACCEPTED,
    ("unrecognised", "known-agrees", "present"): _IGNORED,
    ("unrecognised", "known-conflicts", "absent"): _ACCEPTED,
    ("unrecognised", "known-conflicts", "present"): _IGNORED,
    ("unrecognised", "unknown", "absent"): _ACCEPTED,
    ("unrecognised", "unknown", "present"): _ACCEPTED,
    # Both aliases naming one licence behaves exactly as one identifier.
    ("aliases-agree", "absent", "absent"): _ACCEPTED,
    ("aliases-agree", "absent", "present"): _IGNORED,
    ("aliases-agree", "known-agrees", "absent"): _ACCEPTED,
    ("aliases-agree", "known-agrees", "present"): _IGNORED,
    ("aliases-agree", "known-conflicts", "absent"): _UNDECLARED,
    ("aliases-agree", "known-conflicts", "present"): _UNDECLARED,
    ("aliases-agree", "unknown", "absent"): _ACCEPTED,
    ("aliases-agree", "unknown", "present"): _IGNORED,
    # Aliases naming different licences: undeclared whatever else is present, because there is
    # no basis for choosing and the wrong choice publishes NC data as permissive.
    ("aliases-conflict", "absent", "absent"): _UNDECLARED,
    ("aliases-conflict", "absent", "present"): _UNDECLARED,
    ("aliases-conflict", "known-agrees", "absent"): _UNDECLARED,
    ("aliases-conflict", "known-agrees", "present"): _UNDECLARED,
    ("aliases-conflict", "known-conflicts", "absent"): _UNDECLARED,
    ("aliases-conflict", "known-conflicts", "present"): _UNDECLARED,
    ("aliases-conflict", "unknown", "absent"): _UNDECLARED,
    ("aliases-conflict", "unknown", "present"): _UNDECLARED,
}

_SHAPES = list(itertools.product(_IDENTIFIER_CASES, _NAME_CASES, _OBLIGATION_CASES))


def _declaration(identifier: str, name: str, obligations: str, url: bool = True) -> dict:
    return {
        **_IDENTIFIER_CASES[identifier],
        **_NAME_CASES[name],
        **_OBLIGATION_CASES[obligations],
        **({"url": "https://x"} if url else {}),
    }


def _outcome(declaration: dict) -> str:
    from open_climate_service.shared.licences import licence_declaration_problem

    if parse_licence(declaration) is UNDECLARED:
        return _UNDECLARED
    return _IGNORED if licence_declaration_problem(declaration) else _ACCEPTED


@pytest.mark.parametrize(("identifier", "name", "obligations"), _SHAPES, ids=lambda v: v)
def test_every_declaration_shape_has_the_decided_outcome(identifier: str, name: str, obligations: str) -> None:
    expected = _EXPECTED.get((identifier, name, obligations))
    assert expected is not None, (
        f"undecided declaration shape: identifier={identifier}, name={name}, obligations={obligations}. "
        "Decide what it should do and add it to _EXPECTED."
    )
    assert _outcome(_declaration(identifier, name, obligations)) == expected


def test_the_expectation_table_matches_the_shapes_that_exist() -> None:
    """Both directions. A missing entry leaves a combination undecided; a stale one describes a
    shape that can no longer occur, which reads as coverage that is not there."""
    assert set(_EXPECTED) == set(_SHAPES)


def test_the_url_never_changes_the_outcome() -> None:
    """`url` is carried into the licence link and has no bearing on the terms. Pinned because
    the enumeration above omits it as a dimension, and that omission is only sound while this
    holds."""
    for identifier, name, obligations in _SHAPES:
        with_url = _outcome(_declaration(identifier, name, obligations, url=True))
        without = _outcome(_declaration(identifier, name, obligations, url=False))
        assert with_url == without, f"url changed the outcome for {identifier}/{name}/{obligations}"


def test_the_enumerated_keys_are_the_keys_the_parser_accepts() -> None:
    """Guards the guard. A new key in `parse_licence` with no dimension here would leave its
    interactions untested while the enumeration still claims to be complete."""
    accepted_keys = {"id", "spdx", "name", "url", "obligations"}
    dimensions = (*_IDENTIFIER_CASES.values(), *_NAME_CASES.values(), *_OBLIGATION_CASES.values())
    covered = {key for case in dimensions for key in case}
    assert covered | {"url"} == accepted_keys, f"keys with no dimension: {accepted_keys - covered - {'url'}}"
