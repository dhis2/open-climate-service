"""Dataset licences: declare them and publish them honestly (CLIM-946).

OCS had no per-dataset licence concept — every collection published the constant `various`,
whether the data came from ERA5-Land or a non-commercial imagery release. That was tolerable
while every source was openly licensed. It stopped being tolerable when the August 2026 Nepal
flood produced the only usable imagery of the event under CC-BY-NC-4.0.

The decision recorded in CLIM-946 is to *allow* non-commercial sources, labelled, rather than
refuse them: declining the only imagery of a disaster is the opposite of the point. But
allowing them unlabelled is worse than either, because a derived product is a derivative work.
A water mask computed from CC-BY-NC imagery inherits the restriction, and publishing it as
`various` is licence laundering by accident — on the default path, with nothing to catch it.

## Scope

This module declares and publishes. Propagating a licence to a derived product — the
laundering half — is a separate change, because two licences must be *comparable* for that
and the comparison has to happen at exactly one point in the publish path. Splitting that
decision across the synthesise and reload paths produced four rounds of fixes that
contradicted each other, so the obligations recorded below are the input to that work rather
than the whole of it.

## Why this is not just a string field

A declaration is parsed into a `DatasetLicence` carrying one decisive fact: whether
commercial use is allowed. Three states, not two — `None` means "we do not know", which is
different from "yes" and must not be treated as it.

## SPDX where it exists, a name and URL where it does not

The three built-in sources cover both cases:

    CHIRPS3     CC-BY-4.0                            SPDX
    WorldPop    CC-BY-4.0                            SPDX
    ERA5-Land   Licence to Use Copernicus Products   bespoke, no SPDX identifier

Forcing the Copernicus licence into a near-miss SPDX identifier would be a false statement
about what a user may do, so the field accepts a name plus a URL as well.

CHIRPS3 is the reason to read the source page rather than its tone. CHC says CHIRPS3 "is in
the public domain" and that it has "waived all copyright and related or neighboring rights",
which reads as CC0 — but the same sentence names the instrument, "licensed under a Creative
Commons Attribution 4.0 International License". Declaring the public-domain half would drop
the attribution CC BY keeps, publishing the source as more permissive than its own terms:
the failure this module exists to prevent, committed on the first dataset it declares. Where
a page's wording and its named licence disagree, the named licence governs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# STAC 1.1 wants an SPDX identifier or the literal "other"; it deprecated "proprietary" and
# never defined "various", which is what OCS was publishing on every collection.
STAC_LICENSE_OTHER = "other"

# What a licence requires of anyone redistributing the data, or a work derived from it. A
# derived product may add obligations; it may never drop one.
ATTRIBUTION = "attribution"
SHARE_ALIKE = "share-alike"
NON_COMMERCIAL = "non-commercial"
# Not an obligation you can satisfy — a prohibition on distributing adapted material at all.
# Recording ND alongside attribution and non-commercial would let a derivative be published
# under the same ND licence, which the licence forbids outright.
NO_DERIVATIVES = "no-derivatives"

# SPDX identifiers whose terms have been read, with the obligations each imposes. Deliberately
# a small checked table rather than an attempt at the full SPDX register: an identifier absent
# here is not an SPDX identifier as far as OCS is concerned, which keeps an unvalidated string
# out of the published `license` field.
#
# Commercial use is one obligation among several, not the whole model. Comparing on it alone
# would let CC0 be derived from CC-BY-4.0 — both permit commercial use — silently dropping
# WorldPop's attribution requirement.
_SPDX_OBLIGATIONS: dict[str, frozenset[str]] = {
    "CC0-1.0": frozenset(),
    "PDDL-1.0": frozenset(),
    "MIT": frozenset({ATTRIBUTION}),
    "Apache-2.0": frozenset({ATTRIBUTION}),
    "CC-BY-3.0": frozenset({ATTRIBUTION}),
    "CC-BY-4.0": frozenset({ATTRIBUTION}),
    "OGL-UK-3.0": frozenset({ATTRIBUTION}),
    "ODbL-1.0": frozenset({ATTRIBUTION, SHARE_ALIKE}),
    "CC-BY-SA-3.0": frozenset({ATTRIBUTION, SHARE_ALIKE}),
    "CC-BY-SA-4.0": frozenset({ATTRIBUTION, SHARE_ALIKE}),
    "CC-BY-NC-3.0": frozenset({ATTRIBUTION, NON_COMMERCIAL}),
    "CC-BY-NC-4.0": frozenset({ATTRIBUTION, NON_COMMERCIAL}),
    "CC-BY-NC-SA-3.0": frozenset({ATTRIBUTION, SHARE_ALIKE, NON_COMMERCIAL}),
    "CC-BY-NC-SA-4.0": frozenset({ATTRIBUTION, SHARE_ALIKE, NON_COMMERCIAL}),
    "CC-BY-NC-ND-4.0": frozenset({ATTRIBUTION, NON_COMMERCIAL, NO_DERIVATIVES}),
    "CC-BY-ND-4.0": frozenset({ATTRIBUTION, NO_DERIVATIVES}),
}
# SPDX identifiers that are valid but whose propagation semantics have not been reviewed.
# Kept separate from `_SPDX_OBLIGATIONS` on purpose: a valid identifier belongs in the STAC
# `license` field even when OCS cannot yet reason about deriving from it. Downgrading
# BSD-3-Clause to a free-form name and publishing `other` threw away information the catalogue
# had — while `known=False` still makes propagation fail closed, which is the safe half.
_SPDX_UNREVIEWED = frozenset(
    {
        "BSD-2-Clause",
        "BSD-3-Clause",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-3.0-only",
        "MPL-2.0",
        "Unlicense",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "OGL-Canada-2.0",
    }
)
_SPDX_BY_LOWER = {identifier.lower(): identifier for identifier in (*_SPDX_OBLIGATIONS, *_SPDX_UNREVIEWED)}

# Licences with no SPDX identifier, whose terms have been read. Without this a template author
# has to restate the obligations by hand for every one, and an assertion repeated in five
# templates is one that will eventually be wrong in a sixth.
#
# Note what this is NOT: a default. An unrecognised licence resolves to "obligations unknown",
# never to "no obligations". Defaulting to permissive would mean an author who forgets to
# describe a restrictive source publishes it as freely reusable, which is the laundering this
# module exists to prevent.
_KNOWN_NAMED_LICENCES: dict[str, frozenset[str]] = {
    # "free of charge, worldwide, non-exclusive, royalty free and perpetual", for "any purpose
    # in so far as it is lawful", with clear attribution to Copernicus required.
    "licence to use copernicus products": frozenset({ATTRIBUTION}),
    # Free and open including commercial reuse, subject to the Notice's conditions.
    "copernicus sentinel data legal notice": frozenset({ATTRIBUTION}),
    # NASA Earth science data carries no copyright and no use restrictions.
    "nasa earth science data": frozenset(),
}


@dataclass(frozen=True)
class DatasetLicence:
    """A licence declaration, parsed into something two datasets can be compared on."""

    identifier: str | None
    """Canonical SPDX identifier, or None when the licence is named rather than identified."""

    name: str | None
    url: str | None

    obligations: frozenset[str]
    """What the licence requires. Meaningless unless `known` is true."""

    known: bool
    """Whether the obligations were actually determined, rather than defaulted."""

    @property
    def stac_license(self) -> str:
        """The value for a STAC collection's `license` field.

        A canonical SPDX identifier when there is one, otherwise the literal `other`, which
        STAC 1.1 defines for exactly this case. Never a free-form string: `license` is a
        constrained field, and emitting an unvalidated vendor name there produces a collection
        that does not validate. Never `various` either, which is not a STAC value at all and
        reads as "no restrictions worth mentioning".
        """
        return self.identifier or STAC_LICENSE_OTHER

    @property
    def label(self) -> str:
        return self.identifier or self.name or "not declared"

    @property
    def commercial_use(self) -> bool | None:
        """Whether commercial use is permitted, or None when the licence is not understood."""
        if not self.known:
            return None
        return NON_COMMERCIAL not in self.obligations


UNDECLARED = DatasetLicence(identifier=None, name=None, url=None, obligations=frozenset(), known=False)
"""A template with no `license`, or one that could not be parsed. Published as `other`."""


_NAME_VERSION_SUFFIX = re.compile(r"\s+v?\d+(\.\d+)*$")
"""A trailing version on an otherwise exact name — "… Products v1.2", "… Notice 1.0".

Only a version. This used to be a `startswith` test so that qualifiers came along for free,
which meant any suffix inherited the base licence's terms: "NASA Earth Science Data -
Non-Commercial Terms" matched the NASA entry and was classified as unrestricted commercial
use. A version number cannot change what a licence requires; an arbitrary suffix is exactly
how it gets said that it does.
"""


def _obligations_for_name(name: str | None) -> frozenset[str] | None:
    """Obligations for a licence known by name rather than SPDX identifier.

    Matched exactly, after normalising whitespace and case and dropping a trailing version, so
    a name this module has not had its terms read stays unknown rather than borrowing another
    licence's semantics.
    """
    if not name:
        return None
    normalised = " ".join(name.lower().split())
    for candidate in (normalised, _NAME_VERSION_SUFFIX.sub("", normalised)):
        obligations = _KNOWN_NAMED_LICENCES.get(candidate)
        if obligations is not None:
            return obligations
    return None


def _canonical_spdx(value: str) -> str | None:
    """The canonical SPDX identifier for `value`, or None if it is not one we recognise.

    Case-insensitive, because `cc-by-4.0` is an easy thing to write in YAML and rejecting it
    outright would be unhelpful. An unrecognised string is not treated as SPDX at all: it
    becomes a licence *name*, so the collection publishes `other` rather than an identifier no
    STAC client can resolve.
    """
    return _SPDX_BY_LOWER.get(value.strip().lower())


def _resolve_identifier(declared: dict[str, Any]) -> tuple[str | None, str | None]:
    """The SPDX identifier a mapping declares, and why it cannot be resolved, if so.

    `id` and `spdx` are accepted as aliases for one field. When both are given and disagree
    there is no basis for preferring either, and `declared.get("id") or declared.get("spdx")`
    picked the first silently: `{id: CC0-1.0, spdx: CC-BY-NC-4.0}` published as CC0 with
    commercial use allowed.

    Compared on the canonical form, so `CC0-1.0` and `cc0-1.0` are one identifier rather than
    a conflict. Two unrecognised strings are compared case-folded; they resolve to no
    identifier either way, but saying which two disagreed is more useful than silence.
    """
    given = [
        (key, str(raw).strip()) for key in ("id", "spdx") if isinstance(raw := declared.get(key), str) and raw.strip()
    ]
    if not given:
        return None, None
    if len(given) == 2:
        (first_key, first), (second_key, second) = given
        if (_canonical_spdx(first) or first.lower()) != (_canonical_spdx(second) or second.lower()):
            return None, (
                f"declares conflicting licence identifiers: {first_key} {first!r} and "
                f"{second_key} {second!r}. They are aliases for one field, so there is no basis "
                f"for choosing between them; the licence is treated as undeclared. Give only one."
            )
    return _canonical_spdx(given[0][1]), None


def _contradiction(identifier: str | None, name: str | None) -> str | None:
    """Why an identifier and a name cannot both describe this licence, or None.

    An identifier and a name that require different things is a contradiction, not a
    redundancy. `{id: CC0-1.0, name: <a licence requiring attribution>, url: ...}` would
    publish `license: CC0-1.0` while the `rel: license` link points at terms that demand
    credit, and a STAC client reads the field far more often than it fetches the link. There
    is no basis for picking a winner, so neither is used.

    Only a *recognised* name can be shown to disagree. An unrecognised one is left alone: it
    is a label and a link, the identifier stays authoritative for the terms, and the common
    `{id, name, url}` spelling of a single licence has to keep working.
    """
    if identifier is None or identifier not in _SPDX_OBLIGATIONS:
        return None
    named = _obligations_for_name(name)
    if named is None or named == _SPDX_OBLIGATIONS[identifier]:
        return None
    return (
        f"declares SPDX identifier {identifier} together with the name {name!r}, which OCS "
        f"knows to require {sorted(named) or 'nothing'} rather than "
        f"{sorted(_SPDX_OBLIGATIONS[identifier]) or 'nothing'}; treating the licence as "
        f"undeclared. Declare whichever one describes the terms, not both."
    )


def _declared_semantics(identifier: str | None, name: str | None) -> tuple[frozenset[str], str] | None:
    """The obligations this declaration resolves to, and what supplied them.

    None when neither the identifier nor the name is one OCS knows, which is the only case
    where an explicit `obligations` list is consulted.

    Single owner of that precedence rule. `parse_licence` and `licence_declaration_problem`
    both need to know whether the list was used or ignored, and when each decided it
    separately they disagreed: a contradictory list beside a recognised *name* was silently
    discarded while the same list beside an SPDX identifier was reported.
    """
    if identifier is not None and identifier in _SPDX_OBLIGATIONS:
        return _SPDX_OBLIGATIONS[identifier], f"the SPDX identifier {identifier}"
    named = _obligations_for_name(name)
    if named is not None:
        return named, f"the recognised licence name {name!r}"
    return None


def licence_declaration_problem(declared: Any) -> str | None:
    """A specific complaint about a `license` declaration, for the template validator.

    Kept out of `parse_licence` on purpose. That function runs on every STAC and `/datasets`
    request, and an instance's templates are deliberately re-read each time, so a warning
    raised there would log on every request for as long as the template existed — the exact
    behaviour `_warn_once` was added to stop (CLIM-904). Parsing stays silent and pure; this
    is called once, at the validation boundary, and routed through that deduplicating logger.

    Returns None when there is nothing specific to say. A declaration that is merely
    unreadable is already reported by the validator's generic message.
    """
    if not isinstance(declared, dict):
        return None
    identifier, conflict = _resolve_identifier(declared)
    if conflict is not None:
        return conflict
    raw_name = declared.get("name")
    name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else None

    contradiction = _contradiction(identifier, name)
    if contradiction is not None:
        return contradiction

    semantics = _declared_semantics(identifier, name)
    if semantics is not None and isinstance(declared.get("obligations"), list):
        known_obligations, supplier = semantics
        declared_obligations = sorted(str(o).strip().lower() for o in declared["obligations"] if str(o).strip())
        return (
            f"declares explicit 'obligations' {declared_obligations} alongside {supplier}, which "
            f"OCS knows to require {sorted(known_obligations) or 'nothing'}; the list is ignored. "
            f"Remove it, or give the licence a name of its own if its terms genuinely differ from "
            f"the licence it is named after."
        )
    return None


def parse_licence(declared: Any) -> DatasetLicence:
    """Parse a template's `license` field.

    Accepts either form, because the sources demand both:

        license: CC-BY-4.0
        license:
          name: Licence to Use Copernicus Products
          url: https://apps.ecmwf.int/datasets/licences/copernicus/

    A string that is not a recognised SPDX identifier is kept as a *name*, not passed through
    to STAC as though it were one. Anything unparseable resolves to `UNDECLARED` rather than
    raising: a malformed licence should degrade to "unknown, published as other", not take down
    the catalogue, and the template validator warns about it separately.
    """
    if isinstance(declared, str):
        text = declared.strip()
        if not text:
            return UNDECLARED
        spdx = _canonical_spdx(text)
        if spdx is not None:
            return DatasetLicence(
                identifier=spdx,
                name=None,
                url=None,
                obligations=_SPDX_OBLIGATIONS.get(spdx, frozenset()),
                # A valid-but-unreviewed identifier is still published as SPDX, but propagation
                # must fail closed until someone records what it requires.
                known=spdx in _SPDX_OBLIGATIONS,
            )
        named = _obligations_for_name(text)
        return DatasetLicence(
            identifier=None,
            name=text,
            url=None,
            obligations=named if named is not None else frozenset(),
            known=named is not None,
        )

    if isinstance(declared, dict):
        identifier, conflict = _resolve_identifier(declared)
        if conflict is not None:
            return UNDECLARED
        raw_name = declared.get("name")
        name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else None
        raw_url = declared.get("url")
        url = str(raw_url).strip() if isinstance(raw_url, str) and raw_url.strip() else None

        if identifier is None and name is None:
            return UNDECLARED

        # An identifier and a name that require different things is a contradiction, not a
        # redundancy. `{id: CC0-1.0, name: <a licence requiring attribution>, url: ...}` would
        # publish `license: CC0-1.0` while the `rel: license` link points at terms that demand
        # credit, and a STAC client reads the field far more often than it fetches the link.
        # There is no basis for picking a winner, so neither is used.
        #
        # Only a *recognised* name can be shown to disagree. An unrecognised one is left alone:
        # it is a label and a link, the identifier stays authoritative for the terms, and the
        # common `{id, name, url}` spelling of a single licence has to keep working.
        if _contradiction(identifier, name) is not None:
            return UNDECLARED

        # Order matters. A recognised identifier or name is authoritative, and an explicit
        # `obligations` list is consulted only when neither supplies semantics. Letting the
        # list win would make `{id: CC-BY-NC-4.0, obligations: []}` publish as CC-BY-NC while
        # every compatibility check saw no restrictions — laundering by configuration, which is
        # worse than the silence this module replaced.
        raw_obligations = declared.get("obligations")
        semantics = _declared_semantics(identifier, name)
        if semantics is not None:
            obligations, known = semantics[0], True
        elif isinstance(raw_obligations, list):
            obligations = frozenset(str(o).strip().lower() for o in raw_obligations if str(o).strip())
            known = True
        else:
            obligations, known = frozenset(), False

        return DatasetLicence(identifier=identifier, name=name, url=url, obligations=obligations, known=known)

    return UNDECLARED
