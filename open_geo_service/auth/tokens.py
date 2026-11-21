"""Named API tokens from the ``authentication`` block, and the principal a request runs as."""

from __future__ import annotations

import base64
import binascii
import hmac
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from open_geo_service import config as api_config

MIN_SECRET_LENGTH = 32


class Tier(IntEnum):
    """What a principal may do; a higher tier includes every lower one."""

    PUBLIC = 0
    AUTHENTICATED = 1
    OPERATOR = 2


@dataclass(frozen=True)
class NamedToken:
    """One machine client's token, with the secret read from its environment variable."""

    id: str
    tier: Tier
    secret: str = field(repr=False)


@dataclass(frozen=True)
class Principal:
    """Who a request runs as, and the tier it was granted."""

    id: str
    tier: Tier
    token: NamedToken | None = None

    @property
    def owner_id(self) -> str | None:
        """The owner recorded on what this principal creates.

        None while authentication is off: `OPEN` shares its id with `ANONYMOUS`, so a record it
        owned would become readable without a token once authentication is turned on. An
        ownerless record is reached only by operators.
        """
        return None if self is OPEN else self.id

    def may_access(self, owner: str | None) -> bool:
        """Whether this principal may read or change a record `owner` created.

        A record from before principals were recorded has no owner, and only operators reach it.
        """
        return self.tier is Tier.OPERATOR or (owner is not None and owner == self.id)


def principal_for(token: NamedToken) -> Principal:
    """The principal a configured token runs as."""
    return Principal(id=f"token:{token.id}", tier=token.tier, token=token)


ANONYMOUS = Principal(id="anonymous", tier=Tier.PUBLIC)
"""A caller without a credential while authentication is on."""

OPEN = Principal(id="anonymous", tier=Tier.OPERATOR)
"""Every caller while authentication is off: tiers and ownership are not enforced."""


class _TokenConfig(BaseModel):
    # A secret pasted into `token_env` by mistake must not be echoed in the startup error.
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    token_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    tier: Literal["authenticated", "operator"]


class _AuthenticationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tokens: list[_TokenConfig] = []


def configured_tokens() -> tuple[NamedToken, ...]:
    """The named tokens, read from config and the environment; empty when authentication is off.

    Raises ValueError on an invalid block, an unset variable, a secret shorter than
    MIN_SECRET_LENGTH, or a repeated id or secret, so startup refuses a configuration that would
    leave a client unable to authenticate or two clients indistinguishable.
    """
    raw = api_config.get_config().get("authentication")
    if raw is None:
        return ()
    try:
        settings = _AuthenticationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"authentication in GEOSERVICE_CONFIG is invalid: {exc}") from exc
    tokens: list[NamedToken] = []
    for index, item in enumerate(settings.tokens):
        where = f"authentication.tokens[{index}] ({item.id})"
        secret = os.environ.get(item.token_env, "").strip()
        if not secret:
            raise ValueError(f"{where}: environment variable {item.token_env} is not set")
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"{where}: the secret in {item.token_env} is shorter than {MIN_SECRET_LENGTH} characters")
        if any(token.id == item.id for token in tokens):
            raise ValueError(f"{where}: the id is used by an earlier token")
        if any(hmac.compare_digest(token.secret, secret) for token in tokens):
            raise ValueError(f"{where}: the secret in {item.token_env} is used by an earlier token")
        tokens.append(NamedToken(id=item.id, tier=Tier[item.tier.upper()], secret=secret))
    return tuple(tokens)


def owner_principal(owner: str | None) -> Principal | None:
    """The principal a recorded owner runs as when no request carries it, as a batch job does.

    None, which limits nothing, for work no client started (a pipeline run, a derived dataset
    refresh) and while authentication is off. An owner whose token is no longer configured keeps
    its id and the public tier, so it still reaches only its own records.
    """
    tokens = configured_tokens()
    if owner is None or not tokens:
        return None
    for token in tokens:
        if owner == f"token:{token.id}":
            return principal_for(token)
    return Principal(id=owner, tier=Tier.PUBLIC)


def authenticate(authorization: str | None, tokens: tuple[NamedToken, ...]) -> Principal | None:
    """The principal an ``Authorization`` header names: anonymous without one, None when invalid.

    Accepts ``Bearer <secret>``, openEO's ``Bearer basic//<secret>``, and ``Basic`` with the
    token id and secret, which is what openEO clients send to ``GET /credentials/basic``.
    """
    if not authorization:
        return ANONYMOUS
    scheme, _, credential = authorization.partition(" ")
    credential = credential.strip()
    token_id: str | None = None
    if scheme.lower() == "bearer":
        secret = credential.removeprefix("basic//")
    elif scheme.lower() == "basic":
        try:
            token_id, _, secret = base64.b64decode(credential, validate=True).decode().partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return None
    else:
        return None
    # Every token is compared, so the time taken does not say which one came close.
    matches = [token for token in tokens if hmac.compare_digest(token.secret.encode(), secret.encode())]
    if not matches or (token_id is not None and matches[0].id != token_id):
        return None
    return principal_for(matches[0])
