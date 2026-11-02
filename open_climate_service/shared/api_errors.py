"""The error responses routes declare in OpenAPI, so clients see every outcome, not only success."""

from __future__ import annotations

from typing import Any

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError


class ErrorDetail(BaseModel):
    """The body of an error response."""

    detail: str | dict[str, Any] | list[dict[str, Any]] = Field(
        description=(
            "What went wrong: a message; an object such as the validation checks that failed; "
            "or, for a body that does not fit its schema, one entry per problem."
        )
    )


_DESCRIPTIONS = {
    400: "The request was malformed.",
    403: "This instance is read-only.",
    404: "No such resource.",
    409: "Refused in the resource's current state; the detail says why.",
    422: "The body does not fit its schema, or the change failed a check; the detail says which.",
    503: "Not possible now; try again.",
}


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for these error statuses."""
    return {code: {"model": ErrorDetail, "description": _DESCRIPTIONS[code]} for code in codes}


def body_error(exc: ValidationError) -> RequestValidationError:
    """A body a route validated itself, reported as FastAPI reports a typed body: under ``body``."""
    return RequestValidationError(
        [{**error, "loc": ("body", *error["loc"])} for error in exc.errors(include_url=False, include_context=False)]
    )
