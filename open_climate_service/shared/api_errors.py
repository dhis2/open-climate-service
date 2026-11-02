"""The error responses routes declare in OpenAPI, so clients see every outcome, not only success."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """The body of an error response."""

    detail: str | dict[str, Any] = Field(
        description="What went wrong: a message, or an object such as the validation checks that failed."
    )


_DESCRIPTIONS = {
    400: "The request was malformed.",
    403: "This instance is read-only.",
    404: "No such resource.",
    409: "Refused in the resource's current state; the detail says why.",
    503: "Not possible now; try again.",
}


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for these error statuses."""
    return {code: {"model": ErrorDetail, "description": _DESCRIPTIONS[code]} for code in codes}
