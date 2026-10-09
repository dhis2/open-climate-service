"""Named export definitions: the ``exports`` table of the operational database (CLIM-1089, CLIM-1378).

An export is the mapping from a dataset to a destination (DHIS2 data elements, a CHAP file, an
analytics table) and the declaration its gate checks. It used to be a block in
``climate-service.yaml`` that took a restart to change; here it is edited live through
``/exports`` and validated by its plugin before it is saved. Connections and their secrets stay in
the configuration file and the environment: an export only names one.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from open_climate_service.state import db


def list_definitions() -> list[dict[str, Any]]:
    """Every export definition, by id."""
    return [deepcopy(body) for body in db.list_documents("exports").values()]


def get_definition(export_id: str) -> dict[str, Any] | None:
    """One export definition, or None."""
    body = db.list_documents("exports").get(export_id)
    return deepcopy(body) if body is not None else None


def save_definition(definition: dict[str, Any], *, check: Any = None) -> None:
    """Create or replace one definition; ``check`` is called with it inside the write and may refuse."""
    with db.write(configuration=True) as connection:
        if check is not None:
            check(definition)
        db.put_document(connection, "exports", str(definition["id"]), definition)


def delete_definition(export_id: str) -> bool:
    """Remove one definition; True when it existed."""
    with db.write(configuration=True) as connection:
        return db.delete_document(connection, "exports", export_id)


def replace_definitions(definitions: list[dict[str, Any]]) -> None:
    """Replace every definition at once, for importing an operational configuration document."""
    with db.write(configuration=True) as connection:
        db.replace_documents(connection, "exports", {str(item["id"]): item for item in definitions})
