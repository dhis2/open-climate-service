"""DHIS2 as a pipeline destination: its settings, its export definition and its checks.

Everything a pipeline needs to know about DHIS2 lives here and is reached through the
export plugin hooks, so the pipeline core never names DHIS2, its UIDs or its period types.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from open_climate_service import config as api_config
from open_climate_service.exports.base import (
    DestinationCheck,
    DestinationContext,
    LookupOption,
    PipelineDataset,
    PipelineSettingsForm,
    SettingsFormField,
    SettingsFormSection,
)
from open_climate_service.shared.features import is_dhis2_uid
from open_climate_service.shared.time import normalise_export_period

logger = logging.getLogger(__name__)

_DHIS2_PERIOD_TYPE = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
}
_POLYGON_TYPES = {"Polygon", "MultiPolygon"}
# Lookup name -> DHIS2 metadata resource, for the form's search and the labels of stored UIDs.
_LOOKUPS = {"data_sets": "dataSets", "data_elements": "dataElements"}
_LOOKUP_PAGE = 20
_LABEL_CHUNK = 50
# The settings path of each series' data element: its form field, and where validation keeps its labels.
_DATA_ELEMENT_LABELS = "series.data_element"


class Dhis2Series(BaseModel):
    """One workflow result series going to one DHIS2 data element."""

    model_config = ConfigDict(extra="forbid")

    data_element: str = Field(description="DHIS2 data element UID the values are imported into.")

    @field_validator("data_element")
    @classmethod
    def _uid(cls, value: str) -> str:
        if not is_dhis2_uid(value):
            raise ValueError("data_element must be a DHIS2 UID: 11 characters, a letter first")
        return value


class Dhis2DestinationSettings(BaseModel):
    """The DHIS2 side of a pipeline: connection, optional data set, and the series mapping."""

    model_config = ConfigDict(extra="forbid")

    connection: str = Field(description="Id of a named DHIS2 connection from the instance configuration.")
    data_set: str | None = Field(
        default=None, description="DHIS2 data set UID, checked for period type and assignment."
    )
    series: list[Dhis2Series] = Field(min_length=1)

    @field_validator("data_set")
    @classmethod
    def _data_set_uid(cls, value: str | None) -> str | None:
        if value is not None and not is_dhis2_uid(value):
            raise ValueError("data_set must be a DHIS2 UID: 11 characters, a letter first")
        return value


@dataclass(frozen=True)
class FeatureBinding:
    """Stored geometry plus the DHIS2 template parameters that produced it."""

    ids: list[str]
    geometry_types: list[str]
    connection: str | None
    level: int | None


def connections() -> dict[str, Any]:
    """The named DHIS2 connections the instance configures, by id."""
    from open_climate_service.exports.dhis2_config import parse_connections

    return parse_connections(api_config.get_config().get("dhis2_connections", []))


def settings_schema() -> dict[str, Any]:
    """The settings JSON Schema, with the configured connections offered as choices."""
    schema = Dhis2DestinationSettings.model_json_schema()
    configured = sorted(connections())
    if configured:
        schema["properties"]["connection"]["enum"] = configured
    return schema


def pipeline_settings_form() -> PipelineSettingsForm:
    """The complete DHIS2 form presentation; its values and constraints stay in the schema."""
    return PipelineSettingsForm(
        title="DHIS2",
        description="Choose the DHIS2 instance and map each workflow result series to a data element.",
        sections=(
            SettingsFormSection(
                title="Target",
                description="The connection and optional data set used to validate and deliver the values.",
                fields=("connection", "data_set"),
            ),
            SettingsFormSection(
                title="Series mapping",
                description="Map every result series that should be delivered.",
                fields=("series",),
            ),
        ),
        fields=(
            SettingsFormField(
                name="connection",
                label="Connection",
                help_text="A named DHIS2 connection from the instance configuration.",
                widget="select",
            ),
            SettingsFormField(
                name="data_set",
                label="Data set",
                help_text=(
                    "Optional. Search the connected DHIS2 by name or UID; the data set is checked for period "
                    "type and assignment, and narrows the data element search."
                ),
                widget="lookup",
                lookup="data_sets",
            ),
            SettingsFormField(
                name="series",
                label="Series",
                help_text="Fill the empty row to add a mapping; save to get another.",
                widget="rows",
            ),
            SettingsFormField(
                name=_DATA_ELEMENT_LABELS,
                label="Data element",
                help_text="The aggregate data element that receives the values; search by name or UID.",
                widget="lookup",
                lookup="data_elements",
            ),
        ),
    )


def lookup_options(lookup: str, settings: dict[str, Any], query: str) -> list[LookupOption]:
    """Search the connected DHIS2 for data sets or aggregate data elements, by name, code or UID.

    Reads only the connection, and the data set when one is chosen, from the form's values:
    elements are then narrowed to that data set, so the mapping cannot name one it will not accept.
    """
    resource = _LOOKUPS.get(lookup)
    if resource is None:
        raise ValueError(f"DHIS2 has no lookup '{lookup}'")
    filters = [f"identifiable:token:{query.strip()}"] if query.strip() else []
    fields = "id,displayName,periodType"
    if lookup == "data_elements":
        fields = "id,displayName,valueType"
        filters.append("domainType:eq:AGGREGATE")
        data_set = settings.get("data_set")
        if is_dhis2_uid(data_set):
            filters.append(f"dataSetElements.dataSet.id:eq:{data_set}")
    params: dict[str, Any] = {"fields": fields, "order": "displayName:asc", "pageSize": _LOOKUP_PAGE, "page": 1}
    if filters:
        params["filter"] = filters
    with closing(_lookup_client(settings)) as client:
        answer = client.get(f"/api/{resource}", params=params)
    options: list[LookupOption] = []
    for item in answer.get(resource, []):
        if not isinstance(item, dict) or not is_dhis2_uid(item.get("id")):
            continue
        detail = item.get("periodType") if lookup == "data_sets" else item.get("valueType")
        options.append(
            LookupOption(
                value=item["id"],
                label=str(item.get("displayName") or item["id"]),
                detail=str(detail).replace("_", " ").lower() if detail else "",
            )
        )
    return options


def lookup_labels(lookup: str, settings: dict[str, Any], values: Sequence[str]) -> dict[str, str]:
    """The display names of stored data set or data element UIDs; unknown UIDs are left out."""
    resource = _LOOKUPS.get(lookup)
    if resource is None:
        raise ValueError(f"DHIS2 has no lookup '{lookup}'")
    uids = [value for value in values if is_dhis2_uid(value)]
    labels: dict[str, str] = {}
    if not uids:
        return labels
    with closing(_lookup_client(settings)) as client:
        for start in range(0, len(uids), _LABEL_CHUNK):
            chunk = ",".join(uids[start : start + _LABEL_CHUNK])
            answer = client.get(
                f"/api/{resource}",
                params={"fields": "id,displayName", "filter": f"id:in:[{chunk}]", "paging": "false"},
            )
            for item in answer.get(resource, []):
                if isinstance(item, dict) and is_dhis2_uid(item.get("id")) and item.get("displayName"):
                    labels[item["id"]] = str(item["displayName"])
    return labels


def _lookup_client(settings: dict[str, Any]) -> Any:
    """A client for the connection the form names, or a ValueError saying what to choose first."""
    from open_climate_service.exports.dhis2 import get_connection

    connection = settings.get("connection")
    if not isinstance(connection, str) or not connection:
        raise ValueError("choose a DHIS2 connection first")
    if connection not in connections():
        raise ValueError(f"'{connection}' is not a configured DHIS2 connection")
    return get_connection(connection)


def parse_settings(settings: dict[str, Any]) -> Dhis2DestinationSettings:
    """Validate settings, turning pydantic's errors into one readable ValueError."""
    try:
        return Dhis2DestinationSettings.model_validate(settings)
    except ValidationError as exc:
        raise ValueError(
            "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        ) from None


def _single(items: tuple[Any, ...], noun: str) -> Any:
    if len(items) != 1:
        raise ValueError(f"a DHIS2 destination needs exactly one {noun} in the workflow arguments, found {len(items)}")
    return items[0]


def export_definition(settings: dict[str, Any], context: DestinationContext) -> dict[str, Any]:
    """The named-export definition: the dataset's cadence passes through as the DHIS2 period."""
    parsed = parse_settings(settings)
    dataset: PipelineDataset = _single(context.datasets, "source dataset")
    period_type = normalise_export_period(dataset.cadence)
    if period_type is None:
        raise ValueError(
            f"the dataset is {dataset.cadence}, which is not a DHIS2 period; the workflow does not resample in "
            "time, so choose a daily, weekly, monthly or yearly dataset"
        )
    return {
        "dataset": dataset.id,
        "connection": parsed.connection,
        "period_type": period_type,
        "series": [{"select": {}, "data_element": series.data_element} for series in parsed.series],
    }


def check_destination(settings: dict[str, Any], context: DestinationContext) -> list[DestinationCheck]:
    """Every DHIS2-side binding, in the order a delivery would hit it."""
    try:
        parsed = parse_settings(settings)
    except ValueError as exc:
        return [DestinationCheck(id="dhis2_settings", status="fail", message=str(exc))]
    checks: list[DestinationCheck] = []
    period_type: str | None = None
    if len(context.datasets) == 1:
        dataset = context.datasets[0]
        period_type = normalise_export_period(dataset.cadence)
        if period_type is None:
            checks.append(
                DestinationCheck(
                    id="cadence",
                    status="fail",
                    message=(
                        f"the dataset is {dataset.cadence}, which is not a DHIS2 period; the workflow does not "
                        "resample in time, so choose a daily, weekly, monthly or yearly dataset"
                    ),
                )
            )
        else:
            checks.append(
                DestinationCheck(
                    id="cadence", status="pass", message=f"{dataset.cadence} data exports as {period_type} periods"
                )
            )
        elements = ", ".join(series.data_element for series in parsed.series)
        checks.append(
            DestinationCheck(
                id="series", status="pass", message=f"data element {elements} receives '{dataset.variable}'"
            )
        )
    else:
        checks.append(
            DestinationCheck(
                id="cadence",
                status="fail",
                message=(
                    f"a DHIS2 destination needs exactly one source dataset, the workflow binds {len(context.datasets)}"
                ),
            )
        )

    feature_ids: list[str] = []
    if len(context.feature_collections) != 1:
        checks.append(
            DestinationCheck(
                id="organisation_units",
                status="fail",
                message=(
                    "a DHIS2 destination needs exactly one feature collection of organisation units, the workflow "
                    f"binds {len(context.feature_collections)}"
                ),
            )
        )
    else:
        checks.append(_check_organisation_units(context.feature_collections[0], parsed.connection, feature_ids))

    if parsed.connection in connections():
        checks.append(DestinationCheck(id="connection", status="pass", message=f"'{parsed.connection}' is configured"))
    else:
        checks.append(
            DestinationCheck(
                id="connection", status="fail", message=f"'{parsed.connection}' is not a configured DHIS2 connection"
            )
        )
    checks.append(_check_dhis2_metadata(parsed, period_type, feature_ids))
    return checks


def _check_organisation_units(collection_id: str, connection: str, feature_ids: list[str]) -> DestinationCheck:
    """The collection's ids must be organisation unit UIDs from the same DHIS2 connection."""
    try:
        binding = feature_binding(collection_id)
        feature_ids.extend(binding.ids)
        if not binding.geometry_types or not set(binding.geometry_types).issubset(_POLYGON_TYPES):
            raise ValueError(
                f"zonal aggregation requires Polygon or MultiPolygon geometry; found {binding.geometry_types or 'none'}"
            )
        if binding.connection is None:
            raise ValueError(
                "the collection was not fetched from DHIS2, so its feature ids cannot be verified as "
                "organisation unit UIDs"
            )
        if binding.connection != connection:
            raise ValueError(
                f"the collection was declared for DHIS2 connection {binding.connection!r}, but the destination uses "
                f"{connection!r}"
            )
    except LookupError as exc:
        return DestinationCheck(id="organisation_units", status="fail", message=str(exc))
    except ValueError as exc:
        return DestinationCheck(
            id="organisation_units",
            status="fail",
            message=f"the feature collection cannot be used for this DHIS2 destination: {exc}",
        )
    level = f" at level {binding.level}" if binding.level is not None else ""
    return DestinationCheck(
        id="organisation_units",
        status="pass",
        message=f"{len(binding.ids)} polygon features{level} from connection {binding.connection!r}",
    )


def feature_binding(collection_id: str) -> FeatureBinding:
    """Resolve stored polygon identity and its declared DHIS2 source."""
    from open_climate_service.features import templates as feature_templates
    from open_climate_service.features.services import get_feature_collection_or_404, registered_collections
    from open_climate_service.plugins.processes.load_features import load_features
    from open_climate_service.shared.features import validate_dhis2_feature_ids

    if collection_id not in registered_collections():
        raise LookupError(f"'{collection_id}' is not a registered feature collection; fetch it first")
    record = get_feature_collection_or_404(collection_id)
    template = feature_templates.get_feature_template(collection_id) or {}
    if template.get("provider") != "dhis2":
        raise ValueError(
            f"the collection was not fetched from DHIS2 (provider {template.get('provider')!r}), so its ids are not "
            "organisation unit UIDs; choose a collection fetched through the DHIS2 provider"
        )
    raw_params = template.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    connection = params.get("connection")
    level = params.get("level")
    return FeatureBinding(
        ids=validate_dhis2_feature_ids(load_features(collection_id)),
        geometry_types=record.geometry_types,
        connection=connection if isinstance(connection, str) else None,
        level=level if isinstance(level, int) and not isinstance(level, bool) else None,
    )


def _check_dhis2_metadata(
    settings: Dhis2DestinationSettings, period_type: str | None, feature_ids: list[str]
) -> DestinationCheck:
    """Ask DHIS2 whether the data set, elements and org units line up; skip when it cannot be asked."""
    data_set = settings.data_set
    if data_set is None:
        return DestinationCheck(
            id="dhis2_metadata",
            status="skip",
            message="no data set declared, so period type and assignment are not checked",
        )
    if settings.connection not in connections():
        return DestinationCheck(id="dhis2_metadata", status="skip", message="connection not configured")
    try:
        from open_climate_service.exports.dhis2 import get_connection

        with closing(get_connection(settings.connection)) as client:
            metadata = client.get(
                f"/api/dataSets/{data_set}",
                params={"fields": "id,displayName,periodType,dataSetElements[dataElement[id]],organisationUnits[id]"},
            )
    except Exception as exc:  # the connection, its token, or DHIS2 itself
        return DestinationCheck(
            id="dhis2_metadata", status="fail", message=f"could not read the data set from DHIS2: {exc}"
        )
    problems: list[str] = []
    expected = _DHIS2_PERIOD_TYPE.get(period_type or "")
    if expected and metadata.get("periodType") != expected:
        problems.append(f"data set period type is {metadata.get('periodType')!r}, the export is {expected}")
    elements = {entry.get("dataElement", {}).get("id") for entry in metadata.get("dataSetElements", [])}
    missing = [series.data_element for series in settings.series if series.data_element not in elements]
    if missing:
        problems.append(f"data element(s) not in the data set: {', '.join(missing)}")
    assigned = {entry.get("id") for entry in metadata.get("organisationUnits", [])}
    unassigned = [identifier for identifier in feature_ids if identifier not in assigned]
    if feature_ids and unassigned:
        sample = ", ".join(unassigned[:5]) + (" ..." if len(unassigned) > 5 else "")
        problems.append(
            f"{len(unassigned)} of {len(feature_ids)} organisation units are not assigned to the data set: {sample}"
        )
    if problems:
        return DestinationCheck(id="dhis2_metadata", status="fail", message="; ".join(problems))
    return DestinationCheck(
        id="dhis2_metadata",
        status="pass",
        message=(
            f"'{metadata.get('displayName', data_set)}' is {expected}, holds the element(s), "
            "every organisation unit assigned"
        ),
    )


def preview(
    content: bytes,
    settings: dict[str, Any],
    context: DestinationContext,
    labels: Mapping[str, Mapping[str, str]] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """A sample of a rendered dataValueSet, organisation units and data elements by name.

    The payload runs period by period within each organisation unit, in the feature collection's
    order. The sample instead takes the first value of each unit, units by name as DHIS2 lists
    them, before a second value of any, so a reader sees the extent of the delivery rather than
    one unit's months. Organisation unit names come from the bound feature collection, which
    stores what DHIS2 sent with the geometry. Data element names come from ``labels``, as the
    last validation resolved them; DHIS2 is asked only for any it did not, and a name is left
    out when it cannot answer.
    """
    try:
        values = [value for value in json.loads(content).get("dataValues", []) if isinstance(value, dict)]
    except (ValueError, AttributeError):
        return []
    if not values:
        return []
    org_units: dict[str, str] = {}
    for collection_id in context.feature_collections:
        org_units.update(feature_names(collection_id))
    rows = _spread_over_org_units(values, limit, org_units)
    known = (labels or {}).get(_DATA_ELEMENT_LABELS, {})
    uids = sorted({row["dataElement"] for row in rows if is_dhis2_uid(row.get("dataElement"))})
    elements = {uid: known[uid] for uid in uids if uid in known}
    missing = [uid for uid in uids if uid not in elements]
    if missing:
        try:
            elements.update(lookup_labels("data_elements", settings, missing))
        except Exception as exc:  # the connection, its token, or DHIS2 itself
            logger.warning("Could not resolve data element names for the dry-run sample: %s", exc)

    names_by_key = {"orgUnit": org_units, "dataElement": elements}

    def cell(key: str, value: Any) -> Any:
        names = names_by_key.get(key, {})
        return {"value": value, "label": names[value]} if isinstance(value, str) and value in names else value

    return [{key: cell(key, value) for key, value in row.items()} for row in rows]


def _spread_over_org_units(values: list[dict[str, Any]], limit: int, names: dict[str, str]) -> list[dict[str, Any]]:
    """Up to ``limit`` values: each organisation unit's first value, units by name, then the rest in payload order."""
    first_by_unit: dict[Any, int] = {}
    for index, value in enumerate(values):
        first_by_unit.setdefault(value.get("orgUnit"), index)

    def by_name(item: tuple[Any, int]) -> str:
        unit = item[0]
        return (names.get(unit) if isinstance(unit, str) else None) or str(unit)

    chosen = [index for _, index in sorted(first_by_unit.items(), key=lambda item: by_name(item).casefold())][:limit]
    taken = set(chosen)
    chosen.extend(index for index in range(len(values)) if index not in taken)
    return [values[index] for index in chosen[:limit]]


def feature_names(collection_id: str) -> dict[str, str]:
    """Each stored feature's display name by its id; empty when the collection cannot be read.

    Reads only the id and name columns, never the geometry, which a large hierarchy makes heavy.
    """
    from open_climate_service.features import services as feature_services
    from open_climate_service.features import store as feature_store

    try:
        record = feature_services.registered_collections().get(collection_id)
        if record is None or record.features is None:
            raise ValueError(f"'{collection_id}' is not a registered feature collection")
        id_property = record.features.id_property
        rows = feature_store.read_feature_columns(record, [id_property, "displayName", "name"])
    except Exception as exc:  # not registered, or its store unreadable: the sample then shows ids
        logger.warning("Could not read feature names from '%s': %s", collection_id, exc)
        return {}
    names: dict[str, str] = {}
    for row in rows:
        identifier = row.get(id_property)
        name = row.get("displayName") or row.get("name")
        if isinstance(identifier, str) and isinstance(name, str) and name:
            names[identifier] = name
    return names
