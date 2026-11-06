"""Resolve named destinations and persist their rendered payloads."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from open_climate_service import config
from open_climate_service.destinations.base import BaseDestinationPlugin, RenderedPayload
from open_climate_service.destinations.registry import load_destination_plugins

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedDestination:
    """Invocation-local mapping and source/target references."""

    destination_id: str
    plugin: BaseDestinationPlugin
    mapping: dict[str, Any]
    references: dict[str, str]


def _configured_destination_definitions() -> dict[str, dict[str, Any]]:
    """Validate configured and pipeline-owned destination definitions and index them by ID.

    An ID claimed by both the configuration and a pipeline is left out and listed by
    :func:`_contested_destination_ids`, so resolving it fails instead of picking one.
    """
    by_id, _ = _destination_definitions()
    return by_id


def _destination_definitions() -> tuple[dict[str, dict[str, Any]], set[str]]:
    definitions = config.get_config().get("destinations", [])
    if not isinstance(definitions, list):
        raise ValueError("destinations must be a list of named mappings")
    by_id: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("Each destination definition must be a mapping")
        identifier = definition.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError("Each destination requires an ID containing letters, digits, underscores, or hyphens")
        if identifier in by_id:
            raise ValueError(f"Duplicate destination ID '{identifier}'")
        by_id[identifier] = definition
    # A validated pipeline owns the named destination under its id. Neither side wins a clash.
    from open_climate_service.pipelines.destinations import pipeline_destination_definitions

    contested: set[str] = set()
    for definition in pipeline_destination_definitions():
        identifier = str(definition["id"])
        if identifier in by_id:
            contested.add(identifier)
            logger.error(
                "Destination '%s' is declared in the configuration and owned by the pipeline of the same id; "
                "neither is used until one is renamed or removed",
                identifier,
            )
            continue
        by_id[identifier] = definition
    for identifier in contested:
        by_id.pop(identifier, None)
    return by_id, contested


def resolve_named_destination(fmt: str, options: dict[str, Any]) -> ResolvedDestination:
    """Resolve and validate a mapping without rendering or resolving a target."""
    destination_id = options.get("destination")
    if not isinstance(destination_id, str) or not destination_id.strip() or set(options) != {"destination"}:
        raise ValueError("Named destinations require options containing only a non-empty 'destination' ID")
    by_id, contested = _destination_definitions()
    if destination_id in contested:
        raise ValueError(
            f"Destination '{destination_id}' is declared in the configuration and owned by the pipeline "
            "of the same id; "
            "rename the pipeline or remove the configured destination"
        )
    if destination_id not in by_id:
        raise ValueError(f"Unknown destination '{destination_id}'")
    resolved = resolve_destination_definition(by_id[destination_id])
    if resolved.plugin.format != fmt:
        raise ValueError(f"Destination '{destination_id}' requires format '{resolved.plugin.format}', received '{fmt}'")
    return resolved


def resolve_destination_definition(definition: dict[str, Any]) -> ResolvedDestination:
    """Resolve one explicit definition through the same path as an instance named destination.

    Pipelines use this to validate and dry-run the definition their destination compiles,
    before it is resolvable by name. It deliberately accepts a complete definition rather than
    adding a process-wide temporary destination that concurrent jobs could observe.
    """
    definition = deepcopy(definition)
    destination_id = definition.pop("id", None)
    if not isinstance(destination_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", destination_id):
        raise ValueError("Each destination requires an ID containing letters, digits, underscores, or hyphens")
    plugin_id = definition.pop("plugin", None)
    if not isinstance(plugin_id, str):
        raise ValueError("Destination definition requires a plugin ID")
    plugin = load_destination_plugins().get(plugin_id)
    if plugin is None:
        raise ValueError(f"Unknown destination plugin '{plugin_id}'")
    references: dict[str, str] = {}
    for field in ("dataset", "org_units", "connection"):
        value = definition.pop(field, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Destination {field} must be a non-empty reference string")
        if value is not None:
            references[field] = value.strip()
    from open_climate_service.destinations.manifest import validate_public_mapping

    validate_public_mapping(definition)
    mapping = plugin.validate_mapping(definition)
    validate_public_mapping(mapping)
    resolved = ResolvedDestination(destination_id, plugin, deepcopy(mapping), references)
    check_cadence_declaration(resolved)
    return resolved


def validate_configured_destinations() -> None:
    """Resolve every configured destination, so a cadence contradiction fails startup, not a job.

    Automation already resolves the destinations its triggers deliver through; this covers the
    rest, so a destination nobody has run yet is refused with the same message at the same time.
    """
    # A pipeline's destination is checked when the pipeline validates; a stale one must not stop startup.
    from open_climate_service.pipelines.destinations import pipeline_destination_definitions

    owned = {str(definition["id"]) for definition in pipeline_destination_definitions()}
    for destination_id, definition in _configured_destination_definitions().items():
        if destination_id in owned:
            continue
        plugin_id = definition.get("plugin")
        plugin = load_destination_plugins().get(plugin_id) if isinstance(plugin_id, str) else None
        if plugin is None:
            raise ValueError(f"Destination '{destination_id}' names an unknown plugin {plugin_id!r}")
        resolve_named_destination(plugin.format, {"destination": destination_id})


def declared_dataset_cadence(resolved: ResolvedDestination) -> str | None:
    """The native period_type of the destination's declared dataset, when both are known."""
    dataset_id = resolved.references.get("dataset")
    if dataset_id is None:
        return None
    from open_climate_service.data_registry.services import datasets as registry

    try:
        template = registry.get_dataset(dataset_id)
    except Exception:
        logger.debug("Could not read the dataset template for destination '%s'", resolved.destination_id, exc_info=True)
        return None
    cadence = template.get("period_type") if isinstance(template, dict) else None
    return cadence if isinstance(cadence, str) else None


def check_cadence_declaration(resolved: ResolvedDestination) -> None:
    """Refuse a destination whose period cannot be produced from its dataset, or that is declared wrong.

    With a declared dataset, the three outcomes of `period_reachability` map to: pass-through
    needs no `temporal_aggregation` and refuses one; aggregation requires one; unreachable is
    refused outright. Without a declared dataset nothing is known here, and the data-based
    check at render time is the only guard.
    """
    from open_climate_service.shared.time import Reachability, period_reachability

    destination = resolved.mapping.get("period_type")
    if not isinstance(destination, str):
        return
    source = declared_dataset_cadence(resolved)
    if source is None:
        return
    declared = resolved.mapping.get("temporal_aggregation")
    outcome, reason = period_reachability(source, destination)
    if outcome is Reachability.UNREACHABLE:
        raise ValueError(
            f"Destination '{resolved.destination_id}' cannot emit {destination} periods from the {source} dataset "
            f"'{resolved.references['dataset']}': {reason}"
        )
    if outcome is Reachability.AGGREGATE and declared is None:
        raise ValueError(
            f"Destination '{resolved.destination_id}' emits {destination} periods from the {source} dataset "
            f"'{resolved.references['dataset']}', so it must declare temporal_aggregation "
            "(sum, mean, min or max) to say how source periods are combined"
        )
    if outcome is Reachability.PASS_THROUGH and declared is not None:
        raise ValueError(
            f"Destination '{resolved.destination_id}' declares temporal_aggregation '{declared}' but its dataset "
            f"'{resolved.references['dataset']}' is already {source}; remove the declaration"
        )


def render_named_destination(
    data: Any,
    fmt: str,
    options: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
) -> tuple[BaseDestinationPlugin, RenderedPayload]:
    """Render a declared destination without resolving a connection or credential.

    When execution provenance is available, the destination's declarations are checked
    against it exactly as for a saved batch payload.
    """
    resolved = resolve_named_destination(fmt, options)
    return resolved.plugin, render_resolved_destination(data, resolved, provenance=provenance)


def render_resolved_destination(
    data: Any,
    resolved: ResolvedDestination,
    *,
    provenance: dict[str, Any] | None = None,
) -> RenderedPayload:
    """Render an already resolved destination through every provenance and data-side gate.

    This is the invocation-local counterpart to :func:`render_named_destination`. Draft pipelines
    resolve their generated definition directly, but must still receive the exact cadence and
    incomplete-period checks applied to configured named destinations.
    """
    if provenance is not None:
        check_execution_declarations(resolved, provenance)
    return _render(data, resolved, provenance)


def check_execution_declarations(resolved: ResolvedDestination, provenance: dict[str, Any]) -> None:
    """Refuse a render whose declared dataset or aggregation contradicts what ran.

    A declaration is checked only where provenance can attribute an observation to
    the result. The dataset must be among the observed sources. The aggregation is
    checked only when exactly one spatial aggregation ran with a named method;
    otherwise the manifest lists it as missing evidence.
    """
    sources_value = provenance.get("sources", [])
    if not isinstance(sources_value, list) or not all(isinstance(source, dict) for source in sources_value):
        raise ValueError("Result provenance sources must be a list of mappings")
    sources = cast(list[dict[str, Any]], sources_value)
    declared = resolved.references.get("dataset")
    if (
        declared is not None
        and sources
        and not any(declared in (source.get("collection_id"), source.get("source_dataset_id")) for source in sources)
    ):
        raise ValueError("Declared destination dataset was not observed during execution")

    observed_value = provenance.get("spatial_aggregations", [])
    if not isinstance(observed_value, list) or not all(
        method is None or isinstance(method, str) for method in observed_value
    ):
        raise ValueError("Result provenance spatial_aggregations must be a list of method names")
    observed = cast(list[str | None], observed_value)
    declared_aggregation = resolved.mapping.get("aggregation")
    if (
        declared_aggregation is not None
        and len(observed) == 1
        and observed[0] is not None
        and observed[0] != declared_aggregation
    ):
        raise ValueError(
            f"Declared destination aggregation '{declared_aggregation}' does not match the executed "
            f"spatial aggregation '{observed[0]}'"
        )
    _check_temporal_declaration(resolved, provenance)


def _check_temporal_declaration(resolved: ResolvedDestination, provenance: dict[str, Any]) -> None:
    """Verify the destination's period was produced the way it declares (CLIM-1302).

    A declared `temporal_aggregation` must be matched by exactly one observed temporal
    aggregation that produced the destination's period with that reducer. An undeclared one must
    not have been aggregated to some other period on the way. Incomplete destination periods
    are refused unless the destination says `incomplete_periods: drop`, in which case `_render`
    drops them.
    """
    observed_value = provenance.get("temporal_aggregations", [])
    if not isinstance(observed_value, list) or not all(isinstance(entry, dict) for entry in observed_value):
        raise ValueError("Result provenance temporal_aggregations must be a list of mappings")
    observed = cast(list[dict[str, Any]], observed_value)
    destination = resolved.mapping.get("period_type")
    declared = resolved.mapping.get("temporal_aggregation")
    producing = [entry for entry in observed if entry.get("period") == destination]
    if declared is not None:
        if not producing:
            raise ValueError(
                f"Destination '{resolved.destination_id}' declares temporal_aggregation '{declared}' to {destination} "
                "periods, but no temporal aggregation to that period ran; add aggregate_temporal_period "
                "before save_result, or deliver a dataset already at that period"
            )
        if len(producing) > 1:
            raise ValueError(
                f"Destination '{resolved.destination_id}' saw {len(producing)} temporal aggregations to {destination} "
                "periods; only one can be attributed to the result"
            )
        method = producing[0].get("method")
        if method != declared:
            raise ValueError(
                f"Destination '{resolved.destination_id}' declares temporal_aggregation '{declared}' but the executed "
                f"aggregation to {destination} periods used {method!r}"
            )
    elif observed and not producing:
        periods = sorted({str(entry["period"]) for entry in observed if entry.get("period")}) or ["another period"]
        raise ValueError(
            f"Destination '{resolved.destination_id}' emits {destination} periods but the graph aggregated to "
            f"{', '.join(periods)}; the destination's period_type and the aggregation disagree"
        )
    unverifiable = _unverifiable_completeness(observed, str(destination))
    if unverifiable is not None:
        raise ValueError(
            f"Destination '{resolved.destination_id}' cannot verify that its {destination} periods are complete: "
            f"{unverifiable}. Declare the dataset on the destination and register its period_type, or deliver a "
            "dataset already at that period"
        )
    incomplete = _incomplete_at_destination(observed, str(destination))
    if incomplete and resolved.mapping.get("incomplete_periods", "reject") != "drop":
        raise ValueError(
            f"Destination '{resolved.destination_id}' would emit {destination} periods the source does not fully "
            f"cover: {', '.join(incomplete)}. Sync through the end of the period, narrow the temporal "
            "extent, or declare incomplete_periods: drop on the destination"
        )


def _unverifiable_completeness(entries: list[dict[str, Any]], destination: str) -> str | None:
    """The reason completeness could not be checked by an aggregation feeding this period, if any.

    An entry that could not count its input is not evidence of completeness, and `drop` has
    nothing to drop, so the destination refuses rather than guess. Older provenance without the
    field counted everything it recorded.
    """
    from open_climate_service.shared.time import Reachability, period_reachability

    for entry in entries:
        if entry.get("completeness") != "unknown":
            continue
        period = entry.get("period")
        if not isinstance(period, str):
            return str(entry.get("reason") or "an aggregation produced no exportable period")
        outcome, _ = period_reachability(period, destination)
        if outcome in (Reachability.PASS_THROUGH, Reachability.AGGREGATE):
            return str(entry.get("reason") or "an aggregation could not count its input")
    return None


def _incomplete_at_destination(entries: list[dict[str, Any]], destination: str) -> list[str]:
    """Every incomplete period any temporal aggregation recorded, relabelled at the destination's period.

    A chain of aggregations (daily to monthly to yearly) loses nothing this way: a month the
    first step found short of days marks the year it falls in, even though the second step saw
    twelve whole months. Entries for a period coarser than the destination cannot be mapped
    onto it and are ignored; the period mismatch is reported elsewhere.
    """
    from open_climate_service.shared.time import (
        Reachability,
        export_period_label,
        period_label_start,
        period_reachability,
    )

    relabelled: set[str] = set()
    for entry in entries:
        period = entry.get("period")
        labels = entry.get("incomplete")
        if labels is None:
            labels = []
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("Result provenance temporal_aggregations carry malformed incomplete periods")
        if not isinstance(period, str) or not labels:
            continue
        outcome, _ = period_reachability(period, destination)
        if outcome is Reachability.PASS_THROUGH:
            relabelled.update(labels)
        elif outcome is Reachability.AGGREGATE:
            relabelled.update(export_period_label(period_label_start(label, period), destination) for label in labels)
    return sorted(relabelled)


def incomplete_periods_to_drop(resolved: ResolvedDestination, provenance: dict[str, Any] | None) -> list[str]:
    """Destination periods the destination has chosen to drop rather than refuse."""
    if provenance is None or resolved.mapping.get("incomplete_periods") != "drop":
        return []
    destination = resolved.mapping.get("period_type")
    entries = provenance.get("temporal_aggregations", [])
    if not isinstance(entries, list) or not isinstance(destination, str):
        return []
    return _incomplete_at_destination([entry for entry in entries if isinstance(entry, dict)], destination)


def check_data_cadence(data: Any, resolved: ResolvedDestination) -> None:
    """Refuse a render whose data is spaced differently from the period it would be labelled with.

    The decisive guard, independent of configuration: the cadence the result carries (stamped
    at load, rewritten by each temporal aggregation) must be the destination's period. A mismatch
    means either a temporal step is missing (finer data, CLIM-1139's collision) or the period
    cannot be derived at all (coarser data). A result carrying no cadence is read from its axis
    spacing, which needs two timestamps; a single period is then accepted as is.
    """
    from open_climate_service.shared.time import (
        Reachability,
        cadence_of,
        cadence_to_openeo_period,
        infer_cadence,
        period_reachability,
    )

    destination = resolved.mapping.get("period_type")
    period_field = resolved.mapping.get("period_field", "t")
    if not isinstance(destination, str):
        return
    source = cadence_of(data)
    if source is None:
        values = _period_values(data, period_field)
        source = infer_cadence(values) if values is not None else None
    if source is None:
        return
    outcome, reason = period_reachability(source, destination)
    if outcome is Reachability.PASS_THROUGH:
        return
    if outcome is Reachability.AGGREGATE:
        period = cadence_to_openeo_period(destination)
        step = f"aggregate_temporal_period(period='{period}')" if period else "a temporal aggregation"
        raise ValueError(
            f"Destination '{resolved.destination_id}' emits {destination} periods but the result is spaced {source}; "
            f"each {destination} period would receive several values. Add {step} with the reducer you "
            f"mean before save_result, or deliver {source} periods"
        )
    raise ValueError(
        f"Destination '{resolved.destination_id}' emits {destination} periods but the result is spaced "
        f"{source}: {reason}"
    )


def _period_values(data: Any, period_field: str) -> Any:
    """The datetime values of the period axis, or None when there is no datetime axis to read."""
    import numpy as np

    try:
        import xarray as xr

        if isinstance(data, (xr.Dataset, xr.DataArray)):
            if period_field not in data.coords:
                return None
            values = np.asarray(data[period_field].values)
        else:
            import pandas as pd

            if not isinstance(data, pd.DataFrame) or period_field not in data.columns:
                return None
            values = np.asarray(data[period_field].values)
    except ImportError:
        return None
    if values.size < 2 or not np.issubdtype(values.dtype, np.datetime64):
        return None
    return np.unique(values)


def _drop_periods(data: Any, resolved: ResolvedDestination, labels: list[str]) -> Any:
    """Remove rows or steps whose period label is in ``labels``."""
    from open_climate_service.shared.time import export_period_label

    destination = str(resolved.mapping.get("period_type"))
    period_field = resolved.mapping.get("period_field", "t")
    import numpy as np
    import xarray as xr

    if isinstance(data, (xr.Dataset, xr.DataArray)) and period_field in data.coords:
        keep = np.array([export_period_label(value, destination) not in labels for value in data[period_field].values])
        # isel indexes dimensions. A period coordinate may sit on another dimension, as `t(observation)`
        # does in a vector cube, so index the dimension that coordinate is laid along.
        dims = data[period_field].dims
        if len(dims) != 1:
            raise ValueError(f"Cannot drop periods along '{period_field}': it spans {len(dims)} dimensions")
        return data.isel({str(dims[0]): keep})
    import pandas as pd

    if isinstance(data, pd.DataFrame) and period_field in data.columns:
        keep_rows = [export_period_label(value, destination) not in labels for value in data[period_field].values]
        return data.loc[keep_rows]
    return data


def _render(data: Any, resolved: ResolvedDestination, provenance: dict[str, Any] | None = None) -> RenderedPayload:
    dropped = incomplete_periods_to_drop(resolved, provenance)
    if dropped:
        data = _drop_periods(data, resolved, dropped)
    check_data_cadence(data, resolved)
    rendered = resolved.plugin.render(data, deepcopy(resolved.mapping))
    # External Python plugins are not necessarily type-checked.
    if not isinstance(rendered, RenderedPayload):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("Destination plugin render() must return RenderedPayload")
    return rendered


def write_named_destination(
    data: Any,
    directory: Path,
    fmt: str,
    options: dict[str, Any],
    *,
    job_id: str,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Freeze a resolved invocation and publish its payload and versioned manifest."""
    import hashlib

    from open_climate_service.destinations.manifest import (
        DeliveryManifest,
        plugin_identity,
        publish_manifest,
        target_binding,
        validate_public_mapping,
    )
    from open_climate_service.shared.provenance import json_digest
    from open_climate_service.shared.time import utc_now

    resolved = resolve_named_destination(fmt, options)
    identity = plugin_identity(resolved.plugin)
    target = target_binding(resolved.plugin, resolved.references)
    evidence = (
        deepcopy(provenance)
        if provenance is not None
        else {
            "scope": "unavailable",
            "sources": [],
            "features": [],
            "missing": ["execution_provenance"],
        }
    )
    validate_public_mapping(evidence)
    check_execution_declarations(resolved, evidence)
    rendered = _render(data, resolved, evidence)
    manifest = DeliveryManifest(
        source_job_id=job_id,
        destination_id=resolved.destination_id,
        created_at=utc_now().isoformat(),
        filename=f"export-{uuid4().hex}{identity.extension}",
        payload_sha256=hashlib.sha256(rendered.content).hexdigest(),
        payload_size=len(rendered.content),
        record_count=rendered.record_count,
        skipped_count=rendered.skipped_count,
        periods=sorted(set(rendered.periods)) if rendered.periods is not None else None,
        plugin=identity,
        mapping=resolved.mapping,
        mapping_sha256=json_digest(resolved.mapping),
        references=resolved.references,
        target=target,
        provenance=evidence,
    )
    return publish_manifest(directory, manifest, rendered.content)


def read_payload_metadata(path: Path) -> dict[str, Any] | None:
    """Return usable display metadata; delivery separately validates its manifest."""
    metadata_path = path.parent / ".export.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Could not read payload metadata '%s'; using result-file defaults", metadata_path)
        return None
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(field), str) or not metadata[field].strip()
        for field in ("filename", "format", "media_type")
    ):
        logger.warning("Invalid payload metadata fields in '%s'; using result-file defaults", metadata_path)
        return None
    media_type = metadata["media_type"]
    manifest = metadata.get("manifest")
    if any(ord(character) < 32 or ord(character) > 126 for character in media_type) or (
        "manifest" in metadata
        and (
            not isinstance(manifest, str)
            or not manifest.strip()
            or manifest in {".", ".."}
            or "/" in manifest
            or "\\" in manifest
        )
    ):
        logger.warning("Invalid payload metadata fields in '%s'; using result-file defaults", metadata_path)
        return None
    if metadata["filename"] != path.name:
        return None
    return metadata
