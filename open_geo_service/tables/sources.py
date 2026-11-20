"""The source, licence and providers of a computed table dataset, from the datasets it read (CLIM-1555).

A computed table has no template. Its record names the datasets its latest rows were computed
from (`ingestion_services.recorded_inputs`), and their templates declare the source, licence and
providers the table needs, so the table's are derived from theirs. An input whose template
declares no licence, such as a workflow output without one, is followed back to the datasets it
was computed from, as `via` links are.

Every input counts, the feature collection the values are reported on among them: a row carries
its features' ids and names, so the boundaries' licence applies to the table as much as the
raster's does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from open_geo_service.ingestions.schemas import ArtifactRecord
from open_geo_service.shared.licences import STAC_LICENSE_OTHER, DatasetLicence, declared_providers, parse_licence


@dataclass(frozen=True)
class TableSources:
    """The templates of the datasets a table was computed from, in the order its job read them."""

    templates: tuple[dict[str, Any], ...]

    @property
    def licences(self) -> list[DatasetLicence]:
        """Each distinct licence of the sources, an undeclared one included."""
        licences: list[DatasetLicence] = []
        for template in self.templates:
            licence = parse_licence(template.get("license"))
            if licence not in licences:
                licences.append(licence)
        return licences

    @property
    def license(self) -> str:
        """One SPDX identifier, an SPDX expression joining several with AND, or `other`, for `/datasets`.

        Values computed from several sources are bound by every source's licence, which an SPDX
        expression says. A source whose licence has no SPDX identifier, or declares none, makes it
        `other`.
        """
        licences = self.licences
        identifiers = [licence.identifier for licence in licences]
        if not identifiers or None in identifiers:
            return STAC_LICENSE_OTHER
        return " AND ".join(str(identifier) for identifier in identifiers)

    @property
    def stac_license(self) -> str:
        """The STAC `license`: the one licence's, or `other` for several, with a link to each.

        STAC's schema and Portolan accept an SPDX identifier or `other`, not an expression.
        """
        licences = self.licences
        return licences[0].stac_license if len(licences) == 1 else STAC_LICENSE_OTHER

    def license_links(self) -> list[tuple[str, str]]:
        """Each licence's page and title, for the STAC `license` links: its own URL, or its SPDX page.

        With several licences, `license` says only `other`, so an SPDX licence links its page too.
        """
        several = len(self.licences) > 1
        links: list[tuple[str, str]] = []
        for licence in self.licences:
            page = licence.url or (
                f"https://spdx.org/licenses/{licence.identifier}.html" if several and licence.identifier else None
            )
            if page:
                links.append((page, licence.name or licence.identifier or "Licence"))
        return links

    @property
    def license_url(self) -> str | None:
        """The licence page, when the sources share one licence that has one."""
        licences = self.licences
        return licences[0].url if len(licences) == 1 else None

    @property
    def label(self) -> str:
        """The licences as a reader names them, or "" when no source declares one."""
        licences = self.licences
        if all(not licence.identifier and not licence.name for licence in licences):
            return ""
        if self.license != STAC_LICENSE_OTHER:
            return self.license
        return ", ".join(licence.label for licence in licences)

    @property
    def providers(self) -> list[dict[str, Any]]:
        """The providers the sources declare, each once, by name."""
        providers: list[dict[str, Any]] = []
        for template in self.templates:
            for provider in declared_providers(template):
                if all(provider["name"] != seen["name"] for seen in providers):
                    providers.append(provider)
        return providers

    @property
    def source(self) -> str | None:
        """The sources' own names, joined, for the listing's `source`."""
        from open_geo_service.ingestions.services import _template_source

        names: list[str] = []
        for template in self.templates:
            name = _template_source(template)
            if name is not None and name not in names:
                names.append(name)
        return ", ".join(names) or None

    def description(self, record: ArtifactRecord) -> str:
        """What the table holds and what it was computed from, for a table without a template's own prose.

        What it holds comes from what its columns represent (`TableSchema.holds`).
        """
        from open_geo_service.tables.store import schema_of

        names = [str(template.get("name") or template["id"]) for template in self.templates if template.get("id")]
        computed = ""
        if names:
            joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
            computed = f", computed from {joined}"
        return f"{record.dataset_name}: {schema_of(record).holds(record.period_type)}{computed}."

    def template(self) -> dict[str, Any]:
        """The table's metadata in a template's shape, for the pages that read one.

        Several licences are named together, as one label: no single declaration describes them.
        """
        template: dict[str, Any] = {"providers": self.providers}
        if self.source is not None:
            template["source"] = self.source
        if self.label:
            template["license"] = {"name": self.label, **({"url": self.license_url} if self.license_url else {})}
        return template


def table_sources(record: ArtifactRecord) -> TableSources:
    """The templates of the datasets *record*'s rows were computed from.

    A record without provenance, rows written directly rather than by a job, has no sources.
    """
    from open_geo_service.data_registry.services import datasets as registry_datasets
    from open_geo_service.features import templates as feature_templates
    from open_geo_service.ingestions import services as ingestion_services

    templates: list[dict[str, Any]] = []
    seen: set[str] = set()
    published: dict[str, ArtifactRecord] | None = None

    def template_for(dataset_id: str | None) -> dict[str, Any]:
        if not dataset_id:
            return {}
        return registry_datasets.get_dataset(dataset_id) or feature_templates.get_feature_template(dataset_id) or {}

    def visit(current: ArtifactRecord) -> None:
        nonlocal published
        for recorded in ingestion_services.recorded_inputs(current):
            input_id = str(recorded["collection_id"])
            if input_id in seen:
                continue
            seen.add(input_id)
            template = template_for(input_id) or template_for(recorded.get("source_dataset_id"))
            if template.get("license") is None:
                if published is None:
                    published = ingestion_services.stac_eligible_artifacts_by_dataset()
                upstream = published.get(input_id)
                if upstream is not None and ingestion_services.recorded_inputs(upstream):
                    visit(upstream)
                    continue
            templates.append(template)

    visit(record)
    return TableSources(templates=tuple(templates))
