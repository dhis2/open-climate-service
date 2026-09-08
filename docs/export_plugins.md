# Export plugins and named mappings

Named exports render a computed aggregate using a mapping in instance configuration.
They work with synchronous `POST /result` and batch jobs. Rendering does not contact
DHIS2, resolve a credential, fetch organisation units, or run another aggregation.

## Render one series for DHIS2

Add an export to the file referenced by `CLIMATE_SERVICE_CONFIG`:

```yaml
exports:
  - id: rainfall-monthly
    plugin: dhis2
    period_type: monthly
    org_unit_field: geometry
    period_field: t
    series:
      - select: {}
        data_element: BXgDHhPdFVU
```

Replace the data-element UID with the destination defined in your DHIS2 instance.
Pass the prepared aggregate to `save_result`:

```json
{
  "process_id": "save_result",
  "arguments": {
    "data": {"from_node": "aggregate"},
    "format": "DHIS2JSON",
    "options": {"export": "rainfall-monthly"}
  },
  "result": true
}
```

This is a graph node; `aggregate` must be a preceding node in the full graph.
Named exports accept only the `export` option. Change the configured mapping to
change its destination or fields; per-request overrides are rejected. Existing
`DHIS2JSON` calls using `data_element_id`, `org_unit_field`, and `period_type`
continue to work without a named export.

Each named DHIS2 export currently selects one value series. `select: {}` requires
an unambiguous value column. To select a variable from a result with several
variables, use `select: {variable: precip}`. Additional dimensions such as
quantile or ensemble member must be selected or reduced upstream. Several datasets
can be published through separate single-series exports; no merged cube is required.

The series may also specify `category_option_combo` and `attribute_option_combo`.
All destination and organisation-unit IDs must have DHIS2 UID syntax. Positional
labels such as `0`, missing IDs, duplicate organisation-unit/period keys, and invalid
calendar periods are rejected. Zero values are preserved; missing observations are
omitted and counted. Non-finite and non-numeric values are rejected.

Supported output periods are daily, weekly (ISO), monthly, quarterly, and yearly.
`period_type` formats already prepared observations; it does not aggregate them.
Multiple dekads in the same month cannot be exported simply by labelling them
monthly, since they would produce duplicate destination keys. Explicit incompatible
`period_type` attributes are also rejected. Without execution provenance, the
renderer cannot prove that an arbitrary input value was aggregated correctly.

The optional `dataset`, `org_units`, and `connection` references, and the optional
`aggregation` declaration, describe intended input/delivery configuration. They
do not trigger any work. Batch exports bind these declarations to a manifest and
check an observed dataset reference when execution provenance contains one. A
connection is not required to render or download a payload, but a bound connection
is required for later server-side delivery. Use
[named connections](importing_to_dhis2.md#named-connections-for-server-side-plugins)
for that binding.

## Write a render-only plugin

Export plugins implement the public `BaseExportPlugin` contract and expose an
instance named `plugin` in each discoverable module:

```python
import json

from open_climate_service.exports import BaseExportPlugin, RenderedExport


class SummaryExport(BaseExportPlugin):
    id = "summary"
    format = "SUMMARYJSON"
    extension = ".json"
    media_type = "application/json"

    def validate_mapping(self, mapping):
        if mapping:
            raise ValueError("Summary export does not accept mapping fields")
        return {}

    def render(self, data, mapping):
        payload = {"variables": list(data.data_vars)}
        return RenderedExport(json.dumps(payload).encode(), record_count=len(data.data_vars))


plugin = SummaryExport()
```

Declare `exports: [{id: summary, plugin: summary}]` and use
`save_result(format="SUMMARYJSON", options={"export": "summary"})`.
The format is advertised in `GET /file_formats` with its `export` parameter.

`render` receives the computed result and the validated mapping. It returns bytes
and non-negative `record_count` / `skipped_count` values. The framework owns file
paths, persistence, and HTTP response metadata. Both methods must be pure: no
network calls, credential resolution, file writes, or delivery. Keep invocation
state local because plugin instances may be shared by concurrent jobs.

For an installed package, put the module in `<package>/exports/` and include
`__init__.py`. Use the same `open_climate_service.plugins` entry point as other
[installable plugins](installable_plugins.md). No separate export entry point is
needed. For an instance, put it in `plugins_dir/exports/`; relative helper imports
are supported. Prefix helper filenames with `_` so discovery skips them.

Precedence is built-in, then installed packages, then local instance plugins,
matched by plugin ID. Installed packages and module filenames are sorted for
deterministic loading. Broken modules fail explicitly rather than falling back to
a different renderer. Python modules are cached by Python's import machinery;
restart the service after changing plugin code.

Format identifiers use uppercase letters, digits, and underscores. File extensions
are a dot followed by lowercase letters or digits; Zarr directories are excluded.
Media types use `type/subtype` without parameters. Plugin code is operator-installed
Python code and must be trusted by the deployment.

## Saved results and delivery eligibility

Each batch render writes a fresh payload generation and a versioned JSON manifest,
then atomically replaces `.export.json` as the pointer to that generation. The
manifest records payload and mapping digests, renderer identity and version, record
counts and periods, the source job, public configuration references, a credential-free
target fingerprint, and execution provenance available from OCS processes. The
payload and manifest are both exposed as job result assets. Synchronous rendering
still returns the payload directly and remains available in read-only mode.

Execution provenance records observed managed artifacts, Icechunk snapshot IDs,
and hashes of inline spatial features where those inputs pass through native OCS
processes. The manifest explicitly lists evidence that is unavailable; declarations
alone do not prove aggregation semantics or per-output lineage.

The delivery-input validator accepts only completed jobs with intact payloads and
manifests whose mapping, plugin version, target, references, and process graph still
match. It holds a cross-process lease that prevents job update, rerun, or deletion
while a future delivery worker consumes the bytes. Older named-export assets remain
downloadable but are not eligible for automatic delivery.

Server-side sending, remote dry runs, import reports, retry/recovery, and multi-series
mappings remain subsequent work. Downloaded DHIS2 JSON can still be imported by the
existing client workflow.
