# Export plugins and named mappings

A named export renders a computed aggregate through an export plugin, using a mapping. A
[pipeline](web_interface.md#the-pipeline-pages-pipelines) owns one under its own id: its
destination plugin compiles the destination settings into the mapping each time the pipeline
validates. Named exports work with synchronous `POST /result` and batch jobs. Rendering does not
contact DHIS2, resolve a credential, fetch organisation units, or run another aggregation.

Rendering and delivery are separate. Any process graph can render through a pipeline's named
export by passing the pipeline id to `save_result`; that borrows the validated destination mapping
but does not run the pipeline's workflow or send anything to the external service. The graph's
observed source must match the dataset declared by the pipeline. A client can import the rendered
payload itself, while server-side delivery always goes through the pipeline's export endpoint.
For a one-off payload without a pipeline, pass an ad-hoc mapping to `save_result`; OCS can render
and return it, but cannot deliver it.

## Render one series for DHIS2

A pipeline delivering to DHIS2 compiles a mapping like this one:

```yaml
id: rainfall-monthly          # the pipeline's id
plugin: dhis2
dataset: era5land_precipitation_monthly
connection: national-hmis
period_type: monthly          # the dataset's own cadence
series:
  - select: {}
    data_element: BXgDHhPdFVU
    category_option_combo: FemaleCOC01   # optional
```

The pipeline's DHIS2 settings fill the connection, and per series the data element and the
optional category and attribute option combos. The dataset and the period come from the
workflow the pipeline binds. The rest of this section describes the mapping the DHIS2 renderer
accepts; fields no pipeline setting fills (`org_unit_field`, `period_field`,
`temporal_aggregation`, `incomplete_periods`, `aggregation` and a `select` naming a variable)
keep their defaults.

`org_unit_field: feature_id`, the default, reads each value's organisation unit from the
feature ids a spatial aggregation keeps beside its geometries. CHAP CSV's `location_field`
defaults to `feature_id` too. Geometries are never used as ids: a result that has shapes but
no `feature_id` is refused, so give each GeoJSON Feature an `id`.
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
The built-in `aggregate_to_dhis2_json` workflow wraps this aggregation and
`save_result` call, taking the export ID as its `export` parameter. See
[Importing data to DHIS2](importing_to_dhis2.md#deliver-one-batch-job).
Named exports accept only the `export` option. Edit the pipeline's destination to
change its target or fields; per-request overrides are rejected. Hand-written
`DHIS2JSON` graphs using `data_element_id`, `org_unit_field`, and `period_type`
still render an ad-hoc payload without a named export, but that payload cannot be
delivered by the server.

Each series mapping selects one value series. `select: {}` requires
an unambiguous value column. To select a variable from a result with several
variables, use `select: {variable: precip}`. Additional dimensions such as
quantile can be selected with `select: {variable: precip, quantile: 0.1}`;
other dimensions, such as ensemble member, must be reduced upstream. Several datasets
can be published through separate single-series exports; no merged cube is required.

The series may also specify `category_option_combo` and `attribute_option_combo`.
When several series target the same data element, specify each combo consistently
on every series or omit it on every series. Mixing implicit and explicit defaults
is rejected because the renderer cannot resolve target metadata without a network
request. Explicit duplicate destination keys are always rejected.
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
`aggregation` declaration (`mean`, `sum`, `min`, `max`, or `median`), describe
intended input/delivery configuration. They do not trigger any work. Synchronous
and batch renders check them against execution provenance where it contains an
observation, and batch exports bind them to a manifest. The dataset must match an
observed source. The aggregation is checked when exactly one spatial aggregation 
process ran — either the built-in `aggregate_spatial` with a `reduce_by_method` reducer, 
or the custom `aggregate_spatial_weighted` with a parsed reducer string, as the built-in 
workflow does. 
With no spatial aggregation, several, or another reducer, the aggregation cannot be
attributed to the result; it remains an unverified declaration and the manifest
lists `spatial_aggregation_method` as missing. A
connection is not required to render or download a payload, but a bound connection
is required for later server-side delivery. Use
[named connections](importing_to_dhis2.md#named-connections-for-server-side-plugins)
for that binding.

### The export's period must be reachable from the dataset's cadence

An export exports what exists. `period_type` labels; it never aggregates, and neither does
anything else on the export path. So the export has to say how its period relates to the
dataset it declares, and OCS checks it twice: when the pipeline validates, from the dataset
template's `period_type`, and at render time, from the spacing of the data actually being
exported.

| Dataset cadence | Export `period_type` | Outcome |
| --- | --- | --- |
| the same | the same | Pass-through. No `temporal_aggregation`; declaring one is refused. |
| finer, and tiling the period | coarser | Exportable only after an explicit aggregation. `temporal_aggregation: sum \| mean \| min \| max` declares which, and the export verifies that it ran. |
| anything else | | Refused, with the reason. |

Producing the coarser data is openEO's job, not the export's. Publish a derived dataset
(`load_collection`, `aggregate_temporal_period`, `save_result` as Zarr with a `dataset_id`) and
point the pipeline at that dataset, which then needs no declaration at all; a
[recipe](adding_custom_datasets.md#keep-a-derived-dataset-current-with-a-recipe) keeps it
current. A pipeline's mapping has no `temporal_aggregation`, so the built-in org-unit workflows,
given a dataset finer than the export's period, are refused.

A derived dataset is published by an openEO job, and that job records a `dataset.updated`
event when the publication succeeds, so a pipeline listens for it the way it listens for a
sync. See [Updates from derived datasets](workflow_automation.md#updates-from-derived-datasets).

The pairs currently supported by `aggregate_temporal_period`: hourly into daily, weekly,
monthly and yearly; daily into weekly, monthly and yearly; dekadal into monthly and yearly;
monthly into yearly; and quarterly into yearly. Weekly data tiles nothing, because ISO weeks
straddle months, quarters and years. Calendar-quarter destinations are refused for now: they
tile arithmetically, but the standard process has no calendar-quarter period with which to
produce and record them. Nothing can be made finer than it is stored.

`aggregate_dekads(period="week")` is a separate, day-overlap-weighted transformation rather
than a tiling aggregation. Its weekly result can be exported ad hoc, or published as a derived
weekly dataset and then delivered by a pipeline on it.

When the graph aggregates, a destination period counts only when every source period inside
it is present. A sync that ends on the 14th does not produce that month: the job fails naming
the period, unless the export says `incomplete_periods: drop`, in which case the period is
left out and the rest is delivered. Execution provenance records each temporal aggregation
(its period, reducer and incomplete periods), so a declared `temporal_aggregation` is verified
against what ran, the way the spatial `aggregation` is. A derived dataset carries no such
record; its completeness is the derivation's concern.

A hand-written graph gets the same guard at `save_result`: a `period_type` coarser than the
result's spacing is refused with the `aggregate_temporal_period` step to add, instead of
collapsing several values onto one DHIS2 key.

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

A pipeline whose destination is the plugin owns a named export of it (see
[Act as a pipeline destination](#act-as-a-pipeline-destination)); a graph then uses
`save_result(format="SUMMARYJSON", options={"export": "<pipeline id>"})`. Without a pipeline,
a hand-written graph uses the format with its ad-hoc options. The format is advertised in
`GET /file_formats` with its `export` parameter.

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

## Act as a pipeline destination

A [pipeline](web_interface.md#the-pipeline-pages-pipelines) delivers through an export
plugin, and never interprets the destination's settings itself. A plugin becomes a pipeline
destination by overriding these optional hooks of `BaseExportPlugin`:

| Hook | Purpose |
|---|---|
| `destination_settings_schema()` | JSON Schema of the settings; drives the pipeline form. Returning `None` (the default) means "not a destination". |
| `pipeline_settings_form()` | Optional structured presentation metadata for the plugin's pipeline settings: display title and description, sections, field order, labels, help text and widget hints. Returning `None` keeps schema-derived defaults. |
| `lookup_options(lookup, settings, query, row)` | The choices the destination offers for a `lookup` field as the form types, from the settings entered so far. For a column of a `rows` field, `row` holds the values entered in the asking row, so a choice can follow its neighbour (DHIS2 offers the chosen data element's category option combos). A plugin whose hook takes no `row` is searched without it. May contact the destination; raise `ValueError` naming what is missing. |
| `lookup_labels(lookup, settings, values)` | How stored lookup values read at the destination, by value. Called at validation; the labels are kept with the result so pages never wait for the destination. |
| `export_definition(settings, context)` | Compile the settings into an export definition (mapping plus `dataset`/`connection` references), resolved as the pipeline's named export. Pure; raise `ValueError` for bad settings. |
| `check_destination(settings, context)` | Destination-side checks run on save, re-check and going live, returned as `DestinationCheck`s. May contact the destination. |
| `preview(content, settings, context, labels, limit)` | A few rendered records for the dry-run summary. A cell may be `{"value", "label"}` so an identifier shows by name. `labels` are the names `lookup_labels` resolved at the last validation, so the destination need not be asked again: the DHIS2 plugin names data elements from them, asks DHIS2 only for any missing, and names organisation units from the bound feature collection's id and name columns. |
| `target_binding(references)` | Credential-free identity of the target the export's references name. Recorded in each saved export's manifest and compared before delivery, so a moved target is refused. Default: the `connection` reference as a named instance connection, `{connection_id, url_sha256}`. |
| `delivery_target(binding)` | What `send` receives for that binding, for both a delivery and a pipeline dry run. Default: the named connection's id. |
| `redelivery_safe` | `True` when sending the same payload twice leaves the destination as sending it once (an upsert). Only then may an operator send a pipeline delivery again after an unconfirmed outcome. DHIS2 sets it. |
| `supports_dry_run` | `True` when `send(dry_run=True)` has the destination validate without storing. Without it a pipeline dry run only renders. |

The target hooks apply to every delivery, not only pipelines. The defaults keep the contract
existing delivery plugins rely on, including DHIS2: `send` receives a named connection's id. A
plugin delivering anywhere else, a database or a file store, overrides both hooks, as the
built-in `table` destination does for this instance's own table store.

`context` is a `DestinationContext`: the pipeline's export id, the datasets its workflow binds
(id, variable, cadence, units) and the feature collections it binds. The built-in DHIS2 plugin
implements all of them in `open_climate_service/exports/dhis2_destination.py`; the `table`
destination, which keeps the result as a table dataset, in `open_climate_service/exports/table.py`.

### Configure the plugin's pipeline settings

JSON Schema remains authoritative for property names, types, required values, enums and
validation. `pipeline_settings_form()` only controls presentation. It returns
`PipelineSettingsForm` with ordered `SettingsFormSection`s and optional `SettingsFormField`
overrides:

```python
from open_climate_service.exports import (
    PipelineSettingsForm,
    SettingsFormField,
    SettingsFormSection,
)


def pipeline_settings_form(self):
    return PipelineSettingsForm(
        title="Object storage",
        description="Write each completed result to a configured bucket.",
        sections=(
            SettingsFormSection(
                title="Target",
                description="Where the rendered result is stored.",
                fields=("bucket", "prefix"),
            ),
        ),
        fields=(
            SettingsFormField(name="bucket", label="Bucket", widget="select"),
            SettingsFormField(name="prefix", label="Key prefix", help_text="No leading slash."),
        ),
    )
```

Top-level field names refer to properties in the settings schema. A dotted name such as
`series.data_element` customises a column of an array-of-objects field. Supported widget hints are
`text`, `select`, `checkbox`, `textarea`, `json`, `rows` and `lookup`. A hint is used only
when it is compatible with the property's schema shape; unknown or incompatible hints fall
back to the safe schema-derived control. Schema properties omitted from the declared sections
are appended under **Additional settings**, so presentation metadata cannot hide a new setting.
Properties explicitly marked `deprecated: true` remain valid for stored configuration but are
not offered for new input.

Choices remain part of the schema's `enum`. A plugin can derive that enum from instance
configuration each time `destination_settings_schema()` is called, as the built-in DHIS2
plugin does for named connections. The schema and form hooks must not contact the destination.
Use `option_labels` on `SettingsFormField` when enum values need friendlier labels.

### Choose a setting from the destination

A string setting whose values live at the destination, such as a DHIS2 data element, is declared
with `widget="lookup"` and a `lookup` name of the plugin's choosing:

```python
SettingsFormField(
    name="series.data_element",
    label="Data element",
    widget="lookup",
    lookup="data_elements",
)
```

The form then searches the destination as the operator types, through
`lookup_options(lookup, settings, query)`. `settings` are the form's values so far, read the
same way a save reads them but neither complete nor validated, so the plugin takes from them
only what the search needs: the built-in DHIS2 plugin reads the connection, and the data set
when one is chosen, to narrow the data elements to that set. Each `LookupOption` carries the
`value` that is stored, the `label` the operator sees and an optional `detail` such as a period
or value type. The value stays typeable by hand, and without JavaScript the field is a plain
text input, so a pasted id keeps working.

When a pipeline is validated the core calls `lookup_labels(lookup, settings, values)` for every
lookup field with a stored value and keeps the answer on the validation result, by field path
(`data_set`, `series.data_element`) and value. The pipeline page and the edit form show those
labels beside the stored values and never contact the destination to render. A lookup that
fails to resolve is logged and left out; it is never a check.

Destination settings are persisted in pipeline records and returned by the API. They must not
contain passwords, tokens or other credentials. Store credentials through the instance's
connection configuration and expose only a connection reference in the destination schema.

## Saved results and delivery eligibility

Each batch render writes a fresh payload generation and a versioned JSON manifest,
then atomically replaces `.export.json` as the pointer to that generation. The
manifest records payload and mapping digests, renderer identity and version, record
counts and periods, the source job, public configuration references, a credential-free
target fingerprint, and execution provenance available from OCS processes. The
payload and manifest are both exposed as job result assets. Synchronous rendering
still returns the payload directly and remains available in read-only mode.

Execution provenance records observed managed artifacts, Icechunk snapshot IDs, hashes of 
inline spatial features, and the method of each spatial aggregation call — both built-in 
`aggregate_spatial` and custom `aggregate_spatial_weighted` — where those inputs pass through 
native OCS processes. The manifest explicitly lists evidence that is unavailable; declarations
alone do not prove aggregation semantics or per-output lineage.

The delivery-input validator accepts only completed jobs with intact payloads and
manifests whose mapping, plugin version, target, references, and process graph still
match. It holds a cross-process lease that prevents job update, rerun, or deletion
while a delivery worker consumes the bytes. Older named-export assets remain
downloadable but are not eligible for automatic delivery.

## Deliver a saved export

A pipeline delivers its own runs. To deliver another completed job saved through a pipeline's
export, submit it:

```http
POST /exports/rainfall-monthly
Idempotency-Key: rainfall-job-123-validation
Content-Type: application/json

{"job_id": "job-123", "dry_run": true}
```

The response is `202 Accepted` with a delivery job ID and status/report URL. The
source job also links to that delivery. Inspect `report.outcome`, which distinguishes
success, dry-run validation, partial imports, rejection, cancellation and unknown
remote outcomes. A dry run can be rejected; its counts describe validation rather
than saved writes. Use a new idempotency key to request the actual import with
`dry_run: false`. Reports are scoped to `/exports/{export_id}/jobs/{delivery_job_id}`.

Reservations are persisted before jobs are enqueued. Repeating a key returns the
original job; changing its source, manifest or mode is a conflict. After a crash
between reservation and job creation, repeat the request to create the reserved job.
The worker checks the exact manifest captured at submission, so rerendering the
source while delivery is queued requires a new submission. Old queued delivery
jobs without that binding fail before sending and must be submitted again.

DHIS2 payloads are split into deterministic chunks of at most 1,000 values. Chunk
intent is saved before POST; completed chunks are reused on recovery. An uncertain
POST without a task ID is reported as unknown and is never automatically resent.
Known async tasks are polled through the DHIS2 `DATAVALUE_IMPORT` task summaries
endpoint. Corrupt or incompatible checkpoints stop recovery. Imports for the same
export are serialized; different exports may still overlap in their destination
keys, so operators must coordinate those mappings. `submitted` counts attempted
values, including uncertain writes, rather than the full planned payload.

Delivery and reports require a writable instance and are closed in read-only mode.
Writable deployments must put these endpoints behind an operator access boundary;
OCS does not yet provide per-user authorization. Downloaded JSON remains usable
through the existing client workflow.

## Map multiple series

The renderer takes one entry per output series; a merged raster cube is not required. A
pipeline's series rows all use `select: {}`, so a pipeline delivers one value series to several
data elements or disaggregations; selecting by variable, as below, has no pipeline setting yet.

```yaml
series:
  - select: {variable: temperature}
    data_element: TEMP0000001
  - select: {variable: precipitation}
    data_element: PREC0000001
```

Replace these example UIDs with target metadata. Wide DataFrames, multi-variable
xarray aggregates and merged aggregate cubes are supported. Zero is retained and
missing values are counted separately per series. Named DHIS2 graphs, including
workflow calls, check that original GeoJSON feature IDs are unique DHIS2 UIDs before
spatial aggregation. The renderer also rejects invalid organisation-unit UIDs and
duplicate destination keys.
