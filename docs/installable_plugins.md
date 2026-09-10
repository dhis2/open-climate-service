# Installable plugins

There are two ways to add datasets, processes, workflows, and export renderers to an instance, and they
complement each other:

- **`plugins_dir`** — drop files into the instance's local plugins folder. Ideal for
  instance-specific customisation, one-off datasets, or overriding a built-in. No packaging.
  See [Extensibility](extensibility.md) and [Adding custom datasets](adding_custom_datasets.md).
- **Installable packages** — package a plugin and publish it, so any instance can `uv add` it
  and OCS discovers it automatically ([#118](https://github.com/dhis2/open-climate-service/issues/118)).
  Ideal for a **reusable** plugin shared across countries or organisations, versioned independently.

This guide covers the installable-package option. It is purely additive — `plugins_dir`
keeps working exactly as before, and still takes precedence (see [Precedence](#precedence)).

## Package layout

An importable package can ship any combination of these extension points — **datasets,
processes, workflows, and exports are auto-discovered** when the package is installed. `datasets/`,
`processes/`, and `exports/` hold importable Python, so each needs an `__init__.py` (`workflows/` is plain
JSON and does not):

```
osc_example_plugin/
  __init__.py
  datasets/
    __init__.py
    example.py           # your BaseDatasetPlugin subclass
    example.yaml         # dataset templates
  processes/             # optional: @process-decorated callables
    __init__.py
    my_process.py
  workflows/             # optional: openEO UDP JSON graphs
    my_workflow.json
  exports/               # optional: pure export renderers
    __init__.py
    my_export.py         # exposes plugin = BaseExportPlugin subclass instance
```

The layout mirrors `plugins_dir`, so migrating a `plugins_dir`-based plugin to a distributable
package is mostly moving the files into a package and adding the entry point below.

## Declare the entry point

In the package's `pyproject.toml`, point an [entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
in the `open_climate_service.plugins` group at the top-level package:

```toml
[project.entry-points."open_climate_service.plugins"]
example = "osc_example_plugin"
```

Declare `open-climate-service` as a dependency, but **do not** pin its VCS source — the consuming
instance decides which OCS revision to run:

```toml
dependencies = ["open-climate-service"]
```

The `ingestion.plugin` in a dataset template uses the class's **full dotted path** (not a
`plugins_dir`-relative one), since the package is installed on `PYTHONPATH`:

```yaml
ingestion:
  plugin: osc_example_plugin.datasets.example.ExamplePlugin
```

## Install and discover

An operator installs the package — nothing else, no `plugins_dir` wiring:

```bash
uv add osc-example-plugin
```

OCS auto-discovers every installed package in the `open_climate_service.plugins` group and loads
its `datasets/*.yaml` templates, its `processes/` (`@process`-decorated callables), its
`workflows/*.json` (openEO UDPs), and its `exports/` renderers (see [Export plugins](export_plugins.md)).
The ingestion plugin class is importable by dotted path because
the package is installed. The datasets then appear in `/datasets` and can be ingested like any
built-in.

## Precedence

Templates are merged in increasing order of precedence, per extension point:

**built-in → installed plugins → instance `plugins_dir`**

So `plugins_dir` always wins on an id conflict — an operator can drop a YAML into their local
`plugins/datasets/` to override an installed plugin's dataset. Overrides are logged at load time.

## Naming convention

- **Distribution name:** `osc-<name>-plugin`
- **Import package:** `osc_<name>_plugin`

For example, `osc-senorge-plugin` / `osc_senorge_plugin`.

## Reference implementation

The [seNorge plugin](https://github.com/MasterMaps/osc-senorge-plugin) is the reference
implementation: it ships the seNorge 2018 datasets (source + derived) and the ingestion plugin, and
is consumed by the Norway instance via `uv add`.
