# AI assistants (MCP)

An instance can serve an [MCP](https://modelcontextprotocol.io) endpoint at `/mcp`, so an AI assistant can answer questions about its data, such as *"What was the rainfall in Bo District from May to August?"*, without the user knowing the API or writing a process graph.

MCP is an open standard. Any MCP client works: Claude, ChatGPT, GitHub Copilot, Gemini CLI, code editors, or a model running locally. The assistant chooses the dataset and the area; the instance computes every value, and each result includes the openEO process graph that produced it, so an answer can be checked and reproduced without the assistant.

This is a first version, off by default while it settles (CLIM-851).

## Enable it

Add to `climate-service.yaml`:

```yaml
mcp: true
```

The setting is read on every request, so no restart is needed. It works on [read-only instances](instance_guide.md#read-only-instances): every tool only reads.

## Connect an assistant

Give the client the instance's `/mcp` address:

- **A client on the same computer** (Claude Code, an editor, a local model) can use `http://localhost:<port>/mcp`. With Claude Code:

    ```bash
    claude mcp add --transport http ocs http://localhost:9000/mcp
    ```

- **A cloud client** (claude.ai, ChatGPT) connects from the provider's servers, not from your computer, so it needs the instance at a public HTTPS address. An instance on a laptop needs a tunnel or a hosted deployment for these.

The endpoint has no authentication, like the rest of the API.

## What it offers

| Tool | What it does |
|---|---|
| `list_datasets` | The published datasets: rasters with their variable, unit and time range, and area collections |
| `list_areas` | The areas in an area collection (districts, org units), by name and id |
| `get_timeseries` | Values of a raster at one location over time |
| `get_zonal_statistics` | A statistic (mean, sum, min, max or median) of a raster over each area, per period. Area-weighted |

The catalogue is also available as the resource `ocs://catalogue`.

To answer a question about a place, the assistant lists the areas of a collection, picks one by name and passes its id to `get_zonal_statistics`. An instance needs an area collection for that, such as [Overture Maps divisions](built_in_datasets.md#overture-maps-administrative-divisions-feature-collection) or [DHIS2 org units](built_in_datasets.md#dhis2-organisation-units-feature-collection).

Results are tables (`period`, `area_id`, `area_name`, `value`) with the unit, the process graph, and links to the dataset and to it in the instance's map viewer. `list_datasets` also says when each dataset was last updated. A result is limited to 2,000 rows; a larger request is refused with a hint to narrow the period or the areas.
