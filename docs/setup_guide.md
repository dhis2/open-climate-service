# Quick start

This is the fastest way to try Open Climate Service locally with the built-in datasets:
clone the repository, configure an extent, start the service, then ingest and view data
from the web interface.

!!! tip "Running a service for a country?"
    For an operational deployment — custom datasets, version pinning, configuration kept
    in its own repository — the [instance guide](instance_guide.md) is the recommended
    path. Most people running a real climate service should start there. This quick start
    is for evaluating the service or developing against the core repository.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git
- Make (`make` — available by default on macOS and most Linux distributions; on Windows use [WSL](https://learn.microsoft.com/en-us/windows/wsl/) or run the commands in the Makefile directly)

## Step 1: Clone and install

This quick start runs from a clone of the core repository, which is best for evaluating
the service or developing against it. To run an instance from the released PyPI package
instead — in its own repository, without cloning this one — follow the
[instance guide](instance_guide.md).

```bash
git clone https://github.com/dhis2/open-climate-service.git
cd open-climate-service
make sync
```

## Step 2: Configure the spatial extent

The repo includes `climate-service.yaml.example` with Sierra Leone as a starting point.
Copy it to `climate-service.yaml` (which is gitignored, so your local extent stays out of
version control) and replace the entry with your area of interest:

```bash
cp climate-service.yaml.example climate-service.yaml
```

```yaml
id: rwanda-climate-service
name: Rwanda Climate Service

extent:
  name: Rwanda
  bbox: [28.8, -2.9, 30.9, -1.0]    # [xmin, ymin, xmax, ymax] in WGS84
  country_code: RWA                   # ISO 3166-1 alpha-3, required for WorldPop

data_dir: ./data
```

`bbox` is required; `country_code` is needed for WorldPop downloads. To find a bounding
box, [bboxfinder.com](http://bboxfinder.com) is a useful tool. For the full list of
configuration fields, see the [instance guide](instance_guide.md#step-4-configure-the-instance).

## Step 3: Configure environment variables

```bash
cp .env.example .env
```

`CLIMATE_SERVICE_CONFIG=./climate-service.yaml` is already set, and the remaining defaults
are enough to run the API and ingest CHIRPS3 and WorldPop data. For ERA5-Land downloads,
see [ERA5-Land datasets](era5_land_datasets.md).

## Step 4: Start the service

```bash
make run
```

The service starts on `http://127.0.0.1:9000`. Open it in a browser — the overview reports what
the instance holds and links to the dataset templates you can ingest, the datasets you hold, and
the map viewer.

## Step 5: Ingest and view data

Open **Dataset templates** from the navigation and ingest your first dataset from its page.
CHIRPS3 (daily precipitation) requires no API key and is a good one to start with — open
it, enter a start date, and ingest (leave **Publish** checked). Progress streams live, and
the dataset's page opens when it finishes.

Then open **`http://127.0.0.1:9000/map`** to view it on the map, stepping through time with
the slider.

See [Using the web interface](web_interface.md) for a full walkthrough of the management
and map-viewer pages.

!!! note "Prefer the API?"
    Everything the interface does is also available over HTTP. To ingest, sync, discover,
    and open data programmatically, see [Accessing data](user_guide.md) and the
    [API reference](managed_data_api_guide.md).

## Next steps

- **[Instance guide](instance_guide.md)** — set up a versioned instance repository for your
  country, with custom datasets and deliberate upgrades (the recommended path for operators).
- **[Using the web interface](web_interface.md)** — manage ingestion, sync, and the map viewer.
- **[Accessing data](user_guide.md)** — discover and open datasets from Python or any tool.
- **[Adding custom datasets](adding_custom_datasets.md)** — add sources beyond the built-in templates.
