"""Aggregate a climate dataset to DHIS2 organisation units and import it into DHIS2.

End-to-end pipeline driven from Python:

  1. Fetch organisation unit boundaries from DHIS2 as GeoJSON.
  2. Run the built-in ``aggregate_to_dhis2_json`` workflow on an Open Climate
     Service instance — it loads a dataset, aggregates it within each org-unit
     polygon, and returns a ready-to-import DHIS2 ``dataValueSet`` rendered by a
     named export.
  3. Import that ``dataValueSet`` back into DHIS2.

The workflow uses each org unit's GeoJSON ``id`` (its DHIS2 UID) as the ``orgUnit``,
so the result imports into DHIS2 without any remapping. The data element and period
type come from a named export in the instance configuration, for example:

  exports:
    - id: temperature-monthly
      plugin: dhis2
      dataset: era5land_temperature_monthly
      aggregation: mean
      period_type: monthly
      series:
        - select: {}
          data_element: BXgDHhPdFVU

Mirrors the DHIS2 Climate Tools guides "Fetch org units from DHIS2 Web API" and
"Importing data values", combined into a single Open Climate Service workflow.

Requires:
  pip install open-climate-service "dhis2-client @ git+https://github.com/dhis2/dhis2-python-client.git"
  A running Open Climate Service instance with the dataset published and the
  named export above configured.
  A DHIS2 instance whose org units have geometry, plus a target data element.

Adjust the configuration constants below for your instances.
"""

from __future__ import annotations

from dhis2_client import DHIS2Client
from dhis2_client.settings import ClientSettings

from open_climate_service import ClimateService

# --- Open Climate Service ---------------------------------------------------
OCS_BASE_URL = "http://127.0.0.1:9000"
DATASET_ID = "era5land_temperature_monthly"  # a published collection (see /datasets)
TEMPORAL_EXTENT = ["2025-01-01", "2025-12-31"]
EXPORT_ID = "temperature-monthly"  # named export in the instance configuration
METHOD = "mean"  # mean (default), min, max, sum, or median; must match the export's aggregation

# --- DHIS2 ------------------------------------------------------------------
# `admin` / `district` are the public DHIS2 demo credentials, shown here for
# illustration — replace the URL and credentials with your own instance.
DHIS2_BASE_URL = "https://climate.im.dhis2.org/climate-tools-42"
DHIS2_USERNAME = "admin"
DHIS2_PASSWORD = "district"
ORG_UNIT_LEVEL = 2  # org unit level to aggregate to


def main() -> None:
    """Fetch org units from DHIS2, aggregate them on Open Climate Service, import back."""
    dhis2 = DHIS2Client(
        settings=ClientSettings(
            base_url=DHIS2_BASE_URL,
            username=DHIS2_USERNAME,
            password=DHIS2_PASSWORD,
        )
    )

    # 1. Fetch org unit boundaries from DHIS2 as GeoJSON.
    #    Each feature's `id` is the DHIS2 org unit UID, which the workflow uses as the orgUnit.
    org_units = dhis2.get_org_units_geojson(level=ORG_UNIT_LEVEL)
    print(f"Fetched {len(org_units['features'])} org units from DHIS2 (level {ORG_UNIT_LEVEL})")

    # 2. Run the aggregation workflow on Open Climate Service -> DHIS2 dataValueSet.
    service = ClimateService(OCS_BASE_URL)
    data_value_set = service.execute(
        {
            "agg": {
                "process_id": "aggregate_to_dhis2_json",
                "arguments": {
                    "dataset_id": DATASET_ID,
                    "temporal_extent": TEMPORAL_EXTENT,
                    "geometries": org_units,
                    "export": EXPORT_ID,
                    "method": METHOD,
                },
                "result": True,
            }
        }
    )
    values = data_value_set.get("dataValues", [])
    print(f"Workflow produced {len(values)} data values through export {EXPORT_ID}")
    if not values:
        print("No data values produced — check the dataset id, temporal extent, and org unit geometries.")
        return
    print(f"  e.g. {values[0]}")

    # 3. Import the dataValueSet into DHIS2.
    report = dhis2.post_data_value_set(data_value_set)
    import_count = report.get("response", {}).get("importCount", report)
    print("DHIS2 import summary:", import_count)


if __name__ == "__main__":
    main()
