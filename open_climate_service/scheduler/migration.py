"""One-time migration of legacy YAML schedules to the shared schedules store."""

from __future__ import annotations

from pydantic import ValidationError

from open_climate_service import config as api_config
from open_climate_service.scheduler.config import DatasetSyncSchedule
from open_climate_service.scheduler.store import import_legacy_schedules, schedules_path


def migrate_legacy_schedules() -> int:
    """Import the legacy scheduler.dataset_sync list; never alter the YAML config."""
    if api_config.get_config_path() is None:
        raise ValueError("Set CLIMATE_SERVICE_CONFIG to the instance YAML file before migrating schedules")
    raw_scheduler = api_config.get_config().get("scheduler", {})
    if not isinstance(raw_scheduler, dict):
        raise ValueError("scheduler in CLIMATE_SERVICE_CONFIG must be a mapping")
    raw_entries = raw_scheduler.get("dataset_sync")
    if not isinstance(raw_entries, list):
        raise ValueError("No scheduler.dataset_sync list found in CLIMATE_SERVICE_CONFIG")
    try:
        legacy = [DatasetSyncSchedule.model_validate(item) for item in raw_entries]
    except ValidationError as exc:
        raise ValueError(f"Invalid scheduler.dataset_sync entry: {exc}") from exc
    ids = [item.dataset_id for item in legacy]
    if len(set(ids)) != len(ids):
        raise ValueError("scheduler.dataset_sync contains duplicate dataset ids; no schedules were changed")
    return import_legacy_schedules(legacy)


def main() -> None:
    """Import before deleting the YAML block and starting the upgraded server."""
    count = migrate_legacy_schedules()
    print(f"Imported {count} schedule(s) into {schedules_path()}.")
    print("Verify the saved schedules, remove scheduler.dataset_sync from the YAML, then restart OCS.")


if __name__ == "__main__":
    main()
