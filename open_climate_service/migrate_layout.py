"""Move an instance onto the rasters/vectors folder layout (CLIM-1253).

Run once per instance, with its server stopped, from the directory its ``.env`` is in:

    python -m open_climate_service.migrate_layout [--dry-run]

It renames, in the data directory, ``downloads/`` to ``rasters/`` and ``features/`` to
``vectors/``, and rewrites the store paths in ``artifacts/records.json`` to match. Under
``plugins_dir`` it renames ``datasets/`` to ``rasters/`` and ``features/`` to ``vectors/``,
and rewrites the ``ingestion.plugin`` paths and plugin imports that name the old folders.

OCS reads only the new names, so an instance that has not been migrated starts with no
data sources and no stores. Running it again once done changes nothing. It refuses to run
when an old and a new folder both exist, rather than merging them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import open_climate_service.startup  # noqa: F401  # pyright: ignore[reportUnusedImport]  # loads .env
from open_climate_service import config as api_config

DATA_RENAMES = (("downloads", "rasters"), ("features", "vectors"))
PLUGIN_RENAMES = (("datasets", "rasters"), ("features", "vectors"))

_PATH_FIELDS = ("path", "asset_paths")


class MigrationError(RuntimeError):
    """The layout cannot be migrated without a decision this command will not make."""


@dataclass
class Plan:
    """What a migration does, so a dry run can report it and a real run can carry it out."""

    renames: list[tuple[Path, Path]] = field(default_factory=list)
    records: tuple[Path, list[Any]] | None = None
    rewrites: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.renames and self.records is None and not self.rewrites


def _plan_rename(parent: Path, old: str, new: str, plan: Plan) -> None:
    source, target = parent / old, parent / new
    if not source.exists():
        return
    if target.exists():
        raise MigrationError(f"both {source} and {target} exist; move their contents by hand, then run again")
    plan.renames.append((source, target))


def _migrated_path(raw: str, root: Path) -> str:
    """The same store path with its first folder renamed, relative or absolute under root."""
    mapping = dict(DATA_RENAMES)
    # Absolute is decided by the native path, so a Windows record (`C:\\data\\downloads\\...`)
    # is recognised on Windows; relative records are always written with forward slashes.
    if not Path(raw).is_absolute():
        parts = PurePosixPath(raw).parts
        if parts and parts[0] in mapping:
            return PurePosixPath(mapping[parts[0]], *parts[1:]).as_posix()
        return raw
    candidate = Path(raw)
    try:
        # Resolved, as root is, so a data directory reached through a symlink still matches.
        relative = candidate.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        parts = relative.parts
        if parts and parts[0] in mapping:
            return str(root.joinpath(mapping[parts[0]], *parts[1:]))
        return raw
    # Recorded under another mount, such as a container's /app/data: the store is ours when
    # the same suffix exists under this data root, as store-path rebasing would find it.
    parts = candidate.parts
    for index in range(1, len(parts) - 1):
        if parts[index] in mapping and root.joinpath(*parts[index:]).exists():
            return PurePosixPath(mapping[parts[index]], *parts[index + 1 :]).as_posix()
    return raw  # a store deliberately kept outside the data directory


def _plan_records(data_root: Path, plan: Plan) -> None:
    index = data_root / "artifacts" / "records.json"
    if not index.is_file():
        return
    records = json.loads(index.read_text(encoding="utf-8"))
    changed = False
    for record in records:
        for key in _PATH_FIELDS:
            value = record.get(key)
            if isinstance(value, str):
                new = _migrated_path(value, data_root)
                changed |= new != value
                record[key] = new
            elif isinstance(value, list):
                new_list = [_migrated_path(item, data_root) if isinstance(item, str) else item for item in value]
                changed |= new_list != value
                record[key] = new_list
    if changed:
        plan.records = (index, records)


# In a dataset template, the old folder as the first component of the plugin path or after the
# built-in package: `plugin: datasets.x.Class`, `plugin: open_climate_service.plugins.datasets.x.Class`.
# In a plugin module, only imports of the built-in folders (`from open_climate_service.plugins.datasets.x
# import y`): no plugin imports a sibling by the bare folder name, and a bare `datasets` or `features`
# is as likely another package. Nothing else is touched: the words are common.
_YAML_PLUGIN = re.compile(
    r"^(\s*plugin:\s*['\"]?)((?:open_climate_service\.plugins\.)?)(datasets|features)(?=\.)", re.MULTILINE
)
_PY_IMPORT = re.compile(
    r"^(\s*(?:from|import)\s+)(open_climate_service\.plugins\.)(datasets|features)(?=[.\s,]|$)", re.MULTILINE
)


def _rewritten(text: str, pattern: re.Pattern[str]) -> str:
    mapping = dict(PLUGIN_RENAMES)
    return pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{mapping[m.group(3)]}", text)


def _plan_plugin_files(plugins_root: Path, plan: Plan) -> None:
    """Dataset templates in the template folders; Python modules anywhere under plugins_dir.

    plugins_dir is on `sys.path`, so a module in `processes/` or `exports/` can import from
    `datasets` or `features` as readily as a plugin next to its template can.
    """
    candidates: list[tuple[Path, re.Pattern[str]]] = []
    for folder in {old for old, _ in PLUGIN_RENAMES} | {new for _, new in PLUGIN_RENAMES}:
        directory = plugins_root / folder
        if directory.is_dir():
            candidates += [(path, _YAML_PLUGIN) for path in directory.rglob("*") if path.suffix in {".yaml", ".yml"}]
    for path in plugins_root.rglob("*.py"):
        if not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(plugins_root).parts):
            candidates.append((path, _PY_IMPORT))
    for path, pattern in sorted(candidates):
        text = path.read_text(encoding="utf-8")
        new = _rewritten(text, pattern)
        if new != text:
            plan.rewrites.append((path, new))


def _plugins_root() -> Path | None:
    config = api_config.get_config()
    raw = config.get("plugins_dir") if config else None
    if not raw:
        return None
    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    root = (base / raw).resolve()
    return root if root.is_dir() else None


def build_plan() -> Plan:
    """Work out every change for the configured instance, without making any."""
    plan = Plan()
    data_root = api_config.get_data_root().resolve()
    if data_root.is_dir():
        for old, new in DATA_RENAMES:
            _plan_rename(data_root, old, new, plan)
        _plan_records(data_root, plan)
    plugins_root = _plugins_root()
    if plugins_root is not None:
        for old, new in PLUGIN_RENAMES:
            _plan_rename(plugins_root, old, new, plan)
        _plan_plugin_files(plugins_root, plan)
    return plan


def apply(plan: Plan) -> None:
    """Rewrite files first, while their paths are the ones planned, then rename the folders."""
    for path, text in plan.rewrites:
        path.write_text(text, encoding="utf-8")
    if plan.records is not None:
        index, records = plan.records
        _replace_json(index, records)
    for source, target in plan.renames:
        source.rename(target)


def _replace_json(path: Path, value: Any) -> None:
    """Write records.json in the form the service writes it, replacing the old file whole."""
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".records-", encoding="utf-8", delete=False
    ) as handle:
        handle.write(f"{json.dumps(value, indent=2)}\n")
    os.replace(handle.name, path)


def describe(plan: Plan) -> list[str]:
    """One line per change, in the order `apply` makes them."""
    lines = [f"rename {source} -> {target.name}/" for source, target in plan.renames]
    if plan.records is not None:
        lines.append(f"rewrite store paths in {plan.records[0]}")
    lines += [f"rewrite plugin paths in {path}" for path, _ in plan.rewrites]
    return lines


def main(argv: list[str] | None = None) -> int:
    """Migrate the configured instance, or report what would change with ``--dry-run``."""
    parser = argparse.ArgumentParser(description="Move an instance onto the rasters/vectors folder layout.")
    parser.add_argument("--dry-run", action="store_true", help="report the changes without making them")
    args = parser.parse_args(argv)
    try:
        plan = build_plan()
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if plan.empty:
        print("Nothing to migrate: this instance already uses rasters/ and vectors/.")
        return 0
    for line in describe(plan):
        print(("would " if args.dry_run else "") + line)
    if not args.dry_run:
        apply(plan)
        print("Done. Restart the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
