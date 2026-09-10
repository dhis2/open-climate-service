"""Dataset registry backed by YAML config files."""

import copy
import functools
import importlib.resources
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from open_climate_service import config as api_config
from open_climate_service.shared.time import SUPPORTED_PERIOD_TYPES

logger = logging.getLogger(__name__)

# Overridden in tests via monkeypatch to point to a temporary directory.
# When set, only this directory is loaded (no built-ins, no config override).
CONFIGS_DIR: Path | None = None

# Template issues already logged this process, so a re-read path cannot repeat them.
# Guarded by its own lock: FastAPI runs sync endpoints in worker threads, so the
# check-then-add would otherwise let two threads both decide a warning is new.
_WARNED_TEMPLATE_ISSUES: set[tuple[str, str, str]] = set()
_WARNED_LOCK = threading.Lock()

# Serialises cache population. `lru_cache` protects its own state but calls the wrapped
# function outside that lock, so concurrent cold calls would each parse every template.
# Only contended on a cold cache; the warm path takes it and returns immediately.
# Lock order is one-way — a parse takes this and then `_WARNED_LOCK` while validating.
_PARSE_LOCK = threading.Lock()

SUPPORTED_SYNC_KINDS = {"temporal", "release", "static"}
SUPPORTED_SYNC_EXECUTIONS = {"append", "rematerialize"}

# Which way a dataset's periods run relative to now. Deliberately separate from
# `sync.kind`: a forecast is still `temporal` for sync purposes (re-run it and you get
# fresh data), so this is an orthogonal property rather than a fourth kind. Keeping them
# apart also means the sync engine's decision logic is untouched.
#
# `past` (the default) — periods are historical, so an ingestion must say where to start.
# `future` — every period lies ahead of now, as for a weather forecast. An omitted start
#     means "from now", because a fixed date would be stale the next day.
# `spanning` — periods cross now: WorldPop Global2 runs 2015–2030, and climate projections
#     behave the same way. An explicit start is still required, because defaulting to "now"
#     would silently drop the historical half — which is usually the half you want.
#     Declaring it is not decoration: it tells the ingest form to offer the dataset's
#     declared future end rather than truncating at today.
SUPPORTED_TEMPORAL_DIRECTIONS = {"past", "future", "spanning"}
DEFAULT_TEMPORAL_DIRECTION = "past"


def temporal_direction(dataset: dict[str, Any]) -> str:
    """Return a dataset template's temporal direction, defaulting to ``past``."""
    raw = dataset.get("temporal_direction")
    return raw if isinstance(raw, str) and raw else DEFAULT_TEMPORAL_DIRECTION


def is_future_facing(dataset: dict[str, Any]) -> bool:
    """True when *every* period lies ahead of now, so an omitted start can mean "now".

    Deliberately excludes ``spanning``. A dataset straddling now (WorldPop Global2's
    2015–2030, or a climate projection) must still be given an explicit start: defaulting
    it to "now" would quietly ingest only the projected half and drop all the history.
    """
    return temporal_direction(dataset) == "future"


def declared_temporal_end(dataset: dict[str, Any]) -> str | None:
    """Return the template's declared temporal end, if any.

    Used by the ingest form to offer a ``spanning`` dataset's full range rather than
    stopping at today, which would truncate the projected periods.
    """
    extents = dataset.get("extents")
    temporal = extents.get("temporal") if isinstance(extents, dict) else None
    end = temporal.get("end") if isinstance(temporal, dict) else None
    return str(end) if end is not None else None


def list_datasets() -> list[dict[str, Any]]:
    """Load all dataset templates and return a flat list.

    Templates are merged in increasing order of precedence:

    1. Built-in templates from open_climate_service/plugins/datasets/.
    2. Installed plugin packages that declare an ``open_climate_service.plugins``
       entry point (auto-discovered — no config change beyond installing them, #118).
    3. The instance ``plugins_dir`` from CLIMATE_SERVICE_CONFIG.

    A template with the same id overrides one from an earlier stage, so ``plugins_dir``
    always wins and an installed plugin overrides a built-in. Overrides are logged.

    CONFIGS_DIR (test override via monkeypatch) bypasses this and loads only
    from the given directory, as tests supply a fully controlled set.
    """
    if CONFIGS_DIR is not None:
        return _load_from_dir(CONFIGS_DIR)

    merged: dict[str, dict[str, Any]] = {d["id"]: d for d in _load_builtin_datasets()}

    for plugin_name, dataset in _load_entry_point_datasets():
        ds_id = dataset["id"]
        if ds_id in merged:
            logger.warning("Plugin '%s' template '%s' overrides an existing dataset template", plugin_name, ds_id)
        merged[ds_id] = dataset

    config = api_config.get_config()
    if config.get("templates_dir"):
        raise ValueError(
            "CLIMATE_SERVICE_CONFIG uses the removed 'templates_dir' key. "
            "Rename it to 'plugins_dir' and rename the directory from 'templates/' to 'plugins/'."
        )
    config_plugins_dir = config.get("plugins_dir")
    if config_plugins_dir:
        if not isinstance(config_plugins_dir, (str, Path)):
            raise ValueError(
                f"plugins_dir in CLIMATE_SERVICE_CONFIG must be a path string, got {type(config_plugins_dir).__name__}"
            )
        config_path = api_config.get_config_path()
        base = config_path.parent if config_path else Path()
        root = (base / config_plugins_dir).resolve()
        if not root.is_dir():
            # Startup already warns about this (plugins_diagnostics.log_plugin_loading) and lets the
            # service run, so raising here made the two disagree: the instance reported healthy and
            # served /datasets, /collections and /processes while /dataset-templates/ returned 500 —
            # the one route the ingest form needs. A configured-but-absent plugins_dir is a
            # deployment slip, not a reason to refuse service, so degrade to the built-in and
            # entry-point templates instead. Debug rather than warning because startup has already
            # said it loudly, once, and this runs per request.
            logger.debug("plugins_dir '%s' does not exist; serving built-in and plugin templates only", root)
            return list(merged.values())
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.append(root_str)
        datasets_subdir = root / "datasets"
        if datasets_subdir.is_dir():
            for dataset in _load_from_dir(datasets_subdir):
                ds_id = dataset["id"]
                if ds_id in merged:
                    logger.info("plugins_dir template '%s' overrides an existing dataset template", ds_id)
                merged[ds_id] = dataset

    return list(merged.values())


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    """Get dataset dict for a given id."""
    datasets_lookup = {d["id"]: d for d in list_datasets()}
    return datasets_lookup.get(dataset_id)


def get_instance_datasets_dir(*, create: bool = False) -> Path:
    """Return the writable directory for instance dataset templates.

    When CONFIGS_DIR is set (tests), that directory is used directly. Otherwise,
    templates are written to ``plugins_dir/datasets`` resolved relative to the
    instance config file.
    """
    if CONFIGS_DIR is not None:
        if create:
            CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIGS_DIR.is_dir():
            raise ValueError(f"Path is not a directory: {CONFIGS_DIR}")
        return CONFIGS_DIR

    config = api_config.get_config()
    plugins_dir = config.get("plugins_dir")
    if not plugins_dir:
        raise ValueError("Cannot persist dataset template: plugins_dir is not configured in CLIMATE_SERVICE_CONFIG")
    if not isinstance(plugins_dir, (str, Path)):
        raise ValueError(
            f"plugins_dir in CLIMATE_SERVICE_CONFIG must be a path string, got {type(plugins_dir).__name__}"
        )

    config_path = api_config.get_config_path()
    base = config_path.parent if config_path else Path()
    root = (base / plugins_dir).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"plugins_dir '{root}' does not exist or is not a directory")

    datasets_dir = root / "datasets"
    if create:
        datasets_dir.mkdir(parents=True, exist_ok=True)
    if not datasets_dir.is_dir():
        raise ValueError(f"datasets directory '{datasets_dir}' does not exist or is not a directory")
    return datasets_dir


def write_dataset_template(dataset: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Persist one dataset template YAML into the writable instance datasets directory."""
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset template must define a non-empty string id")
    if Path(dataset_id).name != dataset_id:
        raise ValueError(
            f"Invalid dataset id '{dataset_id}': must be a plain name with no path separators or traversal segments"
        )

    datasets_dir = get_instance_datasets_dir(create=True)
    destination = datasets_dir / f"{dataset_id}.yaml"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Dataset template file already exists: {destination}")

    _validate_dataset_template(dataset, source=str(destination))
    payload = yaml.safe_dump([dataset], sort_keys=False, allow_unicode=False)
    destination.write_text(payload, encoding="utf-8")
    return destination


def reset_template_caches() -> None:
    """Forget the parsed built-in and plugin templates, and which warnings were logged.

    Nothing in a running service needs this: both cached stages read files that cannot
    change without a restart, and an instance's ``plugins_dir`` is not cached. It exists so
    tests that install a fake entry point, or patch package data, do not leak state.
    """
    with _PARSE_LOCK:
        _parse_builtin_datasets.cache_clear()
        _parse_entry_point_datasets.cache_clear()
    with _WARNED_LOCK:
        _WARNED_TEMPLATE_ISSUES.clear()


def _load_builtin_datasets() -> list[dict[str, Any]]:
    """Built-in dataset templates, parsed once per process.

    Package data cannot change while the process runs, but this was re-reading and
    re-validating every built-in YAML on each call — 13.5 ms per call, on a path some
    requests take twice — and re-emitting the same validation warnings each time, which
    is what filled deployment logs (CLIM-904). Under ``--reload`` the process restarts,
    so edits during development are still picked up.

    Callers get a deep copy: the cached parse is 13.5 ms and the copy 0.17 ms, so
    isolating callers from each other costs almost nothing next to what caching saves,
    and a mutation of a returned template cannot poison every later request.
    """
    with _PARSE_LOCK:
        parsed = _parse_builtin_datasets()
    return copy.deepcopy(parsed)


@functools.lru_cache(maxsize=1)
def _parse_builtin_datasets() -> list[dict[str, Any]]:
    """Read and validate the built-in templates from package data.

    Using importlib.resources instead of __file__-relative path arithmetic ensures
    this works correctly in both editable installs and wheel installs, where the
    package lives inside site-packages with no guarantee that the project root
    directory (and its data/ folder) is accessible.
    """
    pkg = importlib.resources.files("open_climate_service") / "plugins" / "datasets"
    datasets: list[dict[str, Any]] = []
    for resource in pkg.iterdir():
        if not resource.name.endswith((".yaml", ".yml")):
            continue
        try:
            content = resource.read_text(encoding="utf-8")
            file_datasets = yaml.safe_load(content)
            if not isinstance(file_datasets, list):
                raise ValueError(f"{resource.name} must contain a list of dataset templates")
            for dataset in file_datasets:
                _validate_dataset_template(dataset, source=resource.name)
            datasets.extend(file_datasets)
        except Exception:
            logger.exception("Error loading %s", resource.name)
            raise
    return datasets


def _load_entry_point_datasets() -> list[tuple[str, dict[str, Any]]]:
    """Dataset templates from installed plugin packages, parsed once per process.

    Cached and copied for the same reasons as :func:`_load_builtin_datasets`: an installed
    package's templates cannot change without a reinstall, which needs a restart anyway.
    """
    with _PARSE_LOCK:
        parsed = _parse_entry_point_datasets()
    return copy.deepcopy(parsed)


@functools.lru_cache(maxsize=1)
def _parse_entry_point_datasets() -> list[tuple[str, dict[str, Any]]]:
    """Read and validate dataset templates contributed by installed plugin packages (#118).

    A plugin's ``datasets/*.yaml`` templates are loaded here; the package's Python —
    the ``ingestion.plugin`` class — is importable by dotted path because the package
    is installed, so no ``sys.path`` handling is needed.

    Returns ``(plugin_name, template)`` pairs so the caller can report conflicts.
    """
    from open_climate_service.plugin_discovery import iter_plugin_subdirs

    results: list[tuple[str, dict[str, Any]]] = []
    for plugin_name, _package, datasets_res in iter_plugin_subdirs("datasets"):
        try:
            for resource in datasets_res.iterdir():
                if not resource.name.endswith((".yaml", ".yml")):
                    continue
                file_datasets = yaml.safe_load(resource.read_text(encoding="utf-8"))
                if not isinstance(file_datasets, list):
                    raise ValueError(f"{plugin_name} ({resource.name}) must contain a list of dataset templates")
                for dataset in file_datasets:
                    _validate_dataset_template(dataset, source=f"plugin '{plugin_name}' ({resource.name})")
                    results.append((plugin_name, dataset))
        except Exception:
            logger.exception("Error loading dataset templates from plugin '%s'", plugin_name)
            raise
    return results


def _load_from_dir(folder: Path) -> list[dict[str, Any]]:
    """Load dataset templates from a directory on disk."""
    datasets: list[dict[str, Any]] = []

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    for file_path in folder.glob("*.y*ml"):
        try:
            with open(file_path, encoding="utf-8") as f:
                file_datasets = yaml.safe_load(f)
                if not isinstance(file_datasets, list):
                    raise ValueError(f"{file_path.name} must contain a list of dataset templates")
                for dataset in file_datasets:
                    _validate_dataset_template(dataset, source=str(file_path))
                datasets.extend(file_datasets)
        except Exception:
            logger.exception("Error loading %s", file_path.name)
            raise

    return datasets


def _warn_once(dataset_id: str, source: str, message: str) -> None:
    """Log a template issue once per (template, source, message) per process.

    Built-in templates are parsed once, but an instance's plugins_dir is deliberately re-read
    on every request, so without this a questionable template there would log on every request
    exactly as the built-ins used to (CLIM-904).
    """
    key = (dataset_id, source, message)
    with _WARNED_LOCK:
        first_time = key not in _WARNED_TEMPLATE_ISSUES
        _WARNED_TEMPLATE_ISSUES.add(key)
    if first_time:
        logger.warning("Dataset template '%s' in %s: %s", dataset_id, source, message)


def _validate_dataset_template(dataset: object, *, source: str) -> None:
    """Validate registry fields required by runtime sync planning."""
    if not isinstance(dataset, dict):
        raise ValueError(f"{source} contains a non-object dataset template")

    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"{source} contains a dataset template with a missing or invalid id")
    sync_block = dataset.get("sync", {})
    sync_kind = sync_block.get("kind") if isinstance(sync_block, dict) else None
    if not isinstance(sync_kind, str) or not sync_kind:
        raise ValueError(f"Dataset template '{dataset_id}' in {source} must define sync.kind")
    if sync_kind not in SUPPORTED_SYNC_KINDS:
        supported = ", ".join(sorted(SUPPORTED_SYNC_KINDS))
        raise ValueError(
            f"Dataset template '{dataset_id}' in {source} has unsupported sync.kind "
            f"'{sync_kind}'. Supported values: {supported}"
        )

    # An unsupported period_type is accepted by every downstream consumer and then
    # silently ignored — no step, no period normalisation, no error. Reject it here so a
    # typo or a cadence this version cannot honour fails at registration.
    period_type = dataset.get("period_type")
    if period_type is not None and period_type not in SUPPORTED_PERIOD_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PERIOD_TYPES))
        raise ValueError(
            f"Dataset template '{dataset_id}' in {source} has unsupported period_type "
            f"{period_type!r}. Supported values: {supported}"
        )
    # A temporal or release dataset must declare its cadence: sync planning and coverage
    # index period_type unguarded, so an absent one registers and then raises a KeyError
    # further in. Static datasets are exempt — an openEO save_result output is static and
    # legitimately has no cadence when none can be derived, and sync planning returns for
    # static before it reads the field.
    if period_type is None and sync_kind != "static":
        raise ValueError(
            f"Dataset template '{dataset_id}' in {source} must define period_type "
            f"(required for sync.kind '{sync_kind}')"
        )

    direction = dataset.get("temporal_direction")
    if direction is not None:
        if not isinstance(direction, str) or direction not in SUPPORTED_TEMPORAL_DIRECTIONS:
            supported = ", ".join(sorted(SUPPORTED_TEMPORAL_DIRECTIONS))
            raise ValueError(
                f"Dataset template '{dataset_id}' in {source} has unsupported temporal_direction "
                f"{direction!r}. Supported values: {supported}"
            )
        if direction == "future" and sync_kind == "static":
            raise ValueError(
                f"Dataset template '{dataset_id}' in {source} declares temporal_direction: future "
                "with sync.kind: static. A static dataset has no upstream to look ahead into."
            )

    sync_execution = sync_block.get("execution") if isinstance(sync_block, dict) else None
    if sync_execution is not None:
        if not isinstance(sync_execution, str) or not sync_execution:
            raise ValueError(f"Dataset template '{dataset_id}' in {source} has invalid sync.execution")
        if sync_execution not in SUPPORTED_SYNC_EXECUTIONS:
            supported = ", ".join(sorted(SUPPORTED_SYNC_EXECUTIONS))
            raise ValueError(
                f"Dataset template '{dataset_id}' in {source} has unsupported sync.execution "
                f"'{sync_execution}'. Supported values: {supported}"
            )

    ingestion = dataset.get("ingestion")
    # Static datasets may omit ingestion entirely — they exist as display-metadata
    # entries for artifacts produced outside the sync pipeline (e.g. openEO jobs).
    if sync_kind != "static":
        if not isinstance(ingestion, dict):
            raise ValueError(f"Dataset template '{dataset_id}' in {source} must define an 'ingestion' block")
        plugin = ingestion.get("plugin")
        has_plugin = isinstance(plugin, str) and bool(plugin)
        if not has_plugin:
            raise ValueError(f"Dataset template '{dataset_id}' in {source} must define ingestion.plugin")

    # Surface non-CF/udunits units so unit-aware processes (xclim indices) don't fail
    # cryptically later. Warn rather than reject — not every variable is a physical
    # quantity (e.g. population counts) (#280).
    # A licence is how a user knows what they may do with a layer, and OCS published the
    # constant "various" on every collection until CLIM-946. Warn rather than reject: an
    # instance should not be unable to serve data because a template is missing metadata, and
    # an undeclared licence still publishes honestly as `other`. But warn every time, because
    # the default path — say nothing, publish something permissive-sounding — is how a
    # non-commercial source gets laundered through a derived product.
    if "license" not in dataset:
        _warn_once(
            dataset_id,
            source,
            "declares no 'license'; the published collection will say 'other'. Set an SPDX "
            "identifier (license: CC-BY-4.0) or a name and URL for a licence that has none.",
        )
    else:
        from open_climate_service.shared.licences import (
            UNDECLARED,
            licence_declaration_problem,
            parse_licence,
        )

        # A specific complaint where there is one — a contradictory id/name pair reads as
        # "unreadable" otherwise, which sends the author looking for a typo.
        problem = licence_declaration_problem(dataset.get("license"))
        if problem is not None:
            _warn_once(dataset_id, source, problem)
        elif parse_licence(dataset.get("license")) is UNDECLARED:
            _warn_once(
                dataset_id,
                source,
                f"has an unreadable 'license' value {dataset.get('license')!r}; treating it as "
                "undeclared. Expected an SPDX identifier, or a mapping with 'name' and 'url'.",
            )

    units = dataset.get("units")
    if isinstance(units, str):
        from open_climate_service.shared.cf import validate_units

        message = validate_units(units)
        if message:
            _warn_once(dataset_id, source, message)
